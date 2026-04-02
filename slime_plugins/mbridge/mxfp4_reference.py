import math

import torch


FP4_VALUES = [
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def convert_moe_packed_tensors_reference(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 32768 * 1024,
) -> torch.Tensor:
    """Reference GPT-OSS MXFP4 dequantization copied from HF transformers.

    This mirrors `transformers.integrations.mxfp4.convert_moe_packed_tensors`.
    """
    blocks = blocks.to(torch.uint8)
    scales = scales.to(torch.int32) - 127
    assert blocks.shape[:-1] == scales.shape, f"{blocks.shape[:-1]=} does not match {scales.shape=}"

    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

    *prefix_shape, g, b = blocks.shape
    rows_total = math.prod(prefix_shape) * g

    blocks = blocks.reshape(rows_total, b)
    scales = scales.reshape(rows_total, 1)
    out = torch.empty(rows_total, b * 2, dtype=dtype, device=blocks.device)

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)
        blk = blocks[r0:r1]
        exp = scales[r0:r1]
        sub = out[r0:r1]

        idx_lo = (blk & 0x0F).to(torch.int)
        sub[:, 0::2] = lut[idx_lo]
        del idx_lo

        idx_hi = (blk >> 4).to(torch.int)
        sub[:, 1::2] = lut[idx_hi]
        del idx_hi

        torch.ldexp(sub, exp, out=sub)
        del blk, exp, sub

    out = out.reshape(*prefix_shape, g, b * 2).view(*prefix_shape, g * b * 2)
    return out.transpose(1, 2).contiguous()
