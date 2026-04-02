import logging
import os

print("[bridge-repro-megatron-utils] package init start", flush=True)

import torch

try:
    print("[bridge-repro-megatron-utils] importing deep_ep", flush=True)
    import deep_ep
    from torch_memory_saver import torch_memory_saver

    print("[bridge-repro-megatron-utils] imported deep_ep", flush=True)

    old_init = deep_ep.Buffer.__init__

    def new_init(self, *args, **kwargs):
        if torch_memory_saver._impl is not None:
            torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(False)
        old_init(self, *args, **kwargs)
        torch.cuda.synchronize()
        if torch_memory_saver._impl is not None:
            torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(True)

    deep_ep.Buffer.__init__ = new_init
except ImportError:
    logging.warning("deep_ep is not installed, some functionalities may be limited.")

if os.environ.get("SLIME_ENABLE_QWEN_ROTARY_PATCH", "1").strip().lower() not in {"0", "false", "no"}:
    try:
        print("[bridge-repro-megatron-utils] importing megatron.bridge qwen rotary patch targets", flush=True)
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import (
            Qwen3VLMoETextRotaryEmbedding,
            Qwen3VLTextRotaryEmbedding,
        )

        print("[bridge-repro-megatron-utils] imported megatron.bridge qwen rotary patch targets", flush=True)

        def patch_rotary_embedding(cls):
            _original_forward = cls.forward

            def _patched_forward(self, *args, packed_seq_params=None, **kwargs):
                return _original_forward(self, *args, **kwargs)

            cls.forward = _patched_forward

        patch_rotary_embedding(Qwen3VLTextRotaryEmbedding)
        patch_rotary_embedding(Qwen3VLMoETextRotaryEmbedding)
    except ImportError:
        pass
else:
    print("[bridge-repro-megatron-utils] skipping qwen rotary patch import by env", flush=True)

logging.getLogger("megatron").setLevel(logging.WARNING)

print("[bridge-repro-megatron-utils] package init done", flush=True)
