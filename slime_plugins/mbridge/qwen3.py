from mbridge.core import register_model
from mbridge.models.qwen3 import Qwen3Bridge as _PyPIQwen3Bridge


@register_model("qwen3")
class Qwen3Bridge(_PyPIQwen3Bridge):
    """
    Qwen3 bridge for SLIME with --transformer-impl local support.

    PyPI Qwen3Bridge only handles TransformerEngine (TE) fused naming:
      self_attention.linear_qkv.layer_norm_weight  -> input_layernorm
      mlp.linear_fc1.layer_norm_weight             -> post_attention_layernorm

    With --transformer-impl local, Megatron stores layernorms separately:
      decoder.layers.X.input_layernorm.weight      -> HF input_layernorm.weight
      decoder.layers.X.pre_mlp_layernorm.weight    -> HF post_attention_layernorm.weight

    This bridge intercepts both cases before the parent dispatcher.
    """

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        # Local impl: standalone input_layernorm (not fused into self_attention.linear_qkv)
        if (
            ".input_layernorm.weight" in mcore_weights_name
            and ".self_attention." not in mcore_weights_name
        ):
            layer_number = mcore_weights_name.split(".")[2]
            return [f"model.layers.{layer_number}.input_layernorm.weight"]

        # Local impl: standalone pre_mlp_layernorm (not fused into mlp.linear_fc1)
        # "mlp" appears in "pre_mlp_layernorm" so the parent would route to _weight_name_mapping_mlp
        # and fail — we intercept it here.
        if ".pre_mlp_layernorm.weight" in mcore_weights_name:
            layer_number = mcore_weights_name.split(".")[2]
            return [f"model.layers.{layer_number}.post_attention_layernorm.weight"]

        return super()._weight_name_mapping_mcore_to_hf(mcore_weights_name)
