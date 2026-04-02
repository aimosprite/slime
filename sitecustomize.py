"""Process-wide compatibility patches loaded automatically by Python."""

from __future__ import annotations

import builtins
import inspect
import os
import sys
from functools import wraps

_ORIG_IMPORT = builtins.__import__
_PATCHING = False
_ENABLE_GLOBAL_PATCH = os.environ.get("SLIME_ENABLE_GLOBAL_SGLANG_PATCH", "0") == "1"

def _patch_sglang_default_weight_loader() -> None:
    global _PATCHING

    if _PATCHING:
        return

    _PATCHING = True
    try:
        weight_utils = sys.modules.get("sglang.srt.model_loader.weight_utils")
        if weight_utils is None:
            return

        default_weight_loader = getattr(weight_utils, "default_weight_loader", None)
        if default_weight_loader is None:
            return

        try:
            sig = inspect.signature(default_weight_loader)
        except (TypeError, ValueError):
            return

        if getattr(default_weight_loader, "_slime_filters_unsupported_kwargs", False):
            return

        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
            return

        supported_kwargs = {
            name
            for name, param in sig.parameters.items()
            if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }

        @wraps(default_weight_loader)
        def _compat_default_weight_loader(*args, **kwargs):
            filtered_kwargs = {name: value for name, value in kwargs.items() if name in supported_kwargs}
            return default_weight_loader(*args, **filtered_kwargs)

        _compat_default_weight_loader._slime_filters_unsupported_kwargs = True
        weight_utils.default_weight_loader = _compat_default_weight_loader

        gpt_oss = sys.modules.get("sglang.srt.models.gpt_oss")
        if gpt_oss is not None and getattr(gpt_oss, "default_weight_loader", None) is default_weight_loader:
            gpt_oss.default_weight_loader = _compat_default_weight_loader
    finally:
        _PATCHING = False


def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
    module = _ORIG_IMPORT(name, globals, locals, fromlist, level)

    if _ENABLE_GLOBAL_PATCH and (
        name.startswith("sglang") or (fromlist and any(str(item).startswith("sglang") for item in fromlist))
    ):
        _patch_sglang_default_weight_loader()

    return module


if _ENABLE_GLOBAL_PATCH:
    builtins.__import__ = _patched_import
