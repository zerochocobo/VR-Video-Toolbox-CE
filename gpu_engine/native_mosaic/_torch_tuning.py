"""Torch inference tuning switches for native_mosaic."""
from __future__ import annotations

import os


_FALSE_VALUES = {"0", "false", "no", "off"}


def inference_tuning_enabled() -> bool:
    # Default OFF: enabling cudnn.benchmark on BasicVSR++ inputs whose internal
    # feature-map shapes vary per clip (deform_align, SPyNet pyramid levels,
    # last-clip remainder length) triggered repeated cudnn autotune passes that
    # added minutes to real 8K restore runs. Opt in with VRVT_INFERENCE_TUNING=1
    # only on workloads with stable shapes.
    return str(os.environ.get("VRVT_INFERENCE_TUNING", "0")).strip().lower() not in _FALSE_VALUES


def channels_last_enabled() -> bool:
    if not inference_tuning_enabled():
        return False
    # Default OFF: BasicVSR++ deform_conv2d + channels_last on Blackwell sm_120
    # has triggered native fast-fail (Windows STATUS_STACK_BUFFER_OVERRUN
    # 0xc0000409) during decode. Re-enable per run with VRVT_CHANNELS_LAST=1
    # only after profiling confirms a stable path.
    return str(os.environ.get("VRVT_CHANNELS_LAST", "0")).strip().lower() not in _FALSE_VALUES


def apply_inference_tuning():
    """Apply process-wide torch inference flags.

    Returns a small metadata dict so profile logs can show which knobs were
    active. All settings are best-effort and can be disabled with
    VRVT_INFERENCE_TUNING=0.
    """
    import torch

    enabled = inference_tuning_enabled()
    if not enabled:
        return {"enabled": False}

    state = {"enabled": True}
    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        state["cudnn_benchmark"] = True
        state["cudnn_allow_tf32"] = True
    except Exception as exc:
        state["cudnn_error"] = f"{type(exc).__name__}: {exc}"
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        state["matmul_allow_tf32"] = True
        state["float32_matmul_precision"] = "high"
    except Exception as exc:
        state["tf32_error"] = f"{type(exc).__name__}: {exc}"
    try:
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        # Keep math SDPA enabled as a safety fallback. Disabling it once caused
        # native fast-fail crashes on layers whose shape neither flash nor
        # mem-efficient kernels accepted.
        state["sdpa_flash"] = True
    except Exception as exc:
        state["sdpa_error"] = f"{type(exc).__name__}: {exc}"
    return state


def convert_module_channels_last(module) -> bool:
    if not channels_last_enabled() or module is None:
        return False
    try:
        import torch

        module.to(memory_format=torch.channels_last)
        return True
    except Exception:
        return False


def to_channels_last_5d(inputs):
    """Apply channels_last to the inner NCHW frames of a BTCHW tensor."""
    if not channels_last_enabled() or getattr(inputs, "ndim", 0) != 5:
        return inputs
    try:
        import torch

        b, t, c, h, w = inputs.shape
        frames = inputs.reshape(b * t, c, h, w).contiguous(memory_format=torch.channels_last)
        return frames.reshape(b, t, c, h, w)
    except Exception:
        return inputs
