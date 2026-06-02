"""GPU-path exception classification and automatic fallback to ffmpeg.

Strategy, see section 7 of the plan:
  - Static routing: probe.decide_backend() routes to ffmpeg before execution.
  - Dynamic runtime: if the GPU path raises during execution, fall back per file and rerun the ffmpeg implementation.
  - Startup: runtime.warmup() failure globally degrades to ffmpeg-only mode.

Backend configuration states:
  - "auto"   : use GPU when available and automatically fall back on errors (default)
  - "gpu"    : force GPU and raise directly on failure for debugging
  - "ffmpeg" : force ffmpeg and skip GPU
"""
from __future__ import annotations

from typing import Callable

from . import runtime


class GpuUnsupportedError(RuntimeError):
    """The source is unsupported by the GPU path according to static routing."""


class GpuRuntimeError(RuntimeError):
    """The GPU path failed during decode, encode, kernel execution, or OOM."""


class OperationCancelled(Exception):
    """User cancellation, propagated directly without ffmpeg fallback."""


def get_backend_mode() -> str:
    """Read transcode_backend from app_config, defaulting to auto."""
    try:
        from utils import app_config

        mode = str(app_config.get("transcode_backend", "auto") or "auto").lower()
    except Exception:
        mode = "auto"
    return mode if mode in {"auto", "gpu", "ffmpeg"} else "auto"


def run_with_fallback(
    gpu_fn: Callable,
    ffmpeg_fn: Callable,
    *,
    gpu_eligible: bool,
    log_callback=None,
    label: str = "",
):
    """Select the GPU or ffmpeg implementation by backend mode, falling back automatically on GPU failures.

    gpu_fn and ffmpeg_fn are zero-argument callables with arguments pre-bound by
    functools.partial or lambda. The return value from the called implementation
    is returned unchanged.
    """
    mode = get_backend_mode()

    def _log(msg: str):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    if mode == "ffmpeg":
        _log(f"[backend=ffmpeg{(' ' + label) if label else ''}] forced by config")
        return ffmpeg_fn()

    if not runtime.gpu_available():
        if mode == "gpu":
            raise GpuRuntimeError(f"backend=gpu forced but GPU unavailable: {runtime.get_state().reason}")
        _log(f"[backend=ffmpeg{(' ' + label) if label else ''}] GPU unavailable: {runtime.get_state().reason}")
        return ffmpeg_fn()

    if not gpu_eligible:
        if mode == "gpu":
            raise GpuUnsupportedError(f"backend=gpu forced but source not GPU-eligible{(': ' + label) if label else ''}")
        _log(f"[backend=ffmpeg{(' ' + label) if label else ''}] source not GPU-eligible, routing to ffmpeg")
        return ffmpeg_fn()

    # GPU path.
    try:
        result = gpu_fn()
        runtime.free_memory_pool()
        return result
    except OperationCancelled:
        # User cancellation: do not fall back; propagate directly.
        runtime.free_memory_pool()
        raise
    except Exception as exc:
        runtime.free_memory_pool()
        if mode == "gpu":
            # Forced GPU mode does not fall back; re-raise unchanged for debugging.
            raise
        _log(f"[gpu→ffmpeg fallback]{(' ' + label) if label else ''} reason={type(exc).__name__}: {exc}")
        return ffmpeg_fn()
