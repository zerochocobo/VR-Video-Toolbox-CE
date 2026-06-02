"""GPU video processing engine.

Keep the whole "decode -> geometry transform -> compose -> encode" path in GPU memory:
  PyNvVideoCodec decode (NVDEC) -> CuPy / RawKernel transform -> PyNvVideoCodec encode (NVENC) -> ffmpeg for audio mux only.

See summary/summary_20260529_GPU_PIPELINE_REFACTOR_PLAN_CN.md for the design.

Public entry points:
  - runtime.warmup() / runtime.gpu_available()
  - probe.probe_video(path) / probe.decide_backend(meta)
  - files.*  (file-in/file-out high-level pipelines used by each logic.py)
  - fallback.with_auto_fallback(...)
"""
from __future__ import annotations

# Configure the CUDA environment before any cupy import.
from . import _cuda_env as _cuda_env

_cuda_env.configure()

# Note: after aligning the full stack to CUDA 12.8 (cupy-cuda12x + torch cu128
# + 12.8 nvrtc wheel), CuPy no longer conflicts with torch's nvrtc, so this file
# no longer has to enforce import order. See the header in _cuda_env for details.

__all__ = [
    "runtime",
    "probe",
    "pynv_io",
    "mux",
    "fallback",
]
