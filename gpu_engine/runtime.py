"""CUDA/CuPy/PyNv runtime: device probing, memory pool management, startup warmup, and capability checks.

main.py calls warmup() at startup. On failure, the app globally degrades to
ffmpeg-only mode.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass

_lock = threading.Lock()
_state: "GpuState | None" = None


@dataclass(frozen=True)
class GpuState:
    available: bool
    reason: str
    summary: str = ""
    name: str = ""
    cc_major: int = 0
    cc_minor: int = 0
    nvenc_hevc: bool = False
    nvenc_hevc_10bit: bool = False

    @property
    def compute_capability(self) -> float:
        return float(f"{self.cc_major}.{self.cc_minor}")


def _detect() -> GpuState:
    try:
        import cupy as cp
    except Exception as exc:
        return GpuState(False, f"cupy import failed: {type(exc).__name__}: {exc}")
    try:
        import PyNvVideoCodec as nvc
    except Exception as exc:
        return GpuState(False, f"PyNvVideoCodec import failed: {type(exc).__name__}: {exc}")

    try:
        ndev = cp.cuda.runtime.getDeviceCount()
        if ndev <= 0:
            return GpuState(False, "no CUDA device found")
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props.get("name", b"")
        name = name.decode("utf-8", "replace") if isinstance(name, bytes) else str(name)
        cc_major = int(props.get("major", 0))
        cc_minor = int(props.get("minor", 0))
    except Exception as exc:
        return GpuState(False, f"CUDA device query failed: {type(exc).__name__}: {exc}")

    from .pynv_io import cuda_device_summary

    summary = cuda_device_summary(0)

    nvenc_hevc = False
    nvenc_hevc_10bit = False
    try:
        caps = nvc.GetEncoderCaps(gpuid=0, codec="hevc")
        # caps is dict[str, int]; key names differ by version, so any value means usable.
        nvenc_hevc = bool(caps)
        # 10-bit encode capability keys vary by version, often like NV_ENC_CAPS_SUPPORT_10BIT_ENCODE.
        for k, v in (caps or {}).items():
            if "10BIT" in k.upper() or "10_BIT" in k.upper():
                nvenc_hevc_10bit = bool(v)
        # If no explicit 10-bit cap is found, assume Turing+ (cc >= 7.5) supports HEVC Main10.
        if nvenc_hevc and not nvenc_hevc_10bit and (cc_major, cc_minor) >= (7, 5):
            nvenc_hevc_10bit = True
    except Exception:
        # GetEncoderCaps failure is non-fatal; infer from compute capability.
        if (cc_major, cc_minor) >= (7, 0):
            nvenc_hevc = True
            nvenc_hevc_10bit = (cc_major, cc_minor) >= (7, 5)

    if not nvenc_hevc:
        return GpuState(
            False, "GPU has no usable HEVC NVENC", summary, name, cc_major, cc_minor, False, False
        )

    return GpuState(
        True, "ok", summary, name, cc_major, cc_minor, nvenc_hevc, nvenc_hevc_10bit
    )


def _warmup_roundtrip() -> None:
    """Run a minimal CuPy kernel and memory-pool warmup to trigger the JIT cache."""
    import cupy as cp

    a = cp.arange(1 << 16, dtype=cp.uint8)
    b = (a + 1).sum()
    cp.cuda.get_current_stream().synchronize()
    _ = int(b)  # Force synchronized evaluation.


_nvrtc_locked = False


def lock_nvrtc() -> bool:
    """Make CuPy resolve and compile nvrtc immediately, caching the nvrtc 13.0 handle.

    Must be called before any `import torch`. torch loads its bundled nvrtc 12.x,
    after which cuda-pathfinder may treat the already-loaded torch nvrtc as the
    current nvrtc, causing CuPy to compile CUDA 13 headers such as
    cuda_fp8/fp6/fp4.hpp with nvrtc 12.x and fail. If CuPy compiles once before
    torch, it caches the correct nvrtc 13.0 handle and remains unaffected after
    torch loads. This is idempotent and non-fatal.
    """
    global _nvrtc_locked
    if _nvrtc_locked:
        return True
    try:
        import cupy as cp
        # Compile a real RawKernel through nvrtc, including fp16 builtins, to trigger the full header chain.
        k = cp.RawKernel(
            r'extern "C" __global__ void _vrtb_lock(float* d){'
            r' d[threadIdx.x] = __half2float(__float2half(d[threadIdx.x])); }',
            "_vrtb_lock",
        )
        d = cp.ones(8, dtype=cp.float32)
        k((1,), (8,), (d,))
        cp.cuda.Stream.null.synchronize()
        _nvrtc_locked = True
        return True
    except Exception:
        return False


def warmup(verbose: bool = False) -> GpuState:
    """Warm up at startup and return GpuState. Thread-safe and idempotent."""
    global _state
    with _lock:
        if _state is not None:
            return _state
        # Keep CuPy memory-pool behavior friendly to upper limits; each file explicitly frees fragmentation afterward.
        state = _detect()
        if state.available:
            try:
                lock_nvrtc()       # Lock CuPy nvrtc 13.0 before torch.
                _warmup_roundtrip()
            except Exception as exc:
                state = GpuState(
                    False, f"warmup roundtrip failed: {type(exc).__name__}: {exc}",
                    state.summary, state.name, state.cc_major, state.cc_minor,
                )
        _state = state
        if verbose or os.environ.get("VRTB_GPU_LOG_VERBOSE"):
            print(f"[gpu_engine] warmup: available={state.available} reason={state.reason}")
            if state.summary:
                print(f"[gpu_engine] {state.summary}")
            if state.available:
                print(f"[gpu_engine] nvenc_hevc={state.nvenc_hevc} nvenc_hevc_10bit={state.nvenc_hevc_10bit}")
        return state


def get_state() -> GpuState:
    """Return the warmed-up state, warming up first if needed."""
    return _state if _state is not None else warmup()


def gpu_available() -> bool:
    return get_state().available


def supports_10bit() -> bool:
    return get_state().nvenc_hevc_10bit


def free_memory_pool() -> None:
    """Free CuPy memory pools after each file to avoid fragmentation OOM in long batches."""
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
