"""CUDA/CuPy/PyNv runtime: device probing, memory pool management, warmup, and capability checks.

GPU tools call warmup() on demand. On failure, callers can degrade to ffmpeg-only
mode.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

_lock = threading.Lock()
_state: "GpuState | None" = None
_vram_lock = threading.Lock()
_vram_cache: dict[tuple[int, str], tuple[float, "VramUsage | None"]] = {}
_vram_refreshing: set[tuple[int, str]] = set()


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


@dataclass(frozen=True)
class VramUsage:
    used_mib: int
    total_mib: int
    device_index: int = 0
    smi_id: str = ""

    @property
    def used_gib(self) -> float:
        return float(self.used_mib) / 1024.0

    @property
    def total_gib(self) -> float:
        return float(self.total_mib) / 1024.0


def _nvidia_smi_path() -> str | None:
    path = shutil.which("nvidia-smi")
    if path:
        return path
    if sys.platform.startswith("win"):
        system_root = os.environ.get("SystemRoot") or os.environ.get("WINDIR")
        if system_root:
            candidate = os.path.join(system_root, "System32", "nvidia-smi.exe")
            if os.path.isfile(candidate):
                return candidate
    return None


def _hidden_subprocess_kwargs(timeout_s: float) -> dict:
    kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.DEVNULL,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": max(0.1, float(timeout_s)),
    }
    if sys.platform.startswith("win"):
        creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0) or 0)
        if creationflags:
            kwargs["creationflags"] = creationflags
        try:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = int(getattr(subprocess, "SW_HIDE", 0) or 0)
            kwargs["startupinfo"] = startupinfo
        except Exception:
            pass
    return kwargs


def _current_cuda_device_index(default: int = 0) -> int:
    cp = sys.modules.get("cupy")
    if cp is not None:
        try:
            return int(cp.cuda.runtime.getDevice())
        except Exception:
            pass
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                return int(torch.cuda.current_device())
        except Exception:
            pass
    return int(default)


def _resolve_cuda_device_index(device_index: int | None) -> int:
    if device_index is None:
        return _current_cuda_device_index(0)
    return int(device_index)


def _nvidia_smi_id_for_cuda_device(device_index: int | None) -> tuple[int, str]:
    logical_index = _resolve_cuda_device_index(device_index)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible.lower() not in {"-1", "none", "nodevfiles"}:
        devices = [part.strip() for part in visible.split(",") if part.strip()]
        if 0 <= logical_index < len(devices):
            return logical_index, devices[logical_index]
    return logical_index, str(logical_index)


def _parse_nvidia_smi_memory(stdout: str, device_index: int, smi_id: str | None = None) -> VramUsage | None:
    lines = [line.strip() for line in str(stdout or "").splitlines() if line.strip()]
    if not lines:
        return None
    parts = [part.strip().replace("MiB", "").replace("Mib", "") for part in lines[0].split(",")]
    if len(parts) < 2:
        return None
    try:
        used = int(float(parts[0]))
        total = int(float(parts[1]))
    except Exception:
        return None
    if used < 0 or total <= 0:
        return None
    return VramUsage(used_mib=used, total_mib=total, device_index=int(device_index), smi_id=str(smi_id or device_index))


def query_vram_usage(*, device_index: int | None = None, min_interval_s: float = 2.0,
                     timeout_s: float = 0.8, force: bool = False) -> VramUsage | None:
    """Return current NVIDIA VRAM usage via nvidia-smi, silently cached/throttled."""
    now = time.monotonic()
    logical_index, smi_id = _nvidia_smi_id_for_cuda_device(device_index)
    key = (logical_index, smi_id)
    min_interval = max(0.0, float(min_interval_s))
    with _vram_lock:
        cached_time, cached = _vram_cache.get(key, (0.0, None))
        if not force and (now - cached_time) < min_interval:
            return cached
        if key in _vram_refreshing:
            return cached
        _vram_refreshing.add(key)

    smi = _nvidia_smi_path()
    if not smi:
        usage = None
    else:
        cmd = [
            smi,
            f"--id={smi_id}",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        try:
            proc = subprocess.run(cmd, **_hidden_subprocess_kwargs(timeout_s))
            usage = _parse_nvidia_smi_memory(
                getattr(proc, "stdout", ""),
                logical_index,
                smi_id=smi_id,
            ) if proc.returncode == 0 else None
        except Exception:
            usage = None

    with _vram_lock:
        _vram_cache[key] = (time.monotonic(), usage)
        _vram_refreshing.discard(key)
        return usage


def format_vram_usage(*, device_index: int | None = None, min_interval_s: float = 2.0) -> str:
    usage = query_vram_usage(device_index=device_index, min_interval_s=min_interval_s)
    if usage is None:
        return ""
    return f" | VRAM {usage.used_gib:.1f}/{usage.total_gib:.1f} GiB"


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
