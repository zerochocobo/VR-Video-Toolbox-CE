"""Preconfigure the CUDA environment before any `import cupy`.

Architecture since 2026-05-30, with the full stack aligned to CUDA 12.8:
  torch(cu128) + cupy-cuda12x + nvidia-cuda-nvrtc-cu12(12.8) + nvidia-cuda-runtime-cu12(12.8)
  + PyNvVideoCodec are all unified on **CUDA 12.8**. CuPy's nvrtc and headers
  come from pip wheels, so the setup is self-contained and does not depend on a
  system CUDA toolkit. Blackwell sm_120 is supported by nvrtc 12.8 + PTX JIT.

Why CUDA 13 / cupy-cuda13x is no longer used: sm_120 previously used cuda13x
with a system v13.0 nvrtc, but adding torch, which ships CUDA 12.8
nvrtc/builtins, caused an nvrtc version conflict between CuPy and torch. CuPy
would compile CUDA 13 headers such as cuda_fp8/fp6/fp4.hpp with 12.x nvrtc and
fail. Aligning everything to 12.8 removes that issue.

At package import time this module:
  1. Sets CUPY_COMPILE_WITH_PTX=1 so kernels emit PTX and the driver JITs for the actual GPU architecture, including sm_120.
  2. Clears system CUDA_PATH/CUDA_HOME; otherwise CuPy may use system v12.6/v13.0 headers that do not match the CCCL bundled with cupy-cuda12x 14.x, which requires 12.8 headers.
     After clearing those variables, CuPy uses cuda-pathfinder to locate the pip wheel's 12.8 nvrtc and headers, keeping versions aligned.
  3. Adds nvidia-* wheel bin directories to the DLL search path for cudart/nvrtc loading and PyNv compatibility.

Packaging in onedir mode includes wheel DLLs and headers. The runtime hook
handles paths, and frozen mode skips system probing.
"""
from __future__ import annotations

import os
import sys

_configured = False


def _runtime_cache_root() -> str:
    """Return the app-local runtime cache root used for GPU tool caches."""
    if getattr(sys, "frozen", False):
        return os.path.join(os.path.dirname(os.path.abspath(sys.executable)), "runtime_cache")
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runtime_cache")


def _ensure_cache_dir(path: str) -> tuple[bool, str]:
    """Create and write-probe a cache directory without leaving test files behind."""
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".write_test")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        try:
            os.unlink(probe)
        except OSError:
            pass
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _configure_cache_dirs(info: dict) -> None:
    """Route CuPy and CUDA driver JIT caches to runtime_cache.

    CuPy defaults to a user-profile cache such as %USERPROFILE%\\.cupy. On some
    Windows accounts that directory may exist but be non-writable, which can
    make first-use JIT operations appear to hang. Keep both CuPy's kernel cache
    and the CUDA driver's compute cache inside the app-local runtime_cache.
    """
    root = _runtime_cache_root()
    requested = {
        "CUPY_CACHE_DIR": os.path.join(root, "cupy_kernel_cache"),
        "CUDA_CACHE_PATH": os.path.join(root, "cuda_compute_cache"),
    }
    cache_info: dict[str, dict[str, str | bool]] = {}
    for key, default_path in requested.items():
        had_value = bool(os.environ.get(key))
        path = os.environ.get(key) or default_path
        ok, error = _ensure_cache_dir(path)
        if ok and not had_value:
            os.environ[key] = path
        cache_info[key] = {
            "path": os.environ.get(key, path),
            "set_by_vrtb": ok and not had_value,
            "writable": ok,
            "error": error,
        }
    info["cache_dirs"] = cache_info


def _nvidia_wheel_bin_dirs() -> list[str]:
    """Return bin directories from installed nvidia-*-cu12 wheels, including cudart/nvrtc DLLs."""
    dirs: list[str] = []
    try:
        import importlib.util
        spec = importlib.util.find_spec("nvidia")
        if not spec or not spec.submodule_search_locations:
            return dirs
        for nvidia_root in spec.submodule_search_locations:
            if not os.path.isdir(nvidia_root):
                continue
            for sub in os.listdir(nvidia_root):
                b = os.path.join(nvidia_root, sub, "bin")
                if os.path.isdir(b):
                    dirs.append(b)
    except Exception:
        pass
    return dirs


def configure() -> dict:
    """Configure the CUDA environment once and return diagnostic information."""
    global _configured
    if _configured:
        return {"already": True}
    _configured = True

    info: dict = {}
    os.environ.setdefault("CUPY_COMPILE_WITH_PTX", "1")
    info["ptx"] = os.environ.get("CUPY_COMPILE_WITH_PTX")
    _configure_cache_dirs(info)

    if getattr(sys, "frozen", False):
        info["frozen"] = True
        return info

    # Clear system CUDA_PATH/CUDA_HOME so CuPy is forced to use the pip wheel's 12.8 nvrtc and headers.
    for key in ("CUDA_PATH", "CUDA_HOME"):
        if os.environ.get(key):
            os.environ.setdefault(f"{key}_ORIGINAL_VRTB", os.environ[key])
            os.environ.pop(key, None)
    info["cleared_cuda_path"] = True

    # Expose nvidia wheel bin directories to the DLL loader for cudart/nvrtc loading and PyNv runtime compatibility.
    bins = _nvidia_wheel_bin_dirs()
    added = []
    for d in bins:
        try:
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(d)
                added.append(d)
        except Exception:
            pass
    if bins:
        os.environ["PATH"] = os.pathsep.join(bins) + os.pathsep + os.environ.get("PATH", "")
    info["nvidia_wheel_bins"] = added
    return info
