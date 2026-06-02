"""PyInstaller runtime hook that configures CUDA in frozen mode before any cupy/torch import.

The onedir distribution bundles CUDA / cuDNN / nvrtc DLLs and headers beside the
exe and under _internal. This hook fully isolates the system CUDA installation so
all lookups are resolved from the packaged app:

  1. Set CUPY_COMPILE_WITH_PTX=1 for cross-generation PTX compatibility, including Blackwell sm_120.
  2. Override CUDA_PATH / CUDA_HOME to the distribution root with no fallback to system v12.6/v13.0 toolkits.
  3. Prepend all bundled CUDA / cuDNN / nvidia wheel bin directories to the DLL search path.
  4. Prepend the exe directory to PATH so shutil.which prefers packaged ffmpeg.exe.

The development-mode counterpart is gpu_engine/_cuda_env.py; frozen mode skips
system probing in _cuda_env.
"""
import os
import sys


def _isdir(p: str) -> bool:
    try:
        return bool(p) and os.path.isdir(p)
    except OSError:
        return False


def _add_dll_dir(d: str) -> None:
    if not _isdir(d):
        return
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(d)
        except (FileNotFoundError, OSError):
            pass


def _setup() -> None:
    if not getattr(sys, "frozen", False):
        return

    os.environ.setdefault("CUPY_COMPILE_WITH_PTX", "1")

    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    meipass = getattr(sys, "_MEIPASS", None) or exe_dir
    # In onedir mode, _MEIPASS may be exe_dir or exe_dir\_internal, so probe both.
    internal_dir = os.path.join(exe_dir, "_internal")
    if not _isdir(internal_dir):
        internal_dir = meipass

    # CUDA_PATH must point to a real toolkit-style layout containing bin\<nvrtc>.dll and include\.
    # cuda-pathfinder / cupy always call add_dll_directory(CUDA_PATH\bin), so that directory must exist.
    nvrtc_names = (
        "nvrtc64_120_0.dll", "nvrtc64_130_0.dll",
        "nvrtc64_128_0.dll",
    )

    def _has_cuda(d: str) -> bool:
        if not _isdir(d):
            return False
        if _isdir(os.path.join(d, "include")):
            return True
        bin_dir = os.path.join(d, "bin")
        if _isdir(bin_dir):
            for n in nvrtc_names:
                if os.path.isfile(os.path.join(bin_dir, n)):
                    return True
        return False

    cuda_root = next(
        (c for c in (internal_dir, exe_dir, meipass) if _has_cuda(c)),
        internal_dir,
    )

    for key in ("CUDA_PATH", "CUDA_HOME"):
        old = os.environ.get(key)
        if old:
            os.environ.setdefault(f"{key}_ORIGINAL_VRTB", old)
        os.environ[key] = cuda_root
    # Ensure CUDA_PATH\bin exists because cupy adds it during import.
    try:
        os.makedirs(os.path.join(cuda_root, "bin"), exist_ok=True)
    except OSError:
        pass

    # Collect every directory that may contain CUDA / cuDNN / torch DLLs.
    candidates = [
        exe_dir,
        os.path.join(exe_dir, "bin"),
        cuda_root,
        os.path.join(cuda_root, "bin"),
        os.path.join(cuda_root, "bin", "x64"),
        internal_dir,
        os.path.join(internal_dir, "bin"),
        # CUDA / cuDNN DLLs shipped by torch.
        os.path.join(internal_dir, "torch", "lib"),
        # DLL data shipped by the cupy wheel.
        os.path.join(internal_dir, "cupy", ".data", "lib"),
        # ffmpeg DLLs shipped by PyNvVideoCodec.
        os.path.join(internal_dir, "PyNvVideoCodec"),
        os.path.join(internal_dir, "pynvvideocodec"),
    ]
    # nvidia-*-cu12 wheels: nvidia\cuda_nvrtc\bin, nvidia\cuda_runtime\bin, and similar directories.
    nvidia_root = os.path.join(internal_dir, "nvidia")
    if _isdir(nvidia_root):
        for sub in os.listdir(nvidia_root):
            sub_dir = os.path.join(nvidia_root, sub)
            if not _isdir(sub_dir):
                continue
            for leaf in ("bin", "lib"):
                p = os.path.join(sub_dir, leaf)
                if _isdir(p):
                    candidates.append(p)

    seen: set[str] = set()
    path_parts: list[str] = []
    for d in candidates:
        key = os.path.normcase(os.path.abspath(d)) if d else ""
        if not _isdir(d) or key in seen:
            continue
        seen.add(key)
        path_parts.append(d)
        _add_dll_dir(d)

    if path_parts:
        existing = os.environ.get("PATH", "")
        os.environ["PATH"] = os.pathsep.join(path_parts) + (os.pathsep + existing if existing else "")


_setup()
