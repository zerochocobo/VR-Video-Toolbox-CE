"""VR Video Toolbox (CUDA EDITION) onedir packaging script.

Execution order:
  1. Clean old build / dist artifacts.
  2. Run PyInstaller with VR_Video_Toolbox.spec for the main GUI, including CuPy / PyNvVideoCodec / torch / CUDA wheels.
  3. Run PyInstaller with VR_DLNA_Server.spec for the standalone lightweight DLNA Server exe with no GPU dependencies.
  4. Merge vr_dlna_server.exe and its _internal directory into the main dist directory.
  5. Copy runtime DLLs and headers from the system CUDA Toolkit / .venv wheels so the release can run without system dependencies.
  6. Verify required files exist and fail if anything is missing.

Design principles for system-independent packaging:
  - CUDA 12.8 nvrtc/runtime/CCCL comes from nvidia-*-cu12 wheels and is collected into _internal\\nvidia\\ by collect_all("nvidia").
  - cuDNN / cuBLAS / cuFFT come from the torch wheel and are collected into _internal\\torch\\lib\\ by collect_all("torch").
  - PyNvVideoCodec ships its own ffmpeg DLLs, which are collected into _internal\\PyNvVideoCodec\\.
  - System CUDA Toolkit headers such as cuda_fp16.h are copied only as a fallback when nvrtc JIT needs them.
  - User-level ffmpeg.exe / lada-cli live beside the exe, matching the old onefile layout; the runtime hook prepends the exe directory to PATH.
  - The runtime hook (packaging/runtime_hook_cuda.py) forces CUDA_PATH / CUDA_HOME to the dist layout and does not read system registry or environment values.
"""
from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP_NAME = "VR_Video_Toolbox"
DLNA_NAME = "vr_dlna_server"
DIST = ROOT / "dist" / APP_NAME
DLNA_DIST = ROOT / "dist" / DLNA_NAME
INTERNAL = DIST / "_internal"
VENV_SITE = ROOT / ".venv" / "Lib" / "site-packages"


class BuildError(RuntimeError):
    pass


def info(msg: str) -> None:
    print(f"[build] {msg}", flush=True)


def fail(msg: str) -> None:
    raise BuildError(msg)


def on_rm_error(func, path, exc_info) -> None:
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        raise


def remove_path(p: Path) -> None:
    if not p.exists():
        return
    if p.is_dir():
        shutil.rmtree(p, onerror=on_rm_error)
    else:
        try:
            p.chmod(stat.S_IWRITE)
        except OSError:
            pass
        p.unlink()
    if p.exists():
        fail(f"Failed to remove {p}. Stop any running packaged process and retry.")


def copy_file(src: Path, dst: Path) -> None:
    src, dst = src.resolve(), dst.resolve()
    if src == dst:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            ss, ds = src.stat(), dst.stat()
            if ss.st_size == ds.st_size and int(ss.st_mtime) == int(ds.st_mtime):
                return
            dst.chmod(stat.S_IWRITE)
        except OSError:
            pass
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path, *, ignore=None) -> None:
    if not src.exists():
        fail(f"Required directory not found: {src}")
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        if ignore and ignore(item, rel):
            continue
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            copy_file(item, target)


# --------- PyInstaller invocation ---------

def python_has_pyinstaller(python: Path) -> bool:
    r = subprocess.run(
        [str(python), "-c", "import PyInstaller"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return r.returncode == 0


def pyinstaller_cmd() -> list[str]:
    python = Path(sys.executable).resolve()
    if python_has_pyinstaller(python):
        return [str(python), "-m", "PyInstaller"]
    uv = shutil.which("uv")
    if uv:
        r = subprocess.run(
            [uv, "run", "python", "-c", "import PyInstaller"],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if r.returncode == 0:
            return [uv, "run", "pyinstaller"]
    fail("PyInstaller not installed. Run `uv sync` or `pip install pyinstaller`.")


def run(cmd: list[str]) -> None:
    info(" ".join(f'"{c}"' if " " in c else c for c in cmd))
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode:
        fail(f"Command failed (exit {r.returncode}): {' '.join(cmd)}")


def clean() -> None:
    info("cleaning build/dist artifacts")
    for p in (
        ROOT / "build",
        ROOT / "dist" / APP_NAME,
        ROOT / "dist" / DLNA_NAME,
    ):
        remove_path(p)


def build_main(pyi: list[str]) -> None:
    info("building main GUI exe (VR_Video_Toolbox)")
    run([*pyi, "--clean", "--noconfirm", "VR_Video_Toolbox.spec"])


def build_dlna(pyi: list[str]) -> None:
    info("building standalone DLNA server exe (vr_dlna_server)")
    run([*pyi, "--clean", "--noconfirm", "VR_DLNA_Server.spec"])


def merge_dlna_into_main() -> None:
    """Merge vr_dlna_server.exe and its _internal directory into the main dist.

    The main dist already has a complete _internal directory. The DLNA _internal
    directory is only the minimal set needed for standalone execution. Most files
    are already included in the main _internal directory, but the PyInstaller
    bootloader for vr_dlna_server.exe still loads its runtime files such as
    python3X.dll and base_library.zip from _internal. Sharing one _internal
    directory is therefore valid when the Python version is the same and the
    hidden imports are a subset.
    """
    if not DLNA_DIST.exists():
        fail(f"DLNA dist not produced: {DLNA_DIST}")
    info(f"merging {DLNA_DIST.name} into {DIST.name}")
    src_exe = DLNA_DIST / f"{DLNA_NAME}.exe"
    if not src_exe.exists():
        fail(f"DLNA exe missing: {src_exe}")
    copy_file(src_exe, DIST / f"{DLNA_NAME}.exe")
    src_internal = DLNA_DIST / "_internal"
    if src_internal.exists():
        # Copy only files absent from the main _internal directory to avoid overwriting the fuller GPU build.
        for item in src_internal.rglob("*"):
            if item.is_dir():
                continue
            rel = item.relative_to(src_internal)
            target = INTERNAL / rel
            if target.exists():
                continue
            copy_file(item, target)


# --------- CUDA / cuDNN runtime bundling ---------

def system_cuda_toolkit() -> Path | None:
    """Detect the system CUDA Toolkit, used only as a small fallback source for headers."""
    for env_key in ("CUDA_PATH", "CUDA_HOME"):
        p = os.environ.get(env_key, "")
        if p and Path(p).exists():
            return Path(p)
    base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if base.exists():
        versions = sorted(
            (d for d in base.iterdir() if d.is_dir() and d.name.startswith("v")),
            reverse=True,
        )
        if versions:
            return versions[0]
    return None


def ensure_cuda_headers() -> None:
    """Ensure nvrtc JIT headers such as cuda_fp16.h, preferring wheel headers and falling back to the system toolkit."""
    # nvidia-cuda-cccl-cu12 wheel ships include/.
    cccl_inc = INTERNAL / "nvidia" / "cuda_cccl" / "include"
    target_inc = DIST / "include"
    if cccl_inc.exists():
        info(f"using nvidia-cuda-cccl wheel headers: {cccl_inc}")
        target_inc.mkdir(parents=True, exist_ok=True)
        copy_tree(cccl_inc, target_inc,
                  ignore=lambda i, r: i.suffix.lower() in {".lib", ".pdb"} or i.name == "__pycache__")
        # Check whether the key headers are already complete.
        if (target_inc / "cuda_fp16.h").exists():
            return

    toolkit = system_cuda_toolkit()
    if toolkit and (toolkit / "include").exists():
        info(f"copying CUDA headers from system toolkit: {toolkit}")
        copy_tree(toolkit / "include", target_inc,
                  ignore=lambda i, r: i.suffix.lower() in {".lib", ".pdb"} or i.name == "__pycache__")
        return

    if not (target_inc / "cuda_fp16.h").exists():
        info("WARNING: cuda_fp16.h not found in dist. nvrtc JIT may fail at runtime.")


def _file_size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return -1


def dedupe_root_vs_torch_lib() -> None:
    """Remove DLLs duplicated between _internal\\ root and _internal\\torch\\lib.

    PyInstaller's collect_all("torch") frequently emits the same DLL twice:
    once at _internal\\torch\\lib\\<name>.dll (where torch's loader expects it)
    and once at _internal\\<name>.dll (PyInstaller flattening). Keep only the
    torch\\lib copy because torch resolves its DLLs relative to that directory.
    """
    torch_lib = INTERNAL / "torch" / "lib"
    if not torch_lib.exists():
        return
    removed = 0
    bytes_freed = 0
    for src in torch_lib.glob("*.dll"):
        dup = INTERNAL / src.name
        if not dup.is_file():
            continue
        if _file_size(dup) != _file_size(src):
            continue
        try:
            sz = _file_size(dup)
            dup.unlink()
            removed += 1
            bytes_freed += max(sz, 0)
        except OSError:
            pass
    if removed:
        info(f"deduped {removed} torch DLL copies from _internal root ({bytes_freed/1024/1024:.1f} MiB)")


def dedupe_root_vs_nvidia_wheels() -> None:
    """Remove DLLs duplicated between _internal\\ root and _internal\\nvidia\\*\\bin."""
    nvidia_root = INTERNAL / "nvidia"
    if not nvidia_root.exists():
        return
    removed = 0
    bytes_freed = 0
    for src in nvidia_root.rglob("*.dll"):
        dup = INTERNAL / src.name
        if not dup.is_file():
            continue
        if _file_size(dup) != _file_size(src):
            continue
        try:
            sz = _file_size(dup)
            dup.unlink()
            removed += 1
            bytes_freed += max(sz, 0)
        except OSError:
            pass
    if removed:
        info(f"deduped {removed} nvidia wheel DLL copies from _internal root ({bytes_freed/1024/1024:.1f} MiB)")


def prune_torch_bloat() -> None:
    """Strip torch artifacts not needed at runtime: C++ headers, static libs, tests, docs.

    These are required for `pip install torch` build extensions but never loaded by the
    frozen GUI. Removing them shaves hundreds of MB without affecting functionality.
    """
    torch_root = INTERNAL / "torch"
    if not torch_root.exists():
        return
    bytes_freed = 0

    # Whole subdirectories that the runtime never reads.
    for rel in (
        "include",                # C++ headers (~100+ MiB)
        "share/cmake",            # cmake configs
        "share/man",
        "test",                   # bundled test scripts/fixtures
        "utils/benchmark",        # benchmarking helpers
        "utils/model_dump",
        "utils/bottleneck",
        "_C_flatbuffer.pyi",
    ):
        target = torch_root / rel
        if target.exists():
            try:
                if target.is_dir():
                    for p in target.rglob("*"):
                        if p.is_file():
                            bytes_freed += _file_size(p)
                    shutil.rmtree(target, onerror=on_rm_error)
                else:
                    bytes_freed += _file_size(target)
                    target.unlink()
            except OSError:
                pass

    # Per-extension trim under torch\lib: *.lib (static), *.pdb (debug), *.exp/.h
    torch_lib = torch_root / "lib"
    if torch_lib.exists():
        for p in list(torch_lib.iterdir()):
            if p.suffix.lower() in {".lib", ".pdb", ".exp", ".h"}:
                try:
                    bytes_freed += _file_size(p)
                    p.unlink()
                except OSError:
                    pass

    # Drop torchgen if collected — pure codegen scripts, not a runtime dep here.
    torchgen = INTERNAL / "torchgen"
    if torchgen.exists():
        try:
            for p in torchgen.rglob("*"):
                if p.is_file():
                    bytes_freed += _file_size(p)
            shutil.rmtree(torchgen, onerror=on_rm_error)
        except OSError:
            pass

    if bytes_freed:
        info(f"pruned torch bloat: freed {bytes_freed/1024/1024:.1f} MiB")


def prune_nvidia_bloat() -> None:
    """Strip nvidia-*-cu12 wheel headers/static libs not needed at runtime."""
    nvidia_root = INTERNAL / "nvidia"
    if not nvidia_root.exists():
        return
    bytes_freed = 0
    for p in list(nvidia_root.rglob("*")):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix in {".lib", ".a", ".pdb", ".exp"}:
            try:
                bytes_freed += _file_size(p)
                p.unlink()
            except OSError:
                pass
    if bytes_freed:
        info(f"pruned nvidia wheel bloat: freed {bytes_freed/1024/1024:.1f} MiB")


# --------- Verification ---------

REQUIRED_DLLS = (
    "cudart64_12.dll",
    "nvrtc64_120_0.dll",
)
OPTIONAL_DLLS = (
    "cudnn64_9.dll",
    "cublas64_12.dll",
    "cufft64_11.dll",
)


def _exists_anywhere(name: str) -> Path | None:
    # Search the actual bundled wheel locations. No DLL consolidation happens any
    # more — the runtime hook exposes torch/lib, nvidia/*/bin, cupy/.data/lib to
    # the DLL loader directly, so we only need to confirm the files exist there.
    for d in (
        INTERNAL,
        INTERNAL / "torch" / "lib",
        INTERNAL / "nvidia",
        INTERNAL / "cupy" / ".data" / "lib",
        DIST,
    ):
        if not d.exists():
            continue
        for hit in d.rglob(name):
            return hit
    return None


def verify() -> None:
    info("verifying dist contents")
    if not (DIST / f"{APP_NAME}.exe").exists():
        fail(f"Main exe missing: {DIST / (APP_NAME + '.exe')}")
    if not (DIST / f"{DLNA_NAME}.exe").exists():
        fail(f"DLNA exe missing: {DIST / (DLNA_NAME + '.exe')}")

    for name in REQUIRED_DLLS:
        if not _exists_anywhere(name):
            fail(f"Required runtime DLL missing in dist: {name}")
    for name in OPTIONAL_DLLS:
        hit = _exists_anywhere(name)
        if hit is None:
            info(f"NOTE: optional DLL not bundled: {name}")
        else:
            info(f"found {name} -> {hit.relative_to(DIST)}")

    # CuPy extension .pyd files.
    carray = list((INTERNAL / "cupy" / "_core").glob("_carray*.pyd")) if (INTERNAL / "cupy" / "_core").exists() else []
    if not carray:
        fail(r"Missing CuPy extension _carray*.pyd under _internal\cupy\_core (CuPy onedir 不完整).")

    # PyNvVideoCodec extension files.
    pynv_dir = INTERNAL / "PyNvVideoCodec"
    if pynv_dir.exists():
        if not list(pynv_dir.glob("PyNvVideoCodec_*.pyd")):
            fail(r"Missing PyNvVideoCodec_*.pyd under _internal\PyNvVideoCodec.")
    else:
        info("WARNING: PyNvVideoCodec directory not found in _internal; GPU decode 不可用.")

    info("verification passed.")


# --------- Entry ---------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build VR Video Toolbox (CUDA EDITION) onedir release.")
    p.add_argument("--skip-clean", action="store_true", help="跳过清理 build/dist 旧产物")
    p.add_argument("--skip-main", action="store_true", help="跳过主 GUI 打包")
    p.add_argument("--skip-dlna", action="store_true", help="跳过 DLNA 打包")
    p.add_argument("--no-verify", action="store_true", help="跳过最终 verify")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    # Disable UPX because it corrupts CUDA DLLs.
    os.environ.pop("VR_TOOLBOX_USE_UPX", None)

    try:
        pyi = pyinstaller_cmd()
        info(f"using pyinstaller: {' '.join(pyi)}")
        if not args.skip_clean:
            clean()
        if not args.skip_main:
            build_main(pyi)
        if not args.skip_dlna:
            build_dlna(pyi)
        merge_dlna_into_main()
        ensure_cuda_headers()
        # Dedupe + prune to slim the release. The runtime hook adds torch/lib +
        # nvidia/*/bin to the DLL search path, so no DLL needs to live at the
        # _internal root or in a separate bin/ directory.
        dedupe_root_vs_torch_lib()
        dedupe_root_vs_nvidia_wheels()
        prune_torch_bloat()
        prune_nvidia_bloat()
        # Stale dist\bin from earlier builds (no longer produced) — clean it up.
        stale_bin = DIST / "bin"
        if stale_bin.exists():
            remove_path(stale_bin)
            info("removed stale dist/bin (DLL consolidation no longer needed)")
        if not args.no_verify:
            verify()
    except BuildError as e:
        print(f"\n[build] FAILED: {e}", file=sys.stderr)
        return 1

    print()
    info(f"onedir build complete: {DIST}")
    info(f"  main exe: {DIST / (APP_NAME + '.exe')}")
    info(f"  dlna exe: {DIST / (DLNA_NAME + '.exe')}")
    info("分发时压缩整个目录；models/、ffmpeg.exe / lada-cli.exe 请放在 exe 同目录后再压缩。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
