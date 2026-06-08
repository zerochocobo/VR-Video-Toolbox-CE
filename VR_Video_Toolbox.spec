# -*- mode: python ; coding: utf-8 -*-
"""VR Video Toolbox (CUDA EDITION) PyInstaller spec using onedir mode with bundled CuPy + PyNvVideoCodec + CUDA.

Key points:
  - Use onedir (COLLECT), with UPX disabled because it corrupts CUDA DLLs.
  - Bundle cupy / pynvvideocodec / cuda.pathfinder and their DLLs through collect_all.
  - packaging/hook-cupy.py fills in CuPy Cython extension .pyd files.
  - packaging/runtime_hook_cuda.py configures the CUDA environment at frozen startup, including PTX and DLL paths.
  - Bundle i18n/ and config/ as data; keep models/ outside beside the exe for the user to provide.

CUDA DLL sources: the development machine may use the system CUDA v13.0 toolkit.
If pip nvidia-cuda-* wheels are used instead, collect_all('cupy') brings the wheel DLLs.
Otherwise, copy cudart/nvrtc/nvrtc-builtins from v13.0\\bin and the include headers
into dist after build; see the copy steps in build_exe.bat.
"""
import os
import sys
from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None
PROJECT_ROOT = os.path.abspath(os.getcwd())

# Make the vendored Lada importable at build time so PyInstaller compiles its
# modules into the PYZ (i.e. into the exe), instead of shipping loose .py files.
_VENDOR_ABS = os.path.join(PROJECT_ROOT, "gpu_engine", "native_mosaic", "_vendor")
if _VENDOR_ABS not in sys.path:
    sys.path.insert(0, _VENDOR_ABS)
_DA3_VENDOR_ABS = os.path.join(PROJECT_ROOT, "tool_2dvr", "_vender", "da3")
if os.path.isdir(_DA3_VENDOR_ABS) and _DA3_VENDOR_ABS not in sys.path:
    sys.path.insert(0, _DA3_VENDOR_ABS)
_QWEN_TTS_VENDOR_ABS = os.path.join(PROJECT_ROOT, "tool_si", "_vendor", "qwen_tts")

datas = [
    ("i18n", "i18n"),
    ("config", "config"),
]
binaries = []
hiddenimports = [
    "cuda.pathfinder",
    "cuda.pathfinder._dynamic_libs.dynamic_lib_subprocess",
    # stdlib modules lazily imported by cupy and missed by PyInstaller static analysis:
    "graphlib",
]
hiddenimports += collect_submodules("gpu_engine")

# Vendored Lada is imported as the top-level package `lada`. Compile every Lada
# submodule into the PYZ (baked into the exe). _VENDOR_ABS is on sys.path + pathex
# so collect_submodules can enumerate them. on_error="ignore" tolerates the
# training-only subpackages (e.g. datasetcreation) that may fail to import at build
# time; they are not needed for the runtime restoration pipeline.
try:
    hiddenimports += collect_submodules("lada", on_error="ignore")
except TypeError:
    # Older PyInstaller without on_error kwarg.
    hiddenimports += collect_submodules("lada")

# Lada's only non-Python runtime asset: encoding_presets.csv, loaded via
# os.path.dirname(__file__). For a PYZ-frozen top-level `lada`, the module's
# __file__ resolves to <_MEIPASS>/lada/utils/video_utils.pyc, so the CSV must sit
# at _internal/lada/utils/encoding_presets.csv.
_LADA_CSV = os.path.join(_VENDOR_ABS, "lada", "utils", "encoding_presets.csv")
if os.path.isfile(_LADA_CSV):
    datas.append((_LADA_CSV, os.path.join("lada", "utils")))

# Vendored Depth Anything 3 is imported as the top-level package
# `depth_anything_3`. Its config creates modules dynamically from strings, so
# collect all submodules explicitly and ship YAML configs as data.
if os.path.isdir(_DA3_VENDOR_ABS):
    try:
        hiddenimports += collect_submodules("depth_anything_3", on_error="ignore")
    except TypeError:
        hiddenimports += collect_submodules("depth_anything_3")
    _DA3_CONFIGS = os.path.join(_DA3_VENDOR_ABS, "depth_anything_3", "configs")
    if os.path.isdir(_DA3_CONFIGS):
        datas.append((_DA3_CONFIGS, os.path.join("depth_anything_3", "configs")))

# Vendored Qwen3-TTS is imported through tool_si._vendor.qwen_tts. Compile all
# modules into the PYZ, and also mirror the source files under _internal because
# Transformers inspects custom model source with open(module.__file__) during
# from_pretrained().
if os.path.isdir(_QWEN_TTS_VENDOR_ABS):
    try:
        hiddenimports += collect_submodules("tool_si._vendor.qwen_tts", on_error="ignore")
    except TypeError:
        hiddenimports += collect_submodules("tool_si._vendor.qwen_tts")
    for root, _dirs, files in os.walk(_QWEN_TTS_VENDOR_ABS):
        for filename in files:
            if not filename.endswith(".py"):
                continue
            module_path = os.path.join(root, filename)
            rel_module = os.path.relpath(module_path, PROJECT_ROOT)[:-3].replace(os.sep, ".")
            if rel_module.endswith(".__init__"):
                rel_module = rel_module[:-9]
            hiddenimports.append(rel_module)
    datas.append((_QWEN_TTS_VENDOR_ABS, os.path.join("tool_si", "_vendor", "qwen_tts")))

# Bundle GPU/AI dependencies, including their DLLs. torch ships runtime DLLs such as cuDNN/cuBLAS/cuFFT.
# nvidia-cuda-*-cu12 wheels ship cudart/nvrtc + headers, so the build uses wheels without system CUDA.
for pkg in (
    "cupy", "cupy_backends", "cupyx",
    "pynvvideocodec", "PyNvVideoCodec",
    "cuda", "fastrlock",
    "torch", "torchvision",
    "ultralytics", "mmengine",
    "faster_whisper", "auditok",
    "nvidia",
    # 2D->Depth VR DA3 runtime deps. DA3 source is vendored under tool_2dvr;
    # model weights stay outside the exe under models/DA3/Small.
    "omegaconf", "safetensors", "einops",
    # Qwen3-TTS runtime deps. The model weights stay outside the exe under
    # models/Qwen3-TTS-12Hz-0.6B-CustomVoice.
    "transformers", "accelerate", "librosa", "soundfile", "torchaudio",
    "huggingface_hub",
    # Vendored Lada + mmengine runtime deps that PyInstaller's static analysis
    # misses because Lada is bundled as data (not analyzed) and mmengine imports
    # some of these lazily/dynamically: termcolor (Lada), addict + yapf (mmengine
    # Config), scipy + cv2 + PIL already arrive via ultralytics but are pinned
    # here for safety.
    "termcolor", "addict", "yapf",
    "scipy", "cv2", "PIL",
):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

a = Analysis(
    ["main.py"],
    pathex=[PROJECT_ROOT, _VENDOR_ABS, _DA3_VENDOR_ABS],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=["packaging"],
    runtime_hooks=[
        "packaging/runtime_hook_logging.py",
        "packaging/runtime_hook_cuda.py",
    ],
    excludes=[],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,          # onedir: binaries go into COLLECT
    name="VR_Video_Toolbox",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                      # Do not use UPX because it corrupts CUDA DLLs.
    console=False,                  # No console window; logs go to runtime_cache/logs via runtime_hook_logging.py.
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="VR_Video_Toolbox",
)
