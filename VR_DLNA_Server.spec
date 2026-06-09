# -*- mode: python ; coding: utf-8 -*-
"""VR DLNA Server PyInstaller spec using onedir mode.

Standalone DLNA server entry point (tool_dlna/main.py), using only SSDP + FastAPI
with no GPU dependencies. After packaging as vr_dlna_server.exe, build_exe.py
merges it beside the main program onedir output.
"""
import os
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None
PROJECT_ROOT = os.path.abspath(os.getcwd())

datas = [
    ("i18n", "i18n"),
    ("config", "config"),
]

hiddenimports = []
hiddenimports += collect_submodules("tool_dlna")
hiddenimports += collect_submodules("uvicorn")
hiddenimports += collect_submodules("fastapi")

a = Analysis(
    [os.path.join("tool_dlna", "main.py")],
    pathex=[PROJECT_ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # DLNA does not need GPU/AI dependencies; exclude them explicitly to reduce size.
        "cupy", "cupy_backends", "cupyx", "fastrlock",
        "pynvvideocodec", "PyNvVideoCodec",
        "torch", "torchvision", "torchaudio", "torchgen", "functorch",
        "ultralytics", "mmengine",
        "faster_whisper", "auditok",
        "nvidia",
    ],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="vr_dlna_server",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
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
    name="vr_dlna_server",
)
