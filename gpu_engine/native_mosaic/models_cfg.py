"""Model path resolution for the built-in engine.

Defaults point to the VR detection v2_accurate model and generic restoration v1.2
model under project `models/`; app_config can override them. In frozen packages,
models/ lives beside the exe.
"""
from __future__ import annotations

import os
import sys

_DEFAULT_DETECTION = "lada_vr_mosaic_detection_model_v2_accurate.pt"
_DEFAULT_RESTORATION = "lada_mosaic_restoration_model_generic_v1.2.pth"


def models_dir() -> str:
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        # gpu_engine/native_mosaic/models_cfg.py -> project root.
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, "models")


def _cfg(key: str, default: str) -> str:
    try:
        from utils import app_config
        v = app_config.get(key, default)
        return default if not v else str(v)
    except Exception:
        return default


def detection_model_path() -> str:
    name = _cfg("native_detection_model", _DEFAULT_DETECTION)
    return name if os.path.isabs(name) else os.path.join(models_dir(), name)


def restoration_model_path() -> str:
    name = _cfg("native_restoration_model", _DEFAULT_RESTORATION)
    return name if os.path.isabs(name) else os.path.join(models_dir(), name)
