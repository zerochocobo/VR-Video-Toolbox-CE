"""Built-in mosaic-removal engine (native_gpu): in-process integration of Lada's torch pipeline.

YOLO11-seg detection + BasicVSR++ restoration run fully on GPU with torch CUDA,
without invoking the lada-cli subprocess. Models are loaded once. The vendored
Lada source lives in `_vendor/lada` under AGPL-3.0; see _vendor/LICENSE.lada.md.

CUDA coexistence: current dependencies are unified on CUDA 12.8 wheels.
`_prepare()` still runs gpu_engine warmup first and redirects mmengine/yapf caches
to the project runtime_cache to avoid blocked writes outside the sandbox.

Public API:
    available() -> bool        whether the engine is available, requiring torch.cuda and model files
    restore_file(in, out, ...) restore one video file in-process
    restore_sbs_stream(...)    one_click SBS streaming path without intermediate video files
"""
from __future__ import annotations

import os
import sys
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR = os.path.join(_HERE, "_vendor")
_engine = None
_lock = threading.Lock()
_prepared = False


def _redirect_yapf_cache():
    """mmengine imports yapf, which writes grammar cache outside the workspace by default.

    In sandboxed/dev runs that external cache write can hang. Patch platformdirs before
    mmengine/yapf import so the cache lives under project runtime_cache.
    """
    try:
        import platformdirs

        if getattr(platformdirs.user_cache_dir, "_vrtb_patched", False):
            return
        cache_root = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "runtime_cache", "yapf_cache")
        os.makedirs(cache_root, exist_ok=True)

        def _user_cache_dir(appname=None, appauthor=None, version=None, *args, **kwargs):
            parts = [cache_root]
            if appname:
                parts.append(str(appname))
            if version:
                parts.append(str(version))
            path = os.path.join(*parts)
            os.makedirs(path, exist_ok=True)
            return path

        _user_cache_dir._vrtb_patched = True
        platformdirs.user_cache_dir = _user_cache_dir
    except Exception:
        pass


def _prepare():
    """Prepare the native engine runtime environment and add vendored Lada to import paths."""
    global _prepared
    if _prepared:
        return
    _redirect_yapf_cache()
    # 1) Warm up GPU/CuPy/PyNv first, matching the main GPU pipeline.
    try:
        from gpu_engine import runtime
        runtime.warmup()
    except Exception:
        pass
    # 2) Add vendored Lada to import paths because it uses absolute imports such as `from lada ...`.
    if _VENDOR not in sys.path:
        sys.path.insert(0, _VENDOR)
    _prepared = True


def available() -> bool:
    """Return true when torch.cuda is available and detection/restoration model files exist."""
    try:
        _prepare()
        import torch
        if not torch.cuda.is_available():
            return False
        from .models_cfg import detection_model_path, restoration_model_path
        return os.path.isfile(detection_model_path()) and os.path.isfile(restoration_model_path())
    except Exception:
        return False


def get_engine():
    """Return the in-process singleton engine, loading models on first use."""
    global _engine
    with _lock:
        if _engine is None:
            _prepare()
            from .engine import NativeMosaicEngine
            _engine = NativeMosaicEngine()
        return _engine


def restore_file(input_path, output_path, *, log_callback=None, cancel_token=None,
                 max_clip_length=180):
    """Restore one file in-process for mosaic removal."""
    return get_engine().restore_file(
        input_path, output_path,
        log_callback=log_callback, cancel_token=cancel_token,
        max_clip_length=max_clip_length,
    )


def restore_sbs_stream(input_path, output_path, *, use_fisheye: bool,
                       start_sec=None, end_sec=None, bitrate_bps=None,
                       log_callback=None, cancel_token=None):
    """one_click dual-eye SBS: GPU frame source -> LADA -> combine/encode, without intermediate video files."""
    return get_engine().restore_sbs_stream(
        input_path, output_path,
        use_fisheye=use_fisheye,
        start_sec=start_sec,
        end_sec=end_sec,
        bitrate_bps=bitrate_bps,
        log_callback=log_callback,
        cancel_token=cancel_token,
    )


def restore_single_eye_stream(input_path, output_path, *, eye_mode: str,
                              use_fisheye: bool, start_sec=None, end_sec=None,
                              bitrate_bps=None, log_callback=None,
                              cancel_token=None):
    """one_click single-eye path: GPU frame source -> LADA -> encode, without intermediate video files."""
    return get_engine().restore_single_eye_stream(
        input_path, output_path,
        eye_mode=eye_mode,
        use_fisheye=use_fisheye,
        start_sec=start_sec,
        end_sec=end_sec,
        bitrate_bps=bitrate_bps,
        log_callback=log_callback,
        cancel_token=cancel_token,
    )
