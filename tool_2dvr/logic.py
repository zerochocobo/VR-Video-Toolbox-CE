"""2D video to depth-based VR conversion.

This module intentionally keeps Depth Anything 3 integration behind a small
adapter. The video pipeline and stereo rendering are local code, inspired by
the common depth-to-stereo workflow but not copied from depth-surge-3d.

``inverse_warp`` is a separate fast inverse-sampling stereo mode. It does not
produce forward-warp hole masks and is intentionally kept out of the default
``soft_shift`` and ``e2fgvi`` hole-fill semantics.

CPU rendering is only a CUDA fallback path and does not implement the S0/S1
temporal stabilization filters.
"""
from __future__ import annotations

import json
import math
import os
import importlib.util
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from utils.ffmpeg_checker import get_startupinfo
except ImportError:
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from utils.ffmpeg_checker import get_startupinfo


DA3_DIR_NAME = "DA3"
DA3_SMALL_DIR_NAME = "Small"
PROJECTION_FLAT_3D = "flat3d"
PROJECTION_HEQUIRECT = "hequirect"
PROJECTION_FISHEYE = "fisheye"
DEFAULT_PROJECTION = PROJECTION_FLAT_3D
DEFAULT_EYE_DISTANCE_MM = 65.0
DEFAULT_DEPTH_BATCH_SIZE = 8
DEFAULT_MAX_DISPARITY_RATIO = 0.035
DEFAULT_HOLE_FILL_MODE = "soft_shift"
HOLE_FILL_E2FGVI = "e2fgvi"
HOLE_FILL_INVERSE_WARP = "inverse_warp"
HOLE_FILL_MODES = {
    "soft_shift", "shift_fill", "background", "inpaint",
    HOLE_FILL_E2FGVI, HOLE_FILL_INVERSE_WARP, "none",
}
DEFAULT_E2FGVI_CHUNK_SIZE = 12
DEFAULT_FLAT_FOV_DEG = 80.0
MIN_FLAT_FOV_DEG = 1.0
MAX_FLAT_FOV_DEG = 179.0
STABILIZE_OFF = "off"
STABILIZE_AUTO = "auto"
STABILIZE_FULL = "full"
DEFAULT_STABILIZE_MODE = STABILIZE_AUTO
STABILIZE_MODES = {STABILIZE_OFF, STABILIZE_AUTO, STABILIZE_FULL}
BACKEND_PYNV = "pynv"
BACKEND_FFMPEG = "ffmpeg"
BACKEND_AUTO = "auto"
DEFAULT_2DVR_BACKEND = BACKEND_AUTO
BACKEND_MODES = {BACKEND_PYNV, BACKEND_FFMPEG, BACKEND_AUTO}
# NVDEC-capable codecs we consider safe to drive the PyNv pipeline. Anything
# outside this set falls back to ffmpeg automatically.
PYNV_SUPPORTED_CODECS = {"h264", "hevc", "h265", "vp9", "av1", "mpeg4", "mpeg2video", "vc1"}
YUV_INVERSE_COEFFS = {
    # (y_scale, r_cr, g_cb, g_cr, b_cb): R = y_scale*(Y-16) + r_cr*(Cr-128) etc.
    "bt601": (1.16438356, 1.59602678, 0.39176229, 0.81296765, 2.01723214),
    "bt709": (1.16438356, 1.79274107, 0.21324861, 0.53290933, 2.11240179),
}
YUV_LIMITED_COEFFS = {
    "bt601": (
        (0.256788, 0.504129, 0.097906),
        (-0.148223, -0.290993, 0.439216),
        (0.439216, -0.367788, -0.071427),
    ),
    "bt709": (
        (0.182586, 0.614231, 0.062007),
        (-0.100644, -0.338572, 0.439216),
        (0.439216, -0.398942, -0.040274),
    ),
}


class TwoDVRRuntimeError(RuntimeError):
    """Raised when the 2D-to-VR pipeline cannot continue."""


class OperationCancelled(Exception):
    """Raised when the user stops the conversion."""


class PipelineProcess:
    """Small process-group handle compatible with the existing GUI stop logic."""

    def __init__(self):
        self._processes: list[subprocess.Popen] = []
        self._cancelled = False

    def add(self, process: subprocess.Popen) -> None:
        self._processes.append(process)
        if self._cancelled:
            self.kill()

    def kill(self) -> None:
        self._cancelled = True
        for proc in list(self._processes):
            try:
                if proc.poll() is None:
                    proc.kill()
            except Exception:
                pass

    def terminate(self) -> None:
        self.kill()

    def poll(self):
        if self._cancelled:
            return 0
        return None

    @property
    def cancelled(self) -> bool:
        return self._cancelled


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float
    codec_name: str = ""


@dataclass(frozen=True)
class ProjectionMap:
    out_w: int
    out_h: int
    map_x: np.ndarray
    map_y: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class StereoRenderResult:
    left: np.ndarray
    right: np.ndarray
    near: np.ndarray
    disparity: np.ndarray
    left_before_fill: np.ndarray
    right_before_fill: np.ndarray
    left_holes: np.ndarray
    right_holes: np.ndarray


class _AsyncSbsBatch:
    def __init__(self, cpu_tensor: Any, event: Any, gpu_tensor: Any | None = None):
        self.cpu_tensor = cpu_tensor
        self.event = event
        self.gpu_tensor = gpu_tensor
        self.frame_count = int(cpu_tensor.shape[0]) if getattr(cpu_tensor, "ndim", 0) > 0 else 0

    def to_numpy(self) -> np.ndarray:
        self.event.synchronize()
        arr = self.cpu_tensor.numpy()
        # The CUDA event has completed, so the copy stream no longer needs this
        # tensor reference to keep source memory alive.
        self.gpu_tensor = None
        return arr


def _sbs_batch_to_numpy(sbs_batch: Any) -> np.ndarray:
    if isinstance(sbs_batch, _AsyncSbsBatch):
        return sbs_batch.to_numpy()
    return np.asarray(sbs_batch)


def _sbs_batch_is_empty(sbs_batch: Any) -> bool:
    if isinstance(sbs_batch, _AsyncSbsBatch):
        return sbs_batch.frame_count <= 0
    return np.asarray(sbs_batch).size == 0


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_da3_dir() -> Path:
    return project_root() / "models" / DA3_DIR_NAME / DA3_SMALL_DIR_NAME


def da3_vendor_root() -> Path:
    return Path(__file__).resolve().parent / "_vendor" / "da3"


def check_dependencies() -> list[str]:
    missing = []
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            missing.append(tool)
    model_dir = default_da3_dir()
    if not (model_dir / "config.json").exists() or not (model_dir / "model.safetensors").exists():
        missing.append(f"Depth Anything 3 Small model ({model_dir})")
    if not (da3_vendor_root() / "depth_anything_3").exists():
        missing.append(f"vendored DA3 source ({da3_vendor_root()})")
    for module in ("omegaconf", "einops", "safetensors"):
        if importlib.util.find_spec(module) is None:
            missing.append(module)
    return missing


def resolve_depth_batch_size() -> int:
    raw = os.environ.get("TOOL_2DVR_BATCH_SIZE") or os.environ.get("TOOL_2DVR_BATCH")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            return DEFAULT_DEPTH_BATCH_SIZE
    return DEFAULT_DEPTH_BATCH_SIZE


def debug_eye_enabled() -> bool:
    raw = os.environ.get("TOOL_2DVR_DEBUG_EYE", "0").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _env_float(name: str, default: float, min_value: float | None = None, max_value: float | None = None) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        value = float(raw) if raw else float(default)
    except ValueError:
        value = float(default)
    if min_value is not None:
        value = max(float(min_value), value)
    if max_value is not None:
        value = min(float(max_value), value)
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return bool(default)
    return raw not in {"0", "false", "no", "off"}


def resolve_hole_fill_mode(value: str | None = None) -> str:
    raw = (value or os.environ.get("TOOL_2DVR_HOLE_FILL") or DEFAULT_HOLE_FILL_MODE).strip().lower()
    return raw if raw in HOLE_FILL_MODES else DEFAULT_HOLE_FILL_MODE


def resolve_stabilize_mode(value: str | None = None) -> str:
    raw = (value or os.environ.get("TOOL_2DVR_STABILIZE") or DEFAULT_STABILIZE_MODE).strip().lower()
    if raw in {"0", "false", "no"}:
        return STABILIZE_OFF
    if raw in {"1", "true", "yes", "on"}:
        return STABILIZE_AUTO
    return raw if raw in STABILIZE_MODES else DEFAULT_STABILIZE_MODE


def resolve_backend(value: str | None = None) -> str:
    raw = (value or os.environ.get("TOOL_2DVR_BACKEND") or DEFAULT_2DVR_BACKEND).strip().lower()
    if raw in {"0", "false", "no", "off", "cpu"}:
        return BACKEND_FFMPEG
    if raw in {"1", "true", "yes", "on", "gpu"}:
        return BACKEND_PYNV
    return raw if raw in BACKEND_MODES else DEFAULT_2DVR_BACKEND


def _normalize_codec_name(codec_name: str) -> str:
    name = str(codec_name or "").strip().lower()
    if name == "h265":
        return "hevc"
    return name


def pynv_supports_codec(codec_name: str) -> bool:
    return _normalize_codec_name(codec_name) in PYNV_SUPPORTED_CODECS


def resolve_e2fgvi_chunk_size() -> int:
    raw = os.environ.get("TOOL_2DVR_E2FGVI_CHUNK", "").strip()
    try:
        value = int(raw) if raw else DEFAULT_E2FGVI_CHUNK_SIZE
    except ValueError:
        value = DEFAULT_E2FGVI_CHUNK_SIZE
    return max(2, value)


def resolve_depth_norm_samples() -> int:
    raw = os.environ.get("TOOL_2DVR_NORM_SAMPLES", "").strip()
    try:
        value = int(raw) if raw else 8192
    except ValueError:
        value = 8192
    return max(1024, value)


def inverse_warp_compat_enabled() -> bool:
    raw = os.environ.get("TOOL_2DVR_FAST_WARP", "0").strip().lower()
    return raw not in {"0", "false", "no", "off", "forward"}


def resolve_pipe_pix_fmt(out_w: int, out_h: int) -> str:
    raw = os.environ.get("TOOL_2DVR_PIPE_PIXFMT", "yuv420p").strip().lower()
    if raw == "rgb24":
        return "rgb24"
    if out_w % 2 == 0 and out_h % 2 == 0:
        return "yuv420p"
    return "rgb24"


def _yuv_matrix_for_shape(width: int, height: int) -> str:
    return "bt601" if int(height) <= 576 else "bt709"


def _yuv_limited_coefficients(width: int, height: int):
    return YUV_LIMITED_COEFFS[_yuv_matrix_for_shape(width, height)]


def _yuv_inverse_coefficients(width: int, height: int):
    return YUV_INVERSE_COEFFS[_yuv_matrix_for_shape(width, height)]


def _torch_nv12_to_rgb_uint8(y_plane, uv_plane, *, matrix: str | None = None, width: int | None = None, height: int | None = None):
    """Convert NV12 GPU planes to RGB uint8 ``(H, W, 3)`` on the same device.

    ``y_plane``   : torch.uint8 (H, W) on CUDA
    ``uv_plane``  : torch.uint8 (H/2, W/2, 2) interleaved Cb/Cr on CUDA
    ``matrix``    : ``"bt601"`` or ``"bt709"``; auto-selected by ``(width, height)`` if omitted
    """
    import torch
    import torch.nn.functional as F

    if y_plane.ndim != 2:
        raise ValueError(f"y_plane must be (H, W) uint8, got {tuple(y_plane.shape)}")
    h, w = int(y_plane.shape[0]), int(y_plane.shape[1])
    if uv_plane.ndim != 3 or uv_plane.shape[-1] != 2:
        raise ValueError(f"uv_plane must be (H/2, W/2, 2) uint8, got {tuple(uv_plane.shape)}")
    if uv_plane.shape[0] != h // 2 or uv_plane.shape[1] != w // 2:
        raise ValueError(f"uv_plane shape {tuple(uv_plane.shape)} does not match Y plane {(h, w)}")
    if matrix is None:
        key = _yuv_matrix_for_shape(width if width is not None else w, height if height is not None else h)
    else:
        key = "bt601" if str(matrix).lower() in {"bt601", "smpte170m", "bt470bg"} else "bt709"
    y_scale, r_cr, g_cb, g_cr, b_cb = YUV_INVERSE_COEFFS[key]

    yf = y_plane.to(torch.float32) - 16.0
    cb_small = uv_plane[..., 0].to(torch.float32) - 128.0  # (H/2, W/2)
    cr_small = uv_plane[..., 1].to(torch.float32) - 128.0
    # Upsample chroma to luma resolution (nearest = simple replicate, matches NV12 default ffmpeg behaviour).
    cb = F.interpolate(cb_small[None, None], size=(h, w), mode="nearest")[0, 0]
    cr = F.interpolate(cr_small[None, None], size=(h, w), mode="nearest")[0, 0]
    yf = yf * y_scale
    r = (yf + r_cr * cr).clamp_(0.0, 255.0)
    g = (yf - g_cb * cb - g_cr * cr).clamp_(0.0, 255.0)
    b = (yf + b_cb * cb).clamp_(0.0, 255.0)
    return torch.stack((r, g, b), dim=-1).round_().to(torch.uint8)


def _torch_rgb_uint8_to_nv12_packed(rgb, *, matrix: str | None = None):
    """Convert RGB uint8 ``(H, W, 3)`` or batched ``(B, H, W, 3)`` to packed NV12 layout.

    Packed shape: ``(H*3//2, W)`` per frame, with Y on top H rows and interleaved
    UV on the bottom H/2 rows (each row = W bytes = W/2 Cb + W/2 Cr). Matches the
    layout expected by ``gpu_engine.pynv_io.GpuNv12AppFrame``.
    """
    import torch
    import torch.nn.functional as F

    if rgb.ndim == 3:
        rgb = rgb.unsqueeze(0)
        squeeze_back = True
    else:
        squeeze_back = False
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"rgb must be (H,W,3) or (B,H,W,3) uint8, got {tuple(rgb.shape)}")
    b, h, w, _ = rgb.shape
    if h % 2 or w % 2:
        raise ValueError(f"NV12 requires even dimensions, got {w}x{h}")

    if matrix is None:
        key = _yuv_matrix_for_shape(w, h)
    else:
        key = "bt601" if str(matrix).lower() in {"bt601", "smpte170m", "bt470bg"} else "bt709"
    y_coeff, u_coeff, v_coeff = YUV_LIMITED_COEFFS[key]

    x = rgb.to(torch.float32)
    r = x[..., 0]
    g = x[..., 1]
    bch = x[..., 2]
    y = (16.0 + y_coeff[0] * r + y_coeff[1] * g + y_coeff[2] * bch).clamp_(0.0, 255.0)
    u_full = (128.0 + u_coeff[0] * r + u_coeff[1] * g + u_coeff[2] * bch).clamp_(0.0, 255.0)
    v_full = (128.0 + v_coeff[0] * r + v_coeff[1] * g + v_coeff[2] * bch).clamp_(0.0, 255.0)
    u = F.avg_pool2d(u_full.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
    v = F.avg_pool2d(v_full.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)

    y_u8 = y.round_().to(torch.uint8)                 # (B, H, W)
    uv_pair = torch.stack((u, v), dim=-1).round_().to(torch.uint8)  # (B, H/2, W/2, 2)
    # (B, H/2, W/2, 2) -> (B, H/2, W) with Cb,Cr interleaved per pair of columns.
    uv_row = uv_pair.reshape(b, h // 2, w)
    packed = torch.cat([y_u8, uv_row], dim=1).contiguous()
    return packed[0] if squeeze_back else packed


def _ffmpeg_output_color_args(out_w: int, out_h: int) -> list[str]:
    matrix = "smpte170m" if _yuv_matrix_for_shape(out_w, out_h) == "bt601" else "bt709"
    return [
        "-colorspace", matrix,
        "-color_primaries", matrix,
        "-color_trc", matrix,
        "-color_range", "tv",
    ]


def _hevc_color_metadata_bsf(out_w: int, out_h: int) -> str:
    code = 6 if _yuv_matrix_for_shape(out_w, out_h) == "bt601" else 1
    return (
        "hevc_metadata="
        f"colour_primaries={code}:"
        f"transfer_characteristics={code}:"
        f"matrix_coefficients={code}:"
        "video_full_range_flag=0"
    )


def default_debug_output_dir() -> Path:
    return project_root() / "debug_output" / "tool_2dvr"


def resolve_debug_output_dir() -> Path:
    raw = os.environ.get("TOOL_2DVR_DEBUG_DIR", "").strip()
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = project_root() / path
        return path
    return default_debug_output_dir()


def debug_output_stem(output: str) -> Path:
    stem = Path(output).with_suffix("").name
    return resolve_debug_output_dir() / stem / stem


def parse_time_to_seconds(value: str | None) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    parts = text.split(":")
    if len(parts) > 3:
        raise ValueError("time must be SS, MM:SS, or HH:MM:SS")
    nums = [float(p) for p in parts]
    if any(n < 0 for n in nums):
        raise ValueError("time cannot be negative")
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return nums[0] * 60.0 + nums[1]
    return nums[0] * 3600.0 + nums[1] * 60.0 + nums[2]


def _safe_time_suffix(value: str | None, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    return "".join(ch for ch in text if ch.isdigit() or ch == ".").replace(".", "p") or fallback


def output_path(input_path: str, output_dir: str | None, projection: str,
                start_time: str | None = None, end_time: str | None = None) -> str:
    src = Path(input_path)
    out_dir = Path(output_dir) if output_dir else src.parent
    stem = src.stem
    seg = ""
    if start_time or end_time:
        seg = f"_S{_safe_time_suffix(start_time, 'START')}_E{_safe_time_suffix(end_time, 'END')}"
    if projection == PROJECTION_FLAT_3D:
        return str(out_dir / f"{stem}{seg}_2dvr_flat3d_LR_SBS.mp4")
    proj = "fisheye" if projection == PROJECTION_FISHEYE else "hequirect"
    return str(out_dir / f"{stem}{seg}_2dvr_{proj}_LR_180_SBS.mp4")


def _parse_fps(rate: str) -> float:
    text = str(rate or "").strip()
    if "/" in text:
        num, den = text.split("/", 1)
        den_f = float(den)
        return float(num) / den_f if den_f else 0.0
    return float(text or 0.0)


def get_video_info(input_path: str) -> VideoInfo:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,codec_name,avg_frame_rate,r_frame_rate:format=duration",
        "-of", "json",
        input_path,
    ]
    raw = subprocess.check_output(
        cmd, startupinfo=get_startupinfo(), text=True, encoding="utf-8", errors="replace"
    )
    data = json.loads(raw)
    streams = data.get("streams") or []
    if not streams:
        raise TwoDVRRuntimeError("No video stream found.")
    stream = streams[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    fps = _parse_fps(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0")
    duration = float((data.get("format") or {}).get("duration") or 0.0)
    if width <= 0 or height <= 0 or fps <= 0:
        raise TwoDVRRuntimeError(f"Invalid video metadata: {width}x{height} fps={fps}")
    codec_name = str(stream.get("codec_name") or "").lower()
    return VideoInfo(width=width, height=height, fps=fps, duration=duration, codec_name=codec_name)


def _add_da3_import_paths() -> None:
    vendor_root = da3_vendor_root()
    if not (vendor_root / "depth_anything_3").exists():
        raise TwoDVRRuntimeError(f"Vendored DA3 source not found: {vendor_root}")
    p = str(vendor_root)
    if p not in sys.path:
        sys.path.insert(0, p)


def _has_model_file(path: Path) -> bool:
    if not path.is_dir():
        return False
    model_names = {
        "model.safetensors",
        "pytorch_model.bin",
        "model.bin",
        "config.json",
        "model_index.json",
    }
    return any((path / name).exists() for name in model_names)


def resolve_da3_model_path(model_root: str | Path | None = None) -> Path:
    root = Path(model_root) if model_root else default_da3_dir()
    candidates = [
        root,
        root / "Small",
        root / "SMALL",
        root / "Depth-Anything-3-Small",
        root / "depth-anything-3-small",
        root / "da3-small",
        root / "small",
    ]
    for candidate in candidates:
        if _has_model_file(candidate):
            return candidate
    raise TwoDVRRuntimeError(
        "Depth Anything 3 Small model files were not found. "
        f"Put the local small model under {root}."
    )


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.cpu().numpy()
    return np.asarray(value)


def _extract_depth(prediction: Any) -> np.ndarray:
    arr = _extract_depths(prediction)
    if arr.shape[0] < 1:
        raise TwoDVRRuntimeError("DA3 prediction returned no depth maps.")
    return arr[0].astype(np.float32, copy=False)


def _extract_depths(prediction: Any) -> np.ndarray:
    value = prediction
    if isinstance(prediction, dict):
        for key in ("depth", "depths", "pred_depth", "depth_map", "depth_maps"):
            if key in prediction:
                value = prediction[key]
                break
        else:
            raise TwoDVRRuntimeError(f"DA3 prediction has no depth key: {list(prediction.keys())}")
    elif hasattr(prediction, "depth"):
        value = prediction.depth
    arr = _to_numpy(value)
    if arr.ndim == 5 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim == 5 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise TwoDVRRuntimeError(f"Unexpected DA3 depth shape: {arr.shape} from {type(value).__name__}")
    return arr.astype(np.float32, copy=False)


def _is_torch_tensor(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "device")


def _concat_depth_batches(batches: list[Any]) -> Any:
    if not batches:
        return np.empty((0, 0, 0), dtype=np.float32)
    if all(_is_torch_tensor(batch) for batch in batches):
        import torch

        return torch.cat(batches, dim=0)
    return np.concatenate([_extract_depths(batch) for batch in batches], axis=0)


def _resize_depth_to_frame(depth: np.ndarray, frame_rgb: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.shape == frame_rgb.shape[:2]:
        return depth.astype(np.float32, copy=False)
    depth_img = Image.fromarray(depth.astype(np.float32, copy=False))
    return np.asarray(
        depth_img.resize((frame_rgb.shape[1], frame_rgb.shape[0]), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )


def _depths_for_cpu_frames(depths: Any, frames_rgb: list[np.ndarray]) -> np.ndarray:
    arr = _extract_depths(depths)
    if arr.shape[0] != len(frames_rgb):
        raise TwoDVRRuntimeError(
            f"Depth batch has {arr.shape[0]} maps for {len(frames_rgb)} frames."
        )
    return np.stack(
        [_resize_depth_to_frame(depth, frame) for depth, frame in zip(arr, frames_rgb)],
        axis=0,
    )


class DA3DepthEstimator:
    """Local Depth Anything 3 Small adapter.

    DA3 code is vendored under tool_2dvr/_vendor/da3. Model weights are
    resolved locally from models/DA3/Small and are not downloaded by this adapter.
    """

    def __init__(self, model_root: str | Path | None = None, log_callback=None):
        self.model_root = Path(model_root) if model_root else default_da3_dir()
        _add_da3_import_paths()
        try:
            from depth_anything_3.api import DepthAnything3
        except Exception as exc:
            raise TwoDVRRuntimeError(
                "Vendored Depth Anything 3 is not importable. Check "
                "tool_2dvr/_vendor/da3 and install omegaconf/einops/safetensors."
            ) from exc

        model_path = resolve_da3_model_path(self.model_root)
        if log_callback:
            log_callback(f"[2dvr] Loading DA3 Small from: {model_path}")
        try:
            self.model = DepthAnything3.from_pretrained(str(model_path))
        except Exception as exc:
            raise TwoDVRRuntimeError(f"Failed to load DA3 Small model from {model_path}: {exc}") from exc

        if hasattr(self.model, "to"):
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = self.model.to(device)
                if log_callback:
                    log_callback(f"[2dvr] DA3 device: {device}")
            except Exception:
                pass
        if hasattr(self.model, "eval"):
            self.model.eval()

    def predict_batch(self, frames_rgb) -> Any:
        """Predict depth for a batch.

        ``frames_rgb`` may be ``list[np.ndarray]`` (CPU frames) or a torch tensor
        ``(B, H, W, 3)`` uint8 already on CUDA (PyNv zero-copy path).
        """
        if _is_torch_tensor(frames_rgb):
            batch_count = int(frames_rgb.shape[0])
            if batch_count <= 0:
                import torch

                return torch.empty((0, 0, 0), device=frames_rgb.device, dtype=torch.float32)
            if hasattr(self.model, "inference_depth_only"):
                depths = self.model.inference_depth_only(frames_rgb)
            else:
                # Legacy path doesn't accept tensor; fall back to CPU list.
                cpu_list = [np.ascontiguousarray(f.detach().cpu().numpy()) for f in frames_rgb]
                prediction = self.model.inference(cpu_list)
                depths = _extract_depths(prediction)
            if depths.shape[0] != batch_count:
                raise TwoDVRRuntimeError(
                    f"DA3 returned {depths.shape[0]} depth maps for {batch_count} input frames."
                )
            return depths

        if not frames_rgb:
            return np.empty((0, 0, 0), dtype=np.float32)
        if hasattr(self.model, "inference_depth_only"):
            depths = self.model.inference_depth_only(list(frames_rgb))
        else:
            prediction = self.model.inference(list(frames_rgb))
            depths = _extract_depths(prediction)
        if depths.shape[0] != len(frames_rgb):
            raise TwoDVRRuntimeError(
                f"DA3 returned {depths.shape[0]} depth maps for {len(frames_rgb)} input frames."
            )
        return depths

    def predict(self, frame_rgb: np.ndarray) -> np.ndarray:
        return _depths_for_cpu_frames(self.predict_batch([frame_rgb]), [frame_rgb])[0]


def _smooth_depth(depth: np.ndarray) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32)
    if arr.size == 0:
        return arr
    # Lightweight 3x3 binomial blur; avoids PIL's mode limitations for float images.
    pad = np.pad(arr, 1, mode="edge")
    out = (
        pad[:-2, :-2] + 2.0 * pad[:-2, 1:-1] + pad[:-2, 2:]
        + 2.0 * pad[1:-1, :-2] + 4.0 * pad[1:-1, 1:-1] + 2.0 * pad[1:-1, 2:]
        + pad[2:, :-2] + 2.0 * pad[2:, 1:-1] + pad[2:, 2:]
    ) / 16.0
    return out.astype(np.float32, copy=False)


def _normalize_near(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 1e-6)
    if not np.any(valid):
        return np.zeros(depth.shape, dtype=np.float32)
    inv_depth = np.zeros(depth.shape, dtype=np.float32)
    inv_depth[valid] = 1.0 / depth[valid]
    finite = inv_depth[valid]
    lo, hi = np.percentile(finite, [5.0, 95.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(depth.shape, dtype=np.float32)
    # Stereo disparity is approximately proportional to inverse depth (1/Z).
    # DA3 depth is distance-like, so smaller raw depth becomes larger near/disparity.
    near = np.clip((inv_depth - lo) / (hi - lo), 0.0, 1.0)
    near[~valid] = 0.0
    return near.astype(np.float32, copy=False)


def _sample_horizontal_rgb(image: np.ndarray, map_x: np.ndarray) -> np.ndarray:
    h, w, _ = image.shape
    x = np.clip(map_x, 0.0, float(w - 1))
    x0 = np.floor(x).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    weight = (x - x0).astype(np.float32)[..., None]
    rows = np.arange(h, dtype=np.int32)[:, None]
    a = image[rows, x0].astype(np.float32)
    b = image[rows, x1].astype(np.float32)
    return np.clip(a * (1.0 - weight) + b * weight, 0.0, 255.0).astype(np.uint8)


def _rgb_batch_to_yuv420p(batch_rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(batch_rgb, dtype=np.uint8)
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise TwoDVRRuntimeError(f"Expected RGB batch BHWC, got {rgb.shape}")
    batch, height, width, _ = rgb.shape
    if height % 2 or width % 2:
        raise TwoDVRRuntimeError(f"yuv420p requires even dimensions, got {width}x{height}")
    x = rgb.astype(np.float32)
    r = x[..., 0]
    g = x[..., 1]
    b = x[..., 2]
    y_coeff, u_coeff, v_coeff = _yuv_limited_coefficients(width, height)
    y = np.clip(
        16.0 + y_coeff[0] * r + y_coeff[1] * g + y_coeff[2] * b,
        0.0,
        255.0,
    ).round().astype(np.uint8)
    u_full = np.clip(
        128.0 + u_coeff[0] * r + u_coeff[1] * g + u_coeff[2] * b,
        0.0,
        255.0,
    )
    v_full = np.clip(
        128.0 + v_coeff[0] * r + v_coeff[1] * g + v_coeff[2] * b,
        0.0,
        255.0,
    )
    u = u_full.reshape(batch, height // 2, 2, width // 2, 2).mean(axis=(2, 4)).round().astype(np.uint8)
    v = v_full.reshape(batch, height // 2, 2, width // 2, 2).mean(axis=(2, 4)).round().astype(np.uint8)
    return np.concatenate(
        [y.reshape(batch, -1), u.reshape(batch, -1), v.reshape(batch, -1)],
        axis=1,
    )


def _max_disparity_pixels(src_w: int, eye_distance_mm: float) -> float:
    eye_scale = max(0.1, float(eye_distance_mm) / DEFAULT_EYE_DISTANCE_MM)
    return max(2.0, min(96.0, float(src_w) * DEFAULT_MAX_DISPARITY_RATIO * eye_scale))


def _ceil_even(value: float) -> int:
    out = int(math.ceil(float(value)))
    return out if (out & 1) == 0 else out + 1


def _coerce_flat_fov_deg(flat_fov_deg: float) -> float:
    try:
        value = float(flat_fov_deg)
    except (TypeError, ValueError):
        value = DEFAULT_FLAT_FOV_DEG
    if not math.isfinite(value):
        value = DEFAULT_FLAT_FOV_DEG
    return max(MIN_FLAT_FOV_DEG, min(MAX_FLAT_FOV_DEG, value))


def _flat_vr_eye_size(src_w: int, src_h: int, flat_fov_deg: float) -> int:
    fov = _coerce_flat_fov_deg(flat_fov_deg)
    return max(2, _ceil_even(max(1, int(src_w), int(src_h)) * 180.0 / fov))


def _fill_holes_horizontal_rgb(image: np.ndarray, holes: np.ndarray, near_buffer: np.ndarray) -> np.ndarray:
    holes = np.asarray(holes, dtype=bool)
    if not np.any(holes):
        return image
    h, w, _ = image.shape
    valid = ~holes
    cols = np.broadcast_to(np.arange(w, dtype=np.int32)[None, :], (h, w))
    rows = np.arange(h, dtype=np.int32)[:, None]

    left_idx = np.where(valid, cols, -1)
    left_idx = np.maximum.accumulate(left_idx, axis=1)
    right_idx = np.where(valid, cols, w)
    right_idx = np.minimum.accumulate(right_idx[:, ::-1], axis=1)[:, ::-1]

    left_ok = left_idx >= 0
    right_ok = right_idx < w
    li = np.clip(left_idx, 0, w - 1)
    ri = np.clip(right_idx, 0, w - 1)

    left_rgb = image[rows, li]
    right_rgb = image[rows, ri]
    left_near = np.where(left_ok, near_buffer[rows, li], np.inf)
    right_near = np.where(right_ok, near_buffer[rows, ri], np.inf)

    # Disocclusion holes usually need background, so prefer the farther side
    # when both horizontal neighbors are available.
    choose_right = right_ok & (~left_ok | (right_near <= left_near))
    fill = np.where(choose_right[:, :, None], right_rgb, left_rgb)
    use_fill = holes & (left_ok | right_ok)

    out = image.copy()
    out[use_fill] = fill[use_fill]
    return out


def _shift_fill_holes_rgb(image: np.ndarray, holes: np.ndarray, direction: int, max_tries: int | None = None) -> np.ndarray:
    out = image.copy()
    remaining = np.asarray(holes, dtype=bool).copy()
    if not np.any(remaining):
        return out
    h, w, _ = image.shape
    tries = w if max_tries is None else max(1, int(max_tries))
    for _ in range(tries):
        if not np.any(remaining):
            break
        if direction < 0:
            src = np.pad(out[:, :-1], ((0, 0), (1, 0), (0, 0)), mode="edge")
            src_valid = np.pad(~remaining[:, :-1], ((0, 0), (1, 0)), constant_values=False)
        else:
            src = np.pad(out[:, 1:], ((0, 0), (0, 1), (0, 0)), mode="edge")
            src_valid = np.pad(~remaining[:, 1:], ((0, 0), (0, 1)), constant_values=False)
        take = remaining & src_valid
        if not np.any(take):
            break
        out[take] = src[take]
        remaining[take] = False
    return out


def _dilate_mask_np(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(iterations))):
        pad = np.pad(out, 1, mode="edge")
        out = (
            pad[:-2, :-2] | pad[:-2, 1:-1] | pad[:-2, 2:]
            | pad[1:-1, :-2] | pad[1:-1, 1:-1] | pad[1:-1, 2:]
            | pad[2:, :-2] | pad[2:, 1:-1] | pad[2:, 2:]
        )
    return out


def _box_blur_rgb(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    radius = max(1, int(kernel_size) // 2)
    src = image.astype(np.float32)
    pad = np.pad(src, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    out = np.zeros_like(src)
    count = 0
    for dy in range(kernel_size):
        for dx in range(kernel_size):
            out += pad[dy:dy + image.shape[0], dx:dx + image.shape[1]]
            count += 1
    return np.clip(out / max(1, count), 0.0, 255.0).astype(np.uint8)


def _soft_blend_holes_rgb(image: np.ndarray, holes: np.ndarray) -> np.ndarray:
    blend_mask = _dilate_mask_np(holes, iterations=1).astype(np.float32)
    if not np.any(blend_mask):
        return image
    alpha = blend_mask[:, :, None] * 0.35
    blurred = _box_blur_rgb(image, kernel_size=5).astype(np.float32)
    base = image.astype(np.float32)
    return np.clip(base * (1.0 - alpha) + blurred * alpha, 0.0, 255.0).astype(np.uint8)


def _inpaint_holes_rgb(image: np.ndarray, holes: np.ndarray) -> np.ndarray:
    if not np.any(holes):
        return image
    try:
        import cv2

        mask = (_dilate_mask_np(holes, iterations=1).astype(np.uint8) * 255)
        return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    except Exception:
        return _soft_blend_holes_rgb(_shift_fill_holes_rgb(image, holes, -1), holes)


def _fill_eye_holes_rgb(
    image: np.ndarray,
    holes: np.ndarray,
    near_buffer: np.ndarray,
    direction: int,
    mode: str,
) -> np.ndarray:
    mode = resolve_hole_fill_mode(mode)
    if mode in {"none", HOLE_FILL_E2FGVI, HOLE_FILL_INVERSE_WARP}:
        return image
    if mode == "background":
        return _fill_holes_horizontal_rgb(image, holes, near_buffer)
    if mode == "inpaint":
        seed = _shift_fill_holes_rgb(image, holes, direction)
        return _inpaint_holes_rgb(seed, holes)
    shifted = _shift_fill_holes_rgb(image, holes, direction)
    if mode == "soft_shift":
        return _soft_blend_holes_rgb(shifted, holes)
    return shifted


def _make_inverse_warp_stereo_result(
    frame_rgb: np.ndarray,
    depth: np.ndarray,
    eye_distance_mm: float = DEFAULT_EYE_DISTANCE_MM,
) -> StereoRenderResult:
    h, w, _ = frame_rgb.shape
    near = _normalize_near(_smooth_depth(depth))
    if near.shape != (h, w):
        raise TwoDVRRuntimeError(f"Depth shape {near.shape} does not match frame shape {(h, w)}")
    max_shift = _max_disparity_pixels(w, eye_distance_mm)
    disparity = near * max_shift
    cols = np.broadcast_to(np.arange(w, dtype=np.float32)[None, :], (h, w))
    left = _sample_horizontal_rgb(frame_rgb, cols - disparity * 0.5)
    right = _sample_horizontal_rgb(frame_rgb, cols + disparity * 0.5)
    holes = np.zeros((h, w), dtype=bool)
    return StereoRenderResult(
        left=left,
        right=right,
        near=near,
        disparity=disparity,
        left_before_fill=left,
        right_before_fill=right,
        left_holes=holes,
        right_holes=holes,
    )


def _forward_warp_eye_rgb(
    frame_rgb: np.ndarray,
    near: np.ndarray,
    max_shift: float,
    eye_sign: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w, _ = frame_rgb.shape
    yy, xx = np.mgrid[0:h, 0:w]
    target_x = np.rint(xx.astype(np.float32) + near * (max_shift * 0.5 * eye_sign)).astype(np.int32)
    valid = (target_x >= 0) & (target_x < w)

    flat_target = (yy * w + target_x).reshape(-1)
    valid_flat = valid.reshape(-1)
    priority = near.reshape(-1).astype(np.float32, copy=False)
    target_valid = flat_target[valid_flat]

    zbuf_flat = np.full(h * w, -1.0, dtype=np.float32)
    if target_valid.size:
        np.maximum.at(zbuf_flat, target_valid, priority[valid_flat])

    safe_target = np.clip(flat_target, 0, h * w - 1)
    winners = valid_flat & (priority >= zbuf_flat[safe_target] - 1e-6)
    out_flat = np.zeros((h * w, 3), dtype=np.uint8)
    out_flat[safe_target[winners]] = frame_rgb.reshape(-1, 3)[winners]

    near_buffer = zbuf_flat.reshape(h, w)
    holes = near_buffer < 0.0
    return out_flat.reshape(h, w, 3), holes, near_buffer


def _make_stereo_result(
    frame_rgb: np.ndarray,
    depth: np.ndarray,
    eye_distance_mm: float = DEFAULT_EYE_DISTANCE_MM,
    hole_fill_mode: str = DEFAULT_HOLE_FILL_MODE,
) -> StereoRenderResult:
    mode = resolve_hole_fill_mode(hole_fill_mode)
    if mode == HOLE_FILL_INVERSE_WARP:
        return _make_inverse_warp_stereo_result(frame_rgb, depth, eye_distance_mm)

    h, w, _ = frame_rgb.shape
    near = _normalize_near(_smooth_depth(depth))
    if near.shape != (h, w):
        raise TwoDVRRuntimeError(f"Depth shape {near.shape} does not match frame shape {(h, w)}")
    max_shift = _max_disparity_pixels(w, eye_distance_mm)
    disparity = near * max_shift

    left_raw, left_holes, left_near = _forward_warp_eye_rgb(frame_rgb, near, max_shift, 1.0)
    right_raw, right_holes, right_near = _forward_warp_eye_rgb(frame_rgb, near, max_shift, -1.0)
    left = _fill_eye_holes_rgb(left_raw, left_holes, left_near, -1, mode)
    right = _fill_eye_holes_rgb(right_raw, right_holes, right_near, 1, mode)

    return StereoRenderResult(
        left=left,
        right=right,
        near=near,
        disparity=disparity,
        left_before_fill=left_raw,
        right_before_fill=right_raw,
        left_holes=left_holes,
        right_holes=right_holes,
    )


def make_stereo_pair(frame_rgb: np.ndarray, depth: np.ndarray,
                     eye_distance_mm: float = DEFAULT_EYE_DISTANCE_MM,
                     hole_fill_mode: str = DEFAULT_HOLE_FILL_MODE) -> tuple[np.ndarray, np.ndarray]:
    result = _make_stereo_result(frame_rgb, depth, eye_distance_mm, hole_fill_mode)
    return result.left, result.right


def _flat_camera_rays_to_source(
    dir_x: np.ndarray,
    dir_y_down: np.ndarray,
    dir_z: np.ndarray,
    src_w: int,
    src_h: int,
    flat_fov_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fov_rad = math.radians(_coerce_flat_fov_deg(flat_fov_deg))
    plane_scale = max(1e-6, math.tan(fov_rad * 0.5))
    valid = dir_z > 1.0e-6

    px = np.zeros_like(dir_x, dtype=np.float32)
    py = np.zeros_like(dir_y_down, dtype=np.float32)
    np.divide(dir_x, dir_z * plane_scale, out=px, where=valid)
    np.divide(dir_y_down, dir_z * plane_scale, out=py, where=valid)
    valid &= (px >= -1.0) & (px <= 1.0) & (py >= -1.0) & (py <= 1.0)

    canvas = float(max(1, int(src_w), int(src_h)))
    x0 = (canvas - float(src_w)) * 0.5
    y0 = (canvas - float(src_h)) * 0.5
    # Match ffmpeg v360's scale(x, s) convention: (0.5*x + 0.5) * (s - 1).
    map_x = (px * 0.5 + 0.5) * (canvas - 1.0) - x0
    map_y = (py * 0.5 + 0.5) * (canvas - 1.0) - y0
    valid &= (
        (map_x >= 0.0) & (map_x <= float(src_w - 1))
        & (map_y >= 0.0) & (map_y <= float(src_h - 1))
    )
    return map_x.astype(np.float32), map_y.astype(np.float32), valid


def _make_hequirect_projection(src_w: int, src_h: int, flat_fov_deg: float) -> ProjectionMap:
    side = _flat_vr_eye_size(src_w, src_h, flat_fov_deg)
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float32)
    yaw = ((xx + 0.5) / float(side) - 0.5) * math.pi
    pitch = (0.5 - (yy + 0.5) / float(side)) * math.pi

    cos_pitch = np.cos(pitch)
    dir_x = cos_pitch * np.sin(yaw)
    dir_y_down = -np.sin(pitch)
    dir_z = cos_pitch * np.cos(yaw)
    map_x, map_y, mask = _flat_camera_rays_to_source(
        dir_x.astype(np.float32),
        dir_y_down.astype(np.float32),
        dir_z.astype(np.float32),
        src_w,
        src_h,
        flat_fov_deg,
    )
    return ProjectionMap(side, side, map_x, map_y, mask)


def _make_fisheye_projection(src_w: int, src_h: int, flat_fov_deg: float) -> ProjectionMap:
    side = _flat_vr_eye_size(src_w, src_h, flat_fov_deg)
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float32)
    cx = float(side) * 0.5
    cy = float(side) * 0.5
    radius = max(1.0, float(side) * 0.5)
    nx_disk = (xx + 0.5 - cx) / radius
    ny_disk = (yy + 0.5 - cy) / radius
    rr = np.sqrt(nx_disk * nx_disk + ny_disk * ny_disk)
    disk_mask = rr <= 1.0

    theta = rr * (math.pi * 0.5)
    azimuth = np.arctan2(-ny_disk, nx_disk)
    sin_theta = np.sin(theta)
    dir_x = sin_theta * np.cos(azimuth)
    dir_y_down = -sin_theta * np.sin(azimuth)
    dir_z = np.cos(theta)
    map_x, map_y, source_mask = _flat_camera_rays_to_source(
        dir_x.astype(np.float32),
        dir_y_down.astype(np.float32),
        dir_z.astype(np.float32),
        src_w,
        src_h,
        flat_fov_deg,
    )
    return ProjectionMap(side, side, map_x, map_y, disk_mask & source_mask)


def _make_flat3d_projection(src_w: int, src_h: int) -> ProjectionMap:
    yy, xx = np.mgrid[0:src_h, 0:src_w].astype(np.float32)
    mask = np.ones((src_h, src_w), dtype=bool)
    return ProjectionMap(src_w, src_h, xx.astype(np.float32), yy.astype(np.float32), mask)


def make_projection_map(src_w: int, src_h: int, projection: str,
                        flat_fov_deg: float = DEFAULT_FLAT_FOV_DEG) -> ProjectionMap:
    if projection == PROJECTION_FLAT_3D:
        return _make_flat3d_projection(src_w, src_h)
    if projection == PROJECTION_FISHEYE:
        return _make_fisheye_projection(src_w, src_h, flat_fov_deg)
    return _make_hequirect_projection(src_w, src_h, flat_fov_deg)


def _sample_rgb_xy(image: np.ndarray, pmap: ProjectionMap) -> np.ndarray:
    h, w, _ = image.shape
    x = np.clip(pmap.map_x, 0.0, float(w - 1))
    y = np.clip(pmap.map_y, 0.0, float(h - 1))
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = (x - x0).astype(np.float32)[..., None]
    wy = (y - y0).astype(np.float32)[..., None]
    a = image[y0, x0].astype(np.float32)
    b = image[y0, x1].astype(np.float32)
    c = image[y1, x0].astype(np.float32)
    d = image[y1, x1].astype(np.float32)
    top = a * (1.0 - wx) + b * wx
    bottom = c * (1.0 - wx) + d * wx
    out = top * (1.0 - wy) + bottom * wy
    out[~pmap.mask] = 0.0
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def render_sbs_frame(frame_rgb: np.ndarray, depth: np.ndarray, projection: str,
                     eye_distance_mm: float, pmap: ProjectionMap,
                     hole_fill_mode: str = DEFAULT_HOLE_FILL_MODE) -> np.ndarray:
    left, right = make_stereo_pair(frame_rgb, depth, eye_distance_mm, hole_fill_mode)
    left_proj = _sample_rgb_xy(left, pmap)
    right_proj = _sample_rgb_xy(right, pmap)
    return np.concatenate([left_proj, right_proj], axis=1)


def _depth_debug_rgb(depth: np.ndarray) -> np.ndarray:
    near = _normalize_near(_smooth_depth(depth))
    img = np.clip(near * 255.0, 0.0, 255.0).astype(np.uint8)
    return np.repeat(img[:, :, None], 3, axis=2)


def _heatmap_debug_rgb(value: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros((*arr.shape, 3), dtype=np.uint8)
    if vmin is None:
        vmin = float(np.percentile(arr[valid], 5.0))
    if vmax is None:
        vmax = float(np.percentile(arr[valid], 95.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1e-6
    t = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    t[~valid] = 0.0
    r = np.clip((t - 0.50) * 2.0, 0.0, 1.0)
    g = np.clip(1.0 - np.abs(t - 0.50) * 2.0, 0.0, 1.0)
    b = np.clip((0.50 - t) * 2.0, 0.0, 1.0)
    return (np.stack([r, g, b], axis=2).astype(np.float32).clip(0.0, 1.0) * 255.0).astype(np.uint8)


def _mask_debug_rgb(mask: np.ndarray) -> np.ndarray:
    img = np.where(np.asarray(mask, dtype=bool), 255, 0).astype(np.uint8)
    return np.repeat(img[:, :, None], 3, axis=2)


def _depth_debug_stats(depth: np.ndarray, near: np.ndarray, disparity: np.ndarray) -> str:
    arr = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return "depth: no finite values"
    raw = np.percentile(arr[valid], [0.0, 5.0, 50.0, 95.0, 100.0])
    near_vals = near[np.isfinite(near)]
    disp_vals = disparity[np.isfinite(disparity)]
    if near_vals.size:
        near_p = np.percentile(near_vals, [5.0, 50.0, 95.0])
    else:
        near_p = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    if disp_vals.size:
        disp_mean = float(np.mean(disp_vals))
        disp_p95 = float(np.percentile(disp_vals, 95.0))
        disp_max = float(np.max(disp_vals))
    else:
        disp_mean = 0.0
        disp_p95 = 0.0
        disp_max = 0.0
    return (
        "depth raw min/p05/p50/p95/max="
        f"{raw[0]:.4f}/{raw[1]:.4f}/{raw[2]:.4f}/{raw[3]:.4f}/{raw[4]:.4f}; "
        f"near p05/p50/p95={near_p[0]:.3f}/{near_p[1]:.3f}/{near_p[2]:.3f}; "
        f"disparity mean/p95/max={disp_mean:.2f}/{disp_p95:.2f}/{disp_max:.2f}px"
    )


def _save_eye_debug_images(output: str, frame_rgb: np.ndarray, depth: np.ndarray,
                           projection: str, eye_distance_mm: float, pmap: ProjectionMap,
                           hole_fill_mode: str = DEFAULT_HOLE_FILL_MODE,
                           stabilize_mode: str = DEFAULT_STABILIZE_MODE,
                           log_callback=None) -> None:
    try:
        stem = debug_output_stem(output)
        stab = resolve_stabilize_mode(stabilize_mode)
        stem = stem.with_name(f"{stem.name}.stab_{stab}")
        stem.parent.mkdir(parents=True, exist_ok=True)
        mode = resolve_hole_fill_mode(hole_fill_mode)
        stereo = _make_stereo_result(frame_rgb, depth, eye_distance_mm, mode)
        near = stereo.near
        disparity = stereo.disparity
        max_shift = _max_disparity_pixels(frame_rgb.shape[1], eye_distance_mm)
        left, right = stereo.left, stereo.right
        left_proj = _sample_rgb_xy(left, pmap)
        right_proj = _sample_rgb_xy(right, pmap)
        lr_diff = np.mean(
            np.abs(left.astype(np.float32) - right.astype(np.float32)),
            axis=2,
        )
        images = {
            "source": frame_rgb,
            "depth_near": np.repeat(np.clip(near * 255.0, 0.0, 255.0).astype(np.uint8)[:, :, None], 3, axis=2),
            "depth_heat": _heatmap_debug_rgb(near, 0.0, 1.0),
            "disparity_heat": _heatmap_debug_rgb(disparity, 0.0, max(max_shift, 1e-6)),
            "lr_absdiff": _heatmap_debug_rgb(lr_diff),
            "left_holes": _mask_debug_rgb(stereo.left_holes),
            "right_holes": _mask_debug_rgb(stereo.right_holes),
            "left_forward_raw": stereo.left_before_fill,
            "right_forward_raw": stereo.right_before_fill,
            "left_stereo_flat": left,
            "right_stereo_flat": right,
            f"left_{projection}": left_proj,
            f"right_{projection}": right_proj,
            f"sbs_{projection}": np.concatenate([left_proj, right_proj], axis=1),
        }
        saved = []
        for name, image in images.items():
            path = stem.with_name(f"{stem.name}.debug_{name}.png")
            Image.fromarray(np.ascontiguousarray(image)).save(path)
            saved.append(str(path))
        if log_callback:
            log_callback(f"[2dvr] eye debug output dir: {stem.parent}")
            log_callback(f"[2dvr] eye debug fill mode: {mode}")
            log_callback(f"[2dvr] eye debug stabilize mode: {stab}")
            log_callback(f"[2dvr] eye debug stats: {_depth_debug_stats(depth, near, disparity)}")
            log_callback(
                "[2dvr] eye debug stats: "
                f"hole pixels left/right={float(np.mean(stereo.left_holes)) * 100.0:.2f}%/"
                f"{float(np.mean(stereo.right_holes)) * 100.0:.2f}%"
            )
            log_callback(
                "[2dvr] eye debug stats: "
                f"lr_absdiff mean/p95/max={float(np.mean(lr_diff)):.2f}/"
                f"{float(np.percentile(lr_diff, 95.0)):.2f}/{float(np.max(lr_diff)):.2f}"
            )
            log_callback("[2dvr] eye debug images saved:")
            for path in saved:
                log_callback(f"[2dvr]   {path}")
    except Exception as exc:
        if log_callback:
            log_callback(f"[2dvr] failed to save eye debug images: {exc}")


def _is_cuda_oom(exc: Exception) -> bool:
    text = str(exc).lower()
    return "cuda" in text and ("out of memory" in text or "allocation" in text)


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _predict_depths_resilient(estimator: DA3DepthEstimator, frames: list[np.ndarray], log_callback=None) -> Any:
    try:
        return estimator.predict_batch(frames)
    except Exception as exc:
        if len(frames) <= 1 or not _is_cuda_oom(exc):
            raise
        _empty_cuda_cache()
        mid = max(1, len(frames) // 2)
        if log_callback:
            log_callback(f"[2dvr] DA3 CUDA OOM at batch={len(frames)}; retrying as {mid}+{len(frames) - mid}")
        first = _predict_depths_resilient(estimator, frames[:mid], log_callback)
        second = _predict_depths_resilient(estimator, frames[mid:], log_callback)
        return _concat_depth_batches([first, second])


class TorchStereoRenderer:
    backend = "torch_cuda_forward_zfill"

    def __init__(self, src_w: int, src_h: int, pmap: ProjectionMap, eye_distance_mm: float,
                 hole_fill_mode: str = DEFAULT_HOLE_FILL_MODE,
                 stabilize_mode: str = DEFAULT_STABILIZE_MODE):
        import torch

        self.torch = torch
        self.src_w = int(src_w)
        self.src_h = int(src_h)
        self.pmap = pmap
        self.eye_distance_mm = float(eye_distance_mm)
        self.hole_fill_mode = resolve_hole_fill_mode(hole_fill_mode)
        self.stabilize_mode = resolve_stabilize_mode(stabilize_mode)
        self.stabilize_enabled = self.stabilize_mode != STABILIZE_OFF
        self.device = torch.device("cuda")
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.inverse_warp = self.hole_fill_mode == HOLE_FILL_INVERSE_WARP or (
            self.hole_fill_mode == "soft_shift" and inverse_warp_compat_enabled()
        )
        self.backend = "torch_cuda_inverse_warp" if self.inverse_warp else self.backend
        depth_beta_default = 0.7 if self.hole_fill_mode == HOLE_FILL_E2FGVI else 0.5
        self.norm_alpha = _env_float("TOOL_2DVR_NORM_ALPHA", 0.15, 0.0, 1.0)
        self.depth_beta = _env_float("TOOL_2DVR_DEPTH_BETA", depth_beta_default, 0.0, 1.0)
        self.adaptive_beta = _env_bool("TOOL_2DVR_ADAPTIVE_BETA", True) and self.depth_beta < 1.0
        self.adaptive_thresh = _env_float("TOOL_2DVR_ADAPTIVE_THRESH", 0.05, 0.0, 1.0)
        self.adaptive_slope = _env_float("TOOL_2DVR_ADAPTIVE_SLOPE", 50.0, 1.0, 500.0)
        self.hole_thresh = _env_float("TOOL_2DVR_HOLE_THRESH", 0.3, 0.0, 1.0)
        self.scene_cut_depth = _env_float("TOOL_2DVR_SCENE_CUT_DEPTH", 0.5, 0.0, 10.0)
        self.scene_cut_hist = _env_float("TOOL_2DVR_SCENE_CUT_HIST", 0.6, 0.0, 10.0)
        self.temporal_identity = self.stabilize_enabled and self.norm_alpha >= 1.0 and self.depth_beta >= 1.0
        self.subpixel_splat_enabled = (
            self.stabilize_enabled
            and not self.temporal_identity
            and self._resolve_subpixel_splat_enabled()
        )
        self.lo_ema = None
        self.hi_ema = None
        self.depth_ema = None
        self.norm_warmup_frames = 4
        self.warmup_frames_remaining = self.norm_warmup_frames
        self.prev_rgb_hist: np.ndarray | None = None
        self.is_identity_projection = (
            pmap.out_w == self.src_w
            and pmap.out_h == self.src_h
            and pmap.mask.shape == (self.src_h, self.src_w)
            and bool(np.all(pmap.mask))
        )

        yi = torch.arange(self.src_h, device=self.device, dtype=torch.long)
        xi = torch.arange(self.src_w, device=self.device, dtype=torch.long)
        pix_y, pix_x = torch.meshgrid(yi, xi, indexing="ij")
        self.pixel_y = pix_y.unsqueeze(0)
        self.pixel_x = pix_x.unsqueeze(0)
        self.source_grid_x = (pix_x.float() / max(1, self.src_w - 1) * 2.0 - 1.0).unsqueeze(0)
        self.source_grid_y = (pix_y.float() / max(1, self.src_h - 1) * 2.0 - 1.0).unsqueeze(0)

        grid_x = torch.from_numpy((pmap.map_x / max(1, self.src_w - 1) * 2.0 - 1.0).astype(np.float32))
        grid_y = torch.from_numpy((pmap.map_y / max(1, self.src_h - 1) * 2.0 - 1.0).astype(np.float32))
        self.proj_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).to(self.device, non_blocking=True)
        self.proj_mask = torch.from_numpy(pmap.mask.astype(np.float32)).to(self.device, non_blocking=True)[None, None]
        self.blur_kernel = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            device=self.device,
            dtype=torch.float32,
        ).view(1, 1, 3, 3) / 16.0

    def _resolve_subpixel_splat_enabled(self) -> bool:
        raw = os.environ.get("TOOL_2DVR_SUBPIXEL_SPLAT", "auto").strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
        if self.stabilize_mode == STABILIZE_FULL:
            return True
        max_pixels = int(_env_float("TOOL_2DVR_SUBPIXEL_MAX_PIXELS", 2_073_600, 1.0, None))
        return (self.src_w * self.src_h) <= max_pixels

    def _smooth_depth(self, depth):
        import torch.nn.functional as F

        return F.conv2d(F.pad(depth, (1, 1, 1, 1), mode="replicate"), self.blur_kernel)

    def _reset_temporal_state(self):
        self.lo_ema = None
        self.hi_ema = None
        self.depth_ema = None
        self.warmup_frames_remaining = self.norm_warmup_frames

    def _ema_scan(self, values, previous, alpha: float):
        torch = self.torch
        if alpha >= 1.0:
            return values
        if alpha <= 0.0:
            return previous.expand_as(values)
        batch = int(values.shape[0])
        t = torch.arange(batch, device=values.device, dtype=values.dtype)
        base = torch.tensor(1.0 - alpha, device=values.device, dtype=values.dtype)
        view_shape = (batch,) + (1,) * (values.ndim - 1)
        decay_prev = base.pow(t + 1).view(view_shape)
        inv_pow = base.pow(-t).view(view_shape)
        contrib = torch.cumsum(alpha * values * inv_pow, dim=0) * base.pow(t).view(view_shape)
        return decay_prev * previous + contrib

    def _ema_scan_variable(self, values, alpha, previous):
        torch = self.torch
        retain = (1.0 - alpha).clamp_min(1.0e-3)
        prod = torch.cumprod(retain, dim=0)
        contrib = torch.cumsum(alpha * values / prod, dim=0) * prod
        return prod * previous + contrib

    def _rgb_histogram(self, frame_rgb: np.ndarray) -> np.ndarray:
        h, w = frame_rgb.shape[:2]
        step_y = max(1, h // 64)
        step_x = max(1, w // 64)
        small = frame_rgb[::step_y, ::step_x][:64, :64]
        hist_parts = [
            np.histogram(small[..., channel], bins=16, range=(0, 256))[0].astype(np.float32)
            for channel in range(3)
        ]
        hist = np.concatenate(hist_parts)
        total = float(np.sum(hist))
        return hist / max(total, 1.0)

    def observe_rgb_batch(self, frames_rgb: list[np.ndarray]) -> None:
        if not self.stabilize_enabled or self.scene_cut_hist <= 0.0 or not frames_rgb:
            return
        current = self._rgb_histogram(frames_rgb[0])
        if self.prev_rgb_hist is not None:
            denom = current + self.prev_rgb_hist + 1.0e-6
            chi_square = 0.5 * float(np.sum(((current - self.prev_rgb_hist) ** 2) / denom))
            if chi_square > self.scene_cut_hist:
                self._reset_temporal_state()
        self.prev_rgb_hist = self._rgb_histogram(frames_rgb[-1])

    def _depth_tensor(self, depths):
        torch = self.torch
        if _is_torch_tensor(depths):
            depth_t = depths.to(self.device, non_blocking=True)
        else:
            depth_np = np.ascontiguousarray(_extract_depths(depths).astype(np.float32, copy=False))
            depth_t = torch.from_numpy(depth_np).to(self.device, non_blocking=True)
        if depth_t.ndim == 3:
            depth_t = depth_t.unsqueeze(1)
        elif depth_t.ndim == 4 and depth_t.shape[-1] == 1:
            depth_t = depth_t.permute(0, 3, 1, 2).contiguous()
        elif depth_t.ndim != 4:
            raise TwoDVRRuntimeError(f"Unexpected depth tensor shape: {tuple(depth_t.shape)}")
        return depth_t.float()

    def _normalize_near(self, depth):
        torch = self.torch
        valid_depth = torch.isfinite(depth) & (depth > 1e-6)
        inv_depth = torch.zeros_like(depth)
        inv_depth[valid_depth] = 1.0 / depth[valid_depth]
        flat = inv_depth.flatten(1)
        valid_flat = valid_depth.flatten(1)
        max_samples = resolve_depth_norm_samples()
        if flat.shape[1] > max_samples:
            stride = max(1, int(math.ceil(flat.shape[1] / max_samples)))
            flat = flat[:, ::stride][:, :max_samples]
            valid_flat = valid_flat[:, ::stride][:, :max_samples]
        valid_count = valid_flat.sum(dim=1)
        values = torch.where(valid_flat, flat, torch.full_like(flat, float("inf"))).float()
        lo, hi = self._normalize_near_percentiles(values, valid_count)
        if self.stabilize_enabled and not self.temporal_identity:
            joint_values = values.reshape(1, -1)
            joint_count = valid_count.sum().reshape(1)
            lo_joint, hi_joint = self._normalize_near_percentiles(joint_values, joint_count)
            lo = torch.where(valid_count >= 2, lo_joint.expand_as(lo), lo)
            hi = torch.where(valid_count >= 2, hi_joint.expand_as(hi), hi)
            lo, hi = self._stabilize_norm_bounds(lo, hi, valid_count)

        view_shape = (depth.shape[0],) + (1,) * (depth.ndim - 1)
        lo = lo.view(view_shape)
        hi = hi.view(view_shape)
        valid_count = valid_count.view(view_shape)
        good = torch.isfinite(lo) & torch.isfinite(hi) & (hi > lo) & (valid_count >= 2)
        cur = ((inv_depth - lo) / (hi - lo).clamp_min(1e-12)).clamp(0.0, 1.0)
        return torch.where(good & valid_depth, cur, torch.zeros_like(cur))

    def _normalize_near_percentiles(self, values, valid_count):
        torch = self.torch
        if valid_count.numel() > 0 and bool((valid_count == valid_count[0]).all()) and int(valid_count[0].item()) >= 2:
            count = int(valid_count[0].item())
            lo_k = max(1, min(count, int(math.floor((count - 1) * 0.05)) + 1))
            hi_k = max(1, min(count, int(math.floor((count - 1) * 0.95)) + 1))
            return (
                torch.kthvalue(values, lo_k, dim=1).values,
                torch.kthvalue(values, hi_k, dim=1).values,
            )
        sorted_values = torch.sort(values, dim=1).values
        rows = torch.arange(sorted_values.shape[0], device=sorted_values.device)
        safe_count = valid_count.clamp_min(1)
        lo_idx = torch.floor((safe_count - 1).float() * 0.05).long()
        hi_idx = torch.floor((safe_count - 1).float() * 0.95).long()
        return sorted_values[rows, lo_idx], sorted_values[rows, hi_idx]

    def _stabilize_norm_bounds(self, lo_frame, hi_frame, valid_count):
        torch = self.torch
        good = torch.isfinite(lo_frame) & torch.isfinite(hi_frame) & (hi_frame > lo_frame) & (valid_count >= 2)
        if not bool(good.any()):
            return lo_frame, hi_frame
        good_f = good.to(dtype=lo_frame.dtype)
        denom = good_f.sum().clamp_min(1.0)
        lo_seed = (torch.where(good, lo_frame, torch.zeros_like(lo_frame)) * good_f).sum() / denom
        hi_seed = (torch.where(good, hi_frame, torch.zeros_like(hi_frame)) * good_f).sum() / denom
        if self.lo_ema is None or self.hi_ema is None or self.warmup_frames_remaining > 0:
            self.lo_ema = lo_seed.clone()
            self.hi_ema = hi_seed.clone()
            self.warmup_frames_remaining = max(0, self.warmup_frames_remaining - int(lo_frame.shape[0]))
            return lo_seed.expand_as(lo_frame), hi_seed.expand_as(hi_frame)
        lo_scan = self._ema_scan(lo_frame, self.lo_ema, self.norm_alpha)
        hi_scan = self._ema_scan(hi_frame, self.hi_ema, self.norm_alpha)
        self.lo_ema = lo_scan[-1].clone()
        self.hi_ema = hi_scan[-1].clone()
        return (
            torch.where(good, lo_scan, lo_frame),
            torch.where(good, hi_scan, hi_frame),
        )

    def _near_from_depths(self, depths):
        import torch.nn.functional as F

        depth_t = self._depth_tensor(depths)
        depth_t = self._apply_depth_ema(depth_t)
        near = self._normalize_near(self._smooth_depth(depth_t))
        if near.shape[-2:] != (self.src_h, self.src_w):
            near = F.interpolate(
                near,
                size=(self.src_h, self.src_w),
                mode="bilinear",
                align_corners=False,
            )
        return near[:, 0].contiguous()

    def _maybe_reset_for_depth_scene_cut(self, depth_t):
        torch = self.torch
        if (
            not self.stabilize_enabled
            or self.depth_ema is None
            or self.lo_ema is None
            or self.hi_ema is None
            or self.scene_cut_depth <= 0.0
            or self.depth_ema.shape[-2:] != depth_t.shape[-2:]
        ):
            return
        cur = depth_t[:1]
        valid_cur = torch.isfinite(cur) & (cur > 1.0e-6)
        valid_prev = torch.isfinite(self.depth_ema) & (self.depth_ema > 1.0e-6)
        cur_inv = torch.zeros_like(cur)
        prev_inv = torch.zeros_like(self.depth_ema)
        cur_inv[valid_cur] = 1.0 / cur[valid_cur]
        prev_inv[valid_prev] = 1.0 / self.depth_ema[valid_prev]
        valid = valid_cur & valid_prev
        if not bool(valid.any()):
            return
        diff = torch.where(valid, (cur_inv - prev_inv).abs(), torch.zeros_like(cur_inv)).mean()
        span = (self.hi_ema - self.lo_ema).abs().clamp_min(1.0e-3)
        if bool((diff / span) > self.scene_cut_depth):
            self._reset_temporal_state()

    def _apply_depth_ema(self, depth_t):
        torch = self.torch
        if not self.stabilize_enabled or self.depth_beta >= 1.0:
            return depth_t
        self._maybe_reset_for_depth_scene_cut(depth_t)
        if self.depth_ema is None or self.depth_ema.shape[-2:] != depth_t.shape[-2:]:
            self.depth_ema = depth_t[:1].clone()
        if self.adaptive_beta and self.lo_ema is not None and self.hi_ema is not None:
            valid_cur = torch.isfinite(depth_t) & (depth_t > 1.0e-6)
            valid_prev = torch.isfinite(self.depth_ema) & (self.depth_ema > 1.0e-6)
            cur_inv = torch.zeros_like(depth_t)
            prev_inv = torch.zeros_like(depth_t)
            cur_inv[valid_cur] = 1.0 / depth_t[valid_cur]
            prev = self.depth_ema.expand_as(depth_t)
            prev_inv[valid_prev.expand_as(depth_t)] = 1.0 / prev[valid_prev.expand_as(depth_t)]
            diff = (cur_inv - prev_inv).abs()
            span = (self.hi_ema - self.lo_ema).abs().clamp_min(1.0e-3)
            beta = self.torch.sigmoid((diff - self.adaptive_thresh * span) * self.adaptive_slope)
            depth_smoothed = self._ema_scan_variable(depth_t, beta, self.depth_ema)
        else:
            depth_smoothed = self._ema_scan(depth_t, self.depth_ema, self.depth_beta)
        self.depth_ema = depth_smoothed[-1:].clone()
        return depth_smoothed


    def _forward_warp_eye(self, frame, near, max_shift: float, eye_sign: float):
        torch = self.torch
        batch, height, width, channels = frame.shape
        shift = max_shift * 0.5 * eye_sign
        tx = self.pixel_x.expand(batch, -1, -1).float() + near * shift
        target_x = torch.round(tx).long()
        valid = (target_x >= 0) & (target_x < width)

        batch_idx = torch.arange(batch, device=self.device, dtype=torch.long).view(batch, 1, 1)
        flat_target = ((batch_idx * height + self.pixel_y) * width + target_x).reshape(-1)
        valid_flat = valid.reshape(-1)
        priority = near.reshape(-1)

        zbuf = torch.full((batch * height * width,), -1.0, device=self.device, dtype=torch.float32)
        zbuf.scatter_reduce_(
            0,
            flat_target[valid_flat],
            priority[valid_flat],
            reduce="amax",
            include_self=True,
        )

        safe_target = flat_target.clamp(0, batch * height * width - 1)
        near_buffer = zbuf.view(batch, height, width)
        if not self.subpixel_splat_enabled:
            winners = valid_flat & (priority >= zbuf[safe_target] - 1e-6)
            out = torch.zeros((batch * height * width, channels), device=self.device, dtype=frame.dtype)
            out[safe_target[winners]] = frame.reshape(-1, channels)[winners]
            holes = near_buffer < 0.0
            return out.view(batch, height, width, channels), holes, near_buffer

        tx0_raw = torch.floor(tx).long()
        tx1_raw = tx0_raw + 1
        w1 = (tx - tx0_raw.float()).clamp(0.0, 1.0)
        w0 = 1.0 - w1
        valid0 = (tx0_raw >= 0) & (tx0_raw < width)
        valid1 = (tx1_raw >= 0) & (tx1_raw < width)
        tx0 = tx0_raw.clamp(0, width - 1)
        tx1 = tx1_raw.clamp(0, width - 1)

        z0 = torch.gather(near_buffer, 2, tx0)
        z1 = torch.gather(near_buffer, 2, tx1)
        keep0 = valid0 & (near >= z0 - 1.0e-6)
        keep1 = valid1 & (near >= z1 - 1.0e-6)

        flat0 = ((batch_idx * height + self.pixel_y) * width + tx0).reshape(-1)
        flat1 = ((batch_idx * height + self.pixel_y) * width + tx1).reshape(-1)
        src = frame.reshape(-1, channels)
        amount0 = (w0 * keep0.to(dtype=w0.dtype)).reshape(-1)
        amount1 = (w1 * keep1.to(dtype=w1.dtype)).reshape(-1)

        out = torch.zeros((batch * height * width, channels), device=self.device, dtype=frame.dtype)
        weight = torch.zeros((batch * height * width,), device=self.device, dtype=frame.dtype)
        out.scatter_add_(0, flat0[:, None].expand(-1, channels), src * amount0[:, None])
        out.scatter_add_(0, flat1[:, None].expand(-1, channels), src * amount1[:, None])
        weight.scatter_add_(0, flat0, amount0.to(dtype=weight.dtype))
        weight.scatter_add_(0, flat1, amount1.to(dtype=weight.dtype))

        weight_img = weight.view(batch, height, width)
        out = out / weight.clamp_min(1.0e-6)[:, None]
        holes = weight_img < self.hole_thresh
        return out.view(batch, height, width, channels), holes, near_buffer

    def _fill_holes_background(self, image, holes, near_buffer):
        torch = self.torch
        batch, height, width, channels = image.shape
        valid = ~holes
        cols = torch.arange(width, device=self.device, dtype=torch.long).view(1, 1, width)
        cols = cols.expand(batch, height, width)

        left_idx = torch.where(valid, cols, torch.full_like(cols, -1))
        left_idx = torch.cummax(left_idx, dim=2).values
        right_idx = torch.where(valid, cols, torch.full_like(cols, width))
        right_idx = torch.cummin(torch.flip(right_idx, dims=(2,)), dim=2).values
        right_idx = torch.flip(right_idx, dims=(2,))

        left_ok = left_idx >= 0
        right_ok = right_idx < width
        li = left_idx.clamp(0, width - 1)
        ri = right_idx.clamp(0, width - 1)

        left_rgb = torch.gather(image, 2, li.unsqueeze(-1).expand(-1, -1, -1, channels))
        right_rgb = torch.gather(image, 2, ri.unsqueeze(-1).expand(-1, -1, -1, channels))
        left_near = torch.gather(near_buffer, 2, li)
        right_near = torch.gather(near_buffer, 2, ri)
        inf = torch.full_like(left_near, float("inf"))
        left_near = torch.where(left_ok, left_near, inf)
        right_near = torch.where(right_ok, right_near, inf)

        choose_right = right_ok & (~left_ok | (right_near <= left_near))
        fill = torch.where(choose_right.unsqueeze(-1), right_rgb, left_rgb)
        use_fill = holes & (left_ok | right_ok)
        return torch.where(use_fill.unsqueeze(-1), fill, image)

    def _shift_fill_holes(self, image, holes, direction: int, max_tries: int | None = None):
        torch = self.torch
        batch, height, width, channels = image.shape
        valid = ~holes
        cols = torch.arange(width, device=self.device, dtype=torch.long).view(1, 1, width)
        cols = cols.expand(batch, height, width)
        if direction < 0:
            src_idx = torch.where(valid, cols, torch.full_like(cols, -1))
            src_idx = torch.cummax(src_idx, dim=2).values
            src_ok = src_idx >= 0
        else:
            src_idx = torch.where(valid, cols, torch.full_like(cols, width))
            src_idx = torch.cummin(torch.flip(src_idx, dims=(2,)), dim=2).values
            src_idx = torch.flip(src_idx, dims=(2,))
            src_ok = src_idx < width
        gathered = torch.gather(
            image,
            2,
            src_idx.clamp(0, width - 1).unsqueeze(-1).expand(-1, -1, -1, channels),
        )
        take = holes & src_ok
        return torch.where(take.unsqueeze(-1), gathered, image)

    def _shift_fill_holes_iterative(self, image, holes, direction: int, max_tries: int | None = None):
        torch = self.torch
        if not bool(holes.any()):
            return image
        out = image.clone()
        remaining = holes.clone()
        width = image.shape[2]
        tries = width if max_tries is None else max(1, int(max_tries))
        for _ in range(tries):
            if not bool(remaining.any()):
                break
            if direction < 0:
                src = torch.cat([out[:, :, :1, :], out[:, :, :-1, :]], dim=2)
                src_valid = torch.cat([torch.zeros_like(remaining[:, :, :1]), ~remaining[:, :, :-1]], dim=2)
            else:
                src = torch.cat([out[:, :, 1:, :], out[:, :, -1:, :]], dim=2)
                src_valid = torch.cat([~remaining[:, :, 1:], torch.zeros_like(remaining[:, :, -1:])], dim=2)
            take = remaining & src_valid
            if not bool(take.any()):
                break
            out = torch.where(take.unsqueeze(-1), src, out)
            remaining = remaining & ~take
        return out

    def _soft_blend_holes(self, image, holes):
        import torch.nn.functional as F

        x = image.permute(0, 3, 1, 2).contiguous()
        mask = holes.unsqueeze(1).to(dtype=x.dtype)
        mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        mask = F.avg_pool2d(mask, kernel_size=5, stride=1, padding=2).clamp(0.0, 1.0) * 0.35
        blur = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        out = x * (1.0 - mask) + blur * mask
        return out.permute(0, 2, 3, 1).contiguous()

    def _fill_eye_holes(self, image, holes, near_buffer, direction: int):
        mode = self.hole_fill_mode
        if mode in {"none", HOLE_FILL_E2FGVI, HOLE_FILL_INVERSE_WARP}:
            return image
        if mode == "background":
            return self._fill_holes_background(image, holes, near_buffer)
        shifted = self._shift_fill_holes(image, holes, direction)
        if mode == "soft_shift":
            return self._soft_blend_holes(shifted, holes)
        return shifted

    def _project_or_concat(self, left, right, batch: int):
        import torch.nn.functional as F

        torch = self.torch
        if self.is_identity_projection:
            return torch.cat([left, right], dim=3)
        proj_grid = self.proj_grid.expand(batch, -1, -1, -1)
        left_proj = F.grid_sample(left, proj_grid, mode="bilinear", padding_mode="border", align_corners=True)
        right_proj = F.grid_sample(right, proj_grid, mode="bilinear", padding_mode="border", align_corners=True)
        left_proj = left_proj * self.proj_mask
        right_proj = right_proj * self.proj_mask
        return torch.cat([left_proj, right_proj], dim=3)

    def _frames_to_gpu_float_bchw(self, frames_rgb):
        """Upload an RGB batch to ``(B, 3, H, W)`` float32 in [0,1] on device.

        Accepts numpy ``(B, H, W, 3)`` uint8 or a torch uint8 tensor of the same
        layout already on CUDA.
        """
        torch = self.torch
        if _is_torch_tensor(frames_rgb):
            frame_t = frames_rgb
            if frame_t.device != self.device:
                frame_t = frame_t.to(self.device, non_blocking=True)
        else:
            frames = np.ascontiguousarray(frames_rgb)
            frame_t = torch.from_numpy(frames).to(self.device, non_blocking=True)
        return frame_t.permute(0, 3, 1, 2).contiguous().float().div_(255.0)

    def _render_batch_fast_tensor(self, frames_rgb, depths: Any):
        import torch.nn.functional as F

        torch = self.torch
        batch = int(frames_rgb.shape[0])

        with torch.inference_mode():
            frame_t = self._frames_to_gpu_float_bchw(frames_rgb)
            near = self._near_from_depths(depths)
            shift_norm = (_max_disparity_pixels(self.src_w, self.eye_distance_mm) * 0.5) * 2.0 / max(1, self.src_w - 1)
            grid_y = self.source_grid_y.expand(batch, -1, -1)
            grid_x = self.source_grid_x.expand(batch, -1, -1)
            left_grid = torch.stack((grid_x - near * shift_norm, grid_y), dim=-1)
            right_grid = torch.stack((grid_x + near * shift_norm, grid_y), dim=-1)
            left = F.grid_sample(frame_t, left_grid, mode="bilinear", padding_mode="border", align_corners=True)
            right = F.grid_sample(frame_t, right_grid, mode="bilinear", padding_mode="border", align_corners=True)
            sbs = self._project_or_concat(left, right, batch)
            return sbs.clamp(0.0, 1.0).mul_(255.0).round_().to(torch.uint8).permute(0, 2, 3, 1).contiguous()

    def _render_batch_tensor(self, frames_rgb, depths: Any):
        torch = self.torch
        if self.inverse_warp:
            return self._render_batch_fast_tensor(frames_rgb, depths)

        batch = int(frames_rgb.shape[0])

        with torch.inference_mode():
            frame_t = self._frames_to_gpu_float_bchw(frames_rgb)
            near = self._near_from_depths(depths)

            max_shift = _max_disparity_pixels(self.src_w, self.eye_distance_mm)
            frame_bhwc = frame_t.permute(0, 2, 3, 1).contiguous()
            left, left_holes, left_near = self._forward_warp_eye(frame_bhwc, near, max_shift, 1.0)
            right, right_holes, right_near = self._forward_warp_eye(frame_bhwc, near, max_shift, -1.0)
            left = self._fill_eye_holes(left, left_holes, left_near, -1)
            right = self._fill_eye_holes(right, right_holes, right_near, 1)
            left = left.permute(0, 3, 1, 2).contiguous()
            right = right.permute(0, 3, 1, 2).contiguous()

            sbs = self._project_or_concat(left, right, batch)
            out = (sbs.clamp(0.0, 1.0).mul_(255.0).round_().to(torch.uint8))
            return out.permute(0, 2, 3, 1).contiguous()

    def render_batch(self, frames_rgb, depths: Any) -> np.ndarray:
        return self._render_batch_tensor(frames_rgb, depths).cpu().numpy()

    def render_batch_nv12_packed(self, frames_rgb, depths: Any):
        """Return packed NV12 ``(B, H*3//2, W)`` uint8 on device, ready for
        ``gpu_engine.pynv_io.GpuNv12AppFrame``. Color matrix follows
        ``_yuv_matrix_for_shape`` (BT.709 for HD/UHD, BT.601 for SD)."""
        sbs_rgb = self._render_batch_tensor(frames_rgb, depths)
        return _torch_rgb_uint8_to_nv12_packed(sbs_rgb)

    def _rgb_u8_to_yuv420p_flat(self, rgb):
        import torch.nn.functional as F

        x = rgb.float()
        r = x[..., 0]
        g = x[..., 1]
        b = x[..., 2]
        height = int(rgb.shape[1])
        width = int(rgb.shape[2])
        y_coeff, u_coeff, v_coeff = _yuv_limited_coefficients(width, height)
        y = (
            16.0 + y_coeff[0] * r + y_coeff[1] * g + y_coeff[2] * b
        ).clamp(0.0, 255.0).round().to(self.torch.uint8)
        u_full = (
            128.0 + u_coeff[0] * r + u_coeff[1] * g + u_coeff[2] * b
        ).clamp(0.0, 255.0)
        v_full = (
            128.0 + v_coeff[0] * r + v_coeff[1] * g + v_coeff[2] * b
        ).clamp(0.0, 255.0)
        u = F.avg_pool2d(u_full.unsqueeze(1), kernel_size=2, stride=2).round().to(self.torch.uint8).squeeze(1)
        v = F.avg_pool2d(v_full.unsqueeze(1), kernel_size=2, stride=2).round().to(self.torch.uint8).squeeze(1)
        return self.torch.cat(
            [y.flatten(1), u.flatten(1), v.flatten(1)],
            dim=1,
        ).contiguous()

    def render_batch_async(self, frames_rgb: np.ndarray, depths: Any, output_pix_fmt: str = "rgb24") -> Any:
        torch = self.torch
        gpu_batch = self._render_batch_tensor(frames_rgb, depths)
        if output_pix_fmt == "yuv420p":
            gpu_batch = self._rgb_u8_to_yuv420p_flat(gpu_batch)
        try:
            cpu_batch = torch.empty(
                tuple(gpu_batch.shape),
                dtype=gpu_batch.dtype,
                device="cpu",
                pin_memory=True,
            )
        except Exception:
            return gpu_batch.cpu().numpy()
        current_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.copy_stream):
            cpu_batch.copy_(gpu_batch, non_blocking=True)
            gpu_batch.record_stream(self.copy_stream)
            event = torch.cuda.Event()
            event.record(self.copy_stream)
        return _AsyncSbsBatch(cpu_batch, event, gpu_batch)

    def stereo_raw_batch(self, frames_rgb: np.ndarray, depths: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        torch = self.torch
        frames = np.ascontiguousarray(frames_rgb)

        with torch.inference_mode():
            frame_t = torch.from_numpy(frames).to(self.device, non_blocking=True)
            frame_t = frame_t.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
            near = self._near_from_depths(depths)

            max_shift = _max_disparity_pixels(self.src_w, self.eye_distance_mm)
            frame_bhwc = frame_t.permute(0, 2, 3, 1).contiguous()
            left, left_holes, _ = self._forward_warp_eye(frame_bhwc, near, max_shift, 1.0)
            right, right_holes, _ = self._forward_warp_eye(frame_bhwc, near, max_shift, -1.0)

            left = left.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
            right = right.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
            return (
                left.contiguous().cpu().numpy(),
                right.contiguous().cpu().numpy(),
                left_holes.contiguous().cpu().numpy(),
                right_holes.contiguous().cpu().numpy(),
            )


def _make_torch_renderer(src_w: int, src_h: int, pmap: ProjectionMap, eye_distance_mm: float,
                         hole_fill_mode: str = DEFAULT_HOLE_FILL_MODE,
                         stabilize_mode: str = DEFAULT_STABILIZE_MODE,
                         log_callback=None):
    try:
        import torch

        if resolve_hole_fill_mode(hole_fill_mode) == "inpaint":
            return None
        if not torch.cuda.is_available():
            return None
        return TorchStereoRenderer(src_w, src_h, pmap, eye_distance_mm, hole_fill_mode, stabilize_mode)
    except Exception as exc:
        if log_callback:
            log_callback(f"[2dvr] CUDA renderer unavailable, using CPU renderer: {exc}")
        return None


def _ffmpeg_time_args(start_sec: float | None, duration_sec: float | None) -> list[str]:
    args: list[str] = []
    if start_sec is not None and start_sec > 0:
        args.extend(["-ss", f"{start_sec:.6f}"])
    if duration_sec is not None and duration_sec > 0:
        args.extend(["-t", f"{duration_sec:.6f}"])
    return args


@lru_cache(maxsize=1)
def _ffmpeg_supports_cuda_hwaccel() -> bool:
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-hwaccels"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            startupinfo=get_startupinfo(),
        )
    except Exception:
        return False
    return proc.returncode == 0 and "cuda" in (proc.stdout or "").lower()


def _resolve_decode_backend(info: VideoInfo) -> str:
    raw = os.environ.get("TOOL_2DVR_NVDEC", "1").strip().lower()
    if raw in {"0", "false", "no", "off", "cpu"}:
        return "cpu"
    if info.codec_name not in {"h264", "hevc", "h265"}:
        return "cpu"
    return "cuda" if _ffmpeg_supports_cuda_hwaccel() else "cpu"


def _build_decode_cmd(
    input_path: str,
    start_sec: float | None,
    duration_sec: float | None,
    *,
    backend: str = "cpu",
) -> list[str]:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        *_ffmpeg_time_args(start_sec, duration_sec),
    ]
    if backend == "cuda":
        cmd.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])
    cmd.extend([
        "-i", input_path,
        "-an", "-sn",
    ])
    if backend == "cuda":
        cmd.extend(["-vf", "scale_cuda=format=nv12,hwdownload,format=nv12,format=rgb24"])
    cmd.extend([
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-vsync", "0",
        "pipe:1",
    ])
    return cmd


def _build_encode_cmd(temp_video: str, out_w: int, out_h: int, fps: float, input_pix_fmt: str = "rgb24") -> list[str]:
    preset = os.environ.get("TOOL_2DVR_NVENC_PRESET", "p1").strip().lower() or "p1"
    cq = os.environ.get("TOOL_2DVR_NVENC_CQ", "20").strip() or "20"
    cmd = [
        "ffmpeg", "-hide_banner", "-y", "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", input_pix_fmt,
        "-s", f"{out_w}x{out_h}",
        "-r", f"{fps:.6f}",
        "-i", "pipe:0",
        "-c:v", "hevc_nvenc",
        "-preset", preset,
        "-rc", "vbr",
        "-cq", cq,
        "-b:v", "0",
        "-maxrate", "80M",
        "-bufsize", "160M",
        "-pix_fmt", "yuv420p",
        *_ffmpeg_output_color_args(out_w, out_h),
        "-bsf:v", _hevc_color_metadata_bsf(out_w, out_h),
    ]
    if os.environ.get("TOOL_2DVR_NVENC_HQ", "0").strip().lower() in {"1", "true", "yes", "on"}:
        cmd.extend(["-tune", "hq", "-multipass", "fullres", "-spatial_aq", "1", "-temporal_aq", "1"])
    cmd.append(temp_video)
    return cmd


@lru_cache(maxsize=1)
def _pynv_available() -> bool:
    """Probe whether the PyNv decode/encode stack is importable.

    Cached: invoking PyNv import errors per attempt is wasteful, and once it
    fails in a process it stays failing.
    """
    try:
        import PyNvVideoCodec  # noqa: F401
        from gpu_engine import pynv_io as _pynv_io  # noqa: F401
        from gpu_engine import runtime as _runtime
    except Exception:
        return False
    try:
        state = _runtime.warmup()
    except Exception:
        return False
    return bool(getattr(state, "available", False))


def _pynv_should_use(info: VideoInfo, backend: str, log_callback=None) -> bool:
    """Decide whether to try the PyNv pipeline for this input.

    ``backend`` is the user-resolved ``TOOL_2DVR_BACKEND`` value
    (``pynv``/``ffmpeg``/``auto``). Codec gating: only the well-supported NVDEC
    formats are routed through PyNv; the rest fall through to ffmpeg.
    """
    if backend == BACKEND_FFMPEG:
        return False
    if not _pynv_available():
        if backend == BACKEND_PYNV and log_callback:
            log_callback("[2dvr] PyNv requested but unavailable; falling back to ffmpeg")
        return False
    if not pynv_supports_codec(info.codec_name):
        if backend == BACKEND_PYNV and log_callback:
            log_callback(
                f"[2dvr] PyNv requested but codec '{info.codec_name}' is not NVDEC-supported; "
                "falling back to ffmpeg"
            )
        return False
    return True


def _pynv_encoder_kwargs(out_w: int, out_h: int, fps: float) -> dict:
    """NVENC kwargs aligned with the ffmpeg path's real-time settings.

    Preset/RC/bitrate defaults match _build_encode_cmd() so visual output is
    comparable across backends.
    """
    preset = os.environ.get("TOOL_2DVR_NVENC_PRESET", "P1").strip().upper() or "P1"
    if preset not in {f"P{i}" for i in range(1, 8)}:
        preset = "P1"
    # Heuristic bitrate ~ 0.07 bit/px/frame; match _quality_bitrate_bps spirit.
    target_bps = max(1_000_000, int(out_w * out_h * max(1.0, fps) * 0.07))
    maxrate_bps = max(target_bps, int(target_bps * 1.5))
    kwargs = {
        "fps": f"{fps:.6f}",
        "gop": "60",
        "bf": "0",
        "tuning_info": "high_quality",
        "preset": preset,
        "rc": "vbr",
        "bitrate": str(target_bps),
        "maxbitrate": str(maxrate_bps),
    }
    return kwargs


def _pynv_color_metadata(out_w: int, out_h: int):
    """Build ColorMetadata for PyNv mux step."""
    from gpu_engine.probe import ColorMetadata

    name = "smpte170m" if _yuv_matrix_for_shape(out_w, out_h) == "bt601" else "bt709"
    return ColorMetadata(color_range="tv", color_space=name, color_transfer=name, color_primaries=name)


def _cupy_to_torch_zero_copy(arr):
    """Convert a CuPy ndarray to a torch tensor on the same CUDA device.

    Prefer the modern ``__dlpack__`` protocol (torch 2.x). Fall back to
    ``cupy → numpy → torch`` only when the dlpack bridge errors out, which is a
    safety net for older CuPy/torch combos and should not trigger in CUDA 12.x
    aligned environments.
    """
    import torch

    try:
        return torch.from_dlpack(arr)
    except Exception:
        # Fallback via CPU round-trip; preserves correctness if dlpack bridge fails.
        import cupy as cp

        return torch.from_numpy(cp.asnumpy(arr)).to("cuda", non_blocking=True)


def _torch_to_cupy_zero_copy(tensor):
    """Bridge a CUDA torch tensor to a CuPy ndarray with shared memory."""
    import cupy as cp

    tensor = tensor.contiguous()
    try:
        return cp.from_dlpack(tensor)
    except Exception:
        return cp.asarray(tensor.detach().cpu().numpy())


def _torch_gpu_frame_to_rgb_uint8(gpu_frame, *, device):
    """Convert a GpuNv12Frame to torch.uint8 (H, W, 3) RGB on the same CUDA device.

    The PyNv plane is u8 contiguous; we re-wrap the raw pointer as a torch tensor
    with the same storage to avoid host round-trip.
    """
    import cupy as cp

    y_cp, uv_cp = gpu_frame.y_uv_cupy()
    y_cp = cp.ascontiguousarray(y_cp)
    uv_cp = cp.ascontiguousarray(uv_cp)
    y_t = _cupy_to_torch_zero_copy(y_cp)
    uv_t = _cupy_to_torch_zero_copy(uv_cp)
    rgb = _torch_nv12_to_rgb_uint8(y_t, uv_t, width=gpu_frame.width, height=gpu_frame.height)
    if rgb.device != device:
        rgb = rgb.to(device, non_blocking=True)
    return rgb


def _torch_packed_nv12_to_cupy(packed_t):
    """Bridge a torch.uint8 (H*3//2, W) tensor to a CuPy ndarray with shared memory."""
    return _torch_to_cupy_zero_copy(packed_t)


def _run_logged(cmd: list[str], log_callback=None, process_callback=None) -> None:
    if log_callback:
        log_callback("Executing: " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        errors="replace",
        startupinfo=get_startupinfo(),
    )
    if process_callback:
        process_callback(proc)
    assert proc.stdout is not None
    for line in proc.stdout:
        if log_callback:
            log_callback(line.rstrip())
    proc.wait()
    if proc.returncode != 0:
        raise TwoDVRRuntimeError(f"Command failed with code {proc.returncode}")


def _validate_params(input_path: str, projection: str, eye_distance_mm: float, flat_fov_deg: float,
                     start_sec: float | None, end_sec: float | None) -> None:
    if not input_path or not os.path.exists(input_path):
        raise TwoDVRRuntimeError("Input video does not exist.")
    if projection not in {PROJECTION_FLAT_3D, PROJECTION_FISHEYE, PROJECTION_HEQUIRECT}:
        raise TwoDVRRuntimeError(f"Unknown projection: {projection}")
    if eye_distance_mm <= 0:
        raise TwoDVRRuntimeError("Eye distance must be greater than zero.")
    if not math.isfinite(flat_fov_deg) or flat_fov_deg < MIN_FLAT_FOV_DEG or flat_fov_deg > MAX_FLAT_FOV_DEG:
        raise TwoDVRRuntimeError(
            f"Flat projection FOV must be between {MIN_FLAT_FOV_DEG:g} and {MAX_FLAT_FOV_DEG:g} degrees."
        )
    if start_sec is not None and end_sec is not None and start_sec >= end_sec:
        raise TwoDVRRuntimeError("Start time must be earlier than end time.")


def convert_2d_to_vr(
    input_path: str,
    output_dir: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    projection: str = DEFAULT_PROJECTION,
    eye_distance_mm: float = DEFAULT_EYE_DISTANCE_MM,
    flat_fov_deg: float = DEFAULT_FLAT_FOV_DEG,
    hole_fill_mode: str = DEFAULT_HOLE_FILL_MODE,
    stabilize_mode: str | None = None,
    log_callback=None,
    process_callback=None,
) -> str:
    start_sec = parse_time_to_seconds(start_time)
    end_sec = parse_time_to_seconds(end_time)
    try:
        flat_fov_deg = float(flat_fov_deg)
    except (TypeError, ValueError) as exc:
        raise TwoDVRRuntimeError("Flat projection FOV must be a number.") from exc
    _validate_params(input_path, projection, float(eye_distance_mm), flat_fov_deg, start_sec, end_sec)

    missing = [m for m in check_dependencies() if m in {"ffmpeg", "ffprobe"}]
    if missing:
        raise TwoDVRRuntimeError(f"Missing required tools: {', '.join(missing)}")

    info = get_video_info(input_path)
    duration_sec = (end_sec - start_sec) if (start_sec is not None and end_sec is not None) else None
    if start_sec is None and end_sec is not None:
        duration_sec = end_sec
    if start_sec is not None and end_sec is None and info.duration > 0:
        duration_sec = max(0.0, info.duration - start_sec)

    output = output_path(input_path, output_dir, projection, start_time, end_time)
    out_dir = Path(output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    estimator = DA3DepthEstimator(default_da3_dir(), log_callback=log_callback)
    pmap = make_projection_map(info.width, info.height, projection, flat_fov_deg)
    out_w, out_h = pmap.out_w * 2, pmap.out_h
    pipe_pix_fmt = resolve_pipe_pix_fmt(out_w, out_h)
    frame_size = info.width * info.height * 3
    temp_video = str(Path(output).with_suffix(".video_only.tmp.mp4"))
    temp_raw_hevc = str(Path(output).with_suffix(".video_only.tmp.raw.hevc"))
    backend_resolved = resolve_backend()
    use_pynv = _pynv_should_use(info, backend_resolved, log_callback=log_callback)
    depth_batch_size = resolve_depth_batch_size()
    fill_mode = resolve_hole_fill_mode(hole_fill_mode)
    stabilize_mode_resolved = resolve_stabilize_mode(stabilize_mode)
    use_e2fgvi = fill_mode == HOLE_FILL_E2FGVI
    e2fgvi_chunk_size = resolve_e2fgvi_chunk_size()
    e2fgvi_inpainter = None
    e2fgvi_left_buffer: list[np.ndarray] = []
    e2fgvi_right_buffer: list[np.ndarray] = []
    e2fgvi_left_mask_buffer: list[np.ndarray] = []
    e2fgvi_right_mask_buffer: list[np.ndarray] = []
    renderer = _make_torch_renderer(
        info.width, info.height, pmap, float(eye_distance_mm),
        hole_fill_mode=fill_mode,
        stabilize_mode=stabilize_mode_resolved,
        log_callback=log_callback,
    )
    render_backend = renderer.backend if renderer is not None else "cpu_forward_zfill"
    save_eye_debug = debug_eye_enabled()
    eye_debug_saved = False

    handle = PipelineProcess()
    if process_callback:
        process_callback(handle)

    decode_backend = _resolve_decode_backend(info)
    decode_cmd = _build_decode_cmd(input_path, start_sec, duration_sec, backend=decode_backend)
    encode_cmd = _build_encode_cmd(temp_video, out_w, out_h, info.fps, input_pix_fmt=pipe_pix_fmt)

    if log_callback:
        log_callback(
            f"[2dvr] source={info.width}x{info.height} fps={info.fps:.3f}, "
            f"output={out_w}x{out_h}, projection={projection}, fov={flat_fov_deg:.1f}, "
            f"yaw=0.0, pitch=0.0, eye={float(eye_distance_mm):.1f}mm, "
            f"hole_fill={fill_mode}, stabilize={stabilize_mode_resolved}"
        )
        log_callback(
            f"[2dvr] backend={backend_resolved} (pynv_active={int(use_pynv)}, codec={info.codec_name or '?'})"
        )
        log_callback(
            f"[2dvr] pipeline: decode=ffmpeg/{decode_backend}/rawvideo, DA3=depth-only batch={depth_batch_size}, "
            f"render={render_backend}, pipe={pipe_pix_fmt}, encode=hevc_nvenc"
        )
        if renderer is not None and getattr(renderer, "stabilize_enabled", False):
            log_callback(
                "[2dvr] stabilize: "
                f"mode={renderer.stabilize_mode}, norm_alpha={renderer.norm_alpha:.3f}, "
                f"depth_beta={renderer.depth_beta:.3f}, adaptive_beta={int(renderer.adaptive_beta)}, "
                f"subpixel_splat={int(renderer.subpixel_splat_enabled)}"
            )
        if use_e2fgvi:
            log_callback(
                f"[2dvr] E2FGVI video inpaint enabled: chunk={e2fgvi_chunk_size}. "
                "E2FGVI is CC BY-NC 4.0; use it for non-commercial evaluation unless separately licensed."
            )

    decode_proc = None
    encode_proc = None
    frames_done = 0
    last_log = time.perf_counter()
    started = time.perf_counter()
    writer_queue: queue.Queue[Any | None] | None = None
    thread_errors: queue.Queue[BaseException] | None = None

    def _raise_thread_error() -> None:
        if thread_errors is None:
            return
        try:
            exc = thread_errors.get_nowait()
        except queue.Empty:
            return
        if isinstance(exc, OperationCancelled):
            raise exc
        raise exc

    def _write_sbs_frames_sync(sbs_batch: Any) -> None:
        nonlocal frames_done, last_log
        sbs_array = _sbs_batch_to_numpy(sbs_batch)
        for sbs in sbs_array:
            if handle.cancelled:
                raise OperationCancelled()
            if encode_proc is None or encode_proc.stdin is None:
                raise TwoDVRRuntimeError("ffmpeg encoder is not running.")
            view = memoryview(sbs)
            if view.ndim != 1:
                view = view.cast("B")
            encode_proc.stdin.write(view)
            frames_done += 1
        now = time.perf_counter()
        if log_callback and now - last_log >= 1.0:
            fps = frames_done / max(0.001, now - started)
            log_callback(f"[2dvr] frames={frames_done} | {fps:.2f} fps | render={render_backend}")
            last_log = now

    def _write_sbs_frames(sbs_batch: Any) -> None:
        if _sbs_batch_is_empty(sbs_batch):
            return
        if writer_queue is None:
            _write_sbs_frames_sync(sbs_batch)
            return
        while True:
            if handle.cancelled:
                raise OperationCancelled()
            _raise_thread_error()
            try:
                writer_queue.put(sbs_batch, timeout=0.1)
                return
            except queue.Full:
                continue

    def _render_with_depths(frames_batch: list[np.ndarray], depths: Any) -> Any:
        nonlocal renderer, render_backend
        if renderer is not None:
            try:
                if hasattr(renderer, "render_batch_async"):
                    return renderer.render_batch_async(np.stack(frames_batch, axis=0), depths, output_pix_fmt=pipe_pix_fmt)
                sbs_rgb = renderer.render_batch(np.stack(frames_batch, axis=0), depths)
                return _rgb_batch_to_yuv420p(sbs_rgb) if pipe_pix_fmt == "yuv420p" else sbs_rgb
            except Exception as exc:
                if _is_cuda_oom(exc):
                    _empty_cuda_cache()
                    if len(frames_batch) > 1:
                        mid = max(1, len(frames_batch) // 2)
                        if log_callback:
                            log_callback(
                                f"[2dvr] CUDA renderer OOM at batch={len(frames_batch)}; "
                                f"retrying as {mid}+{len(frames_batch) - mid}"
                            )
                        first = _sbs_batch_to_numpy(_render_with_depths(frames_batch[:mid], depths[:mid]))
                        second = _sbs_batch_to_numpy(_render_with_depths(frames_batch[mid:], depths[mid:]))
                        return np.concatenate([first, second], axis=0)
                if log_callback:
                    log_callback(f"[2dvr] CUDA renderer failed, falling back to CPU renderer: {exc}")
                renderer = None
                render_backend = "cpu_forward_zfill"
        depths_cpu = _depths_for_cpu_frames(depths, frames_batch)
        sbs_rgb = np.stack(
            [render_sbs_frame(frame, depth, projection, float(eye_distance_mm), pmap, fill_mode)
             for frame, depth in zip(frames_batch, depths_cpu)],
            axis=0,
        )
        return _rgb_batch_to_yuv420p(sbs_rgb) if pipe_pix_fmt == "yuv420p" else sbs_rgb

    def _append_e2fgvi_stereo_batch(frames_batch: list[np.ndarray], depths: Any) -> None:
        nonlocal renderer, render_backend
        if renderer is not None:
            try:
                left, right, left_masks, right_masks = renderer.stereo_raw_batch(np.stack(frames_batch, axis=0), depths)
                e2fgvi_left_buffer.extend(left)
                e2fgvi_right_buffer.extend(right)
                e2fgvi_left_mask_buffer.extend(left_masks)
                e2fgvi_right_mask_buffer.extend(right_masks)
                return
            except Exception as exc:
                if _is_cuda_oom(exc):
                    _empty_cuda_cache()
                    if len(frames_batch) > 1:
                        mid = max(1, len(frames_batch) // 2)
                        if log_callback:
                            log_callback(
                                f"[2dvr] CUDA raw stereo OOM at batch={len(frames_batch)}; "
                                f"retrying as {mid}+{len(frames_batch) - mid}"
                            )
                        _append_e2fgvi_stereo_batch(frames_batch[:mid], depths[:mid])
                        _append_e2fgvi_stereo_batch(frames_batch[mid:], depths[mid:])
                        return
                if log_callback:
                    log_callback(f"[2dvr] CUDA raw stereo failed, falling back to CPU raw stereo: {exc}")
                renderer = None
                render_backend = "cpu_forward_zfill+e2fgvi"

        depths_cpu = _depths_for_cpu_frames(depths, frames_batch)
        for frame, depth in zip(frames_batch, depths_cpu):
            stereo = _make_stereo_result(frame, depth, float(eye_distance_mm), "none")
            e2fgvi_left_buffer.append(stereo.left_before_fill)
            e2fgvi_right_buffer.append(stereo.right_before_fill)
            e2fgvi_left_mask_buffer.append(stereo.left_holes)
            e2fgvi_right_mask_buffer.append(stereo.right_holes)

    def _flush_e2fgvi_buffer(force: bool = False) -> None:
        nonlocal e2fgvi_inpainter, render_backend
        if not use_e2fgvi:
            return
        try:
            from .e2fgvi_backend import E2FGVIBackendError, E2FGVIInpainter
        except ImportError:
            from e2fgvi_backend import E2FGVIBackendError, E2FGVIInpainter
        while len(e2fgvi_left_buffer) >= e2fgvi_chunk_size or (force and e2fgvi_left_buffer):
            take = len(e2fgvi_left_buffer) if force and len(e2fgvi_left_buffer) < e2fgvi_chunk_size else e2fgvi_chunk_size
            left = np.stack(e2fgvi_left_buffer[:take], axis=0)
            right = np.stack(e2fgvi_right_buffer[:take], axis=0)
            left_masks = np.stack(e2fgvi_left_mask_buffer[:take], axis=0)
            right_masks = np.stack(e2fgvi_right_mask_buffer[:take], axis=0)
            del e2fgvi_left_buffer[:take]
            del e2fgvi_right_buffer[:take]
            del e2fgvi_left_mask_buffer[:take]
            del e2fgvi_right_mask_buffer[:take]

            if e2fgvi_inpainter is None:
                e2fgvi_inpainter = E2FGVIInpainter(project_root(), log_callback=log_callback)
            try:
                if log_callback:
                    log_callback(f"[2dvr] E2FGVI inpaint chunk: frames={take}")
                try:
                    left, right = e2fgvi_inpainter.inpaint_pair(left, right, left_masks, right_masks)
                except Exception as exc:
                    if _is_cuda_oom(exc):
                        _empty_cuda_cache()
                        if log_callback:
                            log_callback(
                                f"[2dvr] E2FGVI paired batch OOM at frames={take}; "
                                "retrying left and right separately"
                            )
                        left = e2fgvi_inpainter.inpaint(left, left_masks)
                        right = e2fgvi_inpainter.inpaint(right, right_masks)
                    else:
                        raise
            except E2FGVIBackendError:
                raise
            except Exception as exc:
                raise TwoDVRRuntimeError(f"E2FGVI video inpaint failed: {exc}") from exc

            sbs_batch = np.stack(
                [np.concatenate([_sample_rgb_xy(l, pmap), _sample_rgb_xy(r, pmap)], axis=1)
                 for l, r in zip(left, right)],
                axis=0,
            )
            if pipe_pix_fmt == "yuv420p":
                sbs_batch = _rgb_batch_to_yuv420p(sbs_batch)
            raw_backend = renderer.backend if renderer is not None else "cpu_forward_zfill"
            e2fgvi_device = getattr(e2fgvi_inpainter, "_device", None)
            e2fgvi_backend = "e2fgvi_cuda" if e2fgvi_device is not None and e2fgvi_device.type == "cuda" else "e2fgvi_cpu"
            render_backend = f"{raw_backend}+{e2fgvi_backend}"
            _write_sbs_frames(sbs_batch)

    def _render_frame_batch(frames_batch: list[np.ndarray]) -> Any:
        nonlocal eye_debug_saved
        if renderer is not None and hasattr(renderer, "observe_rgb_batch"):
            renderer.observe_rgb_batch(frames_batch)
        depths = _predict_depths_resilient(estimator, frames_batch, log_callback)
        if save_eye_debug and not eye_debug_saved and len(frames_batch) > 0:
            debug_depth = _depths_for_cpu_frames(depths[:1], frames_batch[:1])[0]
            _save_eye_debug_images(
                output,
                frames_batch[0],
                debug_depth,
                projection,
                float(eye_distance_mm),
                pmap,
                hole_fill_mode=("none" if use_e2fgvi else fill_mode),
                stabilize_mode=stabilize_mode_resolved,
                log_callback=log_callback,
            )
            eye_debug_saved = True
        if use_e2fgvi:
            _append_e2fgvi_stereo_batch(frames_batch, depths)
            _flush_e2fgvi_buffer(False)
            return np.empty((0, out_h, out_w, 3), dtype=np.uint8)
        return _render_with_depths(frames_batch, depths)

    def _write_sbs_batch(frames_batch: list[np.ndarray]) -> None:
        if not frames_batch:
            return
        sbs_batch = _render_frame_batch(frames_batch)
        if not use_e2fgvi:
            _write_sbs_frames(sbs_batch)

    def _queue_put_thread(target_queue, item) -> bool:
        while True:
            try:
                target_queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                if handle.cancelled:
                    return False
                if thread_errors is not None and not thread_errors.empty():
                    return False

    def _run_pynv_mux(raw_hevc_path: str) -> None:
        """Mux raw HEVC + source audio into the final output mp4."""
        if not os.path.exists(raw_hevc_path):
            raise TwoDVRRuntimeError(f"PyNv raw HEVC missing: {raw_hevc_path}")
        from gpu_engine import mux as _gpu_mux

        color = _pynv_color_metadata(out_w, out_h)
        # If start/end was specified, align audio with `-ss/-t` to match the
        # decoded segment. duration_sec is computed against the source clip.
        if log_callback:
            log_callback(f"[2dvr] PyNv mux: raw={raw_hevc_path} -> {output}")
        _gpu_mux.mux_hevc_with_audio(
            raw_hevc_path,
            output,
            fps=float(info.fps),
            color=color,
            audio_source=input_path,
            audio_start_sec=start_sec if (start_sec and start_sec > 0) else None,
            audio_duration=duration_sec if (duration_sec and duration_sec > 0) else None,
            shortest=True,
            log_callback=log_callback,
        )

    def _run_transcode_attempt_pynv() -> None:
        """Single-threaded GPU pipeline: PyNv NVDEC -> torch render -> PyNv NVENC -> raw .hevc."""
        nonlocal frames_done, last_log, started, eye_debug_saved, render_backend

        if use_e2fgvi:
            raise TwoDVRRuntimeError("PyNv backend does not support E2FGVI hole_fill_mode")
        if renderer is None:
            raise TwoDVRRuntimeError("PyNv backend requires a CUDA renderer")

        import torch as _torch
        from gpu_engine import pynv_io as _pynv

        frames_done = 0
        started = time.perf_counter()
        last_log = time.perf_counter()
        eye_debug_saved = False
        if os.path.exists(temp_raw_hevc):
            try:
                os.remove(temp_raw_hevc)
            except Exception:
                pass

        device = _torch.device("cuda")

        # Probe decoder to compute frame range from start/duration.
        probe = _pynv.PyNvSimpleDecoder(Path(input_path), gpu_id=0, bit_depth=8)
        try:
            total_frames = len(probe)
            if start_sec and start_sec > 0:
                start_idx = max(0, min(total_frames - 1, probe.index_at_time(float(start_sec))))
            else:
                start_idx = 0
            if duration_sec and duration_sec > 0:
                stop_time = (start_sec or 0.0) + float(duration_sec)
                end_idx_inclusive = probe.index_at_time(stop_time)
                end_idx = min(total_frames, max(start_idx + 1, end_idx_inclusive + 1))
            else:
                end_idx = total_frames
        finally:
            probe.stop()

        if end_idx <= start_idx:
            raise TwoDVRRuntimeError(f"PyNv frame range empty: start={start_idx} end={end_idx}")

        if log_callback:
            log_callback(
                f"[2dvr] PyNv pipeline: decode=pynv-threaded, frames=[{start_idx},{end_idx}), "
                f"render={render_backend}, encode=pynv hevc_nvenc"
            )

        # Sequential decoder for fast NVDEC throughput. We always start the
        # threaded decoder at frame 0 and discard preroll frames in the main
        # loop. The decoder's strict seek-PTS check rejects most non-keyframe
        # starts, and the overhead of skipping <2s of preroll is negligible
        # compared with running NVDEC random-access SimpleDecoder.
        serial = _pynv.PyNvThreadedSerialDecoder(
            Path(input_path), gpu_id=0, bit_depth=8,
            start_frame=0, batch_size=max(2, depth_batch_size),
        )
        enc_kwargs = _pynv_encoder_kwargs(out_w, out_h, float(info.fps))
        encoder = _pynv.PyNvEncoderSession(out_w, out_h, bit_depth=8, codec="hevc", **enc_kwargs)

        prev_backend = render_backend
        render_backend = f"{prev_backend}+pynv"

        try:
            with open(temp_raw_hevc, "wb") as raw_file:
                rgb_batch: list[_torch.Tensor] = []
                for idx in range(0, end_idx):
                    if handle.cancelled:
                        raise OperationCancelled()
                    gpu_frame = serial.frame_at(idx)
                    if idx < start_idx:
                        # Discard preroll without bridging to torch. PyNv's batched
                        # memory is freed when the next get_batch_frames() is called.
                        continue
                    # NV12 -> RGB on GPU; the conversion reads via cupy view but does
                    # not retain ownership of PyNv's batched memory beyond this call.
                    rgb = _torch_gpu_frame_to_rgb_uint8(gpu_frame, device=device)
                    rgb_batch.append(rgb)
                    if len(rgb_batch) >= depth_batch_size:
                        _flush_pynv_batch(rgb_batch, encoder, raw_file)
                        rgb_batch = []
                if rgb_batch:
                    _flush_pynv_batch(rgb_batch, encoder, raw_file)
                tail = encoder.flush()
                if tail:
                    raw_file.write(tail)
        finally:
            try:
                serial.stop()
            except Exception:
                pass

        if frames_done <= 0:
            raise TwoDVRRuntimeError("No frames were processed via PyNv backend.")

    def _flush_pynv_batch(rgb_batch_list, encoder, raw_file) -> None:
        nonlocal frames_done, last_log, eye_debug_saved
        import torch as _torch
        from gpu_engine import pynv_io as _pynv

        if not rgb_batch_list:
            return
        rgb_batch_t = _torch.stack(rgb_batch_list, dim=0).contiguous()
        depths = _predict_depths_resilient(estimator, rgb_batch_t, log_callback)
        if save_eye_debug and not eye_debug_saved and rgb_batch_t.shape[0] > 0:
            try:
                debug_rgb = rgb_batch_t[0].detach().cpu().numpy()
                debug_depth = _depths_for_cpu_frames(depths[:1], [debug_rgb])[0]
                _save_eye_debug_images(
                    output, debug_rgb, debug_depth, projection,
                    float(eye_distance_mm), pmap,
                    hole_fill_mode=fill_mode,
                    stabilize_mode=stabilize_mode_resolved,
                    log_callback=log_callback,
                )
            except Exception as exc:
                if log_callback:
                    log_callback(f"[2dvr] eye debug skipped on PyNv path: {exc}")
            eye_debug_saved = True

        packed_nv12 = renderer.render_batch_nv12_packed(rgb_batch_t, depths)
        # packed_nv12: torch.uint8 (B, out_h*3//2, out_w) on CUDA.
        for i in range(int(packed_nv12.shape[0])):
            if handle.cancelled:
                raise OperationCancelled()
            frame_packed = packed_nv12[i].contiguous()
            cp_packed = _torch_packed_nv12_to_cupy(frame_packed)
            app = _pynv.GpuNv12AppFrame(cp_packed, out_w, out_h)
            data = encoder.encode(app, force_idr=(frames_done == 0))
            if data:
                raw_file.write(data)
            frames_done += 1
        now = time.perf_counter()
        if log_callback and now - last_log >= 1.0:
            fps_log = frames_done / max(0.001, now - started)
            log_callback(f"[2dvr] frames={frames_done} | {fps_log:.2f} fps | render={render_backend}")
            last_log = now

    def _run_transcode_attempt(current_decode_backend: str, current_decode_cmd: list[str]) -> None:
        nonlocal decode_proc, encode_proc, frames_done, last_log, started
        nonlocal writer_queue, thread_errors, e2fgvi_inpainter, eye_debug_saved

        frames_done = 0
        last_log = time.perf_counter()
        started = time.perf_counter()
        eye_debug_saved = False
        e2fgvi_inpainter = None
        e2fgvi_left_buffer.clear()
        e2fgvi_right_buffer.clear()
        e2fgvi_left_mask_buffer.clear()
        e2fgvi_right_mask_buffer.clear()
        if os.path.exists(temp_video):
            try:
                os.remove(temp_video)
            except Exception:
                pass

        reader_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=4)
        writer_queue = queue.Queue(maxsize=2)
        thread_errors = queue.Queue()

        if log_callback:
            log_callback(f"[2dvr] decode backend attempt: {current_decode_backend}")
            log_callback("Executing: " + " ".join(current_decode_cmd))
            log_callback("Executing: " + " ".join(encode_cmd))

        decode_proc = subprocess.Popen(
            current_decode_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=get_startupinfo(),
        )
        handle.add(decode_proc)
        encode_proc = subprocess.Popen(
            encode_cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=get_startupinfo(),
        )
        handle.add(encode_proc)
        assert decode_proc.stdout is not None
        assert encode_proc.stdin is not None

        def _kill_current_processes() -> None:
            for proc in (decode_proc, encode_proc):
                try:
                    if proc is not None and proc.poll() is None:
                        proc.kill()
                except Exception:
                    pass

        def _reader_loop() -> None:
            try:
                assert decode_proc is not None and decode_proc.stdout is not None
                while not handle.cancelled:
                    raw = decode_proc.stdout.read(frame_size)
                    if not raw:
                        break
                    if len(raw) != frame_size:
                        raise TwoDVRRuntimeError(f"Short frame read: {len(raw)} of {frame_size} bytes")
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape(info.height, info.width, 3).copy()
                    _queue_put_thread(reader_queue, frame)
            except BaseException as exc:
                if thread_errors is not None:
                    thread_errors.put(exc)
                _kill_current_processes()
            finally:
                _queue_put_thread(reader_queue, None)

        def _writer_loop() -> None:
            try:
                assert writer_queue is not None
                while True:
                    sbs_batch = writer_queue.get()
                    if sbs_batch is None:
                        break
                    _write_sbs_frames_sync(sbs_batch)
            except BaseException as exc:
                if thread_errors is not None:
                    thread_errors.put(exc)
                _kill_current_processes()

        reader_thread = threading.Thread(target=_reader_loop, name="2dvr-reader", daemon=True)
        writer_thread = threading.Thread(target=_writer_loop, name="2dvr-writer", daemon=True)
        reader_thread.start()
        writer_thread.start()
        writer_stop_sent = False

        try:
            frame_batch: list[np.ndarray] = []
            while True:
                if handle.cancelled:
                    raise OperationCancelled()
                _raise_thread_error()
                try:
                    frame = reader_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if frame is None:
                    break
                frame_batch.append(frame)
                if len(frame_batch) >= depth_batch_size:
                    _write_sbs_batch(frame_batch)
                    frame_batch = []

            _write_sbs_batch(frame_batch)
            _flush_e2fgvi_buffer(True)
            assert writer_queue is not None
            writer_stop_sent = _queue_put_thread(writer_queue, None)
            writer_thread.join()
            _raise_thread_error()

            try:
                encode_proc.stdin.close()
            except Exception:
                pass
            encode_proc.wait()
            decode_proc.wait()
            reader_thread.join(timeout=2.0)

            dec_err = (decode_proc.stderr.read() if decode_proc.stderr else b"").decode("utf-8", "replace")
            enc_err = (encode_proc.stderr.read() if encode_proc.stderr else b"").decode("utf-8", "replace")
            if decode_proc.returncode not in (0, None):
                raise TwoDVRRuntimeError(f"ffmpeg decode failed ({decode_proc.returncode}): {dec_err.strip()}")
            if encode_proc.returncode not in (0, None):
                raise TwoDVRRuntimeError(f"ffmpeg encode failed ({encode_proc.returncode}): {enc_err.strip()}")
            if frames_done <= 0:
                raise TwoDVRRuntimeError("No frames were processed.")
        finally:
            if writer_queue is not None and not writer_stop_sent:
                _queue_put_thread(writer_queue, None)
            for thread in (reader_thread, writer_thread):
                try:
                    thread.join(timeout=2.0)
                except Exception:
                    pass
            for proc in (decode_proc, encode_proc):
                try:
                    if proc is not None and proc.poll() is None:
                        proc.kill()
                except Exception:
                    pass
            writer_queue = None
            thread_errors = None

    try:
        used_pynv = False
        if use_pynv:
            try:
                _run_transcode_attempt_pynv()
                _run_pynv_mux(temp_raw_hevc)
                used_pynv = True
            except OperationCancelled:
                raise
            except Exception as exc:
                if backend_resolved == BACKEND_PYNV:
                    raise
                if log_callback:
                    log_callback(f"[2dvr] PyNv pipeline failed, falling back to ffmpeg: {exc}")

        if not used_pynv:
            decode_attempts = [(decode_backend, decode_cmd)]
            if decode_backend == "cuda":
                decode_attempts.append(("cpu", _build_decode_cmd(input_path, start_sec, duration_sec, backend="cpu")))

            last_attempt_error: Exception | None = None
            decode_success = False
            for current_decode_backend, current_decode_cmd in decode_attempts:
                try:
                    _run_transcode_attempt(current_decode_backend, current_decode_cmd)
                    decode_backend = current_decode_backend
                    decode_success = True
                    break
                except OperationCancelled:
                    raise
                except Exception as exc:
                    last_attempt_error = exc
                    if current_decode_backend == "cuda" and frames_done <= 0 and not handle.cancelled:
                        if log_callback:
                            log_callback(f"[2dvr] NVDEC failed before first frame; retrying CPU decode: {exc}")
                        continue
                    raise
            if not decode_success:
                if last_attempt_error is not None:
                    raise last_attempt_error
                raise TwoDVRRuntimeError("No decode attempts were available.")

            mux_cmd = [
                "ffmpeg", "-hide_banner", "-y",
                *_ffmpeg_time_args(start_sec, duration_sec),
                "-i", input_path,
                "-i", temp_video,
                "-map", "1:v:0",
                "-map", "0:a?",
                "-c:v", "copy",
                "-c:a", "copy",
                "-shortest",
                output,
            ]
            _run_logged(mux_cmd, log_callback=log_callback)

        if log_callback:
            elapsed = time.perf_counter() - started
            tag = "pynv" if used_pynv else "ffmpeg"
            log_callback(f"[2dvr] Done [{tag}]: {output} ({frames_done} frames, {elapsed:.1f}s)")
        return output
    except OperationCancelled:
        handle.kill()
        raise
    finally:
        for proc in (decode_proc, encode_proc):
            try:
                if proc is not None and proc.poll() is None:
                    proc.kill()
            except Exception:
                pass
        for path in (temp_video, temp_raw_hevc):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
