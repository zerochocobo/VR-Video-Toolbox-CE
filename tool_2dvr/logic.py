"""2D video to depth-based VR conversion.

This module intentionally keeps Depth Anything 3 integration behind a small
adapter. The video pipeline and stereo rendering are local code, inspired by
the common depth-to-stereo workflow but not copied from depth-surge-3d.
"""
from __future__ import annotations

import json
import math
import os
import importlib.util
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
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
HOLE_FILL_MODES = {"soft_shift", "shift_fill", "background", "inpaint", HOLE_FILL_E2FGVI, "none"}
DEFAULT_E2FGVI_CHUNK_SIZE = 12
DEFAULT_FLAT_FOV_DEG = 80.0
MIN_FLAT_FOV_DEG = 1.0
MAX_FLAT_FOV_DEG = 179.0


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


def resolve_hole_fill_mode(value: str | None = None) -> str:
    raw = (value or os.environ.get("TOOL_2DVR_HOLE_FILL") or DEFAULT_HOLE_FILL_MODE).strip().lower()
    return raw if raw in HOLE_FILL_MODES else DEFAULT_HOLE_FILL_MODE


def resolve_e2fgvi_chunk_size() -> int:
    raw = os.environ.get("TOOL_2DVR_E2FGVI_CHUNK", "").strip()
    try:
        value = int(raw) if raw else DEFAULT_E2FGVI_CHUNK_SIZE
    except ValueError:
        value = DEFAULT_E2FGVI_CHUNK_SIZE
    return max(2, value)


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
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate:format=duration",
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
    return VideoInfo(width=width, height=height, fps=fps, duration=duration)


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

    def predict_batch(self, frames_rgb: list[np.ndarray]) -> np.ndarray:
        if not frames_rgb:
            return np.empty((0, 0, 0), dtype=np.float32)
        prediction = self.model.inference(list(frames_rgb))
        depths = _extract_depths(prediction)
        if depths.shape[0] != len(frames_rgb):
            raise TwoDVRRuntimeError(
                f"DA3 returned {depths.shape[0]} depth maps for {len(frames_rgb)} input frames."
            )

        out: list[np.ndarray] = []
        for frame_rgb, depth in zip(frames_rgb, depths):
            if depth.shape != frame_rgb.shape[:2]:
                depth_img = Image.fromarray(depth.astype(np.float32))
                depth = np.asarray(
                    depth_img.resize((frame_rgb.shape[1], frame_rgb.shape[0]), Image.Resampling.BILINEAR),
                    dtype=np.float32,
                )
            out.append(depth.astype(np.float32, copy=False))
        return np.stack(out, axis=0)

    def predict(self, frame_rgb: np.ndarray) -> np.ndarray:
        return self.predict_batch([frame_rgb])[0]


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
    if mode in {"none", HOLE_FILL_E2FGVI}:
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
    h, w, _ = frame_rgb.shape
    near = _normalize_near(_smooth_depth(depth))
    if near.shape != (h, w):
        raise TwoDVRRuntimeError(f"Depth shape {near.shape} does not match frame shape {(h, w)}")
    max_shift = _max_disparity_pixels(w, eye_distance_mm)
    disparity = near * max_shift

    left_raw, left_holes, left_near = _forward_warp_eye_rgb(frame_rgb, near, max_shift, 1.0)
    right_raw, right_holes, right_near = _forward_warp_eye_rgb(frame_rgb, near, max_shift, -1.0)
    mode = resolve_hole_fill_mode(hole_fill_mode)
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
                           log_callback=None) -> None:
    try:
        stem = debug_output_stem(output)
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


def _predict_depths_resilient(estimator: DA3DepthEstimator, frames: list[np.ndarray], log_callback=None) -> np.ndarray:
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
        return np.concatenate([first, second], axis=0)


class TorchStereoRenderer:
    backend = "torch_cuda_forward_zfill"

    def __init__(self, src_w: int, src_h: int, pmap: ProjectionMap, eye_distance_mm: float,
                 hole_fill_mode: str = DEFAULT_HOLE_FILL_MODE):
        import torch

        self.torch = torch
        self.src_w = int(src_w)
        self.src_h = int(src_h)
        self.pmap = pmap
        self.eye_distance_mm = float(eye_distance_mm)
        self.hole_fill_mode = resolve_hole_fill_mode(hole_fill_mode)
        self.device = torch.device("cuda")

        yi = torch.arange(self.src_h, device=self.device, dtype=torch.long)
        xi = torch.arange(self.src_w, device=self.device, dtype=torch.long)
        pix_y, pix_x = torch.meshgrid(yi, xi, indexing="ij")
        self.pixel_y = pix_y.unsqueeze(0)
        self.pixel_x = pix_x.unsqueeze(0)

        grid_x = torch.from_numpy((pmap.map_x / max(1, self.src_w - 1) * 2.0 - 1.0).astype(np.float32))
        grid_y = torch.from_numpy((pmap.map_y / max(1, self.src_h - 1) * 2.0 - 1.0).astype(np.float32))
        self.proj_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).to(self.device, non_blocking=True)
        self.proj_mask = torch.from_numpy(pmap.mask.astype(np.float32)).to(self.device, non_blocking=True)[None, None]
        self.blur_kernel = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            device=self.device,
            dtype=torch.float32,
        ).view(1, 1, 3, 3) / 16.0

    def _smooth_depth(self, depth):
        import torch.nn.functional as F

        return F.conv2d(F.pad(depth, (1, 1, 1, 1), mode="replicate"), self.blur_kernel)

    def _normalize_near(self, depth):
        torch = self.torch
        near = torch.zeros_like(depth)
        valid_depth = torch.isfinite(depth) & (depth > 1e-6)
        inv_depth = torch.zeros_like(depth)
        inv_depth[valid_depth] = 1.0 / depth[valid_depth]
        flat = inv_depth.flatten(1)
        valid_flat = valid_depth.flatten(1)
        for idx in range(depth.shape[0]):
            vals = flat[idx][valid_flat[idx]]
            if vals.numel() < 2:
                continue
            lo, hi = torch.quantile(vals, torch.tensor([0.05, 0.95], device=self.device, dtype=torch.float32))
            if not bool(torch.isfinite(lo) and torch.isfinite(hi) and hi > lo):
                continue
            cur = ((inv_depth[idx] - lo) / (hi - lo)).clamp(0.0, 1.0)
            near[idx] = torch.where(valid_depth[idx], cur, torch.zeros_like(cur))
        return near

    def _forward_warp_eye(self, frame, near, max_shift: float, eye_sign: float):
        torch = self.torch
        batch, height, width, channels = frame.shape
        target_x = torch.round(
            self.pixel_x.expand(batch, -1, -1).float() + near * (max_shift * 0.5 * eye_sign)
        ).long()
        valid = (target_x >= 0) & (target_x < width)

        batch_idx = torch.arange(batch, device=self.device, dtype=torch.long).view(batch, 1, 1)
        flat_target = ((batch_idx * height + self.pixel_y) * width + target_x).reshape(-1)
        valid_flat = valid.reshape(-1)
        priority = near.reshape(-1)

        zbuf = torch.full((batch * height * width,), -1.0, device=self.device, dtype=torch.float32)
        if bool(valid_flat.any()):
            zbuf.scatter_reduce_(
                0,
                flat_target[valid_flat],
                priority[valid_flat],
                reduce="amax",
                include_self=True,
            )

        safe_target = flat_target.clamp(0, batch * height * width - 1)
        winners = valid_flat & (priority >= zbuf[safe_target] - 1e-6)
        out = torch.zeros((batch * height * width, channels), device=self.device, dtype=frame.dtype)
        out[safe_target[winners]] = frame.reshape(-1, channels)[winners]

        near_buffer = zbuf.view(batch, height, width)
        holes = near_buffer < 0.0
        return out.view(batch, height, width, channels), holes, near_buffer

    def _fill_holes_background(self, image, holes, near_buffer):
        torch = self.torch
        if not bool(holes.any()):
            return image
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

        if not bool(holes.any()):
            return image
        x = image.permute(0, 3, 1, 2).contiguous()
        mask = holes.unsqueeze(1).to(dtype=x.dtype)
        mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        mask = F.avg_pool2d(mask, kernel_size=5, stride=1, padding=2).clamp(0.0, 1.0) * 0.35
        blur = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        out = x * (1.0 - mask) + blur * mask
        return out.permute(0, 2, 3, 1).contiguous()

    def _fill_eye_holes(self, image, holes, near_buffer, direction: int):
        mode = self.hole_fill_mode
        if mode in {"none", HOLE_FILL_E2FGVI}:
            return image
        if mode == "background":
            return self._fill_holes_background(image, holes, near_buffer)
        shifted = self._shift_fill_holes(image, holes, direction)
        if mode == "soft_shift":
            return self._soft_blend_holes(shifted, holes)
        return shifted

    def render_batch(self, frames_rgb: np.ndarray, depths: np.ndarray) -> np.ndarray:
        import torch.nn.functional as F

        torch = self.torch
        frames = np.ascontiguousarray(frames_rgb)
        depth_np = np.ascontiguousarray(depths.astype(np.float32, copy=False))
        batch = int(frames.shape[0])

        with torch.inference_mode():
            frame_t = torch.from_numpy(frames).to(self.device, non_blocking=True)
            frame_t = frame_t.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
            depth_t = torch.from_numpy(depth_np).to(self.device, non_blocking=True).unsqueeze(1).float()
            near = self._normalize_near(self._smooth_depth(depth_t))

            max_shift = _max_disparity_pixels(self.src_w, self.eye_distance_mm)
            frame_bhwc = frame_t.permute(0, 2, 3, 1).contiguous()
            left, left_holes, left_near = self._forward_warp_eye(frame_bhwc, near[:, 0], max_shift, 1.0)
            right, right_holes, right_near = self._forward_warp_eye(frame_bhwc, near[:, 0], max_shift, -1.0)
            left = self._fill_eye_holes(left, left_holes, left_near, -1)
            right = self._fill_eye_holes(right, right_holes, right_near, 1)
            left = left.permute(0, 3, 1, 2).contiguous()
            right = right.permute(0, 3, 1, 2).contiguous()

            proj_grid = self.proj_grid.expand(batch, -1, -1, -1)
            left_proj = F.grid_sample(left, proj_grid, mode="bilinear", padding_mode="border", align_corners=True)
            right_proj = F.grid_sample(right, proj_grid, mode="bilinear", padding_mode="border", align_corners=True)
            left_proj = left_proj * self.proj_mask
            right_proj = right_proj * self.proj_mask

            sbs = torch.cat([left_proj, right_proj], dim=3)
            out = (sbs.clamp(0.0, 1.0).mul_(255.0).round_().to(torch.uint8))
            return out.permute(0, 2, 3, 1).contiguous().cpu().numpy()

    def stereo_raw_batch(self, frames_rgb: np.ndarray, depths: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        torch = self.torch
        frames = np.ascontiguousarray(frames_rgb)
        depth_np = np.ascontiguousarray(depths.astype(np.float32, copy=False))

        with torch.inference_mode():
            frame_t = torch.from_numpy(frames).to(self.device, non_blocking=True)
            frame_t = frame_t.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
            depth_t = torch.from_numpy(depth_np).to(self.device, non_blocking=True).unsqueeze(1).float()
            near = self._normalize_near(self._smooth_depth(depth_t))

            max_shift = _max_disparity_pixels(self.src_w, self.eye_distance_mm)
            frame_bhwc = frame_t.permute(0, 2, 3, 1).contiguous()
            left, left_holes, _ = self._forward_warp_eye(frame_bhwc, near[:, 0], max_shift, 1.0)
            right, right_holes, _ = self._forward_warp_eye(frame_bhwc, near[:, 0], max_shift, -1.0)

            left = left.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
            right = right.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
            return (
                left.contiguous().cpu().numpy(),
                right.contiguous().cpu().numpy(),
                left_holes.contiguous().cpu().numpy(),
                right_holes.contiguous().cpu().numpy(),
            )


def _make_torch_renderer(src_w: int, src_h: int, pmap: ProjectionMap, eye_distance_mm: float,
                         hole_fill_mode: str = DEFAULT_HOLE_FILL_MODE, log_callback=None):
    try:
        import torch

        if resolve_hole_fill_mode(hole_fill_mode) == "inpaint":
            return None
        if not torch.cuda.is_available():
            return None
        return TorchStereoRenderer(src_w, src_h, pmap, eye_distance_mm, hole_fill_mode)
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
    frame_size = info.width * info.height * 3
    temp_video = str(Path(output).with_suffix(".video_only.tmp.mp4"))
    depth_batch_size = resolve_depth_batch_size()
    fill_mode = resolve_hole_fill_mode(hole_fill_mode)
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
        log_callback=log_callback,
    )
    render_backend = renderer.backend if renderer is not None else "cpu_forward_zfill"
    save_eye_debug = debug_eye_enabled()
    eye_debug_saved = False

    handle = PipelineProcess()
    if process_callback:
        process_callback(handle)

    decode_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        *_ffmpeg_time_args(start_sec, duration_sec),
        "-i", input_path,
        "-an", "-sn",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-vsync", "0",
        "pipe:1",
    ]
    encode_cmd = [
        "ffmpeg", "-hide_banner", "-y", "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{out_w}x{out_h}",
        "-r", f"{info.fps:.6f}",
        "-i", "pipe:0",
        "-c:v", "hevc_nvenc",
        "-preset", "p7",
        "-cq", "18",
        "-pix_fmt", "yuv420p",
        temp_video,
    ]

    if log_callback:
        log_callback(
            f"[2dvr] source={info.width}x{info.height} fps={info.fps:.3f}, "
            f"output={out_w}x{out_h}, projection={projection}, fov={flat_fov_deg:.1f}, "
            f"yaw=0.0, pitch=0.0, eye={float(eye_distance_mm):.1f}mm, "
            f"hole_fill={fill_mode}"
        )
        log_callback(
            f"[2dvr] pipeline: decode=ffmpeg/rawvideo, DA3=cuda batch={depth_batch_size}, "
            f"render={render_backend}, encode=hevc_nvenc"
        )
        if use_e2fgvi:
            log_callback(
                f"[2dvr] E2FGVI video inpaint enabled: chunk={e2fgvi_chunk_size}. "
                "E2FGVI is CC BY-NC 4.0; use it for non-commercial evaluation unless separately licensed."
            )
        log_callback("Executing: " + " ".join(decode_cmd))
        log_callback("Executing: " + " ".join(encode_cmd))

    decode_proc = None
    encode_proc = None
    frames_done = 0
    last_log = time.perf_counter()
    started = time.perf_counter()

    def _write_sbs_frames(sbs_batch: np.ndarray) -> None:
        nonlocal frames_done, last_log
        for sbs in sbs_batch:
            if handle.cancelled:
                raise OperationCancelled()
            encode_proc.stdin.write(sbs.tobytes())
            frames_done += 1
        now = time.perf_counter()
        if log_callback and now - last_log >= 1.0:
            fps = frames_done / max(0.001, now - started)
            log_callback(f"[2dvr] frames={frames_done} | {fps:.2f} fps | render={render_backend}")
            last_log = now

    def _render_with_depths(frames_batch: list[np.ndarray], depths: np.ndarray) -> np.ndarray:
        nonlocal renderer, render_backend
        if renderer is not None:
            try:
                return renderer.render_batch(np.stack(frames_batch, axis=0), depths)
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
                        first = _render_with_depths(frames_batch[:mid], depths[:mid])
                        second = _render_with_depths(frames_batch[mid:], depths[mid:])
                        return np.concatenate([first, second], axis=0)
                if log_callback:
                    log_callback(f"[2dvr] CUDA renderer failed, falling back to CPU renderer: {exc}")
                renderer = None
                render_backend = "cpu_forward_zfill"
        return np.stack(
            [render_sbs_frame(frame, depth, projection, float(eye_distance_mm), pmap, fill_mode)
             for frame, depth in zip(frames_batch, depths)],
            axis=0,
        )

    def _append_e2fgvi_stereo_batch(frames_batch: list[np.ndarray], depths: np.ndarray) -> None:
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

        for frame, depth in zip(frames_batch, depths):
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
            raw_backend = renderer.backend if renderer is not None else "cpu_forward_zfill"
            e2fgvi_device = getattr(e2fgvi_inpainter, "_device", None)
            e2fgvi_backend = "e2fgvi_cuda" if e2fgvi_device is not None and e2fgvi_device.type == "cuda" else "e2fgvi_cpu"
            render_backend = f"{raw_backend}+{e2fgvi_backend}"
            _write_sbs_frames(sbs_batch)

    def _render_frame_batch(frames_batch: list[np.ndarray]) -> np.ndarray:
        nonlocal eye_debug_saved
        depths = _predict_depths_resilient(estimator, frames_batch, log_callback)
        if save_eye_debug and not eye_debug_saved and len(frames_batch) > 0:
            _save_eye_debug_images(
                output,
                frames_batch[0],
                depths[0],
                projection,
                float(eye_distance_mm),
                pmap,
                hole_fill_mode=("none" if use_e2fgvi else fill_mode),
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

    try:
        decode_proc = subprocess.Popen(
            decode_cmd,
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

        frame_batch: list[np.ndarray] = []
        while True:
            if handle.cancelled:
                raise OperationCancelled()
            raw = decode_proc.stdout.read(frame_size)
            if not raw:
                break
            if len(raw) != frame_size:
                raise TwoDVRRuntimeError(f"Short frame read: {len(raw)} of {frame_size} bytes")
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(info.height, info.width, 3).copy()
            frame_batch.append(frame)
            if len(frame_batch) >= depth_batch_size:
                _write_sbs_batch(frame_batch)
                frame_batch = []

        _write_sbs_batch(frame_batch)
        _flush_e2fgvi_buffer(True)

        encode_proc.stdin.close()
        encode_proc.wait()
        decode_proc.wait()
        dec_err = (decode_proc.stderr.read() if decode_proc.stderr else b"").decode("utf-8", "replace")
        enc_err = (encode_proc.stderr.read() if encode_proc.stderr else b"").decode("utf-8", "replace")
        if decode_proc.returncode not in (0, None):
            raise TwoDVRRuntimeError(f"ffmpeg decode failed ({decode_proc.returncode}): {dec_err.strip()}")
        if encode_proc.returncode not in (0, None):
            raise TwoDVRRuntimeError(f"ffmpeg encode failed ({encode_proc.returncode}): {enc_err.strip()}")
        if frames_done <= 0:
            raise TwoDVRRuntimeError("No frames were processed.")

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
            log_callback(f"[2dvr] Done: {output} ({frames_done} frames, {elapsed:.1f}s)")
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
        try:
            if os.path.exists(temp_video):
                os.remove(temp_video)
        except Exception:
            pass
