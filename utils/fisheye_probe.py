"""Auto-detect whether mosaic removal should convert VR frames to fisheye.

The question "convert to fisheye first?" really asks: *in which projection was
the mosaic applied?*  Studios that mosaic the fisheye master and then convert
to half-equirect for distribution leave a warped mosaic grid in the
distributed frames (curved cell edges, drifting cell pitch, worse away from
the view center).  Studios that mosaic the distributed hequirect frame leave
an axis-aligned square grid.  Whichever projection renders the mosaic grid
more *regular* is the projection the mosaic was applied in.

Measurement: for a detected mosaic ROI we compute a "grid axis score" in both
projections — sum |dx| gradients into a column profile and |dy| into a row
profile (grid lines produce equally spaced spikes), detrend, and take the
normalized autocorrelation peak over plausible cell pitches.  Regular
axis-aligned grids score high; warped grids score low.  Per-ROI votes are
weighted by detection confidence, ROI size, and the expected warp magnitude
at the ROI position (near the view center both projections agree, so those
votes carry little information — conveniently, misclassifying them also
costs nothing).

Geometry follows gpu_engine/v360_lut.py (which matches ffmpeg vf_v360), in
pure numpy so it runs without CUDA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

_PI = math.pi
_PI_2 = math.pi / 2.0


def _cfg_float(key: str, default: float) -> float:
    try:
        from utils import app_config
        return float(app_config.get(key, default) or default)
    except Exception:
        return float(default)


def _cfg_int(key: str, default: int) -> int:
    try:
        from utils import app_config
        return int(app_config.get(key, default) or default)
    except Exception:
        return int(default)


# --- Projection math (numpy port of v360_lut formulas). ---

def _rescale(idx, size):
    return (2.0 * idx + 1.0) / size - 1.0


def _scale(x, size):
    return (0.5 * x + 0.5) * (size - 1.0)


def heq_to_fisheye_coords(x, y, eye_w: int, eye_h: int, fov: float = 180.0):
    """Map hequirect pixel coords -> fisheye pixel coords (same eye size)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    phi = _rescale(x, eye_w) * _PI_2
    theta = _rescale(y, eye_h) * _PI_2
    ct = np.cos(theta)
    vx = ct * np.sin(phi)
    vy = np.sin(theta)
    vz = ct * np.cos(phi)
    hh = np.hypot(vx, vy)
    lh = np.where(hh > 0.0, hh, 1.0)
    phi2 = np.arctan2(hh, vz) / _PI
    fr = fov / 180.0
    uf = vx / lh * phi2 / fr
    vf = vy / lh * phi2 / fr
    return _scale(uf * 2.0, eye_w), _scale(vf * 2.0, eye_h)


def fisheye_to_heq_coords(x, y, eye_w: int, eye_h: int, fov: float = 180.0):
    """Map fisheye pixel coords -> hequirect pixel coords (same eye size)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    fr = fov / 180.0
    uf = fr * _rescale(x, eye_w)
    vf = fr * _rescale(y, eye_h)
    phi = np.arctan2(vf, uf)
    theta = _PI_2 * (1.0 - np.hypot(uf, vf))
    ct = np.cos(theta)
    vx = ct * np.cos(phi)
    vy = ct * np.sin(phi)
    vz = np.sin(theta)
    phi2 = np.arctan2(vx, vz) / _PI_2
    theta2 = np.arcsin(np.clip(vy, -1.0, 1.0)) / _PI_2
    return _scale(phi2, eye_w), _scale(theta2, eye_h)


def bilinear_sample(image: np.ndarray, sx, sy) -> np.ndarray:
    """Sample a 2D array at float coords with bilinear filtering, edge-clamped."""
    h, w = image.shape[:2]
    sx = np.clip(np.asarray(sx, dtype=np.float64), 0.0, w - 1.0)
    sy = np.clip(np.asarray(sy, dtype=np.float64), 0.0, h - 1.0)
    x0 = np.floor(sx).astype(np.int64)
    y0 = np.floor(sy).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = sx - x0
    fy = sy - y0
    img = image.astype(np.float32)
    top = img[y0, x0] * (1.0 - fx) + img[y0, x1] * fx
    bottom = img[y1, x0] * (1.0 - fx) + img[y1, x1] * fx
    return (top * (1.0 - fy) + bottom * fy).astype(np.float32)


def render_fisheye_patch(eye_gray: np.ndarray, fx0: int, fy0: int,
                         fw: int, fh: int, fov: float = 180.0) -> np.ndarray:
    """Render a sub-window of the fisheye view of a hequirect eye frame."""
    eye_h, eye_w = eye_gray.shape[:2]
    xs = np.arange(fx0, fx0 + fw, dtype=np.float64)
    ys = np.arange(fy0, fy0 + fh, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    sx, sy = fisheye_to_heq_coords(grid_x, grid_y, eye_w, eye_h, fov)
    return bilinear_sample(eye_gray, sx, sy)


# --- Grid regularity score. ---

def _profile_periodicity(profile: np.ndarray, min_lag: int = 4, max_lag: int = 64) -> float:
    p = np.asarray(profile, dtype=np.float64)
    n = p.size
    if n < min_lag * 6:
        return 0.0
    max_lag = int(min(max_lag, n // 3))
    if max_lag <= min_lag:
        return 0.0
    # High-pass with a window narrower than one cell pitch: smooth profiles
    # (skin, fabric, defocus) are wiped out, while the equally spaced gradient
    # spikes of a mosaic grid survive.  Without this, the autocorrelation of
    # any smooth signal is close to 1 at short lags and the score would
    # measure smoothness rather than periodicity.
    window = 2 * min_lag + 1
    trend = np.convolve(p, np.ones(window) / window, mode="same")
    p = p - trend
    denom = float(np.dot(p, p))
    if denom < 1e-9:
        return 0.0
    ac = np.correlate(p, p, mode="full")[n - 1:] / denom
    return float(np.clip(ac[min_lag:max_lag + 1].max(), 0.0, 1.0))


def grid_axis_score(gray: np.ndarray) -> float:
    """How much the region looks like an axis-aligned, evenly pitched grid."""
    g = np.asarray(gray, dtype=np.float32)
    if g.ndim != 2 or min(g.shape) < 48:
        return 0.0
    col_profile = np.abs(np.diff(g, axis=1)).sum(axis=0)
    row_profile = np.abs(np.diff(g, axis=0)).sum(axis=1)
    return 0.5 * (_profile_periodicity(col_profile) + _profile_periodicity(row_profile))


# --- Per-ROI classification and vote aggregation. ---

@dataclass
class RoiVote:
    score_heq: float
    score_fish: float
    weight: float


def _geo_weight(cx: float, cy: float, eye_w: int, eye_h: int) -> float:
    """Expected warp magnitude at the ROI center: 0.15 at the view axis, 1 at 90 deg."""
    phi = float(_rescale(np.float64(cx), eye_w)) * _PI_2
    theta = float(_rescale(np.float64(cy), eye_h)) * _PI_2
    vz = math.cos(theta) * math.cos(phi)
    alpha = math.acos(max(-1.0, min(1.0, vz)))
    return 0.15 + 0.85 * max(0.0, min(1.0, alpha / _PI_2))


def classify_box(eye_gray: np.ndarray, box_xyxy, conf: float,
                 fov: float = 180.0, min_grid_score: float | None = None) -> RoiVote | None:
    """Score one mosaic ROI (eye-local coords) in both projections."""
    eye_h, eye_w = eye_gray.shape[:2]
    x1 = max(0, int(math.floor(box_xyxy[0])))
    y1 = max(0, int(math.floor(box_xyxy[1])))
    x2 = min(eye_w, int(math.ceil(box_xyxy[2])))
    y2 = min(eye_h, int(math.ceil(box_xyxy[3])))
    if x2 - x1 < 48 or y2 - y1 < 48:
        return None
    if min_grid_score is None:
        min_grid_score = _cfg_float("fisheye_auto_min_grid_score", 0.30)

    score_heq = grid_axis_score(eye_gray[y1:y2, x1:x2])

    # Fisheye-space bounding box of the ROI via its corner and edge midpoints.
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    px = np.array([x1, cx, x2, x1, x2, x1, cx, x2], dtype=np.float64)
    py = np.array([y1, y1, y1, cy, cy, y2, y2, y2], dtype=np.float64)
    fx, fy = heq_to_fisheye_coords(px, py, eye_w, eye_h, fov)
    margin_x = 0.05 * (fx.max() - fx.min())
    margin_y = 0.05 * (fy.max() - fy.min())
    fx1 = max(0, int(math.floor(fx.min() - margin_x)))
    fy1 = max(0, int(math.floor(fy.min() - margin_y)))
    fx2 = min(eye_w, int(math.ceil(fx.max() + margin_x)))
    fy2 = min(eye_h, int(math.ceil(fy.max() + margin_y)))
    if fx2 - fx1 < 48 or fy2 - fy1 < 48:
        return None
    patch = render_fisheye_patch(eye_gray, fx1, fy1, fx2 - fx1, fy2 - fy1, fov)
    score_fish = grid_axis_score(patch)

    if max(score_heq, score_fish) < min_grid_score:
        return None  # no visible grid: blur-type censor, motion blur, or too soft
    size_weight = max(0.25, min(2.0, min(x2 - x1, y2 - y1) / 64.0))
    weight = float(conf) * _geo_weight(cx, cy, eye_w, eye_h) * size_weight
    return RoiVote(score_heq=score_heq, score_fish=score_fish, weight=weight)


@dataclass
class FisheyeVerdict:
    mode: str            # "fisheye" | "direct" | "uncertain"
    use_fisheye: bool
    margin: float
    votes: int
    total_weight: float
    reason: str


@dataclass
class VoteAccumulator:
    votes: list = field(default_factory=list)

    def add(self, vote: RoiVote | None) -> None:
        if vote is not None:
            self.votes.append(vote)

    def verdict(self) -> FisheyeVerdict:
        min_votes = _cfg_int("fisheye_auto_min_votes", 3)
        min_weight = _cfg_float("fisheye_auto_min_weight", 0.8)
        margin_threshold = _cfg_float("fisheye_auto_margin", 0.06)
        total_weight = sum(v.weight for v in self.votes)
        count = len(self.votes)
        if count < min_votes or total_weight < min_weight:
            return FisheyeVerdict(
                mode="uncertain", use_fisheye=False, margin=0.0,
                votes=count, total_weight=total_weight,
                reason=f"insufficient evidence (votes={count}, weight={total_weight:.2f})",
            )
        margin = sum(v.weight * (v.score_fish - v.score_heq) for v in self.votes) / total_weight
        if margin > margin_threshold:
            mode, use = "fisheye", True
            reason = f"mosaic grid is more regular in fisheye space (margin={margin:+.3f})"
        elif margin < -margin_threshold:
            mode, use = "direct", False
            reason = f"mosaic grid is more regular in the source projection (margin={margin:+.3f})"
        else:
            mode, use = "uncertain", False
            reason = f"projections score too close (margin={margin:+.3f})"
        return FisheyeVerdict(
            mode=mode, use_fisheye=use, margin=margin,
            votes=count, total_weight=total_weight, reason=reason,
        )


# --- Whole-video probe. ---

def probe_video(input_file, *, start_s: float | None = None, end_s: float | None = None,
                sample_count: int | None = None, log_callback=None,
                cancel_token=None) -> FisheyeVerdict:
    """Sample frames, detect mosaic ROIs, and vote on the mosaic projection.

    Assumes a side-by-side source (the one_click pipelines' input); boxes are
    classified per eye.  Raises on unreadable input; detector errors bubble up
    so the caller can fall back to a safe default.
    """
    import cv2
    import torch

    from gpu_engine import probe as gpu_probe
    from gpu_engine.fallback import OperationCancelled
    from utils import mosaic_prescan

    meta = gpu_probe.probe_video(input_file)
    duration = float(meta.duration or 0.0)
    lo = max(0.0, float(start_s or 0.0))
    hi = float(end_s) if end_s else duration
    if hi <= lo:
        lo, hi = 0.0, duration
    samples = int(sample_count or _cfg_int("fisheye_auto_samples", 12))
    min_conf = _cfg_float("fisheye_auto_min_conf", 0.35)
    eye_w = max(2, int(meta.width) // 2)

    detector = mosaic_prescan._get_detector(log_callback, frame_w=meta.width, frame_h=meta.height)
    accumulator = VoteAccumulator()
    cap = cv2.VideoCapture(str(input_file))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video for fisheye probe: {input_file}")
    try:
        for i in range(samples):
            if cancel_token is not None and getattr(cancel_token, "cancelled", False):
                raise OperationCancelled("cancelled by user")
            t = lo + (hi - lo) * (i + 0.5) / samples
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            results = mosaic_prescan._run_detector_batch(
                detector, [torch.from_numpy(np.ascontiguousarray(frame))],
                log_callback=log_callback, boxes_only=True,
            )
            boxes = mosaic_prescan._extract_boxes(results[0], min_conf=min_conf)
            if not boxes:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for box in boxes:
                x1, y1, x2, y2, conf = box
                eye_index = 1 if (x1 + x2) * 0.5 >= eye_w else 0
                offset = eye_index * eye_w
                eye_gray = gray[:, offset:offset + eye_w]
                vote = classify_box(eye_gray, (x1 - offset, y1, x2 - offset, y2), conf)
                if vote is not None and log_callback:
                    log_callback(
                        f"[fisheye-auto] t={t:.1f}s eye={'R' if eye_index else 'L'} "
                        f"roi=({int(x1 - offset)},{int(y1)},{int(x2 - offset)},{int(y2)}) "
                        f"score_src={vote.score_heq:.3f} score_fisheye={vote.score_fish:.3f} "
                        f"weight={vote.weight:.2f}"
                    )
                accumulator.add(vote)
    finally:
        cap.release()

    verdict = accumulator.verdict()
    if log_callback:
        log_callback(
            f"[fisheye-auto] verdict={verdict.mode} -> "
            f"{'convert to fisheye' if verdict.use_fisheye else 'no conversion'}; "
            f"{verdict.reason}; votes={verdict.votes}, weight={verdict.total_weight:.2f}"
        )
    return verdict
