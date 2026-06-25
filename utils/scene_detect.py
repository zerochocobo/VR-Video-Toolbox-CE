"""Shot-boundary (scene cut) detection for pre-extract segmentation.

Why histogram difference and not pixel/edge methods
----------------------------------------------------
The pre-extract scan samples frames sparsely (~0.5s stride) from motion-heavy
VR footage (camera shake + body motion). Under those conditions:

* SAD / per-pixel diff is far too sensitive to motion -> false cuts everywhere.
* Edge-change-ratio is expensive and, per the shot-boundary literature, does
  not beat histogram methods; fisheye distortion makes its edge maps noisy.
* HSV-mean (PySceneDetect ContentDetector) is tuned for *consecutive* frames;
  across a 0.5s gap, in-scene motion alone can exceed its absolute threshold.

A *global colour histogram* is spatially invariant: panning, body movement and
local occlusion barely change the whole-frame colour distribution, while a real
shot cut changes it sharply. This is PySceneDetect's HistogramDetector idea,
hardened with AdaptiveDetector's rolling baseline so a single absolute threshold
need not be tuned per video. Crucially it looks at the *whole frame*, never at
the mosaic boxes, so detection survives occlusion and missed detections.

The detector is deliberately dependency-light (NumPy only) and accepts frames as
plain arrays, so it can run on a CuPy->host thumbnail during the GPU scan or on
CPU frames during the ffmpeg fallback scan.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

__all__ = [
    "compute_histogram",
    "histogram_distance",
    "SceneCutDetector",
    "detect_cuts",
]


def compute_histogram(frame: np.ndarray, bins: int = 8) -> np.ndarray:
    """Return a normalized colour histogram (sums to 1) for one frame.

    ``frame`` is an HxWxC uint8 array (channel order is irrelevant for cut
    detection) or an HxW grayscale array. Colour frames use a joint
    ``bins**3`` histogram; grayscale uses ``bins**3`` luma bins for a
    comparable resolution. The signature is exposure/size invariant because it
    is normalized by the pixel count.
    """
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        c = arr[:, :, :3].reshape(-1, 3).astype(np.float32)
        idx = np.minimum((c * bins / 256.0).astype(np.int64), bins - 1)
        flat = (idx[:, 0] * bins + idx[:, 1]) * bins + idx[:, 2]
        hist = np.bincount(flat, minlength=bins ** 3).astype(np.float64)
    else:
        g = arr.reshape(-1).astype(np.float32)
        nb = bins ** 3
        idx = np.minimum((g * nb / 256.0).astype(np.int64), nb - 1)
        hist = np.bincount(idx, minlength=nb).astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def histogram_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Bhattacharyya distance in [0, 1] between two normalized histograms.

    0 means identical distributions, 1 means no overlap. Symmetric and bounded,
    which makes thresholding and the rolling baseline well behaved.
    """
    bc = float(np.sum(np.sqrt(a * b)))
    bc = min(1.0, max(0.0, bc))
    return float(np.sqrt(max(0.0, 1.0 - bc)))


@dataclass
class _Params:
    min_scene_len_s: float = 1.5
    floor: float = 0.30
    k: float = 3.0
    window: int = 8
    min_baseline: float = 0.02


class SceneCutDetector:
    """Streaming shot-cut detector fed one sampled frame at a time.

    Feed monotonically increasing timestamps. ``update`` returns True when the
    given frame starts a new shot. A cut requires both an absolute jump
    (``> floor``) and a jump large relative to recent inter-frame motion
    (``> k * rolling_median``); ``min_scene_len_s`` enforces hysteresis so a
    single noisy frame cannot fragment a scene.
    """

    def __init__(self, *, min_scene_len_s: float = 1.5, floor: float = 0.30,
                 k: float = 3.0, window: int = 8, min_baseline: float = 0.02) -> None:
        self.p = _Params(
            min_scene_len_s=float(min_scene_len_s),
            floor=float(floor),
            k=float(k),
            window=max(1, int(window)),
            min_baseline=float(min_baseline),
        )
        self._prev_sig: np.ndarray | None = None
        self._recent: deque[float] = deque(maxlen=self.p.window)
        self._last_cut_t: float | None = None

    def update(self, t: float, signature: np.ndarray) -> bool:
        t = float(t)
        prev = self._prev_sig
        self._prev_sig = signature
        if prev is None:
            self._last_cut_t = t
            return False

        diff = histogram_distance(prev, signature)
        baseline = float(np.median(self._recent)) if self._recent else 0.0
        baseline = max(baseline, self.p.min_baseline)
        # Record motion baseline *before* deciding, so a real cut (excluded by
        # the gate below) still informs the next window via its own magnitude.
        self._recent.append(diff)

        is_cut = diff > self.p.floor and diff > self.p.k * baseline
        if is_cut and self._last_cut_t is not None:
            if t - self._last_cut_t < self.p.min_scene_len_s:
                is_cut = False
        if is_cut:
            self._last_cut_t = t
        return is_cut


def detect_cuts(times, signatures, *, min_scene_len_s: float = 1.5,
                floor: float = 0.30, k: float = 3.0, window: int = 8,
                min_baseline: float = 0.02) -> list[float]:
    """Batch helper: return the timestamps that begin a new shot."""
    det = SceneCutDetector(
        min_scene_len_s=min_scene_len_s, floor=floor, k=k,
        window=window, min_baseline=min_baseline,
    )
    cuts: list[float] = []
    for t, sig in zip(times, signatures):
        if det.update(float(t), sig):
            cuts.append(float(t))
    return cuts
