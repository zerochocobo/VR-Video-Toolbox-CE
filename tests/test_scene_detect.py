import unittest

import numpy as np

from utils import scene_detect


def _solid(color, size=64):
    """Solid-colour RGB frame."""
    f = np.zeros((size, size, 3), dtype=np.uint8)
    f[:, :] = color
    return f


def _noisy(color, seed, size=64, amp=10):
    """Solid colour plus small per-pixel noise to mimic in-scene motion."""
    rng = np.random.default_rng(seed)
    base = np.zeros((size, size, 3), dtype=np.int16)
    base[:, :] = color
    base += rng.integers(-amp, amp + 1, size=base.shape, dtype=np.int16)
    return np.clip(base, 0, 255).astype(np.uint8)


class HistogramBasicsTests(unittest.TestCase):
    def test_identical_frames_zero_distance(self):
        h = scene_detect.compute_histogram(_solid((120, 40, 200)))
        self.assertAlmostEqual(scene_detect.histogram_distance(h, h), 0.0, places=6)

    def test_histogram_normalized(self):
        h = scene_detect.compute_histogram(_solid((10, 20, 30)))
        self.assertAlmostEqual(float(h.sum()), 1.0, places=6)

    def test_distinct_colours_large_distance(self):
        a = scene_detect.compute_histogram(_solid((10, 10, 10)))
        b = scene_detect.compute_histogram(_solid((240, 240, 240)))
        self.assertGreater(scene_detect.histogram_distance(a, b), 0.9)

    def test_grayscale_supported(self):
        g = np.full((32, 32), 100, dtype=np.uint8)
        h = scene_detect.compute_histogram(g)
        self.assertAlmostEqual(float(h.sum()), 1.0, places=6)


class SceneCutDetectionTests(unittest.TestCase):
    def _run(self, frames, dt=0.5, **kw):
        times = [i * dt for i in range(len(frames))]
        sigs = [scene_detect.compute_histogram(f) for f in frames]
        return scene_detect.detect_cuts(times, sigs, **kw)

    def test_hard_cut_detected(self):
        frames = [_solid((20, 20, 20))] * 6 + [_solid((230, 200, 40))] * 6
        cuts = self._run(frames)
        self.assertEqual(cuts, [3.0])  # 7th frame at t=3.0 starts new shot

    def test_in_scene_motion_not_cut(self):
        # One shot, only per-pixel noise (camera/body motion) -> no cut.
        frames = [_noisy((120, 80, 60), seed=i) for i in range(12)]
        cuts = self._run(frames)
        self.assertEqual(cuts, [])

    def test_occlusion_does_not_cut(self):
        # Same shot; middle frames have a dark blob (mosaic occluded by a hand).
        scene = _noisy((150, 120, 90), seed=1)
        occluded = scene.copy()
        occluded[20:44, 20:44] = (15, 15, 15)
        frames = [scene, _noisy((150, 120, 90), 2), occluded, occluded,
                  _noisy((150, 120, 90), 3), _noisy((150, 120, 90), 4)]
        cuts = self._run(frames)
        self.assertEqual(cuts, [])

    def test_multiple_cuts(self):
        frames = (
            [_solid((20, 20, 20))] * 5
            + [_solid((220, 30, 30))] * 5
            + [_solid((30, 30, 210))] * 5
        )
        cuts = self._run(frames)
        self.assertEqual(cuts, [2.5, 5.0])

    def test_min_scene_len_hysteresis(self):
        # A real cut then an immediate second jump within min_scene_len is held.
        frames = [_solid((20, 20, 20))] * 3 + [_solid((230, 30, 30))] \
            + [_solid((30, 230, 30))] + [_solid((30, 230, 30))] * 4
        cuts = self._run(frames, dt=0.5, min_scene_len_s=1.5)
        # Only the first cut survives; the one 0.5s later is suppressed.
        self.assertEqual(cuts, [1.5])

    def test_first_frame_never_cut(self):
        det = scene_detect.SceneCutDetector()
        h = scene_detect.compute_histogram(_solid((100, 100, 100)))
        self.assertFalse(det.update(0.0, h))


if __name__ == "__main__":
    unittest.main()
