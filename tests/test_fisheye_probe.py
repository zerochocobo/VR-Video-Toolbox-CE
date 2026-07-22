from __future__ import annotations

import unittest

import numpy as np

from utils import fisheye_probe
from utils.fisheye_probe import RoiVote, VoteAccumulator


def _smooth_texture(h: int, w: int, seed: int) -> np.ndarray:
    """Aperiodic smooth texture: repeatedly box-filtered white noise.

    Deliberately avoids lattice-based generators (upsampled control grids,
    sinusoids): their regular spacing already looks like a mosaic grid to the
    periodicity score.
    """
    rng = np.random.default_rng(seed)
    pad = 64
    img = rng.uniform(0.0, 255.0, size=(h + 2 * pad, w + 2 * pad))
    for k in (31, 17, 9):
        kernel = np.ones(k) / k
        img = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), 1, img)
        img = np.apply_along_axis(lambda c: np.convolve(c, kernel, mode="same"), 0, img)
    img = img[pad:pad + h, pad:pad + w]  # drop zero-padding edge ramps
    span = max(1e-6, float(img.max() - img.min()))
    return (40.0 + (img - img.min()) * (180.0 / span)).astype(np.float32)


def _apply_mosaic(img: np.ndarray, box, cell: int = 16) -> np.ndarray:
    x1, y1, x2, y2 = box
    out = img.copy()
    region = out[y1:y2, x1:x2]
    for by in range(0, region.shape[0], cell):
        for bx in range(0, region.shape[1], cell):
            block = region[by:by + cell, bx:bx + cell]
            block[:] = block.mean()
    return out


def _box_blur(img: np.ndarray, box, passes: int = 3, k: int = 15) -> np.ndarray:
    x1, y1, x2, y2 = box
    out = img.copy()
    region = out[y1:y2, x1:x2].astype(np.float64)
    kernel = np.ones(k) / k
    for _ in range(passes):
        region = np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode="same"), 1, region)
        region = np.apply_along_axis(
            lambda col: np.convolve(col, kernel, mode="same"), 0, region)
    out[y1:y2, x1:x2] = region
    return out


def _render_heq_from_fisheye(fisheye_img: np.ndarray) -> np.ndarray:
    h, w = fisheye_img.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    sx, sy = fisheye_probe.heq_to_fisheye_coords(gx, gy, w, h)
    return fisheye_probe.bilinear_sample(fisheye_img, sx, sy)


def _fisheye_box_to_heq_box(box, eye_w: int, eye_h: int):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    px = np.array([x1, cx, x2, x1, x2, x1, cx, x2], dtype=np.float64)
    py = np.array([y1, y1, y1, cy, cy, y2, y2, y2], dtype=np.float64)
    hx, hy = fisheye_probe.fisheye_to_heq_coords(px, py, eye_w, eye_h)
    return (
        int(np.floor(hx.min())), int(np.floor(hy.min())),
        int(np.ceil(hx.max())), int(np.ceil(hy.max())),
    )


EYE = 768
# Mosaic region below and left of the eye center: warp is significant there.
FISH_BOX = (int(EYE * 0.28), int(EYE * 0.60), int(EYE * 0.62), int(EYE * 0.88))


class GeometryTests(unittest.TestCase):
    def test_fisheye_roundtrip_is_identity_inside_circle(self) -> None:
        pts_x = np.array([EYE * 0.5, EYE * 0.3, EYE * 0.7, EYE * 0.5], dtype=np.float64)
        pts_y = np.array([EYE * 0.5, EYE * 0.6, EYE * 0.35, EYE * 0.8], dtype=np.float64)
        hx, hy = fisheye_probe.fisheye_to_heq_coords(pts_x, pts_y, EYE, EYE)
        bx, by = fisheye_probe.heq_to_fisheye_coords(hx, hy, EYE, EYE)
        np.testing.assert_allclose(bx, pts_x, atol=1.0)
        np.testing.assert_allclose(by, pts_y, atol=1.0)


class GridScoreTests(unittest.TestCase):
    def test_regular_mosaic_scores_high(self) -> None:
        img = _apply_mosaic(_smooth_texture(320, 320, 1), (0, 0, 320, 320))
        self.assertGreater(fisheye_probe.grid_axis_score(img), 0.30)

    def test_smooth_texture_scores_low(self) -> None:
        img = _smooth_texture(320, 320, 2)
        self.assertLess(fisheye_probe.grid_axis_score(img), 0.15)

    def test_tiny_region_scores_zero(self) -> None:
        self.assertEqual(fisheye_probe.grid_axis_score(np.zeros((32, 32))), 0.0)


class ClassifyBoxTests(unittest.TestCase):
    def test_fisheye_mastered_mosaic_prefers_fisheye(self) -> None:
        fisheye_master = _smooth_texture(EYE, EYE, 3)
        mosaicked = _apply_mosaic(fisheye_master, FISH_BOX)
        heq_frame = _render_heq_from_fisheye(mosaicked)
        heq_box = _fisheye_box_to_heq_box(FISH_BOX, EYE, EYE)

        vote = fisheye_probe.classify_box(heq_frame, heq_box, conf=0.9)

        self.assertIsNotNone(vote)
        self.assertGreater(vote.score_fish, vote.score_heq + 0.05)

    def test_direct_mosaic_prefers_source_projection(self) -> None:
        heq_frame = _apply_mosaic(_smooth_texture(EYE, EYE, 4), FISH_BOX)

        vote = fisheye_probe.classify_box(heq_frame, FISH_BOX, conf=0.9)

        self.assertIsNotNone(vote)
        self.assertGreater(vote.score_heq, vote.score_fish + 0.05)

    def test_blur_censor_yields_no_vote(self) -> None:
        heq_frame = _box_blur(_smooth_texture(EYE, EYE, 5), FISH_BOX)

        vote = fisheye_probe.classify_box(heq_frame, FISH_BOX, conf=0.9)

        self.assertIsNone(vote)

    def test_tiny_box_yields_no_vote(self) -> None:
        frame = _apply_mosaic(_smooth_texture(EYE, EYE, 6), FISH_BOX)
        self.assertIsNone(fisheye_probe.classify_box(frame, (10, 10, 40, 40), conf=0.9))

    def test_center_roi_weighs_less_than_edge_roi(self) -> None:
        center = fisheye_probe._geo_weight(EYE * 0.5, EYE * 0.5, EYE, EYE)
        edge = fisheye_probe._geo_weight(EYE * 0.08, EYE * 0.85, EYE, EYE)
        self.assertLess(center, edge)
        self.assertGreater(center, 0.0)


class VerdictTests(unittest.TestCase):
    def _votes(self, deltas, weight=0.6):
        return [RoiVote(score_heq=0.4, score_fish=0.4 + d, weight=weight) for d in deltas]

    def test_consistent_fisheye_votes_yield_fisheye(self) -> None:
        acc = VoteAccumulator(votes=self._votes([0.2, 0.25, 0.18, 0.22]))
        verdict = acc.verdict()
        self.assertEqual(verdict.mode, "fisheye")
        self.assertTrue(verdict.use_fisheye)

    def test_consistent_direct_votes_yield_direct(self) -> None:
        acc = VoteAccumulator(votes=self._votes([-0.2, -0.25, -0.18, -0.22]))
        verdict = acc.verdict()
        self.assertEqual(verdict.mode, "direct")
        self.assertFalse(verdict.use_fisheye)

    def test_too_few_votes_yield_uncertain(self) -> None:
        acc = VoteAccumulator(votes=self._votes([0.3]))
        verdict = acc.verdict()
        self.assertEqual(verdict.mode, "uncertain")
        self.assertFalse(verdict.use_fisheye)

    def test_conflicting_votes_yield_uncertain(self) -> None:
        acc = VoteAccumulator(votes=self._votes([0.2, -0.2, 0.15, -0.18]))
        verdict = acc.verdict()
        self.assertEqual(verdict.mode, "uncertain")
        self.assertFalse(verdict.use_fisheye)

    def test_end_to_end_synthetic_fisheye_video_verdict(self) -> None:
        acc = VoteAccumulator()
        for seed in range(7, 12):
            fisheye_master = _smooth_texture(EYE, EYE, seed)
            heq_frame = _render_heq_from_fisheye(_apply_mosaic(fisheye_master, FISH_BOX))
            heq_box = _fisheye_box_to_heq_box(FISH_BOX, EYE, EYE)
            acc.add(fisheye_probe.classify_box(heq_frame, heq_box, conf=0.8))
        verdict = acc.verdict()
        self.assertEqual(verdict.mode, "fisheye")

    def test_end_to_end_synthetic_direct_video_verdict(self) -> None:
        acc = VoteAccumulator()
        for seed in range(12, 17):
            heq_frame = _apply_mosaic(_smooth_texture(EYE, EYE, seed), FISH_BOX)
            acc.add(fisheye_probe.classify_box(heq_frame, FISH_BOX, conf=0.8))
        verdict = acc.verdict()
        self.assertEqual(verdict.mode, "direct")


if __name__ == "__main__":
    unittest.main()
