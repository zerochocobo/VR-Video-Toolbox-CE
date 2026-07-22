from __future__ import annotations

import unittest

try:
    import cupy  # noqa: F401
    import torch

    _HAS_GPU_STACK = bool(torch.cuda.is_available())
except Exception:
    _HAS_GPU_STACK = False


@unittest.skipUnless(_HAS_GPU_STACK, "torch CUDA + cupy not available")
class FisheyeDeltaTests(unittest.TestCase):
    EYE = 128  # square eye, matches VR180 per-eye aspect

    @classmethod
    def setUpClass(cls):
        import torch
        from gpu_engine.native_mosaic import fisheye_delta

        cls.torch = torch
        cls.delta = fisheye_delta
        cls.device = torch.device("cuda")

    def _rand(self, h, w, seed):
        g = self.torch.Generator(device="cpu").manual_seed(seed)
        return (
            self.torch.randint(0, 256, (h, w, 3), generator=g, dtype=self.torch.uint8)
            .to(self.device)
        )

    def _diff_mask(self, out, base):
        return (out.to(self.torch.int16) - base.to(self.torch.int16)).abs().amax(dim=-1) > 0

    def test_untouched_frame_returns_source_projection_object(self):
        eye = self.EYE
        heq = self._rand(eye, eye * 2, 1)
        fish = self._rand(eye, eye * 2, 2)
        out = self.delta.apply_delta_frame(
            self.torch, self.device, heq, fish, fish.clone(), sbs=True
        )
        self.assertIs(out, heq)  # equality shortcut: no remap at all

    def test_untouched_eye_stays_bit_exact_in_sbs(self):
        eye = self.EYE
        heq = self._rand(eye, eye * 2, 3)
        fish = self._rand(eye, eye * 2, 4)
        restored = fish.clone()
        # Touch only the right eye, near its optical center.
        cy, cx = eye // 2, eye + eye // 2
        restored[cy - 4:cy + 4, cx - 4:cx + 4, :] = 255
        out = self.delta.apply_delta_frame(
            self.torch, self.device, heq, fish, restored, sbs=True
        )
        self.assertTrue(self.torch.equal(out[:, :eye], heq[:, :eye]))
        self.assertFalse(self.torch.equal(out[:, eye:], heq[:, eye:]))

    def test_constant_delta_propagates_as_constant(self):
        eye = self.EYE
        heq = self.torch.full((eye, eye, 3), 100, dtype=self.torch.uint8, device=self.device)
        fish = self._rand(eye, eye, 5)
        restored = (fish.to(self.torch.int16) + 10).clamp(0, 255).to(self.torch.uint8)
        # Avoid clipped pixels so the delta really is +10 everywhere.
        fish = fish.clamp(0, 245)
        restored = (fish.to(self.torch.int16) + 10).to(self.torch.uint8)
        out = self.delta.apply_delta_frame(
            self.torch, self.device, heq, fish, restored, sbs=False
        )
        diff = out.to(self.torch.int16) - heq.to(self.torch.int16)
        # Bilinear sampling of a constant field is the same constant.
        self.assertEqual(int(diff.min()), 10)
        self.assertEqual(int(diff.max()), 10)

    def test_center_patch_lands_at_projection_center(self):
        eye = self.EYE
        heq = self._rand(eye, eye, 6)
        fish = self._rand(eye, eye, 7)
        restored = fish.clone()
        c = eye // 2
        restored[c - 3:c + 3, c - 3:c + 3, :] = 255
        out = self.delta.apply_delta_frame(
            self.torch, self.device, heq, fish, restored, sbs=False
        )
        mask = self._diff_mask(out, heq)
        self.assertGreater(int(mask.sum()), 0)
        ys, xs = self.torch.nonzero(mask, as_tuple=True)
        cy = float(ys.float().mean())
        cx = float(xs.float().mean())
        self.assertAlmostEqual(cy, eye / 2, delta=eye * 0.08)
        self.assertAlmostEqual(cx, eye / 2, delta=eye * 0.08)
        # Corners (outside the fisheye circle) must stay untouched.
        self.assertFalse(bool(mask[:8, :8].any()))
        self.assertFalse(bool(mask[-8:, -8:].any()))

    def test_left_of_center_patch_stays_in_left_half(self):
        eye = self.EYE
        heq = self._rand(eye, eye, 8)
        fish = self._rand(eye, eye, 9)
        restored = fish.clone()
        # Fisheye u = -0.25 -> hequirect longitude -45 deg -> x = 0.25 * eye.
        cy, cx = eye // 2, eye // 4
        restored[cy - 3:cy + 3, cx - 3:cx + 3, :] = 255
        out = self.delta.apply_delta_frame(
            self.torch, self.device, heq, fish, restored, sbs=False
        )
        mask = self._diff_mask(out, heq)
        self.assertGreater(int(mask.sum()), 0)
        _ys, xs = self.torch.nonzero(mask, as_tuple=True)
        self.assertLess(float(xs.float().mean()), eye / 2)

    def test_shape_mismatch_raises(self):
        eye = self.EYE
        heq = self._rand(eye, eye, 10)
        fish = self._rand(eye, eye, 11)
        restored = self._rand(eye, eye // 2, 12)
        with self.assertRaises(RuntimeError):
            self.delta.apply_delta_frame(
                self.torch, self.device, heq, fish, restored, sbs=False
            )

    def test_next_reference_translates_stop_iteration(self):
        with self.assertRaises(RuntimeError):
            self.delta.next_reference(iter(()))

    def test_grid_cache_reuse(self):
        self.delta.clear_cache()
        g1 = self.delta.inverse_grid(self.torch, self.device, 64, 64)
        g2 = self.delta.inverse_grid(self.torch, self.device, 64, 64)
        self.assertIs(g1, g2)
        self.delta.clear_cache()


if __name__ == "__main__":
    unittest.main()
