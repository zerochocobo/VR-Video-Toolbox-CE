from __future__ import annotations

import unittest
from unittest.mock import patch

from gpu_engine.probe import BackendDecision
from tool_split_combine import logic


class SplitCombineGpuAudioTests(unittest.TestCase):
    def test_combine_gpu_path_preserves_first_input_audio(self) -> None:
        seen_kwargs = {}

        def fake_run_with_fallback(gpu_fn, _ffmpeg_fn, **_kwargs):
            return gpu_fn()

        def fake_combine(_input_a, _input_b, _output, _mode, **kwargs):
            seen_kwargs.update(kwargs)
            return True

        with (
            patch("gpu_engine.probe.route", return_value=(None, BackendDecision("gpu_nv12", "ok"))),
            patch("gpu_engine.fallback.run_with_fallback", side_effect=fake_run_with_fallback),
            patch("gpu_engine.files.combine_video", side_effect=fake_combine),
        ):
            ok = logic.combine_video("left.mp4", "right.mp4", "left_right", "out.mp4")

        self.assertTrue(ok)
        self.assertTrue(seen_kwargs["keep_audio"])

    def test_combine_uses_selected_original_as_bitrate_reference(self) -> None:
        seen_kwargs = {}

        def fake_run_with_fallback(gpu_fn, _ffmpeg_fn, **_kwargs):
            return gpu_fn()

        def fake_combine(_input_a, _input_b, _output, _mode, **kwargs):
            seen_kwargs.update(kwargs)
            return True

        with (
            patch("gpu_engine.probe.route", return_value=(None, BackendDecision("gpu_nv12", "ok"))),
            patch("gpu_engine.fallback.run_with_fallback", side_effect=fake_run_with_fallback),
            patch("gpu_engine.files.combine_video", side_effect=fake_combine),
            patch.object(logic, "get_video_bitrate", return_value=42_000_000),
        ):
            ok = logic.combine_video(
                "left.mp4", "right.mp4", "left_right", "out.mp4",
                bitrate_reference_path="original.mp4",
            )

        self.assertTrue(ok)
        self.assertIsNone(seen_kwargs["cq"])
        self.assertEqual(seen_kwargs["bitrate_bps"], 42_000_000)
        self.assertEqual(seen_kwargs["max_bitrate_bps"], 84_000_000)


if __name__ == "__main__":
    unittest.main()
