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


if __name__ == "__main__":
    unittest.main()
