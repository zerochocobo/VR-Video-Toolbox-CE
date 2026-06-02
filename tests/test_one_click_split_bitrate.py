from __future__ import annotations

import unittest
from unittest.mock import patch

from gpu_engine.probe import BackendDecision, VideoMetadata
from one_click import logic


class OneClickSplitBitrateTests(unittest.TestCase):
    def test_single_eye_split_vbr_uses_75_percent_target_and_source_max(self) -> None:
        self.assertEqual(logic._single_eye_split_vbr_bps(20_000_000), (15_000_000, 20_000_000))

    def test_area_scaled_bitrate_uses_rect_area_and_expansion(self) -> None:
        self.assertEqual(
            logic._area_scaled_bitrate_bps(
                20_000_000,
                8192,
                4096,
                512,
                512,
                2.0,
            ),
            312_500,
        )

    def test_dual_fisheye_gpu_split_uses_single_eye_vbr_caps(self) -> None:
        meta = VideoMetadata(path="in.mp4", bitrate_bps=20_000_000)
        decision = BackendDecision("gpu_nv12", "ok")

        with (
            patch("gpu_engine.probe.route", return_value=(meta, decision)),
            patch("gpu_engine.fallback.run_with_fallback", side_effect=lambda gpu_fn, ffmpeg_fn, **kwargs: gpu_fn()),
            patch("gpu_engine.files.split_video") as split_video,
        ):
            logic.split_video_dual_fisheye(
                "in.mp4",
                "left.mp4",
                "right.mp4",
                start_time="00:01",
                end_time="00:02",
                keep_audio=False,
            )

        split_video.assert_called_once()
        kwargs = split_video.call_args.kwargs
        self.assertEqual(kwargs["bitrate_bps"], 15_000_000)
        self.assertEqual(kwargs["max_bitrate_bps"], 20_000_000)
        self.assertIsNone(kwargs["cq"])
        self.assertEqual(kwargs["start_sec"], 1.0)
        self.assertEqual(kwargs["end_sec"], 2.0)
        self.assertFalse(kwargs["keep_audio"])

    def test_dual_fisheye_ffmpeg_fallback_uses_single_eye_vbr_caps(self) -> None:
        meta = VideoMetadata(path="in.mp4", bitrate_bps=20_000_000)
        decision = BackendDecision("ffmpeg_fallback", "unsupported")

        with (
            patch("gpu_engine.probe.route", return_value=(meta, decision)),
            patch("gpu_engine.fallback.run_with_fallback", side_effect=lambda gpu_fn, ffmpeg_fn, **kwargs: ffmpeg_fn()),
            patch("one_click.logic.get_video_info", return_value={"codec": "hevc"}),
            patch("one_click.logic.run_process") as run_process,
        ):
            logic.split_video_dual_fisheye("in.mp4", "left.mp4", "right.mp4")

        cmd = run_process.call_args.args[0]
        self.assertEqual(cmd.count("-b:v"), 2)
        self.assertEqual(cmd.count("15000k"), 2)
        self.assertEqual(cmd.count("-maxrate:v"), 2)
        self.assertEqual(cmd.count("20000k"), 2)
        self.assertNotIn("-cq", cmd)


if __name__ == "__main__":
    unittest.main()
