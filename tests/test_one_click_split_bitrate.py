from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
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

    def test_pipeline_bitrate_splits_intermediate_and_final_multipliers(self) -> None:
        with patch("utils.app_config.get", side_effect=lambda key, default=None: {
            "gpu_bitrate_multiplier": 2.0,
            "gpu_bitrate_final_multiplier": 1.0,
        }.get(key, default)):
            intermediate = logic._resolve_pipeline_bitrate(
                "intermediate",
                512,
                512,
                30.0,
                40_000_000,
                False,
                source_w=8192,
                source_h=4096,
            )
            final = logic._resolve_pipeline_bitrate("final", 8192, 4096, 30.0, 80_000_000, False)
            # Intermediate stage is decoupled from keep_original: even with the
            # user toggle on, intermediate keeps a 2x headroom for downstream
            # re-encode quality.
            kept_intermediate = logic._resolve_pipeline_bitrate(
                "intermediate",
                512,
                512,
                30.0,
                20_000_000,
                True,
                source_w=8192,
                source_h=4096,
            )
            kept_final = logic._resolve_pipeline_bitrate(
                "final",
                8192,
                4096,
                30.0,
                33_000_000,
                True,
            )

        self.assertEqual(intermediate, int(40_000_000 * (512 * 512) / (8192 * 4096) * 2.0))
        self.assertEqual(final, 80_000_000)
        self.assertEqual(kept_intermediate, int(20_000_000 * (512 * 512) / (8192 * 4096) * 2.0))
        self.assertEqual(kept_final, 33_000_000)

    def test_pipeline_final_bitrate_does_not_expand_typical_8k_source(self) -> None:
        logs: list[str] = []

        with patch("utils.app_config.get", side_effect=lambda key, default=None: {
            "gpu_bitrate_final_multiplier": 1.0,
        }.get(key, default)):
            target = logic._resolve_pipeline_bitrate(
                "final",
                8192,
                4096,
                60.0,
                33_000_000,
                False,
                log_callback=logs.append,
            )

        self.assertEqual(target, 33_000_000)
        self.assertFalse(any("baseline applied" in line for line in logs))

    def test_final_bitrate_summary_warns_on_large_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "out.mp4"
            output.write_bytes(b"0" * 2_000_000)
            logs: list[str] = []

            with patch(
                "gpu_engine.probe.probe_video",
                return_value=VideoMetadata(path=str(output), duration=10.0),
            ):
                logic._log_final_bitrate_summary(output, 1_000_000, logs.append)

        self.assertIn("[bitrate] final mp4: 1600 kbps avg (source 1000 kbps, ratio 1.600x)", logs)
        self.assertTrue(any("WARNING final mp4 ratio 1.600x exceeds 1.20x" in line for line in logs))

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
