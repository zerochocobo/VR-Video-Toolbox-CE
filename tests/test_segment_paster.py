from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from gpu_engine.probe import VideoMetadata
from utils.mosaic_prescan import MosaicSegment
from utils.segment_paster import PasteSeg, _paste_segments_ffmpeg, paste_segments_gpu_or_fallback


class SegmentPasterTests(unittest.TestCase):
    def test_gpu_paste_respects_keep_audio_false(self) -> None:
        segment = MosaicSegment(0, 1.0, 2.0, 1.0, 2.0, 10, 20, 128, 128, 0.9)
        meta = VideoMetadata(path="base.mp4", width=1920, height=1080, duration=10.0, source_fps=30.0)

        with (
            patch("gpu_engine.probe.probe_video", return_value=meta),
            patch("gpu_engine.files.paste_segments_gpu") as paste_gpu,
        ):
            paste_segments_gpu_or_fallback(
                "base.mp4",
                "out.mp4",
                [segment],
                ["restored.mp4"],
                keep_audio=False,
            )

        paste_gpu.assert_called_once()
        self.assertFalse(paste_gpu.call_args.kwargs["keep_audio"])

    def test_ffmpeg_fallback_can_disable_audio(self) -> None:
        segment = PasteSeg(
            seg_id=0,
            path=Path("restored.mp4"),
            base_frame_start=30,
            base_frame_end=60,
            start_s=1.0,
            end_s=2.0,
            x=10,
            y=20,
            w=128,
            h=128,
        )
        captured = {}

        class Proc:
            stdout = None
            returncode = 0

            def wait(self):
                return 0

        def fake_popen(cmd, **_kwargs):
            captured["cmd"] = cmd
            return Proc()

        with patch("subprocess.Popen", side_effect=fake_popen):
            _paste_segments_ffmpeg("base.mp4", "out.mp4", [segment], keep_audio=False)

        self.assertIn("-an", captured["cmd"])
        self.assertNotIn("0:a?", captured["cmd"])


if __name__ == "__main__":
    unittest.main()
