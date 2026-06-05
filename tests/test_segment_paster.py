from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from gpu_engine.fallback import OperationCancelled
from gpu_engine.probe import VideoMetadata
from utils.mosaic_prescan import MosaicSegment
from utils.segment_paster import (
    PasteSeg,
    _build_passthrough_plan,
    _paste_segments_ffmpeg,
    paste_segments_gpu_or_fallback,
)


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

    def test_gpu_paste_cancel_does_not_use_ffmpeg_fallback(self) -> None:
        segment = MosaicSegment(0, 1.0, 2.0, 1.0, 2.0, 10, 20, 128, 128, 0.9)
        meta = VideoMetadata(path="base.mp4", width=1920, height=1080, duration=10.0, source_fps=30.0)

        with (
            patch("gpu_engine.probe.probe_video", return_value=meta),
            patch("gpu_engine.files.paste_segments_gpu", side_effect=OperationCancelled("cancelled")),
            patch("utils.segment_paster._paste_segments_ffmpeg") as paste_ffmpeg,
        ):
            with self.assertRaises(OperationCancelled):
                paste_segments_gpu_or_fallback(
                    "base.mp4",
                    "out.mp4",
                    [segment],
                    ["restored.mp4"],
                    keep_audio=False,
                )

        paste_ffmpeg.assert_not_called()

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

    def test_ffmpeg_fallback_materializes_raw_hevc_inputs(self) -> None:
        segment = PasteSeg(
            seg_id=0,
            path=Path("restored.hevc"),
            base_frame_start=30,
            base_frame_end=60,
            start_s=1.0,
            end_s=2.0,
            x=10,
            y=20,
            w=128,
            h=128,
        )
        meta = VideoMetadata(path="restored.hevc", width=128, height=128, duration=1.0, source_fps=30.0)

        with (
            patch("gpu_engine.restored_sidecar.metadata_from_sidecar", return_value=meta),
            patch("gpu_engine.mux.mux_hevc_with_audio") as mux_raw,
            patch("utils.segment_paster._paste_segments_ffmpeg_impl") as paste_impl,
        ):
            _paste_segments_ffmpeg("base.mp4", "out.mp4", [segment], keep_audio=False)

        mux_raw.assert_called_once()
        paste_impl.assert_called_once()
        materialized = paste_impl.call_args.args[2][0]
        self.assertEqual(materialized.path.suffix, ".mp4")

    def test_gpu_paste_raw_hevc_failure_retries_with_wrapped_mp4(self) -> None:
        segment = MosaicSegment(0, 1.0, 2.0, 1.0, 2.0, 10, 20, 128, 128, 0.9)
        base_meta = VideoMetadata(path="base.mp4", width=1920, height=1080, duration=10.0, source_fps=30.0)
        raw_meta = VideoMetadata(path="restored.hevc", width=128, height=128, duration=1.0, source_fps=30.0)

        def fake_gpu(_src, _dst, segs, **_kwargs):
            if segs[0].path.suffix.lower() == ".hevc":
                raise RuntimeError("raw decode failed")
            return None

        with (
            patch("gpu_engine.probe.probe_video", return_value=base_meta),
            patch("gpu_engine.restored_sidecar.metadata_from_sidecar", return_value=raw_meta),
            patch("gpu_engine.mux.mux_hevc_with_audio") as mux_raw,
            patch("gpu_engine.files.paste_segments_gpu", side_effect=fake_gpu) as paste_gpu,
            patch("utils.segment_paster._paste_segments_ffmpeg") as paste_ffmpeg,
        ):
            paste_segments_gpu_or_fallback(
                "base.mp4",
                "out.mp4",
                [segment],
                ["restored.hevc"],
                keep_audio=False,
            )

        self.assertEqual(paste_gpu.call_count, 2)
        self.assertEqual(paste_gpu.call_args_list[0].args[2][0].path.suffix, ".hevc")
        self.assertEqual(paste_gpu.call_args_list[1].args[2][0].path.suffix, ".mp4")
        mux_raw.assert_called_once()
        paste_ffmpeg.assert_not_called()

    def test_passthrough_plan_splits_inactive_keyframe_aligned_gaps(self) -> None:
        segments = [
            PasteSeg(0, Path("a.mp4"), 30, 60, 1.0, 2.0, 10, 20, 128, 128),
            PasteSeg(1, Path("b.mp4"), 120, 150, 4.0, 5.0, 10, 20, 128, 128),
            PasteSeg(2, Path("c.mp4"), 210, 240, 7.0, 8.0, 10, 20, 128, 128),
        ]

        plan = _build_passthrough_plan(
            segments,
            total_frames=300,
            keyframes=[0, 1, 2, 4, 5, 7, 8, 10],
            fps=30.0,
            min_passthrough_frames=15,
            max_passthrough_count=10,
        )

        self.assertEqual(
            [(part.kind, part.start_frame, part.end_frame) for part in plan],
            [
                ("passthrough", 0, 30),
                ("paste", 30, 60),
                ("passthrough", 60, 120),
                ("paste", 120, 150),
                ("passthrough", 150, 210),
                ("paste", 210, 240),
                ("passthrough", 240, 300),
            ],
        )

    def test_passthrough_plan_rejects_short_gaps_and_max_subseg_overflow(self) -> None:
        segments = [
            PasteSeg(0, Path("a.mp4"), 30, 60, 1.0, 2.0, 10, 20, 128, 128),
            PasteSeg(1, Path("b.mp4"), 70, 100, 2.333, 3.333, 10, 20, 128, 128),
        ]

        plan = _build_passthrough_plan(
            segments,
            total_frames=130,
            keyframes=[0, 1, 2, 70 / 30.0, 100 / 30.0, 130 / 30.0],
            fps=30.0,
            min_passthrough_frames=15,
            max_passthrough_count=10,
        )
        self.assertNotIn(("passthrough", 60, 70), [(part.kind, part.start_frame, part.end_frame) for part in plan])
        self.assertIn(("paste", 30, 100), [(part.kind, part.start_frame, part.end_frame) for part in plan])

        overflow = _build_passthrough_plan(
            segments,
            total_frames=130,
            keyframes=[0, 1, 2, 70 / 30.0, 100 / 30.0, 130 / 30.0],
            fps=30.0,
            min_passthrough_frames=1,
            max_passthrough_count=1,
        )
        self.assertEqual(overflow, [])

    def test_gpu_paste_uses_passthrough_subsegments_when_eligible(self) -> None:
        segment = MosaicSegment(0, 1.0, 2.0, 1.0, 2.0, 10, 20, 128, 128, 0.9)
        meta = VideoMetadata(
            path="base.mp4",
            codec_name="hevc",
            width=1920,
            height=1080,
            duration=4.0,
            nb_frames=120,
            source_fps=30.0,
        )

        def fake_cut(_src, dst, _start, _end, **_kwargs):
            Path(dst).write_bytes(b"copy")

        def fake_gpu(_src, _dst, _segments, **_kwargs):
            return None

        def fake_concat(_timeline, _output, **_kwargs):
            return None

        cfg = {
            "paste_passthrough_enabled": True,
            "paste_passthrough_min_frames": 1,
            "paste_passthrough_max_subseg": 32,
        }
        # Expected plan parts for 4s clip @ 30fps with active rect [30, 60]:
        # 0000 passthrough [0, 30], 0001 paste [30, 60], 0002 passthrough [60, 120].
        # Probe returns exact expected counts so source_cursor advances cleanly.
        expected_passthrough_frames = [30, 60]

        def fake_probe(_path):
            return expected_passthrough_frames.pop(0)

        with (
            patch("utils.app_config.get", side_effect=lambda key, default=None: cfg.get(key, default)),
            patch("gpu_engine.probe.probe_video", return_value=meta),
            patch("utils.keyframe_cutter.list_keyframes", return_value=[0.0, 1.0, 2.0, 4.0]),
            patch("utils.keyframe_cutter._cut_copy", side_effect=fake_cut) as cut_copy,
            patch("utils.segment_paster._probe_video_frame_count", side_effect=fake_probe),
            patch("gpu_engine.files.paste_segments_gpu", side_effect=fake_gpu) as paste_gpu,
            patch("utils.sbs_concat.concat_timeline_hevc_fast", side_effect=fake_concat) as concat_fast,
        ):
            paste_segments_gpu_or_fallback(
                "base.mp4",
                "out.mp4",
                [segment],
                ["restored.mp4"],
                keep_audio=False,
            )

        self.assertEqual(cut_copy.call_count, 2)
        paste_gpu.assert_called_once()
        self.assertEqual(paste_gpu.call_args.kwargs["start_frame"], 30)
        self.assertEqual(paste_gpu.call_args.kwargs["end_frame"], 60)
        concat_fast.assert_called_once()

    def test_passthrough_overshoot_shifts_next_paste_start(self) -> None:
        """`-c copy` cuts often overshoot by a few frames at GOP boundaries.
        The following paste subsegment must resume from the actual cursor so the
        concatenated output stays the same length as the source and does not
        double-encode the overshoot frames.
        """
        segment = MosaicSegment(0, 1.0, 2.0, 1.0, 2.0, 10, 20, 128, 128, 0.9)
        meta = VideoMetadata(
            path="base.mp4",
            codec_name="hevc",
            width=1920,
            height=1080,
            duration=4.0,
            nb_frames=120,
            source_fps=30.0,
        )

        def fake_cut(_src, dst, _start, _end, **_kwargs):
            Path(dst).write_bytes(b"copy")

        def fake_gpu(_src, _dst, _segments, **_kwargs):
            return None

        def fake_concat(_timeline, _output, **_kwargs):
            return None

        # First passthrough requested 30 frames but stream-copy returned 32
        # (overshot 2). The next passthrough requested 60 (frames 60..120)
        # also overshoots by 1 → 61 frames, but it's the trailing part so
        # the paste before it absorbs the prior overshoot.
        actual_passthrough_frames = [32, 61]

        def fake_probe(_path):
            return actual_passthrough_frames.pop(0)

        cfg = {
            "paste_passthrough_enabled": True,
            "paste_passthrough_min_frames": 1,
            "paste_passthrough_max_subseg": 32,
        }
        with (
            patch("utils.app_config.get", side_effect=lambda key, default=None: cfg.get(key, default)),
            patch("gpu_engine.probe.probe_video", return_value=meta),
            patch("utils.keyframe_cutter.list_keyframes", return_value=[0.0, 1.0, 2.0, 4.0]),
            patch("utils.keyframe_cutter._cut_copy", side_effect=fake_cut),
            patch("utils.segment_paster._probe_video_frame_count", side_effect=fake_probe),
            patch("gpu_engine.files.paste_segments_gpu", side_effect=fake_gpu) as paste_gpu,
            patch("utils.sbs_concat.concat_timeline_hevc_fast", side_effect=fake_concat),
        ):
            paste_segments_gpu_or_fallback(
                "base.mp4",
                "out.mp4",
                [segment],
                ["restored.mp4"],
                keep_audio=False,
            )

        paste_gpu.assert_called_once()
        # Paste subsegment was planned at frames [30, 60]; the first passthrough
        # actually delivered 32 frames so paste must now resume at 32, not 30.
        # End stays at the planned 60 — paste itself is frame-accurate.
        self.assertEqual(paste_gpu.call_args.kwargs["start_frame"], 32)
        self.assertEqual(paste_gpu.call_args.kwargs["end_frame"], 60)

    def test_passthrough_failure_retries_full_gpu_paste(self) -> None:
        segment = MosaicSegment(0, 1.0, 2.0, 1.0, 2.0, 10, 20, 128, 128, 0.9)
        meta = VideoMetadata(
            path="base.mp4",
            codec_name="hevc",
            width=1920,
            height=1080,
            duration=4.0,
            nb_frames=120,
            source_fps=30.0,
        )

        def fake_gpu(_src, _dst, _segments, **_kwargs):
            return None

        cfg = {
            "paste_passthrough_enabled": True,
            "paste_passthrough_min_frames": 1,
            "paste_passthrough_max_subseg": 32,
        }
        with (
            patch("utils.app_config.get", side_effect=lambda key, default=None: cfg.get(key, default)),
            patch("gpu_engine.probe.probe_video", return_value=meta),
            patch("utils.keyframe_cutter.list_keyframes", return_value=[0.0, 1.0, 2.0, 4.0]),
            patch("utils.keyframe_cutter._cut_copy", side_effect=RuntimeError("copy failed")),
            patch("gpu_engine.files.paste_segments_gpu", side_effect=fake_gpu) as paste_gpu,
            patch("utils.segment_paster._paste_segments_ffmpeg") as paste_ffmpeg,
        ):
            paste_segments_gpu_or_fallback(
                "base.mp4",
                "out.mp4",
                [segment],
                ["restored.mp4"],
                keep_audio=False,
            )

        paste_gpu.assert_called_once()
        self.assertNotIn("start_frame", paste_gpu.call_args.kwargs)
        paste_ffmpeg.assert_not_called()


if __name__ == "__main__":
    unittest.main()
