from __future__ import annotations

import tempfile
import types
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from gpu_engine.probe import VideoMetadata
from utils import keyframe_cutter
from utils.mosaic_prescan import MosaicSegment
from utils.source_time_scanner import TimeInterval


class KeyframeListingTests(unittest.TestCase):
    def test_list_keyframes_uses_pyav_first(self) -> None:
        class Packet:
            def __init__(self, pts: int | None, key: bool):
                self.pts = pts
                self.is_keyframe = key

        class Container:
            streams = types.SimpleNamespace(video=[types.SimpleNamespace(time_base=Fraction(1, 1000))])

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def demux(self, _stream):
                return iter([
                    Packet(0, True),
                    Packet(33, False),
                    Packet(5005, True),
                    Packet(None, True),
                ])

        fake_av = types.SimpleNamespace(open=lambda *_args, **_kwargs: Container())

        with (
            patch.dict("sys.modules", {"av": fake_av}),
            patch("utils.keyframe_cutter.subprocess.check_output") as check_output,
        ):
            keyframes = keyframe_cutter.list_keyframes("source.mp4")

        self.assertEqual(keyframes, [0.0, 5.005])
        check_output.assert_not_called()

    def test_list_keyframes_falls_back_to_packet_key_flags_when_pyav_fails(self) -> None:
        calls = []

        def fake_check_output(cmd, **_kwargs):
            calls.append(cmd)
            return "0.000000,K__\n0.033367,___\n5.005000,K__\n"

        with (
            patch("utils.keyframe_cutter._list_keyframes_from_pyav", side_effect=RuntimeError("pyav failed")),
            patch("utils.keyframe_cutter.shutil.which", return_value="ffprobe"),
            patch("utils.keyframe_cutter.subprocess.check_output", side_effect=fake_check_output),
        ):
            keyframes = keyframe_cutter.list_keyframes("source.mp4")

        self.assertEqual(keyframes, [0.0, 5.005])
        self.assertEqual(len(calls), 1)
        self.assertIn("-show_packets", calls[0])
        self.assertNotIn("-skip_frame", calls[0])

    def test_list_keyframes_falls_back_to_frame_scan_when_packets_are_empty(self) -> None:
        calls = []

        def fake_check_output(cmd, **_kwargs):
            calls.append(cmd)
            if "-show_packets" in cmd:
                return "0.000000,___\n0.033367,___\n"
            return b'{"frames":[{"pts_time":"0.000000"},{"best_effort_timestamp_time":"5.005000"}]}'

        with (
            patch("utils.keyframe_cutter.shutil.which", return_value="ffprobe"),
            patch("utils.keyframe_cutter.subprocess.check_output", side_effect=fake_check_output),
        ):
            keyframes = keyframe_cutter.list_keyframes("source.mp4")

        self.assertEqual(keyframes, [0.0, 5.005])
        self.assertEqual(len(calls), 2)
        self.assertIn("-show_packets", calls[0])
        self.assertIn("-show_frames", calls[1])
        self.assertIn("-skip_frame", calls[1])


class SourceCutTimelineTests(unittest.TestCase):
    def test_cut_source_keyframe_copy_extracts_mosaic_and_virtual_gap_segments(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            src = root / "source.mp4"
            out_dir = root / "tmp"
            src.write_bytes(b"source")
            meta = VideoMetadata(path=str(src), width=4096, height=4096, duration=100.0, source_fps=30.0)
            calls = []

            def fake_run(cmd, **_kwargs):
                calls.append(cmd)
                Path(cmd[-1]).write_bytes(b"mosaic")

            with (
                patch("gpu_engine.probe.probe_video", return_value=meta),
                patch("utils.keyframe_cutter.list_keyframes", return_value=[0.0, 5.0, 25.0, 55.0, 75.0, 100.0]),
                patch("utils.keyframe_cutter._run", side_effect=fake_run),
            ):
                timeline = keyframe_cutter.cut_source_by_intervals(
                    src,
                    [TimeInterval(10.0, 20.0, 0.91), TimeInterval(60.0, 70.0, 0.88)],
                    out_dir,
                )

            self.assertEqual(len(calls), 2)
            self.assertIn("-c:v", calls[0])
            self.assertIn("copy", calls[0])
            self.assertIn("-an", calls[0])
            self.assertEqual(calls[0][calls[0].index("-ss") + 1], "5.000000")
            self.assertEqual(calls[0][calls[0].index("-t") + 1], "20.000000")
            self.assertEqual(calls[1][calls[1].index("-ss") + 1], "55.000000")
            self.assertEqual(calls[1][calls[1].index("-t") + 1], "20.000000")

            mosaic_entries = [entry for entry in timeline if entry.kind == "mosaic"]
            gap_entries = [entry for entry in timeline if entry.kind == "gap"]
            self.assertEqual(len(mosaic_entries), 2)
            self.assertEqual(len(gap_entries), 3)
            self.assertEqual((mosaic_entries[0].start_s, mosaic_entries[0].end_s), (5.0, 25.0))
            self.assertEqual((mosaic_entries[1].start_s, mosaic_entries[1].end_s), (55.0, 75.0))
            self.assertTrue(all(entry.path == src for entry in gap_entries))
            self.assertEqual((gap_entries[0].inpoint_s, gap_entries[0].outpoint_s), (0.0, 5.0))
            self.assertEqual((gap_entries[1].inpoint_s, gap_entries[1].outpoint_s), (25.0, 55.0))
            self.assertEqual((gap_entries[2].inpoint_s, gap_entries[2].outpoint_s), (75.0, 100.0))
            self.assertTrue((out_dir / "mosaic_seg000.mp4").exists())
            self.assertFalse((out_dir / "gap_seg000.mp4").exists())

    def test_cut_source_can_materialize_gap_segments_for_single_eye(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            src = root / "source.mp4"
            out_dir = root / "tmp"
            src.write_bytes(b"source")
            meta = VideoMetadata(path=str(src), width=4096, height=4096, duration=30.0, source_fps=30.0)
            calls = []

            def fake_run(cmd, **_kwargs):
                calls.append(cmd)
                Path(cmd[-1]).write_bytes(b"clip")

            with (
                patch("gpu_engine.probe.probe_video", return_value=meta),
                patch("utils.keyframe_cutter.list_keyframes", return_value=[0.0, 10.0, 20.0, 30.0]),
                patch("utils.keyframe_cutter._run", side_effect=fake_run),
            ):
                timeline = keyframe_cutter.cut_source_by_intervals(
                    src,
                    [TimeInterval(10.0, 20.0, 0.91)],
                    out_dir,
                    materialize_gaps=True,
                )

            self.assertEqual(len(calls), 3)
            gap_entries = [entry for entry in timeline if entry.kind == "gap"]
            self.assertEqual(len(gap_entries), 2)
            self.assertTrue(all(entry.path.parent == out_dir for entry in gap_entries))
            self.assertTrue((out_dir / "gap_seg000.mp4").exists())

    def test_cut_source_can_keep_mosaic_segments_virtual(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            src = root / "source.mp4"
            out_dir = root / "tmp"
            src.write_bytes(b"source")
            meta = VideoMetadata(path=str(src), width=4096, height=4096, duration=30.0, source_fps=30.0)

            with (
                patch("gpu_engine.probe.probe_video", return_value=meta),
                patch("utils.keyframe_cutter.list_keyframes", return_value=[0.0, 10.0, 20.0, 30.0]),
                patch("utils.keyframe_cutter._run") as run_cmd,
            ):
                timeline = keyframe_cutter.cut_source_by_intervals(
                    src,
                    [TimeInterval(10.0, 20.0, 0.91)],
                    out_dir,
                    materialize_mosaic=False,
                )

            run_cmd.assert_not_called()
            mosaic_entries = [entry for entry in timeline if entry.kind == "mosaic"]
            self.assertEqual(len(mosaic_entries), 1)
            self.assertEqual(mosaic_entries[0].path, src)
            self.assertEqual((mosaic_entries[0].inpoint_s, mosaic_entries[0].outpoint_s), (10.0, 20.0))
            self.assertFalse((out_dir / "mosaic_seg000.mp4").exists())

    def test_cut_segment_keeps_ffmpeg_runner_for_inner_pre_extract_rect_crop(self) -> None:
        segment = MosaicSegment(
            seg_id=0,
            start_s=1.0,
            end_s=3.0,
            start_s_kf=0.5,
            end_s_kf=3.5,
            x=16,
            y=32,
            w=640,
            h=480,
            conf_max=0.91,
        )
        meta = VideoMetadata(path="base.mp4", width=4096, height=4096, duration=10.0, source_fps=30.0)

        with (
            patch("gpu_engine.probe.probe_video", return_value=meta),
            patch("utils.keyframe_cutter._run") as run_cmd,
            patch("utils.keyframe_cutter.get_video_size", return_value=(640, 480)),
        ):
            keyframe_cutter.cut_segment("base.mp4", "seg.mp4", segment)

        run_cmd.assert_called_once()
        cmd = run_cmd.call_args.args[0]
        self.assertIn("crop=640:480:16:32", cmd)
        self.assertIn("-ss", cmd)


if __name__ == "__main__":
    unittest.main()
