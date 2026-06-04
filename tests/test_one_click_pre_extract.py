from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from one_click import logic


class OneClickPreExtractTests(unittest.TestCase):
    def test_process_file_logger_writes_utf8_bom_log_next_to_input(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            video = Path(raw) / "clip.mp4"
            video.write_bytes(b"video")
            ui_messages: list[str] = []

            logger = logic._ProcessFileLogger(str(video), ui_messages.append)
            logger("中文路径日志")
            logger.close()

            log_path = Path(raw) / "clip_process.log"
            data = log_path.read_bytes()
            self.assertTrue(data.startswith(b"\xef\xbb\xbf"))
            text = data.decode("utf-8-sig")
            self.assertIn("中文路径日志", text)
            self.assertTrue(any("中文路径日志" in item for item in ui_messages))

    def test_no_mosaic_detection_copies_base_and_skips_full_lada_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw) / "base.mp4"
            restored = Path(raw) / "base.restored.mp4"
            base.write_bytes(b"not a real video, but enough for copy-through")
            logs: list[str] = []

            with patch("utils.mosaic_prescan.scan_segments", return_value=[]) as scan_segments:
                result = logic._run_pre_extract_branch(
                    str(base),
                    str(restored),
                    keep_intermediate=True,
                    log_callback=logs.append,
                    fine_conf=0.5,
                )

            self.assertEqual(result, logic.PreExtractResult.NO_MOSAIC)
            self.assertEqual(scan_segments.call_args.kwargs["min_conf"], 0.5)
            self.assertEqual(restored.read_bytes(), base.read_bytes())
            self.assertTrue(any("skipping lada/jasna" in item for item in logs))

            metadata = json.loads((Path(raw) / "base.segments.json").read_text(encoding="utf-8-sig"))
            self.assertEqual(metadata["segments"], [])

    def test_pre_extract_no_mosaic_removes_metadata_when_not_kept(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw) / "base.mp4"
            restored = Path(raw) / "base.restored.mp4"
            detections = Path(raw) / "base.detections.jsonl"
            base.write_bytes(b"not a real video, but enough for copy-through")
            detections.write_text("{}\n", encoding="utf-8")

            with patch("utils.mosaic_prescan.scan_segments", return_value=[]):
                result = logic._run_pre_extract_branch(
                    str(base),
                    str(restored),
                    keep_intermediate=False,
                )

            self.assertEqual(result, logic.PreExtractResult.NO_MOSAIC)
            self.assertFalse((Path(raw) / "base.segments.json").exists())
            self.assertFalse(detections.exists())

    def test_successful_single_file_run_removes_process_artifacts_when_not_kept(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            video = Path(raw) / "clip.mp4"
            video.write_bytes(b"video")
            detections = Path(raw) / "clip.detections.jsonl"
            intervals = Path(raw) / "clip_SSTART_EEND_sbs.restored.source_intervals.json"
            detections.write_text("{}\n", encoding="utf-8")
            intervals.write_text("{}", encoding="utf-8")

            with (
                patch.object(logic, "get_video_bitrate", return_value=1000000),
                patch.object(logic, "_native_stream_allowed", return_value=True),
                patch.object(logic, "_run_native_sbs_stream", return_value=True),
            ):
                logic.run_single_file_pipeline(
                    str(video),
                    None,
                    None,
                    use_fisheye=False,
                    keep_intermediate=False,
                    source_scan=False,
                )

            self.assertFalse((Path(raw) / "clip_process.log").exists())
            self.assertFalse(detections.exists())
            self.assertFalse(intervals.exists())

    def test_single_file_source_scan_defaults_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            video = Path(raw) / "clip.mp4"
            video.write_bytes(b"video")

            with (
                patch.object(logic, "get_video_bitrate", return_value=1000000),
                patch.object(logic, "_run_source_scan_branch", return_value=logic.PreExtractResult.NO_MOSAIC) as source_scan,
                patch.object(logic, "_run_native_sbs_stream") as native_stream,
            ):
                logic.run_single_file_pipeline(
                    str(video),
                    None,
                    None,
                    use_fisheye=False,
                    keep_intermediate=False,
                )

            source_scan.assert_called_once()
            native_stream.assert_not_called()

    def test_scan_failure_falls_back_to_full_lada_in_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw) / "base.mp4"
            restored = Path(raw) / "base.restored.mp4"
            base.write_bytes(b"video")

            with (
                patch("utils.mosaic_prescan.scan_segments", side_effect=RuntimeError("detector failed")),
                patch.object(logic, "process_lada") as process_lada,
            ):
                result = logic._process_pre_extract_or_lada(
                    str(base),
                    str(restored),
                    pre_extract_enabled=True,
                )

            self.assertEqual(result, logic.PreExtractResult.OK)
            process_lada.assert_called_once_with(str(base), str(restored), log_callback=None, process_callback=None)


if __name__ == "__main__":
    unittest.main()
