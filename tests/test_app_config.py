from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils import app_config
from gpu_engine.files import _encoder_kwargs, _media_temp_path
from gpu_engine.probe import VideoMetadata


class AppConfigLanguageTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_cache = dict(app_config._cache)
        self._old_path = app_config._CONFIG_PATH

    def tearDown(self) -> None:
        app_config._cache = self._old_cache
        app_config._CONFIG_PATH = self._old_path

    def test_normalize_language_aliases(self) -> None:
        self.assertEqual(app_config.normalize_language("zh-CN"), "zh")
        self.assertEqual(app_config.normalize_language("ja-JP"), "ja")
        self.assertEqual(app_config.normalize_language("日本語"), "ja")
        self.assertEqual(app_config.normalize_language("English"), "en")
        self.assertEqual(app_config.normalize_language("unknown"), "")

    def test_get_language_falls_back_to_system_when_unconfigured(self) -> None:
        app_config._cache = {"language": ""}

        with patch.object(app_config, "get_system_language", return_value="zh"):
            self.assertEqual(app_config.get_language(), "zh")

    def test_set_language_persists_global_language(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            app_config._cache = {}
            app_config._CONFIG_PATH = str(Path(raw) / "config.json")

            app_config.set_language("en")

            self.assertEqual(app_config.get_language(), "en")
            self.assertIn('"language": "en"', Path(app_config._CONFIG_PATH).read_text(encoding="utf-8"))

    def test_code_default_only_keys_ignore_stale_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            app_config._cache = {}
            app_config._CONFIG_PATH = str(Path(raw) / "config.json")
            Path(app_config._CONFIG_PATH).write_text(
                json.dumps(
                    {
                        "pre_extract_yolo_conf": 0.99,
                        "pre_extract_yolo_imgsz": 8192,
                        "source_scan_enabled": False,
                        "source_scan_final_merge_mode": "gpu",
                        "gpu_encode_preset": "P4",
                    }
                ),
                encoding="utf-8-sig",
            )

            self.assertEqual(app_config.get("pre_extract_yolo_conf"), app_config._DEFAULTS["pre_extract_yolo_conf"])
            self.assertEqual(app_config.get("pre_extract_yolo_imgsz"), 2048)
            self.assertEqual(app_config.get("source_scan_enabled"), app_config._DEFAULTS["source_scan_enabled"])
            self.assertEqual(
                app_config.get("source_scan_final_merge_mode"),
                app_config._DEFAULTS["source_scan_final_merge_mode"],
            )
            self.assertEqual(app_config.get("gpu_encode_preset"), "P4")

    def test_code_default_only_keys_are_not_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            app_config._cache = {}
            app_config._CONFIG_PATH = str(Path(raw) / "config.json")
            Path(app_config._CONFIG_PATH).write_text(
                json.dumps(
                    {
                        "pre_extract_yolo_conf": 0.99,
                        "source_scan_final_merge_mode": "gpu",
                        "gpu_encode_preset": "P4",
                    }
                ),
                encoding="utf-8",
            )

            app_config.set_language("en")
            saved = json.loads(Path(app_config._CONFIG_PATH).read_text(encoding="utf-8"))

            self.assertNotIn("pre_extract_yolo_conf", saved)
            self.assertNotIn("source_scan_final_merge_mode", saved)
            self.assertEqual(saved["gpu_encode_preset"], "P4")
            self.assertEqual(saved["language"], "en")

    def test_code_default_only_set_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            app_config._cache = {}
            app_config._CONFIG_PATH = str(Path(raw) / "config.json")

            app_config.set("pre_extract_yolo_conf", 0.99)

            self.assertEqual(app_config.get("pre_extract_yolo_conf"), app_config._DEFAULTS["pre_extract_yolo_conf"])
            self.assertFalse(Path(app_config._CONFIG_PATH).exists())

    def test_gpu_encoder_preset_and_strict_maxrate_come_from_config(self) -> None:
        app_config._cache = {"gpu_encode_preset": "P4"}
        meta = VideoMetadata(path="in.mp4", source_fps=59.94)

        kwargs = _encoder_kwargs(meta, 9_677_852)

        self.assertEqual(kwargs["preset"], "P4")
        self.assertEqual(kwargs["bitrate"], "9677852")
        self.assertEqual(kwargs["maxbitrate"], "9677852")

    def test_gpu_encoder_accepts_explicit_max_bitrate(self) -> None:
        app_config._cache = {"gpu_encode_preset": "P4"}
        meta = VideoMetadata(path="in.mp4", source_fps=59.94)

        kwargs = _encoder_kwargs(meta, 15_000_000, max_bitrate_bps=20_000_000)

        self.assertEqual(kwargs["bitrate"], "15000000")
        self.assertEqual(kwargs["maxbitrate"], "20000000")

    def test_gpu_media_temp_path_is_next_to_output(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw) / "nested" / "video.mp4"

            temp_path = _media_temp_path(out, "timeline")

            self.assertEqual(temp_path.parent, out.parent)
            self.assertTrue(out.parent.exists())
            self.assertIn("video.timeline.", temp_path.name)


if __name__ == "__main__":
    unittest.main()
