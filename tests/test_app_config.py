from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils import app_config, encode_config
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

    def test_dlna_si_defaults_match_live_mix_ui(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            app_config._cache = {}
            app_config._CONFIG_PATH = str(Path(raw) / "config.json")

            self.assertTrue(app_config.get("dlna_si_enabled"))
            self.assertEqual(app_config.get("dlna_si_mix_channel"), "both")
            self.assertEqual(app_config.get("dlna_si_original_volume_percent"), 100)
            self.assertEqual(app_config.get("dlna_si_volume_percent"), 100)
            self.assertEqual(app_config.get("dlna_si_delay_seconds"), 1.0)
            self.assertTrue(app_config.get("dlna_si_duck_original"))
            self.assertEqual(app_config.get("dlna_si_duck_preset"), "normal")

    def test_set_language_persists_global_language(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            app_config._cache = {}
            app_config._CONFIG_PATH = str(Path(raw) / "config.json")

            app_config.set_language("en")

            self.assertEqual(app_config.get_language(), "en")
            self.assertIn('"language": "en"', Path(app_config._CONFIG_PATH).read_text(encoding="utf-8"))

    def test_ui_theme_defaults_normalizes_and_persists(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            app_config._cache = {}
            app_config._CONFIG_PATH = str(Path(raw) / "config.json")

            self.assertEqual(app_config.get_ui_theme(), "light")
            app_config.set_ui_theme("dark")
            self.assertEqual(app_config.get_ui_theme(), "dark")

            saved = json.loads(Path(app_config._CONFIG_PATH).read_text(encoding="utf-8-sig"))
            self.assertEqual(saved["ui_theme"], "dark")

            app_config.set_ui_theme("unsupported")
            self.assertEqual(app_config.get_ui_theme(), "light")

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
                        "transcode_backend": "gpu",
                        "mosaic_engine": "jasna",
                        "gpu_log_verbose": True,
                        "gpu_bitrate_multiplier": 9.0,
                        "gpu_bitrate_final_multiplier": 9.0,
                        "gpu_encode_profile": "highest_quality",
                        "gpu_encode_preset": "P7",
                        "gpu_encode_multipass": "off",
                        "gpu_encode_aq": False,
                        "gpu_encode_aq_strength": 12,
                        "gpu_encode_temporal_aq": True,
                        "gpu_encode_maxrate_multiplier": 9.0,
                        "gpu_final_encode_maxrate_multiplier": 9.0,
                        "gpu_final_encode_bframes": 4,
                        "gpu_final_encode_gop_sec": 9.0,
                        "native_stream_enabled": True,
                        "native_detection_model": "custom.pt",
                        "progress_log_interval_s": 99.0,
                        "progress_log_min_pct": 99.0,
                        "progress_log_vram": False,
                        "progress_vram_query_interval_s": 99.0,
                        "progress_native_log_interval_s": 99.0,
                        "progress_native_log_min_pct": 99.0,
                        "output_mp4_faststart": "always",
                        "paste_passthrough_enabled": False,
                        "paste_passthrough_min_frames": 1,
                        "paste_passthrough_max_subseg": 1,
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
            self.assertEqual(app_config.get("transcode_backend"), app_config._DEFAULTS["transcode_backend"])
            self.assertEqual(app_config.get("mosaic_engine"), app_config._DEFAULTS["mosaic_engine"])
            self.assertEqual(app_config.get("gpu_log_verbose"), app_config._DEFAULTS["gpu_log_verbose"])
            self.assertEqual(app_config.get("gpu_bitrate_multiplier"), 1.2)
            self.assertEqual(app_config.get("gpu_bitrate_final_multiplier"), 1.0)
            self.assertEqual(app_config.get("gpu_encode_profile"), "highest_quality")
            self.assertEqual(app_config.get("gpu_encode_preset"), "P4")
            self.assertEqual(app_config.get("gpu_encode_multipass"), "fullres")
            self.assertEqual(app_config.get("gpu_encode_aq"), True)
            self.assertEqual(app_config.get("gpu_encode_aq_strength"), 6)
            self.assertEqual(app_config.get("gpu_encode_temporal_aq"), False)
            self.assertEqual(app_config.get("gpu_encode_maxrate_multiplier"), 2.0)
            self.assertEqual(app_config.get("gpu_final_encode_maxrate_multiplier"), 1.1)
            self.assertEqual(app_config.get("gpu_final_encode_bframes"), 2)
            self.assertEqual(app_config.get("gpu_final_encode_gop_sec"), 2.0)
            self.assertEqual(app_config.get("native_stream_enabled"), False)
            self.assertEqual(app_config.get("native_detection_model"), "")
            self.assertEqual(app_config.get("progress_log_interval_s"), 5.0)
            self.assertEqual(app_config.get("progress_log_min_pct"), 5.0)
            self.assertEqual(app_config.get("progress_log_vram"), True)
            self.assertEqual(app_config.get("progress_vram_query_interval_s"), 5.0)
            self.assertEqual(app_config.get("progress_native_log_interval_s"), 5.0)
            self.assertEqual(app_config.get("progress_native_log_min_pct"), 20.0)
            self.assertEqual(app_config.get("output_mp4_faststart"), "auto")
            self.assertEqual(app_config.get("paste_passthrough_enabled"), True)
            self.assertEqual(app_config.get("paste_passthrough_min_frames"), 60)
            self.assertEqual(app_config.get("paste_passthrough_max_subseg"), 32)
            self.assertEqual(encode_config.resolve_encode_settings().preset, "P7")

    def test_code_default_only_keys_are_not_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            app_config._cache = {}
            app_config._CONFIG_PATH = str(Path(raw) / "config.json")
            Path(app_config._CONFIG_PATH).write_text(
                json.dumps(
                    {
                        "pre_extract_yolo_conf": 0.99,
                        "source_scan_final_merge_mode": "gpu",
                        "transcode_backend": "gpu",
                        "mosaic_engine": "jasna",
                        "gpu_log_verbose": True,
                        "gpu_bitrate_multiplier": 9.0,
                        "gpu_bitrate_final_multiplier": 9.0,
                        "gpu_encode_profile": "highest_quality",
                        "gpu_encode_preset": "P7",
                        "gpu_encode_multipass": "off",
                        "gpu_encode_aq": False,
                        "gpu_encode_aq_strength": 12,
                        "gpu_encode_temporal_aq": True,
                        "gpu_encode_maxrate_multiplier": 9.0,
                        "gpu_final_encode_maxrate_multiplier": 9.0,
                        "gpu_final_encode_bframes": 4,
                        "gpu_final_encode_gop_sec": 9.0,
                        "native_stream_enabled": True,
                        "native_detection_model": "custom.pt",
                        "progress_log_interval_s": 99.0,
                        "progress_log_min_pct": 99.0,
                        "progress_log_vram": False,
                        "progress_vram_query_interval_s": 99.0,
                        "progress_native_log_interval_s": 99.0,
                        "progress_native_log_min_pct": 99.0,
                        "output_mp4_faststart": "always",
                        "paste_passthrough_enabled": False,
                        "paste_passthrough_min_frames": 1,
                        "paste_passthrough_max_subseg": 1,
                    }
                ),
                encoding="utf-8",
            )

            app_config.set_language("en")
            saved = json.loads(Path(app_config._CONFIG_PATH).read_text(encoding="utf-8"))

            self.assertNotIn("pre_extract_yolo_conf", saved)
            self.assertNotIn("source_scan_final_merge_mode", saved)
            self.assertNotIn("transcode_backend", saved)
            self.assertNotIn("mosaic_engine", saved)
            self.assertNotIn("gpu_log_verbose", saved)
            self.assertNotIn("gpu_bitrate_multiplier", saved)
            self.assertNotIn("gpu_bitrate_final_multiplier", saved)
            self.assertNotIn("gpu_encode_preset", saved)
            self.assertNotIn("gpu_encode_multipass", saved)
            self.assertNotIn("gpu_encode_aq", saved)
            self.assertNotIn("gpu_encode_aq_strength", saved)
            self.assertNotIn("gpu_encode_temporal_aq", saved)
            self.assertNotIn("gpu_encode_maxrate_multiplier", saved)
            self.assertNotIn("gpu_final_encode_maxrate_multiplier", saved)
            self.assertNotIn("gpu_final_encode_bframes", saved)
            self.assertNotIn("gpu_final_encode_gop_sec", saved)
            self.assertNotIn("native_stream_enabled", saved)
            self.assertNotIn("native_detection_model", saved)
            self.assertNotIn("progress_log_interval_s", saved)
            self.assertNotIn("progress_log_min_pct", saved)
            self.assertNotIn("progress_log_vram", saved)
            self.assertNotIn("progress_vram_query_interval_s", saved)
            self.assertNotIn("progress_native_log_interval_s", saved)
            self.assertNotIn("progress_native_log_min_pct", saved)
            self.assertNotIn("output_mp4_faststart", saved)
            self.assertNotIn("paste_passthrough_enabled", saved)
            self.assertNotIn("paste_passthrough_min_frames", saved)
            self.assertNotIn("paste_passthrough_max_subseg", saved)
            self.assertEqual(saved["gpu_encode_profile"], "highest_quality")
            self.assertEqual(saved["language"], "en")

    def test_code_default_only_set_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            app_config._cache = {}
            app_config._CONFIG_PATH = str(Path(raw) / "config.json")

            app_config.set("pre_extract_yolo_conf", 0.99)
            app_config.set("output_mp4_faststart", "always")
            app_config.set("gpu_encode_preset", "P7")

            self.assertEqual(app_config.get("pre_extract_yolo_conf"), app_config._DEFAULTS["pre_extract_yolo_conf"])
            self.assertEqual(app_config.get("output_mp4_faststart"), app_config._DEFAULTS["output_mp4_faststart"])
            self.assertEqual(app_config.get("gpu_encode_preset"), app_config._DEFAULTS["gpu_encode_preset"])
            self.assertFalse(Path(app_config._CONFIG_PATH).exists())

    def test_gpu_encoder_preset_and_peak_headroom_use_code_defaults(self) -> None:
        app_config._cache = {"gpu_encode_preset": "P4"}
        meta = VideoMetadata(path="in.mp4", source_fps=59.94)

        kwargs = _encoder_kwargs(meta, 9_677_852)

        self.assertEqual(kwargs["preset"], "P4")
        self.assertEqual(kwargs["bitrate"], "9677852")
        self.assertEqual(kwargs["maxbitrate"], str(int(9_677_852 * 2.0)))
        self.assertEqual(kwargs["multipass"], "fullres")
        self.assertEqual(kwargs["aq"], "1")
        self.assertNotIn("temporalaq", kwargs)

    def test_default_encode_profile_is_balanced_high_quality(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            app_config._cache = {}
            app_config._CONFIG_PATH = str(Path(raw) / "config.json")

            self.assertEqual(encode_config.DEFAULT_ENCODE_PROFILE, "balanced_high_quality")
            self.assertEqual(app_config.get("gpu_encode_profile"), "balanced_high_quality")
            self.assertEqual(app_config.get("gpu_encode_preset"), "P4")
            self.assertEqual(encode_config.current_encode_profile_key(), "balanced_high_quality")

    def test_gpu_encoder_maxrate_multiplier_uses_code_default(self) -> None:
        app_config._cache = {
            "gpu_encode_preset": "P4",
            "gpu_encode_maxrate_multiplier": 1.5,
        }
        meta = VideoMetadata(path="in.mp4", source_fps=59.94)

        kwargs = _encoder_kwargs(meta, 10_000_000)

        self.assertEqual(kwargs["bitrate"], "10000000")
        self.assertEqual(kwargs["maxbitrate"], "20000000")

    def test_gpu_encoder_quality_knobs_follow_profile_defaults(self) -> None:
        app_config._cache = {
            "gpu_encode_profile": "ultra_fast_normal",
            "gpu_encode_preset": "P7",
            "gpu_encode_multipass": "fullres",
            "gpu_encode_aq": False,
            "gpu_encode_temporal_aq": True,
        }
        meta = VideoMetadata(path="in.mp4", source_fps=59.94)

        kwargs = _encoder_kwargs(meta, 9_677_852)

        self.assertEqual(kwargs["preset"], "P1")
        self.assertNotIn("multipass", kwargs)
        self.assertEqual(kwargs["aq"], "1")
        self.assertNotIn("temporalaq", kwargs)

    def test_apply_encode_profile_persists_only_profile_key(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            app_config._cache = {}
            app_config._CONFIG_PATH = str(Path(raw) / "config.json")

            encode_config.apply_encode_profile("highest_quality")

            saved = json.loads(Path(app_config._CONFIG_PATH).read_text(encoding="utf-8"))
            self.assertEqual(saved["gpu_encode_profile"], "highest_quality")
            self.assertNotIn("gpu_encode_preset", saved)
            self.assertNotIn("gpu_encode_multipass", saved)
            self.assertNotIn("gpu_encode_aq", saved)
            self.assertNotIn("gpu_encode_aq_strength", saved)
            self.assertNotIn("gpu_encode_temporal_aq", saved)
            settings = encode_config.resolve_encode_settings()
            self.assertEqual(settings.preset, "P7")
            self.assertEqual(settings.multipass, "fullres")
            self.assertTrue(settings.aq)
            self.assertFalse(settings.temporal_aq)

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
