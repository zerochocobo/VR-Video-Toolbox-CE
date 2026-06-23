from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tool_clonevoice.gui import ClonevoiceToolsApp


class ClonevoiceBatchScanTests(unittest.TestCase):
    def test_scan_clone_batch_videos_ignores_generated_mp4_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "movie.mp4").write_bytes(b"video")
            (root / "movie_SI.mp4").write_bytes(b"si output")
            (root / "movie_DUB.MP4").write_bytes(b"dub output")
            (root / "movie.mkv").write_bytes(b"mkv")

            videos = ClonevoiceToolsApp._scan_clone_batch_videos(object(), str(root))

        self.assertEqual([Path(path).name for path in videos], ["movie.mkv", "movie.mp4"])

    def test_dubbing_available_does_not_import_separator_backend(self) -> None:
        sys.modules.pop("tool_clonevoice.separate", None)
        with tempfile.TemporaryDirectory() as tmp_dir:
            models_root = Path(tmp_dir)
            bandit_dir = models_root / "bandit-v2"
            bandit_dir.mkdir()
            (bandit_dir / "checkpoint-multi.slim.pt").write_bytes(b"weights")
            obj = type("Obj", (), {"models_root": str(models_root)})()

            available = ClonevoiceToolsApp._dubbing_available(obj)

        self.assertTrue(available)
        self.assertNotIn("tool_clonevoice.separate", sys.modules)

    def test_translation_api_key_configured_accepts_saved_key(self) -> None:
        fake_keyring = types.SimpleNamespace(
            get_password=lambda service, name: (
                "secret" if service == "VR_Video_Toolbox" and name == "deepseek_api_key" else None
            )
        )
        obj = object()

        with patch.dict(sys.modules, {"keyring": fake_keyring}):
            configured = ClonevoiceToolsApp._translation_api_key_configured(obj)

        self.assertTrue(configured)

    def test_translation_api_key_configured_rejects_missing_key(self) -> None:
        fake_keyring = types.SimpleNamespace(get_password=lambda _service, _name: "")
        obj = object()

        with patch.dict(sys.modules, {"keyring": fake_keyring}):
            configured = ClonevoiceToolsApp._translation_api_key_configured(obj)

        self.assertFalse(configured)

    def test_selected_target_language_returns_builtin_value(self) -> None:
        obj = types.SimpleNamespace(
            tgt_lang_var=types.SimpleNamespace(get=lambda: "English"),
            _tgt_map={"Chinese": "Chinese", "English": "English"},
        )

        self.assertEqual(ClonevoiceToolsApp._selected_target_language(obj), "English")

    def test_selected_target_language_maps_fixed_korean_display_name(self) -> None:
        obj = types.SimpleNamespace(
            tgt_lang_var=types.SimpleNamespace(get=lambda: "韩语"),
            _tgt_map={"中文": "Chinese", "英语": "English", "韩语": "Korean"},
        )

        self.assertEqual(ClonevoiceToolsApp._selected_target_language(obj), "Korean")

    def test_selected_target_language_maps_fixed_thai_display_name(self) -> None:
        obj = types.SimpleNamespace(
            tgt_lang_var=types.SimpleNamespace(get=lambda: "泰语"),
            _tgt_map={"中文": "Chinese", "英语": "English", "泰语": "Thai"},
        )

        self.assertEqual(ClonevoiceToolsApp._selected_target_language(obj), "Thai")


if __name__ == "__main__":
    unittest.main()
