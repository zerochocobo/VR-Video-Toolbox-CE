from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from utils import app_config, i18n


class I18nTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_cache = dict(app_config._cache)
        i18n.clear_cache()

    def tearDown(self) -> None:
        app_config._cache = self._old_cache
        i18n.clear_cache()

    def test_translation_files_are_utf8_bom(self) -> None:
        self.assertFalse(Path("utils/translations.json").exists())
        for language in ("en", "zh", "ja"):
            path = Path("i18n") / f"{language}.json"
            self.assertTrue(path.exists(), path)
            self.assertEqual(path.read_bytes()[:3], b"\xef\xbb\xbf")

    def test_available_languages_from_json_files(self) -> None:
        self.assertEqual(
            i18n.available_languages(),
            {"en": "English", "zh": "简体中文", "ja": "日本語"},
        )

    def test_japanese_namespace_translation(self) -> None:
        app_config._cache = {"language": "ja"}

        text = i18n.translate("main", "btn_dlna_start")

        self.assertEqual(text, "▶ VR動画DLNAサーバーを起動")

    def test_japanese_file_has_no_question_mark_mojibake(self) -> None:
        app_config._cache = {"language": "ja"}

        values: list[str] = []
        for namespace in i18n.load_language("ja")["namespaces"].values():
            values.extend(str(value) for value in namespace.values())

        self.assertFalse([value for value in values if "???" in value])

    def test_chinese_title_uses_clean_translation(self) -> None:
        app_config._cache = {"language": "zh"}

        text = i18n.translate("main", "title")

        self.assertEqual(text, "VR视频工具箱(CUDA专版){version}")

    def test_dlna_port_note_translations(self) -> None:
        expected = {
            "en": "Make sure the firewall allows this port. UDP port 1900 is also fixed for broadcasting server information.",
            "zh": "请确保防火墙打开该端口，另外还有UDP协议1900端口固定用于广播服务器信息",
            "ja": "ファイアウォールでこのポートを許可してください。また、UDP 1900番ポートはサーバー情報のブロードキャストに固定で使用されます。",
        }
        for language, text in expected.items():
            app_config._cache = {"language": language}
            self.assertEqual(i18n.translate("main", "lbl_dlna_port_note"), text)

    def test_missing_japanese_key_falls_back_to_english(self) -> None:
        app_config._cache = {"language": "ja"}

        data = {
            "ja": {"namespaces": {"main": {}}},
            "en": {"namespaces": {"main": {"unknown_key": "English fallback"}}},
        }
        with patch("utils.i18n.load_language", side_effect=lambda lang: data[lang]):
            text = i18n.translate("main", "unknown_key")

        self.assertEqual(text, "English fallback")


if __name__ == "__main__":
    unittest.main()
