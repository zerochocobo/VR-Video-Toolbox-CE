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

    def test_one_click_encode_profile_translations_are_in_one_click_namespace(self) -> None:
        expected = {
            "zh": "快速高画质",
            "en": "Fast high quality",
            "ja": "高速高画質",
        }
        for language, text in expected.items():
            app_config._cache = {"language": language}
            i18n.clear_cache()
            self.assertEqual(i18n.translate("one_click", "opt_encode_fast_quality"), text)
            self.assertNotEqual(i18n.translate("one_click", "lbl_encode_profile"), "lbl_encode_profile")

    def test_one_click_encode_profile_options_do_not_mark_recommended(self) -> None:
        keys = (
            "opt_encode_highest_quality",
            "opt_encode_balanced_high_quality",
            "opt_encode_fast_quality",
            "opt_encode_ultra_fast_normal",
        )
        forbidden = ("推荐", "recommended", "Recommended", "推奨")
        for language in ("zh", "en", "ja"):
            app_config._cache = {"language": language}
            i18n.clear_cache()
            for key in keys:
                text = i18n.translate("one_click", key)
                self.assertFalse(any(marker in text for marker in forbidden), (language, key, text))

    def test_one_click_pre_extract_hint_translations(self) -> None:
        expected = {
            "zh": ("功能说明", "（本功能不要用在最高画质，视频马赛克位置稳定建议打开）"),
            "en": ("Note", "(Do not use with maximum quality; recommended when mosaic positions are stable)"),
            "ja": ("説明", "（最高画質では使用しないでください。モザイク位置が安定している動画では有効化を推奨します）"),
        }
        for language, (title, text) in expected.items():
            app_config._cache = {"language": language}
            i18n.clear_cache()
            self.assertEqual(i18n.translate("one_click", "opt_pre_extract_hint_title"), title)
            self.assertEqual(i18n.translate("one_click", "opt_pre_extract_hint"), text)

    def test_one_click_pre_extract_label_does_not_mark_experimental(self) -> None:
        forbidden = ("实验功能", "Experimental", "実験機能")
        for language in ("zh", "en", "ja"):
            app_config._cache = {"language": language}
            i18n.clear_cache()
            text = i18n.translate("one_click", "opt_pre_extract")
            self.assertFalse(any(marker in text for marker in forbidden), (language, text))

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
