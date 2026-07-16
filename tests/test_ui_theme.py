from __future__ import annotations

import unittest

from utils import ui_theme


class UiThemeTests(unittest.TestCase):
    def test_normalize_theme(self) -> None:
        self.assertEqual(ui_theme.normalize_theme("dark"), "dark")
        self.assertEqual(ui_theme.normalize_theme("DARK"), "dark")
        self.assertEqual(ui_theme.normalize_theme("unknown"), "light")

    def test_palette_contains_distinct_light_and_dark_colors(self) -> None:
        self.assertNotEqual(ui_theme.LIGHT_PALETTE.HOME_BG, ui_theme.DARK_PALETTE.HOME_BG)
        self.assertNotEqual(ui_theme.LIGHT_PALETTE.CARD_TITLE_FG, ui_theme.DARK_PALETTE.CARD_TITLE_FG)

    def test_icon_matching_uses_page_title_keywords(self) -> None:
        self.assertNotEqual(ui_theme.icon_for_title("批量字幕翻译"), ui_theme.DEFAULT_NAV_ICON)
        self.assertNotEqual(ui_theme.icon_for_title("Screenshot"), ui_theme.DEFAULT_NAV_ICON)
        self.assertNotEqual(ui_theme.icon_for_title("音声クローン"), ui_theme.DEFAULT_NAV_ICON)
        self.assertEqual(ui_theme.icon_for_title("Unmatched page"), ui_theme.DEFAULT_NAV_ICON)

    def test_scroll_text_to_end_forces_scrollbar_to_latest_line(self) -> None:
        class FakeText:
            def __init__(self):
                self.seen = None
                self.moved = None

            def see(self, index):
                self.seen = index

            def yview_moveto(self, fraction):
                self.moved = fraction

        widget = FakeText()
        ui_theme.scroll_text_to_end(widget)

        self.assertEqual(widget.seen, "end")
        self.assertEqual(widget.moved, 1.0)


if __name__ == "__main__":
    unittest.main()
