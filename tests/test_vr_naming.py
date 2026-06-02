from __future__ import annotations

import unittest

from tool_dlna.vr_naming import has_vr_filename_marker, source_display_stem


class VrNamingTests(unittest.TestCase):
    def test_detects_player_suffixes(self) -> None:
        for name in [
            "scene_LR",
            "scene-RLF",
            "scene_HSBS",
            "scene-Half-SBS",
            "scene_Left_Right",
            "scene-TB",
            "scene_HOU",
            "scene-Half-OU",
            "scene_Top_Bottom",
            "scene_FISHEYE190",
            "scene_RF52",
            "scene_MKX200",
            "scene_MKX22",
            "scene_VRCA220",
            "scene_EAC360",
            "scene_360EAC",
        ]:
            with self.subTest(name=name):
                self.assertTrue(has_vr_filename_marker(name))

    def test_does_not_treat_embedded_words_as_markers(self) -> None:
        self.assertFalse(has_vr_filename_marker("holiday180clip"))
        self.assertFalse(has_vr_filename_marker("scene F180"))
        self.assertFalse(has_vr_filename_marker("scene - F180"))

    def test_half_equirectangular_source_gets_lr_180_sbs_display_suffix(self) -> None:
        self.assertEqual(source_display_stem("movie", 3840, 1920), "movie_LR_180_SBS")
        self.assertEqual(source_display_stem("movie_180", 3840, 1920), "movie_180")
        self.assertEqual(source_display_stem("movie", 1920, 1080), "movie")
        self.assertEqual(source_display_stem("movie_LR_180", 3840, 1920), "movie_LR_180_SBS")
        self.assertEqual(source_display_stem("movie_LR_180_SBS", 3840, 1920), "movie_LR_180_SBS")

    def test_stem_with_dots_is_not_truncated(self) -> None:
        stem = "xxxx.com@atvr00067_1_8k"
        self.assertEqual(source_display_stem(stem, 7680, 3840), f"{stem}_LR_180_SBS")
        self.assertEqual(source_display_stem(f"{stem}.mp4", 7680, 3840), f"{stem}_LR_180_SBS")

    def test_non_video_dotted_suffix_is_part_of_stem(self) -> None:
        self.assertFalse(has_vr_filename_marker("scene.F180"))
        self.assertEqual(source_display_stem("scene.F180", 3840, 1920), "scene.F180_LR_180_SBS")


if __name__ == "__main__":
    unittest.main()
