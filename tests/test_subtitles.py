from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tool_dlna.media_library import MediaLibrary, build_media_roots
from tool_dlna.subtitles import find_external_subtitles, subtitle_mime


class SubtitleDiscoveryTests(unittest.TestCase):
    def test_same_stem_subtitles_are_sorted_by_language_priority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "movie.mp4"
            video.write_bytes(b"video")
            (root / "movie.eng.srt").write_text("one", encoding="utf-8")
            (root / "movie.zh.ass").write_text("two", encoding="utf-8")
            (root / "movie.srt").write_text("three", encoding="utf-8")
            library = MediaLibrary(build_media_roots([root]))

            tracks = find_external_subtitles(video, True, library)

        self.assertEqual([track.path.name for track in tracks], ["movie.srt", "movie.zh.ass", "movie.eng.srt"])
        self.assertEqual([track.lang for track in tracks], ["", "zh", "eng"])
        self.assertEqual(tracks[1].mime, "application/x-ass")

    def test_subtitle_discovery_respects_enable_setting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "movie.mp4"
            video.write_bytes(b"video")
            (root / "movie.srt").write_text("one", encoding="utf-8")
            library = MediaLibrary(build_media_roots([root]))

            tracks = find_external_subtitles(video, False, library)

        self.assertEqual(tracks, [])

    def test_subtitle_discovery_stays_inside_media_library(self) -> None:
        with tempfile.TemporaryDirectory() as video_dir, tempfile.TemporaryDirectory() as other_dir:
            root = Path(video_dir)
            other = Path(other_dir)
            video = root / "movie.mp4"
            video.write_bytes(b"video")
            outside_subtitle = other / "movie.srt"
            outside_subtitle.write_text("subtitle", encoding="utf-8")
            library = MediaLibrary(build_media_roots([root]))

            self.assertEqual(find_external_subtitles(video, True, library), [])

    def test_subtitle_mime_defaults(self) -> None:
        self.assertEqual(subtitle_mime(Path("movie.srt")), "application/x-subrip")
        self.assertEqual(subtitle_mime(Path("movie.vtt")), "text/vtt")
        self.assertEqual(subtitle_mime(Path("movie.unknown")), "text/plain")


if __name__ == "__main__":
    unittest.main()
