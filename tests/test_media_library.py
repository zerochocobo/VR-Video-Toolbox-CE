from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tool_dlna.media_library import MediaLibrary, build_media_roots, parse_video_dirs


class MediaLibraryTests(unittest.TestCase):
    def test_parse_pipe_separated_video_dirs(self) -> None:
        roots = parse_video_dirs(r"D:\VR|E:\VR", Path("videos"))

        self.assertEqual(len(roots), 2)
        self.assertTrue(str(roots[0]).endswith("D:\\VR"))
        self.assertTrue(str(roots[1]).endswith("E:\\VR"))

    def test_duplicate_names_are_numbered(self) -> None:
        roots = build_media_roots([Path(r"D:\VR"), Path(r"E:\VR"), Path(r"F:\Movies")])

        self.assertEqual([root.label for root in roots], ["VR", "VR2", "Movies"])

    def test_multi_root_virtual_key_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            root1 = Path(d1)
            root2 = Path(d2)
            roots = build_media_roots([root1, root2])
            library = MediaLibrary(roots)
            video = root2 / "demo.mp4"
            video.write_bytes(b"video")

            key = library.path_to_key(video)
            self.assertEqual(key, f"{root2.name}/demo.mp4")
            self.assertEqual(library.key_to_path(key), video.resolve())

    def test_nested_multi_root_prefers_most_specific_root_key(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "Downloads"
            nested = root / "VR" / "VR110"
            nested.mkdir(parents=True)
            video = nested / "demo.mp4"
            video.write_bytes(b"video")
            library = MediaLibrary(build_media_roots([root, nested]))

            key = library.path_to_key(video)

            self.assertEqual(key, "VR110/demo.mp4")
            self.assertEqual(library.key_to_path(key), video.resolve())
            self.assertIsNone(library.key_to_path("VR110/../outside.mp4"))

    def test_key_to_path_rejects_absolute_key(self) -> None:
        library = MediaLibrary(build_media_roots([Path(r"D:\VR")]))

        self.assertIsNone(library.key_to_path(r"C:\Windows\notepad.exe"))

    def test_key_to_path_rejects_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            library = MediaLibrary(build_media_roots([root]))

            self.assertIsNone(library.key_to_path("../outside.mp4"))

    def test_multi_root_key_to_path_rejects_absolute_rest_and_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            root1 = Path(d1)
            root2 = Path(d2)
            library = MediaLibrary(build_media_roots([root1, root2]))
            label = library.roots[1].label

            self.assertIsNone(library.key_to_path(f"{label}/C:/Windows/notepad.exe"))
            self.assertIsNone(library.key_to_path(f"{label}/../outside.mp4"))


if __name__ == "__main__":
    unittest.main()
