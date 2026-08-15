from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import build_exe
import main
from utils import app_config


class ReleaseReadmeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_cache = dict(app_config._cache)

    def tearDown(self) -> None:
        app_config._cache = self._old_cache

    def test_release_readme_filename_by_language(self) -> None:
        expected = {
            "en": "readme.txt",
            "zh": "说明.txt",
            "ja": "readme_ja.txt",
        }
        for language, filename in expected.items():
            app_config._cache = {"language": language}
            self.assertEqual(main.get_release_readme_filename(), filename)
            self.assertTrue((Path("release_readme") / filename).exists(), filename)

    def test_packaging_copies_all_release_readmes_beside_exe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "release_readme"
            source_dir.mkdir()
            dist = root / "dist" / build_exe.APP_NAME
            for filename in build_exe.RELEASE_README_NAMES:
                (source_dir / filename).write_text(f"guide:{filename}", encoding="utf-8")

            with patch.object(build_exe, "ROOT", root), patch.object(build_exe, "DIST", dist):
                build_exe.seed_release_readmes()

            for filename in build_exe.RELEASE_README_NAMES:
                self.assertEqual((dist / filename).read_text(encoding="utf-8"), f"guide:{filename}")

    def test_packaging_fails_when_a_release_readme_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "release_readme"
            source_dir.mkdir()
            (source_dir / "readme.txt").write_text("English", encoding="utf-8")
            dist = root / "dist" / build_exe.APP_NAME

            with patch.object(build_exe, "ROOT", root), patch.object(build_exe, "DIST", dist):
                with self.assertRaisesRegex(build_exe.BuildError, "说明.txt"):
                    build_exe.seed_release_readmes()


if __name__ == "__main__":
    unittest.main()
