from __future__ import annotations

import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
