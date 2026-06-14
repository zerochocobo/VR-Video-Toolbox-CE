from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
