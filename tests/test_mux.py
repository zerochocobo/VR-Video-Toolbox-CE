from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gpu_engine import mux


class MuxPathTests(unittest.TestCase):
    def test_mux_passes_unicode_paths_directly_and_creates_output_parent(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            raw_hevc = root / "video.hevc"
            raw_hevc.write_bytes(b"hevc")
            audio = root / "研究" / "source.mp4"
            audio.parent.mkdir()
            audio.write_bytes(b"audio")
            target = root / "输出" / "out.mp4"
            seen_cmds: list[list[str]] = []

            def fake_run(cmd, **_kwargs):
                seen_cmds.append(cmd)
                Path(cmd[-1]).write_bytes(b"mp4")
                return subprocess.CompletedProcess(cmd, 0, "")

            with (
                patch("gpu_engine.mux.shutil.which", return_value="ffmpeg"),
                patch("gpu_engine.mux.subprocess.run", side_effect=fake_run),
            ):
                mux.mux_hevc_with_audio(raw_hevc, target, fps=30.0, audio_source=audio)

            self.assertIn(str(audio), seen_cmds[0])
            self.assertEqual(seen_cmds[0][-1], str(target))
            self.assertEqual(target.read_bytes(), b"mp4")


if __name__ == "__main__":
    unittest.main()
