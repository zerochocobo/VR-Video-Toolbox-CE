from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gpu_engine import mux


class MuxPathTests(unittest.TestCase):
    def test_faststart_policy_auto_disables_large_outputs(self) -> None:
        self.assertTrue(mux.should_use_faststart(1024, "auto"))
        self.assertFalse(mux.should_use_faststart(5 * 1024 * 1024 * 1024, "auto"))
        self.assertTrue(mux.should_use_faststart(5 * 1024 * 1024 * 1024, "always"))
        self.assertFalse(mux.should_use_faststart(1024, "off"))

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
                patch("gpu_engine.mux._has_audio_stream", return_value=True),
            ):
                mux.mux_hevc_with_audio(raw_hevc, target, fps=30.0, audio_source=audio)

            self.assertIn(str(audio), seen_cmds[0])
            self.assertEqual(seen_cmds[0][-1], str(target))
            self.assertEqual(target.read_bytes(), b"mp4")

    def test_mux_restores_source_audio_when_first_mux_drops_it(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            raw_hevc = root / "video.hevc"
            raw_hevc.write_bytes(b"hevc")
            audio = root / "source.mp4"
            audio.write_bytes(b"audio")
            target = root / "out.mp4"
            seen_cmds: list[list[str]] = []

            def fake_run(cmd, **_kwargs):
                seen_cmds.append(cmd)
                Path(cmd[-1]).write_bytes(f"mp4-{len(seen_cmds)}".encode("ascii"))
                return subprocess.CompletedProcess(cmd, 0, "")

            with (
                patch("gpu_engine.mux.shutil.which", return_value="ffmpeg"),
                patch("gpu_engine.mux.subprocess.run", side_effect=fake_run),
                patch("gpu_engine.mux._has_audio_stream", side_effect=[False, True, True]),
            ):
                mux.mux_hevc_with_audio(raw_hevc, target, fps=30.0, audio_source=audio)

            self.assertEqual(len(seen_cmds), 2)
            self.assertIn("-shortest", seen_cmds[0])
            self.assertNotIn("-shortest", seen_cmds[1])
            self.assertIn("1:a:0", seen_cmds[1])
            self.assertNotIn("1:a:0?", seen_cmds[1])
            self.assertEqual(target.read_bytes(), b"mp4-2")


if __name__ == "__main__":
    unittest.main()
