from __future__ import annotations

import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from gpu_engine.fallback import OperationCancelled
from gpu_engine import mux


class FakePopen:
    def __init__(self, cmd, returncode=0, output=""):
        self.cmd = cmd
        self.returncode = returncode
        self.stdout = StringIO(output)
        self.cancelled = False

    def poll(self):
        return self.returncode

    def wait(self):
        return self.returncode

    def kill(self):
        self.cancelled = True
        self.returncode = 1

    def terminate(self):
        self.kill()


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

            def fake_popen(cmd, **_kwargs):
                seen_cmds.append(cmd)
                Path(cmd[-1]).write_bytes(b"mp4")
                return FakePopen(cmd, 0, "mux ok\n")

            with (
                patch("gpu_engine.mux.shutil.which", return_value="ffmpeg"),
                patch("gpu_engine.mux.subprocess.Popen", side_effect=fake_popen),
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

            def fake_popen(cmd, **_kwargs):
                seen_cmds.append(cmd)
                Path(cmd[-1]).write_bytes(f"mp4-{len(seen_cmds)}".encode("ascii"))
                return FakePopen(cmd, 0, "mux ok\n")

            with (
                patch("gpu_engine.mux.shutil.which", return_value="ffmpeg"),
                patch("gpu_engine.mux.subprocess.Popen", side_effect=fake_popen),
                patch("gpu_engine.mux._has_audio_stream", side_effect=[False, True, True]),
            ):
                mux.mux_hevc_with_audio(raw_hevc, target, fps=30.0, audio_source=audio)

            self.assertEqual(len(seen_cmds), 2)
            self.assertIn("-shortest", seen_cmds[0])
            self.assertNotIn("-shortest", seen_cmds[1])
            self.assertIn("1:a:0", seen_cmds[1])
            self.assertNotIn("1:a:0?", seen_cmds[1])
            self.assertEqual(target.read_bytes(), b"mp4-2")

    def test_mux_process_callback_can_cancel_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            raw_hevc = root / "video.hevc"
            raw_hevc.write_bytes(b"hevc")
            target = root / "out.mp4"

            class SlowProc(FakePopen):
                def __init__(self, cmd):
                    super().__init__(cmd, None, "")

                def poll(self):
                    return self.returncode

                def wait(self):
                    return self.returncode or 1

            def fake_popen(cmd, **_kwargs):
                return SlowProc(cmd)

            def cancel_immediately(proc):
                proc.kill()

            with (
                patch("gpu_engine.mux.shutil.which", return_value="ffmpeg"),
                patch("gpu_engine.mux.subprocess.Popen", side_effect=fake_popen),
            ):
                with self.assertRaises(OperationCancelled):
                    mux.mux_hevc_with_audio(
                        raw_hevc,
                        target,
                        fps=30.0,
                        process_callback=cancel_immediately,
                    )


if __name__ == "__main__":
    unittest.main()
