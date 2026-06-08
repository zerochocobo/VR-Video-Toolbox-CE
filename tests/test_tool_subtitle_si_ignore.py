from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tool_subtitle import logic


class FakeSubtitleGenerator:
    def __init__(self) -> None:
        self.log_callback = None
        self.transcribed: list[tuple[str, str]] = []

    def transcribe(self, audio_path: str, output_path: str) -> None:
        self.transcribed.append((audio_path, output_path))


class ToolSubtitleSISidecarIgnoreTests(unittest.TestCase):
    def test_si_sidecar_media_detection(self) -> None:
        self.assertTrue(logic.is_si_sidecar_media_file("movie.si.wav"))
        self.assertTrue(logic.is_si_sidecar_media_file("movie.SI.MP4"))
        self.assertFalse(logic.is_si_sidecar_media_file("movie.wav"))
        self.assertFalse(logic.is_si_sidecar_media_file("movie.mp4"))

        self.assertFalse(logic.is_supported_source_media_file("movie.si.wav"))
        self.assertFalse(logic.is_supported_source_media_file("movie.si.mp4"))
        self.assertTrue(logic.is_supported_source_media_file("movie.wav"))
        self.assertTrue(logic.is_supported_source_media_file("movie.mp4"))

    def test_batch_add_srt_ignores_si_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "movie.mp4").write_bytes(b"video")
            (root / "movie.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nA\n", encoding="utf-8")
            (root / "movie.si.mp4").write_bytes(b"si video")
            (root / "movie.si.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nSI\n", encoding="utf-8")
            commands: list[list[str]] = []

            with patch.object(logic, "check_ffmpeg", return_value=True), patch.object(
                logic, "run_process", side_effect=lambda cmd, *_args, **_kwargs: commands.append(cmd)
            ):
                ok = logic.batch_add_srt(str(root), search_subdirs=False)

        self.assertTrue(ok)
        self.assertEqual(len(commands), 1)
        self.assertIn(str(root / "movie.mp4"), commands[0])
        self.assertNotIn(str(root / "movie.si.mp4"), commands[0])

    def test_batch_generate_srt_ignores_si_wav_and_si_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "movie.wav").write_bytes(b"audio")
            (root / "movie.si.wav").write_bytes(b"si audio")
            (root / "clip.si.mp4").write_bytes(b"si video")
            fake_generator = FakeSubtitleGenerator()
            cache_key = ("large-v3", False)
            old_cache = dict(logic._generator_cache)
            logic._generator_cache.clear()
            logic._generator_cache[cache_key] = fake_generator
            extracted: list[tuple[str, str]] = []

            try:
                with patch.object(logic, "check_ffmpeg", return_value=True), patch.object(
                    logic, "check_model_files", return_value=True
                ), patch.object(
                    logic,
                    "extract_audio",
                    side_effect=lambda src, dst, *_args, **_kwargs: extracted.append((src, dst)) or True,
                ):
                    ok = logic.batch_generate_srt(
                        base_dir=str(root),
                        search_subdirs=False,
                        skip_if_exists=False,
                        denoise_preset="none",
                        model_key="large-v3",
                        models_root=str(root),
                        use_gpu=False,
                        log_callback=lambda _message: None,
                    )
            finally:
                logic._generator_cache.clear()
                logic._generator_cache.update(old_cache)

        self.assertTrue(ok)
        self.assertEqual([Path(src).name for src, _dst in extracted], ["movie.wav"])
        self.assertEqual([Path(audio).name for audio, _srt in fake_generator.transcribed], ["movie.asr.wav"])

    def test_srt_to_ass_ignores_si_mp4_when_finding_video_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "config"
            config_dir.mkdir()
            (config_dir / "subtitle_ass_templates.txt").write_text(
                "[Script Info]\nPlayResX: {width}\nPlayResY: {height}\n"
                "[V4+ Styles]\n"
                "Style: Default,Arial,{cn_size},{DefaultPrimaryColour},{DefaultOutlineColour}\n"
                "Style: Secondary,Arial,{jp_size},{SecondaryPrimaryColour},{SecondaryOutlineColour}\n"
                "[Events]\n",
                encoding="utf-8",
            )
            (root / "movie.si.srt").write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nこんにちは\n",
                encoding="utf-8",
            )
            (root / "movie.si.mp4").write_bytes(b"si video")

            with patch.object(logic, "get_config_dir", return_value=str(config_dir)), patch.object(
                logic, "_get_video_resolution", return_value=(1280, 720)
            ) as get_resolution:
                logic.batch_convert_srt_to_ass(
                    base_dir=str(root),
                    alignment=2,
                    base_cn_size=42,
                    base_jp_size=30,
                    search_subdirs=False,
                    skip_exists=False,
                    only_bilingual=False,
                    log_callback=lambda _message: None,
                    stop_event=None,
                )

            get_resolution.assert_not_called()
            ass_text = (root / "movie.si.ass").read_text(encoding="utf-8")
            self.assertIn("PlayResX: 1920", ass_text)
            self.assertIn("PlayResY: 1080", ass_text)


if __name__ == "__main__":
    unittest.main()
