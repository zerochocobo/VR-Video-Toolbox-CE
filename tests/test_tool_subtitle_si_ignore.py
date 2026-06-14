from __future__ import annotations

import re
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


class FakeListenSubtitleGenerator(FakeSubtitleGenerator):
    def transcribe(self, audio_path: str, output_path: str) -> None:
        super().transcribe(audio_path, output_path)
        Path(output_path).write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nこんにちは\n",
            encoding="utf-8",
        )


class FakeLLMClient:
    def __init__(self, *_args, **_kwargs) -> None:
        self.input_tokens = 0
        self.output_tokens = 0

    def complete(self, prompt: str) -> str:
        self.input_tokens += len(prompt)
        tags = re.findall(r"<(\d+)>(.*?)</\d+>", prompt, flags=re.DOTALL)
        self.output_tokens += len(tags)
        return "\n".join(f"<{tag_id}>translated {idx}</{tag_id}>" for idx, (tag_id, _text) in enumerate(tags, start=1))


class ToolSubtitleSISidecarIgnoreTests(unittest.TestCase):
    def test_load_prompt_template_can_use_dubbing_prompt_with_bom(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "translate_prompt.txt").write_text(
                "normal {target_language}\n{subtitles}\n",
                encoding="utf-8",
            )
            (root / "translate_prompt_dubbing.txt").write_text(
                "\ufeffdubbing {target_language}\n{subtitles}\n",
                encoding="utf-8",
            )

            with patch.object(logic, "get_config_dir", return_value=str(root)):
                normal_template = logic._load_prompt_template(adult_content=True, dubbing_optimized=False)
                dubbing_template = logic._load_prompt_template(adult_content=True, dubbing_optimized=True)

        self.assertTrue(normal_template.startswith("normal"))
        self.assertTrue(dubbing_template.startswith("dubbing"))
        self.assertNotIn("\ufeff", dubbing_template)

    def test_si_sidecar_media_detection(self) -> None:
        self.assertTrue(logic.is_si_sidecar_media_file("movie.si.wav"))
        self.assertTrue(logic.is_si_sidecar_media_file("movie.SI.MP4"))
        self.assertFalse(logic.is_si_sidecar_media_file("movie.wav"))
        self.assertFalse(logic.is_si_sidecar_media_file("movie.mp4"))
        self.assertTrue(logic.is_generated_output_mp4("movie_SI.mp4"))
        self.assertTrue(logic.is_generated_output_mp4("movie_DUB.MP4"))
        self.assertFalse(logic.is_generated_output_mp4("movie.mp4"))

        self.assertFalse(logic.is_supported_source_media_file("movie.si.wav"))
        self.assertFalse(logic.is_supported_source_media_file("movie.si.mp4"))
        self.assertFalse(logic.is_supported_source_media_file("movie_SI.mp4"))
        self.assertFalse(logic.is_supported_source_media_file("movie_DUB.mp4"))
        self.assertTrue(logic.is_supported_source_media_file("movie.wav"))
        self.assertTrue(logic.is_supported_source_media_file("movie.mp4"))

    def test_batch_add_srt_ignores_si_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "movie.mp4").write_bytes(b"video")
            (root / "movie.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nA\n", encoding="utf-8")
            (root / "movie.si.mp4").write_bytes(b"si video")
            (root / "movie.si.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nSI\n", encoding="utf-8")
            (root / "movie_SI.mp4").write_bytes(b"si output")
            (root / "movie_SI.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nSI\n", encoding="utf-8")
            (root / "movie_DUB.mp4").write_bytes(b"dub output")
            (root / "movie_DUB.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nDUB\n", encoding="utf-8")
            commands: list[list[str]] = []

            with patch.object(logic, "check_ffmpeg", return_value=True), patch.object(
                logic, "run_process", side_effect=lambda cmd, *_args, **_kwargs: commands.append(cmd)
            ):
                ok = logic.batch_add_srt(str(root), search_subdirs=False)

        self.assertTrue(ok)
        self.assertEqual(len(commands), 1)
        self.assertIn(str(root / "movie.mp4"), commands[0])
        self.assertNotIn(str(root / "movie.si.mp4"), commands[0])
        self.assertNotIn(str(root / "movie_SI.mp4"), commands[0])
        self.assertNotIn(str(root / "movie_DUB.mp4"), commands[0])

    def test_batch_generate_srt_ignores_si_wav_and_si_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "movie.wav").write_bytes(b"audio")
            (root / "movie.si.wav").write_bytes(b"si audio")
            (root / "clip.si.mp4").write_bytes(b"si video")
            (root / "movie_SI.mp4").write_bytes(b"si output")
            (root / "movie_DUB.mp4").write_bytes(b"dub output")
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

    def test_batch_listen_translate_processes_video_only_and_removes_jp_srt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "movie.mp4").write_bytes(b"video")
            (root / "movie.si.mp4").write_bytes(b"si video")
            (root / "audio.wav").write_bytes(b"audio")
            (root / "done.mp4").write_bytes(b"done video")
            (root / "done.srt").write_text("already translated", encoding="utf-8")
            fake_generator = FakeListenSubtitleGenerator()
            cache_key = ("large-v3", False)
            old_cache = dict(logic._generator_cache)
            logic._generator_cache.clear()
            logic._generator_cache[cache_key] = fake_generator
            extracted: list[tuple[str, str]] = []
            config = dict(logic.DEFAULT_TRANS_CONFIG)
            config["keep_original"] = False

            try:
                with patch.object(logic, "check_ffmpeg", return_value=True), patch.object(
                    logic, "check_model_files", return_value=True
                ), patch.object(
                    logic,
                    "extract_audio",
                    side_effect=lambda src, dst, *_args, **_kwargs: extracted.append((src, dst)) or True,
                ), patch.object(logic, "LLMClient", FakeLLMClient):
                    ok = logic.batch_listen_translate_srt(
                        base_dir=str(root),
                        search_subdirs=False,
                        skip_if_translated=True,
                        keep_jp_srt=False,
                        denoise_preset="none",
                        model_key="large-v3",
                        models_root=str(root),
                        use_gpu=False,
                        api_key="test-key",
                        config=config,
                        log_callback=lambda _message: None,
                    )
            finally:
                logic._generator_cache.clear()
                logic._generator_cache.update(old_cache)

            self.assertTrue(ok)
            self.assertEqual([Path(src).name for src, _dst in extracted], ["movie.mp4"])
            self.assertEqual([Path(audio).name for audio, _srt in fake_generator.transcribed], ["movie.asr.wav"])
            self.assertTrue((root / "movie.srt").exists())
            self.assertFalse((root / "movie.jp.srt").exists())
            self.assertIn("translated", (root / "movie.srt").read_text(encoding="utf-8"))
            self.assertEqual((root / "done.srt").read_text(encoding="utf-8"), "already translated")

    def test_batch_listen_translate_uses_existing_jp_srt_without_asr_and_keeps_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "movie.mp4").write_bytes(b"video")
            (root / "movie.jp.srt").write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nこんにちは\n",
                encoding="utf-8",
            )
            config = dict(logic.DEFAULT_TRANS_CONFIG)
            config["keep_original"] = False

            with patch.object(logic, "check_ffmpeg", return_value=True), patch.object(
                logic, "check_model_files", side_effect=AssertionError("ASR model should not be checked")
            ), patch.object(
                logic, "extract_audio", side_effect=AssertionError("audio should not be extracted")
            ), patch.object(logic, "LLMClient", FakeLLMClient):
                ok = logic.batch_listen_translate_srt(
                    base_dir=str(root),
                    search_subdirs=False,
                    skip_if_translated=True,
                    keep_jp_srt=True,
                    denoise_preset="none",
                    model_key="large-v3",
                    models_root=str(root),
                    use_gpu=False,
                    api_key="test-key",
                    config=config,
                    log_callback=lambda _message: None,
                )

            self.assertTrue(ok)
            self.assertTrue((root / "movie.jp.srt").exists())
            self.assertIn("translated", (root / "movie.srt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
