from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tool_si import logic


class FakeTTSModel:
    def __init__(self, fail_batch: bool = False) -> None:
        self.fail_batch = fail_batch
        self.calls: list[dict] = []

    def generate_custom_voice(self, text, speaker, language, max_new_tokens, **kwargs):
        self.calls.append(
            {
                "text": text,
                "speaker": speaker,
                "language": language,
                "max_new_tokens": max_new_tokens,
                **kwargs,
            }
        )
        if isinstance(text, list):
            if self.fail_batch:
                raise RuntimeError("batch failed")
            return [np.ones(12, dtype=np.float32) * (idx + 1) * 0.1 for idx, _ in enumerate(text)], 100
        return [np.ones(12, dtype=np.float32) * 0.2], 100


class SimultaneousInterpretationLogicTests(unittest.TestCase):
    def test_parse_srt_uses_first_non_empty_subtitle_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            srt_path = Path(tmp_dir) / "sample.srt"
            srt_path.write_text(
                "\ufeff1\n"
                "00:00:01,000 --> 00:00:03,500\n"
                "こんにちは\n"
                "Hello\n\n"
                "2\n"
                "00:00:04,000 --> 00:00:05,000\n"
                "{\\an8}<i>次の字幕</i>\n",
                encoding="utf-8",
            )

            entries = logic.parse_srt(srt_path)

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].text, "こんにちは")
        self.assertEqual(entries[1].text, "次の字幕")
        self.assertAlmostEqual(entries[0].start, 1.0)
        self.assertAlmostEqual(entries[0].end, 3.5)

    def test_parse_srt_selects_text_line_for_requested_language(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            srt_path = Path(tmp_dir) / "sample.srt"
            srt_path.write_text(
                "1\n"
                "00:00:01,000 --> 00:00:03,000\n"
                "最近肩膀疼\n"
                "最近肩が痛くて\n",
                encoding="utf-8",
            )

            chinese_entries = logic.parse_srt(srt_path, language="Chinese")
            japanese_entries = logic.parse_srt(srt_path, language="Japanese")

        self.assertEqual(chinese_entries[0].text, "最近肩膀疼")
        self.assertEqual(japanese_entries[0].text, "最近肩が痛くて")

    def test_collect_paired_srt_tasks_finds_srt_matching_mp4_or_mkv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "a.mp4").write_bytes(b"")
            (root / "a.srt").write_text("x", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir()
            (nested / "b.mkv").write_bytes(b"")
            (nested / "b.srt").write_text("x", encoding="utf-8")
            (nested / "ignored.srt").write_text("x", encoding="utf-8")

            tasks = logic.collect_paired_srt_tasks(root)

        self.assertEqual([path.name for path in tasks], ["a.srt", "b.srt"])

    def test_fit_audio_to_duration_pads_short_audio_to_target_samples(self) -> None:
        audio = np.ones(100, dtype=np.float32) * 0.5

        fitted = logic.fit_audio_to_duration(audio, sample_rate=1000, target_duration=0.2)

        self.assertEqual(fitted.shape[0], 200)
        self.assertTrue(np.allclose(fitted[:100], 0.5))
        self.assertTrue(np.allclose(fitted[100:], 0.0))

    def test_resolve_tts_batch_size_clamps_env_value(self) -> None:
        with patch.dict("os.environ", {logic.TTS_BATCH_SIZE_ENV: "99"}):
            self.assertEqual(logic.resolve_tts_batch_size(), logic.MAX_TTS_BATCH_SIZE)
        with patch.dict("os.environ", {logic.TTS_BATCH_SIZE_ENV: "bad"}):
            self.assertEqual(logic.resolve_tts_batch_size(), logic.DEFAULT_TTS_BATCH_SIZE)

    def test_resolve_tts_batch_token_spread_clamps_env_value(self) -> None:
        with patch.dict("os.environ", {logic.TTS_BATCH_TOKEN_SPREAD_ENV: "99"}):
            self.assertEqual(logic.resolve_tts_batch_token_spread(), 10.0)
        with patch.dict("os.environ", {logic.TTS_BATCH_TOKEN_SPREAD_ENV: "bad"}):
            self.assertEqual(logic.resolve_tts_batch_token_spread(), logic.DEFAULT_TTS_BATCH_TOKEN_SPREAD)

    def test_iter_tts_batches_groups_short_entries(self) -> None:
        entries = [
            logic.SubtitleEntry(index=1, start=0.0, end=1.0, text="a"),
            logic.SubtitleEntry(index=2, start=1.0, end=2.0, text="b"),
            logic.SubtitleEntry(index=3, start=2.0, end=3.0, text="c"),
            logic.SubtitleEntry(index=4, start=3.0, end=4.0, text="d"),
            logic.SubtitleEntry(index=5, start=4.0, end=5.0, text="e"),
        ]

        batches = logic._iter_tts_batches(entries, batch_size=2)

        self.assertEqual([[entry.index for entry in batch] for batch in batches], [[1, 2], [3, 4], [5]])

    def test_subtitle_to_audio_uses_batch_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            srt_path = root / "sample.srt"
            output_path = root / "out.wav"
            srt_path.write_text(
                "1\n00:00:00,000 --> 00:00:00,100\nA\n\n"
                "2\n00:00:00,100 --> 00:00:00,200\nB\n",
                encoding="utf-8",
            )
            model_dir = root / logic.MODEL_DIR_NAME
            model_dir.mkdir()
            fake_model = FakeTTSModel()

            with patch.object(logic, "check_model_files", return_value=True), patch.dict(
                "os.environ", {logic.TTS_BATCH_SIZE_ENV: "2"}
            ):
                result = logic.subtitle_to_audio(
                    srt_path=srt_path,
                    output_path=output_path,
                    language="Chinese",
                    speaker="Vivian",
                    models_root=root,
                    log_callback=lambda _message: None,
                    tts_model=fake_model,
                )

            self.assertEqual(result, str(output_path))
            self.assertTrue(output_path.exists())
            self.assertEqual(len(fake_model.calls), 1)
            self.assertEqual(fake_model.calls[0]["text"], ["A", "B"])

    def test_subtitle_to_audio_can_limit_test_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            srt_path = root / "sample.srt"
            output_path = root / "out.wav"
            srt_path.write_text(
                "1\n00:00:00,000 --> 00:00:00,100\nA\n\n"
                "2\n00:00:00,100 --> 00:00:00,200\nB\n\n"
                "3\n00:00:00,200 --> 00:00:00,300\nC\n",
                encoding="utf-8",
            )
            (root / logic.MODEL_DIR_NAME).mkdir()
            fake_model = FakeTTSModel()
            messages: list[str] = []

            with patch.object(logic, "check_model_files", return_value=True), patch.dict(
                "os.environ", {logic.TTS_BATCH_SIZE_ENV: "4"}
            ):
                logic.subtitle_to_audio(
                    srt_path=srt_path,
                    output_path=output_path,
                    language="Chinese",
                    speaker="Vivian",
                    models_root=root,
                    log_callback=messages.append,
                    tts_model=fake_model,
                    max_entries=2,
                )

            self.assertEqual(len(fake_model.calls), 1)
            self.assertEqual(fake_model.calls[0]["text"], ["A", "B"])
            self.assertTrue(any("converting first 2 for test" in message for message in messages))

    def test_subtitle_to_audio_preserves_srt_order_when_limited(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            srt_path = root / "sample.srt"
            output_path = root / "out.wav"
            long_text = "长" * 80
            srt_path.write_text(
                "1\n00:00:00,000 --> 00:00:00,100\nA\n\n"
                f"2\n00:00:00,100 --> 00:00:03,100\n{long_text}\n\n"
                "3\n00:00:03,100 --> 00:00:03,200\nC\n",
                encoding="utf-8",
            )
            (root / logic.MODEL_DIR_NAME).mkdir()
            fake_model = FakeTTSModel()

            with patch.object(logic, "check_model_files", return_value=True), patch.dict(
                "os.environ", {logic.TTS_BATCH_SIZE_ENV: "4"}
            ):
                logic.subtitle_to_audio(
                    srt_path=srt_path,
                    output_path=output_path,
                    language="Chinese",
                    speaker="Vivian",
                    models_root=root,
                    log_callback=lambda _message: None,
                    tts_model=fake_model,
                    max_entries=3,
                )

            seen_texts: list[str] = []
            for call in fake_model.calls:
                text = call["text"]
                if isinstance(text, list):
                    seen_texts.extend(text)
                else:
                    seen_texts.append(text)
            self.assertEqual(seen_texts, ["A", long_text, "C"])

    def test_subtitle_to_audio_retries_one_by_one_when_batch_generation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            srt_path = root / "sample.srt"
            output_path = root / "out.wav"
            srt_path.write_text(
                "1\n00:00:00,000 --> 00:00:00,100\nA\n\n"
                "2\n00:00:00,100 --> 00:00:00,200\nB\n",
                encoding="utf-8",
            )
            model_dir = root / logic.MODEL_DIR_NAME
            model_dir.mkdir()
            fake_model = FakeTTSModel(fail_batch=True)

            with patch.object(logic, "check_model_files", return_value=True), patch.dict(
                "os.environ", {logic.TTS_BATCH_SIZE_ENV: "2"}
            ):
                logic.subtitle_to_audio(
                    srt_path=srt_path,
                    output_path=output_path,
                    language="Chinese",
                    speaker="Vivian",
                    models_root=root,
                    log_callback=lambda _message: None,
                    tts_model=fake_model,
                )

            self.assertEqual(len(fake_model.calls), 3)
            self.assertEqual(fake_model.calls[0]["text"], ["A", "B"])
            self.assertEqual([call["text"] for call in fake_model.calls[1:]], ["A", "B"])

    def test_iter_tts_batches_packs_similar_length_entries_after_sorting(self) -> None:
        # Interleave short / long entries the way a real SRT does. With pre-sort the
        # short ones cluster together and the batch packer can fill `batch_size` cleanly
        # instead of breaking at every length spike.
        entries = [
            logic.SubtitleEntry(index=1, start=0.0, end=0.2, text="a"),     # very short
            logic.SubtitleEntry(index=2, start=1.0, end=11.0, text="x" * 80),  # very long
            logic.SubtitleEntry(index=3, start=12.0, end=12.2, text="b"),
            logic.SubtitleEntry(index=4, start=13.0, end=23.0, text="y" * 80),
            logic.SubtitleEntry(index=5, start=24.0, end=24.2, text="c"),
        ]

        batches = logic._iter_tts_batches(entries, batch_size=4)

        # The two long entries must not share a batch with the short ones (token_spread
        # would otherwise force every short entry to decode out to ~120 tokens).
        for batch in batches:
            budgets = [
                logic._max_new_tokens_for_duration(e.duration, e.text) for e in batch
            ]
            self.assertLessEqual(max(budgets), min(budgets) * logic.resolve_tts_batch_token_spread() + 1)

        # Shorts should pack into a single batch (sort-then-pack), not split across many.
        short_indices = {1, 3, 5}
        short_batches = [b for b in batches if {e.index for e in b} & short_indices]
        self.assertEqual(len(short_batches), 1,
                         f"shorts split across batches: {[[e.index for e in b] for b in batches]}")
        self.assertEqual({e.index for e in short_batches[0]}, short_indices)

    def test_subtitle_to_audio_places_entries_by_start_time_independent_of_batch_order(self) -> None:
        # Pre-sort means batch generation order differs from SRT order. Output WAV must
        # still place each entry at its `entry.start`. We verify the timeline contains a
        # non-zero signal at each entry's start window — regardless of which batch the
        # entry was generated in.
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            srt_path = root / "interleaved.srt"
            output_path = root / "out.wav"
            srt_path.write_text(
                # Short / long / short / long / short interleaved — forces the sort path
                # to reorder for batching. Entry text length controls token budget.
                "1\n00:00:01,000 --> 00:00:01,500\nA\n\n"
                "2\n00:00:02,000 --> 00:00:08,000\n" + ("X " * 30).strip() + "\n\n"
                "3\n00:00:09,000 --> 00:00:09,500\nC\n",
                encoding="utf-8",
            )
            model_dir = root / logic.MODEL_DIR_NAME
            model_dir.mkdir()
            fake_model = FakeTTSModel()

            with patch.object(logic, "check_model_files", return_value=True), patch.dict(
                "os.environ", {logic.TTS_BATCH_SIZE_ENV: "4"}
            ):
                logic.subtitle_to_audio(
                    srt_path=srt_path,
                    output_path=output_path,
                    language="English",
                    speaker="Ryan",
                    models_root=root,
                    log_callback=lambda _message: None,
                    tts_model=fake_model,
                )

            audio, sr = logic.read_wav_mono(output_path)
            # All three entries should be present at their start positions.
            for entry_start in (1.0, 2.0, 9.0):
                idx = int(entry_start * sr)
                window = audio[idx : idx + int(0.05 * sr)]
                self.assertGreater(
                    float(np.abs(window).max()), 0.0,
                    f"no audio placed at start={entry_start}s — timeline placement broke after sort",
                )

    def test_subtalker_greedy_kwargs_default_to_argmax(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            os_environ_pop = logic.os.environ.pop(logic.TTS_SUBTALKER_GREEDY_ENV, None)
            try:
                kw = logic._subtalker_generate_kwargs()
                self.assertEqual(kw.get("subtalker_dosample"), False)
                self.assertEqual(kw.get("subtalker_top_k"), 1)
            finally:
                if os_environ_pop is not None:
                    logic.os.environ[logic.TTS_SUBTALKER_GREEDY_ENV] = os_environ_pop

    def test_subtalker_greedy_can_be_disabled_via_env(self) -> None:
        with patch.dict("os.environ", {logic.TTS_SUBTALKER_GREEDY_ENV: "0"}):
            self.assertEqual(logic._subtalker_generate_kwargs(), {})

    def test_fit_audio_to_duration_uses_fast_path_for_small_stretch(self) -> None:
        # Audio is ~10% too long → factor 1.1, should NOT call into librosa /
        # ffmpeg, just linearly resample.
        sample_rate = 24000
        target_duration = 1.0
        source = np.linspace(-0.5, 0.5, int(target_duration * sample_rate * 1.10), dtype=np.float32)

        with patch.object(logic, "_speed_up_in_memory") as expensive, \
             patch.object(logic, "_speed_up_with_ffmpeg") as expensive_ffmpeg:
            result = logic.fit_audio_to_duration(source, sample_rate, target_duration)

        expensive.assert_not_called()
        expensive_ffmpeg.assert_not_called()
        self.assertEqual(result.shape[0], int(target_duration * sample_rate))


if __name__ == "__main__":
    unittest.main()
