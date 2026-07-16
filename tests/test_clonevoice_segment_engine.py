"""CloneTranscriber reuses the shared collect pipeline (2026-07-15).

The dubbing transcriber must inherit word anchoring, pause/silence splitting,
tail extension and the acoustic hallucination checks from tool_subtitle's
SubtitleGenerator, while keeping acoustic (non-readability) timing and
language-agnostic text handling.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tool_clonevoice.segment_engine import CloneTranscriber


def make_transcriber(language=None) -> CloneTranscriber:
    transcriber = CloneTranscriber.__new__(CloneTranscriber)
    transcriber.log_callback = lambda _msg: None
    transcriber.model_preset = "kotoba"
    transcriber.language = language
    transcriber.last_speech_probs = None
    transcriber.set_vad_sensitivity("high")
    return transcriber


def word(start, end, text):
    return SimpleNamespace(start=start, end=end, word=text)


def segment(text, start, end, words=None, lp=-0.2, ns=0.05):
    return SimpleNamespace(text=text, start=start, end=end, words=words,
                           avg_logprob=lp, no_speech_prob=ns)


def audio_with_speech(duration, spans, sampling_rate=16000):
    audio = np.zeros(int(duration * sampling_rate), dtype=np.float32)
    for lo, hi in spans:
        n = int(hi * sampling_rate) - int(lo * sampling_rate)
        tone = 0.1 * np.sin(np.linspace(0, 440 * 2 * np.pi * (hi - lo), n))
        audio[int(lo * sampling_rate):int(lo * sampling_rate) + n] = tone
    return audio


class CleanTextOverrideTests(unittest.TestCase):
    def test_inner_spaces_survive_for_non_japanese(self) -> None:
        self.assertEqual(CloneTranscriber.clean_text("  Hello there  "), "Hello there")


class TranscribeWordsPipelineTests(unittest.TestCase):
    def run_transcribe(self, segments, audio, offset=100.0):
        transcriber = make_transcriber()
        transcriber.model = SimpleNamespace(
            transcribe=lambda *_a, **_k: (segments, SimpleNamespace(language="ja")),
        )
        chunk = {"array": audio, "offset_sec": offset,
                 "duration_sec": len(audio) / 16000.0}
        with patch.object(CloneTranscriber, "_split_chunks", return_value=[chunk]):
            return transcriber.transcribe_words("dummy.wav")

    def test_entries_carry_absolute_word_times_and_split_at_pause(self) -> None:
        audio = audio_with_speech(6.0, [(0.5, 1.0), (2.5, 3.5)])
        segments = [segment("はいそうです", 0.0, 3.6, words=[
            word(0.5, 1.0, "はい"),
            word(2.5, 3.5, "そうです"),
        ])]
        result = self.run_transcribe(segments, audio)

        self.assertEqual(result["language"], "ja")
        lines = result["segments"]
        self.assertEqual(len(lines), 2)  # 1.5s pause -> two dub lines
        self.assertEqual(lines[0]["text"], "はい")
        self.assertEqual(lines[1]["text"], "そうです")
        self.assertAlmostEqual(lines[0]["words"][0]["start"], 100.5)
        self.assertAlmostEqual(lines[1]["words"][0]["start"], 102.5)
        # tail extension keeps the audible decay inside the dub window
        self.assertGreaterEqual(lines[1]["end"], 103.5)

    def test_quiet_hallucination_is_dropped(self) -> None:
        audio = audio_with_speech(6.0, [(0.5, 1.5)])
        segments = [
            segment("こんにちは", 0.5, 1.5, words=[word(0.5, 1.5, "こんにちは")]),
            # decoded over silence at 3.0-4.0: quiet peak + high no_speech
            segment("また会おうね", 3.0, 4.0, ns=0.9,
                    words=[word(3.0, 4.0, "また会おうね")]),
        ]
        result = self.run_transcribe(segments, audio)
        lines = result["segments"]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["text"], "こんにちは")

    def test_no_duration_remap_is_applied(self) -> None:
        # A short word-anchored line keeps its acoustic duration; the subtitle
        # pipeline would stretch it to the readability minimum.
        audio = audio_with_speech(6.0, [(1.0, 1.4)])
        segments = [segment("はい", 1.0, 1.4, words=[word(1.0, 1.4, "はい")])]
        lines = self.run_transcribe(segments, audio)["segments"]
        self.assertEqual(len(lines), 1)
        self.assertLess(lines[0]["end"] - lines[0]["start"], 0.7)


class FilterAcousticTests(unittest.TestCase):
    def test_filter_drops_acoustic_hallucination_directly(self) -> None:
        transcriber = make_transcriber()
        items = [
            {"start": 10.0, "end": 11.0, "text": "こんにちは", "anchored": True,
             "words": [], "avg_logprob": -0.2, "no_speech_prob": 0.05,
             "peak_db": -20.0, "speech_coverage": 0.9},
            {"start": 20.0, "end": 21.0, "text": "また会おうね", "anchored": True,
             "words": [], "avg_logprob": -0.3, "no_speech_prob": 0.9,
             "peak_db": -50.0, "speech_coverage": 0.8},
        ]
        kept = transcriber._filter_keep_timing(items)
        self.assertEqual([k["text"] for k in kept], ["こんにちは"])


class SplitOnWordGapsBoundsTests(unittest.TestCase):
    """Manifest assembly must not clamp energy-verified bounds back to the
    word extent (SI_TEST_2 lines 1/2/11 lost their tail-extended speech)."""

    @staticmethod
    def words(*spans):
        return [{"w": f"w{i}", "start": s, "end": e} for i, (s, e) in enumerate(spans)]

    def test_tail_extended_end_survives(self) -> None:
        from tool_clonevoice.logic import _split_on_word_gaps
        subs = _split_on_word_gaps(11.90, 13.10, "早いね", self.words((11.90, 12.58)))
        self.assertEqual(len(subs), 1)
        self.assertAlmostEqual(subs[0]["end"], 13.10)
        self.assertAlmostEqual(subs[0]["start"], 11.90)

    def test_wildly_stray_bounds_fall_back_to_word_extent(self) -> None:
        from tool_clonevoice.logic import _split_on_word_gaps
        subs = _split_on_word_gaps(5.0, 30.0, "text", self.words((11.90, 12.58)))
        self.assertEqual(len(subs), 1)
        self.assertAlmostEqual(subs[0]["start"], 11.90)
        self.assertAlmostEqual(subs[0]["end"], 12.58)

    def test_no_words_passthrough(self) -> None:
        from tool_clonevoice.logic import _split_on_word_gaps
        subs = _split_on_word_gaps(1.0, 2.0, "text", [])
        self.assertEqual(subs, [{"start": 1.0, "end": 2.0, "text": "text", "words": []}])


if __name__ == "__main__":
    unittest.main()
