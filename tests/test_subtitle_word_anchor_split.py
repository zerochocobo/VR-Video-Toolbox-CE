"""Word-anchored segment timing (mdvr-433 debug session, 2026-07-14).

The decoder merged two utterances separated by a 1.4s pause into one segment
(19.2-25.6s), DTW absorbed the pause into the first word after it, and the
reading-speed cap in postprocess then cut the last ~2s of real speech. These
tests pin the three fixes: pause splitting, energy tightening of long words,
and the cap never shrinking word-anchored lines.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tool_subtitle.logic import SubtitleGenerator


def make_generator(vad_sensitivity: str = "high") -> SubtitleGenerator:
    generator = SubtitleGenerator.__new__(SubtitleGenerator)
    generator.log_callback = lambda _msg: None
    generator.model_preset = "kotoba"
    generator.last_speech_probs = None
    generator.set_vad_sensitivity(vad_sensitivity)
    return generator


def word(start: float, end: float, text: str) -> SimpleNamespace:
    return SimpleNamespace(start=start, end=end, word=text)


def segment(text: str, start: float, end: float, words=None) -> SimpleNamespace:
    return SimpleNamespace(
        text=text, start=start, end=end, words=words,
        avg_logprob=-0.2, no_speech_prob=0.05,
    )


def audio_with_speech(duration: float, spans: list, sampling_rate: int = 16000) -> np.ndarray:
    audio = np.zeros(int(duration * sampling_rate), dtype=np.float32)
    for lo, hi in spans:
        n = int(hi * sampling_rate) - int(lo * sampling_rate)
        tone = 0.1 * np.sin(np.linspace(0, 440 * 2 * np.pi * (hi - lo), n))
        audio[int(lo * sampling_rate):int(lo * sampling_rate) + n] = tone
    return audio


class TightenWordStartTests(unittest.TestCase):
    def test_leading_silence_is_trimmed(self) -> None:
        audio = audio_with_speech(4.0, [(2.4, 3.0)])
        start = SubtitleGenerator.tighten_word_start(1.0, 3.0, audio)
        self.assertAlmostEqual(start, 2.3, delta=0.06)

    def test_short_word_and_missing_audio_are_untouched(self) -> None:
        audio = audio_with_speech(2.0, [(0.9, 1.0)])
        self.assertEqual(SubtitleGenerator.tighten_word_start(0.5, 1.0, audio), 0.5)
        self.assertEqual(SubtitleGenerator.tighten_word_start(0.5, 3.0, None), 0.5)

    def test_short_lead_is_untouched(self) -> None:
        audio = audio_with_speech(4.0, [(1.2, 3.0)])
        self.assertEqual(SubtitleGenerator.tighten_word_start(1.0, 3.0, audio), 1.0)


class CollectSegmentEntriesTests(unittest.TestCase):
    def test_pause_absorbed_by_word_is_split(self) -> None:
        generator = make_generator()
        audio = audio_with_speech(4.0, [(0.5, 1.0), (2.4, 3.2)])
        seg = segment("は通常", 0.0, 3.5, words=[
            word(0.5, 1.0, "は"),
            word(1.0, 3.0, "通"),
            word(3.0, 3.2, "常"),
        ])
        entries, moved_far = generator.collect_segment_entries(seg, 10.0, chunk_audio=audio)
        self.assertEqual(len(entries), 2)
        self.assertEqual([entry["text"] for entry in entries], ["は", "通常"])
        self.assertAlmostEqual(entries[0]["start"], 10.5)
        self.assertAlmostEqual(entries[0]["end"], 11.0)
        self.assertAlmostEqual(entries[1]["start"], 12.3, delta=0.06)
        self.assertAlmostEqual(entries[1]["end"], 13.2)
        self.assertTrue(all(entry["anchored"] for entry in entries))
        self.assertEqual(moved_far, 0)  # 0.5s from the decoder stamp is not "far" (>0.5s)

    def test_explicit_word_gap_is_split_without_audio(self) -> None:
        generator = make_generator()
        seg = segment("はいいいえ", 0.0, 4.0, words=[
            word(0.2, 0.6, "はい"),
            word(2.0, 2.6, "いいえ"),
        ])
        entries, _ = generator.collect_segment_entries(seg, 0.0)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["text"], "はい")
        self.assertEqual(entries[1]["text"], "いいえ")

    def test_contiguous_words_stay_one_entry(self) -> None:
        generator = make_generator()
        seg = segment("よろしくお願いします", 0.0, 2.5, words=[
            word(0.24, 0.7, "よろしく"),
            word(0.7, 2.24, "お願いします"),
        ])
        entries, moved_far = generator.collect_segment_entries(seg, 6.02)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["text"], "よろしくお願いします")
        self.assertAlmostEqual(entries[0]["start"], 6.26)
        self.assertAlmostEqual(entries[0]["end"], 8.26)
        self.assertEqual(moved_far, 0)
        # absolute-time word list rides along (consumed by tool_clonevoice)
        self.assertEqual(len(entries[0]["words"]), 2)
        self.assertAlmostEqual(entries[0]["words"][0]["start"], 6.26)
        self.assertAlmostEqual(entries[0]["words"][1]["end"], 8.26)
        self.assertEqual(entries[0]["words"][1]["word"], "お願いします")

    def test_missing_words_fall_back_to_segment_times(self) -> None:
        generator = make_generator()
        seg = segment("本日は", 1.0, 3.0, words=None)
        entries, moved_far = generator.collect_segment_entries(seg, 10.0)
        self.assertEqual(len(entries), 1)
        self.assertAlmostEqual(entries[0]["start"], 11.0)
        self.assertAlmostEqual(entries[0]["end"], 13.0)
        self.assertFalse(entries[0]["anchored"])
        self.assertEqual(moved_far, 0)


class ReadingSpeedCapTests(unittest.TestCase):
    TEXT = "本日は通常コースでよろしかったですか"  # 18 chars -> 4.26s cap

    def test_cap_shrinks_decoder_timed_lines(self) -> None:
        generator = make_generator()
        result = generator.postprocess_segments([
            {"start": 19.24, "end": 25.64, "text": self.TEXT, "anchored": False,
             "avg_logprob": -0.23, "no_speech_prob": 0.03},
        ])
        self.assertAlmostEqual(result[0]["end"], 19.24 + 18 / 5.2 + 0.8, places=3)

    def test_cap_never_shrinks_word_anchored_lines(self) -> None:
        generator = make_generator()
        result = generator.postprocess_segments([
            {"start": 19.24, "end": 25.64, "text": self.TEXT, "anchored": True,
             "avg_logprob": -0.23, "no_speech_prob": 0.03},
        ])
        self.assertAlmostEqual(result[0]["end"], 25.64)

    def test_cap_still_extends_too_short_anchored_lines(self) -> None:
        generator = make_generator()
        result = generator.postprocess_segments([
            {"start": 5.0, "end": 5.2, "text": "はいそうですね", "anchored": True,
             "avg_logprob": -0.23, "no_speech_prob": 0.03},
        ])
        self.assertGreater(result[0]["end"], 5.2)


class AcousticHallucinationChecksTests(unittest.TestCase):
    """Postprocess kills for silence-span and no-speech-coverage lines."""

    @staticmethod
    def line(text="今日はいい天気ですね", *, anchored=True, peak_db=None, coverage=None,
             lp=-0.3, ns=0.05, start=100.0, end=102.0):
        return {"start": start, "end": end, "text": text, "anchored": anchored,
                "avg_logprob": lp, "no_speech_prob": ns,
                "peak_db": peak_db, "speech_coverage": coverage}

    def test_anchored_line_over_silence_is_removed(self) -> None:
        generator = make_generator()
        removal_log: list = []
        result = generator.postprocess_segments(
            [self.line(peak_db=-80.0)], removal_log=removal_log)
        self.assertEqual(result, [])
        self.assertEqual(removal_log[0]["reason"], "no_audio_energy")

    def test_decoder_timed_confident_line_over_silence_survives(self) -> None:
        # A mis-stamped decoder span can cover real silence; without word
        # anchoring the silence kill also needs low decoder confidence.
        generator = make_generator()
        result = generator.postprocess_segments(
            [self.line(anchored=False, peak_db=-80.0, lp=-0.2, ns=0.03)])
        self.assertEqual(len(result), 1)

    def test_quiet_whisper_above_floor_survives(self) -> None:
        # high sensitivity: rms gate -58 -> floor -63; a -57dB whisper stays.
        generator = make_generator("high")
        result = generator.postprocess_segments([self.line(peak_db=-57.0)])
        self.assertEqual(len(result), 1)

    def test_low_coverage_needs_low_confidence_too(self) -> None:
        generator = make_generator()
        confident = self.line(coverage=0.05, lp=-0.2, ns=0.03)
        unsure = self.line(coverage=0.05, lp=-1.0, ns=0.03, start=200.0, end=202.0)
        removal_log: list = []
        result = generator.postprocess_segments([confident, unsure], removal_log=removal_log)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["start"], 100.0)
        self.assertEqual(removal_log[0]["reason"], "low_speech_coverage")

    def test_good_coverage_keeps_unsure_line(self) -> None:
        generator = make_generator()
        result = generator.postprocess_segments([self.line(coverage=0.9, lp=-1.0)])
        self.assertEqual(len(result), 1)

    def test_lines_without_stats_are_untouched(self) -> None:
        generator = make_generator()
        result = generator.postprocess_segments([self.line(peak_db=None, coverage=None)])
        self.assertEqual(len(result), 1)


class AcousticStatsCollectionTests(unittest.TestCase):
    def test_entries_carry_peak_db_and_coverage(self) -> None:
        generator = make_generator()
        # 20ms whisperseg frames: speech probability 1.0 during 10.5-11.0s only.
        probs = np.zeros(2000, dtype=np.float32)
        probs[int(10.5 / 0.02):int(11.0 / 0.02)] = 1.0
        generator.last_speech_probs = probs
        audio = audio_with_speech(4.0, [(0.5, 1.0)])
        seg = segment("はい", 0.0, 3.5, words=[word(0.5, 1.0, "はい")])

        entries, _ = generator.collect_segment_entries(seg, 10.0, chunk_audio=audio)

        self.assertEqual(len(entries), 1)
        self.assertGreater(entries[0]["peak_db"], -30.0)  # 0.1 sine ~ -23dB
        self.assertGreater(entries[0]["speech_coverage"], 0.9)

    def test_stats_default_to_none_without_audio_or_probs(self) -> None:
        generator = make_generator()
        seg = segment("はい", 0.0, 1.0, words=[word(0.2, 0.8, "はい")])
        entries, _ = generator.collect_segment_entries(seg, 0.0)
        self.assertIsNone(entries[0]["peak_db"])
        self.assertIsNone(entries[0]["speech_coverage"])


class TailExtensionTests(unittest.TestCase):
    """DTW word ends are early; extend through trailing voiced audio."""

    def test_anchored_end_extends_through_voiced_tail(self) -> None:
        audio = audio_with_speech(6.0, [(1.0, 3.2)])
        entries = [{"start": 10.5, "end": 12.0, "text": "はい", "anchored": True}]
        SubtitleGenerator.extend_entry_tails(entries, audio, 9.5)
        # voiced until rel 3.2 (abs 12.7) plus the 0.12s release pad
        self.assertAlmostEqual(entries[0]["end"], 12.82, delta=0.06)

    def test_extension_capped_at_max_seconds(self) -> None:
        audio = audio_with_speech(6.0, [(1.0, 5.5)])
        entries = [{"start": 10.5, "end": 12.0, "text": "はい", "anchored": True}]
        SubtitleGenerator.extend_entry_tails(entries, audio, 9.5)
        self.assertAlmostEqual(entries[0]["end"], 13.0, delta=0.03)

    def test_extension_stops_before_next_entry(self) -> None:
        audio = audio_with_speech(6.0, [(1.0, 5.5)])
        entries = [
            {"start": 10.5, "end": 12.0, "text": "はい", "anchored": True},
            {"start": 12.5, "end": 13.5, "text": "そうです", "anchored": True},
        ]
        SubtitleGenerator.extend_entry_tails(entries, audio, 9.5)
        self.assertAlmostEqual(entries[0]["end"], 12.45, delta=0.03)
        self.assertAlmostEqual(entries[1]["start"], 12.5)

    def test_decoder_timed_and_silent_tails_untouched(self) -> None:
        audio = audio_with_speech(6.0, [(1.0, 2.4)])
        entries = [
            {"start": 10.5, "end": 12.0, "text": "はい", "anchored": False},
            {"start": 13.0, "end": 13.4, "text": "ええ", "anchored": True},
        ]
        SubtitleGenerator.extend_entry_tails(entries, audio, 9.5)
        self.assertEqual(entries[0]["end"], 12.0)  # not word-anchored
        self.assertEqual(entries[1]["end"], 13.4)  # tail already silent


class TailExtensionHangoverTests(unittest.TestCase):
    def test_extension_rides_through_glottal_dip(self) -> None:
        # 60ms dip inside the tail (mdvr-433 44.78-44.84s) must not stop the scan.
        audio = audio_with_speech(6.0, [(1.0, 2.5), (2.56, 3.2)])
        entries = [{"start": 10.5, "end": 12.0, "text": "はい", "anchored": True}]
        SubtitleGenerator.extend_entry_tails(entries, audio, 9.5)
        self.assertAlmostEqual(entries[0]["end"], 12.82, delta=0.06)

    def test_extension_still_stops_at_real_silence(self) -> None:
        # 0.5s of silence then more speech: the scan must not bridge it.
        audio = audio_with_speech(6.0, [(1.0, 2.0), (2.5, 3.5)])
        entries = [{"start": 10.5, "end": 11.4, "text": "はい", "anchored": True}]
        SubtitleGenerator.extend_entry_tails(entries, audio, 9.5)
        self.assertAlmostEqual(entries[0]["end"], 11.62, delta=0.06)


class SilenceValleySplitTests(unittest.TestCase):
    """A pause hidden across several word boundaries splits on energy."""

    def test_entry_splits_at_internal_silence_run(self) -> None:
        generator = make_generator()
        audio = audio_with_speech(4.0, [(0.5, 1.0), (2.5, 3.5)])
        # zero-gap words spanning the 1.5s silence; none long enough for
        # tighten_word_start, no inter-word gap for the pause split.
        seg = segment("ねえ生理前って", 0.0, 3.6, words=[
            word(0.5, 1.2, "ねえ"),
            word(1.2, 1.9, "生理"),
            word(1.9, 2.7, "前"),
            word(2.7, 3.5, "って"),
        ])
        entries, _ = generator.collect_segment_entries(seg, 10.0, chunk_audio=audio)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["text"], "ねえ生理")
        self.assertEqual(entries[1]["text"], "前って")
        self.assertAlmostEqual(entries[0]["start"], 10.5)
        self.assertAlmostEqual(entries[0]["end"], 11.12, delta=0.06)
        self.assertAlmostEqual(entries[1]["start"], 12.38, delta=0.06)
        self.assertAlmostEqual(entries[1]["end"], 13.5)

    def test_continuous_speech_is_not_split(self) -> None:
        generator = make_generator()
        audio = audio_with_speech(4.0, [(0.5, 3.5)])
        seg = segment("よろしくお願いします", 0.0, 3.6, words=[
            word(0.5, 1.9, "よろしく"),
            word(1.9, 3.5, "お願いします"),
        ])
        entries, _ = generator.collect_segment_entries(seg, 10.0, chunk_audio=audio)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["text"], "よろしくお願いします")


class QuietNoSpeechHallucinationTests(unittest.TestCase):
    @staticmethod
    def line(*, ns, pk, text="次の動画でまた会おうね"):
        return {"start": 329.8, "end": 331.4, "text": text, "anchored": True,
                "avg_logprob": -0.32, "no_speech_prob": ns,
                "peak_db": pk, "speech_coverage": 0.87}

    def test_quiet_high_no_speech_line_is_removed(self) -> None:
        generator = make_generator()
        removal_log: list = []
        result = generator.postprocess_segments(
            [self.line(ns=0.74, pk=-49.0)], removal_log=removal_log)
        self.assertEqual(result, [])
        self.assertEqual(removal_log[0]["reason"], "quiet_no_speech")

    def test_loud_or_confident_lines_survive(self) -> None:
        generator = make_generator()
        self.assertEqual(len(generator.postprocess_segments([self.line(ns=0.74, pk=-30.0)])), 1)
        self.assertEqual(len(generator.postprocess_segments([self.line(ns=0.2, pk=-49.0)])), 1)

    def test_next_video_outro_is_a_known_hallucination(self) -> None:
        self.assertTrue(SubtitleGenerator.is_known_hallucination(
            "次の動画でお会いしましょう", 329.8, 331.4, 2000.0))

    def test_truncated_outro_phrase_is_still_a_hallucination(self) -> None:
        # ASR dropped the leading ご (mdvr-433 #2 entry 18).
        self.assertTrue(SubtitleGenerator.is_known_hallucination(
            "視聴ありがとうございました", 750.8, 751.7, 2000.0))

    def test_plain_thanks_is_not_a_truncated_outro(self) -> None:
        # ありがとうございました is a real line, not a truncation of ご視聴….
        self.assertFalse(SubtitleGenerator.is_known_hallucination(
            "ありがとうございました", 100.0, 103.0, 2000.0,
            avg_logprob=-0.2, no_speech_prob=0.05))

    def test_very_high_no_speech_uses_relaxed_peak_threshold(self) -> None:
        generator = make_generator()
        removal_log: list = []
        result = generator.postprocess_segments(
            [self.line(ns=0.86, pk=-41.0)], removal_log=removal_log)
        self.assertEqual(result, [])
        self.assertEqual(removal_log[0]["reason"], "quiet_no_speech")
        # A -30dB peak is real speech territory even at high ns.
        self.assertEqual(len(generator.postprocess_segments([self.line(ns=0.86, pk=-30.0)])), 1)

    def test_goodnight_breathing_is_a_short_hallucination(self) -> None:
        # sleep-scene breathing transcribed as goodnight: short + unsure.
        self.assertTrue(SubtitleGenerator.is_known_hallucination(
            "おやすみなさい", 1485.65, 1487.55, 2900.0,
            avg_logprob=-0.50, no_speech_prob=0.70))
        # A confidently decoded mid-video goodnight is real.
        self.assertFalse(SubtitleGenerator.is_known_hallucination(
            "おやすみなさい", 1450.0, 1452.0, 2900.0,
            avg_logprob=-0.30, no_speech_prob=0.20))

    def test_sparse_stretched_phrase_is_removed(self) -> None:
        # 7 chars spanning 11s (breathing tracked as words, mdvr-433 #3 #78).
        generator = make_generator()
        removal_log: list = []
        stretched = {"start": 2399.8, "end": 2410.86, "text": "おやすみなさい",
                     "anchored": True, "avg_logprob": -0.65, "no_speech_prob": 0.49,
                     "peak_db": -13.0, "speech_coverage": 0.62}
        result = generator.postprocess_segments([stretched], removal_log=removal_log)
        self.assertEqual(result, [])
        self.assertEqual(removal_log[0]["reason"], "sparse_text")

    def test_normally_paced_line_is_not_sparse(self) -> None:
        generator = make_generator()
        normal = {"start": 100.0, "end": 105.5, "text": "いらっしゃいませ初めまして担当するミヨと申します",
                  "anchored": True, "avg_logprob": -0.26, "no_speech_prob": 0.45,
                  "peak_db": -17.0, "speech_coverage": 0.97}
        self.assertEqual(len(generator.postprocess_segments([normal])), 1)


class MergeAdjacentFragmentsTests(unittest.TestCase):
    """Decoder-fragmented sentences (zero-gap segments) merge back."""

    @staticmethod
    def piece(start, end, text):
        return {"start": start, "end": end, "text": text}

    def test_zero_gap_fragments_merge(self) -> None:
        generator = make_generator()
        result = generator.merge_adjacent_fragments([
            self.piece(22.1, 23.9, "通常コースで"),
            self.piece(23.9, 25.38, "よろしかったですか?"),
        ])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "通常コースでよろしかったですか?")
        self.assertAlmostEqual(result[0]["end"], 25.38)

    def test_real_pause_is_not_bridged(self) -> None:
        generator = make_generator()
        result = generator.merge_adjacent_fragments([
            self.piece(19.26, 21.1, "本日は"),
            self.piece(22.1, 23.9, "通常コースで"),
        ])
        self.assertEqual(len(result), 2)

    def test_sentence_final_punctuation_keeps_own_line(self) -> None:
        generator = make_generator()
        result = generator.merge_adjacent_fragments([
            self.piece(10.0, 11.0, "よろしかったですか?"),
            self.piece(11.1, 11.6, "はい"),
        ])
        self.assertEqual(len(result), 2)

    def test_long_combined_text_is_not_merged(self) -> None:
        generator = make_generator()
        result = generator.merge_adjacent_fragments([
            self.piece(33.98, 37.28, "お支払いが現金のみなんですけど"),
            self.piece(37.28, 41.26, "現金のご用意は大丈夫ですかね"),
        ])
        self.assertEqual(len(result), 2)

    def test_chained_fragments_merge_into_one(self) -> None:
        generator = make_generator()
        result = generator.merge_adjacent_fragments([
            self.piece(11.6, 12.7, "初めまして"),
            self.piece(12.7, 15.2, "担当するミヨと申します"),
        ])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "初めまして担当するミヨと申します")


class RawDebugSegmentsTests(unittest.TestCase):
    def test_last_raw_segments_survive_postprocess_mutation(self) -> None:
        generator = make_generator()
        generator.model = SimpleNamespace(
            transcribe=lambda *_args, **_kwargs: (
                [segment("ではそちらで説明させていただきます", 0.0, 20.0, words=None)],
                None,
            ),
        )
        chunk = {"array": np.zeros(16000, dtype=np.float32), "offset_sec": 0.0, "duration_sec": 20.0}
        with patch.object(SubtitleGenerator, "split_audio_for_profile", return_value=[chunk]):
            final = generator.transcribe_profile("dummy.wav", "stable")

        # 17 chars -> 4.07s reading-speed cap applies to the decoder-timed line...
        self.assertAlmostEqual(final[0]["end"], 17 / 5.2 + 0.8, places=3)
        # ...but .raw.srt data must keep the decoder's original 20s end.
        self.assertAlmostEqual(generator.last_raw_segments[0]["end"], 20.0)


if __name__ == "__main__":
    unittest.main()
