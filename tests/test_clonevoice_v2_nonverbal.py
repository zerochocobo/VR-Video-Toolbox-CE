"""Torn-fragment repair and non-verbal removal for the dub (2026-09-02).

Both stages were added after the benchmark corpus (scripts/asr_bench, 5.5 h /
1630 lines) showed 22.7% of lines at three characters or fewer, 77 of them
sitting within a third of a second of a neighbour: one word the chunked decode
had cut in half. The cases pinned here are all real lines from that corpus.
"""
from __future__ import annotations

import unittest

from tool_clonevoice_v2 import segment_engine as se
from tool_subtitle import logic as subl
from tool_subtitle import nonverbal


def make_transcriber() -> se.CloneTranscriber:
    transcriber = se.CloneTranscriber.__new__(se.CloneTranscriber)
    transcriber.log_callback = lambda _msg: None
    return transcriber


def line(start, end, text):
    return {"start": start, "end": end, "text": text, "words": []}


class NonverbalClassifierTests(unittest.TestCase):
    def test_pure_vocalisation_is_nonverbal(self):
        for text in ("あぁん", "はぁはぁ", "んんっ", "ちゅぱちゅぱ"):
            with self.subTest(text=text):
                self.assertTrue(nonverbal.is_nonverbal(text))

    def test_dialogue_spelled_only_in_sound_kana_survives(self):
        # Every kana in these belongs to the sound alphabet, so only the
        # evidence layer keeps them. They were being deleted before 2026-09-02.
        for text in ("ごめん", "じゃあ", "いっぱい", "おめえ", "ふるえる", "おっぱい"):
            with self.subTest(text=text):
                self.assertFalse(nonverbal.is_nonverbal(text))

    def test_gemination_is_folded_before_the_evidence_check(self):
        self.assertTrue(nonverbal.has_language_evidence("すっごい"))
        self.assertFalse(nonverbal.is_nonverbal("すっごい"))

    def test_kanji_always_counts_as_evidence(self):
        self.assertFalse(nonverbal.is_nonverbal("気持ちいい"))

    def test_a_dialogue_line_keeps_the_whole_entry(self):
        self.assertFalse(nonverbal.is_nonverbal("はぁはぁ\nもう我慢できない"))

    def test_lone_token_but_not_backchannel(self):
        self.assertTrue(nonverbal.is_lone_nonverbal_token("ん。"))
        self.assertTrue(nonverbal.is_lone_nonverbal_token("あ"))
        for text in ("うん。", "はい。", "あー。", "あ、あ。", "はは。"):
            with self.subTest(text=text):
                self.assertFalse(nonverbal.is_lone_nonverbal_token(text))


class TornFragmentTests(unittest.TestCase):
    def setUp(self):
        self.transcriber = make_transcriber()

    def test_adjacent_halves_are_rejoined(self):
        merged = self.transcriber._merge_torn_fragments(
            [line(904.32, 906.96, "ご"), line(906.96, 907.34, "ちそう")])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["text"], "ごちそう")
        self.assertEqual((merged[0]["start"], merged[0]["end"]), (904.32, 907.34))

    def test_halves_separated_by_a_bad_timestamp_are_rejoined(self):
        # "ごめ" was stamped 0.06s long and its "ん" only came back 1.42s later.
        merged = self.transcriber._merge_torn_fragments(
            [line(230.09, 230.15, "ごめ"), line(231.57, 232.05, "ん")])
        self.assertEqual([entry["text"] for entry in merged], ["ごめん"])

    def test_two_plausible_fragments_are_left_apart(self):
        # Both carry dialogue evidence, so neither is doomed and the wide gap
        # does not apply -- these are two things that were said.
        merged = self.transcriber._merge_torn_fragments(
            [line(10.0, 10.5, "はい"), line(11.6, 12.0, "うん")])
        self.assertEqual([entry["text"] for entry in merged], ["はい", "うん"])

    def test_a_repeated_seam_is_trimmed(self):
        # Chunks overlap by two seconds, so the boundary syllables come back
        # from both decodes; plain concatenation would say "そわ" twice.
        merged = self.transcriber._merge_torn_fragments(
            [line(1.0, 2.0, "そわ"), line(2.0, 3.5, "そわしてきちゃった")])
        self.assertEqual(merged[0]["text"], "そわしてきちゃった")
        # The words no longer spell the text once a seam is dropped.
        self.assertEqual(merged[0]["words"], [])

    def test_two_full_lines_are_never_merged(self):
        pair = [line(1.0, 2.0, "そうですね"), line(2.1, 3.0, "わかりました")]
        self.assertEqual(len(self.transcriber._merge_torn_fragments(pair)), 2)

    def test_a_finished_sentence_does_not_absorb_the_next_fragment(self):
        pair = [line(1.0, 2.0, "そうですね。"), line(2.1, 2.3, "え")]
        self.assertEqual(len(self.transcriber._merge_torn_fragments(pair)), 2)

    def test_words_survive_a_clean_join(self):
        first = line(1.0, 1.4, "ご")
        first["words"] = [{"word": "ご", "start": 1.0, "end": 1.4}]
        second = line(1.5, 2.0, "ちそう")
        second["words"] = [{"word": "ちそう", "start": 1.5, "end": 2.0}]
        merged = self.transcriber._merge_torn_fragments([first, second])
        self.assertEqual(len(merged[0]["words"]), 2)


class NonverbalRemovalTests(unittest.TestCase):
    def setUp(self):
        self.transcriber = make_transcriber()
        self.Gen = subl.SubtitleGenerator

    def reason(self, text, duration):
        return self.transcriber._nonverbal_reason(text, duration, self.Gen)

    def test_short_fragments_are_not_judged(self):
        # Two characters is not enough to tell half of "ごめん" from a moan, so
        # the classifier's verdict is not acted on at that length.
        self.assertIsNone(self.reason("ごめ", 0.54))

    def test_long_vocalisation_is_removed(self):
        self.assertEqual(self.reason("あぁん", 1.0), "nonverbal")

    def test_lone_token_is_removed_at_any_length(self):
        self.assertEqual(self.reason("ん。", 0.36), "lone-token")

    def test_a_syllable_smeared_over_seconds_is_removed(self):
        self.assertEqual(self.reason("メ", 16.48), "stretched")
        self.assertEqual(self.reason("す", 10.98), "stretched")

    def test_real_words_with_bad_timing_survive(self):
        # Same shape as the stretched case, but there is enough text to mean
        # something; a short take in a long slot is not a defect.
        self.assertIsNone(self.reason("みーちゃん", 9.76))
        self.assertIsNone(self.reason("うまい", 4.62))

    def test_ordinary_dialogue_is_untouched(self):
        self.assertIsNone(self.reason("何かでどこから出てきたのそれ", 2.94))

    def test_drop_runs_over_a_list(self):
        kept = self.transcriber._drop_nonverbal(
            [line(0.0, 1.0, "あぁん"), line(1.0, 3.0, "そうですね")])
        self.assertEqual([entry["text"] for entry in kept], ["そうですね"])


class OrderingTests(unittest.TestCase):
    """Merging must happen before filtering, or real speech is deleted."""

    def test_torn_apology_survives_the_filter_only_when_merged_first(self):
        transcriber = make_transcriber()
        torn = [line(230.09, 230.15, "ごめ"), line(231.57, 232.05, "ん")]

        filtered_first = transcriber._drop_nonverbal([dict(entry) for entry in torn])
        self.assertEqual([entry["text"] for entry in filtered_first], ["ごめ"],
                         "the tail is a lone token and goes, leaving half a word")

        merged_first = transcriber._drop_nonverbal(
            transcriber._merge_torn_fragments([dict(entry) for entry in torn]))
        self.assertEqual([entry["text"] for entry in merged_first], ["ごめん"])


if __name__ == "__main__":
    unittest.main()


class DecodeWindowTests(unittest.TestCase):
    """The dub owns its decode window; tool_subtitle's must not leak back in.

    WhisperJAV lost eight days to exactly this shape of bug: a later assignment
    silently overrode the grouping parameters, so every knob above it -- GUI
    slider, CLI flag, preset -- looked live and did nothing. This pins which
    values reach the splitter.
    """

    def test_the_engine_window_is_what_reaches_the_splitter(self):
        transcriber = make_transcriber()
        seen = {}

        def capture(audio_file, chunk_seconds, profile_name, **kwargs):
            seen.update(kwargs, chunk_seconds=chunk_seconds)
            return [{"array": [], "offset_sec": 0.0, "duration_sec": 0.0}]

        transcriber.split_audio_for_profile = capture
        transcriber._split_chunks("audio.wav")

        # The values the engine owns are the ones that reach the splitter --
        # whether or not they currently differ from tool_subtitle's.
        self.assertEqual(seen["chunk_seconds"], se.CHUNK_SECONDS)
        self.assertEqual(seen["merge_gap_seconds"], se.WHISPERSEG_MERGE_GAP_SECONDS)

    def test_the_clean_vad_copy_is_passed_through(self):
        transcriber = make_transcriber()
        seen = {}

        def capture(audio_file, chunk_seconds, profile_name, **kwargs):
            seen.update(kwargs)
            return []

        transcriber.split_audio_for_profile = capture
        transcriber._split_chunks("audio.wav", "clean.wav")
        self.assertEqual(seen["vad_audio_path"], "clean.wav")


class StartTightenTests(unittest.TestCase):
    """A smeared start is repaired from the audio, not guessed from text length.

    WhisperJAV's CpsStartRetimer solves the same defect on a finished SRT, so it
    must estimate the onset as ``end - len(text) / reading_speed``. Running while
    the audio is still in hand, we can measure it.
    """

    SAMPLE_RATE = 16000

    def setUp(self):
        self.transcriber = make_transcriber()

    def audio(self, silent_seconds, voiced_seconds, level=0.3):
        import numpy as np
        silence = np.zeros(int(silent_seconds * self.SAMPLE_RATE), dtype=np.float32)
        # White-ish noise well above the -45 dB floor.
        voiced = np.full(int(voiced_seconds * self.SAMPLE_RATE), level, dtype=np.float32)
        return np.concatenate([silence, voiced])

    def test_the_start_moves_to_the_audible_onset(self):
        entries = [line(0.0, 5.0, "うまい")]
        moved = self.transcriber.tighten_entry_starts(
            entries, self.audio(4.0, 1.0), offset=0.0)
        self.assertEqual(moved, 1)
        self.assertAlmostEqual(entries[0]["start"], 4.0 - 0.15, places=1)
        self.assertEqual(entries[0]["end"], 5.0, "the end is never touched")

    def test_a_normally_paced_line_is_left_alone(self):
        entries = [line(0.0, 5.0, "何かでどこから出てきたのそれ")]
        self.assertEqual(
            self.transcriber.tighten_entry_starts(
                entries, self.audio(4.0, 1.0), offset=0.0),
            0,
        )
        self.assertEqual(entries[0]["start"], 0.0)

    def test_a_short_lead_is_the_line_s_own_onset(self):
        entries = [line(0.0, 5.0, "うまい")]
        self.assertEqual(
            self.transcriber.tighten_entry_starts(
                entries, self.audio(0.2, 4.8), offset=0.0),
            0,
        )

    def test_a_line_over_pure_silence_is_left_for_the_acoustic_checks(self):
        import numpy as np
        entries = [line(0.0, 5.0, "うまい")]
        self.assertEqual(
            self.transcriber.tighten_entry_starts(
                entries, np.zeros(5 * self.SAMPLE_RATE, dtype=np.float32), offset=0.0),
            0,
        )

    def test_the_line_keeps_a_usable_slot(self):
        entries = [line(0.0, 5.0, "うまい")]
        self.transcriber.tighten_entry_starts(
            entries, self.audio(4.95, 0.05), offset=0.0)
        self.assertAlmostEqual(entries[0]["end"] - entries[0]["start"],
                               se.START_TIGHTEN_MIN_DURATION_SECONDS, places=6)

    def test_word_times_before_the_new_start_are_dropped(self):
        entry = line(0.0, 5.0, "うまい")
        entry["words"] = [{"word": "う", "start": 0.1, "end": 0.3},
                          {"word": "まい", "start": 4.0, "end": 4.6}]
        self.transcriber.tighten_entry_starts([entry], self.audio(4.0, 1.0), offset=0.0)
        self.assertEqual([w["word"] for w in entry["words"]], ["まい"])

    def test_the_chunk_offset_is_honoured(self):
        entries = [line(100.0, 105.0, "うまい")]
        self.transcriber.tighten_entry_starts(
            entries, self.audio(4.0, 1.0), offset=100.0)
        self.assertAlmostEqual(entries[0]["start"], 104.0 - 0.15, places=1)


class ModelResolutionTests(unittest.TestCase):
    """anime-whisper is the Japanese default and a 2.9 GB optional download.

    An install that has not fetched it must land somewhere sensible rather than
    fail, and for a Japanese source that means the other Japanese model first --
    large-v3 is worse on this material than either of them, not merely
    different.
    """

    def resolve(self, key, language, present):
        from unittest.mock import patch
        with patch.object(se, "_model_present",
                          side_effect=lambda k, _root: k in present):
            return se.resolve_model_for_language(key, language, "models", lambda _m: None)

    def test_the_default_is_used_when_it_is_installed(self):
        self.assertEqual(
            self.resolve("anime-whisper", "ja", {"anime-whisper", "kotoba", "large-v3"}),
            "anime-whisper")

    def test_a_missing_default_falls_back_to_the_other_japanese_model(self):
        self.assertEqual(
            self.resolve("anime-whisper", "ja", {"kotoba", "large-v3"}), "kotoba")

    def test_large_v3_is_the_last_resort_for_japanese(self):
        self.assertEqual(
            self.resolve("anime-whisper", "ja", {"large-v3"}), "large-v3")

    def test_a_non_japanese_source_never_gets_a_japanese_model(self):
        for key in ("anime-whisper", "kotoba"):
            with self.subTest(key=key):
                self.assertEqual(self.resolve(key, "en", {key, "large-v3"}), "large-v3")

    def test_a_multilingual_choice_is_left_alone(self):
        self.assertEqual(self.resolve("large-v3", "ja", set()), "large-v3")

    def test_a_missing_kotoba_lands_on_anime_not_on_large_v3(self):
        self.assertEqual(
            self.resolve("kotoba", "ja", {"anime-whisper", "large-v3"}), "anime-whisper",
            "the fallback list is ordered best-first; large-v3 is the one we "
            "measured to be worst on this material")
