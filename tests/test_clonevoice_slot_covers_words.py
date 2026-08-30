"""A subtitle slot must be able to hold the words attached to it.

On sivr-314 part1, 34 of 81 lines reached the synthesizer attached to a slot
far shorter than their own word extent — the worst being a full sentence on a
0.12 s slot. ``fit_audio_to_duration`` then time-compressed the whole utterance
into that sliver, so the dub was inaudible. Three places could produce it and
each is covered here.
"""
from __future__ import annotations

import pytest

from tool_clonevoice import logic as v1
from tool_clonevoice_v2 import logic as v2
from tool_subtitle.logic import SubtitleGenerator


def _words(*spans):
    return [{"w": text, "start": start, "end": end} for start, end, text in spans]


# --- the invariant helper itself ---

def test_cover_own_words_grows_a_slot_that_starts_after_its_first_word():
    seg = {"start": 108.88, "end": 109.00, "src_text": "降るって言ってたでしょ?",
           "words": _words((107.68, 108.78, "降るって言ってた"), (108.78, 109.00, "でしょ?"))}
    assert v2._cover_own_words(seg) is True
    assert seg["start"] == 107.68
    assert seg["end"] == 109.00


def test_cover_own_words_leaves_a_slot_that_already_contains_them():
    """A tail-extended end is deliberate and must survive untouched."""
    seg = {"start": 10.0, "end": 13.0, "src_text": "ok",
           "words": _words((10.2, 12.0, "ok"))}
    assert v2._cover_own_words(seg) is False
    assert (seg["start"], seg["end"]) == (10.0, 13.0)


def test_a_folded_fragment_keeps_the_neighbour_slot_when_it_still_fits():
    """The merged line drops the fragment's untrustworthy timing on purpose;
    only a slot that cannot speak the text is repaired."""
    seg = {"start": 11.64, "end": 15.18,
           "src_text": "本日はご来店いただきありがとうございます",
           "words": _words((9.96, 10.68, "本"), (11.64, 15.18, "日はご来店いただきありがとうございます"))}
    assert v2._cover_own_words(seg) is False
    assert (seg["start"], seg["end"]) == (11.64, 15.18)


def test_slot_can_hold_line_separates_the_two_cases():
    assert v2._slot_can_hold_line(
        {"start": 11.64, "end": 15.18, "src_text": "本日はご来店いただきありがとうございます"}
    ) is True
    assert v2._slot_can_hold_line(
        {"start": 108.88, "end": 109.0, "src_text": "降るって言ってたでしょ?"}
    ) is False
    # No text to speak: nothing to be too short for.
    assert v2._slot_can_hold_line({"start": 1.0, "end": 1.1, "src_text": ""}) is True


def test_cover_own_words_ignores_a_wordless_segment():
    seg = {"start": 1.0, "end": 1.1, "src_text": "あいうえおかきくけこ", "words": []}
    assert v2._cover_own_words(seg) is False


# --- route 1: the orphan absorb ---

def test_absorbing_into_a_sliver_widens_it_over_the_merged_words():
    """The real sivr-314 id 4 shape: a 0.12 s line becomes the merge target."""
    target = {"start": 108.88, "end": 109.00, "src_text": "でしょ?",
              "words": _words((108.78, 109.00, "でしょ?"))}
    other = {"start": 107.68, "end": 108.78, "src_text": "降るって言ってた",
             "words": _words((107.68, 108.78, "降るって言ってた"))}

    v2._absorb_fragment(target, other, other_first=True, extend=False)

    assert target["src_text"] == "降るって言ってたでしょ?"
    # Without the fix this stayed 0.12 s while carrying the whole sentence.
    assert target["start"] == 107.68
    assert target["end"] == 109.00
    assert target["dur"] == pytest.approx(1.32)


def test_merge_pass_reports_and_repairs_lines_it_could_not_hold():
    segments = [
        {"id": 1, "start": 5.0, "end": 5.12, "src_text": "あいうえおかきくけこ", "speaker": "",
         "words": _words((4.0, 5.12, "あいうえおかきくけこ"))},
        {"id": 2, "start": 30.0, "end": 31.0, "src_text": "c", "speaker": "",
         "words": _words((30.0, 31.0, "c"))},
    ]
    messages: list[str] = []
    result = v2._merge_dub_fragments(segments, log=messages.append)

    assert result[0]["start"] == 4.0
    assert any("could not hold their own words" in m for m in messages)


# --- route 2: the word-gap split, shared by v1 and v2 ---

@pytest.mark.parametrize("module", [v1, v2], ids=["v1", "v2"])
def test_split_on_word_gaps_never_returns_a_slot_inside_its_words(module):
    words = _words((107.68, 108.20, "a"), (108.20, 109.00, "b"))
    out = module._split_on_word_gaps(108.88, 109.00, "ab", words)

    assert len(out) == 1
    assert out[0]["start"] == 107.68
    assert out[0]["end"] == 109.00


@pytest.mark.parametrize("module", [v1, v2], ids=["v1", "v2"])
def test_split_on_word_gaps_keeps_a_deliberately_extended_tail(module):
    """extend_entry_tails pushes the end past the last word on purpose."""
    words = _words((10.0, 11.0, "a"))
    out = module._split_on_word_gaps(10.0, 11.4, "a", words)

    assert out[0]["start"] == 10.0
    assert out[0]["end"] == 11.4


@pytest.mark.parametrize("module", [v1, v2], ids=["v1", "v2"])
def test_split_on_word_gaps_still_rejects_a_stray_timestamp(module):
    """Bounds far outside the word extent are the old stray-stamp bug."""
    words = _words((100.0, 101.0, "a"))
    out = module._split_on_word_gaps(100.0, 140.0, "a", words)

    assert out[0]["end"] == 101.0


# --- tool_subtitle: deliberately NOT repaired here ---

def test_subtitle_silence_clamp_may_still_sit_inside_unreliable_word_stamps():
    """The shared front-end clamps a piece to the measured silence even when
    that puts the span inside the DTW word stamps — there the stamps are the
    unreliable side (a word stamped inside a measured pause). Plain subtitles
    only lose display timing by it; the dub damage is repaired downstream by
    :func:`_cover_own_words`, which is why this clamp is left alone.
    """
    generator = SubtitleGenerator.__new__(SubtitleGenerator)
    group = [(0.5, 1.2, "a"), (1.2, 1.9, "b")]
    generator.find_silence_runs = lambda *_a, **_k: [(1.0, 2.5)]

    pieces = generator.split_group_on_silence(group, chunk_audio=object())

    assert len(pieces) == 1
    _words_out, start, end = pieces[0]
    assert start == 0.5
    assert end < 1.2, "the energy clamp must win over the word stamp"
