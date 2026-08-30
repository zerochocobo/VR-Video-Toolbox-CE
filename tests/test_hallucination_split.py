"""A stock hallucination emitted across two entries.

HNVR-174 opened with SRT #1 "この動画の字幕は視聴者の方によって作成" and #2
"されました。" -- YouTube's community-subtitle credit, which Whisper produces
over the music of a title card. 15 seconds of invented dialogue at the head of
the file, and the dub spoke it.

Neither per-line guard could catch it: the phrase was in no list, and once
added, the first half is only 19 of its 23 characters while the second half is
an ordinary Japanese ending that no per-line rule may ever drop.
"""
from __future__ import annotations

import pytest

from tool_subtitle.logic import SubtitleGenerator, drop_split_stock_phrases

CREDIT_HEAD = "この動画の字幕は視聴者の方によって作成"
CREDIT_TAIL = "されました。"


def _line(seg_id, text):
    return {"id": seg_id, "src_text": text}


# --- the per-line guard ---

def test_the_whole_credit_on_one_line_is_caught():
    assert SubtitleGenerator.is_known_hallucination(
        CREDIT_HEAD + CREDIT_TAIL, 23.6, 39.0, total_end=3467.0)


def test_a_long_leading_fragment_of_it_is_caught():
    """The decoder cuts at a chunk boundary, so the first entry is a prefix."""
    assert SubtitleGenerator.is_known_hallucination(CREDIT_HEAD, 23.6, 35.7, total_end=3467.0)


def test_a_common_phrase_that_merely_ends_a_stock_one_is_kept():
    """ありがとうございます is 77% of ご視聴ありがとうございます and an entirely
    ordinary thing to say. Matching substrings rather than prefixes broke this,
    and an existing test caught it."""
    assert not SubtitleGenerator.is_known_hallucination(
        "ありがとうございます", 79.6, 80.3, total_end=3467.0)


def test_an_ordinary_ending_is_never_dropped_on_its_own():
    """されました is normal Japanese. Only its context condemns it."""
    assert not SubtitleGenerator.is_known_hallucination(
        CREDIT_TAIL, 100.0, 104.0, total_end=3467.0)


@pytest.mark.parametrize("text", [
    "初めてなんですけど初めて?はい",
    "顔写真付きの身分証明書何かお持ちですか?",
    "どうもありがとうございますそうしますとこちらで身分証明書をお預かりしておりますので",
])
def test_real_dialogue_survives(text):
    assert not SubtitleGenerator.is_known_hallucination(text, 44.9, 51.5, total_end=3467.0)


# --- the run-level pass ---

def test_the_split_credit_is_dropped_as_a_run():
    kept = drop_split_stock_phrases(
        [_line(1, CREDIT_HEAD), _line(2, CREDIT_TAIL), _line(3, "初めてなんですけど初めて?はい")],
        text_key="src_text")
    assert [x["id"] for x in kept] == [3]


def test_the_same_ending_elsewhere_is_kept():
    """Only the run that spells the phrase goes; an identical line that does
    not follow the first half stays."""
    kept = drop_split_stock_phrases(
        [_line(1, "初めてなんですけど初めて?はい"), _line(2, CREDIT_TAIL)], text_key="src_text")
    assert [x["id"] for x in kept] == [1, 2]


def test_a_run_of_real_lines_is_never_matched():
    kept = drop_split_stock_phrases(
        [_line(1, "どうもありがとうございます"), _line(2, "そうしますとこちらで"),
         _line(3, "身分証明書をお預かりしております")], text_key="src_text")
    assert len(kept) == 3


def test_a_single_line_is_left_to_the_per_line_guard():
    """The run pass only fires on two or more entries, so it can never delete
    a lone line the per-line rules chose to keep."""
    kept = drop_split_stock_phrases([_line(1, CREDIT_HEAD + CREDIT_TAIL)], text_key="src_text")
    assert len(kept) == 1


def test_empty_input_is_returned_unchanged():
    assert drop_split_stock_phrases([], text_key="src_text") == []


# --- wiring ---

def test_v2_renumbers_after_dropping():
    """Leaving a gap in the ids would desynchronise the SRT from the manifest."""
    import inspect

    from tool_clonevoice_v2 import logic

    source = inspect.getsource(logic.run_transcribe_diarize)
    assert "drop_split_stock_phrases" in source
    assert 'item["srt_index"] = index' in source


def test_the_credit_is_in_the_phrase_list():
    from tool_subtitle.logic import HARD_HALLUCINATION_NORMS

    norms = {n for n in HARD_HALLUCINATION_NORMS if "視聴者" in n}
    assert norms, "the YouTube subtitle credit must be a known stock phrase"


# --- the orphan tail left behind by the per-line guard ---

def test_the_tail_is_dropped_when_its_head_was_just_removed():
    """The per-line guard runs during transcription and removes the head, so
    the later run-level pass only ever sees the orphan and cannot match it.
    The tail has to be caught where the head was dropped."""
    import inspect

    from tool_clonevoice_v2 import segment_engine

    source = inspect.getsource(segment_engine)
    assert "pending_tail" in source
    assert "stock-phrase tail" in source
    # It must be armed only by a prefix drop, and disarmed after one line.
    assert "phrase.startswith(norm)" in source
    assert "pending_tail = None" in source


def test_the_tail_rule_needs_the_head_and_a_deadline():
    """Two guards against eating a real line: the text must complete the very
    phrase that was removed, and it must follow closely in time."""
    import inspect

    from tool_clonevoice_v2 import segment_engine

    source = inspect.getsource(segment_engine)
    assert 'remainder.startswith(norm)' in source
    assert 'item["start"] <= deadline' in source
