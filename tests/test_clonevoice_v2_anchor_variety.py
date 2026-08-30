"""A timbre anchor must carry varied speech, not one sound repeated.

Judged on acoustics alone a sustained moan wins: it is long, loud and almost
perfectly periodic. On ipvr-385 the two top-ranked candidates offered to the
user were "くっくりくれくれかいかい" (0.905) and "やばいやばいやばい" ->
"太爽了太爽了太爽了" (0.895), both above every real sentence, and the batch
anchor -- which scored on acoustics with no text at all -- picked the same kind
of clip.
"""
from __future__ import annotations

import pytest

from tool_clonevoice_v2 import backend, refsel


REAL_LINES = [
    "でも最近忙しいかなって思ってなんか誘いづらかったんだよね",
    "なんかまだもうちょっと一緒にいたいな一緒にいたくない?",
    "気持ちよかったちょっと一緒に寝よ",
    "私ってなんか突っつきづらいかな",
    "チューしたい あなたからして",
    "たくさんいちゃいちゃできて幸せだったよ",   # 擬態語 reduplication, still real
    "私は最近ぼちぼちかな",                      # ...and again
]
MOANS = [
    "やばいやばいやばい",
    "くっくりくれくれかいかい",
    "じゃあ クリンクリンクイクリン",
]


@pytest.mark.parametrize("text", REAL_LINES)
def test_a_real_line_clears_the_variety_bar(text):
    assert backend.phonetic_variety(text) >= backend.ANCHOR_GOOD_TEXT_VARIETY


@pytest.mark.parametrize("text", MOANS)
def test_a_moan_does_not(text):
    assert backend.phonetic_variety(text) < backend.ANCHOR_GOOD_TEXT_VARIETY


def test_the_two_scorers_agree_on_the_bar():
    """The candidate list and the automatic anchor must not disagree about what
    counts as a usable reference."""
    assert refsel.GOOD_TEXT_VARIETY == backend.ANCHOR_GOOD_TEXT_VARIETY
    for text in REAL_LINES + MOANS:
        assert refsel._phonetic_variety(text) == backend.phonetic_variety(text)


def test_variety_ignores_spacing_and_punctuation():
    assert backend.phonetic_variety("あ, あ. あ") == pytest.approx(1 / 3)
    assert backend.phonetic_variety("") == 0.0
    assert backend.phonetic_variety("   ") == 0.0


def test_the_anchor_scorer_prefers_a_quieter_line_over_a_louder_moan():
    """The moan wins on every acoustic term -- longer, louder, more periodic --
    so the variety factor has to be strong enough to overturn all of them."""
    moan = {"duration": 6.0, "active_seconds": 5.8, "periodic_ratio": 0.98,
            "active_rms_dbfs": -30.0}
    line = {"duration": 5.0, "active_seconds": 4.0, "periodic_ratio": 0.75,
            "active_rms_dbfs": -38.0}
    median = -38.0
    assert backend._anchor_score(moan, median) > backend._anchor_score(line, median), (
        "precondition: on acoustics alone the moan must win"
    )
    moan_score = backend._anchor_score(moan, median, {"src_text": "やばいやばいやばい"})
    line_score = backend._anchor_score(
        line, median, {"src_text": "気持ちよかったちょっと一緒に寝よ"}
    )
    assert line_score > moan_score


def test_a_title_with_nothing_better_still_gets_an_anchor():
    """The factor scales the score, it does not veto: a video that is all moans
    must still end up with a reference rather than none."""
    entries = [{"speaker": "A", "src_text": t} for t in MOANS]
    profiles = [
        {"duration": 5.0, "active_seconds": 4.5, "periodic_ratio": 0.9,
         "active_rms_dbfs": -35.0}
        for _ in MOANS
    ]
    anchors = backend.select_speaker_timbre_anchors(entries, profiles)
    assert "A" in anchors
