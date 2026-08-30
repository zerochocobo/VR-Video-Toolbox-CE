"""Splitting a line at a real speaker change.

``assign_speakers`` only tags a line with whoever dominates it, so a question
and its answer reached the synthesizer as one inference and came back in one
voice. On a two-person title that was 23% of the lines. The shapes below are
taken from that title (URVRSP-566 part 1) and from the diarizer output measured
on it.
"""
from __future__ import annotations

import pytest

from tool_clonevoice_v2 import diarize as diar


def _words(*spans):
    return [{"w": text, "start": start, "end": end} for start, end, text in spans]


def _segment(seg_id, words, text=None, **extra):
    return {
        "id": seg_id, "srt_index": seg_id,
        "start": float(words[0]["start"]), "end": float(words[-1]["end"]),
        "dur": round(float(words[-1]["end"]) - float(words[0]["start"]), 3),
        "src_text": text if text is not None else "".join(w["w"] for w in words),
        "tgt_text": "", "emotion_ref": "", "words": words, **extra,
    }


# --- fragment clusters ---

def test_substantial_speakers_drops_the_fragment_clusters():
    """The real title clustered two people into four labels: the extra two
    averaged 0.50s and 0.31s per turn against the real pair's 1.37s and 1.07s."""
    turns = []
    for i in range(40):  # a real speaker: long turns, most of the audio
        turns.append((i * 10.0, i * 10.0 + 1.4, "SPEAKER_00"))
    for i in range(20):  # the other real speaker
        turns.append((i * 10.0 + 2.0, i * 10.0 + 3.1, "SPEAKER_02"))
    for i in range(60):  # jitter: many turns, almost no time
        turns.append((i * 3.0 + 0.5, i * 3.0 + 0.8, "SPEAKER_01"))

    keep = diar.substantial_speakers(turns)

    assert keep == {"SPEAKER_00", "SPEAKER_02"}


def test_substantial_speakers_never_returns_nothing():
    """A title of only short turns must still be usable rather than empty."""
    turns = [(i * 2.0, i * 2.0 + 0.3, "SPEAKER_00") for i in range(10)]
    assert diar.substantial_speakers(turns) == {"SPEAKER_00"}


# --- the split ---

def test_a_question_and_its_answer_become_two_lines():
    """seg 2: そうなの?疲れてるの? / 残業多いじゃん."""
    words = _words(
        (9.48, 9.90, "そう"), (9.90, 10.30, "なの?"), (10.30, 10.82, "疲れてるの?"),
        (11.42, 11.80, "残業"), (11.80, 12.10, "多い"), (12.10, 12.40, "じゃん"),
    )
    turns = [(9.40, 10.90, "SPEAKER_02"), (11.30, 12.50, "SPEAKER_00")]

    out = diar.split_segments_by_turns([_segment(1, words)], turns, log=lambda _m: None)

    assert len(out) == 2
    assert out[0]["src_text"] == "そうなの?疲れてるの?"
    assert out[1]["src_text"] == "残業多いじゃん"
    assert out[0]["speaker"] == "SPEAKER_02"
    assert out[1]["speaker"] == "SPEAKER_00"
    assert out[0]["end"] == 10.82
    assert out[1]["start"] == 11.42
    assert [s["id"] for s in out] == [1, 2]


def test_a_split_line_drops_the_translation_of_the_whole_line():
    words = _words((0.0, 1.0, "a"), (1.0, 2.0, "b"), (2.0, 3.0, "c"),
                   (3.0, 4.0, "d"), (4.0, 5.0, "e"), (5.0, 6.0, "f"))
    turns = [(0.0, 3.0, "SPEAKER_00"), (3.0, 6.0, "SPEAKER_01")]
    parent = _segment(1, words, tgt_text="整行的旧译文")

    out = diar.split_segments_by_turns([parent], turns, log=lambda _m: None)

    assert len(out) == 2
    assert all(s["tgt_text"] == "" for s in out)


def test_one_speaker_is_left_alone_however_long_the_pause():
    """seg 15 has a 0.76s breath but the diarizer keeps one speaker across it,
    which a pause threshold alone could not tell from a turn change."""
    words = _words((67.16, 68.20, "それってさ"), (68.20, 69.25, "逆に天才じゃね?"),
                   (70.01, 71.10, "ギリギリ狙うって"), (71.10, 72.12, "俺できないよ"))
    turns = [(67.10, 69.30, "SPEAKER_00"), (69.95, 72.20, "SPEAKER_00")]

    out = diar.split_segments_by_turns([_segment(1, words)], turns, log=lambda _m: None)

    assert len(out) == 1
    assert out[0]["src_text"] == "それってさ逆に天才じゃね?ギリギリ狙うって俺できないよ"


def test_a_backchannel_does_not_split_the_line():
    """seg 6/7: a 0.4s foreign turn sitting inside a longer one is jitter."""
    words = _words((23.40, 24.20, "仕事"), (24.20, 25.00, "できない"),
                   (25.10, 25.45, "人に"), (25.51, 26.40, "対して"),
                   (26.40, 27.30, "どう教えて"))
    turns = [(23.40, 25.41, "SPEAKER_00"), (25.11, 25.51, "SPEAKER_01"),
             (25.51, 27.06, "SPEAKER_00")]

    out = diar.split_segments_by_turns([_segment(1, words)], turns, log=lambda _m: None)

    assert len(out) == 1


def test_fragment_clusters_cannot_cause_a_split():
    """seg 48: a 2.16s line held seven turns, three under 0.1s, and the jitter
    cluster split it even though both halves were the same person."""
    words = _words((263.30, 263.90, "そっ"), (263.90, 264.40, "かー、"),
                   (264.40, 264.90, "やっ"), (264.90, 265.46, "たー!"))
    turns = [(262.29, 262.96, "SPEAKER_01"), (263.70, 263.75, "SPEAKER_01"),
             (264.16, 264.19, "SPEAKER_00"), (264.19, 264.51, "SPEAKER_01"),
             (264.51, 264.60, "SPEAKER_00"), (264.60, 264.80, "SPEAKER_01"),
             (264.97, 265.36, "SPEAKER_01")]
    # Give SPEAKER_00 the long turns it really has elsewhere in the title.
    turns += [(i * 20.0, i * 20.0 + 4.0, "SPEAKER_00") for i in range(40)]

    out = diar.split_segments_by_turns([_segment(1, words)], turns, log=lambda _m: None)

    assert len(out) == 1, "the jitter cluster must not be allowed to split a line"


# --- guards ---

def test_no_turns_leaves_the_segments_untouched():
    words = _words((0.0, 1.0, "a"), (1.0, 2.0, "b"), (2.0, 3.0, "c"))
    segments = [_segment(1, words)]
    assert diar.split_segments_by_turns(segments, [], log=lambda _m: None) is segments


def test_a_line_shorter_than_the_hysteresis_is_never_split():
    words = _words((0.0, 0.4, "a"), (0.4, 0.8, "b"))
    turns = [(0.0, 0.4, "SPEAKER_00"), (0.4, 0.8, "SPEAKER_01")]
    out = diar.split_segments_by_turns([_segment(1, words)], turns, log=lambda _m: None)
    assert len(out) == 1


def test_words_outside_every_turn_inherit_their_neighbour():
    """A word in a diarization gap must not invent a third speaker."""
    labels = diar._word_speakers(
        _words((0.0, 1.0, "a"), (1.5, 2.0, "b"), (3.0, 4.0, "c")),
        [(0.0, 1.0, "SPEAKER_00"), (3.0, 4.0, "SPEAKER_00")],
    )
    assert labels == ["SPEAKER_00"] * 3


# --- the batch panel switch ---

@pytest.fixture
def batch_panel():
    """Build the real panel; catches ordering bugs a source scan cannot."""
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # no display
        pytest.skip(f"no Tk display: {exc}")
    root.withdraw()
    from tool_clonevoice_v2 import gui as gui_module

    app_cls = next(
        value for value in vars(gui_module).values()
        if isinstance(value, type) and hasattr(value, "_setup_ui")
    )
    app = app_cls(root)
    yield app
    root.destroy()


@pytest.mark.parametrize("page", ["batch_num_speakers_var", "single_clone_num_speakers_var"])
def test_both_clone_pages_default_to_one_voice_held_together(batch_panel, page):
    """語音克隆 and 批量克隆 both start at one voice, which is now a count of
    voices to keep together rather than an instruction to skip diarization."""
    assert batch_panel._speaker_choice(getattr(batch_panel, page).get()) == ("auto", 1)


def test_speaker_count_is_the_only_control(batch_panel):
    """Every count goes to auto, which resolves to pyannote when its bundle is
    present. A backend picker would be noise: resolve_backend never selects
    ECAPA on its own. 1 no longer means "skip diarization" -- a title with one
    voice that matters still has the odd line from someone else, and with no
    labels that line was cloned from the main speaker's anchor."""
    assert batch_panel._speaker_choice("1") == ("auto", 1)
    assert batch_panel._speaker_choice("2") == ("auto", 2)
    assert batch_panel._speaker_choice("7") == ("auto", 7)
    assert batch_panel._speaker_choice("自动") == ("auto", None)
    assert not hasattr(batch_panel, "batch_diarize_var")


def test_auto_never_resolves_to_a_backend_the_pages_do_not_offer(tmp_path):
    """The premise behind dropping the picker."""
    assert diar.resolve_backend("auto", str(tmp_path)) == "single"
    (tmp_path / "speaker-diarization-community-1").mkdir()
    (tmp_path / "speaker-diarization-community-1" / "config.yaml").write_text("x", encoding="utf-8")
    assert diar.resolve_backend("auto", str(tmp_path)) == "pyannote"


def test_both_entry_points_accept_what_the_pages_produce():
    import inspect

    from tool_clonevoice_v2 import logic, single_clone

    targets = [
        (logic.run_full, "none"), (logic.run_batch, "none"),
        (single_clone.run_single_transcribe, "none"),
    ]
    for func, default in targets:
        params = inspect.signature(func).parameters
        assert "diarize_backend" in params, func.__name__
        assert "num_speakers" in params, func.__name__
        assert params["diarize_backend"].default == default, func.__name__
        assert params["num_speakers"].default is None, func.__name__


def test_a_brief_but_real_speaker_is_kept():
    """3dsvr-1911: the man introduces the new hire in the opening minute and
    says nothing after -- 26.3s of 673.5s, 3.9%. Judged by share he was a
    fragment cluster, so no line was ever cut at his boundary and his dialogue
    reached the synthesizer in the woman's voice."""
    turns = [(float(i) * 2.0, float(i) * 2.0 + 0.73, "SPEAKER_00") for i in range(36)]
    turns += [(200.0 + i * 2.0, 200.0 + i * 2.0 + 1.68, "SPEAKER_01") for i in range(384)]
    keep = diar.substantial_speakers(turns)
    assert keep == {"SPEAKER_00", "SPEAKER_01"}


def test_the_clusters_the_rule_was_written_for_are_still_dropped():
    """hnvr-152: 489 turns averaging 0.48s and 390 averaging 0.65s, each
    holding about a third of the speech. Share would have kept both; how they
    talk is what gives them away."""
    turns = [(float(i) * 0.6, float(i) * 0.6 + 0.48, "FRAGMENTS") for i in range(489)]
    turns += [(1000.0 + i * 1.2, 1000.0 + i * 1.2 + 0.93, "REAL") for i in range(273)]
    turns += [(3000.0 + i * 0.8, 3000.0 + i * 0.8 + 0.65, "BREATHS") for i in range(390)]
    assert diar.substantial_speakers(turns) == {"REAL"}


def test_one_stray_turn_is_not_a_speaker():
    """urvrsp-566 part2 had exactly this: a single 0.3s turn. A cluster that
    long-winded by accident must not start voting on speaker changes."""
    turns = [(0.0, 0.3, "STRAY")]
    turns += [(10.0 + i * 2.0, 10.0 + i * 2.0 + 0.94, "REAL") for i in range(382)]
    assert diar.substantial_speakers(turns) == {"REAL"}

    # ...nor two turns that happen to be long
    turns = [(0.0, 3.0, "STRAY"), (5.0, 8.0, "STRAY")]
    turns += [(20.0 + i * 2.0, 20.0 + i * 2.0 + 0.94, "REAL") for i in range(382)]
    assert diar.substantial_speakers(turns) == {"REAL"}


def test_share_no_longer_decides_anything():
    """The same speaker, once holding 4% of the video and once 40%: the verdict
    must not move."""
    brief = [(float(i) * 2.0, float(i) * 2.0 + 0.9, "GUEST") for i in range(20)]
    host_short = [(100.0 + i * 2.0, 100.0 + i * 2.0 + 1.5, "HOST") for i in range(20)]
    host_long = [(100.0 + i * 2.0, 100.0 + i * 2.0 + 1.5, "HOST") for i in range(500)]
    assert diar.substantial_speakers(brief + host_short) == {"GUEST", "HOST"}
    assert diar.substantial_speakers(brief + host_long) == {"GUEST", "HOST"}
