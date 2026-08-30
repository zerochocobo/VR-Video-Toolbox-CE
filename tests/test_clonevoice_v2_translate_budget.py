"""Giving the translator each line's real time budget, and checking it landed.

The dubbing prompt already asked for a duration-matched translation and the
model complied -- median 0.71 against the requested 0.6-0.8 character ratio.
It was not enough: that ratio silently assumes the source is spoken at a normal
rate, and on sivr-314 part1 lines run to 11.2 kana/s against the 6.5 the
pipeline assumes. 23 of 80 lines could not fit their slot even so. This gives
the model the measured seconds instead of a rule of thumb, and checks the
answer rather than trusting it.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from tool_clonevoice_v2 import logic
from tool_subtitle import logic as sl


def _seg(seg_id, start, end, tgt=""):
    return {"id": seg_id, "start": start, "end": end, "src_text": "x", "tgt_text": tgt}


# --- the budget the model is told ---

def test_the_budget_carries_both_seconds_and_a_character_count():
    budgets = logic.translation_budgets([_seg(1, 0.0, 0.98)], "Chinese")
    assert budgets[1] == 'dur="0.98" max="5"'


def test_the_character_budget_matches_what_the_synthesizer_will_allow():
    """The number the model is given and the length the synthesizer treats as
    natural must come from the same rate, or the advice is a lie."""
    from tool_clonevoice_v2 import backend

    for seconds in (0.98, 1.42, 3.0):
        chars = logic.line_budget(seconds, logic._target_chars_per_second("Chinese"))
        assert backend.natural_reading_seconds("字" * chars, "zh") == pytest.approx(seconds, abs=0.2)


def test_a_line_without_timing_is_simply_left_out():
    budgets = logic.translation_budgets(
        [{"id": 1, "start": None, "end": None}, _seg(2, 1.0, 1.0), _seg(3, 0.0, 2.0)], "Chinese")
    assert set(budgets) == {3}


# --- the check afterwards ---

def test_a_line_that_cannot_be_spoken_in_time_is_flagged():
    """sivr-314 line 5: 0.98s holds 5 characters; the translation was 9."""
    segs = [_seg(1, 0.0, 0.98, "回来的时候都淋湿了")]
    assert logic.overlong_translations(segs, "Chinese") == [1]


def test_a_line_within_tolerance_is_left_alone():
    """The engine speaks ~20% faster on demand, so a small overrun is free."""
    assert logic.overlong_translations([_seg(1, 0.0, 1.0, "六个字刚好")], "Chinese") == []


def test_a_short_translation_is_never_flagged():
    """Finishing early costs nothing; only overruns break sync."""
    assert logic.overlong_translations([_seg(1, 0.0, 5.0, "短")], "Chinese") == []


# --- the retry keeps the better answer, not merely the newer one ---

class _Client:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return self.reply


def _retighten(reply, seg, monkeypatch):
    segs = [seg]
    captured = {}

    def fake_pass(client, chunk, template, placeholders, retries, log, stop, budgets=None):
        captured["budgets"] = budgets
        captured["template"] = template
        return {sid: reply for sid in chunk}

    monkeypatch.setattr(sl, "_llm_chunk_pass", fake_pass)
    logic._retighten_overlong(_Client(reply), segs, {}, "Chinese",
                              logic.translation_budgets(segs, "Chinese"),
                              2, True, lambda _m: None, None)
    return segs[0]["tgt_text"], captured


def test_a_shorter_retry_replaces_the_overlong_line(monkeypatch):
    out, _ = _retighten("淋湿着回来了", _seg(1, 0.0, 0.98, "回来的时候都淋湿了"), monkeypatch)
    assert out == "淋湿着回来了"


def test_a_longer_retry_is_discarded(monkeypatch):
    original = "回来的时候都淋湿了"
    out, _ = _retighten("回来的时候全身都被淋湿透了", _seg(1, 0.0, 0.98, original), monkeypatch)
    assert out == original


def test_an_empty_retry_never_wipes_the_line(monkeypatch):
    """An overrunning line still beats a line the dub cannot speak."""
    original = "回来的时候都淋湿了"
    out, _ = _retighten("", _seg(1, 0.0, 0.98, original), monkeypatch)
    assert out == original


def test_a_retry_that_leaks_source_kana_is_discarded(monkeypatch):
    original = "回来的时候都淋湿了"
    out, _ = _retighten("濡れて帰った", _seg(1, 0.0, 0.98, original), monkeypatch)
    assert out == original


def test_the_retry_gets_the_budget_and_the_clone_prompt(monkeypatch):
    _, captured = _retighten("淋湿着回来了", _seg(1, 0.0, 0.98, "回来的时候都淋湿了"), monkeypatch)
    assert captured["budgets"][1] == 'dur="0.98" max="5"'
    assert "max=" in captured["template"]


# --- v1 and tool_subtitle must be untouched ---

def test_the_tagging_is_unchanged_without_budgets():
    """Every existing caller passes no budgets and must get the old bytes."""
    chunk = {7: "hello", 9: "world"}
    assert sl.sequential_ids(chunk) == sl.sequential_ids(chunk, None)
    tagged, mapping = sl.sequential_ids(chunk)
    assert tagged == "<1>hello</1>\n<2>world</2>"
    assert mapping == {1: 7, 2: 9}


def test_budgets_only_change_the_opening_tag():
    tagged, _ = sl.sequential_ids({7: "hello"}, {7: 'dur="1.00" max="5"'})
    assert tagged == '<1 dur="1.00" max="5">hello</1>'


def test_every_new_parameter_defaults_to_the_old_behaviour():
    for func, names in (
        (sl.translate_entries, ("budgets", "prompt_name")),
        (sl._load_prompt_template, ("prompt_name",)),
        (sl._run_entries_llm, ("budgets",)),
        (sl._llm_chunk_pass, ("budgets",)),
        (sl.sequential_ids, ("budgets",)),
    ):
        params = inspect.signature(func).parameters
        for name in names:
            assert name in params, f"{func.__name__}.{name}"
            assert params[name].default is None, f"{func.__name__}.{name}"


def test_the_default_prompt_choice_is_unchanged():
    """tool_subtitle and v1 keep picking between the same two files."""
    src = inspect.getsource(sl._load_prompt_template)
    assert 'translate_prompt_dubbing.txt" if dubbing_optimized else "translate_prompt.txt' in src
    assert "if prompt_name is None:" in src


def test_v1_does_not_use_the_clone_prompt():
    v1 = Path("tool_clonevoice/logic.py").read_text(encoding="utf-8")
    assert "translate_prompt_clonevoice" not in v1
    assert "budgets" not in v1


def test_the_clone_prompt_file_exists_and_documents_the_tag():
    text = Path("config/translate_prompt_clonevoice.txt").read_text(encoding="utf-8")
    assert "{target_language}" in text
    assert 'dur="' in text and 'max="' in text
    assert "<id>translation</id>" in text


@pytest.mark.parametrize("name", ["translate_prompt.txt", "translate_prompt_dubbing.txt",
                                  "translate_prompt_clonevoice.txt"])
def test_every_prompt_carries_the_payload_placeholder(name):
    """Without {subtitles} the lines are never inserted and the model is asked
    to translate nothing -- it answers by asking for the content, and the whole
    file comes back untranslated. That is exactly what happened on the first
    run of the clone prompt."""
    text = Path("config") / name
    assert "{subtitles}" in text.read_text(encoding="utf-8"), name
