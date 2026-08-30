"""Re-exporting after proofreading should only synthesize what changed.

Fixing "skip when the .SI.WAV exists" so it means "when it is up to date"
turned silently discarding the edits into re-running the whole title for them.
model.infer is the expensive step and the only one worth reusing: level
matching and time fitting depend on whole-title statistics and are recomputed
every run regardless.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tool_clonevoice_v2 import backend


class CountingModel:
    def __init__(self):
        self.calls: list[dict] = []

    def infer(self, **kwargs):
        self.calls.append(dict(kwargs))
        wav = np.full(backend.SAMPLE_RATE // 2, 0.05, dtype=np.float32)
        sf.write(kwargs["output_path"], wav, backend.SAMPLE_RATE)
        return kwargs["output_path"]


@pytest.fixture
def scene(tmp_path):
    source = tmp_path / "audio16k.wav"
    sr = backend.SAMPLE_RATE
    t = np.arange(int(40.0 * sr)) / sr
    sf.write(source, (0.06 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), sr)
    return source, tmp_path / "out.wav"


def _segments():
    return [
        {"id": 11, "start": 1.0, "end": 3.0, "text": "第一句", "speaker": "A"},
        {"id": 22, "start": 5.0, "end": 7.0, "text": "第二句", "speaker": "A"},
        {"id": 33, "start": 9.0, "end": 11.0, "text": "第三句", "speaker": "A"},
    ]


def _run(model, scene, segments, **kw):
    source, output = scene
    backend.synthesize_manifest(
        model, source, segments, output, language="zh",
        fit_duration=False, level_match=False, timbre_anchor=False,
        log=kw.pop("log", lambda _m: None), **kw,
    )


def test_an_unchanged_rerun_synthesizes_nothing(scene):
    first, second = CountingModel(), CountingModel()
    _run(first, scene, _segments())
    assert len(first.calls) == 3
    _run(second, scene, _segments())
    assert second.calls == []


def test_only_the_edited_line_is_resynthesized(scene):
    first, second = CountingModel(), CountingModel()
    _run(first, scene, _segments())

    edited = _segments()
    edited[1]["text"] = "改过的第二句"
    _run(second, scene, edited)

    assert len(second.calls) == 1
    assert second.calls[0]["text"] == "改过的第二句"


def test_reassigning_a_speaker_changes_nothing_on_its_own(scene):
    """The speaker picks the prompt; with no anchor in play the prompts are
    identical, so there is nothing to redo. What must never happen is a line
    keeping audio built from a *different* prompt -- covered below."""
    first, second = CountingModel(), CountingModel()
    _run(first, scene, _segments())
    moved = _segments()
    moved[0]["speaker"] = "B"
    _run(second, scene, moved)
    assert second.calls == []


def test_a_changed_prompt_invalidates_the_line(scene, tmp_path):
    """The key hashes the prompt's *bytes*, not its path: the reference WAVs are
    re-cut every run, so a path match proves nothing."""
    source, output = scene
    first, second = CountingModel(), CountingModel()
    _run(first, scene, _segments())

    # Same spans, different source audio underneath them.
    sr = backend.SAMPLE_RATE
    t = np.arange(int(40.0 * sr)) / sr
    sf.write(source, (0.06 * np.sin(2 * np.pi * 400 * t)).astype(np.float32), sr)
    _run(second, scene, _segments())
    assert len(second.calls) == 3


def test_clips_are_named_for_the_segment_not_the_loop_position(scene):
    """Proofreading can merge lines away, which shifts every later index by
    one; an index-named cache would hand a clip to a different sentence."""
    model = CountingModel()
    _run(model, scene, _segments())
    work = Path(scene[1]).parent / backend.MANIFEST_INTERMEDIATE_DIR_NAME
    names = sorted(p.name for p in work.glob("out_*.wav"))
    assert names == ["out_00011.wav", "out_00022.wav", "out_00033.wav"]


def test_removing_a_line_does_not_shift_the_others_onto_its_audio(scene):
    first, second = CountingModel(), CountingModel()
    _run(first, scene, _segments())
    without_middle = [s for s in _segments() if s["id"] != 22]
    _run(second, scene, without_middle)
    # The survivors keep their own clips; nothing is regenerated and nothing
    # inherits the removed line's audio.
    assert second.calls == []


def test_an_orphaned_clip_is_cleaned_up(scene):
    model = CountingModel()
    _run(model, scene, _segments())
    work = Path(scene[1]).parent / backend.MANIFEST_INTERMEDIATE_DIR_NAME
    assert (work / "out_00022.wav").is_file()

    _run(CountingModel(), scene, [s for s in _segments() if s["id"] != 22])
    assert not (work / "out_00022.wav").exists()
    assert (work / "out_00011.wav").is_file()


def test_a_missing_clip_is_regenerated_even_though_the_key_matches(scene):
    first, second = CountingModel(), CountingModel()
    _run(first, scene, _segments())
    work = Path(scene[1]).parent / backend.MANIFEST_INTERMEDIATE_DIR_NAME
    (work / "out_00022.wav").unlink()
    _run(second, scene, _segments())
    assert len(second.calls) == 1


def test_a_corrupt_cache_file_is_ignored_rather_than_fatal(scene):
    first, second = CountingModel(), CountingModel()
    _run(first, scene, _segments())
    work = Path(scene[1]).parent / backend.MANIFEST_INTERMEDIATE_DIR_NAME
    (work / backend.LINE_CACHE_NAME).write_text("{not json", encoding="utf-8")
    _run(second, scene, _segments())
    assert len(second.calls) == 3


def test_the_reuse_is_reported(scene):
    _run(CountingModel(), scene, _segments())
    messages: list[str] = []
    _run(CountingModel(), scene, _segments(), log=messages.append)
    assert any("reused 3 of 3" in m for m in messages)
    assert sum("cached" in m for m in messages) == 3


def test_the_cache_file_records_one_entry_per_line(scene):
    _run(CountingModel(), scene, _segments())
    work = Path(scene[1]).parent / backend.MANIFEST_INTERMEDIATE_DIR_NAME
    data = json.loads((work / backend.LINE_CACHE_NAME).read_text(encoding="utf-8"))
    assert sorted(data["lines"]) == ["11", "22", "33"]
