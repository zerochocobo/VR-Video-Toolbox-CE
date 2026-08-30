"""The line-by-line source/dub comparison behind the pacing complaint."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tool_clonevoice_v2 import logic, pace


SR = 24000


def write_dub(path, spans, total=20.0, sr=SR):
    """A dub track: a tone over each (start, seconds) span, silence elsewhere."""
    track = np.zeros(int(total * sr), dtype=np.float32)
    for start, seconds in spans:
        begin = int(start * sr)
        track[begin:begin + int(seconds * sr)] = 0.4
    sf.write(str(path), track, sr)


@pytest.fixture
def video(tmp_path, monkeypatch):
    """A video whose clone dir the module helpers point at."""
    path = tmp_path / "title.mp4"
    path.write_bytes(b"")
    clone_dir = tmp_path / "title.clone"
    clone_dir.mkdir()
    monkeypatch.setattr(logic, "clone_dir", lambda _v: clone_dir)
    monkeypatch.setattr(logic, "manifest_path", lambda _v: clone_dir / "manifest.json")
    return path


def set_manifest(video, segments):
    (Path(video).parent / "title.clone" / "manifest.json").write_text(
        json.dumps({"segments": segments}), encoding="utf-8"
    )


SEGMENTS = [
    {"id": 1, "start": 1.0, "end": 5.0, "src_text": "ああ、そうですか", "tgt_text": "是吗"},
    {"id": 2, "start": 6.0, "end": 10.0, "src_text": "とても長い日本語の台詞", "tgt_text": "很长的台词"},
]


def test_a_line_is_measured_to_where_its_sound_stops(tmp_path):
    dub = tmp_path / "dub.wav"
    write_dub(dub, [(1.0, 1.2), (6.0, 3.5)])

    spoken = pace.measure_clone_seconds(dub, SEGMENTS)

    assert spoken[0] == pytest.approx(1.2, abs=0.01)
    assert spoken[1] == pytest.approx(3.5, abs=0.01)


def test_a_line_with_no_rendition_measures_zero(tmp_path):
    """Distinguishable from "no dub track at all", which measures None."""
    dub = tmp_path / "dub.wav"
    write_dub(dub, [(1.0, 1.2)])

    assert pace.measure_clone_seconds(dub, SEGMENTS)[1] == 0.0


def test_measuring_stops_at_the_next_line(tmp_path):
    """A rendition running past the next line's start must not be credited with
    that line's audio as well."""
    dub = tmp_path / "dub.wav"
    write_dub(dub, [(1.0, 8.0)])  # runs from 1.0 to 9.0, over line 2's start

    assert pace.measure_clone_seconds(dub, SEGMENTS)[0] == pytest.approx(5.0, abs=0.01)


def test_the_last_line_may_run_past_its_slot(tmp_path):
    """Lines are allowed to overrun into the gap behind them, and the last line
    has only the gap."""
    dub = tmp_path / "dub.wav"
    write_dub(dub, [(6.0, 7.0)])

    assert pace.measure_clone_seconds(dub, SEGMENTS)[1] == pytest.approx(7.0, abs=0.01)


def test_a_missing_dub_track_measures_nothing(tmp_path):
    assert pace.measure_clone_seconds(tmp_path / "absent.wav", SEGMENTS) == [None, None]


def test_coverage_and_the_exposed_tail_are_reported_per_line(video, monkeypatch):
    set_manifest(video, SEGMENTS)
    dub = Path(video).parent / "title.clone" / "dub.wav"
    write_dub(dub, [(1.0, 1.0), (6.0, 3.0)])
    monkeypatch.setattr(
        "tool_clonevoice_v2.proofread.cloned_track_path", lambda _v: dub
    )

    rows = pace.analyze_video(video)["rows"]

    assert rows[0]["slot"] == pytest.approx(4.0)
    assert rows[0]["clone"] == pytest.approx(1.0, abs=0.01)
    assert rows[0]["coverage"] == pytest.approx(0.25, abs=0.01)
    assert rows[0]["exposed"] == pytest.approx(3.0, abs=0.01), "ducked original, uncovered"
    assert rows[1]["coverage"] == pytest.approx(0.75, abs=0.01)


def test_the_text_side_reads_without_a_dub_track(video, monkeypatch):
    """Comparing source against translation is worth doing before export."""
    set_manifest(video, SEGMENTS)
    monkeypatch.setattr(
        "tool_clonevoice_v2.proofread.cloned_track_path", lambda _v: None
    )

    result = pace.analyze_video(video)

    assert [r["src_text"] for r in result["rows"]] == [s["src_text"] for s in SEGMENTS]
    assert [r["tgt_text"] for r in result["rows"]] == [s["tgt_text"] for s in SEGMENTS]
    assert all(r["clone"] is None for r in result["rows"])
    assert result["summary"]["median_coverage"] is None
    assert result["summary"]["median_char_ratio"] is not None, "still worth reading"


def test_lines_are_ordered_by_time_whatever_the_manifest_says(video, monkeypatch):
    set_manifest(video, list(reversed(SEGMENTS)))
    monkeypatch.setattr(
        "tool_clonevoice_v2.proofread.cloned_track_path", lambda _v: None
    )

    assert [r["id"] for r in pace.analyze_video(video)["rows"]] == [1, 2]


def test_summary_counts_the_lines_that_leave_the_original_exposed(video, monkeypatch):
    set_manifest(video, SEGMENTS)
    dub = Path(video).parent / "title.clone" / "dub.wav"
    write_dub(dub, [(1.0, 1.0), (6.0, 3.9)])
    monkeypatch.setattr(
        "tool_clonevoice_v2.proofread.cloned_track_path", lambda _v: dub
    )

    summary = pace.analyze_video(video)["summary"]

    assert summary["lines"] == 2
    # 0.25 leaves three seconds of ducked original; 0.975 leaves a tenth of a
    # second, which nobody hears as the Japanese still talking.
    assert summary["under_exposed"] == 1
    assert summary["under_half"] == 1
    assert summary["exposed_total"] == pytest.approx(3.1, abs=0.02)


def test_the_character_ratio_is_reported_since_it_is_the_structural_cause(video, monkeypatch):
    """Chinese runs about 0.6 the characters of the Japanese it replaces, which
    is most of the gap and is not something the fitter can undo."""
    set_manifest(video, [
        {"id": 1, "start": 0.0, "end": 4.0, "src_text": "あいうえおかきくけこ", "tgt_text": "一二三四五"},
    ])
    monkeypatch.setattr(
        "tool_clonevoice_v2.proofread.cloned_track_path", lambda _v: None
    )

    assert pace.analyze_video(video)["summary"]["median_char_ratio"] == pytest.approx(0.5)


def test_a_missing_manifest_says_what_to_do(video):
    with pytest.raises(FileNotFoundError, match="transcribe"):
        pace.analyze_video(video)
