from __future__ import annotations

import json
from pathlib import Path

from tool_clonevoice import logic
from tool_clonevoice import proofread as pf


def _save_manifest(video: Path, segments: list[dict], *, proofread: dict | None = None) -> None:
    manifest = {
        "video": str(video),
        "language": "ja",
        "target_language": "Chinese",
        "speakers": {},
        "segments": segments,
    }
    if proofread is not None:
        manifest["proofread"] = proofread
    logic.save_manifest(video, manifest)


def _segments() -> list[dict]:
    return [
        {
            "id": 1,
            "start": 0.0,
            "end": 2.0,
            "speaker": "SPEAKER_00",
            "src_text": "source one",
            "tgt_text": "target one",
        },
        {
            "id": 2,
            "start": 2.0,
            "end": 5.0,
            "speaker": "SPEAKER_01",
            "src_text": "source two",
            "tgt_text": "target two",
        },
    ]


def test_parse_srt_seconds_handles_bom_multiline_and_bad_blocks(tmp_path: Path):
    srt = tmp_path / "movie.srt"
    srt.write_text(
        "\ufeff1\n"
        "00:00:01,000 --> 00:00:02,500\n"
        "<i>Hello</i>\n"
        "world\n\n"
        "bad\n"
        "not a time\n\n"
        "3\n"
        "00:00:03.000 --> 00:00:04.000\n"
        "Second&nbsp;line\n",
        encoding="utf-8",
    )

    cues = pf.parse_srt_seconds(srt)

    assert cues == [
        {"start": 1.0, "end": 2.5, "text": "Hello world"},
        {"start": 3.0, "end": 4.0, "text": "Second line"},
    ]


def test_align_reference_handles_one_to_many_and_ref_only_rows():
    segments = [
        {"id": 1, "start": 0.0, "end": 4.0, "speaker": "A", "src_text": "a", "tgt_text": "ta"},
        {"id": 2, "start": 4.0, "end": 8.0, "speaker": "B", "src_text": "b", "tgt_text": "tb"},
    ]
    cues = [
        {"start": 0.0, "end": 1.0, "text": "ref one"},
        {"start": 1.2, "end": 3.0, "text": "ref two"},
        {"start": 3.0, "end": 6.2, "text": "cross cue"},
        {"start": 9.0, "end": 10.0, "text": "extra ref"},
        {"start": 20.0, "end": 20.1, "text": "too far"},
    ]

    rows = pf.align_reference(segments, cues)

    assert rows[0]["seg_id"] == 1
    assert rows[0]["ref_text"] == "ref one ref two"
    assert rows[1]["seg_id"] == 2
    assert rows[1]["ref_text"] == "cross cue"
    assert rows[2]["kind"] == "ref_only"
    assert rows[2]["ref_text"] == "extra ref"
    assert rows[3]["kind"] == "ref_only"
    assert rows[3]["ref_text"] == "too far"


def test_align_reference_rejects_low_overlap():
    rows = pf.align_reference(
        [{"id": 1, "start": 0.0, "end": 10.0, "src_text": "a", "tgt_text": "b"}],
        [{"start": 9.95, "end": 10.95, "text": "tiny overlap"}],
    )

    assert rows[0]["kind"] == "seg"
    assert rows[0]["ref_text"] == ""
    assert rows[1]["kind"] == "ref_only"
    assert rows[1]["ref_text"] == "tiny overlap"


def test_load_rows_detects_same_name_reference_srt(tmp_path: Path):
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    _save_manifest(video, _segments())
    (tmp_path / "movie.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\n参考\n", encoding="utf-8")
    (tmp_path / "movie.si.srt").write_text("ignored", encoding="utf-8")

    data = pf.load_rows(video)

    assert data["reference_srt"] == str(tmp_path / "movie.srt")
    assert data["rows"][0]["ref_text"] == "参考"


def test_save_rows_updates_manifest_backs_up_once_and_rewrites_srt(tmp_path: Path):
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    _save_manifest(video, _segments())
    cdir = logic.clone_dir(video)
    cdir.mkdir(parents=True, exist_ok=True)
    translated = cdir / "translated.srt"
    translated.write_text("original ai srt", encoding="utf-8")
    rows = pf.load_rows(video)["rows"]
    rows[0]["tgt_text"] = "edited one"
    rows[1]["tgt_text"] = ""

    result = pf.save_rows(video, rows)

    manifest = logic.load_manifest(video)
    assert manifest["segments"][0]["tgt_text"] == "edited one"
    assert manifest["segments"][1]["tgt_text"] == ""
    assert manifest["proofread"]["edited_ids"] == [1, 2]
    assert (cdir / "translated_org.srt").read_text(encoding="utf-8") == "original ai srt"
    rewritten = translated.read_text(encoding="utf-8")
    assert "[SPEAKER_00] edited one" in rewritten
    assert "SPEAKER_01" not in rewritten
    assert result["changed_count"] == 2

    rows[0]["original_tgt_text"] = rows[0]["tgt_text"]
    rows[0]["tgt_text"] = "edited again"
    translated.write_text("second version", encoding="utf-8")
    pf.save_rows(video, rows)

    assert (cdir / "translated_org.srt").read_text(encoding="utf-8") == "original ai srt"


def test_cleared_line_stays_translated_and_skips_retranslation(tmp_path: Path):
    from tool_clonevoice import single_clone as sc

    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    _save_manifest(video, _segments())
    rows = pf.load_rows(video)["rows"]
    rows[1]["tgt_text"] = ""

    pf.save_rows(video, rows)

    manifest = logic.load_manifest(video)
    assert manifest["proofread"]["cleared_ids"] == [2]
    # The cleared line counts as done: no re-translation on export, and the
    # panel keeps showing the video as translated/proofread.
    assert sc.manifest_has_target_translation(video, "Chinese")
    status = pf.video_status(video)
    assert status["status"] == "proofread"
    assert status["translated"] == 2

    # Refilling the line removes it from cleared_ids again.
    rows = pf.load_rows(video)["rows"]
    rows[1]["tgt_text"] = "target two again"
    pf.save_rows(video, rows)
    manifest = logic.load_manifest(video)
    assert manifest["proofread"]["cleared_ids"] == []
    assert sc.manifest_has_target_translation(video, "Chinese")


def test_never_translated_segment_still_counts_as_untranslated(tmp_path: Path):
    from tool_clonevoice import single_clone as sc

    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    segs = _segments()
    segs[1]["tgt_text"] = ""
    _save_manifest(video, segs)
    rows = pf.load_rows(video)["rows"]
    rows[0]["tgt_text"] = "edited one"

    pf.save_rows(video, rows)

    manifest = logic.load_manifest(video)
    # Segment 2 was never translated and never deliberately cleared, so it must
    # not be treated as done.
    assert manifest["proofread"]["cleared_ids"] == []
    assert not sc.manifest_has_target_translation(video, "Chinese")
    assert pf.video_status(video)["status"] == "untranslated"


def test_cut_segment_preview_clamps_and_requires_audio(tmp_path: Path):
    import pytest

    sf = pytest.importorskip("soundfile")
    np = pytest.importorskip("numpy")

    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    cdir = logic.clone_dir(video)
    cdir.mkdir(parents=True, exist_ok=True)
    sr = 16000
    sf.write(str(cdir / "audio16k.wav"), np.zeros(sr * 2, dtype="float32"), sr)

    # Pad is clamped at both file edges: 0.0..0.5 with 0.2 pad -> 0.0..0.7.
    clip = pf.cut_segment_preview(video, 0.0, 0.5, pad=0.2)
    data, clip_sr = sf.read(str(clip))
    assert clip_sr == sr
    assert len(data) == int(0.7 * sr)

    clip = pf.cut_segment_preview(video, 1.9, 5.0, pad=0.2)
    data, _sr = sf.read(str(clip))
    assert len(data) == sr * 2 - int(1.7 * sr)

    with pytest.raises(ValueError):
        pf.cut_segment_preview(video, 10.0, 11.0)

    (cdir / "audio16k.wav").unlink()
    with pytest.raises(FileNotFoundError):
        pf.cut_segment_preview(video, 0.0, 0.5)


def test_video_status_states(tmp_path: Path):
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")

    assert pf.video_status(video)["status"] == "no_manifest"

    segs = _segments()
    segs[1]["tgt_text"] = ""
    _save_manifest(video, segs)
    assert pf.video_status(video)["status"] == "untranslated"

    _save_manifest(video, _segments())
    status = pf.video_status(video)
    assert status["status"] == "translated"
    assert status["translated"] == 2

    _save_manifest(video, _segments(), proofread={"edited_ids": [1]})
    status = pf.video_status(video)
    assert status["status"] == "proofread"
    assert status["edited"] == 1
