"""A translation pass has to leave the video counted as translated.

Regenerating in the refined tab re-ran the AI source proofread and the whole
translation on a video nothing had changed about. The "already translated"
check requires every line with source text to have a translation or a cleared
mark, and lines the LLM produces nothing for got neither -- so the check stayed
false, every export sent all 187 entries back to the API, and the next run was
identical: the LLM has nothing to translate in "おこそとのほまよももを".
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tool_clonevoice_v2 import logic, proofread, single_clone


def settled(manifest, target="Chinese"):
    """What manifest_has_target_translation asks: nothing left blocking."""
    cleared = proofread.effective_cleared_segment_ids(manifest)
    return not [
        s for s in manifest["segments"]
        if str(s.get("src_text") or "").strip()
        and not str(s.get("tgt_text") or "").strip()
        and int(s.get("id", -1)) not in cleared
    ]


def test_the_completion_check_agrees_once_the_pass_has_settled(tmp_path, monkeypatch):
    """End to end through the function the refined tab actually calls."""
    video = tmp_path / "title.mp4"
    video.write_bytes(b"")
    cdir = tmp_path / "title.clone"
    cdir.mkdir()
    (cdir / "translated.srt").write_text("", encoding="utf-8")
    monkeypatch.setattr(logic, "clone_dir", lambda _v: cdir)

    segments = [
        {"id": 1, "src_text": "おはよう", "tgt_text": "早上好"},
        {"id": 2, "src_text": "ヘゲヘヘヘホンゴボン", "tgt_text": ""},
    ]
    manifest = {"segments": segments, "target_language": "Chinese", "proofread": {}}
    monkeypatch.setattr(logic, "load_manifest", lambda _v: manifest)

    assert not single_clone.manifest_has_target_translation(video, "Chinese")
    manifest["proofread"]["cleared_ids"] = [2]
    assert single_clone.manifest_has_target_translation(video, "Chinese")


def test_a_different_target_language_still_retranslates(tmp_path, monkeypatch):
    """Settling must not make the video immune to an actual change."""
    video = tmp_path / "title.mp4"
    video.write_bytes(b"")
    cdir = tmp_path / "title.clone"
    cdir.mkdir()
    monkeypatch.setattr(logic, "clone_dir", lambda _v: cdir)
    manifest = {
        "segments": [{"id": 1, "src_text": "おはよう", "tgt_text": "早上好"}],
        "target_language": "Chinese", "proofread": {},
    }
    monkeypatch.setattr(logic, "load_manifest", lambda _v: manifest)

    assert single_clone.manifest_has_target_translation(video, "Chinese")
    assert not single_clone.manifest_has_target_translation(video, "English")


def run_real_translate(tmp_path, monkeypatch, segments, *, translates):
    """Drive the real run_translate with the LLM calls stubbed out.

    Reimplementing the bookkeeping in the test would let the test and the code
    drift apart, which is exactly how the gap this covers survived.
    """
    from tool_subtitle import logic as sl

    video = tmp_path / "title.mp4"
    video.write_bytes(b"")
    cdir = tmp_path / "title.clone"
    cdir.mkdir()
    manifest = {"segments": segments, "language": "ja"}
    monkeypatch.setattr(logic, "clone_dir", lambda _v: cdir)
    monkeypatch.setattr(logic, "load_manifest", lambda _v: manifest)
    saved = {}
    monkeypatch.setattr(logic, "save_manifest", lambda _v, m: saved.update(m) or cdir)
    monkeypatch.setattr(logic, "write_srt", lambda *_a, **_k: None)

    monkeypatch.setattr(sl, "load_trans_config", lambda: {
        "target_language": "Chinese", "tokens_per_chunk": 1000,
        "adult_content": True, "source_correction": False, "model_name": "stub",
    })
    monkeypatch.setattr(sl, "make_llm_client", lambda *_a, **_k: object())
    monkeypatch.setattr(sl, "log_llm_usage", lambda *_a, **_k: None)

    def fake_translate(_client, entries, *_a, **_k):
        for sid, info in entries.items():
            rendered = translates.get(int(sid))
            if rendered is None:
                # The shape the gap lived in, and the state the real manifest
                # is in: the pass reports the line as handled and returns
                # nothing for it. The write-back requires non-empty text so it
                # writes no translation, and the old bookkeeping asked only
                # whether the line was "translated", so it recorded nothing
                # either -- leaving a line that blocks the check forever.
                info["translated"] = True
                info["text"] = ""
            else:
                info["translated"] = True
                info["text"] = rendered

    monkeypatch.setattr(sl, "translate_entries", fake_translate)
    monkeypatch.setattr(logic, "_retighten_overlong",
                        lambda *_a, **_k: 0)
    return logic.run_translate(video, target_language="Chinese", api_key="k",
                               source_correction=False, log=lambda _m: None)


def test_the_real_pass_settles_a_line_it_cannot_translate(tmp_path, monkeypatch):
    """3dsvr-1911 part 1: hallucinated source, reported translated, no text."""
    segments = [
        {"id": 1, "src_text": "みんなおはようございます", "tgt_text": ""},
        {"id": 129, "src_text": "おこそとのほまよももを", "tgt_text": ""},
    ]

    manifest = run_real_translate(
        tmp_path, monkeypatch, segments, translates={1: "大家早上好"})

    assert manifest["segments"][1]["tgt_text"] == "", "untranslatable, as expected"
    assert settled(manifest), "and it must stop blocking every future export"
    assert 129 in proofread.effective_cleared_segment_ids(manifest)


def test_the_real_pass_leaves_a_total_failure_to_be_retried(tmp_path, monkeypatch):
    segments = [
        {"id": 1, "src_text": "みんなおはようございます", "tgt_text": ""},
        {"id": 2, "src_text": "よろしく", "tgt_text": ""},
    ]

    manifest = run_real_translate(tmp_path, monkeypatch, segments, translates={})

    assert not settled(manifest), "an API failure is not a finished video"
    assert not proofread.effective_cleared_segment_ids(manifest)
