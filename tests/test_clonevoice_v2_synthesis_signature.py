"""«Skip when the target .SI.WAV is up to date» has to mean up to date.

It used to mean "does the file exist". Proofreading a video and exporting again
with the box ticked therefore skipped the whole video and dropped the edits,
with `skipped` in the log as the only trace -- the more useful proofreading
became, the more this cost.
"""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from tool_clonevoice_v2 import logic


def _exported(tmp_path, segments):
    from tool_si import logic as si

    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    manifest = {
        "video": str(video), "language": "ja", "target_language": "Chinese",
        "segments": segments,
    }
    logic.save_manifest(video, manifest)
    sf.write(si.default_si_audio_path(str(video)),
             np.zeros(2205, dtype=np.float32), 22050)
    return video, manifest


def _segments():
    return [
        {"id": 1, "start": 0.0, "end": 1.0, "src_text": "a", "tgt_text": "甲",
         "speaker": "SPEAKER_00"},
        {"id": 2, "start": 2.0, "end": 3.0, "src_text": "b", "tgt_text": "乙",
         "speaker": "SPEAKER_01"},
    ]


def _mark_exported(video, manifest, **kwargs):
    manifest["synthesis"] = {"signature": logic.synthesis_signature(manifest, **kwargs)}
    logic.save_manifest(video, manifest)


def test_an_untouched_export_is_up_to_date(tmp_path):
    video, manifest = _exported(tmp_path, _segments())
    _mark_exported(video, manifest)
    assert logic.synthesis_is_current(video)


def test_editing_a_translation_makes_it_stale(tmp_path):
    video, manifest = _exported(tmp_path, _segments())
    _mark_exported(video, manifest)
    manifest["segments"][0]["tgt_text"] = "改过了"
    logic.save_manifest(video, manifest)
    assert not logic.synthesis_is_current(video)


def test_reassigning_a_speaker_makes_it_stale(tmp_path):
    """The speaker decides which cloned voice the line gets, so the audio is
    stale even though every word is unchanged."""
    video, manifest = _exported(tmp_path, _segments())
    _mark_exported(video, manifest)
    manifest["segments"][1]["speaker"] = "SPEAKER_00"
    logic.save_manifest(video, manifest)
    assert not logic.synthesis_is_current(video)


def test_retiming_a_line_makes_it_stale(tmp_path):
    video, manifest = _exported(tmp_path, _segments())
    _mark_exported(video, manifest)
    manifest["segments"][0]["end"] = 1.4
    logic.save_manifest(video, manifest)
    assert not logic.synthesis_is_current(video)


def test_dropping_a_line_makes_it_stale(tmp_path):
    video, manifest = _exported(tmp_path, _segments())
    _mark_exported(video, manifest)
    manifest["segments"][0]["tgt_text"] = ""
    logic.save_manifest(video, manifest)
    assert not logic.synthesis_is_current(video)


@pytest.mark.parametrize("setting", [
    {"tempo_fit": "strong"}, {"level_match": False}, {"timbre_anchor": False},
    {"language": "English"},
])
def test_changing_a_synthesis_setting_makes_it_stale(tmp_path, setting):
    video, manifest = _exported(tmp_path, _segments())
    _mark_exported(video, manifest)
    assert not logic.synthesis_is_current(video, **setting)


def test_an_untranslated_line_does_not_affect_the_signature(tmp_path):
    """Nothing is synthesized for it, so it cannot change the output."""
    video, manifest = _exported(tmp_path, _segments())
    before = logic.synthesis_signature(manifest)
    manifest["segments"].append(
        {"id": 3, "start": 5.0, "end": 6.0, "src_text": "c", "tgt_text": ""}
    )
    assert logic.synthesis_signature(manifest) == before


def test_a_missing_output_is_never_up_to_date(tmp_path):
    from tool_si import logic as si

    video, manifest = _exported(tmp_path, _segments())
    _mark_exported(video, manifest)
    Path = type(video)
    Path(si.default_si_audio_path(str(video))).unlink()
    assert not logic.synthesis_is_current(video)


def test_an_export_from_before_the_marker_is_redone_once(tmp_path):
    """Old .si.wav files carry no signature; their provenance is unknown, so
    they are re-exported rather than trusted."""
    video, _manifest = _exported(tmp_path, _segments())
    assert not logic.synthesis_is_current(video)


def test_both_entry_points_ask_the_same_question(tmp_path):
    import inspect

    from tool_clonevoice_v2 import single_clone

    for fn in (logic.run_batch, single_clone.translate_and_synthesize):
        assert "synthesis_is_current" in inspect.getsource(fn), fn.__name__


def test_neither_checkbox_still_promises_to_skip_on_mere_existence():
    """Both boxes now gate on the recorded signature, so a label saying "if it
    exists" describes behaviour the code no longer has -- and the batch one
    covers checkpoint reuse as well, which its label has to admit."""
    import json
    from pathlib import Path

    stale = {
        "zh": ["存在就忽略", "存在则忽略", "跳过已存在"],
        "en": ["if it exists", "when it exists", "skip existing"],
        "ja": ["存在する場合はスキップ", "があればスキップ"],
    }
    for lang, phrases in stale.items():
        data = json.loads(Path(f"i18n/{lang}.json").read_text(encoding="utf-8-sig"))
        flat: dict[str, str] = {}

        def walk(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    walk(value) if isinstance(value, dict) else flat.__setitem__(key, value)

        walk(data)
        for key in ("chk_skip_existing", "chk_single_skip_existing_si"):
            text = str(flat.get(key, ""))
            assert text, f"{lang}/{key} is missing"
            for phrase in phrases:
                assert phrase.lower() not in text.lower(), f"{lang}/{key}: {text}"


def test_no_message_still_tells_the_user_to_delete_or_untick():
    """Advice from when "skip if the .SI.WAV exists" meant exactly that: the
    only way to make an edit count was to delete the output or untick the box.
    The export detects staleness on its own now, and deleting the file would
    only throw away the lines that did not change."""
    import json
    from pathlib import Path

    stale = {
        "zh": ["请删除", "删除该文件", "取消勾选"],
        "en": ["delete it before", "turn off the skip"],
        "ja": ["削除するか", "無効にしてください"],
    }
    for lang, phrases in stale.items():
        data = json.loads(Path(f"i18n/{lang}.json").read_text(encoding="utf-8-sig"))
        flat: dict[str, str] = {}

        def walk(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    walk(value) if isinstance(value, dict) else flat.__setitem__(key, value)

        walk(data)
        text = str(flat.get("msg_pf_si_exists_warn", ""))
        assert text, f"{lang}: message is missing"
        for phrase in phrases:
            assert phrase.lower() not in text.lower(), f"{lang}: {text}"
