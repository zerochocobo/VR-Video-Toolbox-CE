"""A basis chosen by hand has to be the voice the dub is cloned in.

The refined-clone tab's step 2 -- audition the candidates, pick a speaker's
voice -- wrote its choice to manifest["speakers"][spk]["ref_audio"], and
synthesis never read it. Every title was cloned from the automatic anchor no
matter what was picked. On 3dsvr-1911 part 2 that anchor was a line from the
middle of a sex scene, 10.5dB over the title median and 88% periodic, and it
set the voice of all 247 lines.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tool_clonevoice_v2 import logic


@pytest.fixture
def video(tmp_path, monkeypatch):
    path = tmp_path / "title.mp4"
    path.write_bytes(b"")
    cdir = tmp_path / "title.clone"
    cdir.mkdir()
    monkeypatch.setattr(logic, "clone_dir", lambda _v: cdir)
    return path


def manifest_with(speakers):
    return {"segments": [{"id": 1, "start": 0.0, "end": 2.0, "tgt_text": "你好"}],
            "speakers": speakers}


def test_a_chosen_basis_is_handed_to_synthesis(video):
    cdir = logic.clone_dir(video)
    (cdir / "SPEAKER_00.basis.wav").write_bytes(b"RIFF")

    refs = logic.chosen_speaker_refs(
        video, manifest_with({"SPEAKER_00": {"ref_audio": "SPEAKER_00.basis.wav"}})
    )

    assert refs == {"SPEAKER_00": str(cdir / "SPEAKER_00.basis.wav")}


def test_a_basis_that_no_longer_exists_is_reported_not_swallowed(video):
    """The fallback is a different voice, so it must not be silent."""
    said = []

    refs = logic.chosen_speaker_refs(
        video, manifest_with({"SPEAKER_00": {"ref_audio": "gone.wav"}}),
        log=said.append,
    )

    assert refs == {}
    assert any("missing" in line for line in said)


def test_speakers_without_a_basis_are_left_to_the_ranking(video):
    cdir = logic.clone_dir(video)
    (cdir / "SPEAKER_01.basis.wav").write_bytes(b"RIFF")

    refs = logic.chosen_speaker_refs(video, manifest_with({
        "SPEAKER_00": {"ref_audio": ""},
        "SPEAKER_01": {"ref_audio": "SPEAKER_01.basis.wav"},
    }))

    assert set(refs) == {"SPEAKER_01"}


def test_a_manifest_with_no_speakers_block_is_fine(video):
    assert logic.chosen_speaker_refs(video, {"segments": []}) == {}


def test_changing_the_basis_re_exports_the_dub(video):
    """Swapping a basis changes every line that speaker has, so it has to
    invalidate the export as surely as editing the text does."""
    cdir = logic.clone_dir(video)
    basis = cdir / "SPEAKER_00.basis.wav"
    basis.write_bytes(b"first take")
    manifest = manifest_with({"SPEAKER_00": {"ref_audio": "SPEAKER_00.basis.wav"}})

    refs = logic.chosen_speaker_refs(video, manifest)
    before = logic.synthesis_signature(manifest, speaker_refs=refs)
    basis.write_bytes(b"a different clip entirely")
    after = logic.synthesis_signature(manifest, speaker_refs=refs)

    assert before != after


def test_the_same_clip_under_a_different_name_is_not_a_change(video):
    """Content, not path: re-picking the same audio must not force a re-export."""
    cdir = logic.clone_dir(video)
    (cdir / "a.wav").write_bytes(b"same audio")
    (cdir / "b.wav").write_bytes(b"same audio")
    manifest = manifest_with({"SPEAKER_00": {"ref_audio": "a.wav"}})

    first = logic.synthesis_signature(
        manifest, speaker_refs=logic.chosen_speaker_refs(video, manifest))
    manifest["speakers"]["SPEAKER_00"]["ref_audio"] = "b.wav"
    second = logic.synthesis_signature(
        manifest, speaker_refs=logic.chosen_speaker_refs(video, manifest))

    assert first == second


def test_run_synthesize_hands_the_chosen_basis_down_to_the_backend(video, monkeypatch):
    """The seam that was broken: the choice reaching the synthesizer at all."""
    from tool_clonevoice_v2 import backend

    cdir = logic.clone_dir(video)
    (cdir / "SPEAKER_00.basis.wav").write_bytes(b"RIFF")
    (cdir / logic.AUDIO16K_NAME).write_bytes(b"RIFF")
    manifest = manifest_with({"SPEAKER_00": {"ref_audio": "SPEAKER_00.basis.wav"}})
    manifest["target_language"] = "Chinese"
    (cdir / logic.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

    seen = {}

    def fake_synthesize(_model, _source, _segments, output_path, **kwargs):
        seen.update(kwargs)
        Path(output_path).write_bytes(b"RIFF")
        return str(output_path)

    monkeypatch.setattr(backend, "synthesize_manifest", fake_synthesize)
    monkeypatch.setattr(logic, "_release_cuda_cache", lambda: None)
    monkeypatch.setattr(logic, "log_memory", lambda *_a, **_k: None)
    monkeypatch.setattr("tool_si.logic.write_duck_key_wav", lambda *_a, **_k: None)

    logic.run_synthesize(video, models_root="models", model=object(), log=lambda _m: None)

    assert seen.get("speaker_refs") == {"SPEAKER_00": str(cdir / "SPEAKER_00.basis.wav")}
