"""The batch "keep intermediate files (debug)" option.

The box was threaded from the GUI through every stage but never read, so it had
never removed anything. These cover both the deletion itself and the wiring,
since the wiring is what was actually broken.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tool_clonevoice_v2 import backend, logic


def _make_clone_dir(tmp_path: Path) -> tuple[Path, Path]:
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"")
    cdir = logic.clone_dir(video)
    (cdir / "indextts_v2_manifest").mkdir(parents=True)
    (cdir / "indextts_v2_manifest" / "out_00001.wav").write_bytes(b"")
    (cdir / "audio16k.wav").write_bytes(b"")
    (cdir / "manifest.json").write_text("{}", encoding="utf-8")
    (cdir / "translated.srt").write_text("", encoding="utf-8")
    return video, cdir


def test_cleanup_removes_the_whole_clone_directory(tmp_path):
    video, cdir = _make_clone_dir(tmp_path)
    assert cdir.is_dir()

    assert logic.cleanup_clone_dir(video, log=lambda _m: None) is True
    assert not cdir.exists()


def test_cleanup_keeps_the_dub_background_bed(tmp_path):
    """The bed lives beside the video, not inside .clone, and dub reuses it."""
    video, _cdir = _make_clone_dir(tmp_path)
    bed = video.with_suffix(".dub_bg.wav")
    bed.write_bytes(b"")
    sidecar = Path(str(bed) + ".json")
    sidecar.write_text("{}", encoding="utf-8")

    logic.cleanup_clone_dir(video, log=lambda _m: None)

    assert bed.is_file()
    assert sidecar.is_file()
    assert video.is_file()


def test_cleanup_is_a_noop_when_there_is_nothing_to_remove(tmp_path):
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"")
    assert logic.cleanup_clone_dir(video, log=lambda _m: None) is False


def test_cleanup_refuses_a_path_that_is_not_a_clone_dir(tmp_path, monkeypatch):
    """Safety net: a malformed video path must not turn this into a wide delete."""
    victim = tmp_path / "important"
    victim.mkdir()
    (victim / "keep.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(logic, "clone_dir", lambda _v: victim)

    messages: list[str] = []
    assert logic.cleanup_clone_dir(tmp_path / "movie.mp4", log=messages.append) is False
    assert victim.is_dir()
    assert (victim / "keep.txt").is_file()
    assert any("refusing" in m for m in messages)


# --- wiring: this is what was broken ---

@pytest.fixture
def stubbed_stages(monkeypatch, tmp_path):
    """Replace the three real stages so run_full/run_batch can be driven."""
    cleaned: list[Path] = []

    def fake_synthesize(video, **kwargs):
        out = Path(video).with_suffix(".si.wav")
        out.write_bytes(b"")
        return str(out)

    # run_batch loads the shared IndexTTS model itself; without this it would
    # try to fetch the real checkpoints and hang.
    monkeypatch.setattr(backend, "load_model", lambda *a, **k: object())
    monkeypatch.setattr(logic, "run_transcribe_diarize", lambda *a, **k: {})
    monkeypatch.setattr(logic, "run_translate", lambda *a, **k: {})
    monkeypatch.setattr(logic, "run_synthesize", fake_synthesize)
    monkeypatch.setattr(logic, "release_model_holder", lambda *a, **k: None)
    monkeypatch.setattr(logic, "transcription_checkpoint_ready", lambda *a, **k: False)

    real_cleanup = logic.cleanup_clone_dir

    def tracking_cleanup(video_path, **kwargs):
        cleaned.append(Path(video_path))
        return real_cleanup(video_path, **kwargs)

    monkeypatch.setattr(logic, "cleanup_clone_dir", tracking_cleanup)
    return cleaned


def test_run_full_removes_intermediates_when_the_box_is_unchecked(tmp_path, stubbed_stages):
    video, cdir = _make_clone_dir(tmp_path)

    logic.run_full(
        video, models_root=str(tmp_path), keep_intermediate=False,
        skip_existing=False, log=lambda _m: None,
    )

    assert stubbed_stages == [video]
    assert not cdir.exists()


def test_run_full_keeps_intermediates_when_the_box_is_checked(tmp_path, stubbed_stages):
    video, cdir = _make_clone_dir(tmp_path)

    logic.run_full(
        video, models_root=str(tmp_path), keep_intermediate=True,
        skip_existing=False, log=lambda _m: None,
    )

    assert stubbed_stages == []
    assert (cdir / "manifest.json").is_file()


def test_run_batch_removes_intermediates_for_every_video_it_synthesized(tmp_path, stubbed_stages):
    videos, dirs = [], []
    for name in ("a", "b"):
        video = tmp_path / f"{name}.mp4"
        video.write_bytes(b"")
        cdir = logic.clone_dir(video)
        cdir.mkdir()
        (cdir / "manifest.json").write_text("{}", encoding="utf-8")
        videos.append(video)
        dirs.append(cdir)

    logic.run_batch(
        videos, models_root=str(tmp_path), keep_intermediate=False,
        skip_existing=False, log=lambda _m: None,
    )

    assert stubbed_stages == videos
    assert not any(d.exists() for d in dirs)


def test_run_batch_leaves_a_skipped_video_alone(tmp_path, stubbed_stages):
    """A video skipped because its .SI.WAV is up to date had no work done."""
    import json

    video = tmp_path / "done.mp4"
    video.write_bytes(b"")
    video.with_suffix(".si.wav").write_bytes(b"")
    cdir = logic.clone_dir(video)
    cdir.mkdir()
    # An empty manifest is "up to date" only once it says so: the check is on
    # the recorded signature now, not on the output file merely existing.
    manifest = {"segments": []}
    manifest["synthesis"] = {
        # run_batch defaults to Chinese; the signature covers the target
        # language, so it has to be built with the same one.
        "signature": logic.synthesis_signature(manifest, language="Chinese")
    }
    (cdir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )

    logic.run_batch(
        [video], models_root=str(tmp_path), keep_intermediate=False,
        skip_existing=True, log=lambda _m: None,
    )

    assert stubbed_stages == []
    assert (cdir / "manifest.json").is_file()
