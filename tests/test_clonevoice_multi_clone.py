from __future__ import annotations

import json
import wave
from pathlib import Path
from unittest.mock import patch

from tool_clonevoice import logic
from tool_clonevoice import multi_clone as mc


def _write_wav(path: Path, frames: int, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(b"\0\0" * frames)


def _save_multi_manifest(video: Path) -> None:
    logic.save_manifest(
        video,
        {
            "video": str(video),
            "language": "ja",
            "target_language": "Chinese",
            "speakers": {
                "SPEAKER_00": {"ref_audio": "", "ref_text": "", "score": 0.0},
                "SPEAKER_01": {"ref_audio": "", "ref_text": "", "score": 0.0},
            },
            "segments": [
                {"id": 1, "start": 0.0, "end": 2.0, "dur": 2.0, "speaker": "SPEAKER_01", "src_text": "b"},
                {"id": 2, "start": 2.0, "end": 7.5, "dur": 5.5, "speaker": "SPEAKER_00", "src_text": "a"},
                {"id": 3, "start": 8.0, "end": 9.0, "dur": 1.0, "speaker": "SPEAKER_00", "src_text": "c"},
            ],
        },
    )


def test_list_speakers_sorted_by_duration(tmp_path: Path):
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    _save_multi_manifest(video)

    speakers = mc.list_speakers(video)

    assert speakers == [
        {"speaker": "SPEAKER_00", "total_dur": 6.5, "seg_count": 2},
        {"speaker": "SPEAKER_01", "total_dur": 2.0, "seg_count": 1},
    ]


def test_collect_speaker_candidates_uses_isolated_directory_and_filters_empty_text(tmp_path: Path):
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    _save_multi_manifest(video)
    cdir = logic.clone_dir(video)
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / logic.AUDIO16K_NAME).write_bytes(b"wav")
    logs = []

    def fake_collect(video_arg, manifest, audio16k, clone_dir, *, speaker, top_n, output_dir_name, log):
        assert video_arg == str(video)
        assert speaker == "SPEAKER_01"
        assert top_n == 3
        assert output_dir_name == "candidates_SPEAKER_01"
        return [
            {"id": "empty", "src_text": "", "score": 99.0},
            {"id": "spoken", "src_text": "hello", "score": 1.0},
        ]

    with patch("tool_clonevoice.multi_clone.refsel.collect_reference_candidates", side_effect=fake_collect):
        candidates = mc.collect_speaker_candidates(video, "SPEAKER_01", top_n=3, log=logs.append)

    assert [cand["id"] for cand in candidates] == ["spoken"]
    assert candidates[0]["global_rank"] == 1
    assert any("without source transcript text" in message for message in logs)


def test_save_speaker_basis_updates_only_target_speaker_and_preserves_segments(tmp_path: Path):
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    _save_multi_manifest(video)
    basis = tmp_path / "basis.wav"
    basis.write_bytes(b"wav")

    wav_path, txt_path = mc.save_speaker_basis(
        video,
        "SPEAKER_01",
        basis_wav=basis,
        basis_text="fixed target sentence",
        target_language="Chinese",
        source_kind="candidate_target_sample",
        meta={"candidate_id": "cand_002"},
        log=lambda _m: None,
    )

    cdir = logic.clone_dir(video)
    assert Path(wav_path) == cdir / "SPEAKER_01.basis.wav"
    assert Path(txt_path).read_text(encoding="utf-8") == "fixed target sentence"
    manifest = logic.load_manifest(video)
    assert manifest["speakers"]["SPEAKER_01"]["ref_audio"] == "SPEAKER_01.basis.wav"
    assert manifest["speakers"]["SPEAKER_01"]["ref_text"] == "fixed target sentence"
    assert manifest["speakers"]["SPEAKER_01"]["ref_language"] == "Chinese"
    assert manifest["speakers"]["SPEAKER_01"]["skip_work_ref"] is True
    assert manifest["speakers"]["SPEAKER_00"]["ref_audio"] == ""
    assert [seg["speaker"] for seg in manifest["segments"]] == ["SPEAKER_01", "SPEAKER_00", "SPEAKER_00"]
    meta = json.loads((cdir / "SPEAKER_01.basis.meta.json").read_text(encoding="utf-8"))
    assert meta["candidate_id"] == "cand_002"
    assert meta["basis_wav"] == "SPEAKER_01.basis.wav"


def test_all_speakers_have_basis_respects_skipped_speakers(tmp_path: Path):
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    _save_multi_manifest(video)
    basis = tmp_path / "basis.wav"
    basis.write_bytes(b"wav")

    assert mc.all_speakers_have_basis(video) is False
    mc.save_speaker_basis(
        video,
        "SPEAKER_00",
        basis_wav=basis,
        basis_text="speaker zero",
        target_language="Chinese",
        source_kind="candidate_target_sample",
        log=lambda _m: None,
    )

    assert mc.all_speakers_have_basis(video) is False
    assert mc.all_speakers_have_basis(video, skipped={"SPEAKER_01"}) is True


def test_set_skipped_speakers_persists_and_clears_flags(tmp_path: Path):
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    _save_multi_manifest(video)

    mc.set_speaker_skipped(video, "SPEAKER_01", skipped=True, log=lambda _m: None)
    manifest = logic.load_manifest(video)
    assert manifest["speakers"]["SPEAKER_01"]["skip_synthesis"] is True
    assert "skip_synthesis" not in manifest["speakers"]["SPEAKER_00"]

    mc.set_skipped_speakers(video, {"SPEAKER_00"}, log=lambda _m: None)
    manifest = logic.load_manifest(video)
    assert manifest["speakers"]["SPEAKER_00"]["skip_synthesis"] is True
    assert "skip_synthesis" not in manifest["speakers"]["SPEAKER_01"]


def test_generate_voice_design_basis_uses_speaker_specific_output(tmp_path: Path):
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    calls = []

    def fake_generate(model, *, target_language, instruct, output_wav, log, stop_event):
        calls.append((target_language, instruct, output_wav))
        Path(output_wav).write_bytes(b"wav")
        return output_wav, "fixed target sample"

    with patch("tool_clonevoice.omnivoice_backend.generate_voice_design_sample_with_model", side_effect=fake_generate):
        wav_path, text = mc.generate_voice_design_basis_with_model(
            video,
            "SPEAKER_01",
            model=object(),
            target_language="Chinese",
            instruct="female, young adult",
            log=lambda _m: None,
        )

    assert Path(wav_path) == logic.clone_dir(video) / "SPEAKER_01.design.wav"
    assert Path(wav_path).is_file()
    assert text == "fixed target sample"
    assert calls[0][0] == "Chinese"
    assert calls[0][1] == "female, young adult"


def test_export_speaker_basis_writes_reusable_files(tmp_path: Path):
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    _save_multi_manifest(video)
    basis = tmp_path / "basis.wav"
    basis.write_bytes(b"wav")
    mc.save_speaker_basis(
        video,
        "SPEAKER_01",
        basis_wav=basis,
        basis_text="fixed target sentence",
        target_language="Chinese",
        source_kind="candidate_target_sample",
        meta={"candidate_id": "cand_001"},
        log=lambda _m: None,
    )

    out_dir = tmp_path / "exports"
    wav_path, txt_path, meta_path = mc.export_speaker_basis(video, "SPEAKER_01", out_dir, log=lambda _m: None)

    assert Path(wav_path) == out_dir / "movie.SPEAKER_01.basis.wav"
    assert Path(wav_path).read_bytes() == b"wav"
    assert Path(txt_path).read_text(encoding="utf-8") == "fixed target sentence"
    assert meta_path is not None
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    assert meta["candidate_id"] == "cand_001"


def test_save_speaker_basis_for_videos_only_updates_videos_containing_speaker(tmp_path: Path):
    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.mp4"
    video_a.write_bytes(b"video")
    video_b.write_bytes(b"video")
    _save_multi_manifest(video_a)
    _save_multi_manifest(video_b)
    manifest_b = logic.load_manifest(video_b)
    manifest_b["speakers"].pop("SPEAKER_01")
    for segment in manifest_b["segments"]:
        segment["speaker"] = "SPEAKER_00"
    logic.save_manifest(video_b, manifest_b)
    basis = tmp_path / "basis.wav"
    basis.write_bytes(b"wav")

    saved = mc.save_speaker_basis_for_videos(
        [video_a, video_b],
        "SPEAKER_01",
        basis_wav=basis,
        basis_text="speaker one",
        target_language="Chinese",
        source_kind="basis_reuse",
        log=lambda _m: None,
    )

    assert [Path(item[0]).name for item in saved] == ["a.mp4"]
    assert mc.speaker_has_basis(video_a, "SPEAKER_01") is True
    assert mc.speaker_has_basis(video_b, "SPEAKER_01") is False


def test_estimate_total_video_duration_uses_probe(tmp_path: Path):
    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.mp4"
    video_a.write_bytes(b"video")
    video_b.write_bytes(b"video")

    class Meta:
        def __init__(self, duration):
            self.duration = duration

    with patch(
        "gpu_engine.probe.probe_video",
        side_effect=[Meta(10.5), Meta(20.0)],
    ):
        total = mc.estimate_total_video_duration([video_a, video_b], log=lambda _m: None)

    assert total == 30.5


def test_split_turns_to_video_offsets_and_clips():
    turns = [
        (0.0, 1.0, "SPEAKER_00"),
        (1.8, 3.4, "SPEAKER_01"),
        (4.9, 5.05, "SPEAKER_00"),
        (5.5, 6.0, "SPEAKER_02"),
    ]

    local = mc.split_turns_to_video(turns, offset=2.0, duration=3.0)

    assert local == [(0.0, 1.4, "SPEAKER_01")]


def test_prescan_global_diarize_extracts_concatenates_and_splits(tmp_path: Path):
    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.mp4"
    video_a.write_bytes(b"video")
    video_b.write_bytes(b"video")

    def fake_extract(video, audio_out, **_kwargs):
        frames = 16000 if Path(video).name == "a.mp4" else 32000
        _write_wav(Path(audio_out), frames)

    diar_calls = []

    def fake_diarize(audio_path, **kwargs):
        diar_calls.append((audio_path, kwargs))
        return [
            (0.0, 0.6, "SPEAKER_00"),
            (1.2, 2.4, "SPEAKER_01"),
            (3.0, 4.1, "SPEAKER_00"),
        ]

    with patch("tool_clonevoice.multi_clone.wx.extract_audio_16k", side_effect=fake_extract), patch(
        "tool_clonevoice.multi_clone.diar.diarize", side_effect=fake_diarize
    ), patch("tool_clonevoice.multi_clone.wx.resolve_device", return_value=("cpu", None)):
        turns = mc.prescan_global_diarize(
            [video_a, video_b],
            models_root=str(tmp_path / "models"),
            diarize_backend="pyannote",
            num_speakers=2,
            silence_gap=1.0,
            log=lambda _m: None,
        )

    assert diar_calls[0][1]["backend"] == "pyannote"
    assert diar_calls[0][1]["num_speakers"] == 2
    assert turns[str(video_a)] == [(0.0, 0.6, "SPEAKER_00")]
    assert turns[str(video_b)] == [(0.0, 0.4, "SPEAKER_01"), (1.0, 2.0, "SPEAKER_00")]
    assert not (logic.clone_dir(video_a) / "global_diarize_concat.wav").exists()


def test_extract_shared_references_applies_one_ref_to_all_videos(tmp_path: Path):
    import numpy as np
    from tool_clonevoice import refsel

    videos = []
    for name in ("a.mp4", "b.mp4"):
        video = tmp_path / name
        video.write_bytes(b"video")
        logic.save_manifest(video, {
            "video": str(video),
            "language": "ja",
            "target_language": "Chinese",
            "speakers": {"SPEAKER_00": {"ref_audio": "", "ref_text": "", "score": 0.0}},
            "segments": [{"id": 1, "start": 0.0, "end": 6.0, "dur": 6.0,
                          "speaker": "SPEAKER_00", "src_text": "hi"}],
        })
        cdir = logic.clone_dir(video)
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / logic.AUDIO16K_NAME).write_bytes(b"wav")
        videos.append(str(video))

    def fake_cands(spk, manifest, segs, audio, sr):
        return [{"start": 0.0, "end": 6.0, "dur": 6.0, "text": "hi", "score": 0.9,
                 "source": "segment", "source_srt_refs": [], "speaker": spk}]

    with (
        patch.object(refsel, "_read_wav_mono", side_effect=lambda _p: (np.zeros(16000, np.float32), 16000)),
        patch.object(refsel, "_speaker_candidates", side_effect=fake_cands),
        patch.object(refsel, "_cut_ref", side_effect=lambda video, s, e, out, log: Path(out).write_bytes(b"ref")),
    ):
        mc.extract_shared_references(videos, log=lambda _m: None)

    for video in videos:
        manifest = logic.load_manifest(video)
        spk = manifest["speakers"]["SPEAKER_00"]
        assert spk["ref_audio"] == "ref_SPEAKER_00.wav"
        assert spk["ref_text"] == "hi"
        assert "skip_work_ref" not in spk  # auto uses source ref + work_ref at synth
        assert (logic.clone_dir(video) / "ref_SPEAKER_00.wav").is_file()


def test_run_multi_transcribe_passes_backend_and_speaker_count(tmp_path: Path):
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    calls = []

    def fake_run(video_arg, **kwargs):
        calls.append((video_arg, kwargs))
        return {"ok": True}

    with patch("tool_clonevoice.multi_clone.logic.run_transcribe_diarize", side_effect=fake_run):
        result = mc.run_multi_transcribe(
            video,
            model_key="large-v3",
            language="ja",
            target_language="Chinese",
            models_root=str(tmp_path / "models"),
            diarize_backend="ecapa",
            num_speakers=3,
            denoise="mild",
            precomputed_turns=[(0.0, 1.0, "SPEAKER_00")],
            log=lambda _m: None,
        )

    assert result == {"ok": True}
    assert calls[0][0] == video
    assert calls[0][1]["diarize_backend"] == "ecapa"
    assert calls[0][1]["num_speakers"] == 3
    assert calls[0][1]["denoise"] == "mild"
    assert calls[0][1]["precomputed_turns"] == [(0.0, 1.0, "SPEAKER_00")]
