from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tool_clonevoice import logic
from tool_clonevoice import omnivoice_backend as ov
from tool_clonevoice import single_clone as sc
from tool_si import logic as si_logic


def test_user_visible_speaker1_paths_avoid_single_file_overwrite(tmp_path: Path):
    video = tmp_path / "sample.mp4"
    batch_root = tmp_path / "batch"

    single_wav, single_txt = sc.user_visible_speaker1_paths(video, batch=False)
    batch_wav, batch_txt = sc.user_visible_speaker1_paths(batch_root, batch=True)

    assert single_wav == tmp_path / "sample.SPEAKER1.wav"
    assert single_txt == tmp_path / "sample.SPEAKER1.txt"
    assert batch_wav == batch_root / "SPEAKER1.wav"
    assert batch_txt == batch_root / "SPEAKER1.txt"


def test_collect_candidates_for_videos_filters_empty_source_text():
    logs = []

    def fake_collect(video, *, top_n, log):
        return [
            {"id": "silent", "src_text": "", "score": 999.0},
            {"id": "spoken", "src_text": "hello", "score": 1.0},
        ]

    with patch("tool_clonevoice.single_clone.collect_single_candidates", side_effect=fake_collect):
        result = sc.collect_candidates_for_videos(["movie.mp4"], per_video=2, total=2, log=logs.append)

    assert [candidate["id"] for candidate in result] == ["spoken"]
    assert result[0]["global_rank"] == 1
    assert any("without source transcript text" in message for message in logs)


def test_load_existing_candidates_for_videos_filters_and_ranks(tmp_path: Path):
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    cand_dir = logic.clone_dir(video) / "single_candidates"
    cand_dir.mkdir(parents=True)
    source_a = cand_dir / "cand_001_src.wav"
    source_b = cand_dir / "cand_002_src.wav"
    target_a = cand_dir / "cand_001_target.wav"
    translated_a = cand_dir / "cand_001_translated.wav"
    source_a.write_bytes(b"src-a")
    source_b.write_bytes(b"src-b")
    target_a.write_bytes(b"target-a")
    translated_a.write_bytes(b"translated-a")
    (cand_dir / "candidates.json").write_text(
        json.dumps(
            [
                {
                    "id": "cand_001",
                    "source_audio": str(source_a),
                    "target_sample_audio": str(target_a),
                    "translated_audio": str(translated_a),
                    "target_sample_text": "fixed sample",
                    "src_text": "source a",
                    "dur": 4.0,
                    "start": 2.0,
                    "ecapa_similarity": 0.8,
                    "ecapa_similarity_basis": "translated_audio",
                },
                {
                    "id": "cand_002",
                    "source_audio": str(source_b),
                    "target_sample_audio": str(cand_dir / "missing_target.wav"),
                    "translated_audio": "",
                    "src_text": "source b",
                    "dur": 9.0,
                    "start": 1.0,
                    "ecapa_similarity": None,
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    loaded = sc.load_existing_candidates_for_videos([str(video)], total=12, log=lambda _m: None)

    assert [cand["id"] for cand in loaded] == ["cand_001", "cand_002"]
    assert loaded[0]["global_rank"] == 1
    assert loaded[0]["translated_audio"] == str(translated_a)
    assert loaded[1]["target_sample_audio"] == ""


def test_collect_candidates_with_existing_collects_missing_videos(tmp_path: Path):
    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.mp4"
    video_c = tmp_path / "c.mp4"
    for video in (video_a, video_b, video_c):
        video.write_bytes(b"video")
    existing = [
        {
            "id": "cached_a",
            "video": str(video_a),
            "source_audio": str(tmp_path / "cached_a.wav"),
            "src_text": "cached",
            "dur": 8.0,
            "start": 0.0,
            "ecapa_similarity": 0.9,
        }
    ]
    calls: list[str] = []

    def fake_collect(video, *, top_n, log):
        calls.append(Path(video).name)
        return [
            {
                "id": f"{Path(video).stem}_cand",
                "video": str(video),
                "source_audio": str(tmp_path / f"{Path(video).stem}.wav"),
                "src_text": "fresh",
                "dur": 6.0,
                "start": 1.0,
            }
        ]

    with patch("tool_clonevoice.single_clone.collect_single_candidates", side_effect=fake_collect):
        result = sc.collect_candidates_with_existing_for_videos(
            [str(video_a), str(video_b), str(video_c)],
            existing,
            per_video=2,
            total=12,
            log=lambda _m: None,
        )

    assert calls == ["b.mp4", "c.mp4"]
    assert [cand["id"] for cand in result] == ["cached_a", "b_cand", "c_cand"]


def test_save_speaker1_basis_writes_manifest_basis_and_visible_copy(tmp_path: Path):
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    basis_wav = tmp_path / "basis.wav"
    basis_wav.write_bytes(b"wav")
    logic.save_manifest(video, {
        "video": str(video),
        "language": "ja",
        "target_language": "Chinese",
        "speakers": {},
        "segments": [
            {"id": 1, "start": 0.0, "end": 1.0, "speaker": "OLD", "src_text": "a"},
            {"id": 2, "start": 1.0, "end": 2.0, "speaker": "OLD", "src_text": "b"},
        ],
    })

    visible_wav, visible_txt = sc.save_speaker1_basis(
        [str(video)],
        basis_wav=str(basis_wav),
        basis_text="fixed target sentence",
        target_language="Chinese",
        visible_target=str(video),
        batch=False,
        source_kind="candidate_target_sample",
        meta={"candidate_id": "cand_001"},
        log=lambda _m: None,
    )

    assert Path(visible_wav) == tmp_path / "movie.SPEAKER1.wav"
    assert Path(visible_txt).read_text(encoding="utf-8") == "fixed target sentence"
    clone_dir = logic.clone_dir(video)
    assert (clone_dir / "SPEAKER1.wav").read_bytes() == b"wav"
    assert (clone_dir / "SPEAKER1.txt").read_text(encoding="utf-8") == "fixed target sentence"
    manifest = logic.load_manifest(video)
    speaker = manifest["speakers"][sc.SPEAKER_ID]
    assert speaker["ref_audio"] == "SPEAKER1.wav"
    assert speaker["ref_text"] == "fixed target sentence"
    assert speaker["ref_language"] == "Chinese"
    assert speaker["skip_work_ref"] is True
    assert {seg["speaker"] for seg in manifest["segments"]} == {sc.SPEAKER_ID}
    meta = json.loads((clone_dir / "SPEAKER1.meta.json").read_text(encoding="utf-8"))
    assert meta["candidate_id"] == "cand_001"
    assert meta["basis_wav"] == "SPEAKER1.wav"


def test_save_speaker1_basis_accepts_existing_visible_wav(tmp_path: Path):
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    visible_wav = tmp_path / "movie.SPEAKER1.wav"
    visible_wav.write_bytes(b"existing wav")

    wav_path, txt_path = sc.save_speaker1_basis(
        [str(video)],
        basis_wav=visible_wav,
        basis_text="fixed target sentence",
        target_language="Chinese",
        visible_target=video,
        batch=False,
        source_kind="existing_manifest",
        log=lambda _m: None,
    )

    assert Path(wav_path) == visible_wav
    assert Path(txt_path).read_text(encoding="utf-8") == "fixed target sentence"
    assert (logic.clone_dir(video) / "SPEAKER1.wav").read_bytes() == b"existing wav"


def test_translate_and_synthesize_uses_translate_then_synthesize_not_run_full(tmp_path: Path):
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    calls = []

    with (
        patch("tool_si.logic.default_si_audio_path", return_value=str(tmp_path / "movie.si.wav")),
        patch("tool_clonevoice.single_clone.logic.run_translate", side_effect=lambda video, **kwargs: calls.append(("translate", str(video), kwargs))),
        patch("tool_clonevoice.single_clone.logic.run_synthesize", side_effect=lambda video, **kwargs: calls.append(("synthesize", str(video), kwargs)) or str(tmp_path / "movie.si.wav")),
        patch("tool_clonevoice.single_clone.logic.run_full", side_effect=AssertionError("run_full must not be used")),
        patch("tool_clonevoice.single_clone.logic.run_extract_references", side_effect=AssertionError("run_extract_references must not be used")),
    ):
        result = sc.translate_and_synthesize(
            [str(video)],
            target_language="Chinese",
            models_root=str(tmp_path / "models"),
            api_key="secret",
            skip_existing=False,
            log=lambda _m: None,
        )

    assert result["outputs"] == [str(tmp_path / "movie.si.wav")]
    assert result["written"] == [str(tmp_path / "movie.si.wav")]
    assert result["skipped"] == []
    assert [call[0] for call in calls] == ["translate", "synthesize"]
    assert calls[0][2]["api_key"] == "secret"
    assert calls[1][2]["text_field"] == "tgt_text"
    assert calls[1][2]["language"] == "Chinese"


def test_translate_and_synthesize_skip_existing_does_not_translate_or_clone(tmp_path: Path):
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    out = tmp_path / "movie.si.wav"
    out.write_bytes(b"existing")
    logs = []

    with (
        patch("tool_si.logic.default_si_audio_path", return_value=str(out)),
        patch("tool_clonevoice.single_clone.logic.run_translate", side_effect=AssertionError("translate must not run")),
        patch("tool_clonevoice.single_clone.logic.run_synthesize", side_effect=AssertionError("synthesize must not run")),
    ):
        result = sc.translate_and_synthesize(
            [str(video)],
            target_language="Chinese",
            models_root=str(tmp_path / "models"),
            skip_existing=True,
            log=logs.append,
        )

    assert result["outputs"] == [str(out)]
    assert result["written"] == []
    assert result["skipped"] == [str(out)]
    assert "skipped translation and cloning" in logs[0]


def test_language_normalization_supports_aliases_and_generic_texts():
    assert ov.same_language("Chinese", "zh")
    assert ov.same_language("中文", "chinese")
    assert ov.same_language("Thai", "tha")
    text, duration = ov.generic_ref_text("zh-CN")
    assert text
    assert 6.0 <= duration <= 14.0


def test_prepare_prompt_reference_audio_adds_silent_tail(tmp_path: Path):
    import wave

    source = tmp_path / "ref.wav"
    pcm = (np.full(2400, 0.2, dtype=np.float32) * 32767.0).astype("<i2")
    with wave.open(str(source), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(pcm.tobytes())

    out = Path(ov.prepare_prompt_reference_audio(source, log=lambda _m: None))
    wav, sr = ov._read_pcm_wav_mono_f32(out)

    assert out.name == "ref_prompt.wav"
    assert sr == 24000
    assert wav.size > 2400
    assert np.max(np.abs(wav[-int(0.25 * sr) :])) < 1e-4


def test_build_candidate_target_sample_job_uses_existing_model(tmp_path: Path):
    source = tmp_path / "cand_001_src.wav"
    candidate = {"id": "cand_001", "source_audio": str(source), "src_text": "hello"}
    model = _FakePromptModel()

    job = sc.build_candidate_target_sample_job(
        candidate,
        model=model,
        target_language="Chinese",
        log=lambda _m: None,
    )

    assert model.generate_calls == ov.WORK_REF_TAKES
    assert job["candidate"] is candidate
    assert Path(job["output_wav"]).name == "cand_001_target.wav"
    assert job["generic"]


def test_build_candidate_target_sample_job_defers_translated_preview(tmp_path: Path):
    source = tmp_path / "cand_001_src.wav"
    candidate = {
        "id": "cand_001",
        "source_audio": str(source),
        "src_text": "hello",
        "tgt_text": "hello target",
    }
    model = _FakePromptModel()

    job = sc.build_candidate_target_sample_job(
        candidate,
        model=model,
        target_language="Chinese",
        log=lambda _m: None,
    )

    assert model.generate_calls == ov.WORK_REF_TAKES
    assert "translated_audio" not in job
    assert "translated_audio" not in candidate


def test_generate_candidate_translated_preview_uses_target_sample_prompt(tmp_path: Path):
    source = tmp_path / "cand_001_src.wav"
    target = tmp_path / "cand_001_target.wav"
    source.write_bytes(b"not a real wav")
    target.write_bytes(b"not a real wav")
    candidate = {
        "id": "cand_001",
        "source_audio": str(source),
        "target_sample_audio": str(target),
        "target_sample_text": "fixed target sample",
        "tgt_text": "translated sentence",
    }
    model = _FakePromptModel()

    updated = sc.generate_candidate_translated_previews_with_model(
        [candidate],
        model=model,
        target_language="Chinese",
        log=lambda _m: None,
    )

    assert updated == [candidate]
    assert model.prompts == [(str(target), "fixed target sample", True)]
    assert model.generate_calls == 1
    assert "voice_clone_prompt" in model.generate_kwargs[0]
    assert "ref_audio" not in model.generate_kwargs[0]
    assert Path(candidate["translated_audio"]).name == "cand_001_translated.wav"
    assert Path(candidate["translated_audio"]).is_file()


def test_finish_candidate_target_sample_jobs_batches_ecapa_work(tmp_path: Path):
    candidate_a = {"id": "cand_001", "source_audio": str(tmp_path / "a_src.wav")}
    candidate_b = {"id": "cand_002", "source_audio": str(tmp_path / "b_src.wav")}
    existing = {
        "id": "cand_003",
        "source_audio": str(tmp_path / "c_src.wav"),
        "target_sample_audio": str(tmp_path / "c_target.wav"),
        "translated_audio": str(tmp_path / "c_translated.wav"),
        "ecapa_similarity": None,
    }
    jobs = [
        {"candidate": candidate_a, "device": "cpu"},
        {"candidate": candidate_b, "device": "cpu"},
    ]

    def fake_process(jobs_arg, *, score_pairs, models_root, device, log):
        assert jobs_arg == jobs
        assert score_pairs == [(existing["source_audio"], existing["translated_audio"])]
        assert device == "cpu"
        return [
            (str(tmp_path / "a_target.wav"), "text a", 0.91),
            (str(tmp_path / "b_target.wav"), "text b", 0.82),
        ], [0.73]

    with patch("tool_clonevoice.omnivoice_backend.process_target_reference_batch", side_effect=fake_process) as proc:
        updated = sc.finish_candidate_target_sample_jobs(
            jobs,
            models_root=str(tmp_path / "models"),
            score_candidates=[existing],
            log=lambda _m: None,
        )

    assert proc.call_count == 1
    assert updated == [candidate_a, candidate_b, existing]
    assert candidate_a["target_sample_text"] == "text a"
    assert candidate_a["target_sample_similarity"] == 0.91
    assert candidate_a["ecapa_similarity"] == 0.91
    assert candidate_b["target_sample_audio"].endswith("b_target.wav")
    assert existing["ecapa_similarity"] == 0.73
    assert existing["ecapa_similarity_basis"] == "translated_audio"


def test_score_candidate_similarities_keeps_candidates_when_ecapa_missing(tmp_path: Path):
    candidate = {
        "id": "cand_001",
        "source_audio": str(tmp_path / "source.wav"),
        "translated_audio": str(tmp_path / "translated.wav"),
        "target_sample_similarity": 0.61,
        "ecapa_similarity": 0.61,
    }

    with patch(
        "tool_clonevoice.omnivoice_backend.process_target_reference_batch",
        side_effect=RuntimeError("ECAPA speaker-similarity model is unavailable."),
    ):
        updated = sc.score_candidate_similarities(
            [candidate],
            models_root=str(tmp_path / "models"),
            log=lambda _m: None,
        )

    assert updated == [candidate]
    assert candidate["ecapa_similarity"] == 0.61


def test_omnivoice_synthesize_writes_duck_key_from_manifest_segments(tmp_path: Path):
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    clone_dir = tmp_path / "movie.clone"
    clone_dir.mkdir()
    manifest = {
        "segments": [
            {"id": 1, "speaker": "SPEAKER_00", "start": 1.0, "end": 3.0, "tgt_text": "hello"},
            {"id": 2, "speaker": "SPEAKER_00", "start": 4.0, "end": 5.0, "tgt_text": ""},
        ],
        "speakers": {},
    }
    model = _FakePromptModel()

    with patch("tool_clonevoice.omnivoice_backend._build_speaker_prompts", return_value={"SPEAKER_00": {"prompt": True}}):
        out = ov.synthesize(
            model,
            manifest,
            str(video),
            clone_dir,
            text_field="tgt_text",
            language="Chinese",
            loudness_mode="flat",
            log=lambda _m: None,
        )

    duck_path = Path(si_logic.default_si_duck_key_path(out))
    duck, sr = si_logic.read_wav_mono(duck_path)

    assert Path(out).name == "movie.si.wav"
    assert duck_path.name == "movie.si.duck.wav"
    assert sr == model.sampling_rate
    assert duck.size >= 3 * sr
    assert np.max(np.abs(duck[: sr // 2])) == 0
    assert np.max(duck[int(1.2 * sr) : int(2.8 * sr)]) > 0.2
    assert np.max(np.abs(duck[int(3.2 * sr) :])) == 0


class _FakePromptModel:
    device = "cpu"
    sampling_rate = 24000

    def __init__(self, *, fail_generate: bool = False):
        self.fail_generate = fail_generate
        self.generate_calls = 0
        self.generate_kwargs = []
        self.prompts = []

    def generate(self, **kwargs):
        self.generate_calls += 1
        self.generate_kwargs.append(kwargs)
        if self.fail_generate:
            raise AssertionError("generate should not be called")
        return [np.full(2400, 0.1, dtype=np.float32)]

    def create_voice_clone_prompt(self, audio, text, preprocess_prompt=True):
        self.prompts.append((audio, text, preprocess_prompt))
        return {"audio": audio, "text": text}


def test_build_speaker_prompts_skip_work_ref_for_target_language_basis(tmp_path: Path):
    clone_dir = tmp_path / "video.clone"
    clone_dir.mkdir()
    (clone_dir / "SPEAKER1.wav").write_bytes(b"wav")
    manifest = {
        "speakers": {
            "SPEAKER_00": {
                "ref_audio": "SPEAKER1.wav",
                "ref_text": "target basis text",
                "ref_language": "zh",
                "skip_work_ref": True,
            }
        },
        "segments": [],
    }
    model = _FakePromptModel(fail_generate=True)

    prompts = ov._build_speaker_prompts(
        model,
        manifest,
        clone_dir,
        "Chinese",
        24000,
        ov.DEFAULT_NUM_STEP,
        ov.DEFAULT_GUIDANCE,
        str(tmp_path / "models"),
        log=lambda _m: None,
        text_field="tgt_text",
    )

    assert "SPEAKER_00" in prompts
    assert model.generate_calls == 0
    assert Path(model.prompts[0][0]).name == "SPEAKER1.wav"
    assert model.prompts[0][1] == "target basis text"


def test_build_speaker_prompts_skip_work_ref_is_tgt_text_only(tmp_path: Path):
    clone_dir = tmp_path / "video.clone"
    clone_dir.mkdir()
    (clone_dir / "SPEAKER1.wav").write_bytes(b"wav")
    manifest = {
        "speakers": {
            "SPEAKER_00": {
                "ref_audio": "SPEAKER1.wav",
                "ref_text": "target basis text",
                "ref_language": "Chinese",
                "skip_work_ref": True,
            }
        },
        "segments": [],
    }
    model = _FakePromptModel()

    fake_soundfile = types.SimpleNamespace(write=lambda path, wav, sr: Path(path).write_bytes(b"work_ref"))
    with (
        patch("tool_clonevoice.omnivoice_backend._load_speaker_sim_model", return_value=None),
        patch.dict(sys.modules, {"soundfile": fake_soundfile}),
    ):
        prompts = ov._build_speaker_prompts(
            model,
            manifest,
            clone_dir,
            "Chinese",
            24000,
            ov.DEFAULT_NUM_STEP,
            ov.DEFAULT_GUIDANCE,
            str(tmp_path / "models"),
            log=lambda _m: None,
            text_field="src_text",
        )

    assert "SPEAKER_00" in prompts
    assert model.generate_calls == ov.WORK_REF_TAKES
    assert Path(model.prompts[0][0]).name == "work_ref_SPEAKER_00.wav"
