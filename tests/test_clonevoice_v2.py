from pathlib import Path
import wave

import numpy as np
import soundfile as sf
import torch

from tool_clonevoice_v2 import backend, logic, segment_engine, single_clone


def test_vendored_pcm16_writer_does_not_use_torchaudio_codec(tmp_path: Path):
    backend._ensure_vendor_importable()
    from indextts.utils.audio_io import save_pcm16_wav

    output = tmp_path / "pcm16.wav"
    source = torch.tensor([[-1.0, -0.25, 0.0, 0.25, 1.0]], dtype=torch.float32)
    save_pcm16_wav(output, source, 22050)

    with wave.open(str(output), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 22050
        pcm = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2")
    assert pcm.tolist() == [-32767, -8192, 0, 8192, 32767]


def test_vendored_pcm16_writer_preserves_indextts_prescaled_amplitude(tmp_path: Path):
    """IndexTTS holds `wav` as float scaled to +-32767 and casts before saving.

    Without the cast the writer treats it as normalized [-1, 1] and every
    audible sample clamps to full scale, turning speech into a square wave.
    """
    backend._ensure_vendor_importable()
    from indextts.utils.audio_io import save_pcm16_wav

    quiet_speech = torch.tensor([[0.0, 0.2, -0.2]], dtype=torch.float32)
    prescaled = torch.clamp(32767 * quiet_speech, -32767.0, 32767.0)
    output = tmp_path / "prescaled.wav"

    save_pcm16_wav(output, prescaled.type(torch.int16), 22050)

    with wave.open(str(output), "rb") as wav_file:
        pcm = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2")
    assert pcm.tolist() == [0, 6553, -6553]


def test_load_segments_accepts_utf8_bom_and_ignores_speaker(tmp_path: Path):
    path = tmp_path / "segments.json"
    path.write_text(
        '[{"start":0,"end":1,"text":"你好","speaker":"SPEAKER_99"}]',
        encoding="utf-8-sig",
    )
    assert logic.load_segments(path) == [{"start": 0.0, "end": 1.0, "text": "你好"}]


def test_extract_reference_uses_sentence_time_range(tmp_path: Path):
    sr = 1000
    source = tmp_path / "source.wav"
    output = tmp_path / "ref.wav"
    sf.write(source, np.linspace(-0.5, 0.5, 3000, dtype=np.float32), sr)
    backend.extract_reference(source, 0.5, 1.25, output)
    clip, clip_sr = sf.read(output, dtype="float32")
    assert clip_sr == sr
    assert 749 <= clip.size <= 751


def test_synthesize_segments_uses_each_sentence_as_its_reference(tmp_path: Path):
    sr = backend.SAMPLE_RATE
    source = tmp_path / "source.wav"
    output = tmp_path / "result.wav"
    sf.write(source, np.zeros(sr * 3, dtype=np.float32), sr)

    class FakeModel:
        def __init__(self):
            self.calls = []

        def infer(self, **kwargs):
            self.calls.append(kwargs)
            index = len(self.calls)
            sf.write(kwargs["output_path"], np.full(sr // 2, index * 0.1, dtype=np.float32), sr)
            return kwargs["output_path"]

    model = FakeModel()
    segments = [
        {"start": 0.0, "end": 0.5, "text": "one"},
        {"start": 1.0, "end": 1.5, "text": "two"},
    ]
    backend.synthesize_segments(model, source, segments, output, language="en", fit_duration=True)
    assert len(model.calls) == 2
    assert model.calls[0]["spk_audio_prompt"] != model.calls[1]["spk_audio_prompt"]
    assert "speaker" not in model.calls[0]
    intermediate = tmp_path / backend.SEGMENT_INTERMEDIATE_DIR_NAME
    assert intermediate.is_dir()
    assert (intermediate / "ref_00001.wav").is_file()
    assert (intermediate / "out_00001.wav").is_file()
    wav, out_sr = sf.read(output, dtype="float32")
    assert out_sr == sr
    assert np.mean(np.abs(wav[: sr // 2])) > 0.05
    assert np.mean(np.abs(wav[sr: sr + sr // 2])) > 0.1


def test_synthesize_manifest_keeps_intermediate_wavs_in_video_clone_dir(tmp_path: Path):
    sr = backend.SAMPLE_RATE
    clone_dir = tmp_path / "movie.clone"
    clone_dir.mkdir()
    source = clone_dir / logic.AUDIO16K_NAME
    output = tmp_path / "movie.si.wav"
    sf.write(source, np.zeros(sr * 3, dtype=np.float32), sr)
    intermediate = clone_dir / backend.MANIFEST_INTERMEDIATE_DIR_NAME
    intermediate.mkdir()
    (intermediate / "speaker_ref_00001.wav").write_bytes(b"obsolete")
    (intermediate / "retry_00001.wav").write_bytes(b"obsolete")
    (intermediate / "sentence_context_00001.wav").write_bytes(b"stale")

    class FakeModel:
        def __init__(self):
            self.calls = []

        def infer(self, **kwargs):
            self.calls.append(kwargs)
            sf.write(kwargs["output_path"], np.full(sr // 2, 0.1, dtype=np.float32), sr)
            return kwargs["output_path"]

    model = FakeModel()
    backend.synthesize_manifest(
        model,
        source,
        [{"id": 1, "start": 0.0, "end": 0.5, "text": "hello"}],
        output,
        language="en",
    )

    assert intermediate.is_dir()
    assert len(model.calls) == 1
    sentence_ref = intermediate / "sentence_ref_00001.wav"
    assert sentence_ref.is_file()
    assert not (intermediate / "sentence_context_00001.wav").exists()
    assert not (intermediate / "speaker_ref_00001.wav").exists()
    assert not (intermediate / "retry_00001.wav").exists()
    assert (intermediate / "out_00001.wav").is_file()
    assert Path(model.calls[0]["output_path"]).parent == intermediate
    assert Path(model.calls[0]["spk_audio_prompt"]).parent == intermediate
    reference, reference_sr = sf.read(sentence_ref, dtype="float32")
    assert reference_sr == sr
    assert 0.49 <= reference.size / reference_sr <= 0.51


def test_vendored_indextts_imports_with_host_transformers():
    backend._ensure_vendor_importable()
    from indextts.infer_v2_5 import IndexTTS2

    assert IndexTTS2.__name__ == "IndexTTS2"


def test_short_sentence_uses_its_own_reference_without_eager_fallback(tmp_path: Path):
    source = tmp_path / "source.wav"
    waveform = np.concatenate([
        np.full(1000, 0.1, dtype=np.float32),
        np.full(400, 0.5, dtype=np.float32),
        np.full(4600, 0.8, dtype=np.float32),
    ])
    sf.write(source, waveform, 1000)
    segment = {"id": 2, "start": 1.0, "end": 1.4, "speaker": "SPEAKER_1"}
    ref = backend.extract_sentence_reference(source, segment, tmp_path)
    audio, sr = sf.read(ref, dtype="float32")
    assert sr == 1000
    assert 0.39 <= len(audio) / sr <= 0.41
    assert np.isclose(np.mean(audio), 0.5, atol=5e-5)


def test_reference_quality_detects_short_quiet_effective_speech() -> None:
    sr = 16000
    time = np.arange(int(sr * 1.36), dtype=np.float32) / sr
    good = (0.04 * np.sin(2.0 * np.pi * 220.0 * time)).astype(np.float32)
    weak = np.zeros(int(sr * 0.94), dtype=np.float32)
    weak[int(sr * 0.35):int(sr * 0.58)] = 0.0035

    good_profile = backend.analyze_reference_quality(good, sr)
    weak_profile = backend.analyze_reference_quality(weak, sr)

    assert good_profile["active_seconds"] > 0.9
    assert good_profile["periodic_ratio"] > 0.3
    assert not backend.reference_needs_timbre_fallback(good_profile)
    assert weak_profile["active_seconds"] < 0.3
    assert weak_profile["rms_dbfs"] < -48.0
    assert backend.reference_needs_timbre_fallback(weak_profile)


def test_reference_quality_rejects_loud_unvoiced_breath_noise() -> None:
    sr = 16000
    rng = np.random.default_rng(1234)
    breath_noise = rng.normal(0.0, 0.02, int(sr * 1.5)).astype(np.float32)

    profile = backend.analyze_reference_quality(breath_noise, sr)

    assert profile["active_seconds"] > 1.0
    assert profile["rms_dbfs"] > -40.0
    assert profile["periodic_ratio"] < backend.UNVOICED_REFERENCE_MAX_PERIODIC_RATIO
    assert backend.reference_needs_timbre_fallback(profile)


def test_manifest_uses_nearest_stable_timbre_and_current_weak_emotion(tmp_path: Path):
    """The weak-reference fallback, which now only runs with anchoring off.

    With the speaker anchor on (the default) the timbre comes from the best
    sentence in the whole title rather than the nearest stable one, so this
    path is exercised explicitly; see the companion assertion at the end.
    """
    sr = backend.SAMPLE_RATE
    source = tmp_path / "source.wav"
    output = tmp_path / "result.wav"
    waveform = np.zeros(sr * 12, dtype=np.float32)
    near_time = np.arange(sr * 3, dtype=np.float32) / sr
    far_time = np.arange(sr * 4, dtype=np.float32) / sr
    waveform[0:sr * 3] = 0.1 * np.sin(2.0 * np.pi * 220.0 * near_time)
    rng = np.random.default_rng(5678)
    waveform[sr * 4:sr * 5] = rng.normal(0.0, 0.02, sr).astype(np.float32)
    waveform[sr * 8:sr * 12] = 0.08 * np.sin(2.0 * np.pi * 260.0 * far_time)
    sf.write(source, waveform, sr)

    class FakeModel:
        def __init__(self):
            self.calls = []

        def infer(self, **kwargs):
            self.calls.append(dict(kwargs))
            sf.write(kwargs["output_path"], np.full(sr // 2, 0.1, dtype=np.float32), sr)
            return kwargs["output_path"]

    model = FakeModel()
    messages: list[str] = []
    backend.synthesize_manifest(
        model,
        source,
        [
            {"id": 1, "start": 0.0, "end": 3.0, "text": "near stable"},
            {"id": 2, "start": 4.0, "end": 5.0, "text": "weak breath"},
            {"id": 3, "start": 8.0, "end": 12.0, "text": "far stable"},
        ],
        output,
        language="en",
        fit_duration=False,
        log=messages.append,
        timbre_anchor=False,
    )

    assert len(model.calls) == 3
    weak_call = model.calls[1]
    assert Path(weak_call["spk_audio_prompt"]).name == "sentence_ref_00001.wav"
    assert Path(weak_call["emo_audio_prompt"]).name == "sentence_ref_00002.wav"
    assert weak_call["emo_alpha"] == backend.UNVOICED_EMOTION_ALPHA
    # One line per sentence now, not two: the reference detail is folded into
    # the duration line, and the per-sentence profile numbers are gone -- they
    # were debugging output and the GUI log holds a limited number of lines.
    assert any("weak-ref" in message for message in messages)
    assert any("emo-alpha=" in message for message in messages)
    assert not any("periodic=" in message for message in messages)

    # Same manifest with the default on. The discriminator is segment 3: it is
    # stable, so the fallback path leaves it cloning from itself, while the
    # anchor path makes it borrow the title's best line (segment 1 -- louder
    # than segment 3, and loudness is now judged against the title's median).
    anchored = FakeModel()
    # Its own intermediate directory: sharing one with the run above would let
    # the unchanged lines come back from the per-line cache instead of reaching
    # the model -- correct behaviour, but it leaves nothing here to inspect.
    anchored_dir = Path(output).parent / "anchored_manifest"
    backend.synthesize_manifest(
        anchored, source,
        [
            {"id": 1, "start": 0.0, "end": 3.0, "text": "near stable"},
            {"id": 2, "start": 4.0, "end": 5.0, "text": "weak breath"},
            {"id": 3, "start": 8.0, "end": 12.0, "text": "far stable"},
        ],
        output, language="en", fit_duration=False, log=lambda _m: None,
        intermediate_dir=anchored_dir,
    )
    assert Path(anchored.calls[1]["spk_audio_prompt"]).name == "sentence_ref_00001.wav"
    assert anchored.calls[1]["emo_alpha"] == backend.UNVOICED_EMOTION_ALPHA
    assert Path(anchored.calls[2]["spk_audio_prompt"]).name == "sentence_ref_00001.wav"
    assert Path(model.calls[2]["spk_audio_prompt"]).name == "sentence_ref_00003.wav"


def test_manifest_retries_failed_short_prompt_with_local_context(tmp_path: Path):
    sr = backend.SAMPLE_RATE
    source = tmp_path / "source.wav"
    output = tmp_path / "result.wav"
    sf.write(source, np.full(sr * 6, 0.1, dtype=np.float32), sr)

    class RetryModel:
        def __init__(self):
            self.calls = []

        def infer(self, **kwargs):
            self.calls.append(dict(kwargs))
            if len(self.calls) == 1:
                raise RuntimeError("Input size per channel is too small")
            sf.write(kwargs["output_path"], np.full(sr // 2, 0.1, dtype=np.float32), sr)
            return kwargs["output_path"]

    model = RetryModel()
    backend.synthesize_manifest(
        model,
        source,
        [{"id": 1, "start": 2.0, "end": 2.4, "text": "short"}],
        output,
        language="en",
        fit_duration=False,
    )

    assert len(model.calls) == 2
    first_ref = Path(model.calls[0]["spk_audio_prompt"])
    retry_ref = Path(model.calls[1]["spk_audio_prompt"])
    first_audio, first_sr = sf.read(first_ref, dtype="float32")
    retry_audio, retry_sr = sf.read(retry_ref, dtype="float32")
    assert first_ref.name == "sentence_ref_00001.wav"
    assert retry_ref.name == "sentence_context_00001.wav"
    assert first_sr == retry_sr == sr
    assert 0.39 <= first_audio.size / sr <= 0.41
    assert 2.99 <= retry_audio.size / sr <= 3.01


def test_candidate_preview_writes_legacy_table_fields(tmp_path: Path):
    source = tmp_path / "cand_src.wav"
    sf.write(source, np.zeros(22050 * 3, dtype=np.float32), 22050)
    candidate = {
        "id": "cand_001", "source_audio": str(source), "src_text": "你好",
        "tgt_text": "Hello", "score": 0.72,
    }

    class FakeModel:
        def infer(self, **kwargs):
            sf.write(kwargs["output_path"], np.ones(2205, dtype=np.float32) * 0.1, 22050)
            return kwargs["output_path"]

    single_clone.generate_indextts_candidate_previews([candidate], model=FakeModel(), target_language="en")
    preview, preview_sr = sf.read(candidate["target_sample_audio"], dtype="float32")
    assert candidate["target_sample_audio"] == candidate["translated_audio"]
    assert candidate["target_sample_text"] == "Hello"
    assert candidate["ecapa_similarity"] == 0.72
    assert preview_sr == 22050
    assert np.isclose(np.max(np.abs(preview)), 0.1, atol=5e-5)


# --- ASR model choice -------------------------------------------------------

def test_kotoba_is_kept_for_a_japanese_source(tmp_path: Path):
    (tmp_path / "kotoba-whisper-v2.0-faster").mkdir()
    (tmp_path / "kotoba-whisper-v2.0-faster" / "model.bin").write_bytes(b"x")
    (tmp_path / "kotoba-whisper-v2.0-faster" / "config.json").write_text("{}")
    lines: list[str] = []
    assert segment_engine.resolve_model_for_language(
        "kotoba", "ja", str(tmp_path), lines.append
    ) == "kotoba"
    assert lines == []


def test_kotoba_falls_back_on_a_non_japanese_source(tmp_path: Path):
    lines: list[str] = []
    assert segment_engine.resolve_model_for_language(
        "kotoba", "en", str(tmp_path), lines.append
    ) == segment_engine.FALLBACK_MODEL_KEY
    assert any("Japanese-only" in line for line in lines)


def test_kotoba_falls_back_when_its_files_are_absent(tmp_path: Path):
    lines: list[str] = []
    assert segment_engine.resolve_model_for_language(
        "kotoba", "ja", str(tmp_path), lines.append
    ) == segment_engine.FALLBACK_MODEL_KEY
    assert any("missing" in line for line in lines)


def test_a_non_japanese_only_model_is_never_second_guessed(tmp_path: Path):
    lines: list[str] = []
    assert segment_engine.resolve_model_for_language(
        "large-v3", "en", str(tmp_path), lines.append
    ) == "large-v3"
    assert lines == []


def test_the_v2_transcription_defaults():
    """anime-whisper since 2026-09-03; "high" because "max" answers every breath
    with a stock phrase.

    anime-whisper is a Whisper large-v2 fine-tune on anime/drama Japanese, the
    register this material is actually in. Measured on the benchmark corpus it
    recognises 13% more kanji than kotoba -- content words, not characters --
    and gets words right that kotoba misses ("同時ないように" ->
    "動じないように"). kotoba stays selectable and is the first fallback when
    anime-whisper is not installed; large-v3 remains last, because it invents
    stock phrases over this material rather than transcribing it.

    "high" over "max": counted over the transcribed titles, the runs at "max"
    carry 12-15% stock-phrase lines and 31-46% three-character fragments,
    against 0-5% and 3-13% at "high" -- and a hallucinated line is spoken aloud
    in the dub, where a missing one is only silence."""
    import inspect

    for fn in (logic.run_transcribe_diarize, logic.run_full, logic.run_batch):
        params = inspect.signature(fn).parameters
        assert params["model_key"].default == "anime-whisper"
        assert params["vad_sensitivity"].default == "high"
