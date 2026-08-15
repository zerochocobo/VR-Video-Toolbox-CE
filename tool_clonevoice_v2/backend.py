from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

LogCallback = Callable[[str], None]
MODEL_DIR_NAME = "IndexTTS-2.5"
SUPPORTED_LANGUAGES = {"zh", "en", "ja", "es", "ar"}
SAMPLE_RATE = 22050
MANIFEST_INTERMEDIATE_DIR_NAME = "indextts_v2_manifest"
SEGMENT_INTERMEDIATE_DIR_NAME = "indextts_v2_segments"
LOCAL_CONTEXT_RETRY_SECONDS = 3.0
REFERENCE_ACTIVITY_TOP_DB = 35.0
WEAK_REFERENCE_MIN_ACTIVE_SECONDS = 0.60
WEAK_REFERENCE_MIN_RMS_DBFS = -48.0
WEAK_REFERENCE_MIN_ACTIVE_RMS_DBFS = -45.0
STABLE_REFERENCE_MIN_SECONDS = 2.0
STABLE_REFERENCE_MIN_ACTIVE_SECONDS = 1.2
STABLE_REFERENCE_MIN_RMS_DBFS = -45.0
STABLE_REFERENCE_MIN_ACTIVE_RMS_DBFS = -42.0
STABLE_REFERENCE_MAX_DISTANCE_SECONDS = 120.0
PERIODICITY_CORRELATION_THRESHOLD = 0.45
UNVOICED_REFERENCE_MAX_PERIODIC_RATIO = 0.15
STABLE_REFERENCE_MIN_PERIODIC_RATIO = 0.30
UNVOICED_EMOTION_ALPHA = 0.35
MAIN_MODEL_FILES = (
    "config.yaml", "gpt.pth", "s2mel.pth", "codec.pth", "feat1.pt", "feat2.pt",
    "wav2vec2bert_stats.pt", "multilingual_zh_ja_yue_char_del.tiktoken",
)
MIN_MODEL_FILE_BYTES = {
    "gpt.pth": 100_000_000,
    "s2mel.pth": 100_000_000,
    "codec.pth": 100_000_000,
    "feat1.pt": 1_000,
    "feat2.pt": 1_000,
    "wav2vec2bert_stats.pt": 1_000,
    "multilingual_zh_ja_yue_char_del.tiktoken": 100_000,
    "aux_models/campplus_cn_common.bin": 1_000_000,
    "aux_models/bigvgan/config.json": 100,
    "aux_models/bigvgan/bigvgan_generator.pt": 100_000_000,
}
AUX_MODEL_FILES = (
    "aux_models/campplus_cn_common.bin",
    "aux_models/bigvgan/config.json",
    "aux_models/bigvgan/bigvgan_generator.pt",
)


def model_dir(models_root: str | Path) -> Path:
    root = Path(models_root)
    if (root / "config.yaml").is_file():
        return root
    return root / MODEL_DIR_NAME


def check_model(models_root: str | Path) -> bool:
    return not missing_model_files(models_root)


def missing_main_model_files(models_root: str | Path) -> list[str]:
    root = model_dir(models_root)
    return [name for name in MAIN_MODEL_FILES if not _valid_model_file(root / name, name)]


def _valid_model_file(path: Path, relative_name: str) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= MIN_MODEL_FILE_BYTES.get(relative_name, 1)
    except OSError:
        return False


def missing_model_files(models_root: str | Path) -> list[str]:
    root = model_dir(models_root)
    missing = missing_main_model_files(models_root)
    missing.extend(name for name in AUX_MODEL_FILES if not _valid_model_file(root / name, name))
    w2v = root / "aux_models" / "w2v-bert-2.0"
    for name, min_bytes in (
        ("config.json", 100),
        ("preprocessor_config.json", 100),
        ("model.safetensors", 100_000_000),
    ):
        path = w2v / name
        try:
            valid = path.is_file() and path.stat().st_size >= min_bytes
        except OSError:
            valid = False
        if not valid:
            missing.append(f"aux_models/w2v-bert-2.0/{name}")
    return missing


def normalize_language(language: str | None) -> str:
    key = (language or "").strip().lower().replace("_", "-")
    aliases = {
        "zh-cn": "zh", "zh-hans": "zh", "chinese": "zh", "中文": "zh",
        "en-us": "en", "english": "en", "英语": "en",
        "ja-jp": "ja", "japanese": "ja", "日语": "ja", "日本語": "ja",
        "es-es": "es", "spanish": "es", "西班牙语": "es",
        "ar-sa": "ar", "arabic": "ar", "阿拉伯语": "ar",
    }
    return aliases.get(key, key.split("-", 1)[0])


def _ensure_vendor_importable() -> None:
    vendor_root = str(Path(__file__).resolve().parent / "vendor")
    if vendor_root not in sys.path:
        sys.path.insert(0, vendor_root)


def _ensure_main_model_available(root: Path, missing: list[str], log: LogCallback) -> None:
    """Resume the main IndexTTS repository download into its model directory."""
    _ensure_vendor_importable()
    log(f"[indextts-v2] main model files missing ({len(missing)}); downloading to {root}")
    try:
        from indextts.utils.model_download import snapshot_download
        snapshot_download("IndexTeam/IndexTTS-2.5", local_dir=str(root))
    except Exception as exc:
        names = "\n  - ".join(missing)
        raise FileNotFoundError(
            f"IndexTTS-2.5 main model files are incomplete under {root}:\n  - {names}\n"
            f"Automatic download failed: {exc}"
        ) from exc


def load_model(models_root: str | Path, *, device: str | None = None,
               log: LogCallback = print):
    main_missing = missing_main_model_files(models_root)
    if main_missing:
        _ensure_main_model_available(model_dir(models_root), main_missing, log)
        main_missing = missing_main_model_files(models_root)
        if main_missing:
            missing = "\n  - ".join(main_missing)
            raise FileNotFoundError(f"IndexTTS-2.5 main model files remain incomplete:\n  - {missing}")
    _ensure_vendor_importable()
    aux_missing = [name for name in missing_model_files(models_root) if name.startswith("aux_models/")]
    if aux_missing:
        log("[indextts-v2] auxiliary models missing; downloading to the fixed aux_models directory")
        try:
            from indextts.utils.model_download import ensure_models_available
            ensure_models_available(str(model_dir(models_root)))
        except Exception as exc:
            missing = "\n  - ".join(aux_missing)
            raise FileNotFoundError(
                f"IndexTTS-2.5 auxiliary models are incomplete under {model_dir(models_root) / 'aux_models'}:\n"
                f"  - {missing}\nDownload failed: {exc}"
            ) from exc
        remaining = missing_model_files(models_root)
        if remaining:
            missing = "\n  - ".join(remaining)
            raise FileNotFoundError(f"IndexTTS-2.5 model files remain incomplete:\n  - {missing}")
    try:
        import torch
        from indextts.infer_v2_5 import IndexTTS2
    except Exception as exc:
        raise RuntimeError(
            "Failed to import vendored IndexTTS-2.5. The vendor compatibility "
            f"layer or host dependencies are incomplete: {exc}"
        ) from exc
    root = model_dir(models_root)
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    use_bf16 = str(device).startswith("cuda")
    log(f"[indextts-v2] loading {root} on {device} (bf16={use_bf16})")
    return IndexTTS2(cfg_path=str(root / "config.yaml"), model_dir=str(root),
                     use_bf16=use_bf16, device=device, use_cuda_kernel=False,
                     use_deepspeed=False, use_accel=False, use_torch_compile=False,
                     use_qwen_emo=False)


def release_model(model) -> None:
    try:
        del model
    finally:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return wav.mean(axis=1), int(sr)


def _write_audio(path: str | Path, wav: np.ndarray, sr: int) -> None:
    import soundfile as sf
    sf.write(str(path), np.asarray(wav, dtype=np.float32), int(sr))


def extract_reference(source_audio: str | Path, start: float, end: float,
                     output_path: str | Path) -> str:
    wav, sr = _read_audio(source_audio)
    return _extract_reference_from_audio(wav, sr, start, end, output_path, source_audio)


def _extract_reference_from_audio(wav: np.ndarray, sr: int, start: float, end: float,
                                  output_path: str | Path,
                                  source_label: str | Path = "audio") -> str:
    lo = max(0, int(round(float(start) * sr)))
    hi = min(wav.size, max(lo + 1, int(round(float(end) * sr))))
    clip = wav[lo:hi]
    if clip.size == 0:
        raise ValueError(f"Empty reference range {start}-{end}: {source_label}")
    _write_audio(output_path, clip, sr)
    return str(output_path)


def _fit_audio(wav: np.ndarray, sr: int, duration: float) -> np.ndarray:
    from tool_si.logic import fit_audio_to_duration
    return fit_audio_to_duration(wav, sr, duration)


def _persistent_intermediate_dir(
    source_audio: str | Path,
    output_path: str | Path,
    intermediate_dir: str | Path | None,
    default_name: str,
) -> Path:
    """Create an inference work directory outside the system temp folder."""
    if intermediate_dir is not None:
        work_dir = Path(intermediate_dir)
    else:
        source_parent = Path(source_audio).parent
        base_dir = source_parent if source_parent.name.lower().endswith(".clone") else Path(output_path).parent
        work_dir = base_dir / default_name
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def duration_factor_for_slot(text: str, slot_seconds: float, language: str, tempo_fit: str = "moderate") -> float:
    visible = max(1, len("".join((text or "").split())))
    cps = {"zh": 5.5, "ja": 6.5, "en": 13.0, "es": 13.5, "ar": 11.0}.get(language, 12.0)
    natural = max(0.3, visible / cps)
    # IndexTTS semantics: >1 is slower. Keep moderate fitting within the model
    # range; the final WAV is still hard-fitted after generation.
    bands = {"moderate": (0.80, 1.30), "strong": (0.65, 1.55)}
    lo, hi = bands.get(tempo_fit, (0.80, 1.30))
    return float(np.clip(float(slot_seconds) / natural, lo, hi))


def synthesize_segments(model, source_audio: str | Path, segments: Iterable[dict],
                        output_path: str | Path, *, language: str,
                        fit_duration: bool = True, tempo_fit: str = "moderate", log: LogCallback = print,
                        stop_event=None, intermediate_dir: str | Path | None = None) -> str:
    lang = normalize_language(language)
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"IndexTTS-2.5 supports only: {', '.join(sorted(SUPPORTED_LANGUAGES))}; got {language}")
    entries = [s for s in segments if (s.get("text") or "").strip()]
    total = max((float(s["end"]) for s in entries), default=0.0)
    timeline = np.zeros(max(1, int(round((total + 1.0) * SAMPLE_RATE))), dtype=np.float32)
    work_dir = _persistent_intermediate_dir(
        source_audio, output_path, intermediate_dir, SEGMENT_INTERMEDIATE_DIR_NAME
    )
    log(f"[indextts-v2] intermediate WAV directory: {work_dir}")
    for idx, segment in enumerate(entries, 1):
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("Stopped by user.")
        start, end = float(segment["start"]), float(segment["end"])
        text = str(segment["text"]).strip()
        ref_path = work_dir / f"ref_{idx:05d}.wav"
        out_path = work_dir / f"out_{idx:05d}.wav"
        extract_reference(source_audio, start, end, ref_path)
        kwargs = {
            "spk_audio_prompt": str(ref_path), "text": text,
            "lang": lang.upper(), "output_path": str(out_path),
            "verbose": False,
        }
        if fit_duration:
            kwargs["duration_factor"] = duration_factor_for_slot(text, end - start, lang, tempo_fit)
        log(f"[indextts-v2] {idx}/{len(entries)} {start:.2f}-{end:.2f}s")
        result = model.infer(**kwargs)
        generated = Path(result or out_path)
        if not generated.is_file():
            raise RuntimeError(f"IndexTTS returned no WAV for segment {idx}")
        clip, clip_sr = _read_audio(generated)
        generated_duration = clip.size / float(clip_sr)
        if clip_sr != SAMPLE_RATE:
            import torchaudio
            import torch
            clip = torchaudio.functional.resample(torch.from_numpy(clip), clip_sr, SAMPLE_RATE).numpy()
            clip_sr = SAMPLE_RATE
        if fit_duration:
            clip = _fit_audio(clip, clip_sr, end - start)
        log(f"[indextts-v2] {idx}/{len(entries)} duration: model={generated_duration:.2f}s, final={(clip.size / SAMPLE_RATE):.2f}s, slot={(end-start):.2f}s")
        lo = max(0, int(round(start * SAMPLE_RATE)))
        hi = min(timeline.size, lo + clip.size)
        if hi > lo:
            timeline[lo:hi] += clip[:hi - lo]
    _write_audio(output_path, timeline, SAMPLE_RATE)
    return str(output_path)


def extract_sentence_reference(source_audio: str | Path, segment: dict,
                               temp_dir: str | Path, *,
                               source_wav: np.ndarray | None = None,
                               source_sr: int | None = None) -> str:
    """Extract exactly the current manifest sentence as the IndexTTS prompt."""
    path = Path(temp_dir) / f"sentence_ref_{int(segment.get('id', 0)):05d}.wav"
    if source_wav is not None and source_sr is not None:
        return _extract_reference_from_audio(
            source_wav, source_sr, float(segment["start"]), float(segment["end"]),
            path, source_audio,
        )
    return extract_reference(
        source_audio, float(segment["start"]), float(segment["end"]), path
    )


def extract_local_context_reference(source_audio: str | Path, segment: dict,
                                    temp_dir: str | Path,
                                    *, target_seconds: float = LOCAL_CONTEXT_RETRY_SECONDS,
                                    source_wav: np.ndarray | None = None,
                                    source_sr: int | None = None) -> str:
    """Expand around the current sentence for a short-prompt retry.

    This is deliberately local: a failed short sentence must never be replaced
    by an unrelated longest sentence elsewhere in the video. The initial
    inference always uses ``extract_sentence_reference``; this wider window is
    created only after IndexTTS actually fails to produce a WAV.
    """
    if source_wav is None or source_sr is None:
        wav, sr = _read_audio(source_audio)
    else:
        wav, sr = source_wav, int(source_sr)
    sentence_lo = max(0, min(wav.size, int(round(float(segment["start"]) * sr))))
    sentence_hi = max(sentence_lo + 1, int(round(float(segment["end"]) * sr)))
    sentence_hi = min(wav.size, sentence_hi)
    target_samples = min(wav.size, max(1, int(round(float(target_seconds) * sr))))
    sentence_samples = max(1, sentence_hi - sentence_lo)
    missing = max(0, target_samples - sentence_samples)
    lo = max(0, sentence_lo - missing // 2)
    hi = min(wav.size, sentence_hi + (missing - missing // 2))
    if hi - lo < target_samples:
        if lo == 0:
            hi = min(wav.size, target_samples)
        elif hi == wav.size:
            lo = max(0, wav.size - target_samples)
    clip = wav[lo:hi]
    if clip.size == 0:
        raise ValueError(
            f"Empty local reference range {segment['start']}-{segment['end']}: {source_audio}"
        )
    path = Path(temp_dir) / f"sentence_context_{int(segment.get('id', 0)):05d}.wav"
    _write_audio(path, clip, sr)
    return str(path)


def _dbfs(value: float) -> float:
    return float(20.0 * np.log10(max(float(value), 1e-12)))


def _periodic_voice_ratio(wav: np.ndarray, sr: int, activity_threshold: float) -> tuple[float, float]:
    """Estimate stable vocal-fold periodicity with a lightweight FFT autocorrelation."""
    frame = max(64, int(round(sr * 0.040)))
    hop = max(1, int(round(sr * 0.010)))
    clip = np.asarray(wav, dtype=np.float32).reshape(-1)
    if clip.size < frame:
        clip = np.pad(clip, (0, frame - clip.size))
    count = 1 + (clip.size - frame) // hop
    frames = np.lib.stride_tricks.as_strided(
        clip,
        shape=(count, frame),
        strides=(clip.strides[0] * hop, clip.strides[0]),
    ).copy()
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    active = frame_rms >= activity_threshold
    if not np.any(active):
        return 0.0, 0.0
    frames -= np.mean(frames, axis=1, keepdims=True)
    frames *= np.hanning(frame).astype(np.float32)
    fft_size = 1 << int(np.ceil(np.log2(frame * 2 - 1)))
    spectrum = np.fft.rfft(frames, n=fft_size, axis=1)
    autocorrelation = np.fft.irfft(
        spectrum * np.conj(spectrum), n=fft_size, axis=1
    )[:, :frame]
    autocorrelation /= np.maximum(autocorrelation[:, :1], 1e-12)
    min_lag = max(1, int(sr / 700.0))
    max_lag = min(frame - 1, int(sr / 60.0))
    strength = np.max(autocorrelation[:, min_lag:max_lag + 1], axis=1)
    periodic = active & (strength >= PERIODICITY_CORRELATION_THRESHOLD)
    periodic_ratio = float(np.count_nonzero(periodic) / max(1, np.count_nonzero(active)))
    periodic_seconds = float(np.count_nonzero(periodic) * hop / float(sr))
    return periodic_ratio, periodic_seconds


def analyze_reference_quality(wav: np.ndarray, sr: int) -> dict:
    """Measure usable speech length and level without changing the waveform."""
    clip = np.asarray(wav, dtype=np.float32).reshape(-1)
    duration = clip.size / float(max(1, sr))
    if clip.size == 0:
        return {
            "duration": 0.0, "active_seconds": 0.0,
            "rms_dbfs": -240.0, "active_rms_dbfs": -240.0,
            "periodic_ratio": 0.0, "periodic_seconds": 0.0,
        }
    rms = float(np.sqrt(np.mean(clip * clip)))
    peak = float(np.max(np.abs(clip)))
    frame = max(1, int(round(sr * 0.025)))
    hop = max(1, int(round(sr * 0.010)))
    if clip.size < frame:
        frame_rms = np.asarray([rms], dtype=np.float32)
    else:
        count = 1 + (clip.size - frame) // hop
        frames = np.lib.stride_tricks.as_strided(
            clip,
            shape=(count, frame),
            strides=(clip.strides[0] * hop, clip.strides[0]),
        )
        frame_rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    threshold = max(peak * (10.0 ** (-REFERENCE_ACTIVITY_TOP_DB / 20.0)), 1e-5)
    active = frame_rms >= threshold
    active_seconds = float(np.count_nonzero(active) * hop / float(sr))
    active_rms = float(np.sqrt(np.mean(frame_rms[active] ** 2))) if np.any(active) else 0.0
    periodic_ratio, periodic_seconds = _periodic_voice_ratio(clip, sr, threshold)
    return {
        "duration": float(duration),
        "active_seconds": active_seconds,
        "rms_dbfs": _dbfs(rms),
        "active_rms_dbfs": _dbfs(active_rms),
        "periodic_ratio": periodic_ratio,
        "periodic_seconds": periodic_seconds,
    }


def reference_needs_timbre_fallback(profile: dict) -> bool:
    return (
        float(profile["active_seconds"]) < WEAK_REFERENCE_MIN_ACTIVE_SECONDS
        or float(profile["rms_dbfs"]) < WEAK_REFERENCE_MIN_RMS_DBFS
        or float(profile["active_rms_dbfs"]) < WEAK_REFERENCE_MIN_ACTIVE_RMS_DBFS
        or float(profile["periodic_ratio"]) < UNVOICED_REFERENCE_MAX_PERIODIC_RATIO
    )


def _stable_timbre_reference(profile: dict) -> bool:
    return (
        float(profile["duration"]) >= STABLE_REFERENCE_MIN_SECONDS
        and float(profile["active_seconds"]) >= STABLE_REFERENCE_MIN_ACTIVE_SECONDS
        and float(profile["rms_dbfs"]) >= STABLE_REFERENCE_MIN_RMS_DBFS
        and float(profile["active_rms_dbfs"]) >= STABLE_REFERENCE_MIN_ACTIVE_RMS_DBFS
        and float(profile["periodic_ratio"]) >= STABLE_REFERENCE_MIN_PERIODIC_RATIO
    )


def select_nearest_stable_reference(entries: list[dict], profiles: list[dict],
                                    current_index: int) -> int | None:
    """Find a nearby long, audible sentence for timbre only."""
    current = entries[current_index]
    current_start, current_end = float(current["start"]), float(current["end"])
    current_speaker = str(current.get("speaker") or "")
    choices: list[tuple[float, float, float, int]] = []
    for index, (candidate, profile) in enumerate(zip(entries, profiles)):
        if index == current_index or not _stable_timbre_reference(profile):
            continue
        candidate_speaker = str(candidate.get("speaker") or "")
        if current_speaker and candidate_speaker and candidate_speaker != current_speaker:
            continue
        candidate_start, candidate_end = float(candidate["start"]), float(candidate["end"])
        distance = max(0.0, candidate_start - current_end, current_start - candidate_end)
        if distance > STABLE_REFERENCE_MAX_DISTANCE_SECONDS:
            continue
        choices.append((
            distance,
            -float(profile["active_seconds"]),
            -float(profile["active_rms_dbfs"]),
            index,
        ))
    return min(choices)[-1] if choices else None


def _reference_profiles(source_wav: np.ndarray, source_sr: int,
                        entries: list[dict]) -> list[dict]:
    profiles = []
    for segment in entries:
        lo = max(0, int(round(float(segment["start"]) * source_sr)))
        hi = min(source_wav.size, max(lo + 1, int(round(float(segment["end"]) * source_sr))))
        profiles.append(analyze_reference_quality(source_wav[lo:hi], source_sr))
    return profiles


def _short_prompt_runtime_error(exc: RuntimeError) -> bool:
    return "input size per channel" in str(exc).lower()


def _remove_legacy_manifest_references(work_dir: Path, log: LogCallback) -> None:
    """Remove obsolete global-longest fallback files from the old workflow."""
    removed = 0
    for pattern in ("speaker_ref_*.wav", "retry_*.wav"):
        for path in work_dir.glob(pattern):
            path.unlink(missing_ok=True)
            removed += 1
    if removed:
        log(f"[indextts-v2] removed {removed} obsolete speaker/longest reference WAV(s)")


def synthesize_manifest(model, source_audio: str | Path, segments: Iterable[dict],
                        output_path: str | Path, *, language: str,
                        fit_duration: bool = True, tempo_fit: str = "moderate", log: LogCallback = print,
                        stop_event=None, intermediate_dir: str | Path | None = None) -> str:
    """Synthesize manifest segments with per-sentence IndexTTS references."""
    entries = [dict(s) for s in segments if (s.get("text") or "").strip()]
    total = max((float(s["end"]) for s in entries), default=0.0)
    timeline = np.zeros(max(1, int(round((total + 1.0) * SAMPLE_RATE))), dtype=np.float32)
    work_dir = _persistent_intermediate_dir(
        source_audio, output_path, intermediate_dir, MANIFEST_INTERMEDIATE_DIR_NAME
    )
    log(f"[indextts-v2] intermediate WAV directory: {work_dir}")
    _remove_legacy_manifest_references(work_dir, log)
    source_wav, source_sr = _read_audio(source_audio)
    profiles = _reference_profiles(source_wav, source_sr, entries)
    for idx, segment in enumerate(entries, 1):
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("Stopped by user.")
        start, end = float(segment["start"]), float(segment["end"])
        text = str(segment["text"]).strip()
        context_path = work_dir / f"sentence_context_{int(segment.get('id', 0)):05d}.wav"
        context_path.unlink(missing_ok=True)
        ref_path = extract_sentence_reference(
            source_audio, segment, work_dir, source_wav=source_wav, source_sr=source_sr
        )
        out_path = work_dir / f"out_{idx:05d}.wav"
        kwargs = {"spk_audio_prompt": ref_path, "text": text,
                  "lang": normalize_language(language).upper(),
                  "output_path": str(out_path), "verbose": False}
        profile = profiles[idx - 1]
        timbre_fallback_index = None
        if reference_needs_timbre_fallback(profile):
            timbre_fallback_index = select_nearest_stable_reference(entries, profiles, idx - 1)
        if timbre_fallback_index is not None:
            timbre_segment = entries[timbre_fallback_index]
            timbre_path = extract_sentence_reference(
                source_audio, timbre_segment, work_dir,
                source_wav=source_wav, source_sr=source_sr,
            )
            kwargs["spk_audio_prompt"] = timbre_path
            kwargs["emo_audio_prompt"] = ref_path
            kwargs["emo_alpha"] = (
                UNVOICED_EMOTION_ALPHA
                if float(profile["periodic_ratio"]) < UNVOICED_REFERENCE_MAX_PERIODIC_RATIO
                else 1.0
            )
        if fit_duration:
            kwargs["duration_factor"] = duration_factor_for_slot(text, end - start, normalize_language(language), tempo_fit)
        if timbre_fallback_index is not None:
            log(
                f"[indextts-v2] {idx}/{len(entries)} {start:.2f}-{end:.2f}s "
                f"timbre={Path(kwargs['spk_audio_prompt']).name} emotion={Path(ref_path).name} "
                f"weak-ref(active={profile['active_seconds']:.2f}s, "
                f"rms={profile['rms_dbfs']:.1f}dBFS, "
                f"active-rms={profile['active_rms_dbfs']:.1f}dBFS, "
                f"periodic={profile['periodic_ratio']:.0%}, "
                f"emo-alpha={kwargs['emo_alpha']:.2f})"
            )
        else:
            log(f"[indextts-v2] {idx}/{len(entries)} {start:.2f}-{end:.2f}s ref={Path(ref_path).name}")
        out_path.unlink(missing_ok=True)
        retry_reason = ""
        try:
            result = model.infer(**kwargs)
        except RuntimeError as exc:
            if not _short_prompt_runtime_error(exc):
                raise
            result = None
            retry_reason = str(exc)
        generated = Path(result or out_path)
        if not generated.is_file():
            retry_reason = retry_reason or "IndexTTS returned no WAV"
        if retry_reason:
            if timbre_fallback_index is not None:
                kwargs["emo_audio_prompt"] = kwargs["spk_audio_prompt"]
                retry_reference = Path(kwargs["spk_audio_prompt"])
                retry_description = "stable timbre reference for both timbre and emotion"
            else:
                context_path = Path(extract_local_context_reference(
                    source_audio, segment, work_dir,
                    source_wav=source_wav, source_sr=source_sr,
                ))
                kwargs["spk_audio_prompt"] = str(context_path)
                retry_reference = context_path
                retry_description = "local sentence context"
            out_path.unlink(missing_ok=True)
            log(
                f"[indextts-v2] retry segment {idx} with {retry_description} "
                f"({retry_reference.name}); first attempt failed: {retry_reason}"
            )
            result = model.infer(**kwargs)
        generated = Path(result or out_path)
        if not generated.is_file():
            raise RuntimeError(f"IndexTTS returned no WAV for segment {idx}")
        clip, clip_sr = _read_audio(generated)
        generated_duration = clip.size / float(clip_sr)
        if clip_sr != SAMPLE_RATE:
            import torchaudio, torch
            clip = torchaudio.functional.resample(torch.from_numpy(clip), clip_sr, SAMPLE_RATE).numpy()
        if fit_duration:
            clip = _fit_audio(clip, SAMPLE_RATE, end - start)
        log(f"[indextts-v2] {idx}/{len(entries)} duration: model={generated_duration:.2f}s, final={(clip.size / SAMPLE_RATE):.2f}s, slot={(end-start):.2f}s")
        lo = max(0, int(round(start * SAMPLE_RATE)))
        hi = min(timeline.size, lo + clip.size)
        if hi > lo:
            timeline[lo:hi] += clip[:hi - lo]
    _write_audio(output_path, timeline, SAMPLE_RATE)
    return str(output_path)
