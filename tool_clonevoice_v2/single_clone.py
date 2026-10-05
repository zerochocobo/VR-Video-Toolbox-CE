"""Single-speaker voice-clone workflow helpers.

This module keeps the new guided GUI thin: it orchestrates existing
``tool_clonevoice_v2`` stages without calling ``run_full`` so a user-confirmed
``SPEAKER1.wav/txt`` basis cannot be overwritten by automatic ref selection.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from threading import Event
from typing import Callable, Optional

from tool_clonevoice_v2 import logic
from tool_clonevoice_v2 import refsel

LogCallback = Callable[[str], None]

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v")
GENERATED_MP4_SUFFIXES = ("_si.mp4", "_dub.mp4")
# Evaluate this many times the displayed candidate count: pick 2x the longest
# clips (deterministic), generate + ECAPA-rank them, then show the best N. Gives
# the best-cloning clip a real chance to surface without generating every span.
CANDIDATE_POOL_FACTOR = 2
SPEAKER_ID = "SPEAKER_00"
SPEAKER1_WAV = "SPEAKER1.wav"
SPEAKER1_TXT = "SPEAKER1.txt"
SPEAKER1_META = "SPEAKER1.meta.json"


def is_generated_output_mp4(path: str | os.PathLike[str]) -> bool:
    return Path(path).name.lower().endswith(GENERATED_MP4_SUFFIXES)


def scan_videos(input_path: str, *, batch: bool) -> list[str]:
    base = Path(input_path)
    if not batch:
        return [str(base)] if base.is_file() else []
    videos: list[str] = []
    if not base.is_dir():
        return videos
    for root, _dirs, files in os.walk(base):
        for name in files:
            if name.lower().endswith(VIDEO_EXTENSIONS) and not is_generated_output_mp4(name):
                videos.append(str(Path(root) / name))
    return sorted(videos, key=lambda p: p.lower())


def user_visible_speaker1_paths(video_or_root: str | os.PathLike[str], *, batch: bool) -> tuple[Path, Path]:
    path = Path(video_or_root)
    if batch:
        root = path
        return root / SPEAKER1_WAV, root / SPEAKER1_TXT
    video = path
    return video.parent / f"{video.stem}.SPEAKER1.wav", video.parent / f"{video.stem}.SPEAKER1.txt"


def run_single_transcribe(
    video_path: str | os.PathLike[str],
    *,
    model_key: str,
    language: Optional[str],
    target_language: str,
    models_root: str,
    denoise: str = "none",
    vad_sensitivity: str = "high",
    diarize_backend: str = "none",
    num_speakers: Optional[int] = None,
    log: LogCallback = print,
    stop_event: Optional[Event] = None,
    model_holder: Optional[list] = None,
) -> dict:
    return logic.run_transcribe_diarize(
        video_path,
        model_key=model_key,
        language=language,
        diarize_backend=diarize_backend,
        num_speakers=num_speakers,
        target_language=target_language,
        models_root=models_root,
        denoise=denoise,
        vad_sensitivity=vad_sensitivity,
        skip_diarization=(diarize_backend == "none"),
        log=log,
        stop_event=stop_event,
        model_holder=model_holder,
    )


def collect_single_candidates(
    video_path: str | os.PathLike[str],
    *,
    top_n: int = 12,
    log: LogCallback = print,
) -> list[dict]:
    video = Path(video_path)
    manifest = logic.load_manifest(video)
    if not manifest:
        raise FileNotFoundError(f"Manifest not found; run transcription first: {logic.manifest_path(video)}")
    cdir = logic.clone_dir(video)
    audio16k = cdir / logic.AUDIO16K_NAME
    if not audio16k.is_file():
        raise FileNotFoundError(f"Intermediate audio missing: {audio16k}")
    return refsel.collect_reference_candidates(
        str(video),
        manifest,
        str(audio16k),
        cdir,
        speaker=SPEAKER_ID,
        top_n=top_n,
        log=log,
    )


def collect_candidates_for_videos(
    videos: list[str],
    *,
    per_video: int = 6,
    total: int = 12,
    log: LogCallback = print,
) -> list[dict]:
    all_candidates: list[dict] = []
    for video in videos:
        candidates = collect_single_candidates(video, top_n=per_video, log=log)
        usable = [cand for cand in candidates if (cand.get("src_text") or "").strip()]
        skipped = len(candidates) - len(usable)
        if skipped:
            log(f"[single-ref] skipped {skipped} candidate(s) without source transcript text.")
        all_candidates.extend(usable)
    # Duration-first (longest clip wins), matching the per-video pool selection —
    # deterministic and prefers stronger references.
    all_candidates.sort(key=lambda c: (-float(c.get("dur") or 0.0), float(c.get("start") or 0.0)))
    selected = all_candidates[: max(1, int(total))]
    for idx, cand in enumerate(selected, 1):
        cand["global_rank"] = idx
    return selected


def _candidate_audio_file(candidate: dict, key: str) -> str:
    path = candidate.get(key) or ""
    return path if path and Path(path).is_file() else ""


def load_candidate_json(path: str | os.PathLike[str], *, log: LogCallback = print) -> list[dict]:
    json_path = Path(path)
    if not json_path.is_file():
        return []
    try:
        data = json.loads(json_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        log(f"[single-ref] could not load existing candidates: {json_path} ({exc})")
        return []
    if not isinstance(data, list):
        return []

    loaded: list[dict] = []
    for raw in data:
        if not isinstance(raw, dict):
            continue
        cand = dict(raw)
        source_audio = _candidate_audio_file(cand, "source_audio")
        if not source_audio or not (cand.get("src_text") or "").strip():
            continue
        cand["source_audio"] = source_audio
        cand["target_sample_audio"] = _candidate_audio_file(cand, "target_sample_audio")
        cand["translated_audio"] = _candidate_audio_file(cand, "translated_audio")
        if not cand["target_sample_audio"]:
            cand["target_sample_similarity"] = None
            if cand.get("ecapa_similarity_basis") == "target_sample_audio":
                cand["ecapa_similarity"] = None
        if not cand["translated_audio"] and cand.get("ecapa_similarity_basis") == "translated_audio":
            cand["ecapa_similarity"] = None
        loaded.append(cand)
    return loaded


def rank_loaded_candidates(candidates: list[dict], *, total: int = 12) -> list[dict]:
    def sort_key(cand: dict) -> tuple:
        sim = cand.get("ecapa_similarity")
        has_sim = sim is not None
        try:
            sim_value = float(sim)
        except Exception:
            sim_value = -999.0
        return (
            1 if has_sim else 0,
            sim_value,
            float(cand.get("dur") or 0.0),
            -float(cand.get("start") or 0.0),
        )

    selected = sorted(candidates, key=sort_key, reverse=True)[: max(1, int(total))]
    for idx, cand in enumerate(selected, 1):
        cand["global_rank"] = idx
    return selected


def load_existing_candidates_for_videos(
    videos: list[str | os.PathLike[str]],
    *,
    total: int = 12,
    log: LogCallback = print,
) -> list[dict]:
    loaded = load_all_existing_candidates_for_videos(videos, log=log)
    return rank_loaded_candidates(loaded, total=total) if loaded else []


def load_all_existing_candidates_for_videos(
    videos: list[str | os.PathLike[str]],
    *,
    log: LogCallback = print,
) -> list[dict]:
    loaded: list[dict] = []
    for video_arg in videos:
        path = logic.clone_dir(video_arg) / "single_candidates" / "candidates.json"
        loaded.extend(load_candidate_json(path, log=log))
    return loaded


def _video_key(path: str | os.PathLike[str]) -> str:
    return os.path.normcase(str(Path(path).resolve(strict=False)))


def _candidate_video_key(candidate: dict, videos: list[str | os.PathLike[str]]) -> Optional[str]:
    video = (candidate.get("video") or "").strip()
    if video:
        return _video_key(video)
    source_audio = candidate.get("source_audio") or ""
    if not source_audio:
        return None
    source_path = Path(source_audio).resolve(strict=False)
    for video_arg in videos:
        clone = logic.clone_dir(video_arg).resolve(strict=False)
        try:
            source_path.relative_to(clone)
        except ValueError:
            continue
        return _video_key(video_arg)
    return None


def missing_candidate_videos(
    videos: list[str | os.PathLike[str]],
    candidates: list[dict],
) -> list[str | os.PathLike[str]]:
    requested = {_video_key(video) for video in videos}
    covered = {
        key
        for key in (_candidate_video_key(cand, videos) for cand in candidates)
        if key in requested
    }
    return [video for video in videos if _video_key(video) not in covered]


def collect_candidates_with_existing_for_videos(
    videos: list[str | os.PathLike[str]],
    existing_candidates: list[dict],
    *,
    per_video: int = 6,
    total: int = 12,
    log: LogCallback = print,
) -> list[dict]:
    existing = list(existing_candidates or [])
    if not existing:
        return collect_candidates_for_videos(
            [str(video) for video in videos],
            per_video=per_video,
            total=total,
            log=log,
        )

    missing = missing_candidate_videos(videos, existing)
    covered_count = max(0, len(videos) - len(missing))
    if missing:
        log(
            f"[single-ref] existing candidate cache covers {covered_count}/{len(videos)} video(s); "
            f"collecting {len(missing)} missing video(s)."
        )
        fresh = collect_candidates_for_videos(
            [str(video) for video in missing],
            per_video=per_video,
            total=total,
            log=log,
        )
        candidates = existing + fresh
        for idx, cand in enumerate(candidates, 1):
            cand["global_rank"] = idx
        return candidates

    log(f"[single-ref] existing candidate cache covers all {len(videos)} video(s); reusing cache.")
    return rank_loaded_candidates(existing, total=total)


def _candidate_json_path(candidate: dict) -> Optional[Path]:
    source = candidate.get("source_audio") or ""
    if not source:
        return None
    path = Path(source)
    return path.parent / "candidates.json"


def _persist_candidate_update(candidate: dict) -> None:
    path = _candidate_json_path(candidate)
    if path is None or not path.is_file():
        return
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        return
    for item in data:
        if item.get("id") == candidate.get("id"):
            item.update(candidate)
            break
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _candidate_target_output_path(candidate: dict) -> Path:
    source_audio = candidate.get("source_audio") or ""
    if not source_audio:
        raise ValueError("Candidate source audio is missing.")
    source = Path(source_audio)
    return source.with_name(f"{source.stem.replace('_src', '')}_target.wav")


def _candidate_translated_output_path(candidate: dict) -> Path:
    source_audio = candidate.get("source_audio") or ""
    if not source_audio:
        raise ValueError("Candidate source audio is missing.")
    source = Path(source_audio)
    return source.with_name(f"{source.stem.replace('_src', '')}_translated.wav")


def _write_preview_wav(path: str | os.PathLike[str], clip, sr: int) -> None:
    import wave
    import numpy as np

    arr = np.asarray(clip, dtype=np.float32).reshape(-1)
    pcm = (np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sr))
        wav.writeframes(pcm.tobytes())


def generate_indextts_candidate_previews(
    candidates: list[dict], *, model, target_language: str,
    log_label: Callable[[int, int, dict], str] | None = None,
    log: LogCallback = print, stop_event: Optional[Event] = None,
) -> list[dict]:
    """Generate candidate previews directly from each source reference WAV."""
    from tool_clonevoice_v2 import backend

    total = len(candidates)
    for idx, candidate in enumerate(candidates, 1):
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("Stopped by user.")
        text = (candidate.get("tgt_text") or candidate.get("src_text") or "").strip()
        # Candidate windows may contain several manifest lines. Keep previews
        # bounded so IndexTTS receives a normal utterance instead of a long
        # concatenated paragraph.
        text = text[:80].strip()
        source_audio = candidate.get("source_audio") or ""
        if not text or not source_audio or not Path(source_audio).is_file():
            continue
        label = log_label(idx, total, candidate) if log_label else f"candidate {idx}/{total}"
        output = _candidate_translated_output_path(candidate)
        log(f"[indextts-v2] {label}: source WAV -> target preview")
        try:
            result = model.infer(
                spk_audio_prompt=source_audio,
                text=text,
                lang=backend.normalize_language(target_language).upper(),
                output_path=str(output),
                verbose=False,
                max_text_tokens_per_segment=80,
                num_beams=1,
                repetition_penalty=1.2,
                top_p=0.9,
                temperature=0.8,
            )
        except Exception as exc:
            log(f"[indextts-v2] {label}: preview failed ({exc})")
            candidate["target_sample_text"] = text
            candidate["preview_error"] = str(exc)
            if candidate.get("ecapa_similarity") is None:
                candidate["ecapa_similarity"] = candidate.get("score")
                candidate["ecapa_similarity_basis"] = "candidate_score"
            _persist_candidate_update(candidate)
            continue
        generated = Path(result or output)
        if generated.is_file():
            candidate["target_sample_audio"] = str(generated)
            candidate["target_sample_text"] = text
            candidate["translated_audio"] = str(generated)
            # The extraction score remains useful as a visible quality/similarity
            # value when the optional ECAPA scoring bundle is unavailable.
            if candidate.get("ecapa_similarity") is None:
                candidate["ecapa_similarity"] = candidate.get("score")
                candidate["ecapa_similarity_basis"] = "candidate_score"
            _persist_candidate_update(candidate)
    return candidates


def build_candidate_target_sample_job(
    candidate: dict,
    *,
    model,
    target_language: str,
    log_label: str = "",
    log: LogCallback = print,
    stop_event: Optional[Event] = None,
) -> dict:
    """Generate target-language takes for a candidate using an existing model.

    The returned job intentionally does not score or write the final WAV. Callers
    can release OmniVoice first, then batch-finalize many jobs with one ECAPA load.
    """
    from tool_clonevoice_v2 import omnivoice_backend as ov

    source_audio = candidate.get("source_audio") or ""
    source_text = (candidate.get("src_text") or "").strip()
    if not source_audio:
        raise ValueError("Candidate source audio is missing.")
    if not source_text:
        label = log_label or str(candidate.get("id") or "candidate")
        raise ValueError(f"{label} source transcript text is empty.")
    label = log_label or str(candidate.get("id") or "candidate")

    takes, generic, sr, device = ov._generate_target_reference_takes_with_model(
        model,
        source_ref_audio=source_audio,
        source_ref_text=source_text,
        target_language=target_language,
        log_label=label,
        log=log,
        stop_event=stop_event,
    )
    return {
        "candidate": candidate,
        "takes": takes,
        "generic": generic,
        "source_ref_audio": source_audio,
        "take_sr": sr,
        "device": device,
        "output_wav": str(_candidate_target_output_path(candidate)),
        "label": label,
    }


def finish_candidate_target_sample_jobs(
    jobs: list[dict],
    *,
    models_root: str,
    score_candidates: Optional[list[dict]] = None,
    log: LogCallback = print,
) -> list[dict]:
    """Finalize generated candidate jobs and optionally score existing samples.

    Both operations share one ECAPA model load when possible.
    """
    from tool_clonevoice_v2 import omnivoice_backend as ov

    jobs = list(jobs)
    score_items: list[dict] = []
    score_pairs: list[tuple[str, str]] = []
    score_kinds: list[str] = []
    for cand in score_candidates or []:
        source_audio = cand.get("source_audio") or ""
        translated_audio = cand.get("translated_audio") or ""
        target_audio = translated_audio or cand.get("target_sample_audio") or ""
        if source_audio and target_audio:
            score_items.append(cand)
            score_pairs.append((source_audio, target_audio))
            score_kinds.append("translated_audio" if translated_audio else "target_sample_audio")

    device = str(jobs[0].get("device") or "") if jobs else None
    job_results, pair_results = ov.process_target_reference_batch(
        jobs,
        score_pairs=score_pairs,
        models_root=models_root,
        device=device,
        log=log,
    )

    updated: list[dict] = []
    for job, (wav_path, text, sim) in zip(jobs, job_results):
        candidate = job["candidate"]
        candidate["target_sample_audio"] = wav_path
        candidate["target_sample_text"] = text
        candidate["target_sample_similarity"] = sim
        candidate["ecapa_similarity"] = sim
        _persist_candidate_update(candidate)
        updated.append(candidate)

    for cand, sim, score_kind in zip(score_items, pair_results, score_kinds):
        cand["ecapa_similarity"] = sim
        cand["ecapa_similarity_basis"] = score_kind
        _persist_candidate_update(cand)
        updated.append(cand)

    return updated


def generate_candidate_translated_previews_with_model(
    candidates: list[dict],
    *,
    model,
    target_language: str,
    label_func: Optional[Callable[[int, int, dict], str]] = None,
    log: LogCallback = print,
    stop_event: Optional[Event] = None,
) -> list[dict]:
    """Generate final-chain target-text previews from the frozen target sample.

    This mirrors final ``.SI.WAV`` synthesis for the single-speaker tab:
    source clip -> fixed target-language sample -> translated sentence. The
    preview therefore exposes second-hop drift instead of directly cloning the
    translated text from the source-language clip.
    """
    from tool_clonevoice_v2 import omnivoice_backend as ov
    import numpy as np

    updated: list[dict] = []
    total = len(candidates)
    sr = int(getattr(model, "sampling_rate", None) or 24000)
    for idx, candidate in enumerate(candidates, 1):
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("Stopped by user.")
        label = (
            label_func(idx, total, candidate)
            if label_func is not None
            else f"candidate {idx}/{total} {candidate.get('id') or ''}".strip()
        )
        translated_text = (candidate.get("tgt_text") or "").strip()
        target_audio = candidate.get("target_sample_audio") or ""
        target_text = (candidate.get("target_sample_text") or "").strip()
        if not translated_text:
            log(f"[synth] {label}: no target-language text; final-chain preview skipped")
            continue
        if not target_audio or not Path(target_audio).is_file() or not target_text:
            log(f"[synth] {label}: fixed target sample missing; final-chain preview skipped")
            continue

        translated_out = _candidate_translated_output_path(candidate)
        existing_translated = candidate.get("translated_audio") or ""
        if existing_translated and Path(existing_translated).is_file():
            log(f"[synth] {label}: existing final-chain preview reused")
            continue
        if translated_out.is_file():
            candidate["translated_audio"] = str(translated_out)
            _persist_candidate_update(candidate)
            log(f"[synth] {label}: existing final-chain preview reused")
            continue
        prompt_audio = ov.prepare_prompt_reference_audio(target_audio, log=log)
        prompt = model.create_voice_clone_prompt(prompt_audio, target_text, preprocess_prompt=True)
        # Reproducible preview: same fixed sample + text -> same preview every run,
        # so the "translated vs source" similarity ranking is stable.
        ov._seed_generation(ov._stable_seed(target_audio, translated_text))
        log(f"[synth] {label}: generating final-chain target-language preview audio")
        clip = ov._normalize_peak(
            np.asarray(
                model.generate(
                    text=translated_text,
                    voice_clone_prompt=prompt,
                    language=target_language,
                    duration=None,
                    num_step=ov.DEFAULT_NUM_STEP,
                    guidance_scale=ov.DEFAULT_GUIDANCE,
                )[0],
                dtype=np.float32,
            ).reshape(-1),
            0.85,
        )
        _write_preview_wav(translated_out, clip, sr)
        candidate["translated_audio"] = str(translated_out)
        _persist_candidate_update(candidate)
        updated.append(candidate)
        log(f"[synth] {label}: final-chain target-language preview audio generated")
    return updated


def score_candidate_similarities(
    candidates: list[dict],
    *,
    models_root: str,
    log: LogCallback = print,
) -> list[dict]:
    try:
        return finish_candidate_target_sample_jobs(
            [],
            models_root=models_root,
            score_candidates=candidates,
            log=log,
        )
    except RuntimeError as exc:
        if "ECAPA speaker-similarity model is unavailable" not in str(exc):
            raise
        log(f"[synth] translated preview similarity skipped ({exc})")
        return candidates


def manifest_has_target_translation(video: str | os.PathLike[str], target_language: str) -> bool:
    from tool_clonevoice_v2 import proofread

    manifest = logic.load_manifest(video)
    if not manifest:
        return False
    if (manifest.get("target_language") or "") != target_language:
        return False
    # Lines the user deliberately emptied while proofreading count as done;
    # re-translating here would overwrite every proofread edit.
    cleared = proofread.effective_cleared_segment_ids(manifest)
    translated_srt = logic.clone_dir(video) / "translated.srt"
    segments = [s for s in manifest.get("segments", []) if (s.get("src_text") or "").strip()]
    legacy_completed = (
        "proofread" not in manifest
        and translated_srt.is_file()
        and any((seg.get("tgt_text") or "").strip() for seg in segments)
    )

    def _done(seg: dict) -> bool:
        if (seg.get("tgt_text") or "").strip():
            return True
        if legacy_completed:
            return True
        try:
            return int(seg.get("id")) in cleared
        except Exception:
            return False

    return bool(segments) and all(_done(s) for s in segments)


def ensure_translated(
    video: str | os.PathLike[str],
    *,
    target_language: str,
    api_key: Optional[str] = None,
    source_correction: Optional[bool] = None,
    log: LogCallback = print,
    stop_event: Optional[Event] = None,
) -> dict:
    if manifest_has_target_translation(video, target_language):
        log(f"[single] translated text already exists -> {video}")
        return logic.load_manifest(video) or {}
    log(f"[single] translate candidate text -> {video}")
    return logic.run_translate(
        video,
        target_language=target_language,
        api_key=api_key,
        source_correction=source_correction,
        log=log,
        stop_event=stop_event,
    )


def ensure_translated_for_videos(
    videos: list[str],
    *,
    target_language: str,
    api_key: Optional[str] = None,
    source_correction: Optional[bool] = None,
    log: LogCallback = print,
    stop_event: Optional[Event] = None,
) -> None:
    for video in videos:
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("Stopped by user.")
        ensure_translated(
            video,
            target_language=target_language,
            api_key=api_key,
            source_correction=source_correction,
            log=log,
            stop_event=stop_event,
        )


def generate_voice_design_basis_with_model(
    video_or_root: str | os.PathLike[str],
    *,
    batch: bool,
    model,
    target_language: str,
    instruct: str,
    log: LogCallback = print,
    stop_event: Optional[Event] = None,
) -> tuple[str, str]:
    from tool_clonevoice_v2 import omnivoice_backend as ov

    wav_path, _txt_path = user_visible_speaker1_paths(video_or_root, batch=batch)
    output = wav_path.with_name(wav_path.stem + ".design.wav")
    return ov.generate_voice_design_sample_with_model(
        model,
        target_language=target_language,
        instruct=instruct,
        output_wav=str(output),
        log=log,
        stop_event=stop_event,
    )


def _copy_file_if_different(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
    src_path = Path(src)
    dst_path = Path(dst)
    try:
        if src_path.resolve() == dst_path.resolve():
            return
    except OSError:
        pass
    shutil.copyfile(str(src_path), str(dst_path))


def _copy_basis_to_clone_dir(
    video: Path,
    *,
    basis_wav: str | os.PathLike[str],
    basis_text: str,
    target_language: str,
    source_kind: str,
    meta: Optional[dict],
) -> None:
    cdir = logic.clone_dir(video)
    cdir.mkdir(parents=True, exist_ok=True)
    _copy_file_if_different(basis_wav, cdir / SPEAKER1_WAV)
    (cdir / SPEAKER1_TXT).write_text(basis_text, encoding="utf-8")
    meta_data = dict(meta or {})
    meta_data.update({
        "source": source_kind,
        "target_language": target_language,
        "basis_wav": SPEAKER1_WAV,
        "basis_txt": SPEAKER1_TXT,
    })
    (cdir / SPEAKER1_META).write_text(json.dumps(meta_data, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = logic.load_manifest(video) or {
        "video": str(video),
        "language": "",
        "target_language": target_language,
        "speakers": {},
        "segments": [],
    }
    manifest["target_language"] = target_language
    manifest.setdefault("speakers", {})[SPEAKER_ID] = {
        "ref_audio": SPEAKER1_WAV,
        "ref_text": basis_text,
        "ref_language": target_language,
        "ref_kind": "target_language_basis",
        "skip_work_ref": True,
        "score": 1.0,
        "source": source_kind,
    }
    for seg in manifest.get("segments", []):
        seg["speaker"] = SPEAKER_ID
    logic.save_manifest(video, manifest)


def save_speaker1_basis(
    videos: list[str],
    *,
    basis_wav: str | os.PathLike[str],
    basis_text: str,
    target_language: str,
    visible_target: str | os.PathLike[str],
    batch: bool,
    source_kind: str,
    meta: Optional[dict] = None,
    log: LogCallback = print,
) -> tuple[str, str]:
    if not videos:
        raise ValueError("No videos are available for applying SPEAKER1.")
    if not Path(basis_wav).is_file():
        raise FileNotFoundError(f"SPEAKER1 source WAV not found: {basis_wav}")
    text = (basis_text or "").strip()
    if not text:
        raise ValueError("SPEAKER1 text is empty.")

    visible_wav, visible_txt = user_visible_speaker1_paths(visible_target, batch=batch)
    visible_wav.parent.mkdir(parents=True, exist_ok=True)
    _copy_file_if_different(basis_wav, visible_wav)
    visible_txt.write_text(text, encoding="utf-8")
    log(f"[single] visible SPEAKER1 -> {visible_wav}")

    for video in videos:
        _copy_basis_to_clone_dir(
            Path(video),
            basis_wav=visible_wav,
            basis_text=text,
            target_language=target_language,
            source_kind=source_kind,
            meta=meta,
        )
        log(f"[single] applied SPEAKER1 -> {logic.clone_dir(video) / SPEAKER1_WAV}")
    return str(visible_wav), str(visible_txt)


def translate_and_synthesize(
    videos: list[str],
    *,
    target_language: str,
    models_root: str,
    api_key: Optional[str] = None,
    source_correction: Optional[bool] = None,
    tempo_fit: str = "moderate",
    level_match: bool = True,
    skip_existing: bool = True,
    log: LogCallback = print,
    stop_event: Optional[Event] = None,
    model_holder: Optional[list] = None,
) -> dict:
    from tool_clonevoice_v2 import backend
    from tool_si import logic as si

    outputs: list[str] = []
    written: list[str] = []
    skipped: list[str] = []
    # One model for the whole run. Leaving ``model`` unset made run_synthesize
    # load its own for every video and nothing released the previous one, so a
    # three-part title carried three copies: measured on hnvr-174, VRAM
    # allocated went 0.01 -> 5.60 -> 11.18 GB across the three parts, which is
    # what "the card fills up and it crawls" looks like. run_batch already
    # loads once and passes it down; this path did not.
    shared_model = None
    for video in videos:
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("Stopped by user.")
        out_path = si.default_si_audio_path(video)
        # "Up to date", not merely "present": proofreading a video and exporting
        # again with this ticked used to skip the whole video and drop the edits
        # on the floor, with `skipped` in the log as the only trace.
        if skip_existing and logic.synthesis_is_current(
            video, language=target_language, tempo_fit=tempo_fit,
            level_match=level_match,
        ):
            log(f"[single] target .SI.WAV is up to date; skipped: {out_path}")
            si.prepare_dlna_audio(video, out_path, log_callback=log, stop_event=stop_event)
            outputs.append(out_path)
            skipped.append(out_path)
            continue
        ensure_translated(
            video,
            target_language=target_language,
            api_key=api_key,
            source_correction=source_correction,
            log=log,
            stop_event=stop_event,
        )
        log(f"[single] synthesize -> {video}")
        # Loaded on the first video that actually needs it, so a run where
        # every output already exists never pays for it.
        if shared_model is None:
            shared_model = backend.load_model(models_root, log=log)
            if model_holder is not None:
                model_holder.append(shared_model)
        out = logic.run_synthesize(
            video,
            models_root=models_root,
            model=shared_model,
            text_field="tgt_text",
            language=target_language,
            tempo_fit=tempo_fit,
            level_match=level_match,
            log=log,
            stop_event=stop_event,
            model_holder=model_holder,
        )
        outputs.append(out)
        written.append(out)
    return {"outputs": outputs, "written": written, "skipped": skipped}
