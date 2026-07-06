"""Multi-speaker voice-clone workflow helpers.

This module keeps the first multi-speaker GUI pass focused on single-file
workflows: transcribe with diarization, choose a basis per speaker, and reuse
the existing multi-speaker synthesis path.
"""
from __future__ import annotations

import json
import os
import shutil
import wave
from pathlib import Path
from threading import Event
from typing import Callable, Optional

import numpy as np

from tool_clonevoice import diarize as diar
from tool_clonevoice import logic
from tool_clonevoice import refsel
from tool_clonevoice import whisperx_backend as wx

LogCallback = Callable[[str], None]

BASIS_WAV_SUFFIX = ".basis.wav"
BASIS_TXT_SUFFIX = ".basis.txt"
BASIS_META_SUFFIX = ".basis.meta.json"
# Evaluate this many times the displayed candidate count (see single_clone).
CANDIDATE_POOL_FACTOR = 2
GLOBAL_DIARIZE_WARN_SECONDS = 2 * 60 * 60


def run_multi_transcribe(
    video_path: str | os.PathLike[str],
    *,
    model_key: str,
    language: Optional[str],
    target_language: str,
    models_root: str,
    diarize_backend: str = "auto",
    num_speakers: Optional[int] = None,
    denoise: str = "none",
    log: LogCallback = print,
    stop_event: Optional[Event] = None,
    model_holder: Optional[list] = None,
    precomputed_turns: Optional[list] = None,
) -> dict:
    """Transcribe one video with multi-speaker diarization enabled."""
    return logic.run_transcribe_diarize(
        video_path,
        model_key=model_key,
        language=language,
        diarize_backend=diarize_backend,
        num_speakers=num_speakers,
        target_language=target_language,
        models_root=models_root,
        denoise=denoise,
        precomputed_turns=precomputed_turns,
        log=log,
        stop_event=stop_event,
        model_holder=model_holder,
    )


def list_speakers(video_path: str | os.PathLike[str]) -> list[dict]:
    """Return speaker stats from a manifest, sorted by total speech duration."""
    manifest = logic.load_manifest(video_path)
    if not manifest:
        return []
    stats: dict[str, dict] = {}
    for speaker in manifest.get("speakers", {}) or {}:
        stats[str(speaker)] = {"speaker": str(speaker), "total_dur": 0.0, "seg_count": 0}
    for segment in manifest.get("segments", []) or []:
        speaker = str(segment.get("speaker") or "SPEAKER_00")
        item = stats.setdefault(speaker, {"speaker": speaker, "total_dur": 0.0, "seg_count": 0})
        try:
            dur = float(segment.get("dur", float(segment.get("end", 0.0)) - float(segment.get("start", 0.0))))
        except Exception:
            dur = 0.0
        item["total_dur"] += max(0.0, dur)
        item["seg_count"] += 1
    return sorted(
        (
            {
                "speaker": item["speaker"],
                "total_dur": round(float(item["total_dur"]), 3),
                "seg_count": int(item["seg_count"]),
            }
            for item in stats.values()
        ),
        key=lambda item: (-float(item["total_dur"]), item["speaker"]),
    )


def speaker_ids(video_path: str | os.PathLike[str]) -> list[str]:
    return [item["speaker"] for item in list_speakers(video_path)]


def list_global_speakers(videos: list[str | os.PathLike[str]]) -> list[dict]:
    """Return union speaker stats across already-transcribed videos."""
    stats: dict[str, dict] = {}
    for video in videos:
        for item in list_speakers(video):
            speaker = str(item["speaker"])
            dst = stats.setdefault(speaker, {"speaker": speaker, "total_dur": 0.0, "seg_count": 0})
            dst["total_dur"] += float(item.get("total_dur") or 0.0)
            dst["seg_count"] += int(item.get("seg_count") or 0)
    return sorted(
        (
            {
                "speaker": item["speaker"],
                "total_dur": round(float(item["total_dur"]), 3),
                "seg_count": int(item["seg_count"]),
            }
            for item in stats.values()
        ),
        key=lambda item: (-float(item["total_dur"]), item["speaker"]),
    )


def videos_with_speaker(videos: list[str | os.PathLike[str]], speaker: str) -> list[str]:
    out: list[str] = []
    for video in videos:
        manifest = logic.load_manifest(video) or {}
        if speaker in (manifest.get("speakers") or {}):
            out.append(str(video))
    return out


def _speaker_total_dur(segments: list[dict], speaker: str) -> float:
    total = 0.0
    for s in segments:
        if s.get("speaker") != speaker:
            continue
        try:
            total += max(0.0, float(s.get("dur", float(s.get("end", 0.0)) - float(s.get("start", 0.0)))))
        except Exception:
            pass
    return total


def extract_shared_references(
    videos: list[str | os.PathLike[str]],
    *,
    log: LogCallback = print,
    stop_event: Optional[Event] = None,
) -> None:
    """Auto-pick ONE source reference per global speaker and share it across videos.

    The one-click "same folder" mode: after global diarization + transcription,
    pool each global speaker's candidate spans across every video, auto-pick the
    best (3-10s duration pool + _score, with a stitched fallback for sparse
    speakers), cut it once, and write it into every video's manifest so the whole
    folder clones that speaker from the same reference.
    """
    from tool_clonevoice import refsel

    speakers = [item["speaker"] for item in list_global_speakers(videos)]
    for speaker in speakers:
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("Stopped by user.")
        vids = videos_with_speaker(videos, speaker)
        if not vids:
            continue
        pooled: list[dict] = []
        seg_by_video: dict[str, list[dict]] = {}
        for video in vids:
            manifest = logic.load_manifest(video)
            if not manifest:
                continue
            segs = manifest.get("segments", [])
            seg_by_video[str(Path(video))] = segs
            audio16k = logic.clone_dir(video) / logic.AUDIO16K_NAME
            if not audio16k.is_file():
                continue
            audio, sr = refsel._read_wav_mono(str(audio16k))
            for cand in refsel._speaker_candidates(speaker, manifest, segs, audio, sr):
                cand["_video"] = str(Path(video))
                pooled.append(cand)
        if not pooled:
            log(f"[shared-ref] {speaker}: no candidates, skipped.")
            continue

        pool = refsel._select_candidate_pool(pooled, refsel.AUTO_REF_POOL)
        best = max(pool, key=lambda c: float(c.get("score", 0.0)))
        origin = best.get("_video") or str(Path(vids[0]))
        # No continuous 3-10s span anywhere: stitch within the richest video.
        if float(best.get("dur", 0.0)) < refsel.CAND_POOL_MIN_DUR:
            origin = max(
                (str(Path(v)) for v in vids),
                key=lambda v: _speaker_total_dur(seg_by_video.get(v, []), speaker),
                default=str(Path(vids[0])),
            )
            stitched = refsel._build_stitched_candidate(speaker, seg_by_video.get(origin, []))
            if stitched is not None:
                best = stitched
                log(f"[shared-ref] {speaker}: no 3-10s span; stitched {len(stitched['pieces'])} clips "
                    f"-> {stitched['dur']:.1f}s.")

        ref_name = f"ref_{speaker}.wav"
        origin_ref = logic.clone_dir(Path(origin)) / ref_name
        if best.get("pieces"):
            refsel._cut_ref_multi(origin, best["pieces"], str(origin_ref), log)
        else:
            refsel._cut_ref(origin, float(best["start"]), float(best["end"]), str(origin_ref), log)

        for video in vids:
            dst = logic.clone_dir(Path(video)) / ref_name
            if str(Path(video)) != str(Path(origin)):
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(str(origin_ref), str(dst))
            manifest = logic.load_manifest(video)
            manifest.setdefault("speakers", {})[speaker] = {
                "ref_audio": ref_name,
                "ref_text": (best.get("text") or "").strip(),
                "score": round(float(best.get("score", 0.0)), 3),
                "source": best.get("source", "segment"),
                "source_srt_refs": best.get("source_srt_refs", []),
                "start": round(float(best.get("start", 0.0)), 3),
                "end": round(float(best.get("end", 0.0)), 3),
                "dur": round(float(best.get("dur", 0.0)), 3),
            }
            logic.save_manifest(video, manifest)
        log(f"[shared-ref] {speaker}: shared reference {ref_name} ({float(best.get('dur',0.0)):.1f}s) "
            f"applied to {len(vids)} video(s).")


def estimate_total_video_duration(videos: list[str | os.PathLike[str]], *, log: LogCallback = print) -> float:
    """Return total probed duration in seconds; unknown files count as 0."""
    from gpu_engine import probe

    total = 0.0
    for video in videos:
        try:
            meta = probe.probe_video(video)
            total += max(0.0, float(meta.duration or 0.0))
        except Exception as exc:
            log(f"[multi-global] could not probe duration for {video}: {exc}")
    return total


def _read_wav_mono_f32(path: str | os.PathLike[str]) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        channels = int(wav.getnchannels())
        sample_width = int(wav.getsampwidth())
        sr = int(wav.getframerate())
        frames = wav.readframes(wav.getnframes())
    if sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio.astype(np.float32, copy=False), sr


def _write_wav_mono_f32(path: str | os.PathLike[str], audio: np.ndarray, sr: int) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    with wave.open(str(out), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sr))
        wav.writeframes(pcm16.tobytes())


def split_turns_to_video(
    turns: list[tuple[float, float, str]],
    *,
    offset: float,
    duration: float,
    min_duration: float = 0.2,
) -> list[tuple[float, float, str]]:
    """Clip global diarization turns to one video's local timeline."""
    start_bound = float(offset)
    end_bound = start_bound + float(duration)
    local: list[tuple[float, float, str]] = []
    for start, end, speaker in turns:
        clipped_start = max(float(start), start_bound)
        clipped_end = min(float(end), end_bound)
        if clipped_end - clipped_start < min_duration:
            continue
        local.append((round(clipped_start - start_bound, 3), round(clipped_end - start_bound, 3), str(speaker)))
    return local


def prescan_global_diarize(
    videos: list[str | os.PathLike[str]],
    *,
    models_root: str,
    diarize_backend: str = "pyannote",
    num_speakers: Optional[int],
    denoise: str = "none",
    silence_gap: float = 1.0,
    log: LogCallback = print,
    stop_event: Optional[Event] = None,
) -> dict[str, list[tuple[float, float, str]]]:
    """Run one diarization pass over concatenated batch audio.

    Returns ``{video_path: local_turns}`` where speaker labels are global across
    the batch. The temporary concatenated WAV is removed after diarization.
    """
    if not videos:
        return {}
    # num_speakers=None -> let the diarizer auto-detect the count (used by the
    # per-subfolder batch mode where each folder may have a different headcount).

    def _check_stop() -> None:
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("Stopped by user.")

    parts: list[np.ndarray] = []
    offsets: dict[str, float] = {}
    durations: dict[str, float] = {}
    sr: Optional[int] = None
    cursor = 0.0
    for index, video_arg in enumerate(videos, 1):
        _check_stop()
        video = Path(video_arg)
        cdir = logic.clone_dir(video)
        cdir.mkdir(parents=True, exist_ok=True)
        audio16k = cdir / logic.AUDIO16K_NAME
        log(f"[multi-global] extract audio {index}/{len(videos)} -> {video}")
        wx.extract_audio_16k(str(video), str(audio16k), log=log, stop_event=stop_event, denoise=denoise)
        audio, part_sr = _read_wav_mono_f32(audio16k)
        if sr is None:
            sr = part_sr
        elif part_sr != sr:
            raise ValueError(f"Unexpected sample rate {part_sr}; expected {sr}: {audio16k}")
        offsets[str(video)] = cursor
        duration = audio.size / float(sr or 1)
        durations[str(video)] = duration
        parts.append(audio)
        cursor += duration
        if index < len(videos):
            gap = np.zeros(max(0, int(round(float(silence_gap) * float(sr)))), dtype=np.float32)
            parts.append(gap)
            cursor += gap.size / float(sr)

    if sr is None:
        return {}
    concat = np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)
    concat_path = logic.clone_dir(Path(videos[0])) / "global_diarize_concat.wav"
    _write_wav_mono_f32(concat_path, concat, sr)
    try:
        _check_stop()
        torch_device, _ = wx.resolve_device()
        log(
            f"[multi-global] diarize {len(videos)} video(s), "
            f"{concat.size / float(sr):.1f}s concat, speakers={num_speakers}"
        )
        global_turns = diar.diarize(
            str(concat_path),
            backend=diarize_backend,
            num_speakers=num_speakers,
            models_root=models_root,
            device=torch_device,
            log=log,
        )
    finally:
        try:
            concat_path.unlink()
        except OSError:
            pass

    out: dict[str, list[tuple[float, float, str]]] = {}
    for video_arg in videos:
        key = str(Path(video_arg))
        local = split_turns_to_video(global_turns, offset=offsets[key], duration=durations[key])
        out[key] = local
        log(f"[multi-global] {Path(key).name}: {len(local)} local global-speaker turn(s)")
    return out


def collect_speaker_candidates(
    video_path: str | os.PathLike[str],
    speaker: str,
    *,
    top_n: int = 12,
    log: LogCallback = print,
) -> list[dict]:
    """Collect Top-N source candidates for a specific diarized speaker."""
    video = Path(video_path)
    manifest = logic.load_manifest(video)
    if not manifest:
        raise FileNotFoundError(f"Manifest not found; run transcription first: {logic.manifest_path(video)}")
    cdir = logic.clone_dir(video)
    audio16k = cdir / logic.AUDIO16K_NAME
    if not audio16k.is_file():
        raise FileNotFoundError(f"Intermediate audio missing: {audio16k}")
    candidates = refsel.collect_reference_candidates(
        str(video),
        manifest,
        str(audio16k),
        cdir,
        speaker=speaker,
        top_n=top_n,
        output_dir_name=f"candidates_{speaker}",
        log=log,
    )
    filtered = [cand for cand in candidates if (cand.get("src_text") or "").strip()]
    skipped = len(candidates) - len(filtered)
    if skipped:
        log(f"[multi] {speaker}: skipped {skipped} candidate(s) without source transcript text")
    for index, cand in enumerate(filtered, 1):
        cand["global_rank"] = index
    return filtered


def collect_speaker_candidates_for_videos(
    videos: list[str | os.PathLike[str]],
    speaker: str,
    *,
    per_video: int = 6,
    total: int = 12,
    log: LogCallback = print,
) -> list[dict]:
    all_candidates: list[dict] = []
    for video in videos:
        if speaker not in speaker_ids(video):
            continue
        try:
            all_candidates.extend(collect_speaker_candidates(video, speaker, top_n=per_video, log=log))
        except FileNotFoundError as exc:
            log(f"[multi] {speaker}: candidate collection skipped for {video} ({exc})")
    # Duration-first (longest clip wins), matching the per-video pool selection —
    # deterministic and prefers stronger references.
    all_candidates.sort(key=lambda c: (-float(c.get("dur") or 0.0), float(c.get("start") or 0.0)))
    selected = all_candidates[: max(1, int(total))]
    for index, cand in enumerate(selected, 1):
        cand["global_rank"] = index
    return selected


def _speaker_basis_names(speaker: str) -> tuple[str, str, str]:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in (speaker or "SPEAKER"))
    return f"{safe}{BASIS_WAV_SUFFIX}", f"{safe}{BASIS_TXT_SUFFIX}", f"{safe}{BASIS_META_SUFFIX}"


def save_speaker_basis(
    video_path: str | os.PathLike[str],
    speaker: str,
    *,
    basis_wav: str | os.PathLike[str],
    basis_text: str,
    target_language: str,
    source_kind: str,
    meta: Optional[dict] = None,
    log: LogCallback = print,
) -> tuple[str, str]:
    """Apply a target-language basis to one speaker without changing segments."""
    video = Path(video_path)
    if not Path(basis_wav).is_file():
        raise FileNotFoundError(f"Basis WAV not found: {basis_wav}")
    text = (basis_text or "").strip()
    if not text:
        raise ValueError("Basis text is empty.")

    cdir = logic.clone_dir(video)
    cdir.mkdir(parents=True, exist_ok=True)
    wav_name, txt_name, meta_name = _speaker_basis_names(speaker)
    shutil.copyfile(str(basis_wav), str(cdir / wav_name))
    (cdir / txt_name).write_text(text, encoding="utf-8")
    meta_data = dict(meta or {})
    meta_data.update(
        {
            "speaker": speaker,
            "source": source_kind,
            "target_language": target_language,
            "basis_wav": wav_name,
            "basis_txt": txt_name,
        }
    )
    (cdir / meta_name).write_text(json.dumps(meta_data, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = logic.load_manifest(video)
    if not manifest:
        raise FileNotFoundError(f"Manifest not found; run transcription first: {logic.manifest_path(video)}")
    manifest["target_language"] = target_language
    manifest.setdefault("speakers", {})[speaker] = {
        "ref_audio": wav_name,
        "ref_text": text,
        "ref_language": target_language,
        "ref_kind": "target_language_basis",
        "skip_work_ref": True,
        "score": 1.0,
        "source": source_kind,
    }
    logic.save_manifest(video, manifest)
    log(f"[multi] applied {speaker} basis -> {cdir / wav_name}")
    return str(cdir / wav_name), str(cdir / txt_name)


def save_speaker_basis_for_videos(
    videos: list[str | os.PathLike[str]],
    speaker: str,
    *,
    basis_wav: str | os.PathLike[str],
    basis_text: str,
    target_language: str,
    source_kind: str,
    meta: Optional[dict] = None,
    log: LogCallback = print,
) -> list[tuple[str, str, str]]:
    """Apply one global speaker basis to every video containing that speaker."""
    saved: list[tuple[str, str, str]] = []
    for video in videos:
        if speaker not in speaker_ids(video):
            continue
        wav_path, txt_path = save_speaker_basis(
            video,
            speaker,
            basis_wav=basis_wav,
            basis_text=basis_text,
            target_language=target_language,
            source_kind=source_kind,
            meta=meta,
            log=log,
        )
        saved.append((str(video), wav_path, txt_path))
    if not saved:
        raise ValueError(f"Speaker {speaker} is not present in the selected video(s).")
    return saved


def speaker_has_basis(video_path: str | os.PathLike[str], speaker: str) -> bool:
    manifest = logic.load_manifest(video_path)
    if not manifest:
        return False
    info = (manifest.get("speakers") or {}).get(speaker) or {}
    ref_audio = info.get("ref_audio") or ""
    ref_text = (info.get("ref_text") or "").strip()
    return bool(ref_audio and ref_text and (logic.clone_dir(video_path) / ref_audio).is_file())


def set_speaker_skipped(
    video_path: str | os.PathLike[str],
    speaker: str,
    *,
    skipped: bool,
    log: LogCallback = print,
) -> None:
    """Mark one speaker as excluded from SI synthesis."""
    manifest = logic.load_manifest(video_path)
    if not manifest:
        raise FileNotFoundError(f"Manifest not found; run transcription first: {logic.manifest_path(video_path)}")
    info = manifest.setdefault("speakers", {}).setdefault(speaker, {})
    if skipped:
        info["skip_synthesis"] = True
        log(f"[multi] {speaker}: marked skipped; no SI voice will be generated for this speaker")
    else:
        info.pop("skip_synthesis", None)
        log(f"[multi] {speaker}: skip cleared")
    logic.save_manifest(video_path, manifest)


def set_skipped_speakers(
    video_path: str | os.PathLike[str],
    skipped: set[str],
    *,
    log: LogCallback = print,
) -> None:
    """Persist the current skip set and clear stale skip flags."""
    manifest = logic.load_manifest(video_path)
    if not manifest:
        raise FileNotFoundError(f"Manifest not found; run transcription first: {logic.manifest_path(video_path)}")
    skipped = set(skipped or set())
    for speaker, info in manifest.setdefault("speakers", {}).items():
        if speaker in skipped:
            info["skip_synthesis"] = True
        else:
            info.pop("skip_synthesis", None)
    logic.save_manifest(video_path, manifest)
    if skipped:
        log(f"[multi] skipped speakers: {', '.join(sorted(skipped))}")


def set_skipped_speakers_for_videos(
    videos: list[str | os.PathLike[str]],
    skipped: set[str],
    *,
    log: LogCallback = print,
) -> None:
    for video in videos:
        set_skipped_speakers(str(video), skipped, log=log)


def generate_voice_design_basis_with_model(
    video_path: str | os.PathLike[str],
    speaker: str,
    *,
    model,
    target_language: str,
    instruct: str,
    log: LogCallback = print,
    stop_event: Optional[Event] = None,
) -> tuple[str, str]:
    """Generate a target-language designed-voice basis WAV for one speaker."""
    from tool_clonevoice import omnivoice_backend as ov

    cdir = logic.clone_dir(video_path)
    cdir.mkdir(parents=True, exist_ok=True)
    wav_name, _txt_name, _meta_name = _speaker_basis_names(speaker)
    output = cdir / wav_name.replace(BASIS_WAV_SUFFIX, ".design.wav")
    return ov.generate_voice_design_sample_with_model(
        model,
        target_language=target_language,
        instruct=instruct,
        output_wav=str(output),
        log=log,
        stop_event=stop_event,
    )


def export_speaker_basis(
    video_path: str | os.PathLike[str],
    speaker: str,
    target_dir: str | os.PathLike[str],
    *,
    log: LogCallback = print,
) -> tuple[str, str, Optional[str]]:
    """Copy one speaker basis out of the clone directory for reuse."""
    video = Path(video_path)
    manifest = logic.load_manifest(video)
    if not manifest:
        raise FileNotFoundError(f"Manifest not found; run transcription first: {logic.manifest_path(video)}")
    info = (manifest.get("speakers") or {}).get(speaker) or {}
    ref_audio = info.get("ref_audio") or ""
    ref_text = (info.get("ref_text") or "").strip()
    if not ref_audio or not ref_text:
        raise ValueError(f"No basis is configured for {speaker}.")

    cdir = logic.clone_dir(video)
    src_wav = cdir / ref_audio
    if not src_wav.is_file():
        raise FileNotFoundError(f"Basis WAV not found: {src_wav}")
    dst_dir = Path(target_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    safe_video = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in video.stem)
    safe_speaker = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in speaker)
    prefix = f"{safe_video}.{safe_speaker}.basis"
    dst_wav = dst_dir / f"{prefix}.wav"
    dst_txt = dst_dir / f"{prefix}.txt"
    dst_meta = dst_dir / f"{prefix}.meta.json"
    shutil.copyfile(str(src_wav), str(dst_wav))
    dst_txt.write_text(ref_text, encoding="utf-8")

    _, _txt_name, meta_name = _speaker_basis_names(speaker)
    src_meta = cdir / meta_name
    meta_out: Optional[str] = None
    if src_meta.is_file():
        shutil.copyfile(str(src_meta), str(dst_meta))
        meta_out = str(dst_meta)
    log(f"[multi] exported {speaker} basis -> {dst_wav}")
    return str(dst_wav), str(dst_txt), meta_out


def all_speakers_have_basis(
    video_path: str | os.PathLike[str],
    *,
    skipped: Optional[set[str]] = None,
) -> bool:
    skipped = set(skipped or set())
    speakers = speaker_ids(video_path)
    required = [speaker for speaker in speakers if speaker not in skipped]
    return bool(speakers) and all(speaker_has_basis(video_path, speaker) for speaker in required)


def all_videos_have_basis(
    videos: list[str | os.PathLike[str]],
    *,
    skipped: Optional[set[str]] = None,
) -> bool:
    skipped = set(skipped or set())
    speakers = [item["speaker"] for item in list_global_speakers(videos)]
    if not speakers:
        return False
    for speaker in speakers:
        if speaker in skipped:
            continue
        for video in videos_with_speaker(videos, speaker):
            if not speaker_has_basis(video, speaker):
                return False
    return True
