"""tool_clonevoice orchestration logic.

Voice clone translation pipeline. See
``summary/summary_20260610_CLONEVOICE_TRANSLATE_DEV_PLAN_CN.md`` for the full plan.

Stages (each reads/writes a shared manifest JSON, resumable when intermediate
files are kept):
    1. WhisperX transcription + word-level alignment + speaker diarization   [P1]
    2. Per-speaker reference sample extraction (+ per-line emotion refs)      [P2]
    3. AI translation (default config/translate_prompt_dubbing.txt)          [P3]
    4. OmniVoice voice cloning with hard ``duration`` alignment              [P4]
    5. Single-timeline assembly -> ``<video>.si.wav``                        [P4]
    6. (optional) remux into the video (reuses tool_si)                      [P5]

Downstream audio helpers (timeline mix, fit-to-duration, wav IO, ffmpeg remux,
HF download, CUDA release) are reused from ``tool_si.logic``; the LLM client and
translation config come from ``tool_subtitle.logic``.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
from threading import Event
from typing import Callable, Optional

from tool_clonevoice import diarize as diar
from tool_clonevoice import refsel
from tool_clonevoice import whisperx_backend as wx

LogCallback = Callable[[str], None]

MANIFEST_NAME = "manifest.json"
AUDIO16K_NAME = "audio16k.wav"


# --- intermediate directory / manifest IO ---

def clone_dir(video_path: str | Path) -> Path:
    video = Path(video_path)
    return video.parent / (video.stem + ".clone")


def manifest_path(video_path: str | Path) -> Path:
    return clone_dir(video_path) / MANIFEST_NAME


def load_manifest(video_path: str | Path) -> Optional[dict]:
    path = manifest_path(video_path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(video_path: str | Path, manifest: dict) -> Path:
    cdir = clone_dir(video_path)
    cdir.mkdir(parents=True, exist_ok=True)
    path = cdir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _release_cuda_cache() -> None:
    try:
        import torch
    except Exception:
        return
    if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def release_model_holder(model_holder: Optional[list]) -> None:
    """Release native/torch models collected by a caller-controlled holder."""
    if model_holder is not None:
        model_holder.clear()
    gc.collect()
    _release_cuda_cache()


def _format_srt_ts(t: float) -> str:
    t = max(0.0, float(t))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    if ms >= 1000:
        ms -= 1000
        s += 1
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path: str | Path, segments: list, text_key: str, speaker_prefix: bool = False) -> Path:
    """Write a debug SRT from manifest segments using ``text_key`` for the body."""
    blocks = []
    idx = 0
    for s in segments:
        txt = (s.get(text_key) or "").strip()
        if not txt:
            continue
        idx += 1
        if speaker_prefix and s.get("speaker"):
            txt = f"[{s['speaker']}] {txt}"
        blocks.append(
            f"{idx}\n{_format_srt_ts(s['start'])} --> {_format_srt_ts(s['end'])}\n{txt}\n"
        )
    path = Path(path)
    path.write_text("\n".join(blocks), encoding="utf-8")
    return path


# --- Stage 1: transcribe + align + diarize ---

# faster-whisper occasionally emits a stray word timestamp far from the rest of
# a segment (a 0.3s word at t=2522 while the real speech is at t=2623), which
# inflates the whisper segment to 100s+. Placed at the stray word's time, such a
# segment leaves its whole slot silent until the next one. Split a segment
# wherever consecutive words are farther apart than this, so each burst of
# speech lands at its true time.
MAX_WORD_GAP = 2.0


def _split_on_word_gaps(start: float, end: float, text: str, words: list, max_gap: float = MAX_WORD_GAP) -> list:
    """Split one segment into bursts separated by word gaps > ``max_gap``.

    ``words`` are ``{"w", "start", "end"}`` dicts. With no usable words the
    segment is returned unchanged; a single burst keeps the original (richer)
    text but clamps its bounds to the real word extent; multiple bursts rebuild
    each sub-text from the per-word tokens.
    """
    if not words:
        return [{"start": start, "end": end, "text": text, "words": []}]
    groups = [[words[0]]]
    for w in words[1:]:
        if float(w["start"]) - float(groups[-1][-1]["end"]) > max_gap:
            groups.append([w])
        else:
            groups[-1].append(w)
    if len(groups) == 1:
        return [{"start": float(words[0]["start"]), "end": float(words[-1]["end"]),
                 "text": text, "words": words}]
    subs = []
    for g in groups:
        t = "".join(x.get("w", "") for x in g).strip()
        subs.append({"start": float(g[0]["start"]), "end": float(g[-1]["end"]),
                     "text": t, "words": g})
    return subs


def run_transcribe_diarize(
    video_path: str | Path,
    *,
    model_key: str = "large-v3",
    language: Optional[str] = None,
    diarize_backend: str = "auto",
    num_speakers: Optional[int] = None,
    target_language: str = "",
    ref_strategy: str = "hybrid",
    models_root: str,
    keep_intermediate: bool = True,
    log: LogCallback = print,
    stop_event: Optional[Event] = None,
    model_holder: Optional[list] = None,
) -> dict:
    """Transcribe + word-align + diarize a video, writing the manifest.

    ``model_holder`` collects native (CTranslate2/torch) models so the caller
    can release them on the main thread, avoiding a background-thread C++
    destructor crash.
    """
    video = Path(video_path)
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")

    def _check_stop():
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("Stopped by user.")

    cdir = clone_dir(video)
    cdir.mkdir(parents=True, exist_ok=True)
    audio_wav = cdir / AUDIO16K_NAME

    _check_stop()
    wx.extract_audio_16k(str(video), str(audio_wav), log=log, stop_event=stop_event)

    import whisperx

    audio = whisperx.load_audio(str(audio_wav))
    asr_device, asr_compute = wx.resolve_asr_device()
    torch_device, _ = wx.resolve_device()
    model_arg = wx.resolve_model_arg(model_key, models_root)
    log(f"[device] transcription={asr_device}/{asr_compute}, align+diarize={torch_device}")
    if asr_device == "cpu" and torch_device == "cuda":
        log("[device] CTranslate2 cuDNN 8 DLLs not found; transcription runs on CPU to avoid a hard crash.")

    with wx.torch_load_compat():
        _check_stop()
        result = wx.transcribe(
            audio, model_arg, asr_device, asr_compute, language, log=log, model_holder=model_holder
        )
        detected_lang = result.get("language") or (language or "")
        raw_segments = result.get("segments", [])

        _check_stop()
        resolved_backend = diar.resolve_backend(diarize_backend, models_root)
        turns = diar.diarize(
            str(audio_wav), backend=diarize_backend, num_speakers=num_speakers,
            models_root=models_root, device=torch_device, log=log,
        )

    segments = []
    sidx = 0
    for s in raw_segments:
        if s.get("start") is None or s.get("end") is None:
            continue
        words = [
            {"w": w.get("word", ""), "start": float(w["start"]), "end": float(w["end"])}
            for w in s.get("words", [])
            if w.get("start") is not None and w.get("end") is not None
        ]
        # Repair faster-whisper's stray-timestamp segments (see _split_on_word_gaps).
        for sub in _split_on_word_gaps(float(s["start"]), float(s["end"]),
                                       (s.get("text") or "").strip(), words):
            sidx += 1
            segments.append({
                "id": sidx,
                "srt_index": sidx,
                "start": round(sub["start"], 3),
                "end": round(sub["end"], 3),
                "dur": round(sub["end"] - sub["start"], 3),
                "speaker": "SPEAKER_00",
                "src_text": sub["text"],
                "tgt_text": "",
                "emotion_ref": "",
                "words": sub["words"],
            })

    diar.assign_speakers(segments, turns)
    speaker_ids = sorted({seg["speaker"] for seg in segments})

    manifest = {
        "video": str(video),
        "language": detected_lang,
        "target_language": target_language,
        "diarize_backend": resolved_backend,
        "ref_strategy": ref_strategy,
        "diarization_turns": [
            {"start": round(float(ts), 3), "end": round(float(te), 3), "speaker": spk}
            for ts, te, spk in turns
        ],
        "speakers": {spk: {"ref_audio": "", "ref_text": "", "score": 0.0} for spk in speaker_ids},
        "segments": segments,
    }
    save_manifest(video, manifest)
    write_srt(cdir / "source.srt", segments, "src_text", speaker_prefix=True)
    log(f"[manifest] {len(segments)} segments, {len(speaker_ids)} speaker(s) -> {manifest_path(video)}")
    log(f"[srt] source subtitles -> {cdir / 'source.srt'}")

    _release_cuda_cache()
    return manifest


# --- Stage 2: per-speaker reference extraction ---

def run_extract_references(
    video_path: str | Path,
    *,
    models_root: str,
    log: LogCallback = print,
    stop_event: Optional[Event] = None,
) -> dict:
    """Select + cut one reference clip per speaker, updating the manifest.

    Requires the manifest produced by :func:`run_transcribe_diarize` and the
    intermediate ``audio16k.wav`` to be present.
    """
    video = Path(video_path)
    manifest = load_manifest(video)
    if not manifest:
        raise FileNotFoundError(f"Manifest not found; run transcription first: {manifest_path(video)}")

    cdir = clone_dir(video)
    audio_wav = cdir / AUDIO16K_NAME
    if not audio_wav.is_file():
        raise FileNotFoundError(f"Intermediate audio missing: {audio_wav}")

    if stop_event is not None and stop_event.is_set():
        raise RuntimeError("Stopped by user.")

    refsel.extract_references(str(video), manifest, str(audio_wav), cdir, log=log)
    save_manifest(video, manifest)
    log(f"[manifest] references updated -> {manifest_path(video)}")
    return manifest


# --- Stage 3: AI translation (reuses tool_subtitle's LLM client + dubbing prompt) ---

def run_translate(
    video_path: str | Path,
    *,
    target_language: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.5,
    max_retries: int = 3,
    log: LogCallback = print,
    stop_event: Optional[Event] = None,
) -> dict:
    """Translate each segment's ``src_text`` into ``tgt_text`` for dubbing.

    Reuses tool_subtitle's LLM client, chunking, and the dubbing-optimized prompt
    (config/translate_prompt_dubbing.txt). The API endpoint/model come from the
    shared translation config; the key from the argument or the saved keyring.
    """
    from tool_subtitle import logic as sl

    video = Path(video_path)
    manifest = load_manifest(video)
    if not manifest:
        raise FileNotFoundError(f"Manifest not found; run earlier stages first: {manifest_path(video)}")

    cfg = sl.load_trans_config()
    target_language = target_language or manifest.get("target_language") or cfg.get("target_language") or "Chinese"

    if not api_key:
        try:
            import keyring

            api_key = (
                keyring.get_password("VR_Video_Toolbox", "deepseek_api_key")
                or keyring.get_password("VR_Mosaic_Removal", "deepseek_api_key")
            )
        except Exception:
            api_key = None
    if not api_key:
        raise RuntimeError("No translation API key provided or saved.")

    client = sl.LLMClient(
        cfg.get("api_base_url", ""), api_key, cfg.get("model_name", ""), temperature=temperature
    )

    segments = manifest.get("segments", [])
    entries = {
        int(s["id"]): {"text": s["src_text"]}
        for s in segments
        if (s.get("src_text") or "").strip()
    }
    if not entries:
        raise ValueError("No source text to translate.")

    log(f"[translate] {len(entries)} segments -> {target_language} ({cfg.get('model_name')})")
    sl.translate_entries(
        client,
        entries,
        target_language,
        int(cfg.get("tokens_per_chunk", 500000)),
        keep_original=False,
        adult_content=bool(cfg.get("adult_content", True)),
        dubbing_optimized=bool(cfg.get("dubbing_optimized", True)),
        max_retries=max_retries,
        log_callback=log,
        stop_event=stop_event,
    )

    translated = 0
    for s in segments:
        sid = int(s["id"])
        if sid in entries and entries[sid]["text"].strip():
            s["tgt_text"] = entries[sid]["text"].strip()
            translated += 1
    manifest["target_language"] = target_language
    save_manifest(video, manifest)
    write_srt(clone_dir(video) / "translated.srt", segments, "tgt_text", speaker_prefix=True)
    log(f"[translate] {translated}/{len(entries)} translated -> {manifest_path(video)}")
    log(f"[srt] translated subtitles -> {clone_dir(video) / 'translated.srt'}")
    return manifest


# --- Stage 4: OmniVoice voice-clone synthesis -> <video>.si.wav ---

def run_synthesize(
    video_path: str | Path,
    *,
    models_root: str,
    text_field: str = "tgt_text",
    language: Optional[str] = None,
    num_step: int = 32,
    guidance_scale: float = 2.0,
    batch_size: int = 6,
    loudness_mode: str = "envelope",
    envelope_alpha: float = 0.6,
    max_segments: Optional[int] = None,
    log: LogCallback = print,
    stop_event: Optional[Event] = None,
    model_holder: Optional[list] = None,
) -> str:
    """Synthesize the cloned dub and write ``<video>.si.wav``.

    ``language`` defaults to the manifest's target language when synthesizing
    ``tgt_text`` and the source language when synthesizing ``src_text``.
    """
    from tool_clonevoice import omnivoice_backend as ov

    video = Path(video_path)
    manifest = load_manifest(video)
    if not manifest:
        raise FileNotFoundError(f"Manifest not found; run earlier stages first: {manifest_path(video)}")
    if not ov.check_model(models_root):
        raise FileNotFoundError(f"OmniVoice model files are missing: {ov.model_dir(models_root)}")

    if language is None:
        language = manifest.get("target_language") if text_field == "tgt_text" else manifest.get("language")

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    model = ov.load_model(models_root, device, log)
    if model_holder is not None:
        model_holder.append(model)

    out = ov.synthesize(
        model, manifest, str(video), clone_dir(video),
        text_field=text_field, language=language, models_root=models_root,
        num_step=num_step, guidance_scale=guidance_scale, batch_size=batch_size,
        loudness_mode=loudness_mode, envelope_alpha=envelope_alpha,
        max_segments=max_segments, log=log, stop_event=stop_event,
    )
    _release_cuda_cache()
    return out


# --- Full pipeline: transcribe+diarize -> references -> translate -> synthesize ---

def run_full(
    video_path: str | Path,
    *,
    model_key: str = "large-v3",
    language: Optional[str] = None,
    diarize_backend: str = "auto",
    num_speakers: Optional[int] = None,
    target_language: str = "Chinese",
    models_root: str,
    keep_intermediate: bool = True,
    skip_existing: bool = True,
    num_step: int = 32,
    guidance_scale: float = 2.0,
    loudness_mode: str = "envelope",
    envelope_alpha: float = 0.6,
    log: LogCallback = print,
    stop_event: Optional[Event] = None,
    model_holder: Optional[list] = None,
) -> str:
    """Run the whole voice-clone-translation pipeline, returning the .si.wav path."""
    from tool_si import logic as si

    video = Path(video_path)
    out_path = si.default_si_audio_path(str(video))
    if skip_existing and Path(out_path).exists():
        log(f"[full] output exists, skipping: {out_path}")
        return str(out_path)

    log("=== [1/4] Transcription + diarization ===")
    run_transcribe_diarize(
        video, model_key=model_key, language=language, diarize_backend=diarize_backend,
        num_speakers=num_speakers, target_language=target_language, models_root=models_root,
        keep_intermediate=keep_intermediate, log=log, stop_event=stop_event, model_holder=model_holder,
    )
    # ASR/diarization models can occupy several GB. Release them before
    # OmniVoice loads, otherwise batch runs may carry both stages in VRAM.
    release_model_holder(model_holder)

    log("=== [2/4] Reference extraction ===")
    run_extract_references(video, models_root=models_root, log=log, stop_event=stop_event)

    log("=== [3/4] AI translation ===")
    run_translate(video, target_language=target_language, log=log, stop_event=stop_event)

    log("=== [4/4] Voice-clone synthesis ===")
    out = run_synthesize(
        video, models_root=models_root, text_field="tgt_text", language=target_language,
        num_step=num_step, guidance_scale=guidance_scale,
        loudness_mode=loudness_mode, envelope_alpha=envelope_alpha,
        log=log, stop_event=stop_event,
        model_holder=model_holder,
    )
    log(f"=== Done -> {out} ===")
    return out
