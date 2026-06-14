"""OmniVoice voice-clone synthesis for tool_clonevoice (P4).

Loads OmniVoice once, builds a reusable VoiceClonePrompt per speaker from the
reference clips chosen in P2, then synthesizes each segment's text in that
speaker's cloned voice with a hard per-line duration (so the dub lands in the
original time slot). Generated clips are mixed onto a single timeline and
written to ``<video>.si.wav`` (compatible with the tool_si remix tab).

Downstream timeline/wav helpers are reused from ``tool_si.logic``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

LogCallback = Callable[[str], None]

MODEL_DIR_NAME = "OmniVoice"
DEFAULT_NUM_STEP = 32
DEFAULT_GUIDANCE = 2.0
DEFAULT_BATCH_SIZE = 6


def model_dir(models_root: str) -> Path:
    return Path(models_root) / MODEL_DIR_NAME


def check_model(models_root: str) -> bool:
    d = model_dir(models_root)
    return (d / "config.json").exists() and (d / "model.safetensors").exists()


def load_model(models_root: str, device: str, log: LogCallback = print):
    import torch
    from omnivoice.models.omnivoice import OmniVoice

    path = str(model_dir(models_root))
    dtype = torch.float16 if device == "cuda" else torch.float32
    log(f"[synth] loading OmniVoice from {path} ({device}/{dtype})")
    model = OmniVoice.from_pretrained(path, device_map=device, dtype=dtype)
    return model


# Generic, clear, ~5 s, phonetically rich target-language sentences. For these
# target languages we build a SAME-LANGUAGE working reference (see below); other
# languages fall back to the cross-lingual video reference.
_GENERIC_REF_TEXTS = {
    "chinese": "你好，很高兴认识你。今天天气真不错，我们慢慢聊，有什么想法都可以告诉我，希望你过得开心。",
    "english": "Hello, it is really nice to meet you. The weather is lovely today, so let us talk slowly, "
               "and please feel free to tell me anything that is on your mind.",
}

# Fixed target duration (seconds) for the work_ref generic line, sized to the
# text at a natural pace. Without an explicit duration, OmniVoice estimates it
# from the (cross-lingual, often mismatched) ref_text/ref_audio ratio, which
# makes the working reference wildly slow for one speaker and fast for another.
# Pinning it keeps every speaker's work_ref at the same sane speaking rate.
_GENERIC_REF_DURATION = {
    "chinese": 9.5,
    "english": 12.0,
}


def _rms(clip: np.ndarray) -> float:
    return float(np.sqrt(np.mean(clip * clip))) if clip.size else 0.0


WORK_REF_TAKES = 3  # work_ref candidates per speaker; best picked by ECAPA similarity


def _read_wav_mono_f32(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    return wav.mean(axis=1), int(sr)


def _resample_f32(wav: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return wav
    import torch
    import torchaudio

    return torchaudio.functional.resample(
        torch.from_numpy(wav.astype(np.float32)), sr_from, sr_to
    ).numpy()


def _load_speaker_sim_model(models_root: str, device: str, log: LogCallback):
    """ECAPA-WavLM speaker-similarity model (OmniVoice's own eval model)."""
    try:
        from tool_clonevoice import diarize as diar

        return diar._load_ecapa_model(models_root, device, log)
    except Exception as exc:
        log(f"[synth] ECAPA similarity model unavailable ({exc}); "
            "work_ref takes will be picked by loudness instead.")
        return None


def _ecapa_embed(ecapa, wav: np.ndarray, sr: int, device: str):
    import torch
    import torch.nn.functional as F

    wav16 = _resample_f32(wav, sr, 16000)
    if wav16.size < 16000:
        wav16 = np.pad(wav16, (0, 16000 - wav16.size))
    with torch.no_grad():
        emb = ecapa([torch.from_numpy(wav16.astype(np.float32)).to(device)])
        emb = F.normalize(emb, dim=-1)
    return emb[0].detach().cpu()


def _ecapa_cosine(emb_a, emb_b) -> float:
    import torch

    return float(torch.dot(emb_a, emb_b))


def _fill_missing_ref_texts(manifest: dict, clone_dir: Path, models_root: str,
                            device: str, log: LogCallback) -> None:
    """Transcribe ref audios whose ref_text is empty (turn-only refs).

    An audio prompt without its transcript is out-of-distribution for OmniVoice
    (training always pairs prompt audio with its text), and silently degrades
    cloning to a near-random generic voice. Fill the gap with the local
    faster-whisper model instead of letting OmniVoice download Whisper Turbo.
    """
    missing = [
        (spk, info) for spk, info in manifest.get("speakers", {}).items()
        if not (info.get("ref_text") or "").strip()
        and (clone_dir / (info.get("ref_audio") or "")).is_file()
        and (info.get("ref_audio") or "")
    ]
    if not missing:
        return
    # CPU on purpose: only a few short clips, and OmniVoice already fills the
    # GPU — loading another large model on CUDA here stalls generation.
    asr = _load_trim_asr(models_root, "cpu", log)
    if asr is None:
        log(f"[synth] WARNING: {len(missing)} ref(s) have no ref_text and local ASR "
            "is unavailable; cloning quality will degrade badly.")
        return
    src_lang = (manifest.get("language") or "").strip() or None
    for spk, info in missing:
        try:
            segs, _info = asr.transcribe(
                str(clone_dir / info["ref_audio"]), beam_size=1, language=src_lang
            )
            text = "".join(s.text for s in segs).strip()
        except Exception as exc:
            log(f"[synth] {spk}: ref_text ASR failed ({exc}).")
            continue
        if text:
            info["ref_text"] = text
            log(f"[synth] {spk}: ref_text was empty, transcribed -> {text[:40]}")
        else:
            log(f"[synth] {spk}: WARNING ref_text empty and ASR heard nothing in "
                f"{info['ref_audio']}; cloning will be unreliable.")
    del asr


def _build_speaker_prompts(model, manifest: dict, clone_dir: Path, language, sr,
                           num_step, guidance_scale, models_root: str,
                           log: LogCallback):
    """One reusable voice-clone prompt per speaker.

    OmniVoice clones reliably (down to very short lines) only when the reference
    is in the SAME language as the target. So for Chinese/English targets we first
    synthesize a clean same-language "working reference" — clone the speaker's
    source-language video reference saying a generic sentence — then clone from
    THAT. For other target languages we fall back to the cross-lingual video
    reference directly (less reliable for short lines).

    The working reference is generated several times and the take whose ECAPA
    speaker embedding is closest to the ORIGINAL video reference wins, so the
    second cloning hop drifts as little as possible from the real voice.
    """
    import soundfile as sf

    device = "cuda" if str(model.device).startswith("cuda") else "cpu"
    # Empty ref_text + ref audio = untranscribed prompt: out-of-distribution for
    # OmniVoice and the root cause of "both speakers sound the same generic voice".
    _fill_missing_ref_texts(manifest, clone_dir, models_root, device, log)

    generic = _GENERIC_REF_TEXTS.get((language or "").strip().lower())
    work_dur = _GENERIC_REF_DURATION.get((language or "").strip().lower(), 9.5)
    prompts = {}

    # VRAM discipline: OmniVoice (fp16) nearly fills the GPU during generation,
    # so the ECAPA scorer must NOT be resident at the same time. Phase 1
    # generates all work_ref takes for all speakers; phase 2 loads ECAPA once,
    # scores everything, and frees it before any further generation.
    takes_by_spk: dict = {}  # spk -> (ref_audio path, [takes])
    for spk, info in manifest.get("speakers", {}).items():
        ref_rel = info.get("ref_audio") or ""
        ref_audio = clone_dir / ref_rel
        if not ref_rel or not ref_audio.is_file():
            log(f"[synth] {spk}: missing ref_audio, skipped.")
            continue
        # Do not coerce empty ref_text to None. OmniVoice auto-transcribes when
        # ref_text is None, which downloads openai/whisper-large-v3-turbo.
        ref_text = (info.get("ref_text") or "").strip()
        try:
            if generic:
                takes = []
                for take in range(1, WORK_REF_TAKES + 1):
                    cand = _normalize_peak(
                        np.asarray(model.generate(
                            text=generic, ref_audio=str(ref_audio), ref_text=ref_text,
                            language=language, duration=work_dur,
                            num_step=num_step, guidance_scale=guidance_scale,
                        )[0], dtype=np.float32).reshape(-1),
                        0.85,
                    )
                    takes.append(cand)
                    log(f"[synth] {spk}: work_ref take {take}/{WORK_REF_TAKES} generated")
                takes_by_spk[spk] = (ref_audio, takes)
            else:
                prompts[spk] = model.create_voice_clone_prompt(str(ref_audio), ref_text, preprocess_prompt=True)
                log(f"[synth] {spk}: cross-lingual video ref {ref_audio.name} (no generic ref for {language})")
        except Exception as exc:
            log(f"[synth] {spk}: failed to build prompt ({exc}).")
        _release_cuda_cache()

    if not takes_by_spk:
        return prompts

    ecapa = _load_speaker_sim_model(models_root, device, log)
    best_by_spk = {}
    try:
        for spk, (ref_audio, takes) in takes_by_spk.items():
            audible = [t for t in takes if _rms(t) > 0.02]
            wav = None
            best_sim = None
            if ecapa is not None and audible:
                ref_wav, ref_sr = _read_wav_mono_f32(str(ref_audio))
                ref_emb = _ecapa_embed(ecapa, ref_wav, ref_sr, device)
                sims = []
                for cand in audible:
                    sim = _ecapa_cosine(ref_emb, _ecapa_embed(ecapa, cand, sr, device))
                    sims.append(sim)
                    if best_sim is None or sim > best_sim:
                        best_sim, wav = sim, cand
                log(f"[synth] {spk}: work_ref sims " +
                    ", ".join(f"{s:.3f}" for s in sims) + f" -> best {best_sim:.3f}")
            else:
                for cand in (audible or takes):  # loudest fallback
                    if wav is None or _rms(cand) > _rms(wav):
                        wav = cand
            best_by_spk[spk] = (wav, best_sim)
    finally:
        if ecapa is not None:
            del ecapa
            _release_cuda_cache()

    for spk, (wav, best_sim) in best_by_spk.items():
        try:
            work_path = clone_dir / f"work_ref_{spk}.wav"
            sf.write(str(work_path), wav, sr)
            prompts[spk] = model.create_voice_clone_prompt(str(work_path), generic, preprocess_prompt=True)
            sim_note = f", sim={best_sim:.3f}" if best_sim is not None else ""
            log(f"[synth] {spk}: same-language working ref ({language}) -> {work_path.name}{sim_note}")
        except Exception as exc:
            log(f"[synth] {spk}: failed to build prompt ({exc}).")
    return prompts


_CJK_RE = re.compile(r"[㐀-鿿豈-﫿぀-ヿ가-힯]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]+")


def _clone_char_count(text: str) -> int:
    """Rough 'amount of speech' measure: CJK characters + Latin words."""
    return len(_CJK_RE.findall(text or "")) + len(_LATIN_WORD_RE.findall(text or ""))


def _merge_units(segments: List[dict], text_field: str, merge_gap: float) -> List[dict]:
    """Merge consecutive same-speaker segments (small time gaps) into synthesis
    units, so each unit has enough text for reliable cross-lingual cloning."""
    ordered = sorted(
        (s for s in segments if (s.get(text_field) or "").strip()),
        key=lambda s: float(s["start"]),
    )
    units: List[dict] = []
    cur: Optional[dict] = None
    for s in ordered:
        text = (s.get(text_field) or "").strip()
        start, end, spk = float(s["start"]), float(s["end"]), s["speaker"]
        if cur and spk == cur["speaker"] and (start - cur["end"]) <= merge_gap:
            cur["text"] = (cur["text"] + " " + text).strip()
            cur["end"] = end
        else:
            if cur:
                units.append(cur)
            cur = {"speaker": spk, "start": start, "end": end, "text": text}
    if cur:
        units.append(cur)
    return units


# --- short-text padding (carrier) so short lines still clone in the speaker's voice ---

# A neutral, punctuation-free target-language sentence appended after a short
# line so OmniVoice has enough text to clone reliably; trimmed off afterwards.
# Each: (carrier_text, verify_probes, boundary_marker). The marker is a
# distinctive token inside the carrier (unlikely to appear in a short line);
# the carrier start = marker position minus the marker's offset in the carrier.
_CARRIERS = {
    "chinese": ("我们改天再慢慢聊聊这件事情吧", ("改天", "这件事", "聊聊", "慢慢"), "改天"),
    "english": ("let us talk about this matter slowly some other day", ("other day", "matter", "slowly"), "matter"),
    "japanese": ("この件についてまた今度ゆっくり話しましょう", ("今度", "ゆっくり", "話し"), "今度"),
}
_ASR_LANG = {"chinese": "zh", "english": "en", "japanese": "ja"}


def _carrier_for(language: Optional[str]):
    key = (language or "chinese").strip().lower()
    return _CARRIERS.get(key, _CARRIERS["chinese"])


def _asr_lang_for(language: Optional[str]) -> str:
    return _ASR_LANG.get((language or "chinese").strip().lower(), None)  # type: ignore[return-value]


def _normalize_peak(clip: np.ndarray, target_peak: float = 0.5, max_gain: float = 25.0) -> np.ndarray:
    """Scale a clip so its peak hits ``target_peak`` (bounded gain), for
    consistent loudness; OmniVoice output is otherwise very quiet."""
    if clip.size == 0:
        return clip
    peak = float(np.max(np.abs(clip)))
    if peak < 1e-6:
        return clip
    gain = min(max_gain, target_peak / peak)
    return np.clip(clip * gain, -1.0, 1.0).astype(np.float32)


def _rms_db(clip: np.ndarray, floor_db: float = -55.0) -> float:
    if clip.size == 0:
        return floor_db
    value = _rms(clip)
    if value <= 1e-8:
        return floor_db
    return max(floor_db, 20.0 * np.log10(value))


def _slice_audio(audio: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    s = max(0, int(round(float(start) * sr)))
    e = min(audio.size, int(round(float(end) * sr)))
    if e <= s:
        return np.zeros(0, dtype=np.float32)
    return audio[s:e].astype(np.float32, copy=False)


def _match_sentence_loudness(
    clip: np.ndarray,
    source_clip: Optional[np.ndarray],
    *,
    target_peak_limit: float = 0.88,
    min_gain: float = 0.35,
    max_gain: float = 2.8,
) -> tuple[np.ndarray, float, float, float]:
    """Match a synthesized sentence to the original sentence's RMS loudness.

    Returns ``(clip, gain, source_db, synth_db)``. The target is sentence-level
    dynamics only; phrase/word envelope following is intentionally not applied.
    """
    if clip.size == 0:
        return clip, 1.0, -55.0, -55.0
    synth_db = _rms_db(clip)
    source_db = _rms_db(source_clip) if source_clip is not None and source_clip.size else synth_db
    gain = 10.0 ** ((source_db - synth_db) / 20.0)
    gain = float(np.clip(gain, min_gain, max_gain))
    peak = float(np.max(np.abs(clip)))
    if peak > 1e-6:
        gain = min(gain, target_peak_limit / peak)
    return np.clip(clip * gain, -1.0, 1.0).astype(np.float32), gain, source_db, synth_db


def _short_time_rms_envelope(
    audio: np.ndarray, sr: int, win_ms: float = 80.0, hop_ms: float = 20.0
) -> tuple[np.ndarray, int]:
    """Frame-wise RMS envelope. Returns ``(envelope, hop_samples)``."""
    n = audio.size
    if n == 0:
        return np.zeros(0, dtype=np.float32), 1
    win = max(1, int(round(win_ms * sr / 1000.0)))
    hop = max(1, int(round(hop_ms * sr / 1000.0)))
    if n <= win:
        return np.array([_rms(audio)], dtype=np.float32), hop
    n_frames = 1 + (n - win) // hop
    env = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        s = i * hop
        env[i] = _rms(audio[s : s + win])
    return env, hop


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or x.size <= 2:
        return x
    k = min(k, x.size)
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(x, kernel, mode="same").astype(np.float32)


def _follow_energy_envelope(
    clip: np.ndarray,
    source_clip: Optional[np.ndarray],
    clip_sr: int,
    source_sr: int,
    alpha: float,
    *,
    win_ms: float = 80.0,
    hop_ms: float = 20.0,
    max_db: float = 6.0,
    floor_pct: float = 40.0,
) -> np.ndarray:
    """Make ``clip`` follow the coarse loudness contour of ``source_clip``.

    Position-normalized (NOT time-aligned): source and synth speak different
    languages of different lengths, so we map the source sentence's energy
    *shape* onto the synth sentence by normalized position, heavily smooth it,
    clamp to +/- ``max_db``, and blend it in with strength ``alpha``. Absolute
    level is left to :func:`_match_sentence_loudness`. The source shape is
    floored (``floor_pct`` percentile) so internal pauses don't punch holes in
    the dub.
    """
    if clip.size == 0 or source_clip is None or source_clip.size == 0 or alpha <= 0:
        return clip
    src_env, _ = _short_time_rms_envelope(source_clip, source_sr, win_ms, hop_ms)
    syn_env, hop = _short_time_rms_envelope(clip, clip_sr, win_ms, hop_ms)
    if src_env.size < 2 or syn_env.size < 2:
        return clip
    # shape only: normalize each envelope by its own mean
    src_env = src_env / (src_env.mean() + 1e-8)
    syn_env = syn_env / (syn_env.mean() + 1e-8)
    src_env = np.maximum(src_env, np.percentile(src_env, floor_pct))
    # map the source shape onto the synth timeline by normalized position
    src_on_syn = np.interp(
        np.linspace(0.0, 1.0, syn_env.size),
        np.linspace(0.0, 1.0, src_env.size),
        src_env,
    ).astype(np.float32)
    ratio = _smooth(src_on_syn / (syn_env + 1e-8), 5)
    lim = 10.0 ** (max_db / 20.0)
    ratio = np.clip(ratio, 1.0 / lim, lim)
    gain_frames = (1.0 - alpha) + alpha * ratio
    win = max(1, int(round(win_ms * clip_sr / 1000.0)))
    centers = np.arange(syn_env.size, dtype=np.float32) * hop + win / 2.0
    sample_idx = np.arange(clip.size, dtype=np.float32)
    gain_samples = np.interp(sample_idx, centers, gain_frames).astype(np.float32)
    return (clip * gain_samples).astype(np.float32)


def _carrier_trim_cut(words, carrier: str, marker: str) -> Optional[float]:
    """Boundary time between the target line and the trailing carrier.

    Locates the carrier's distinctive ``marker`` in the recognized characters and
    backtracks by the marker's offset within the carrier to the carrier start —
    robust to ASR length errors and to the model not pausing before the carrier.
    """
    clean = lambda s: re.sub(r"[^\w]", "", s or "")
    offset = clean(carrier).find(clean(marker))
    if offset < 0:
        offset = 0
    chars = []  # (char, start_time)
    for w, ws, we in words:
        cw = clean(w)
        if not cw:
            continue
        step = max(1e-3, (float(we) - float(ws))) / len(cw)
        for k, ch in enumerate(cw):
            chars.append((ch, float(ws) + k * step))
    text = "".join(c[0] for c in chars)
    pos = text.find(clean(marker))
    if pos < 0:
        return None
    cut_char = pos - offset
    if cut_char <= 0:
        return None
    return chars[cut_char][1]


def _synth_short_padded(
    model,
    text: str,
    prompt,
    language: Optional[str],
    asr_model,
    sr: int,
    num_step: int,
    guidance_scale: float,
    log: LogCallback,
    max_retries: int = 5,
) -> np.ndarray:
    """Synthesize a short line in the speaker's cloned voice via carrier padding.

    Appends a carrier sentence so the generation is long enough to clone
    correctly, verifies the carrier was actually produced (else retries), then
    trims the carrier off at the longest pause.
    """
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

    clean = lambda s: re.sub(r"[^\w]", "", s or "")
    carrier, _probes, marker = _carrier_for(language)
    cmk = clean(marker)
    carrier_offset = max(0, clean(carrier).find(cmk))
    target_clean = clean(text)
    padded = text.strip().rstrip("。.!?！？") + "。" + carrier
    cfg = OmniVoiceGenerationConfig(postprocess_output=False)
    asr_lang = _asr_lang_for(language)

    fallback = None
    for attempt in range(1, max_retries + 1):
        au = np.asarray(
            model.generate(
                text=padded, voice_clone_prompt=prompt, language=language,
                generation_config=cfg, num_step=num_step, guidance_scale=guidance_scale,
            )[0],
            dtype=np.float32,
        ).reshape(-1)
        au = _normalize_peak(au, 0.6)  # OmniVoice output is quiet; ASR needs level
        if fallback is None:
            fallback = au
        if asr_model is None:
            return au
        segs, _info = asr_model.transcribe(au, word_timestamps=True, beam_size=1, language=asr_lang)
        chars = []  # (char, start_time)
        for s in segs:
            for w in (s.words or []):
                cw = clean(w.word)
                step = max(1e-3, (float(w.end) - float(w.start))) / max(1, len(cw))
                for k, ch in enumerate(cw):
                    chars.append((ch, float(w.start) + k * step))
        ctext = "".join(c[0] for c in chars)
        mpos = ctext.find(cmk)
        if mpos < 0:
            log(f"[synth]   short retry {attempt}/{max_retries}: carrier not produced")
            continue
        tend = max(0, mpos - carrier_offset)  # target ends here, carrier begins
        pre = ctext[:tend]
        overlap = (len(set(pre) & set(target_clean)) / len(set(target_clean))) if target_clean else 1.0
        if overlap < 0.45:
            log(f"[synth]   short retry {attempt}/{max_retries}: wrong content '{pre[:12]}' vs '{text[:8]}'")
            continue
        if tend <= 0 or tend >= len(chars):
            continue
        return au[: int(chars[tend][1] * sr)]
    log(f"[synth]   short '{text[:10]}': no clean take after {max_retries} tries; best effort")
    return fallback if fallback is not None else np.zeros(1, dtype=np.float32)


def _load_trim_asr(models_root: str, device: str, log: LogCallback):
    try:
        from faster_whisper import WhisperModel

        path = str(Path(models_root) / "faster-whisper-large-v3")
        compute = "float16" if device == "cuda" else "int8"
        log("[synth] loading local faster-whisper (ref transcription / boundary detection)")
        return WhisperModel(path, device=device, compute_type=compute)
    except Exception as exc:
        log(f"[synth] trim ASR unavailable ({exc}); short lines will not be trimmed precisely.")
        return None


def synthesize(
    model,
    manifest: dict,
    video_path: str,
    clone_dir: Path,
    *,
    text_field: str = "tgt_text",
    language: Optional[str] = None,
    models_root: str = "models",
    num_step: int = DEFAULT_NUM_STEP,
    guidance_scale: float = DEFAULT_GUIDANCE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    duration_mode: str = "natural",
    min_clone_chars: int = 7,
    merge_gap: float = 0.35,
    loudness_mode: str = "envelope",
    envelope_alpha: float = 0.6,
    max_segments: Optional[int] = None,
    log: LogCallback = print,
    stop_event=None,
) -> str:
    """Synthesize all segments and write ``<video>.si.wav``. Returns its path.

    ``duration_mode``:
      - ``"natural"`` (default): let OmniVoice speak the translation at its own
        natural pace; place each clip at the segment start with no time-stretch.
        Best intelligibility; total length may drift from the source.
      - ``"fit"``: pass the source-segment duration to OmniVoice so it paces the
        line to roughly that length (single, model-side fit; no post-stretch).
    """
    from tool_si import logic as si

    sr = int(getattr(model, "sampling_rate", None) or 24000)
    needs_source = loudness_mode in ("sentence", "envelope")
    source_audio = None
    source_sr = 16000
    source_path = clone_dir / "audio16k.wav"
    if needs_source and source_path.is_file():
        try:
            source_audio, source_sr = _read_wav_mono_f32(str(source_path))
        except Exception as exc:
            log(f"[synth] source loudness reference unavailable ({exc})")
            source_audio = None
    elif needs_source:
        log(f"[synth] source loudness reference unavailable: {source_path} not found")

    segments = manifest.get("segments", [])
    if max_segments is not None:
        segments = segments[:max_segments]

    prompts = _build_speaker_prompts(
        model, manifest, clone_dir, language, sr, num_step, guidance_scale,
        models_root, log
    )

    units = _merge_units(segments, text_field, merge_gap)
    log(f"[synth] {len(segments)} segments -> {len(units)} synthesis units")

    total_dur = max((float(s["end"]) for s in segments), default=0.0)
    # Headroom so natural-paced clips that run past their slot are not clipped.
    timeline = np.zeros(max(1, int(round((total_dur + 30.0) * sr))), dtype=np.float32)
    max_end_sample = 0

    n_clone = n_noref = 0
    for idx, unit in enumerate(units, 1):
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("Stopped by user.")
        text = unit["text"]
        prompt = prompts.get(unit["speaker"])
        duration = float(unit["end"] - unit["start"]) if duration_mode == "fit" else None

        # Same-language working ref clones reliably even for short lines. Short
        # lines are still the riskiest; retry a few times and keep the least-silent
        # take (catches the occasional empty/garbled generation) — no carrier.
        tries = 3 if (prompt is not None and _clone_char_count(text) < min_clone_chars) else 1
        clip = None
        for _ in range(tries):
            cand = np.asarray(
                model.generate(text=text, voice_clone_prompt=prompt, language=language,
                               duration=duration, num_step=num_step, guidance_scale=guidance_scale)[0],
                dtype=np.float32,
            ).reshape(-1)
            if clip is None or _rms(cand) > _rms(clip):
                clip = cand
            if _rms(cand) > 0.02:
                break

        if loudness_mode == "flat":
            clip = _normalize_peak(clip, 0.6)
            loudness_note = ""
        else:
            source_clip = (
                _slice_audio(source_audio, source_sr, unit["start"], unit["end"])
                if source_audio is not None else None
            )
            if loudness_mode == "envelope":
                clip = _follow_energy_envelope(clip, source_clip, sr, source_sr, envelope_alpha)
            clip, gain, src_db, synth_db = _match_sentence_loudness(clip, source_clip)
            loudness_note = f" gain={gain:.2f} src={src_db:.1f}dB gen={synth_db:.1f}dB"
            if loudness_mode == "envelope":
                loudness_note += f" env(a={envelope_alpha:g})"
        n_noref += int(prompt is None)
        n_clone += int(prompt is not None)
        start_sample = max(0, int(round(unit["start"] * sr)))
        si._mix_timeline_segment(timeline, start_sample, clip)
        max_end_sample = max(max_end_sample, start_sample + clip.size)
        log(
            f"[synth] {idx}/{len(units)} {unit['speaker']} "
            f"{unit['start']:.1f}-{unit['end']:.1f}s{loudness_note}"
        )
        _release_cuda_cache()

    timeline = timeline[: max(1, min(timeline.size, max_end_sample))]
    out_path = si.default_si_audio_path(video_path)
    si.write_wav_mono(out_path, timeline, sr)
    log(f"[synth] wrote {out_path} ({timeline.size / sr:.1f}s; {n_clone} cloned, {n_noref} no-ref)")
    return str(out_path)


def _release_cuda_cache() -> None:
    try:
        import gc

        import torch

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
    except Exception:
        pass
