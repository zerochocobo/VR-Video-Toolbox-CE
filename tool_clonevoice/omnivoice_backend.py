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


def _duck_spans_for_segments(segments: List[dict], text_field: str,
                             skipped_speakers: set[str] | None = None) -> list[dict]:
    """Return ducking spans only for lines that will actually be synthesized.

    AI source correction and manual proofreading represent a deleted line by
    clearing its active text field. Such a line must not suppress the original
    voice in the generated ``.si.duck.wav``.
    """
    skipped = skipped_speakers or set()
    return [
        {"start": float(segment["start"]), "end": float(segment["end"])}
        for segment in segments
        if (segment.get(text_field) or "").strip()
        and str(segment.get("speaker") or "") not in skipped
    ]


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


def resolve_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _stable_seed(*parts: object) -> int:
    """Deterministic 31-bit seed from arbitrary parts (stable across processes)."""
    import hashlib

    digest = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


def _seed_generation(seed: int) -> None:
    """Seed torch/numpy RNG so OmniVoice generation is reproducible.

    OmniVoice samples token positions with Gumbel noise (position_temperature>0),
    which reads the global torch RNG. Without a fixed seed, the same reference clip
    produces a different take every run, so a good candidate can vanish next time.
    Seeding per candidate keeps takes diverse but reproducible.
    """
    seed = int(seed) & 0x7FFFFFFF
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    try:
        np.random.seed(seed)
    except Exception:
        pass


# Generic, clear, ~7-8 s target-language sentences. For these
# target languages we build a SAME-LANGUAGE working reference (see below); other
# languages fall back to the cross-lingual video reference.
_GENERIC_REF_TEXTS = {
    "chinese": (
        "你好，很高兴认识你。今天天气很舒服，我们可以聊聊最近的计划、"
        "喜欢的音乐，也分享一些温暖有趣的想法。"
    ),
    "english": (
        "Hello, it is nice to meet you today. The weather is calm, "
        "so we can speak clearly, share a few ideas, and talk about what matters."
    ),
    "korean": (
        "안녕하세요, 오늘 만나서 반갑습니다. 날씨가 편안해서 천천히 이야기하고, "
        "최근 계획과 좋아하는 음악, 따뜻한 생각을 나눌 수 있습니다."
    ),
    "thai": (
        "สวัสดีครับ วันนี้ยินดีที่ได้พบกัน อากาศสงบและน่าพอใจ "
        "เราคุยกันช้าๆ ได้ชัดเจน แบ่งปันความคิดง่ายๆ และเรื่องที่อยู่ในใจ"
    ),
    "german": (
        "Hallo, es freut mich, dich heute kennenzulernen. Das Wetter ist ruhig, "
        "wir können klar sprechen, einfache Gedanken teilen und über Wichtiges reden."
    ),
    "french": (
        "Bonjour, je suis ravi de vous rencontrer aujourd'hui. Le temps est calme, "
        "parlons clairement, partageons des idées simples et évoquons ce qui compte."
    ),
    "spanish": (
        "Hola, me alegra conocerte hoy. El clima está tranquilo, "
        "podemos hablar con claridad, compartir ideas sencillas y conversar sobre lo importante."
    ),
    "portuguese": (
        "Olá, é bom conhecer você hoje. O tempo está calmo, "
        "podemos falar com clareza, compartilhar ideias simples e conversar sobre o que importa."
    ),
    "italian": (
        "Ciao, sono felice di incontrarti oggi. Il tempo è calmo, "
        "possiamo parlare con chiarezza, condividere idee semplici e discutere ciò che conta."
    ),
    "russian": (
        "Здравствуйте, приятно встретиться с вами сегодня. Погода спокойная, "
        "мы можем говорить ясно, делиться простыми мыслями и обсуждать важное."
    ),
}

# Fixed target duration (seconds) for the work_ref generic line, sized to the
# text at a natural pace. Without an explicit duration, OmniVoice estimates it
# from the (cross-lingual, often mismatched) ref_text/ref_audio ratio, which
# makes the working reference wildly slow for one speaker and fast for another.
# Pinning it keeps every speaker's work_ref at the same sane speaking rate.
_GENERIC_REF_DURATION = {
    "chinese": 8.0,
    "english": 8.0,
    "korean": 8.0,
    "thai": 8.0,
    "german": 8.0,
    "french": 8.0,
    "spanish": 8.0,
    "portuguese": 8.0,
    "italian": 8.0,
    "russian": 8.0,
}


_LANG_ALIASES = {
    "zh": "chinese",
    "zho": "chinese",
    "cn": "chinese",
    "chinese": "chinese",
    "中文": "chinese",
    "中国語": "chinese",
    "en": "english",
    "eng": "english",
    "english": "english",
    "英语": "english",
    "英語": "english",
    "ja": "japanese",
    "jpn": "japanese",
    "japanese": "japanese",
    "日语": "japanese",
    "日本語": "japanese",
    "ko": "korean",
    "kor": "korean",
    "korean": "korean",
    "韩语": "korean",
    "韓国語": "korean",
    "th": "thai",
    "tha": "thai",
    "thai": "thai",
    "泰语": "thai",
    "タイ語": "thai",
    "de": "german",
    "deu": "german",
    "ger": "german",
    "german": "german",
    "德语": "german",
    "ドイツ語": "german",
    "fr": "french",
    "fra": "french",
    "fre": "french",
    "french": "french",
    "法语": "french",
    "フランス語": "french",
    "es": "spanish",
    "spa": "spanish",
    "spanish": "spanish",
    "西班牙语": "spanish",
    "スペイン語": "spanish",
    "pt": "portuguese",
    "por": "portuguese",
    "portuguese": "portuguese",
    "葡萄牙语": "portuguese",
    "ポルトガル語": "portuguese",
    "it": "italian",
    "ita": "italian",
    "italian": "italian",
    "意大利语": "italian",
    "イタリア語": "italian",
    "ru": "russian",
    "rus": "russian",
    "russian": "russian",
    "俄语": "russian",
    "ロシア語": "russian",
}


def normalize_language_name(language: Optional[str]) -> str:
    key = (language or "").strip().lower().replace("_", "-")
    if key in _LANG_ALIASES:
        return _LANG_ALIASES[key]
    if "-" in key:
        primary = key.split("-", 1)[0]
        return _LANG_ALIASES.get(primary, primary)
    return key


def same_language(a: Optional[str], b: Optional[str]) -> bool:
    na = normalize_language_name(a)
    nb = normalize_language_name(b)
    return bool(na and nb and na == nb)


def _estimate_ref_duration(text: str) -> float:
    cjk_count = len(re.findall(r"[\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]", text or ""))
    word_count = len(re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", text or ""))
    visible_count = sum(1 for ch in (text or "") if not ch.isspace())
    if word_count > 0 and cjk_count < 10:
        duration = 1.2 + word_count / 2.45
    else:
        duration = 1.0 + max(cjk_count, visible_count) / 5.0
    return float(np.clip(duration, 6.0, 14.0))


def _resolve_generic_ref(language: Optional[str]) -> tuple[Optional[str], float]:
    key = normalize_language_name(language)
    generic = _GENERIC_REF_TEXTS.get(key)
    duration = _GENERIC_REF_DURATION.get(key)
    if duration is None and generic:
        duration = _estimate_ref_duration(generic)
    return generic, duration or 9.5


def generic_ref_text(language: Optional[str]) -> tuple[Optional[str], float]:
    """Return the built-in target-language fixed reference sentence and duration."""
    return _resolve_generic_ref(language)


def _rms(clip: np.ndarray) -> float:
    return float(np.sqrt(np.mean(clip * clip))) if clip.size else 0.0


WORK_REF_TAKES = 3  # work_ref candidates per speaker; best picked by ECAPA similarity
PROMPT_TAIL_FADE_S = 0.08
PROMPT_TAIL_SILENCE_S = 0.32


def _read_wav_mono_f32(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    return wav.mean(axis=1), int(sr)


def _read_pcm_wav_mono_f32(path: str | Path) -> tuple[np.ndarray, int]:
    import wave

    with wave.open(str(path), "rb") as wav_file:
        sr = int(wav_file.getframerate())
        channels = int(wav_file.getnchannels())
        sampwidth = int(wav_file.getsampwidth())
        raw = wav_file.readframes(wav_file.getnframes())
    if sampwidth != 2:
        raise ValueError(f"Only 16-bit PCM WAV fallback is supported, got sample width {sampwidth}.")
    wav = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        wav = wav.reshape(-1, channels).mean(axis=1)
    return wav, sr


def _write_pcm_wav_mono_f32(path: str | Path, wav: np.ndarray, sr: int) -> None:
    import wave

    arr = np.asarray(wav, dtype=np.float32).reshape(-1)
    pcm = (np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sr))
        wav_file.writeframes(pcm.tobytes())


def prepare_prompt_reference_audio(
    ref_audio: str | Path,
    *,
    fade_out_s: float = PROMPT_TAIL_FADE_S,
    tail_silence_s: float = PROMPT_TAIL_SILENCE_S,
    log: Optional[LogCallback] = None,
) -> str:
    """Create a prompt-only copy whose tail fades into silence.

    OmniVoice conditions generated tokens immediately after the prompt audio
    tokens. When a reference clip ends mid-breath or with no pause, the model can
    leak the reference tail into the beginning of the generated sentence. The
    original reference is still used for playback and ECAPA; this copy is only
    for prompt construction.
    """
    path = Path(ref_audio)
    if not path.is_file() or path.stem.endswith("_prompt"):
        return str(path)
    out = path.with_name(f"{path.stem}_prompt.wav")
    try:
        if out.is_file() and out.stat().st_mtime >= path.stat().st_mtime:
            return str(out)
        try:
            wav, sr = _read_wav_mono_f32(str(path))
        except Exception:
            wav, sr = _read_pcm_wav_mono_f32(path)
        if wav.size == 0:
            return str(path)
        prepared = wav.astype(np.float32, copy=True)
        fade_n = min(prepared.size, max(0, int(round(float(fade_out_s) * sr))))
        if fade_n > 1:
            prepared[-fade_n:] *= np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
        pad_n = max(0, int(round(float(tail_silence_s) * sr)))
        if pad_n:
            prepared = np.concatenate([prepared, np.zeros(pad_n, dtype=np.float32)])
        try:
            import soundfile as sf

            sf.write(str(out), prepared, sr)
        except Exception:
            _write_pcm_wav_mono_f32(out, prepared, sr)
        if log is not None:
            log(f"[synth] prompt tail separator -> {out.name} (+{tail_silence_s:.2f}s silence)")
        return str(out)
    except Exception as exc:
        if log is not None:
            log(f"[synth] prompt tail separator skipped for {path.name} ({exc})")
        return str(path)


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


def score_audio_similarity(
    audio_a: str,
    audio_b: str,
    *,
    models_root: str,
    device: Optional[str] = None,
    log: LogCallback = print,
) -> float:
    """Score two audios with OmniVoice's ECAPA-WavLM speaker similarity model."""
    if device is None:
        device = resolve_device()
    _job_results, pair_results = process_target_reference_batch(
        [],
        score_pairs=[(audio_a, audio_b)],
        models_root=models_root,
        device=device,
        log=log,
    )
    return pair_results[0]


def score_audio_similarity_pairs(
    pairs: list[tuple[str, str]],
    *,
    models_root: str,
    device: Optional[str] = None,
    log: LogCallback = print,
) -> list[float]:
    """Score audio pairs with one ECAPA model load."""
    if device is None:
        device = resolve_device()
    _job_results, pair_results = process_target_reference_batch(
        [],
        score_pairs=pairs,
        models_root=models_root,
        device=device,
        log=log,
    )
    return pair_results


def _pick_best_take_against_ref(
    takes: list[np.ndarray],
    *,
    ref_audio: Path,
    take_sr: int,
    models_root: str,
    device: str,
    log: LogCallback,
) -> tuple[np.ndarray, Optional[float]]:
    audible = [t for t in takes if _rms(t) > 0.02]
    candidates = audible or takes
    if not candidates:
        return np.zeros(1, dtype=np.float32), None

    ecapa = _load_speaker_sim_model(models_root, device, log)
    try:
        if ecapa is not None and audible:
            ref_wav, ref_sr = _read_wav_mono_f32(str(ref_audio))
            ref_emb = _ecapa_embed(ecapa, ref_wav, ref_sr, device)
            best_wav = None
            best_sim = None
            sims = []
            for cand in audible:
                sim = _ecapa_cosine(ref_emb, _ecapa_embed(ecapa, cand, take_sr, device))
                sims.append(sim)
                if best_sim is None or sim > best_sim:
                    best_sim = sim
                    best_wav = cand
            log("[synth] target-ref sims " + ", ".join(f"{s:.3f}" for s in sims) + f" -> best {best_sim:.3f}")
            return best_wav, best_sim  # type: ignore[return-value]

        best = max(candidates, key=_rms)
        return best, None
    finally:
        if ecapa is not None:
            del ecapa
        _release_cuda_cache()


def _generate_target_reference_takes_with_model(
    model,
    *,
    source_ref_audio: str,
    source_ref_text: str,
    target_language: str,
    num_step: int = DEFAULT_NUM_STEP,
    guidance_scale: float = DEFAULT_GUIDANCE,
    take_count: int = WORK_REF_TAKES,
    instruct: Optional[str] = None,
    log_label: str = "target sample",
    log: LogCallback = print,
    stop_event=None,
) -> tuple[list[np.ndarray], str, int, str]:
    """Generate target-language work_ref takes without loading ECAPA."""
    generic, duration = _resolve_generic_ref(target_language)
    if not generic:
        raise ValueError(f"No built-in target-language reference sentence for: {target_language}")
    sr = int(getattr(model, "sampling_rate", None) or 24000)
    device = "cuda" if str(getattr(model, "device", "")).startswith("cuda") else "cpu"
    ref_audio = Path(prepare_prompt_reference_audio(source_ref_audio, log=log))
    ref_text = (source_ref_text or "").strip()
    if not ref_text:
        raise ValueError("Source reference text is empty; cannot build a reliable OmniVoice prompt.")

    # Reproducible per-candidate: the same source clip yields the same takes every
    # run (still diverse across takes for ECAPA to pick from), so a good candidate
    # does not randomly disappear on the next "collect + generate" pass.
    _seed_generation(_stable_seed(source_ref_audio, target_language, instruct or "", take_count))

    takes: list[np.ndarray] = []
    for take in range(1, max(1, int(take_count)) + 1):
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("Stopped by user.")
        kwargs = {
            "text": generic,
            "ref_audio": str(ref_audio),
            "ref_text": ref_text,
            "language": target_language,
            "duration": duration,
            "num_step": num_step,
            "guidance_scale": guidance_scale,
        }
        if instruct:
            kwargs["instruct"] = instruct
        cand = _normalize_peak(
            np.asarray(model.generate(**kwargs)[0], dtype=np.float32).reshape(-1),
            0.85,
        )
        takes.append(cand)
        log(f"[synth] {log_label}: fixed sample take {take}/{max(1, int(take_count))} generated")
        _release_cuda_cache()
    return takes, generic, sr, device


def _write_best_target_reference_sample(
    takes: list[np.ndarray],
    *,
    generic: str,
    source_ref_audio: str,
    take_sr: int,
    models_root: str,
    device: str,
    output_wav: str,
    log: LogCallback,
) -> tuple[str, str, Optional[float]]:
    results, _pair_results = process_target_reference_batch(
        [{
            "takes": takes,
            "generic": generic,
            "source_ref_audio": source_ref_audio,
            "take_sr": take_sr,
            "output_wav": output_wav,
        }],
        models_root=models_root,
        device=device,
        log=log,
    )
    return results[0]


def process_target_reference_batch(
    jobs: list[dict],
    *,
    score_pairs: Optional[list[tuple[str, str]]] = None,
    models_root: str,
    device: Optional[str] = None,
    log: LogCallback = print,
) -> tuple[list[tuple[str, str, Optional[float]]], list[float]]:
    """Finalize target-language reference jobs and/or score pairs with one ECAPA load.

    ``jobs`` contain generated takes from ``_generate_target_reference_takes_with_model``.
    Each job must include ``takes``, ``generic``, ``source_ref_audio``, ``take_sr`` and
    ``output_wav``. Generated jobs fall back to loudness if ECAPA is unavailable;
    explicit ``score_pairs`` still raise because they have no useful fallback score.
    """
    import soundfile as sf

    jobs = list(jobs)
    pairs = list(score_pairs or [])
    if not jobs and not pairs:
        return [], []
    if device is None:
        device = str(jobs[0].get("device") or "") if jobs else ""
    if not device:
        device = resolve_device()

    ecapa = _load_speaker_sim_model(models_root, device, log)
    embed_cache = {}

    def _file_emb(path: str):
        if ecapa is None:
            raise RuntimeError("ECAPA speaker-similarity model is unavailable.")
        key = str(Path(path))
        if key not in embed_cache:
            wav, sr = _read_wav_mono_f32(key)
            embed_cache[key] = _ecapa_embed(ecapa, wav, sr, device)
        return embed_cache[key]

    job_results: list[tuple[str, str, Optional[float]]] = []
    pair_results: list[float] = []
    try:
        for job in jobs:
            label = job.get("label") or "target-ref"
            takes = list(job.get("takes") or [])
            audible = [t for t in takes if _rms(t) > 0.02]
            candidates = audible or takes
            if not candidates:
                candidates = [np.zeros(1, dtype=np.float32)]

            best_wav = None
            best_sim = None
            if ecapa is not None and audible:
                ref_emb = _file_emb(str(job["source_ref_audio"]))
                sims = []
                for cand in audible:
                    sim = _ecapa_cosine(ref_emb, _ecapa_embed(ecapa, cand, int(job["take_sr"]), device))
                    sims.append(sim)
                    if best_sim is None or sim > best_sim:
                        best_sim = sim
                        best_wav = cand
                log("[synth] " + str(label) + " sims " +
                    ", ".join(f"{s:.3f}" for s in sims) + f" -> best {best_sim:.3f}")
            if best_wav is None:
                best_wav = max(candidates, key=_rms)

            out = Path(job["output_wav"])
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out), best_wav, int(job["take_sr"]))
            sim_note = f" sim={best_sim:.3f}" if best_sim is not None else ""
            log(f"[synth] {label}: fixed target-language sample -> {out.name}{sim_note}")
            job_results.append((str(out), str(job["generic"]), best_sim))

        if pairs:
            if ecapa is None:
                raise RuntimeError("ECAPA speaker-similarity model is unavailable.")
            for audio_a, audio_b in pairs:
                pair_results.append(_ecapa_cosine(_file_emb(audio_a), _file_emb(audio_b)))
        return job_results, pair_results
    finally:
        if ecapa is not None:
            del ecapa
        _release_cuda_cache()


def generate_target_reference_sample_with_model(
    model,
    *,
    models_root: str,
    source_ref_audio: str,
    source_ref_text: str,
    target_language: str,
    output_wav: str,
    num_step: int = DEFAULT_NUM_STEP,
    guidance_scale: float = DEFAULT_GUIDANCE,
    take_count: int = WORK_REF_TAKES,
    instruct: Optional[str] = None,
    log: LogCallback = print,
    stop_event=None,
) -> tuple[str, str, Optional[float]]:
    """Generate one frozen target-language work_ref from a source-language ref.

    The text is one built-in fixed sentence; internally several takes are sampled
    and ECAPA-pick the one closest to the source reference.
    """
    takes, generic, sr, device = _generate_target_reference_takes_with_model(
        model,
        source_ref_audio=source_ref_audio,
        source_ref_text=source_ref_text,
        target_language=target_language,
        num_step=num_step,
        guidance_scale=guidance_scale,
        take_count=take_count,
        instruct=instruct,
        log=log,
        stop_event=stop_event,
    )
    return _write_best_target_reference_sample(
        takes,
        generic=generic,
        source_ref_audio=source_ref_audio,
        take_sr=sr,
        models_root=models_root,
        device=device,
        output_wav=output_wav,
        log=log,
    )


def generate_target_reference_sample(
    *,
    models_root: str,
    source_ref_audio: str,
    source_ref_text: str,
    target_language: str,
    output_wav: str,
    num_step: int = DEFAULT_NUM_STEP,
    guidance_scale: float = DEFAULT_GUIDANCE,
    take_count: int = WORK_REF_TAKES,
    instruct: Optional[str] = None,
    log: LogCallback = print,
    stop_event=None,
) -> tuple[str, str, Optional[float]]:
    device = resolve_device()
    model = load_model(models_root, device, log)
    try:
        takes, generic, sr, model_device = _generate_target_reference_takes_with_model(
            model,
            source_ref_audio=source_ref_audio,
            source_ref_text=source_ref_text,
            target_language=target_language,
            num_step=num_step,
            guidance_scale=guidance_scale,
            take_count=take_count,
            instruct=instruct,
            log=log,
            stop_event=stop_event,
        )
    finally:
        del model
        _release_cuda_cache()
    return _write_best_target_reference_sample(
        takes,
        generic=generic,
        source_ref_audio=source_ref_audio,
        take_sr=sr,
        models_root=models_root,
        device=model_device,
        output_wav=output_wav,
        log=log,
    )


def generate_voice_design_sample(
    *,
    models_root: str,
    target_language: str,
    instruct: str,
    output_wav: str,
    num_step: int = DEFAULT_NUM_STEP,
    guidance_scale: float = DEFAULT_GUIDANCE,
    log: LogCallback = print,
    stop_event=None,
) -> tuple[str, str]:
    """Generate a target-language SPEAKER1 sample from OmniVoice voice design."""
    device = resolve_device()
    model = load_model(models_root, device, log)
    try:
        return generate_voice_design_sample_with_model(
            model,
            target_language=target_language,
            instruct=instruct,
            output_wav=output_wav,
            num_step=num_step,
            guidance_scale=guidance_scale,
            log=log,
            stop_event=stop_event,
        )
    finally:
        del model
        _release_cuda_cache()


def generate_voice_design_sample_with_model(
    model,
    *,
    target_language: str,
    instruct: str,
    output_wav: str,
    num_step: int = DEFAULT_NUM_STEP,
    guidance_scale: float = DEFAULT_GUIDANCE,
    log: LogCallback = print,
    stop_event=None,
) -> tuple[str, str]:
    """Generate a target-language SPEAKER1 sample from an already-loaded model."""
    import soundfile as sf

    generic, duration = _resolve_generic_ref(target_language)
    if not generic:
        raise ValueError(f"No built-in target-language reference sentence for: {target_language}")
    if stop_event is not None and stop_event.is_set():
        raise RuntimeError("Stopped by user.")
    sr = int(getattr(model, "sampling_rate", None) or 24000)
    wav = _normalize_peak(
        np.asarray(
            model.generate(
                text=generic,
                language=target_language,
                instruct=instruct,
                duration=duration,
                num_step=num_step,
                guidance_scale=guidance_scale,
            )[0],
            dtype=np.float32,
        ).reshape(-1),
        0.85,
    )
    out = Path(output_wav)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), wav, sr)
    log(f"[synth] voice-design SPEAKER1 sample -> {out.name}")
    return str(out), generic


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
                            log: LogCallback, text_field: str = "tgt_text"):
    """One reusable voice-clone prompt per speaker.

    OmniVoice clones reliably (down to very short lines) only when the reference
    is in the SAME language as the target. So for fixed supported targets we first
    synthesize a clean same-language "working reference" — clone the speaker's
    source-language video reference saying a generic sentence — then clone from
    THAT. For other target languages we fall back to the cross-lingual video
    reference directly (less reliable for short lines).

    The working reference is generated several times and the take whose ECAPA
    speaker embedding is closest to the ORIGINAL video reference wins, so the
    second cloning hop drifts as little as possible from the real voice.
    """
    device = "cuda" if str(model.device).startswith("cuda") else "cpu"
    # Empty ref_text + ref audio = untranscribed prompt: out-of-distribution for
    # OmniVoice and the root cause of "both speakers sound the same generic voice".
    _fill_missing_ref_texts(manifest, clone_dir, models_root, device, log)

    generic, work_dur = _resolve_generic_ref(language)
    prompts = {}

    # VRAM discipline: OmniVoice (fp16) nearly fills the GPU during generation,
    # so the ECAPA scorer must NOT be resident at the same time. Phase 1
    # generates all work_ref takes for all speakers; phase 2 loads ECAPA once,
    # scores everything, and frees it before any further generation.
    takes_by_spk: dict = {}  # spk -> (ref_audio path, [takes])
    for spk, info in manifest.get("speakers", {}).items():
        if (info or {}).get("skip_synthesis"):
            log(f"[synth] {spk}: marked skipped, prompt not built.")
            continue
        ref_rel = info.get("ref_audio") or ""
        ref_audio = clone_dir / ref_rel
        if not ref_rel or not ref_audio.is_file():
            log(f"[synth] {spk}: missing ref_audio, skipped.")
            continue
        # Do not coerce empty ref_text to None. OmniVoice auto-transcribes when
        # ref_text is None, which downloads openai/whisper-large-v3-turbo.
        ref_text = (info.get("ref_text") or "").strip()
        try:
            prompt_ref_audio = prepare_prompt_reference_audio(ref_audio, log=log)
            if (
                text_field == "tgt_text"
                and info.get("skip_work_ref")
                and same_language(info.get("ref_language"), language)
            ):
                prompts[spk] = model.create_voice_clone_prompt(prompt_ref_audio, ref_text, preprocess_prompt=True)
                log(f"[synth] {spk}: target-language SPEAKER1 basis -> {ref_audio.name}")
                continue
            if generic:
                takes = []
                for take in range(1, WORK_REF_TAKES + 1):
                    cand = _normalize_peak(
                        np.asarray(model.generate(
                            text=generic, ref_audio=prompt_ref_audio, ref_text=ref_text,
                            language=language, duration=work_dur,
                            num_step=num_step, guidance_scale=guidance_scale,
                        )[0], dtype=np.float32).reshape(-1),
                        0.85,
                    )
                    takes.append(cand)
                    log(f"[synth] {spk}: work_ref take {take}/{WORK_REF_TAKES} generated")
                takes_by_spk[spk] = (ref_audio, takes)
            else:
                prompts[spk] = model.create_voice_clone_prompt(prompt_ref_audio, ref_text, preprocess_prompt=True)
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
            import soundfile as sf

            work_path = clone_dir / f"work_ref_{spk}.wav"
            sf.write(str(work_path), wav, sr)
            prompt_work_path = prepare_prompt_reference_audio(work_path, log=log)
            prompts[spk] = model.create_voice_clone_prompt(prompt_work_path, generic, preprocess_prompt=True)
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
    min_gain: float = 0.0,
    max_gain: float = 25.0,
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


# Tempo-fit: make each dubbed line roughly track its source segment's pace so a
# slow original does not leave big silent gaps and a fast one is not rushed. We
# clamp OmniVoice's speed factor to a band so pacing stays natural (extreme
# slow-downs sound unnatural); anything beyond the band leaves a residual pause
# rather than distorting speech.
_TEMPO_FIT_SPEED_BANDS = {
    "off": None,
    "moderate": (0.85, 1.15),
    "strong": (0.72, 1.30),
}

# Approximate natural speaking rate in non-space characters per second, keyed by
# (normalized) TARGET language. Alphabetic/Cyrillic scripts pack ~12-14 chars/s;
# CJK and Korean ~5-7 chars/s because each glyph is roughly a syllable. Only the
# TARGET language matters here — the SOURCE side uses the measured segment slot
# (real timing), not an estimate. These need only be good enough to decide how
# far to stretch within the clamped speed band.
_LANG_CHARS_PER_SEC = {
    "chinese": 5.5,
    "japanese": 6.5,
    "korean": 6.5,
    "thai": 8.0,
    "english": 13.0,
    "german": 13.0,
    "french": 13.0,
    "spanish": 13.5,
    "italian": 13.5,
    "portuguese": 13.0,
    "russian": 12.0,
}
_DEFAULT_CHARS_PER_SEC = 12.0


def _estimate_natural_duration(text: str, language: Optional[str] = None) -> float:
    """Rough natural spoken length (seconds) of ``text`` in its TARGET language."""
    visible = len(re.sub(r"\s+", "", text or ""))
    if visible == 0:
        return 0.3
    cps = _LANG_CHARS_PER_SEC.get(normalize_language_name(language), _DEFAULT_CHARS_PER_SEC)
    return float(max(0.3, visible / cps))


def _tempo_fit_speed(text: str, start: float, end: float, language: Optional[str],
                     tempo_fit: str) -> Optional[float]:
    """Speed factor to pace ``text`` toward the source slot, clamped to the band."""
    band = _TEMPO_FIT_SPEED_BANDS.get(tempo_fit or "off")
    if not band:
        return None
    src = float(end) - float(start)
    natural = _estimate_natural_duration(text, language)
    if src <= 0.2 or natural <= 0.05:
        return None
    return float(np.clip(natural / src, band[0], band[1]))


def _is_fatal_generation_error(exc: Exception) -> bool:
    text = str(exc).lower()
    fatal_markers = (
        "out of memory",
        "cuda error",
        "cublas",
        "cudnn",
        "device-side assert",
    )
    return any(marker in text for marker in fatal_markers)


def _synthesize_take(
    model,
    *,
    text: str,
    prompt,
    language: Optional[str],
    duration: Optional[float],
    speed: Optional[float],
    num_step: int,
    guidance_scale: float,
    postprocess_output: bool = True,
) -> np.ndarray:
    kwargs = {
        "text": text,
        "voice_clone_prompt": prompt,
        "language": language,
        "duration": duration,
        "speed": speed,
        "num_step": num_step,
        "guidance_scale": guidance_scale,
    }
    if not postprocess_output:
        kwargs["postprocess_output"] = False
    return np.asarray(model.generate(**kwargs)[0], dtype=np.float32).reshape(-1)


def _silence_fallback(unit: dict, sr: int) -> np.ndarray:
    dur = max(0.15, min(3.0, float(unit["end"]) - float(unit["start"])))
    return np.zeros(max(1, int(round(dur * sr))), dtype=np.float32)


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
    tempo_fit: str = "off",
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
    speakers_info = manifest.get("speakers", {}) or {}
    skipped_speakers = {
        str(spk)
        for spk, info in speakers_info.items()
        if (info or {}).get("skip_synthesis")
    }
    if skipped_speakers:
        log(f"[synth] skipped speakers: {', '.join(sorted(skipped_speakers))}")

    prompts = _build_speaker_prompts(
        model, manifest, clone_dir, language, sr, num_step, guidance_scale,
        models_root, log, text_field=text_field
    )

    units = _merge_units(segments, text_field, merge_gap)
    if skipped_speakers:
        before = len(units)
        units = [u for u in units if str(u.get("speaker") or "") not in skipped_speakers]
        log(f"[synth] skipped {before - len(units)} synthesis units for skipped speakers")
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
        # Tempo-fit paces the line toward its source slot (bounded); it overrides
        # the hard "fit" duration when enabled. duration takes priority over speed
        # in OmniVoice, so pass only one.
        speed = _tempo_fit_speed(text, unit["start"], unit["end"], language, tempo_fit)
        if speed is not None:
            duration = None

        # Same-language working ref clones reliably even for short lines. Short
        # lines are still the riskiest; retry a few times and keep the least-silent
        # take (catches the occasional empty/garbled generation) — no carrier.
        tries = 3 if (prompt is not None and _clone_char_count(text) < min_clone_chars) else 1
        clip = None
        generation_warnings: list[str] = []
        for _ in range(tries):
            cand = None
            try:
                cand = _synthesize_take(
                    model,
                    text=text,
                    prompt=prompt,
                    language=language,
                    duration=duration,
                    speed=speed,
                    num_step=num_step,
                    guidance_scale=guidance_scale,
                )
            except Exception as exc:
                if _is_fatal_generation_error(exc):
                    raise
                generation_warnings.append(f"normal={exc}")

            if cand is None or cand.size == 0 or _rms(cand) <= 1e-6:
                try:
                    retry = _synthesize_take(
                        model,
                        text=text,
                        prompt=prompt,
                        language=language,
                        duration=duration,
                        speed=speed,
                        num_step=num_step,
                        guidance_scale=guidance_scale,
                        postprocess_output=False,
                    )
                    if retry.size > 0:
                        if cand is None or _rms(retry) > _rms(cand):
                            cand = retry
                    else:
                        generation_warnings.append("retry_no_postprocess=empty")
                except Exception as exc:
                    if _is_fatal_generation_error(exc):
                        raise
                    generation_warnings.append(f"retry_no_postprocess={exc}")

            if cand is None:
                cand = np.zeros(0, dtype=np.float32)
            if clip is None or _rms(cand) > _rms(clip):
                clip = cand
            if _rms(cand) > 0.02:
                break

        used_silence_fallback = False
        if clip is None or clip.size == 0:
            used_silence_fallback = True
            clip = _silence_fallback(unit, sr)

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
            out_db = _rms_db(clip)
            loudness_note = f" gain={gain:.2f} src={src_db:.1f}dB gen={synth_db:.1f}dB out={out_db:.1f}dB"
            if loudness_mode == "envelope":
                loudness_note += f" env(a={envelope_alpha:g})"
        if generation_warnings:
            loudness_note += f" warn=gen_retry({'; '.join(generation_warnings[:2])})"
        if used_silence_fallback:
            loudness_note += " warn=silence_fallback"
        n_noref += int(prompt is None)
        n_clone += int(prompt is not None)
        start_sample = max(0, int(round(unit["start"] * sr)))
        si._mix_timeline_segment(timeline, start_sample, clip)
        max_end_sample = max(max_end_sample, start_sample + clip.size)
        tempo_note = f" speed={speed:.2f}" if speed is not None else ""
        log(
            f"[synth] {idx}/{len(units)} {unit['speaker']} "
            f"{unit['start']:.1f}-{unit['end']:.1f}s{loudness_note}{tempo_note}"
        )
        _release_cuda_cache()

    timeline = timeline[: max(1, min(timeline.size, max_end_sample))]
    out_path = si.default_si_audio_path(video_path)
    si.write_wav_mono(out_path, timeline, sr)
    log(f"[synth] wrote {out_path} ({timeline.size / sr:.1f}s; {n_clone} cloned, {n_noref} no-ref)")
    duck_spans = _duck_spans_for_segments(segments, text_field, skipped_speakers)
    duck_duration = max(total_dur, timeline.size / float(sr))
    duck_path = si.default_si_duck_key_path(out_path)
    si.write_duck_key_wav(duck_path, duck_spans, duck_duration, sr)
    log(f"[synth] wrote duck key {duck_path} ({duck_duration:.1f}s; {len(duck_spans)} spans)")
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
