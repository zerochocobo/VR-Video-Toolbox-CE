from __future__ import annotations

import gc
import re
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
# Shortest audio that still makes a usable IndexTTS prompt. Deliberately small:
# a short line cloning from itself is a design decision, not a defect, so this
# catches only spans that cannot work at all. Observed on hnvr-174 part1 -- a
# 0.06s slot gives a mel spectrogram too small for the 3x3 kernels ("Calculated
# padded input size per channel: (2 x 1024)") and a zero-length slot degenerates
# to a single sample that vanishes on resampling. Ordinary short sentences
# (0.3s and up) still prompt from exactly themselves.
MIN_REFERENCE_SECONDS = 0.3
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
# --- speaker timbre anchor ---
# IndexTTS re-derives the voice from whatever prompt it is given, so cloning
# every sentence from itself lets the voice drift line to line. Anchoring the
# timbre on one reference per speaker and passing the sentence only as the
# emotion prompt keeps the voice fixed. The window follows refsel's: under
# ~3s the prompt is too thin to fix a voice, over ~10s the extra material
# dilutes it rather than sharpening it, so "best" is not "longest".
ANCHOR_MIN_SECONDS = 3.0
ANCHOR_MAX_SECONDS = 10.0
ANCHOR_IDEAL_SECONDS = 6.0
# IndexTTS-2's emotion prompt is NOT fully disentangled from timbre: at alpha
# 1.0 the sentence's own voice leaks back in and cancels the anchor. The
# model's own default is 1.0 (vendor/indextts/infer_v2.py), so an anchored
# call must always pass this explicitly.
SPEAKER_ANCHOR_EMOTION_ALPHA = 0.7
# Absolute dBFS gates do not transfer across titles. Measured over the 32
# existing manifests, the whisper-heavy ones sit far under the fixed
# stable-reference floor -- on sivr-434 part1, 174 of 182 sentences fail
# active-RMS >= -42 dBFS and the loudest 3-10s candidate reads -43 dBFS -- so
# an absolute floor rejects the entire video rather than its quiet lines.
# Rank the anchor against the title's own median instead, the same relative
# treatment level matching already needs.
ANCHOR_MAX_BELOW_MEDIAN_DB = 6.0
# Some titles have no line long enough to anchor on at all: sivr-314 part1 has
# 80 lines, the longest 2.7s, median 1.2s. There the anchor is stitched from
# adjacent lines of the same speaker instead. Whispers are excluded by the same
# median test above -- a stitched whisper would pin the voice at the one level
# we least want it fixed at -- and so are unvoiced pieces.
STITCH_MIN_PIECE_SECONDS = 0.45
STITCH_JOIN_FADE_SECONDS = 0.012
STITCH_MAX_GAP_SECONDS = 30.0
# IndexTTS normalises every sentence to its own comfortable level, so a whisper
# and a shout come back at the same loudness. Measured over three titles, the
# correlation between source and generated active-RMS is ~0 (-0.15..0.39) with
# per-segment errors from -20 dB to +51 dB. These bound the correction.
# Boosting lifts the model's noise floor with the signal, so this used to be a
# flat 12 dB. On hnvr-174 part1 that flat cap was the binding constraint on 100
# of 120 clips -- IndexTTS returns this title at a -45.9 dBFS median because it
# tracks the prompt's level (measured slope 0.48 dB per dB of prompt level, so a
# quiet title stays quiet), and 12 dB could not reach an audible target. The
# floor is what actually matters, not the gain, so the real limit is applied
# per clip by ``noise_floor_boost_cap`` below and this is only a backstop.
LEVEL_MATCH_MAX_BOOST_DB = 24.0
# A boosted clip's own hiss is harmless while the film's background covers it,
# so the per-clip limit is set on the floor rather than on the gain. How much
# headroom to give it is a direct trade against the imbalance this whole change
# exists to remove: on hnvr-174 part1, holding every floor at or under the local
# background left the worst speaker (SPEAKER_01, one of the quiet women) 7.5 dB
# below the rest, because her clips have the worst internal SNR and hit the cap
# first. Allowing the floor 6 dB of headroom closes that spread to 2.1 dB while
# the floors still land at a -49.2 dBFS median against a -46.0 dBFS background
# -- the median clip is quieter than the bed either way.
NOISE_FLOOR_MAX_ABOVE_BACKGROUND_DB = 6.0
# Silence needed before a clip's floor is worth believing (10 ms hops, so this
# is 100 ms of non-speech). A line with less than that gets no floor reading at
# all and falls back to the flat cap, rather than having a quiet syllable
# mistaken for its hiss.
MIN_SILENT_FRAMES_FOR_FLOOR = 10
# How quiet a line may get is bounded by the floor below, not by this clamp,
# which only exists to catch an absurd measurement. Set too low it becomes the
# real limiter and the loudest outliers never come down: on sivr-314 a 24 dB
# clamp left 11 clips short and held the source/dub correlation at 0.76, while
# 40 dB reaches 0.97 with a 1.7 dB residual.
LEVEL_MATCH_MAX_CUT_DB = 40.0
# A whisper 60 dB down is inaudible on its own, and reproducing that literally
# would make the translated line unusable. Keep quiet lines quiet but audible:
# 18 dB below the median put lines at -67 dBFS, which nobody can hear over a
# background bed, so this is the audibility limit rather than a fidelity one.
LEVEL_FLOOR_BELOW_MEDIAN_DB = 12.0
LEVEL_MATCH_PEAK_CEILING = 0.99
# Reproducing the source's dynamics literally makes a quiet title inaudible.
# sivr-314 part1's own speech median is -52 dBFS with a p10 of -60.4, so lines
# were faithfully matched down to -61 and -64 dBFS -- nobody can hear those.
# The relative floor below could not catch it: 12 dB under a -52 dB median is
# -64, which almost nothing reaches. Reproduce a fraction of each line's
# deviation instead, so a whisper still reads as quieter than a normal line
# without being placed below audibility.
LEVEL_DYNAMIC_RANGE_RATIO = 0.6
# ...and lift a title whose speech is quiet in absolute terms until its median
# line is audible. Only ever a lift: a title already at or above this keeps its
# levels exactly as before, so the titles where matching already worked are
# untouched. Anchoring on an absolute figure rather than on the model's own
# output keeps this independent of whatever level IndexTTS happens to produce.
#
# -42 dBFS was set as "the quietest line one can still make out in isolation",
# which is the wrong question: the dub is heard against the original, not in
# isolation. Measured on hnvr-174 part1, where the anchor was doing its job and
# put the dub median at exactly -42.0: the dub landed +3.0 dB over the source
# mixture, but the source itself is a -45.6 dBFS title, so the finished _SI.mp4
# carried its dialogue at -44..-49 dBFS -- 0.7..7.3 dB *below* the original
# film. That is the "turned it all the way up and still cannot make it out"
# report. A normal dialogue mix sits near -27 dBFS RMS, which leaves ~13 dB
# over this material's background and still ~12 dB of peak headroom.
LEVEL_AUDIBLE_MEDIAN_DBFS = -27.0
# Per-speaker level is a microphone-placement artefact, not intent: on hnvr-174
# part1 the near-mic male read -38.4 dBFS while the three women read -56.1,
# -51.5 and -66.1, and the dub reproduced that faithfully, so the women stayed
# inaudible while the man was fine. Bring every speaker's median to the common
# median before shaping; each speaker's own dynamics are still preserved by
# LEVEL_DYNAMIC_RANGE_RATIO within their line set.
SPEAKER_MEDIAN_MATCH_RATIO = 1.0
# Below this many lines a speaker's median is noise, so leave that speaker be.
MIN_SPEAKER_LINES_FOR_MEDIAN_MATCH = 4
# Hard time compression after generation had no limit at all: the model's
# duration_factor is clamped to 0.80..1.30, but whatever came back was then
# squeezed into the slot regardless. On sivr-314 that reached 2.33x, and 46 of
# 80 lines were over 1.3x -- which is what "listen but cannot make it out"
# sounds like. Every one of those 46 had an empty gap after it (median 14.6s),
# so the squeeze bought nothing.
MAX_HARD_COMPRESSION = 1.15
SLOT_OVERFLOW_GUARD_SECONDS = 0.15
SILENT_CLIP_DBFS = -200.0
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


def natural_reading_seconds(text: str, language: str) -> float:
    """How long a person would take to say this line, unhurried."""
    visible = max(1, len("".join((text or "").split())))
    cps = {"zh": 5.5, "ja": 6.5, "en": 13.0, "es": 13.5, "ar": 11.0}.get(language, 12.0)
    return max(0.3, visible / cps)


def duration_factor_for_slot(text: str, slot_seconds: float, language: str, tempo_fit: str = "moderate") -> float:
    natural = natural_reading_seconds(text, language)
    # IndexTTS semantics: >1 is slower. Never ask it to slow down: measured over
    # sivr-314's 80 lines it returns 1.40x the requested duration at the median
    # and up to 4.15x on short lines, so a "speak slower" request comes back as
    # a badly dragged reading that then has to be crushed to fit. Asking for
    # natural speed and letting the line run into the gap behind it is the
    # better trade. Speeding up is still allowed -- a slot genuinely shorter
    # than the translated line needs it.
    bands = {"moderate": (0.80, 1.00), "strong": (0.65, 1.00)}
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


def widen_degenerate_span(start: float, end: float, total_seconds: float) -> tuple[float, float]:
    """Grow a span that is too short to survive resampling into a usable prompt.

    Word anchoring can collapse a line onto a single timestamp -- hnvr-174
    part1 id=139 arrived as 1109.08-1109.08, an exactly zero-length slot. The
    extractor's ``max(lo + 1, ...)`` then wrote a one-sample WAV, and resampling
    that to 22 kHz left a zero-dimensional tensor: "Input waveform must have
    only one dimension, shape is ()". Widen symmetrically instead, clamped to
    the file, so the prompt is always long enough to have a spectrogram.
    """
    start, end = float(start), float(end)
    if end - start >= MIN_REFERENCE_SECONDS:
        return start, end
    centre = 0.5 * (start + end)
    half = 0.5 * MIN_REFERENCE_SECONDS
    lo = centre - half
    hi = centre + half
    if lo < 0.0:
        lo, hi = 0.0, min(total_seconds, MIN_REFERENCE_SECONDS)
    elif hi > total_seconds:
        hi = total_seconds
        lo = max(0.0, total_seconds - MIN_REFERENCE_SECONDS)
    return lo, hi


def extract_sentence_reference(source_audio: str | Path, segment: dict,
                               temp_dir: str | Path, *,
                               source_wav: np.ndarray | None = None,
                               source_sr: int | None = None) -> str:
    """Extract exactly the current manifest sentence as the IndexTTS prompt.

    A slot too short to be a prompt at all is widened around its centre; the
    manifest's own timing is untouched, only the audio handed to the model.
    """
    path = Path(temp_dir) / f"sentence_ref_{int(segment.get('id', 0)):05d}.wav"
    if source_wav is not None and source_sr is not None:
        start, end = widen_degenerate_span(
            segment["start"], segment["end"], source_wav.size / float(source_sr)
        )
        return _extract_reference_from_audio(
            source_wav, source_sr, start, end, path, source_audio,
        )
    wav, sr = _read_audio(source_audio)
    start, end = widen_degenerate_span(
        segment["start"], segment["end"], wav.size / float(sr)
    )
    return _extract_reference_from_audio(wav, sr, start, end, path, source_audio)


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


def phonetic_variety(text: str) -> float:
    """How much *different* sound a line carries, from 0 to 1.

    A timbre anchor has to show the voice across varied phonemes. Judged on
    acoustics alone, a sustained moan wins every time -- it is long, loud and
    almost perfectly periodic -- and that is what the automatic anchor kept
    choosing. Measured on the ipvr-385 candidate list, the distinct-character
    ratio separates the two cleanly: the moans read 0.33
    ("やばいやばいやばい") and 0.50 ("くっくりくれくれかいかい"), while every
    real line lands between 0.67 and 0.93. Repetition detection does not catch
    these -- they are far too short for it.
    """
    chars = [ch for ch in (text or "") if not ch.isspace() and ch.isalnum()]
    if not chars:
        return 0.0
    return len(set(chars)) / len(chars)


# Below this the line is mostly one repeated sound. Not a hard gate: a title
# with nothing better must still get an anchor, so this scales the score
# instead of rejecting, and a clean line always outranks a moan.
ANCHOR_GOOD_TEXT_VARIETY = 0.65


def _anchor_score(profile: dict, median_active_rms: float,
                  segment: dict | None = None) -> float:
    """Rank a sentence as a speaker's timbre anchor; higher is better.

    Deliberately not "longest": a long sentence that is mostly silence, or a
    breathy one, is a worse voice print than a shorter clean one. Periodicity
    weighs most because an unvoiced prompt is exactly what made per-sentence
    cloning drift. Loudness is judged against the title, never in absolute
    dBFS, so a quiet video still yields its own best line.
    """
    duration = float(profile["duration"])
    active = float(profile["active_seconds"])
    if not (ANCHOR_MIN_SECONDS <= duration <= ANCHOR_MAX_SECONDS):
        return float("-inf")
    if active < STABLE_REFERENCE_MIN_ACTIVE_SECONDS:
        return float("-inf")
    if float(profile["periodic_ratio"]) < STABLE_REFERENCE_MIN_PERIODIC_RATIO:
        return float("-inf")
    relative_db = float(profile["active_rms_dbfs"]) - float(median_active_rms)
    if relative_db < -ANCHOR_MAX_BELOW_MEDIAN_DB:
        return float("-inf")
    length = -abs(duration - ANCHOR_IDEAL_SECONDS) / ANCHOR_IDEAL_SECONDS
    density = active / max(duration, 1e-6)
    loudness = min(1.0, max(0.0, (relative_db + ANCHOR_MAX_BELOW_MEDIAN_DB) / 12.0))
    acoustic = (2.0 * float(profile["periodic_ratio"])
                + 1.5 * loudness + 1.0 * density + 1.0 * length)
    if segment is None:
        return acoustic
    # Scale by how varied the line's sounds are, so a clean sentence beats a
    # moan that happens to be louder and more periodic than anything else.
    variety = phonetic_variety(
        str(segment.get("src_text") or segment.get("text") or "")
    )
    return acoustic * min(1.0, variety / ANCHOR_GOOD_TEXT_VARIETY)


def select_speaker_timbre_anchors(entries: list[dict],
                                  profiles: list[dict]) -> dict[str, int]:
    """Pick one timbre reference per speaker: the best-scoring sentence.

    Segments carrying no speaker label share the key "", so a single-speaker
    title gets one anchor for the whole video without needing diarization at
    all -- which is the case the drift complaint came from. A speaker with no
    qualifying sentence simply gets no anchor and keeps cloning per sentence.
    """
    median_active_rms = _median_active_rms(profiles)
    best: dict[str, tuple[float, int]] = {}
    for index, (segment, profile) in enumerate(zip(entries, profiles)):
        score = _anchor_score(profile, median_active_rms, segment)
        if score == float("-inf"):
            continue
        key = str(segment.get("speaker") or "")
        if key not in best or score > best[key][0]:
            best[key] = (score, index)
    return {key: index for key, (_score, index) in best.items()}


def _median_active_rms(profiles: list[dict]) -> float:
    """The title's own speech level, which every anchor test is relative to."""
    levels = [
        float(profile["active_rms_dbfs"]) for profile in profiles
        if float(profile["active_seconds"]) > 0.0
    ]
    return float(np.median(levels)) if levels else -240.0


def _stitch_candidate(profile: dict, median_active_rms: float) -> bool:
    """May this line be one piece of a stitched anchor?

    Looser than a whole anchor on length -- that is the entire point -- but not
    on voice or level: a breath contributes no timbre, and a whisper stitched
    into the anchor would fix the cloned voice at whisper level for the whole
    title.
    """
    return (
        float(profile["duration"]) >= STITCH_MIN_PIECE_SECONDS
        and float(profile["active_seconds"]) > 0.0
        and float(profile["periodic_ratio"]) >= STABLE_REFERENCE_MIN_PERIODIC_RATIO
        and float(profile["active_rms_dbfs"]) >= median_active_rms - ANCHOR_MAX_BELOW_MEDIAN_DB
    )


def _best_stitch_run(entries: list[dict], profiles: list[dict], usable: list[int],
                     max_gap: float) -> list[int]:
    best: tuple[float, list[int]] | None = None
    for start in range(len(usable)):
        run: list[int] = []
        total = 0.0
        for index in usable[start:]:
            if run:
                gap = float(entries[index]["start"]) - float(entries[run[-1]]["end"])
                if gap > max_gap:
                    break
            piece = min(float(profiles[index]["duration"]), ANCHOR_MAX_SECONDS - total)
            if piece <= 0.0:
                break
            run.append(index)
            total += piece
            if total >= ANCHOR_IDEAL_SECONDS:
                break
        if total < ANCHOR_MIN_SECONDS:
            continue
        score = (
            float(np.mean([profiles[i]["periodic_ratio"] for i in run]))
            + float(np.mean([profiles[i]["active_seconds"] / max(profiles[i]["duration"], 1e-6)
                             for i in run]))
            - abs(total - ANCHOR_IDEAL_SECONDS) / ANCHOR_IDEAL_SECONDS
            - 0.05 * len(run)          # fewer, longer pieces splice more cleanly
        )
        if best is None or score > best[0]:
            best = (score, run)
    return best[1] if best else []


def select_stitch_pieces(entries: list[dict], profiles: list[dict], speaker: str,
                         median_active_rms: float) -> list[int]:
    """Pick lines to concatenate into one anchor, preferring neighbours.

    Neighbouring first: consecutive lines share a scene, a distance to the
    microphone and a delivery, so they splice into something that still sounds
    like one utterance. But a speaker whose usable lines are scattered across
    the title still deserves an anchor -- refusing one there costs more than
    splicing across a scene change -- so the adjacency limit is a preference,
    retried without it. Returns [] only when there is genuinely too little
    non-whisper, voiced material, and the caller then clones per sentence.
    """
    usable = [
        index for index, (segment, profile) in enumerate(zip(entries, profiles))
        if str(segment.get("speaker") or "") == speaker
        and _stitch_candidate(profile, median_active_rms)
    ]
    if not usable:
        return []
    return (_best_stitch_run(entries, profiles, usable, STITCH_MAX_GAP_SECONDS)
            or _best_stitch_run(entries, profiles, usable, float("inf")))


def write_stitched_anchor(source_wav: np.ndarray, source_sr: int, entries: list[dict],
                          pieces: list[int], out_path: str | Path) -> str:
    """Concatenate the chosen lines into one prompt WAV, fading each join."""
    fade = max(1, int(round(STITCH_JOIN_FADE_SECONDS * source_sr)))
    limit = int(round(ANCHOR_MAX_SECONDS * source_sr))
    clips: list[np.ndarray] = []
    total = 0
    for index in pieces:
        segment = entries[index]
        lo = max(0, int(round(float(segment["start"]) * source_sr)))
        hi = min(source_wav.size, max(lo + 1, int(round(float(segment["end"]) * source_sr))))
        clip = np.asarray(source_wav[lo:hi], dtype=np.float32).copy()
        if clip.size > limit - total:
            clip = clip[: limit - total]
        if clip.size <= 2 * fade:
            continue
        # Fade both ends: a butt join between two lines clicks, and the click
        # is exactly the kind of transient the model would take for a consonant.
        clip[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        clip[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        clips.append(clip)
        total += clip.size
        if total >= limit:
            break
    if not clips:
        raise ValueError("no usable pieces for a stitched anchor")
    _write_audio(out_path, np.concatenate(clips), source_sr)
    return str(out_path)


def _reference_profiles(source_wav: np.ndarray, source_sr: int,
                        entries: list[dict]) -> list[dict]:
    profiles = []
    for segment in entries:
        lo = max(0, int(round(float(segment["start"]) * source_sr)))
        hi = min(source_wav.size, max(lo + 1, int(round(float(segment["end"]) * source_sr))))
        profiles.append(analyze_reference_quality(source_wav[lo:hi], source_sr))
    return profiles


def match_speaker_medians(entries: list[dict], targets: list[float]) -> tuple[list[float], dict[str, float]]:
    """Remove the per-speaker level imbalance from a set of loudness targets.

    Returns the adjusted targets and the shift applied to each speaker, so the
    caller can report it. Speakers with too few lines, and titles where only
    one speaker qualifies, are returned untouched.
    """
    per_speaker: dict[str, list[float]] = {}
    for entry, value in zip(entries, targets):
        name = entry.get("speaker")
        if name:
            per_speaker.setdefault(str(name), []).append(float(value))
    medians = {
        name: float(np.median(values))
        for name, values in per_speaker.items()
        if len(values) >= MIN_SPEAKER_LINES_FOR_MEDIAN_MATCH
    }
    if len(medians) < 2:
        return list(targets), {}
    common = float(np.median(list(medians.values())))
    shifts = {
        name: SPEAKER_MEDIAN_MATCH_RATIO * (common - median)
        for name, median in medians.items()
    }
    adjusted = [
        float(value) + shifts.get(str(entry.get("speaker")), 0.0)
        for entry, value in zip(entries, targets)
    ]
    return adjusted, shifts


def segment_level_targets(entries: list[dict], profiles: list[dict]) -> tuple[list[float], str]:
    """Collect the per-segment loudness target and decide how to apply it.

    The target is the source mixture's own level over each line's window,
    measured from audio16k.wav. This used to prefer a bandit-separated speech
    stem where one existed, on the theory that an isolated voice is the truer
    target. It was not worth what it cost: on hnvr-174 only 140 of 304 lines
    produced a usable stem, its failures read as much as 52 dB under the
    mixture and had to be guarded against one by one, and where it did work the
    final dub level agreed with the mixture to a median 0.0-0.3 dB (86% of
    lines within 2 dB, urvrsp-566) -- for a 13 GB VRAM peak the separator asked
    on every run.

    ``mixture`` mode is absolute: these are real dBFS readings, so they get the
    audibility anchor. Leaving the overall level where IndexTTS put it (what
    ``relative`` does) means -45.9 dBFS on this material, which is the
    inaudible dub all over again. What the mixture cannot do as well is shape:
    it is inflated by music and effects, so a whisper under loud music reads as
    loud there. LEVEL_DYNAMIC_RANGE_RATIO reproduces only a fraction of each
    line's deviation, which is what keeps that error small.

    ``relative`` is left for the case where there is no usable reading at all.
    """
    mixture = [float(profile["active_rms_dbfs"]) for profile in profiles]
    audible = [value for value in mixture if value > SILENT_CLIP_DBFS]
    return mixture, ("mixture" if audible else "relative")


def clip_noise_floor_dbfs(clip: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float | None:
    """The level of the clip's *silence*, or ``None`` when it has none.

    Measured only on frames the activity test calls non-speech, using the same
    threshold as :func:`analyze_reference_quality` so both agree on what speech
    is. A blanket 5th percentile over every frame -- what this did first --
    reads the quietest *syllable* of a line that never pauses as though it were
    hiss. On hnvr-174 that reading tracked duration (r=-0.35) and fell 11 dB
    from the sub-0.5s lines to the 3s+ ones while their speech level barely
    moved, so the boost cap was penalising short lines for having no gap in
    them -- exactly the lines that were hardest to make out.

    ``None`` means "no opinion": the caller falls back to the flat cap rather
    than inventing a floor from speech.
    """
    audio = np.asarray(clip, dtype=np.float32).reshape(-1)
    frame = max(1, int(round(sample_rate * 0.025)))
    hop = max(1, int(round(sample_rate * 0.010)))
    if audio.size < frame:
        return None
    peak = float(np.max(np.abs(audio)))
    if peak <= 0.0:
        return None
    count = 1 + (audio.size - frame) // hop
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(count, frame),
        strides=(audio.strides[0] * hop, audio.strides[0]),
    )
    frame_rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    threshold = max(peak * (10.0 ** (-REFERENCE_ACTIVITY_TOP_DB / 20.0)), 1e-5)
    silent = frame_rms[frame_rms < threshold]
    if silent.size < MIN_SILENT_FRAMES_FOR_FLOOR:
        return None
    return float(20.0 * np.log10(max(float(np.median(silent)), 1e-12)))


def noise_floor_boost_cap(noise_floor: float | None, background: float | None) -> float:
    """How far a clip may be boosted before its hiss escapes the film's bed."""
    if noise_floor is None or background is None:
        return LEVEL_MATCH_MAX_BOOST_DB
    allowed = float(background) + NOISE_FLOOR_MAX_ABOVE_BACKGROUND_DB - float(noise_floor)
    return float(min(LEVEL_MATCH_MAX_BOOST_DB, max(0.0, allowed)))


def compute_level_gains(measured: list[float], targets: list[float], mode: str,
                        noise_floors: list[float | None] | None = None,
                        backgrounds: list[float | None] | None = None) -> list[float]:
    """Return the per-segment gain in dB that matches the dub to the source.

    ``noise_floors`` and ``backgrounds`` are optional; supplying both caps each
    clip's boost at the point where its own hiss would rise above the film's
    background. Without them every clip shares the flat backstop, which is the
    behaviour from before the cap existed.
    """
    if not measured:
        return []
    audible = [value for value in measured if value > SILENT_CLIP_DBFS]
    target_median = float(np.median(targets))
    if mode == "relative":
        offset = float(np.median(audible)) - target_median if audible else 0.0
    else:
        # Absolute mode places the dub at the source's own level, which on a
        # quiet title is below audibility. Lift the whole set until its median
        # line can be heard; never push it down.
        offset = max(0.0, LEVEL_AUDIBLE_MEDIAN_DBFS - target_median)
    floor = target_median - LEVEL_FLOOR_BELOW_MEDIAN_DB
    gains: list[float] = []
    for index, (value, target) in enumerate(zip(measured, targets)):
        if value <= SILENT_CLIP_DBFS:
            gains.append(0.0)
            continue
        # Keep the shape of the source's dynamics, at a reduced ratio.
        shaped = target_median + LEVEL_DYNAMIC_RANGE_RATIO * (float(target) - target_median)
        desired = max(shaped, floor) + offset
        boost_cap = LEVEL_MATCH_MAX_BOOST_DB
        if noise_floors is not None and backgrounds is not None:
            boost_cap = noise_floor_boost_cap(noise_floors[index], backgrounds[index])
        gains.append(float(np.clip(desired - value, -LEVEL_MATCH_MAX_CUT_DB, boost_cap)))
    return gains


def apply_level_gain(clip: np.ndarray, gain_db: float) -> np.ndarray:
    """Scale a clip by ``gain_db``, backing off if that would clip the output."""
    if abs(gain_db) < 0.05:
        return clip
    scaled = clip * (10.0 ** (gain_db / 20.0))
    peak = float(np.max(np.abs(scaled))) if scaled.size else 0.0
    if peak > LEVEL_MATCH_PEAK_CEILING:
        scaled *= LEVEL_MATCH_PEAK_CEILING / peak
    return scaled.astype(np.float32, copy=False)


_PROMPT_SHAPE_ERROR_MARKERS = (
    # mel spectrogram too short for the convolution kernels
    "input size per channel",
    # the prompt degenerated on resampling (zero-dimensional waveform)
    "must have only one dimension",
    "input waveform",
)


def _short_prompt_runtime_error(exc: BaseException) -> bool:
    """Whether a failed inference looks like a prompt that was too short.

    These all mean the same thing operationally -- widen the reference and try
    again -- and letting an unmatched variant escape aborts the whole title
    after hours of work, which is what happened on hnvr-174 part1 at line 134
    of 352.
    """
    message = str(exc).lower()
    return any(marker in message for marker in _PROMPT_SHAPE_ERROR_MARKERS)


def _fit_to_slot(clip: np.ndarray, text: str, language: str, start: float,
                 end: float, overflow_end: float, fit_duration: bool) -> np.ndarray:
    """Place a generated clip in its slot without crushing it.

    Aim at what a person would take to say the line, not at what the model
    returned: compressing a dragged reading back towards natural is a repair,
    while compressing a natural reading into a slot sized for the *source*
    language is what made lines unintelligible. Never stretch to fill either --
    finishing early and leaving silence costs nothing.

    But `natural` is an estimate of reading time, not a licence to crush the
    rendition down to it. On ipvr-385 the estimate for a one-character line is
    0.30s, the model rendered it as a 1.7-2.9s delivery, and the fit squeezed it
    by up to 9.6x -- 97% of the title over 1.15x, 39% over 3x, which is the
    "flashes past, cannot make out a word" report. MAX_HARD_COMPRESSION was
    added for exactly this and then never wired to anything. Give the clip the
    room it needs; the gap behind it is usually enormous (84s on line 36).
    """
    if not fit_duration:
        return clip
    generated = clip.size / SAMPLE_RATE
    natural = natural_reading_seconds(text, normalize_language(language))
    room = max(0.0, overflow_end - start)
    allowed = max(room, end - start)
    wanted = min(natural, allowed)
    least_crushed = generated / MAX_HARD_COMPRESSION
    return _fit_audio(clip, SAMPLE_RATE, min(allowed, max(wanted, least_crushed)))


def _synth_line(idx: int, total: int, start: float, end: float, note: str,
                generated_duration: float, clip: np.ndarray, fit_duration: bool) -> str:
    final = clip.size / SAMPLE_RATE
    overflow = final - (end - start)
    return (
        f"[indextts-v2] {idx}/{total} {start:.2f}-{end:.2f}s {note} "
        f"model={generated_duration:.2f}s -> {final:.2f}s (slot {(end - start):.2f}s"
        + (f", +{overflow:.2f}s into the gap" if fit_duration and overflow > 0.02 else "")
        + ")"
    )


LINE_CACHE_NAME = "line_cache.json"


def _file_digest(path: str | Path) -> str:
    import hashlib

    try:
        return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


def line_cache_key(kwargs: dict) -> str:
    """Fingerprint of everything handed to ``model.infer`` for one line.

    Prompts are hashed by content rather than by path: the reference WAVs are
    re-cut from the same audio on every run, so the path is stable while the
    bytes are what actually decide the result. ``output_path`` and ``verbose``
    are excluded -- they do not affect what comes out.
    """
    import hashlib

    parts = [
        f"v1|{kwargs.get('text') or ''}|{kwargs.get('lang') or ''}|"
        f"{float(kwargs.get('duration_factor') or 0.0):.4f}|"
        f"{float(kwargs.get('emo_alpha') or 0.0):.4f}"
    ]
    for field in ("spk_audio_prompt", "emo_audio_prompt"):
        value = kwargs.get(field)
        parts.append(f"{field}={_file_digest(value) if value else ''}")
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


def load_line_cache(work_dir: Path) -> dict[str, str]:
    import json

    path = Path(work_dir) / LINE_CACHE_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in (data.get("lines") or {}).items() if isinstance(v, str)}


def save_line_cache(work_dir: Path, lines: dict[str, str]) -> None:
    import json

    try:
        (Path(work_dir) / LINE_CACHE_NAME).write_text(
            json.dumps({"version": 1, "lines": lines}, indent=2), encoding="utf-8"
        )
    except OSError:
        pass


def prune_unreferenced_clips(work_dir: Path, keep: set[str], log: LogCallback) -> None:
    """Drop generated clips no line points at any more.

    Merging lines away during proofreading orphans their clips; without this
    the directory only ever grows, and clips left over from the old
    index-based naming would sit there forever.
    """
    removed = 0
    for path in Path(work_dir).glob("out_*.wav"):
        if path.name not in keep:
            path.unlink(missing_ok=True)
            removed += 1
    if removed:
        log(f"[indextts-v2] removed {removed} clip(s) no longer referenced by any line")


def _remove_legacy_manifest_references(work_dir: Path, log: LogCallback) -> None:
    """Remove obsolete global-longest fallback files from the old workflow."""
    removed = 0
    for pattern in ("speaker_ref_*.wav", "retry_*.wav"):
        for path in work_dir.glob(pattern):
            path.unlink(missing_ok=True)
            removed += 1
    if removed:
        log(f"[indextts-v2] removed {removed} obsolete speaker/longest reference WAV(s)")


ANCHOR_DEFAULT_SPEAKER = "SPEAKER_00"


def publish_anchor(clone_dir: Path, speaker: str, wav_path: str,
                   detail: str, spoken: str) -> Path:
    """Write the chosen timbre anchor out the way the refined-clone tab does.

    "Lock timbre (one anchor clip per speaker)" picked a clip and then left it
    inside indextts_v2_manifest/ under a sentence_ref_NNNNN.wav name
    indistinguishable from the hundred per-line references beside it -- ticking
    the box produced nothing anyone could listen to or argue with. The anchor
    decides the voice of every line, so it lands in the clone directory under
    the same ``<speaker>.basis.wav`` name the refined-clone tab saves a
    hand-picked basis to, with the same .txt and .meta.json beside it. Same
    place, same format, whether a person chose it or the ranking did.
    """
    import json
    import shutil

    safe = "".join(
        ch if ch.isalnum() or ch in "._-" else "_"
        for ch in (speaker or ANCHOR_DEFAULT_SPEAKER)
    )
    out = clone_dir / f"{safe}.basis.wav"
    try:
        clone_dir.mkdir(parents=True, exist_ok=True)
        if Path(wav_path).resolve() != out.resolve():
            shutil.copyfile(wav_path, out)
        (clone_dir / f"{safe}.basis.txt").write_text(spoken + "\n", encoding="utf-8")
        (clone_dir / f"{safe}.basis.meta.json").write_text(
            json.dumps(
                {"source": "auto-timbre-anchor", "speaker": speaker or ANCHOR_DEFAULT_SPEAKER,
                 "detail": detail, "basis_wav": out.name},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
    except OSError:
        return Path(wav_path)
    return out


def synthesize_manifest(model, source_audio: str | Path, segments: Iterable[dict],
                        output_path: str | Path, *, language: str,
                        fit_duration: bool = True, tempo_fit: str = "moderate", log: LogCallback = print,
                        stop_event=None, intermediate_dir: str | Path | None = None,
                        level_match: bool = True, timbre_anchor: bool = True,
                        speaker_refs: dict | None = None) -> str:
    """Synthesize manifest segments with IndexTTS references.

    ``speaker_refs`` maps a speaker to a basis WAV chosen by hand; those
    speakers take that voice and are not ranked for an anchor at all.

    With ``timbre_anchor`` (the default) each speaker's voice comes from one
    anchor sentence and the current sentence supplies only emotion; otherwise
    every sentence clones from itself and the split prompt is used only as a
    fallback for unusable references.
    """
    entries = [dict(s) for s in segments if (s.get("text") or "").strip()]
    total = max((float(s["end"]) for s in entries), default=0.0)
    timeline = np.zeros(max(1, int(round((total + 1.0) * SAMPLE_RATE))), dtype=np.float32)
    clips: list[np.ndarray] = []
    skipped: list[int] = []
    cached: dict[str, str] = {}
    fresh_cache: dict[str, str] = {}
    keep_clips: set[str] = set()
    reused = 0
    work_dir = _persistent_intermediate_dir(
        source_audio, output_path, intermediate_dir, MANIFEST_INTERMEDIATE_DIR_NAME
    )
    log(f"[indextts-v2] intermediate WAV directory: {work_dir}")
    _remove_legacy_manifest_references(work_dir, log)
    cached = load_line_cache(work_dir)
    source_wav, source_sr = _read_audio(source_audio)
    profiles = _reference_profiles(source_wav, source_sr, entries)
    # Where each line may run to if the model overshoots its slot: the next
    # line's start, less a guard. Overrunning into silence is always better
    # than crushing the delivery to fit.
    overflow_end = [
        float(entries[index + 1]["start"]) - SLOT_OVERFLOW_GUARD_SECONDS
        if index + 1 < len(entries) else float("inf")
        for index in range(len(entries))
    ]
    # A basis chosen by hand outranks anything the ranking can find. The
    # refined-clone tab's whole step 2 -- audition the candidates, pick a
    # speaker's voice -- wrote its choice to manifest["speakers"][spk]
    # ["ref_audio"] and nothing here ever read it, so every title was cloned
    # from the automatic anchor no matter what the user picked. On
    # 3dsvr-1911 part 2 that anchor was a line from the middle of a sex
    # scene: 10.5dB above the title's median and 88% periodic, which is what
    # the score rewards, and it set the voice of all 247 lines.
    chosen: dict[str, str] = {}
    for key, path in (speaker_refs or {}).items():
        if path and Path(path).is_file():
            chosen[str(key)] = str(path)
    anchors: dict[str, int] = (
        select_speaker_timbre_anchors(entries, profiles) if timbre_anchor else {}
    )
    anchor_paths: dict[str, str] = dict(chosen)
    for key in sorted(chosen):
        anchors.pop(key, None)
        log(f"[indextts-v2] timbre anchor {key or '(single speaker)'}: "
            f"chosen basis -> {Path(chosen[key]).name}")
    for key, index in sorted(anchors.items()):
        anchor_segment, anchor_profile = entries[index], profiles[index]
        anchor_paths[key] = extract_sentence_reference(
            source_audio, anchor_segment, work_dir,
            source_wav=source_wav, source_sr=source_sr,
        )
        detail = (
            f"segment {index + 1}/{len(entries)} "
            f"{float(anchor_segment['start']):.2f}-{float(anchor_segment['end']):.2f}s "
            f"({anchor_profile['duration']:.2f}s, "
            f"active={anchor_profile['active_seconds']:.2f}s, "
            f"active-rms={anchor_profile['active_rms_dbfs']:.1f}dBFS, "
            f"periodic={anchor_profile['periodic_ratio']:.0%})"
        )
        spoken = str(
            anchor_segment.get("src_text") or anchor_segment.get("text") or ""
        ).strip()
        published = publish_anchor(work_dir.parent, key, anchor_paths[key], detail, spoken)
        log(
            f"[indextts-v2] timbre anchor {key or '(single speaker)'}: {detail} "
            f"-> {published.name}"
        )
    if timbre_anchor:
        # Speakers with no line long enough to anchor on: stitch neighbouring
        # ones instead of falling back to per-sentence cloning.
        median_active_rms = _median_active_rms(profiles)
        # Chosen bases are excluded as well as ranked ones: a speaker who
        # already has the voice a person picked must not have one stitched
        # over the top of it.
        unanchored = {str(item.get("speaker") or "") for item in entries}
        for key in sorted(unanchored - set(anchors) - set(chosen)):
            pieces = select_stitch_pieces(entries, profiles, key, median_active_rms)
            if not pieces:
                log(
                    f"[indextts-v2] {key or '(single speaker)'}: no line long enough to "
                    "anchor on and too little non-whisper material to stitch one; "
                    "these lines keep cloning from themselves"
                )
                continue
            safe = re.sub(r"[^A-Za-z0-9_-]", "_", key) or "single"
            path = work_dir / f"anchor_stitched_{safe}.wav"
            anchor_paths[key] = write_stitched_anchor(
                source_wav, source_sr, entries, pieces, path
            )
            spanned = sum(float(profiles[i]["duration"]) for i in pieces)
            detail = (
                f"stitched {len(pieces)} line(s) -> {spanned:.2f}s from segments "
                f"{', '.join(str(i + 1) for i in pieces)} "
                f"(longest single line {max(float(profiles[i]['duration']) for i in pieces):.2f}s)"
            )
            published = publish_anchor(work_dir.parent, key, anchor_paths[key], detail, "")
            log(
                f"[indextts-v2] timbre anchor {key or '(single speaker)'}: {detail} "
                f"-> {published.name}"
            )
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
        # Named for the segment, not the loop position. Proofreading can merge
        # lines away, which shifts every later index by one and would silently
        # hand a cached clip to a different sentence.
        seg_id = int(segment.get("id", idx))
        out_path = work_dir / f"out_{seg_id:05d}.wav"
        kwargs = {"spk_audio_prompt": ref_path, "text": text,
                  "lang": normalize_language(language).upper(),
                  "output_path": str(out_path), "verbose": False}
        profile = profiles[idx - 1]
        speaker_key = str(segment.get("speaker") or "")
        anchor_path = anchor_paths.get(speaker_key)
        if anchors.get(speaker_key) == idx - 1:
            anchor_path = None  # the anchor sentence still clones from itself
        timbre_fallback_index = None
        if anchor_path is None and reference_needs_timbre_fallback(profile):
            timbre_fallback_index = select_nearest_stable_reference(entries, profiles, idx - 1)
        timbre_path = anchor_path
        if timbre_path is None and timbre_fallback_index is not None:
            timbre_path = extract_sentence_reference(
                source_audio, entries[timbre_fallback_index], work_dir,
                source_wav=source_wav, source_sr=source_sr,
            )
        if timbre_path is not None:
            kwargs["spk_audio_prompt"] = timbre_path
            kwargs["emo_audio_prompt"] = ref_path
            # An unvoiced sentence -- a breath, or a moan bandit routed to the
            # sfx stem -- is a bad emotion prompt as much as a bad timbre one,
            # so it keeps its downweight whether or not an anchor is in use.
            if float(profile["periodic_ratio"]) < UNVOICED_REFERENCE_MAX_PERIODIC_RATIO:
                kwargs["emo_alpha"] = UNVOICED_EMOTION_ALPHA
            elif anchor_path is not None:
                kwargs["emo_alpha"] = SPEAKER_ANCHOR_EMOTION_ALPHA
            else:
                kwargs["emo_alpha"] = 1.0
        if fit_duration:
            kwargs["duration_factor"] = duration_factor_for_slot(text, end - start, normalize_language(language), tempo_fit)
        # The per-sentence reference detail is held back and folded into the
        # single duration line below: two lines per sentence plus the vendor's
        # own chatter filled the GUI log window on an 80-line title.
        if timbre_path is not None:
            prompt_note = (
                f"{'anchor' if anchor_path is not None else 'weak-ref'}"
                f" emo-alpha={kwargs['emo_alpha']:.2f}"
            )
        else:
            prompt_note = "self-ref"
        # Everything above is cheap: cutting reference WAVs and picking prompts.
        # model.infer is the expensive step, and it is the only one worth
        # reusing -- level matching and time fitting depend on whole-title
        # statistics and are recomputed every run regardless.
        cache_key = line_cache_key(kwargs)
        cache_id = str(seg_id)
        keep_clips.add(out_path.name)
        if cached.get(cache_id) == cache_key and out_path.is_file():
            reused += 1
            fresh_cache[cache_id] = cache_key
            clip, clip_sr = _read_audio(out_path)
            generated_duration = clip.size / float(clip_sr)
            if clip_sr != SAMPLE_RATE:
                import torchaudio, torch
                clip = torchaudio.functional.resample(
                    torch.from_numpy(clip), clip_sr, SAMPLE_RATE
                ).numpy()
            clip = _fit_to_slot(
                clip, text, language, start, end, overflow_end[idx - 1], fit_duration
            )
            log(_synth_line(idx, len(entries), start, end, prompt_note + " cached",
                            generated_duration, clip, fit_duration))
            clips.append(clip)
            continue
        out_path.unlink(missing_ok=True)
        retry_reason = ""
        try:
            result = model.infer(**kwargs)
        except Exception as exc:
            # Not only RuntimeError: the degenerate-prompt failure surfaces from
            # the resampler as a ValueError, and catching just RuntimeError let
            # it abort a title 134 lines into 352.
            if not _short_prompt_runtime_error(exc):
                raise
            result = None
            retry_reason = f"{type(exc).__name__}: {exc}"
        generated = Path(result or out_path)
        if not generated.is_file():
            retry_reason = retry_reason or "IndexTTS returned no WAV"
        if retry_reason:
            if timbre_path is not None:
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
            try:
                result = model.infer(**kwargs)
            except Exception as exc:
                if not _short_prompt_runtime_error(exc):
                    raise
                result = None
        generated = Path(result or out_path)
        if not generated.is_file():
            # One unusable line must not throw away the whole title. Leave its
            # slot silent, keep going, and report the count at the end.
            skipped.append(idx)
            log(
                f"[indextts-v2] {idx}/{len(entries)} {start:.2f}-{end:.2f}s skipped: "
                f"IndexTTS produced nothing usable for this reference"
            )
            clips.append(np.zeros(1, dtype=np.float32))
            continue
        clip, clip_sr = _read_audio(generated)
        generated_duration = clip.size / float(clip_sr)
        if clip_sr != SAMPLE_RATE:
            import torchaudio, torch
            clip = torchaudio.functional.resample(torch.from_numpy(clip), clip_sr, SAMPLE_RATE).numpy()
        clip = _fit_to_slot(
            clip, text, language, start, end, overflow_end[idx - 1], fit_duration
        )
        fresh_cache[cache_id] = cache_key
        log(_synth_line(idx, len(entries), start, end, prompt_note,
                        generated_duration, clip, fit_duration))
        clips.append(clip)

    save_line_cache(work_dir, fresh_cache)
    prune_unreferenced_clips(work_dir, keep_clips, log)
    if reused:
        log(
            f"[indextts-v2] reused {reused} of {len(entries)} line(s) unchanged since "
            "the last run; only the rest were synthesized"
        )
    if skipped:
        log(
            f"[indextts-v2] {len(skipped)} of {len(entries)} line(s) left silent "
            f"(unusable reference): {', '.join(str(i) for i in skipped[:10])}"
            + (" ..." if len(skipped) > 10 else "")
        )

    gains = [0.0] * len(clips)
    if level_match and clips:
        measured = [
            float(analyze_reference_quality(clip, SAMPLE_RATE)["active_rms_dbfs"])
            for clip in clips
        ]
        targets, mode = segment_level_targets(entries, profiles)
        targets, speaker_shifts = match_speaker_medians(entries, targets)
        if speaker_shifts:
            log(
                "[level] speaker balance: "
                + ", ".join(
                    f"{name} {shift:+.1f} dB"
                    for name, shift in sorted(speaker_shifts.items())
                )
            )
        noise_floors = [clip_noise_floor_dbfs(clip, SAMPLE_RATE) for clip in clips]
        backgrounds = [float(profile["active_rms_dbfs"]) for profile in profiles]
        gains = compute_level_gains(measured, targets, mode, noise_floors, backgrounds)
        applied = np.asarray(gains, dtype=np.float32)
        # Two different reasons a clip stops short, and they need different
        # fixes: its own hiss would escape the film's bed (lower the target or
        # accept more hiss), or it simply ran into the flat backstop (raise
        # LEVEL_MATCH_MAX_BOOST_DB). Reporting them as one number blamed the
        # noise floor for 72 of 100 clips on hnvr-174 that were only hitting
        # the backstop, with the floor allowing a median +30.6 dB.
        floor_limited = backstop_limited = 0
        for gain, floor, bed in zip(gains, noise_floors, backgrounds):
            if gain < noise_floor_boost_cap(floor, bed) - 0.05:
                continue
            allowed = (
                float(bed) + NOISE_FLOOR_MAX_ABOVE_BACKGROUND_DB - float(floor)
                if floor is not None and bed is not None
                else None
            )
            if allowed is not None and allowed < LEVEL_MATCH_MAX_BOOST_DB:
                floor_limited += 1
            else:
                backstop_limited += 1
        log(
            f"[level] matching {len(clips)} clip(s) to the source in {mode} mode: "
            f"gain mean {applied.mean():+.1f} dB, range {applied.min():+.1f}..{applied.max():+.1f} dB, "
            f"|gain|>6dB on {int(np.count_nonzero(np.abs(applied) > 6.0))} clip(s); "
            f"{floor_limited} stopped by their own noise floor, "
            f"{backstop_limited} by the {LEVEL_MATCH_MAX_BOOST_DB:.0f} dB boost limit"
        )
        for (idx, segment), value, target, gain in zip(
            enumerate(entries, 1), measured, targets, gains
        ):
            if abs(gain) >= 6.0:
                log(
                    f"[level] {idx}/{len(entries)} {float(segment['start']):.2f}s "
                    f"src={target:.1f} dBFS tts={value:.1f} dBFS -> {gain:+.1f} dB"
                )

    for segment, clip, gain in zip(entries, clips, gains):
        clip = apply_level_gain(clip, gain)
        lo = max(0, int(round(float(segment["start"]) * SAMPLE_RATE)))
        hi = min(timeline.size, lo + clip.size)
        if hi > lo:
            timeline[lo:hi] += clip[:hi - lo]
    _write_audio(output_path, timeline, SAMPLE_RATE)
    return str(output_path)
