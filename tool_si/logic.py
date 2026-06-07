from __future__ import annotations

import importlib.util
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave
import warnings
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable

import numpy as np


MODEL_REPO_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
MODEL_DIR_NAME = "Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_TTS_BATCH_SIZE = 4
MAX_TTS_BATCH_SIZE = 16
TTS_BATCH_SIZE_ENV = "VRTB_TTS_BATCH_SIZE"
TTS_TIME_FIT_MODE_ENV = "VRTB_TTS_TIME_FIT_MODE"
DEFAULT_TTS_BATCH_TOKEN_SPREAD = 1.5
TTS_BATCH_TOKEN_SPREAD_ENV = "VRTB_TTS_BATCH_TOKEN_SPREAD"
# When 1/true (default), the sub-talker (code predictor) runs argmax instead of sampling.
# It runs 31 inner steps for every main codec token, so dropping sampling there is a
# free 5-15% throughput win with no audible main-talker quality change. Set to 0 to
# restore the upstream stochastic sub-talker if you need its tiny prosody/timbre detail.
TTS_SUBTALKER_GREEDY_ENV = "VRTB_TTS_SUBTALKER_GREEDY"
SUPPORTED_LANGUAGES = (
    "Chinese",
    "English",
    "German",
    "French",
    "Italian",
    "Portuguese",
    "Spanish",
    "Japanese",
    "Korean",
    "Russian",
)

LANGUAGE_CODES = {
    "Chinese": "chinese",
    "English": "english",
    "German": "german",
    "French": "french",
    "Italian": "italian",
    "Portuguese": "portuguese",
    "Spanish": "spanish",
    "Japanese": "japanese",
    "Korean": "korean",
    "Russian": "russian",
}

ALL_SPEAKERS = (
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
)

LANGUAGE_NATIVE_SPEAKERS = {
    "Chinese": ("Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"),
    "English": ("Ryan", "Aiden"),
    "Japanese": ("Ono_Anna",),
    "Korean": ("Sohee",),
}

UI_LANGUAGE_TO_TTS = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
}

LogCallback = Callable[[str], None]


@dataclass(frozen=True)
class SubtitleEntry:
    index: int
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def get_model_dir(models_root: str | os.PathLike[str]) -> str:
    return str(Path(models_root) / MODEL_DIR_NAME)


def default_tts_language(ui_language: str | None) -> str:
    return UI_LANGUAGE_TO_TTS.get((ui_language or "").lower(), "English")


def speakers_for_language(language: str) -> tuple[str, ...]:
    return LANGUAGE_NATIVE_SPEAKERS.get(language, ALL_SPEAKERS)


def default_speaker_for_language(language: str) -> str:
    speakers = speakers_for_language(language)
    return speakers[0] if speakers else ALL_SPEAKERS[0]


def check_model_files(models_root: str | os.PathLike[str]) -> bool:
    model_dir = Path(get_model_dir(models_root))
    if not model_dir.exists():
        return False
    has_config = (model_dir / "config.json").exists()
    has_main_weights = any(
        path.is_file()
        and (
            path.name == "model.safetensors"
            or path.name == "pytorch_model.bin"
            or path.name.endswith(".safetensors.index.json")
            or re.match(r"model-\d+-of-\d+\.safetensors$", path.name)
        )
        for path in model_dir.iterdir()
    )
    has_speech_tokenizer = (model_dir / "speech_tokenizer" / "model.safetensors").exists()
    return has_config and has_main_weights and has_speech_tokenizer


def download_model(models_root: str | os.PathLike[str], log_callback: LogCallback = print) -> bool:
    model_dir = get_model_dir(models_root)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(Path(models_root) / ".hf_home"))
    try:
        from huggingface_hub import snapshot_download
        import huggingface_hub.constants
    except ImportError:
        log_callback("Error: huggingface_hub package is not installed.")
        return False

    try:
        log_callback(f"HuggingFace Endpoint: {huggingface_hub.constants.ENDPOINT}")
        log_callback(f"Downloading {MODEL_REPO_ID} to {model_dir}")
        snapshot_download(repo_id=MODEL_REPO_ID, local_dir=model_dir)
        log_callback("Model download finished.")
        return check_model_files(models_root)
    except TypeError:
        try:
            snapshot_download(repo_id=MODEL_REPO_ID, local_dir=model_dir)
            log_callback("Model download finished.")
            return check_model_files(models_root)
        except Exception as exc:
            log_callback(f"Download failed: {exc}")
            return False
    except Exception as exc:
        log_callback(f"Download failed: {exc}")
        return False


def _read_text_with_fallback(path: str | os.PathLike[str]) -> str:
    data = Path(path).read_bytes()
    encodings = ("utf-8-sig", "utf-8", "gb18030", "shift_jis", "cp932", "latin-1")
    last_error: UnicodeDecodeError | None = None
    for encoding in encodings:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error:
        raise last_error
    return data.decode("utf-8-sig")


_TIMECODE_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})"
)
_TAG_RE = re.compile(r"<[^>]+>")
_ASS_OVERRIDE_RE = re.compile(r"\{\\[^}]*\}")
_JAPANESE_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7af]")
_HAN_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")


def _parse_timecode(value: str) -> float:
    time_part, ms_part = value.replace(",", ".").split(".", 1)
    hours, minutes, seconds = (int(part) for part in time_part.split(":"))
    milliseconds = int(ms_part.ljust(3, "0")[:3])
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0


def _clean_subtitle_line(line: str) -> str:
    line = line.replace("\\N", " ").strip()
    line = _ASS_OVERRIDE_RE.sub("", line)
    line = _TAG_RE.sub("", line)
    return re.sub(r"\s+", " ", line).strip()


def _select_subtitle_text_line(text_lines: list[str], language: str | None = None) -> str:
    cleaned_lines = [_clean_subtitle_line(line) for line in text_lines]
    cleaned_lines = [line for line in cleaned_lines if line]
    if not cleaned_lines:
        return ""

    language_key = (language or "").lower()
    if language_key == "japanese":
        return next((line for line in cleaned_lines if _JAPANESE_KANA_RE.search(line)), cleaned_lines[0])
    if language_key == "korean":
        return next((line for line in cleaned_lines if _HANGUL_RE.search(line)), cleaned_lines[0])
    if language_key == "chinese":
        return next(
            (
                line
                for line in cleaned_lines
                if _HAN_RE.search(line) and not _JAPANESE_KANA_RE.search(line) and not _HANGUL_RE.search(line)
            ),
            cleaned_lines[0],
        )
    if language_key == "english":
        return next(
            (
                line
                for line in cleaned_lines
                if _LATIN_WORD_RE.search(line) and not _CJK_CHAR_RE.search(line)
            ),
            cleaned_lines[0],
        )
    return cleaned_lines[0]


def parse_srt(path: str | os.PathLike[str], language: str | None = None) -> list[SubtitleEntry]:
    text = _read_text_with_fallback(path)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n+", text.strip())
    entries: list[SubtitleEntry] = []

    for block_index, block in enumerate(blocks, 1):
        lines = [line.strip("\ufeff") for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        time_line_index = next((i for i, line in enumerate(lines) if "-->" in line), -1)
        if time_line_index < 0:
            continue
        match = _TIMECODE_RE.search(lines[time_line_index])
        if not match:
            continue
        start = _parse_timecode(match.group("start"))
        end = _parse_timecode(match.group("end"))
        if end <= start:
            continue

        text_lines = lines[time_line_index + 1 :]
        selected_line = _select_subtitle_text_line(text_lines, language)
        if not selected_line:
            continue
        entries.append(SubtitleEntry(index=block_index, start=start, end=end, text=selected_line))

    return entries


def default_output_path(srt_path: str | os.PathLike[str]) -> str:
    path = Path(srt_path)
    return str(path.with_name(f"{path.stem}.si.wav"))


def collect_paired_srt_tasks(base_dir: str | os.PathLike[str]) -> list[Path]:
    root_path = Path(base_dir)
    seen: set[Path] = set()
    tasks: list[Path] = []
    for video in root_path.rglob("*"):
        if not video.is_file() or video.suffix.lower() not in {".mp4", ".mkv"}:
            continue
        srt_path = video.with_suffix(".srt")
        if not srt_path.is_file():
            continue
        resolved = srt_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        tasks.append(srt_path)
    return sorted(tasks, key=lambda path: str(path).lower())


def _build_startupinfo():
    if sys.platform != "win32":
        return None
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0
    return startupinfo


def _to_mono_float32(audio) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    arr = np.asarray(audio)
    source_dtype = arr.dtype
    arr = arr.astype(np.float32, copy=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if np.issubdtype(source_dtype, np.integer):
        max_value = max(1, np.iinfo(source_dtype).max)
        arr = arr / float(max_value)
    elif arr.size and float(np.nanmax(np.abs(arr))) > 2.0:
        arr = arr / 32768.0
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(arr, -1.0, 1.0).astype(np.float32, copy=False)


def write_wav_mono(path: str | os.PathLike[str], audio: np.ndarray, sample_rate: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mono = _to_mono_float32(audio)
    pcm = np.clip(mono, -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm_i16.tobytes())


def read_wav_mono(path: str | os.PathLike[str]) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if sample_width != 2:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")
    pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return pcm.astype(np.float32, copy=False), sample_rate


def _atempo_chain(factor: float) -> str:
    if factor <= 0:
        return "atempo=1.0"
    parts: list[float] = []
    remaining = factor
    while remaining > 2.0:
        parts.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        parts.append(0.5)
        remaining /= 0.5
    parts.append(remaining)
    return ",".join(f"atempo={part:.6f}" for part in parts)


def _speed_up_with_ffmpeg(audio: np.ndarray, sample_rate: int, factor: float) -> np.ndarray:
    if not shutil.which("ffmpeg"):
        return audio
    with tempfile.TemporaryDirectory(prefix="si_tts_") as tmp_dir:
        src = Path(tmp_dir) / "src.wav"
        dst = Path(tmp_dir) / "dst.wav"
        write_wav_mono(src, audio, sample_rate)
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-filter:a",
            _atempo_chain(factor),
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(dst),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            startupinfo=_build_startupinfo(),
        )
        if result.returncode != 0 or not dst.exists():
            return audio
        converted, converted_rate = read_wav_mono(dst)
        if converted_rate != sample_rate:
            return audio
        return converted


def _time_fit_mode() -> str:
    mode = os.environ.get(TTS_TIME_FIT_MODE_ENV, "fast").strip().lower()
    if mode in {"ffmpeg", "atempo"}:
        return "ffmpeg"
    return "fast"


def _speed_up_fast_resample(audio: np.ndarray, target_samples: int) -> np.ndarray:
    if target_samples <= 0:
        return np.zeros(1, dtype=np.float32)
    if audio.size <= 1:
        return np.resize(audio, target_samples).astype(np.float32, copy=False)
    source_positions = np.linspace(0.0, 1.0, num=audio.size, endpoint=True, dtype=np.float32)
    target_positions = np.linspace(0.0, 1.0, num=target_samples, endpoint=True, dtype=np.float32)
    return np.interp(target_positions, source_positions, audio).astype(np.float32, copy=False)


def _speed_up_in_memory(audio: np.ndarray, target_samples: int, factor: float) -> np.ndarray:
    if audio.size < 2048:
        return _speed_up_fast_resample(audio, target_samples)
    try:
        import librosa

        stretched = librosa.effects.time_stretch(audio.astype(np.float32, copy=False), rate=float(factor))
        return np.asarray(stretched, dtype=np.float32)
    except Exception:
        return _speed_up_fast_resample(audio, target_samples)


def fit_audio_to_duration(audio: np.ndarray, sample_rate: int, target_duration: float) -> np.ndarray:
    target_samples = max(1, int(round(target_duration * sample_rate)))
    mono = _to_mono_float32(audio)
    if mono.size == 0:
        return np.zeros(target_samples, dtype=np.float32)

    if mono.size > target_samples:
        factor = mono.size / float(target_samples)
        # Pitch-preserving stretch (librosa phase vocoder or ffmpeg atempo) is ~100ms / 5s
        # of audio on CPU and serialises the post-generation tail of every batch. Up to
        # ~1.15× speedup the pitch drift of a plain linear resample is <~ a quarter-tone
        # and not noticeable in subtitle TTS, so reserve the expensive path for bigger
        # stretches where the pitch artefact would actually be audible.
        if factor > 1.15:
            if _time_fit_mode() == "ffmpeg":
                mono = _speed_up_with_ffmpeg(mono, sample_rate, factor)
            else:
                mono = _speed_up_in_memory(mono, target_samples, factor)
        elif factor > 1.0:
            mono = _speed_up_fast_resample(mono, target_samples)

    if mono.size > target_samples:
        mono = mono[:target_samples]
    elif mono.size < target_samples:
        mono = np.pad(mono, (0, target_samples - mono.size))
    return np.clip(mono, -1.0, 1.0).astype(np.float32, copy=False)


def _load_tts_model(model_dir: str, log_callback: LogCallback):
    try:
        import torch
        from tool_si._vendor.qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    except Exception as exc:
        raise RuntimeError(
            "Failed to import vendored Qwen3-TTS runtime. Check runtime dependencies "
            "(torch, transformers, accelerate, librosa, soundfile, torchaudio). "
            f"Original error: {exc}"
        ) from exc

    has_cuda = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    dtype = torch.bfloat16 if has_cuda else torch.float32
    device_map = "cuda:0" if has_cuda else "cpu"

    attempts = []
    if has_cuda and importlib.util.find_spec("flash_attn") is not None:
        attempts.append({"device_map": device_map, "dtype": dtype, "attn_implementation": "flash_attention_2"})
    elif has_cuda:
        log_callback("FlashAttention2 is not installed; using PyTorch SDPA attention.")
    if has_cuda:
        attempts.append({"device_map": device_map, "dtype": dtype, "attn_implementation": "sdpa"})
    attempts.append({"device_map": device_map, "dtype": dtype, "attn_implementation": "eager"})
    attempts.append({"device_map": device_map, "dtype": dtype})
    attempts.append({"device_map": device_map, "torch_dtype": dtype})
    attempts.append({})

    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            log_callback(f"Loading Qwen3-TTS model from {model_dir}")
            return Qwen3TTSModel.from_pretrained(model_dir, **kwargs)
        except Exception as exc:
            last_error = exc
            if kwargs.get("attn_implementation") == "flash_attention_2":
                log_callback(f"FlashAttention load failed, retrying without it: {exc}")
                continue
            if kwargs.get("attn_implementation") == "sdpa":
                log_callback(f"SDPA attention load failed, retrying with eager attention: {exc}")
                continue
    raise RuntimeError(f"Failed to load Qwen3-TTS model: {last_error}") from last_error


_CJK_CHAR_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")


def _speech_token_floor(text: str) -> int:
    cjk_count = len(_CJK_CHAR_RE.findall(text or ""))
    latin_word_count = len(_LATIN_WORD_RE.findall(text or ""))
    return int(math.ceil(cjk_count * 2.2 + latin_word_count * 4.5)) + 3


def _max_new_tokens_for_duration(duration: float, text: str = "") -> int:
    # 12Hz model decodes one codec token to about 80ms of audio. The duration
    # budget keeps generation bounded; the text floor reduces accidental truncation.
    duration_budget = int(math.ceil(max(0.1, duration) * 12.0)) + 2
    text_budget = _speech_token_floor(text)
    return max(10, min(1024, max(duration_budget, text_budget)))


def resolve_tts_batch_size() -> int:
    raw_value = os.environ.get(TTS_BATCH_SIZE_ENV, str(DEFAULT_TTS_BATCH_SIZE))
    try:
        batch_size = int(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_TTS_BATCH_SIZE
    return max(1, min(MAX_TTS_BATCH_SIZE, batch_size))


def resolve_tts_batch_token_spread() -> float:
    raw_value = os.environ.get(TTS_BATCH_TOKEN_SPREAD_ENV, str(DEFAULT_TTS_BATCH_TOKEN_SPREAD))
    try:
        spread = float(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_TTS_BATCH_TOKEN_SPREAD
    return max(1.0, min(10.0, spread))


def _subtalker_greedy_enabled() -> bool:
    raw_value = os.environ.get(TTS_SUBTALKER_GREEDY_ENV, "1").strip().lower()
    return raw_value not in {"0", "false", "no", "off"}


def _subtalker_generate_kwargs() -> dict:
    if not _subtalker_greedy_enabled():
        return {}
    return {
        "subtalker_dosample": False,
        "subtalker_top_k": 1,
        "subtalker_top_p": 1.0,
        "subtalker_temperature": 1.0,
    }


def _iter_tts_batches(
    entries: list[SubtitleEntry],
    batch_size: int,
    preserve_order: bool = False,
) -> list[list[SubtitleEntry]]:
    if batch_size <= 1:
        return [[entry] for entry in entries]

    if preserve_order:
        ordered = list(entries)
    else:
        # Sort by token budget before forming batches. Batched generation runs every member
        # of a batch out to `max(budget for entry in batch)` autoregressive steps, so mixing
        # a 30-token short line with a 250-token long line in one batch wastes ~8x decode on
        # the short one. For full-file conversion, placement is restored via `entry.start`.
        ordered = sorted(
            entries,
            key=lambda entry: _max_new_tokens_for_duration(entry.duration, entry.text),
        )

    token_spread = resolve_tts_batch_token_spread()
    batches: list[list[SubtitleEntry]] = []
    current: list[SubtitleEntry] = []
    current_min_tokens = 0
    current_max_tokens = 0

    for entry in ordered:
        token_budget = _max_new_tokens_for_duration(entry.duration, entry.text)
        if not current:
            current = [entry]
            current_min_tokens = current_max_tokens = token_budget
            continue

        next_min = min(current_min_tokens, token_budget)
        next_max = max(current_max_tokens, token_budget)
        too_many_items = len(current) >= batch_size
        too_much_spread = next_max > max(1, next_min) * token_spread
        if too_many_items or too_much_spread:
            batches.append(current)
            current = [entry]
            current_min_tokens = current_max_tokens = token_budget
        else:
            current.append(entry)
            current_min_tokens = next_min
            current_max_tokens = next_max

    if current:
        batches.append(current)
    return batches


def _generate_audio(
    tts_model,
    text: str,
    speaker: str,
    language: str,
    target_duration: float,
    log_callback: LogCallback,
) -> tuple[np.ndarray, int]:
    max_new_tokens = _max_new_tokens_for_duration(target_duration, text)
    extra_kwargs = _subtalker_generate_kwargs()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Passing `repetition_penalty` with `inputs_embeds`.*",
                category=UserWarning,
            )
            wavs, sample_rate = tts_model.generate_custom_voice(
                text=text,
                speaker=speaker,
                language=language,
                max_new_tokens=max_new_tokens,
                **extra_kwargs,
            )
    except TypeError:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Passing `repetition_penalty` with `inputs_embeds`.*",
                category=UserWarning,
            )
            wavs, sample_rate = tts_model.generate_custom_voice(
                text,
                speaker=speaker,
                language=language,
                max_new_tokens=max_new_tokens,
            )
    if not wavs:
        log_callback("Warning: model returned empty audio for one subtitle entry.")
        return np.zeros(1, dtype=np.float32), int(sample_rate or 24000)
    return _to_mono_float32(wavs[0]), int(sample_rate)


def _generate_audio_batch(
    tts_model,
    entries: list[SubtitleEntry],
    speaker: str,
    language: str,
    log_callback: LogCallback,
) -> list[tuple[np.ndarray, int]]:
    if len(entries) == 1:
        entry = entries[0]
        return [_generate_audio(tts_model, entry.text, speaker, language, entry.duration, log_callback)]

    texts = [entry.text for entry in entries]
    max_new_tokens = max(_max_new_tokens_for_duration(entry.duration, entry.text) for entry in entries)
    extra_kwargs = _subtalker_generate_kwargs()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Passing `repetition_penalty` with `inputs_embeds`.*",
                category=UserWarning,
            )
            wavs, sample_rate = tts_model.generate_custom_voice(
                text=texts,
                speaker=speaker,
                language=language,
                max_new_tokens=max_new_tokens,
                **extra_kwargs,
            )
    except TypeError:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Passing `repetition_penalty` with `inputs_embeds`.*",
                category=UserWarning,
            )
            wavs, sample_rate = tts_model.generate_custom_voice(
                texts,
                speaker=speaker,
                language=language,
                max_new_tokens=max_new_tokens,
            )

    if len(wavs or []) != len(entries):
        raise RuntimeError(f"Batch TTS returned {len(wavs or [])} wavs for {len(entries)} subtitle entries.")

    return [(_to_mono_float32(wav), int(sample_rate or 24000)) for wav in wavs]


def subtitle_to_audio(
    srt_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None,
    language: str,
    speaker: str,
    models_root: str | os.PathLike[str],
    log_callback: LogCallback = print,
    stop_event: Event | None = None,
    tts_model=None,
    max_entries: int | None = None,
) -> str:
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")
    if speaker not in ALL_SPEAKERS:
        raise ValueError(f"Unsupported speaker: {speaker}")
    if stop_event and stop_event.is_set():
        raise RuntimeError("Stopped by user.")

    srt_path = Path(srt_path)
    if not srt_path.is_file():
        raise FileNotFoundError(f"SRT file not found: {srt_path}")
    output_path = Path(output_path or default_output_path(srt_path))
    all_entries = parse_srt(srt_path, language=language)
    if max_entries is not None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive or None.")
        entries = all_entries[:max_entries]
    else:
        entries = all_entries
    if not entries:
        raise ValueError("No valid subtitle entries found.")
    model_dir = get_model_dir(models_root)
    if not check_model_files(models_root):
        raise FileNotFoundError(f"Qwen3-TTS model files are missing: {model_dir}")

    if tts_model is None:
        tts_model = _load_tts_model(model_dir, log_callback)

    timeline: np.ndarray | None = None
    sample_rate = 24000
    total_duration = max(entry.end for entry in entries)
    batch_size = resolve_tts_batch_size()
    batches = _iter_tts_batches(entries, batch_size, preserve_order=max_entries is not None)
    if max_entries is not None and len(all_entries) > len(entries):
        log_callback(
            f"Parsed {len(all_entries)} subtitle entries; converting first {len(entries)} for test. "
            f"Output duration: {total_duration:.3f}s. TTS batch size: {batch_size} ({len(batches)} batches)."
        )
    else:
        log_callback(
            f"Parsed {len(entries)} subtitle entries. Output duration: {total_duration:.3f}s. "
            f"TTS batch size: {batch_size} ({len(batches)} batches)."
        )

    processed_count = 0
    for batch_index, batch in enumerate(batches, 1):
        if stop_event and stop_event.is_set():
            raise RuntimeError("Stopped by user.")
        if len(batch) > 1:
            log_callback(f"Batch [{batch_index}/{len(batches)}] generating {len(batch)} subtitle entries...")
        try:
            generated = _generate_audio_batch(tts_model, batch, speaker, language, log_callback)
        except Exception as exc:
            if len(batch) == 1:
                raise
            log_callback(f"Batch generation failed; retrying one by one: {exc}")
            generated = [
                _generate_audio(tts_model, entry.text, speaker, language, entry.duration, log_callback)
                for entry in batch
            ]

        for entry, (audio, sample_rate) in zip(batch, generated):
            if stop_event and stop_event.is_set():
                raise RuntimeError("Stopped by user.")
            processed_count += 1
            log_callback(f"[{processed_count}/{len(entries)}] {entry.start:.3f}-{entry.end:.3f}s {entry.text}")
            segment = fit_audio_to_duration(audio, sample_rate, entry.duration)

            if timeline is None:
                total_samples = max(1, int(round(total_duration * sample_rate)))
                timeline = np.zeros(total_samples, dtype=np.float32)

            start_sample = max(0, int(round(entry.start * sample_rate)))
            end_sample = min(timeline.size, start_sample + segment.size)
            if end_sample <= start_sample:
                continue
            segment = segment[: end_sample - start_sample]
            mixed = timeline[start_sample:end_sample] + segment
            timeline[start_sample:end_sample] = np.clip(mixed, -1.0, 1.0)

    if timeline is None:
        timeline = np.zeros(1, dtype=np.float32)
    write_wav_mono(output_path, timeline, sample_rate)
    log_callback(f"Saved audio: {output_path}")
    return str(output_path)


def batch_subtitle_to_audio(
    base_dir: str | os.PathLike[str],
    language: str,
    speaker: str,
    models_root: str | os.PathLike[str],
    log_callback: LogCallback = print,
    stop_event: Event | None = None,
) -> list[str]:
    tasks = collect_paired_srt_tasks(base_dir)
    if not tasks:
        raise ValueError("No paired SRT files found.")
    if not check_model_files(models_root):
        raise FileNotFoundError(f"Qwen3-TTS model files are missing: {get_model_dir(models_root)}")

    model = _load_tts_model(get_model_dir(models_root), log_callback)
    outputs: list[str] = []
    for index, srt_path in enumerate(tasks, 1):
        if stop_event and stop_event.is_set():
            raise RuntimeError("Stopped by user.")
        log_callback(f"=== [{index}/{len(tasks)}] {srt_path} ===")
        output = subtitle_to_audio(
            srt_path=srt_path,
            output_path=default_output_path(srt_path),
            language=language,
            speaker=speaker,
            models_root=models_root,
            log_callback=log_callback,
            stop_event=stop_event,
            tts_model=model,
        )
        outputs.append(output)
    return outputs
