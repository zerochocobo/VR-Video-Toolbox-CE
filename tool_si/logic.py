from __future__ import annotations

import importlib.util
import json
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

import gc

import numpy as np


def _release_cuda_cache() -> None:
    """Drop the PyTorch allocator's free segments back to the driver.

    Qwen3-TTS forces ``output_hidden_states=True`` so each generate() call
    accumulates large per-step hidden states on GPU; combined with the
    sort-then-pack batches whose shapes vary every call, the caching allocator
    keeps growing reserved memory across batches. Calling empty_cache() between
    batches keeps a 16GB card from filling up over a long SRT.
    """
    try:
        import torch
    except Exception:
        return
    if getattr(torch, "cuda", None) is None or not torch.cuda.is_available():
        return
    gc.collect()
    torch.cuda.empty_cache()


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

SPEAKER_NOTE_KEYS = {
    speaker: f"speaker_note_{speaker.lower()}"
    for speaker in ALL_SPEAKERS
}

LANGUAGE_NATIVE_SPEAKERS = {
    "Chinese": ("Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"),
    "English": ("Ryan", "Aiden"),
    "Japanese": ("Ono_Anna",),
    "Korean": ("Sohee",),
}

DEFAULT_LANGUAGE_SPEAKERS = {
    "Chinese": "Serena",
}

UI_LANGUAGE_TO_TTS = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
}

LogCallback = Callable[[str], None]
SI_MIX_CHANNELS = ("left", "right", "both")
ORIGINAL_VOLUME_CHOICES = (70, 80, 90, 100)
SI_VOLUME_CHOICES = (50, 60, 70, 80, 90, 100)
SI_DELAY_SECONDS_CHOICES = (0.0, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0)
DEFAULT_ORIGINAL_VOLUME_PERCENT = 100
DEFAULT_SI_VOLUME_PERCENT = 50
DEFAULT_SI_DELAY_SECONDS = 1.0
SI_DUCK_THRESHOLD = "0.025"
SI_DUCK_RATIO = "5"
SI_DUCK_ATTACK_MS = "30"
SI_DUCK_RELEASE_MS = "600"
SI_DUCK_MAKEUP = "1"
MAX_SUBTITLE_ENTRY_DURATION = 300.0
MAX_SUBTITLE_TIMECODE_SECONDS = 6 * 60 * 60


@dataclass(frozen=True)
class SubtitleEntry:
    index: int
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class SITrackMixTask:
    video_path: Path
    si_audio_path: Path
    output_path: Path


def get_model_dir(models_root: str | os.PathLike[str]) -> str:
    return str(Path(models_root) / MODEL_DIR_NAME)


def default_tts_language(ui_language: str | None) -> str:
    return UI_LANGUAGE_TO_TTS.get((ui_language or "").lower(), "English")


def speakers_for_language(language: str) -> tuple[str, ...]:
    return LANGUAGE_NATIVE_SPEAKERS.get(language, ALL_SPEAKERS)


def default_speaker_for_language(language: str) -> str:
    speakers = speakers_for_language(language)
    preferred = DEFAULT_LANGUAGE_SPEAKERS.get(language)
    if preferred in speakers:
        return preferred
    return speakers[0] if speakers else ALL_SPEAKERS[0]


def speaker_note_key(speaker: str) -> str:
    return SPEAKER_NOTE_KEYS.get(speaker, "")


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
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"Invalid subtitle timecode: {value}")
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


def parse_srt(
    path: str | os.PathLike[str],
    language: str | None = None,
    log_callback: LogCallback | None = None,
) -> list[SubtitleEntry]:
    text = _read_text_with_fallback(path)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n+", text.strip())
    entries: list[SubtitleEntry] = []
    skipped_invalid_timecode = 0
    skipped_too_long = 0
    skipped_beyond_max = 0

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
        try:
            start = _parse_timecode(match.group("start"))
            end = _parse_timecode(match.group("end"))
        except ValueError:
            skipped_invalid_timecode += 1
            continue
        if end <= start:
            skipped_invalid_timecode += 1
            continue
        if end > MAX_SUBTITLE_TIMECODE_SECONDS:
            skipped_beyond_max += 1
            continue
        if (end - start) > MAX_SUBTITLE_ENTRY_DURATION:
            skipped_too_long += 1
            continue

        text_lines = lines[time_line_index + 1 :]
        selected_line = _select_subtitle_text_line(text_lines, language)
        if not selected_line:
            continue
        entries.append(SubtitleEntry(index=block_index, start=start, end=end, text=selected_line))

    if log_callback is not None:
        if skipped_invalid_timecode:
            log_callback(f"Skipped {skipped_invalid_timecode} subtitle entries with invalid timecodes.")
        if skipped_too_long:
            log_callback(
                f"Skipped {skipped_too_long} subtitle entries longer than "
                f"{MAX_SUBTITLE_ENTRY_DURATION:g}s (single-entry duration cap)."
            )
        if skipped_beyond_max:
            log_callback(
                f"Skipped {skipped_beyond_max} subtitle entries past "
                f"{MAX_SUBTITLE_TIMECODE_SECONDS / 3600:g}h (timecode cap)."
            )

    return entries


def _coerce_non_negative_seconds(value: int | float, name: str) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative number of seconds.") from exc
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"{name} must be a non-negative number of seconds.")
    return seconds


def _entries_for_time_window(
    entries: list[SubtitleEntry],
    start_seconds: int | float = 0.0,
    duration_seconds: int | float | None = None,
) -> tuple[list[SubtitleEntry], float | None, bool]:
    start = _coerce_non_negative_seconds(start_seconds, "start_seconds")
    if duration_seconds is None:
        duration = None
        end = None
    else:
        duration = _coerce_non_negative_seconds(duration_seconds, "duration_seconds")
        if duration <= 0:
            raise ValueError("duration_seconds must be positive or None.")
        end = start + duration

    time_limited = start > 0 or duration is not None
    if not time_limited:
        return list(entries), None, False

    selected: list[SubtitleEntry] = []
    for entry in entries:
        clipped_start = max(entry.start, start)
        clipped_end = entry.end if end is None else min(entry.end, end)
        if clipped_end <= clipped_start:
            continue
        selected.append(
            SubtitleEntry(
                index=entry.index,
                start=max(0.0, clipped_start - start),
                end=max(0.0, clipped_end - start),
                text=entry.text,
            )
        )
    return selected, duration, True


def default_output_path(srt_path: str | os.PathLike[str]) -> str:
    path = Path(srt_path)
    return _format_path_like_source(path.with_name(f"{path.stem}.si.wav"), srt_path)


def default_si_audio_path(video_path: str | os.PathLike[str]) -> str:
    path = Path(video_path)
    return _format_path_like_source(path.with_suffix(".si.wav"), video_path)


def default_si_mix_output_path(video_path: str | os.PathLike[str]) -> str:
    path = Path(video_path)
    return _format_path_like_source(path.with_name(f"{path.stem}_SI.mp4"), video_path)


def _format_path_like_source(path: Path, source_path: str | os.PathLike[str]) -> str:
    result = str(path)
    source = os.fspath(source_path)
    forward_index = source.rfind("/")
    backward_index = source.rfind("\\")
    if forward_index > backward_index:
        return result.replace("\\", "/")
    if backward_index > forward_index:
        return result.replace("/", "\\")
    return result


def collect_paired_si_mix_tasks(base_dir: str | os.PathLike[str], recursive: bool = True) -> list[SITrackMixTask]:
    root_path = Path(base_dir)
    seen: set[Path] = set()
    tasks: list[SITrackMixTask] = []
    candidates = root_path.rglob("*") if recursive else root_path.iterdir()
    for video in candidates:
        if not video.is_file() or video.suffix.lower() not in {".mp4", ".mkv"}:
            continue
        resolved = video.resolve()
        if resolved in seen:
            continue
        si_audio = Path(default_si_audio_path(video))
        if not si_audio.is_file():
            continue
        seen.add(resolved)
        tasks.append(
            SITrackMixTask(
                video_path=video,
                si_audio_path=si_audio,
                output_path=Path(default_si_mix_output_path(video)),
            )
        )
    return sorted(tasks, key=lambda task: str(task.video_path).lower())


def _validate_si_mix_channel(channel: str) -> str:
    normalized = (channel or "").strip().lower()
    if normalized not in SI_MIX_CHANNELS:
        raise ValueError(f"Unsupported SI mix channel: {channel}")
    return normalized


def _validate_original_volume(percent: int | float) -> int:
    try:
        value = int(percent)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid volume percent: {percent}") from exc
    if value not in ORIGINAL_VOLUME_CHOICES:
        raise ValueError("Original volume percent must be one of 70, 80, 90, 100.")
    return value


def _validate_si_volume(percent: int | float) -> int:
    try:
        value = int(percent)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid SI volume percent: {percent}") from exc
    if value not in SI_VOLUME_CHOICES:
        raise ValueError("SI volume percent must be one of 50, 60, 70, 80, 90, 100.")
    return value


def _validate_si_delay_seconds(seconds: int | float) -> float:
    try:
        value = round(float(seconds), 1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid SI delay seconds: {seconds}") from exc
    if value not in SI_DELAY_SECONDS_CHOICES:
        raise ValueError("SI delay must be one of 0, 0.3, 0.5, 0.7, 1, 1.2, 1.5, 2 seconds.")
    return value


def _filter_number(value: int | float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def build_si_mix_filter(
    mix_channel: str,
    original_volume_percent: int | float,
    si_volume_percent: int | float,
    si_delay_seconds: int | float = DEFAULT_SI_DELAY_SECONDS,
    duck_original: bool = False,
) -> str:
    channel = _validate_si_mix_channel(mix_channel)
    original_volume = _filter_number(_validate_original_volume(original_volume_percent) / 100.0)
    si_volume = _filter_number(_validate_si_volume(si_volume_percent) / 100.0)
    si_delay_ms = int(round(_validate_si_delay_seconds(si_delay_seconds) * 1000))

    if channel == "both":
        # SI overlaid equally on BOTH channels (no channel split).
        if duck_original:
            compressor = (
                f"threshold={SI_DUCK_THRESHOLD}:"
                f"ratio={SI_DUCK_RATIO}:"
                f"attack={SI_DUCK_ATTACK_MS}:"
                f"release={SI_DUCK_RELEASE_MS}:"
                f"makeup={SI_DUCK_MAKEUP}"
            )
            return (
                "[0:a:0]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
                f"volume={original_volume}[orig_base];"
                "[1:a:0]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=mono,"
                f"adelay={si_delay_ms},volume={si_volume},apad,asplit=2[si_key][si_mono];"
                f"[orig_base][si_key]sidechaincompress={compressor}[orig];"
                "[si_mono]aformat=channel_layouts=stereo[si];"
                "[orig][si]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
                "alimiter=limit=0.95[si_track]"
            )
        return (
            f"[0:a:0]aresample=48000,aformat=channel_layouts=stereo,volume={original_volume}[orig];"
            f"[1:a:0]aresample=48000,aformat=channel_layouts=mono,adelay={si_delay_ms},"
            f"volume={si_volume},aformat=channel_layouts=stereo[si];"
            "[orig][si]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
            "alimiter=limit=0.95[si_track]"
        )

    if channel == "left":
        mix_part = (
            "[ol][si]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[left_mix_raw];"
            "[left_mix_raw][or]join=inputs=2:channel_layout=stereo,"
            "alimiter=limit=0.95[si_track]"
        )
    else:
        mix_part = (
            "[or][si]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[right_mix_raw];"
            "[ol][right_mix_raw]join=inputs=2:channel_layout=stereo,"
            "alimiter=limit=0.95[si_track]"
        )

    if duck_original:
        compressor = (
            f"threshold={SI_DUCK_THRESHOLD}:"
            f"ratio={SI_DUCK_RATIO}:"
            f"attack={SI_DUCK_ATTACK_MS}:"
            f"release={SI_DUCK_RELEASE_MS}:"
            f"makeup={SI_DUCK_MAKEUP}"
        )
        return (
            "[0:a:0]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"volume={original_volume}[orig_base];"
            "[1:a:0]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=mono,"
            f"adelay={si_delay_ms},volume={si_volume},apad,asplit=2[si_key][si];"
            f"[orig_base][si_key]sidechaincompress={compressor}[orig];"
            "[orig]channelsplit=channel_layout=stereo[ol][or];"
            f"{mix_part}"
        )

    return (
        f"[0:a:0]aresample=48000,aformat=channel_layouts=stereo,volume={original_volume}[orig];"
        f"[1:a:0]aresample=48000,aformat=channel_layouts=mono,adelay={si_delay_ms},volume={si_volume}[si];"
        "[orig]channelsplit=channel_layout=stereo[ol][or];"
        f"{mix_part}"
    )


def probe_audio_stream_count(video_path: str | os.PathLike[str], log_callback: LogCallback | None = None) -> int | None:
    if not shutil.which("ffprobe"):
        if log_callback:
            log_callback("ffprobe not found; audio stream count is unavailable.")
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            errors="replace",
            startupinfo=_build_startupinfo(),
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        if log_callback:
            log_callback("ffprobe audio stream probe timed out after 15 seconds.")
        return None
    except Exception as exc:
        if log_callback:
            log_callback(f"ffprobe audio stream probe failed: {exc}")
        return None
    if result.returncode != 0:
        if log_callback:
            message = (result.stderr or result.stdout or "").strip()
            log_callback(f"ffprobe audio stream probe failed: {message}")
        return None
    try:
        data = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return None
    streams = data.get("streams", [])
    if not isinstance(streams, list):
        return None
    return len(streams)


def build_si_audio_mix_command(
    video_path: str | os.PathLike[str],
    si_audio_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    mix_channel: str,
    original_volume_percent: int | float,
    si_volume_percent: int | float,
    si_delay_seconds: int | float = DEFAULT_SI_DELAY_SECONDS,
    audio_stream_count: int | None = 1,
    add_independent_track: bool = False,
    duck_original: bool = False,
) -> list[str]:
    if audio_stream_count is not None and audio_stream_count < 1:
        raise ValueError("Input video must contain at least one audio stream.")

    filter_arg = build_si_mix_filter(
        mix_channel,
        original_volume_percent,
        si_volume_percent,
        si_delay_seconds,
        duck_original=duck_original,
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-stats",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(si_audio_path),
        "-filter_complex",
        filter_arg,
        "-map",
        "0:v?",
    ]
    if add_independent_track:
        si_audio_index = audio_stream_count if audio_stream_count is not None else 1
        cmd.extend(["-map", "0:a?" if audio_stream_count is not None else "0:a:0", "-map", "[si_track]"])
    else:
        si_audio_index = 0
        cmd.extend(["-map", "[si_track]"])
        if audio_stream_count is not None:
            for audio_index in range(1, audio_stream_count):
                cmd.extend(["-map", f"0:a:{audio_index}?"])
        else:
            cmd.extend(["-map", "0:a?", "-map", "-0:a:0"])

    cmd.extend(
        [
            "-map",
            "0:s?",
            "-map_metadata",
            "0",
            "-map_chapters",
            "0",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            f"-c:a:{si_audio_index}",
            "aac",
            f"-b:a:{si_audio_index}",
            "192k",
            f"-ar:a:{si_audio_index}",
            "48000",
            f"-ac:a:{si_audio_index}",
            "2",
            "-c:s",
            "copy",
            f"-metadata:s:a:{si_audio_index}",
            "title=SI",
            f"-metadata:s:a:{si_audio_index}",
            "handler_name=SI",
            "-disposition:a:0",
            "default",
        ]
    )
    if add_independent_track:
        cmd.extend([f"-disposition:a:{si_audio_index}", "0"])
    cmd.extend(["-movflags", "+faststart", str(output_path)])
    return cmd


def _format_command_for_log(cmd: list[str]) -> str:
    if sys.platform == "win32":
        return subprocess.list2cmdline(cmd)
    return " ".join(cmd)


def _terminate_process(process: subprocess.Popen, timeout: float = 2.0) -> None:
    try:
        if process.poll() is not None:
            return
    except Exception:
        return
    try:
        process.terminate()
        process.wait(timeout=timeout)
        return
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                timeout=5,
                startupinfo=_build_startupinfo(),
            )
            process.wait(timeout=timeout)
            return
        except Exception:
            pass
    try:
        process.kill()
        process.wait(timeout=timeout)
    except Exception:
        pass


def mix_si_audio_track(
    video_path: str | os.PathLike[str],
    si_audio_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
    mix_channel: str = "left",
    original_volume_percent: int | float = DEFAULT_ORIGINAL_VOLUME_PERCENT,
    si_volume_percent: int | float = DEFAULT_SI_VOLUME_PERCENT,
    si_delay_seconds: int | float = DEFAULT_SI_DELAY_SECONDS,
    add_independent_track: bool = False,
    duck_original: bool = False,
    log_callback: LogCallback = print,
    stop_event: Event | None = None,
    process_callback: Callable[[subprocess.Popen | None], None] | None = None,
) -> str:
    if not shutil.which("ffmpeg"):
        raise FileNotFoundError("ffmpeg not found.")

    video = Path(video_path)
    si_audio = Path(si_audio_path)
    output = Path(output_path or default_si_mix_output_path(video))
    if not video.is_file():
        raise FileNotFoundError(f"Video file not found: {video}")
    if not si_audio.is_file():
        raise FileNotFoundError(f"SI audio file not found: {si_audio}")
    if video.resolve() == output.resolve():
        raise ValueError("Output file must be different from the input video.")
    output.parent.mkdir(parents=True, exist_ok=True)

    audio_stream_count = probe_audio_stream_count(video, log_callback)
    if audio_stream_count is None:
        if add_independent_track:
            log_callback("Warning: audio stream count unavailable; preserving the first original audio track before SI.")
        else:
            log_callback(
                "Warning: audio stream count unavailable; copying original audio streams except the first after "
                "the mixed SI track."
            )
    cmd = build_si_audio_mix_command(
        video_path=video,
        si_audio_path=si_audio,
        output_path=output,
        mix_channel=mix_channel,
        original_volume_percent=original_volume_percent,
        si_volume_percent=si_volume_percent,
        si_delay_seconds=si_delay_seconds,
        audio_stream_count=audio_stream_count,
        add_independent_track=add_independent_track,
        duck_original=duck_original,
    )
    log_callback(f"Executing: {_format_command_for_log(cmd)}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        errors="replace",
        startupinfo=_build_startupinfo(),
    )
    if process_callback:
        process_callback(process)
    try:
        if process.stdout:
            try:
                for line in process.stdout:
                    text = line.strip()
                    if text:
                        log_callback(text)
                    if stop_event and stop_event.is_set():
                        _terminate_process(process)
                        break
            except OSError:
                if not (stop_event and stop_event.is_set()):
                    raise
        process.wait()
    finally:
        if process_callback:
            process_callback(None)
        try:
            if process.stdout:
                process.stdout.close()
        except Exception:
            pass

    if stop_event and stop_event.is_set():
        raise RuntimeError("Stopped by user.")
    if process.returncode != 0:
        err_msg = f"FFmpeg SI audio mix failed with code {process.returncode}"
        try:
            from utils import ffmpeg_checker

            ffmpeg_checker.handle_ffmpeg_error(cmd, err_msg, log_callback)
        except Exception:
            pass
        raise RuntimeError(err_msg)
    log_callback(f"Saved mixed video: {output}")
    return str(output)


def batch_mix_si_audio_tracks(
    base_dir: str | os.PathLike[str],
    mix_channel: str = "left",
    original_volume_percent: int | float = DEFAULT_ORIGINAL_VOLUME_PERCENT,
    si_volume_percent: int | float = DEFAULT_SI_VOLUME_PERCENT,
    si_delay_seconds: int | float = DEFAULT_SI_DELAY_SECONDS,
    add_independent_track: bool = False,
    duck_original: bool = False,
    log_callback: LogCallback = print,
    stop_event: Event | None = None,
    recursive: bool = True,
    process_callback: Callable[[subprocess.Popen | None], None] | None = None,
) -> list[str]:
    tasks = collect_paired_si_mix_tasks(base_dir, recursive=recursive)
    if not tasks:
        raise ValueError("No paired MP4/MKV + .si.wav files found.")

    outputs: list[str] = []
    for index, task in enumerate(tasks, 1):
        if stop_event and stop_event.is_set():
            raise RuntimeError("Stopped by user.")
        log_callback(f"=== [{index}/{len(tasks)}] {task.video_path} ===")
        output = mix_si_audio_track(
            video_path=task.video_path,
            si_audio_path=task.si_audio_path,
            output_path=task.output_path,
            mix_channel=mix_channel,
            original_volume_percent=original_volume_percent,
            si_volume_percent=si_volume_percent,
            si_delay_seconds=si_delay_seconds,
            add_independent_track=add_independent_track,
            duck_original=duck_original,
            log_callback=log_callback,
            stop_event=stop_event,
            process_callback=process_callback,
        )
        outputs.append(output)
    return outputs


def collect_paired_srt_tasks(base_dir: str | os.PathLike[str], recursive: bool = True) -> list[Path]:
    root_path = Path(base_dir)
    seen: set[Path] = set()
    tasks: list[Path] = []
    candidates = root_path.rglob("*") if recursive else root_path.iterdir()
    for video in candidates:
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


def _speed_up_with_ffmpeg(audio: np.ndarray, sample_rate: int, factor: float, target_samples: int | None = None) -> np.ndarray:
    if not shutil.which("ffmpeg"):
        return _speed_up_fast_resample(audio, target_samples) if target_samples else audio
    # Windows: if the timeout fires, ffmpeg's killed handle may briefly hold the
    # temp wav files. ignore_cleanup_errors avoids a PermissionError leaking out
    # of TemporaryDirectory.__exit__ (Python 3.10+).
    with tempfile.TemporaryDirectory(prefix="si_tts_", ignore_cleanup_errors=True) as tmp_dir:
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
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                startupinfo=_build_startupinfo(),
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return _speed_up_fast_resample(audio, target_samples) if target_samples else audio
        if result.returncode != 0 or not dst.exists():
            return _speed_up_fast_resample(audio, target_samples) if target_samples else audio
        converted, converted_rate = read_wav_mono(dst)
        if converted_rate != sample_rate:
            return _speed_up_fast_resample(audio, target_samples) if target_samples else audio
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
                mono = _speed_up_with_ffmpeg(mono, sample_rate, factor, target_samples)
            else:
                mono = _speed_up_in_memory(mono, target_samples, factor)
        elif factor > 1.0:
            mono = _speed_up_fast_resample(mono, target_samples)

    if mono.size > target_samples:
        mono = mono[:target_samples]
    elif mono.size < target_samples:
        mono = np.pad(mono, (0, target_samples - mono.size))
    return np.clip(mono, -1.0, 1.0).astype(np.float32, copy=False)


def _mix_timeline_segment(timeline: np.ndarray, start_sample: int, segment: np.ndarray) -> None:
    if start_sample >= timeline.size:
        return
    end_sample = min(timeline.size, start_sample + segment.size)
    if end_sample <= start_sample:
        return
    segment = segment[: end_sample - start_sample]
    existing = timeline[start_sample:end_sample]
    # If any sample in this range already has audio from a previous entry, the
    # two subtitles overlap. Halve the whole new segment AND the existing audio
    # in the overlap window uniformly so volume stays flat across the boundary,
    # instead of per-sample halving which would step-jump at the overlap edge.
    overlapping = np.any((np.abs(existing) > 1e-6) & (np.abs(segment) > 1e-6))
    if overlapping:
        mixed = existing * 0.5 + segment * 0.5
    else:
        mixed = existing + segment
    timeline[start_sample:end_sample] = np.clip(mixed, -1.0, 1.0)


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
    start_seconds: int | float = 0.0,
    duration_seconds: int | float | None = None,
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
    all_entries = parse_srt(srt_path, language=language, log_callback=log_callback)
    window_entries, fixed_duration, time_limited = _entries_for_time_window(
        all_entries,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
    )
    if max_entries is not None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive or None.")
        entries = window_entries[:max_entries]
    else:
        entries = window_entries
    if not entries:
        if all_entries and time_limited:
            raise ValueError("No subtitle entries found in the selected time range.")
        raise ValueError("No valid subtitle entries found.")
    model_dir = get_model_dir(models_root)
    if not check_model_files(models_root):
        raise FileNotFoundError(f"Qwen3-TTS model files are missing: {model_dir}")

    if tts_model is None:
        tts_model = _load_tts_model(model_dir, log_callback)

    timeline: np.ndarray | None = None
    sample_rate = 24000
    total_duration = fixed_duration if fixed_duration is not None else max(entry.end for entry in entries)
    batch_size = resolve_tts_batch_size()
    batches = _iter_tts_batches(entries, batch_size, preserve_order=max_entries is not None)
    if time_limited:
        start = _coerce_non_negative_seconds(start_seconds, "start_seconds")
        if duration_seconds is None:
            range_desc = f"from {start:.3f}s"
        else:
            duration = _coerce_non_negative_seconds(duration_seconds, "duration_seconds")
            range_desc = f"{start:.3f}-{start + duration:.3f}s"
        limit_desc = ""
        if max_entries is not None and len(window_entries) > len(entries):
            limit_desc = f" Converting first {len(entries)} selected entries for test."
        log_callback(
            f"Parsed {len(all_entries)} subtitle entries; selected {len(window_entries)} in {range_desc}."
            f"{limit_desc} Output duration: {total_duration:.3f}s. "
            f"TTS batch size: {batch_size} ({len(batches)} batches)."
        )
    elif max_entries is not None and len(all_entries) > len(entries):
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
            _mix_timeline_segment(timeline, start_sample, segment)

        # Sort-then-pack batches have a different (batch, seq) shape almost every
        # iteration, so the allocator's reserved pool keeps climbing on a 16GB
        # card. Drop free segments back to the driver after each batch.
        del generated
        _release_cuda_cache()

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
    recursive: bool = True,
) -> list[str]:
    tasks = collect_paired_srt_tasks(base_dir, recursive=recursive)
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
        # Free per-file scratch (timeline, intermediate tensors held by allocator)
        # before moving on to the next SRT so cumulative reserved memory stays flat.
        _release_cuda_cache()
    return outputs
