"""Dubbing mode: replace a video's original dialogue with the cloned voice.

Pipeline per video:
  1. extract the original audio (mono 48 kHz);
  2. bandit-v2 separation -> keep the music+sfx bed, drop the original speech;
  3. mix the bed with the cloned/translated ``.si.wav`` track;
  4. mux the mixed audio back as the sole audio track -> ``<name>_DUB.mp4``.

All code lives in tool_clonevoice; tool_si is only reused read-only for the
``.si.wav`` path convention, pairing, and process helpers.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from threading import Event
from typing import Callable

import torch
import torchaudio as ta

from tool_si import logic as sl
from tool_clonevoice.separate import FS, BanditSeparator

LogCallback = Callable[[str], None]
ProcessCallback = Callable[["subprocess.Popen | None"], None]


def default_dub_output_path(video_path: str | os.PathLike[str]) -> str:
    path = Path(video_path)
    return sl._format_path_like_source(path.with_name(f"{path.stem}_DUB.mp4"), video_path)


def _extracted_mix_path(video_path: str | os.PathLike[str]) -> Path:
    return Path(video_path).with_suffix(".dub_mix48k.wav")


def default_background_path(video_path: str | os.PathLike[str]) -> str:
    return str(Path(video_path).with_suffix(".dub_bg.wav"))


def _run_ffmpeg(
    cmd: list[str],
    log_callback: LogCallback,
    stop_event: Event | None,
    process_callback: ProcessCallback | None,
    error_label: str,
) -> None:
    log_callback(f"Executing: {sl._format_command_for_log(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        errors="replace",
        startupinfo=sl._build_startupinfo(),
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
                        sl._terminate_process(process)
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
        raise RuntimeError(f"{error_label} failed with code {process.returncode}")


def _extract_mixdown(
    video_path: Path,
    out_wav: Path,
    log_callback: LogCallback,
    stop_event: Event | None,
    process_callback: ProcessCallback | None,
) -> None:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-stats", "-y",
        "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(FS), "-c:a", "pcm_s16le",
        str(out_wav),
    ]
    _run_ffmpeg(cmd, log_callback, stop_event, process_callback, "FFmpeg audio extraction")


def _mux_dub(
    video_path: Path,
    dub_audio: Path,
    output_path: Path,
    log_callback: LogCallback,
    stop_event: Event | None,
    process_callback: ProcessCallback | None,
    add_independent_track: bool = False,
) -> None:
    # Video copied as-is. By default the dub audio becomes the sole primary
    # audio track; optional independent-track mode keeps original audio too.
    audio_stream_count = sl.probe_audio_stream_count(video_path, log_callback)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-stats", "-y",
        "-i", str(video_path),
        "-i", str(dub_audio),
        "-map", "0:v?",
    ]
    if add_independent_track:
        if audio_stream_count is None:
            log_callback("Warning: audio stream count unavailable; preserving the first original audio track before DUB.")
            cmd.extend(["-map", "0:a:0", "-map", "1:a:0"])
            dub_audio_index = 1
        else:
            cmd.extend(["-map", "0:a?", "-map", "1:a:0"])
            dub_audio_index = audio_stream_count
    else:
        cmd.extend(["-map", "1:a:0"])
        dub_audio_index = 0
    cmd.extend([
        "-map", "0:s?",
        "-map_metadata", "0", "-map_chapters", "0",
        "-c:v", "copy", "-c:a", "copy",
        f"-c:a:{dub_audio_index}", "aac",
        f"-b:a:{dub_audio_index}", "192k",
        f"-ar:a:{dub_audio_index}", "48000",
        f"-ac:a:{dub_audio_index}", "2",
        "-c:s", "copy",
        f"-metadata:s:a:{dub_audio_index}", "title=DUB",
        f"-metadata:s:a:{dub_audio_index}", "handler_name=DUB",
        "-disposition:a:0", "default",
    ])
    if add_independent_track:
        cmd.extend([f"-disposition:a:{dub_audio_index}", "0"])
    cmd.extend(["-movflags", "+faststart", str(output_path)])
    _run_ffmpeg(cmd, log_callback, stop_event, process_callback, "FFmpeg dub mux")


def _mix_background_and_voice(
    background_path: Path,
    voice_path: Path,
    out_path: Path,
    background_volume_percent: int | float,
    voice_volume_percent: int | float,
) -> None:
    bg, bg_fs = ta.load(str(background_path))
    if bg.shape[0] > 1:
        bg = bg.mean(0, keepdim=True)
    if bg_fs != FS:
        bg = ta.functional.resample(bg, bg_fs, FS)

    voice, v_fs = ta.load(str(voice_path))
    if voice.shape[0] > 1:
        voice = voice.mean(0, keepdim=True)
    if v_fs != FS:
        voice = ta.functional.resample(voice, v_fs, FS)

    n = max(bg.shape[-1], voice.shape[-1])
    bg = torch.nn.functional.pad(bg, (0, n - bg.shape[-1]))
    voice = torch.nn.functional.pad(voice, (0, n - voice.shape[-1]))

    mix = bg * (background_volume_percent / 100.0) + voice * (voice_volume_percent / 100.0)

    # Simple peak limiter to avoid clipping after summing the two tracks.
    peak = float(mix.abs().max())
    if peak > 0.97:
        mix = mix * (0.97 / peak)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ta.save(str(out_path), mix, FS)


def dub_video(
    video_path: str | os.PathLike[str],
    si_audio_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None,
    separator: BanditSeparator,
    *,
    background_volume_percent: int | float = 100,
    voice_volume_percent: int | float = 100,
    skip_existing_background: bool = True,
    add_independent_track: bool = False,
    log_callback: LogCallback = print,
    stop_event: Event | None = None,
    process_callback: ProcessCallback | None = None,
) -> str:
    if not shutil.which("ffmpeg"):
        raise FileNotFoundError("ffmpeg not found.")

    video = Path(video_path)
    si_audio = Path(si_audio_path)
    output = Path(output_path or default_dub_output_path(video))
    if not video.is_file():
        raise FileNotFoundError(f"Video file not found: {video}")
    if not si_audio.is_file():
        raise FileNotFoundError(f"Cloned voice (.si.wav) not found: {si_audio}")
    if video.resolve() == output.resolve():
        raise ValueError("Output file must be different from the input video.")

    background = Path(default_background_path(video))
    mix_wav = _extracted_mix_path(video)
    dub_audio = video.with_suffix(".dub_audio.wav")

    try:
        if skip_existing_background and background.is_file():
            log_callback(f"Reusing existing background bed: {background.name}")
        else:
            log_callback("Extracting original audio (mono 48 kHz) ...")
            _extract_mixdown(video, mix_wav, log_callback, stop_event, process_callback)
            if stop_event and stop_event.is_set():
                raise RuntimeError("Stopped by user.")
            log_callback("Separating dialogue from music+sfx (bandit-v2) ...")
            separator.separate_background(mix_wav, background, stop_event=stop_event)

        if stop_event and stop_event.is_set():
            raise RuntimeError("Stopped by user.")

        log_callback("Mixing background bed with cloned voice ...")
        _mix_background_and_voice(
            background, si_audio, dub_audio,
            background_volume_percent, voice_volume_percent,
        )

        log_callback("Muxing dubbed audio into video ...")
        _mux_dub(
            video, dub_audio, output, log_callback, stop_event, process_callback,
            add_independent_track=add_independent_track,
        )
    finally:
        for tmp in (mix_wav, dub_audio):
            try:
                if tmp.is_file():
                    tmp.unlink()
            except Exception:
                pass

    log_callback(f"Saved dubbed video: {output}")
    return str(output)


def batch_dub_videos(
    base_dir: str | os.PathLike[str],
    models_root: str | os.PathLike[str],
    *,
    background_volume_percent: int | float = 100,
    voice_volume_percent: int | float = 100,
    skip_existing_background: bool = True,
    add_independent_track: bool = False,
    log_callback: LogCallback = print,
    stop_event: Event | None = None,
    recursive: bool = True,
    process_callback: ProcessCallback | None = None,
    separator: BanditSeparator | None = None,
) -> list[str]:
    tasks = sl.collect_paired_si_mix_tasks(base_dir, recursive=recursive)
    if not tasks:
        raise ValueError("No paired MP4/MKV + .si.wav files found.")

    owns_separator = separator is None
    if owns_separator:
        separator = BanditSeparator(models_root, log=log_callback)
    outputs: list[str] = []
    try:
        for index, task in enumerate(tasks, 1):
            if stop_event and stop_event.is_set():
                raise RuntimeError("Stopped by user.")
            log_callback(f"=== [{index}/{len(tasks)}] {task.video_path} ===")
            output = dub_video(
                video_path=task.video_path,
                si_audio_path=task.si_audio_path,
                output_path=default_dub_output_path(task.video_path),
                separator=separator,
                background_volume_percent=background_volume_percent,
                voice_volume_percent=voice_volume_percent,
                skip_existing_background=skip_existing_background,
                add_independent_track=add_independent_track,
                log_callback=log_callback,
                stop_event=stop_event,
                process_callback=process_callback,
            )
            outputs.append(output)
    finally:
        if owns_separator:
            separator.close()
    return outputs
