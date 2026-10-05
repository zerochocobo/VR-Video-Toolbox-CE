"""Prepare a complete AAC mix for PTMediaServer without copying the video."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from threading import Event

from utils.si_prepared_audio import (
    find_prepared_audio, input_signature, make_manifest, prepared_paths,
)


def mix_inputs(video: Path, voice: Path) -> tuple[str, Path | None]:
    from tool_si import logic as si

    # Explicit PTMediaServer defaults: toolbox remix defaults are different.
    candidate = video.with_suffix(".si.duck.wav")
    duck = candidate if video.suffix.lower() == ".mp4" and candidate.is_file() else None
    filt = si.build_si_mix_filter(
        "both", 100, 120 if duck else 100, 0.0,
        duck_original=True, duck_preset="strong" if duck else "normal",
        duck_key_input=duck is not None, duck_band=False,
    )
    return filt, duck


def _check_cancelled(stop_event: Event | None) -> None:
    if stop_event is not None and stop_event.is_set():
        raise RuntimeError("Stopped by user.")


def prepare_audio(video_path, voice_path=None, *, log=print, stop_event=None) -> str:
    """Create/reuse the default PT mix; failures leave the WAVs untouched."""
    from tool_si import logic as si

    _check_cancelled(stop_event)
    video = Path(video_path)
    voice = Path(voice_path) if voice_path else video.with_suffix(".si.wav")
    if video.suffix.lower() != ".mp4":
        raise ValueError("PT virtual SI audio preparation requires an MP4 source.")
    if not video.is_file() or not voice.is_file():
        raise FileNotFoundError(f"Missing MP4 or SI WAV: {video}, {voice}")
    filt, duck = mix_inputs(video, voice)
    existing = find_prepared_audio(video, voice, duck, filt)
    if existing is not None:
        log(f"[DLNA] prepared mix is up to date: {existing}")
        return str(existing)
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise FileNotFoundError("ffmpeg and ffprobe are required for DLNA audio preparation.")
    inputs = input_signature(video, voice, duck)
    output, metadata = prepared_paths(video)
    token = uuid.uuid4().hex
    temp_audio = output.with_name(f".{output.stem}.{token}.tmp.m4a")
    temp_metadata = metadata.with_name(f".{metadata.stem}.{token}.tmp.json")
    cmd = [ffmpeg, "-hide_banner", "-v", "error", "-nostdin", "-y",
           "-i", str(video), "-i", str(voice)]
    if duck is not None:
        cmd += ["-i", str(duck)]
    cmd += ["-filter_complex", filt, "-map", "[si_track]", "-vn", "-sn", "-dn",
            "-map_metadata", "-1", "-map_chapters", "-1",
            "-c:a", "aac", "-profile:a", "aac_low", "-b:a", "192k",
            "-ar", "48000", "-ac", "2", "-movflags", "+faststart", "-f", "mp4", str(temp_audio)]
    log(f"[DLNA] preparing full mixed AAC audio (48 kHz, stereo, 192 kb/s): {output}")
    try:
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=errors,
                startupinfo=si._build_startupinfo(),
            )
            try:
                while True:
                    _check_cancelled(stop_event)
                    try:
                        code = process.wait(timeout=0.2)
                        break
                    except subprocess.TimeoutExpired:
                        continue
            finally:
                if process.poll() is None:
                    si._terminate_process(process)
            if code:
                errors.seek(0)
                detail = errors.read().decode("utf-8", errors="replace")[-3000:]
                raise RuntimeError(f"DLNA audio encoding failed ({code}): {detail}")
        _check_cancelled(stop_event)
        probe = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries",
             "stream=codec_type,codec_name,profile,sample_rate,channels:format=duration",
             "-of", "json", str(temp_audio)],
            capture_output=True, text=True, errors="replace", timeout=30,
            startupinfo=si._build_startupinfo(), check=True,
        )
        info = json.loads(probe.stdout)
        streams = info.get("streams", [])
        if (len(streams) != 1 or streams[0].get("codec_name") != "aac"
                or streams[0].get("profile") != "LC"
                or streams[0].get("sample_rate") != "48000"
                or streams[0].get("channels") != 2
                or float(info.get("format", {}).get("duration", 0)) <= 0):
            raise RuntimeError("Prepared DLNA audio failed AAC format validation.")
        _check_cancelled(stop_event)
        current_filter, current_duck = mix_inputs(video, voice)
        if (filt != current_filter or inputs != input_signature(video, voice, current_duck)):
            raise RuntimeError("Source audio changed during DLNA preparation; retry the export.")
        manifest = make_manifest(inputs, filt, temp_audio)
        temp_metadata.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp_audio.replace(output)
        # The manifest is the commit marker; a partial/mismatched pair is ignored.
        temp_metadata.replace(metadata)
    finally:
        temp_audio.unlink(missing_ok=True)
        temp_metadata.unlink(missing_ok=True)
    log(f"[DLNA] saved prepared mixed audio: {output}")
    return str(output)


def prepare_after_synthesis(video_path, voice_path=None, *, log=print, stop_event=None):
    """Best effort extra output: report errors without losing successful TTS."""
    _check_cancelled(stop_event)
    video = Path(video_path)
    if video.suffix.lower() != ".mp4":
        log(f"[DLNA] skipping prepared audio: PT virtual SI requires MP4 ({video.name})")
        return None
    try:
        return prepare_audio(video, voice_path, log=log, stop_event=stop_event)
    except Exception as exc:
        _check_cancelled(stop_event)
        log(f"[DLNA] WARNING: WAV saved, but prepared mixed audio is unavailable: {exc}")
        return None


def main() -> int:
    """python -m tool_si.dlna_audio VIDEO_OR_DIRECTORY [--recursive]."""
    import argparse

    parser = argparse.ArgumentParser(description="Prepare PTMediaServer AAC mixes from existing SI WAVs.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()
    videos = set()
    for path in args.paths:
        if path.is_dir():
            candidates = path.rglob("*") if args.recursive else path.iterdir()
            videos.update(p for p in candidates if p.is_file() and p.suffix.lower() == ".mp4"
                          and p.with_suffix(".si.wav").is_file())
        else:
            videos.add(path)
    if not videos:
        parser.error("No paired MP4 / .si.wav files found.")
    failures = 0
    for video in sorted(videos):
        try:
            prepare_audio(video)
        except Exception as exc:
            failures += 1
            print(f"[DLNA] FAILED {video}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
