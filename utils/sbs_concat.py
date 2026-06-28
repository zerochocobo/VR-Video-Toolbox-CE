"""Concat source-scan timeline clips with a copy-first strategy."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from gpu_engine import mux
from gpu_engine import probe
from gpu_engine import restored_sidecar
from gpu_engine.fallback import OperationCancelled
from utils import encode_config

_SIGNATURE_FIELDS = (
    "codec",
    "profile",
    "pix_fmt",
    "width",
    "height",
    "bit_depth",
    "color_range",
    "color_space",
    "color_transfer",
    "color_primaries",
)


class _TrackedProcess:
    def __init__(self, proc: subprocess.Popen):
        self._proc = proc
        self.cancelled = False

    def kill(self):
        self.cancelled = True
        return self._proc.kill()

    def terminate(self):
        self.cancelled = True
        return self._proc.terminate()

    def poll(self):
        return self._proc.poll()

    def __getattr__(self, name: str):
        return getattr(self._proc, name)


def _hidden_kwargs() -> dict:
    if sys.platform.startswith("win"):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        return {"startupinfo": startupinfo}
    return {}


def _run(cmd: list[str], log_callback=None, process_callback=None, label: str = "command") -> None:
    if log_callback:
        log_callback(f"Executing: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        **_hidden_kwargs(),
    )
    tracked = _TrackedProcess(proc)
    if process_callback:
        process_callback(tracked)
    try:
        if proc.stdout:
            for line in proc.stdout:
                if log_callback:
                    log_callback(line.rstrip())
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.wait()
    if proc.returncode != 0:
        if tracked.cancelled:
            raise OperationCancelled("cancelled by user")
        raise RuntimeError(f"ffmpeg {label} failed with code {proc.returncode}")


def _concat_path(path: Path) -> str:
    text = str(path.resolve()).replace("\\", "/")
    return text.replace("'", "'\\''")


def _entry_inpoint(entry) -> float | None:
    value = getattr(entry, "inpoint_s", None)
    return None if value is None else float(value)


def _entry_outpoint(entry) -> float | None:
    value = getattr(entry, "outpoint_s", None)
    return None if value is None else float(value)


def _write_concat_list(entries: list, directory: str | Path | None = None) -> Path:
    if directory is not None:
        Path(directory).mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        prefix="source_scan_concat_",
        suffix=".ffconcat",
        delete=False,
        mode="w",
        encoding="utf-8",
        dir=str(directory) if directory is not None else None,
    )
    try:
        tmp.write("ffconcat version 1.0\n")
        for entry in entries:
            path = Path(entry.path)
            tmp.write(f"file '{_concat_path(path)}'\n")
            inpoint = _entry_inpoint(entry)
            outpoint = _entry_outpoint(entry)
            if inpoint is not None:
                tmp.write(f"inpoint {inpoint:.6f}\n")
            if outpoint is not None:
                tmp.write(f"outpoint {outpoint:.6f}\n")
    finally:
        tmp.close()
    return Path(tmp.name)


def _video_signature(path: Path) -> tuple:
    meta = restored_sidecar.metadata_from_sidecar(path) or probe.probe_video(path)
    color = meta.color
    return (
        meta.codec_name.lower(),
        meta.profile.lower(),
        meta.pix_fmt.lower(),
        int(meta.width),
        int(meta.height),
        int(meta.bit_depth),
        color.color_range.lower(),
        color.color_space.lower(),
        color.color_transfer.lower(),
        color.color_primaries.lower(),
    )


def _collect_signatures(paths: list[Path]) -> tuple[list[tuple[Path, tuple]], list[str]]:
    signatures: list[tuple[Path, tuple]] = []
    errors: list[str] = []
    for path in paths:
        try:
            signatures.append((path, _video_signature(path)))
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    return signatures, errors


def _signature_mismatch_messages(signatures: list[tuple[Path, tuple]]) -> list[str]:
    if len(signatures) <= 1:
        return []
    base_path, base_sig = signatures[0]
    messages: list[str] = []
    for path, sig in signatures[1:]:
        diffs = [
            f"{field} {base_sig[idx]!r}!={sig[idx]!r}"
            for idx, field in enumerate(_SIGNATURE_FIELDS)
            if base_sig[idx] != sig[idx]
        ]
        if diffs:
            messages.append(f"{path} differs from {base_path}: " + "; ".join(diffs))
    return messages


def _format_signature(signature: tuple) -> str:
    return ", ".join(f"{field}={signature[idx]!r}" for idx, field in enumerate(_SIGNATURE_FIELDS))


def _all_params_match(paths: list[Path]) -> bool:
    if len(paths) <= 1:
        return True
    signatures, errors = _collect_signatures(paths)
    if errors:
        return False
    return not _signature_mismatch_messages(signatures)


def _unique_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = os.path.normcase(str(path.resolve()))
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _extract_entry_hevc(
    entry,
    output: Path,
    *,
    log_callback=None,
    process_callback=None,
) -> None:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    input_path = Path(entry.path)
    inpoint = _entry_inpoint(entry)
    outpoint = _entry_outpoint(entry)
    if input_path.suffix.lower() == ".hevc" and inpoint is None and outpoint is None:
        shutil.copyfile(input_path, output)
        return
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-stats", "-y"]
    if inpoint is not None and inpoint > 0.0:
        cmd += ["-ss", f"{inpoint:.6f}"]
    cmd += ["-i", str(input_path)]
    if outpoint is not None:
        start = inpoint if inpoint is not None else 0.0
        duration = max(0.001, float(outpoint) - float(start))
        cmd += ["-t", f"{duration:.6f}"]
    cmd += [
        "-map", "0:v:0",
        "-c:v", "copy",
        "-bsf:v", "hevc_mp4toannexb",
        "-f", "hevc",
        str(output),
    ]
    _run(cmd, log_callback=log_callback, process_callback=process_callback, label="hevc extract")


def _append_binary(paths: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as dst:
        for path in paths:
            with path.open("rb") as src:
                shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)


def _validate_fast_output(output: Path, source_meta: probe.VideoMetadata,
                          *, log_callback=None) -> None:
    out_meta = probe.probe_video(output)
    if int(out_meta.width) != int(source_meta.width) or int(out_meta.height) != int(source_meta.height):
        raise RuntimeError(
            f"fast merge output size mismatch: {out_meta.width}x{out_meta.height}, "
            f"expected {source_meta.width}x{source_meta.height}"
        )
    expected = float(source_meta.duration or 0.0)
    actual = float(out_meta.duration or 0.0)
    if expected > 0.0 and actual > 0.0:
        tolerance = max(2.0, 2.0 / max(1.0, float(source_meta.source_fps or 30.0)))
        if abs(actual - expected) > tolerance:
            raise RuntimeError(f"fast merge duration mismatch: {actual:.3f}s, expected {expected:.3f}s")
    if log_callback:
        log_callback(
            f"[source-scan] fast HEVC merge validated: "
            f"duration={actual:.3f}s expected={expected:.3f}s size={out_meta.width}x{out_meta.height}"
        )


def _concat_timeline_hevc_demuxer(
    entries: list,
    output: Path,
    *,
    source_meta: probe.VideoMetadata,
    audio_source: str | Path | None = None,
    log_callback=None,
    process_callback=None,
) -> None:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    list_file = _write_concat_list(entries, output.parent)
    fd, temp_name = tempfile.mkstemp(
        prefix=f"{output.stem}.hevc_demuxer_",
        suffix=output.suffix,
        dir=str(output.parent),
    )
    os.close(fd)
    temp_video = Path(temp_name)
    try:
        try:
            temp_video.unlink()
        except OSError:
            pass
        paths = [Path(entry.path) for entry in entries]
        size_hint = sum(path.stat().st_size for path in paths if path.exists())
        if log_callback:
            log_callback(
                f"[source-scan] Stage 4 fast HEVC demuxer attempt: "
                f"entries={len(entries)}, audio_source={audio_source or 'none'}, list={list_file}"
            )
        cmd = [
            ffmpeg,
            "-hide_banner", "-loglevel", "error", "-stats", "-y",
            "-f", "concat", "-safe", "0", "-i", str(list_file),
            "-map", "0:v:0",
            "-c:v", "copy",
            "-an",
            "-avoid_negative_ts", "make_zero",
        ]
        cmd += mux.faststart_args(size_hint)
        cmd += [str(temp_video)]
        _run(cmd, log_callback=log_callback, process_callback=process_callback, label="fast hevc demuxer concat")

        if audio_source is not None:
            _mux_mp4_video_with_source_audio(
                temp_video,
                output,
                audio_source,
                log_callback=log_callback,
                process_callback=process_callback,
            )
        else:
            try:
                output.unlink()
            except OSError:
                pass
            temp_video.replace(output)
        _validate_fast_output(output, source_meta, log_callback=log_callback)
    finally:
        try:
            list_file.unlink()
        except OSError:
            pass
        try:
            temp_video.unlink()
        except OSError:
            pass


def concat_timeline_hevc_fast(
    timeline: list,
    output: str | Path,
    *,
    source_src: str | Path,
    audio_source: str | Path | None = None,
    log_callback=None,
    process_callback=None,
) -> None:
    """Fast source-scan final merge by copying HEVC bitstreams, not reencoding.

    Virtual gap entries may point to the original source with inpoint/outpoint.
    Mosaic entries should point to full restored clips. Each entry is converted to
    AnnexB HEVC, concatenated as a raw stream, then muxed once with source audio.
    """
    entries = sorted(timeline, key=lambda entry: (float(entry.start_s), float(entry.end_s), str(entry.kind)))
    if not entries:
        raise ValueError("concat_timeline_hevc_fast requires at least one timeline entry")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_src = Path(source_src)
    source_meta = probe.probe_video(source_src)
    if source_meta.codec_name.lower() != "hevc":
        raise RuntimeError(f"fast HEVC merge requires HEVC source, got {source_meta.codec_name or 'unknown'}")

    paths = _unique_paths([source_src] + [Path(entry.path) for entry in entries])
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"timeline clip missing: {missing[0]}")
    signatures, signature_errors = _collect_signatures(paths)
    mismatch_messages = _signature_mismatch_messages(signatures)
    if signature_errors:
        raise RuntimeError("fast merge signature probe failed: " + "; ".join(signature_errors[:3]))
    if mismatch_messages:
        raise RuntimeError("fast merge parameter mismatch: " + "; ".join(mismatch_messages[:3]))

    try:
        from utils import app_config

        use_demuxer = bool(app_config.get("source_scan_fast_hevc_demuxer", False))
    except Exception:
        use_demuxer = False
    if use_demuxer:
        try:
            _concat_timeline_hevc_demuxer(
                entries,
                output,
                source_meta=source_meta,
                audio_source=audio_source,
                log_callback=log_callback,
                process_callback=process_callback,
            )
            return
        except OperationCancelled:
            raise
        except Exception as exc:
            if log_callback:
                log_callback(
                    f"[source-scan] fast HEVC demuxer concat failed; "
                    f"falling back to annex-b stream merge: {type(exc).__name__}: {exc}"
                )
            try:
                output.unlink()
            except OSError:
                pass
    elif log_callback:
        log_callback("[source-scan] fast HEVC demuxer attempt disabled; using annex-b stream merge")

    temp_dir = Path(tempfile.mkdtemp(prefix=f"{output.stem}.fast_hevc_", dir=str(output.parent)))
    raw_parts: list[Path] = []
    combined_raw = temp_dir / f"{output.stem}.combined.hevc"
    try:
        if log_callback:
            log_callback(
                f"[source-scan] Stage 4 fast HEVC merge: entries={len(entries)}, "
                f"source={source_src}, temp={temp_dir}"
            )
            for idx, entry in enumerate(entries[:30]):
                inpoint = _entry_inpoint(entry)
                outpoint = _entry_outpoint(entry)
                bounds = ""
                if inpoint is not None or outpoint is not None:
                    bounds = f" inpoint={inpoint if inpoint is not None else ''} outpoint={outpoint if outpoint is not None else ''}"
                log_callback(f"[source-scan] fast merge {idx}: kind={getattr(entry, 'kind', '')} path={entry.path}{bounds}")
            if len(entries) > 30:
                log_callback(f"[source-scan] ... {len(entries) - 30} more fast merge entries")

        for idx, entry in enumerate(entries):
            raw = temp_dir / f"part{idx:04d}.hevc"
            _extract_entry_hevc(entry, raw, log_callback=log_callback, process_callback=process_callback)
            if raw.stat().st_size <= 0:
                raise RuntimeError(f"fast merge extracted empty HEVC part: {entry.path}")
            raw_parts.append(raw)

        _append_binary(raw_parts, combined_raw)
        if combined_raw.stat().st_size <= 0:
            raise RuntimeError("fast merge produced empty combined HEVC stream")
        # shortest=False: combined.hevc is binary-concatenated from multiple
        # AnnexB segments, so total frame count can drift from source audio by
        # sub-second amounts. In -c copy mode, -shortest can misclassify and cut
        # the whole audio stream, appearing as an output with no audio, so disable it explicitly.
        mux.mux_hevc_with_audio(
            combined_raw,
            output,
            fps=source_meta.source_fps or 30.0,
            color=source_meta.color,
            audio_source=str(audio_source) if audio_source is not None else None,
            shortest=False,
            log_callback=log_callback,
        )
        _validate_fast_output(output, source_meta, log_callback=log_callback)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def concat_timeline(
    timeline: list,
    output: str | Path,
    *,
    audio_source: str | Path | None = None,
    log_callback=None,
    process_callback=None,
    reencode: str = "auto",
    cq: int | None = 18,
    bitrate_bps: int | None = None,
    max_bitrate_bps: int | None = None,
) -> None:
    """Concat timeline entries using video-only concat, then mux source audio once.

    ``auto`` uses concat demuxer + ``-c copy`` when all video parameters match.
    Audio is intentionally ignored for segment compatibility and, when
    ``audio_source`` is provided, is copied from the original source at the end.
    """
    entries = sorted(timeline, key=lambda entry: (float(entry.start_s), float(entry.end_s), str(entry.kind)))
    paths = [Path(entry.path) for entry in entries]
    if not paths:
        raise ValueError("concat_timeline requires at least one timeline entry")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"timeline clip missing: {missing[0]}")

    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    list_file = _write_concat_list(entries, output.parent)
    mode = str(reencode or "auto").lower()
    signatures: list[tuple[Path, tuple]] = []
    signature_errors: list[str] = []
    mismatch_messages: list[str] = []
    if mode == "auto":
        signatures, signature_errors = _collect_signatures(paths)
        mismatch_messages = _signature_mismatch_messages(signatures)
        copy_mode = not signature_errors and not mismatch_messages
    elif mode == "never":
        copy_mode = True
    elif mode == "always":
        copy_mode = False
    else:
        raise ValueError(f"unknown concat reencode mode: {reencode}")

    temp_video: Path | None = None
    try:
        concat_output = output
        if audio_source is not None:
            fd, temp_name = tempfile.mkstemp(
                prefix=f"{output.stem}.video_concat_",
                suffix=output.suffix,
                dir=str(output.parent),
            )
            os.close(fd)
            temp_video = Path(temp_name)
            try:
                temp_video.unlink()
            except OSError:
                pass
            concat_output = temp_video

        if log_callback:
            log_callback(
                f"[source-scan] Stage 4 concat: entries={len(paths)}, "
                f"mode={'copy' if copy_mode else 'nvenc-reencode'}, "
                f"audio_source={audio_source or 'none'}, list={list_file}"
            )
            for idx, entry in enumerate(entries[:30]):
                inpoint = _entry_inpoint(entry)
                outpoint = _entry_outpoint(entry)
                bounds = ""
                if inpoint is not None or outpoint is not None:
                    bounds = f" inpoint={inpoint if inpoint is not None else ''} outpoint={outpoint if outpoint is not None else ''}"
                log_callback(f"[source-scan] concat {idx}: kind={getattr(entry, 'kind', '')} path={entry.path}{bounds}")
            if len(paths) > 30:
                log_callback(f"[source-scan] ... {len(paths) - 30} more concat entries")
            if signatures:
                for idx, (path, signature) in enumerate(signatures[:30]):
                    log_callback(f"[source-scan] concat signature {idx}: {path.name}: {_format_signature(signature)}")
                if len(signatures) > 30:
                    log_callback(f"[source-scan] ... {len(signatures) - 30} more concat signatures")
            if mode == "auto" and not copy_mode:
                for err in signature_errors[:20]:
                    log_callback(f"[source-scan] concat signature probe failed: {err}")
                for msg in mismatch_messages[:20]:
                    log_callback(f"[source-scan] concat parameter mismatch: {msg}")
                log_callback("[source-scan] concat falling back to NVENC reencode")

        cmd = [
            ffmpeg,
            "-hide_banner", "-loglevel", "error", "-stats", "-y",
            "-f", "concat", "-safe", "0", "-i", str(list_file),
        ]
        if copy_mode:
            cmd += ["-map", "0:v:0", "-c:v", "copy", "-an", "-avoid_negative_ts", "make_zero"]
        else:
            first_meta = probe.probe_video(paths[0])
            cmd += ["-map", "0:v:0"]
            cmd += encode_config.build_ffmpeg_nvenc_base_args()
            cmd += ["-rc", "vbr"]
            cmd += encode_config.build_ffmpeg_pix_fmt_args(first_meta.bit_depth)
            if bitrate_bps and bitrate_bps > 0:
                kbps = max(1, int(bitrate_bps / 1000))
                if max_bitrate_bps and max_bitrate_bps > 0:
                    maxrate = max(kbps, int(max_bitrate_bps / 1000))
                else:
                    # Final delivered re-encode: tighten the VBR peak so the output
                    # converges near source instead of drifting toward a 2x peak.
                    maxrate = max(kbps, int(kbps * encode_config.final_maxrate_multiplier()))
                bufsize = max(int(kbps * 2), int(maxrate * 2))
                cmd += ["-b:v", f"{kbps}k", "-maxrate:v", f"{maxrate}k", "-bufsize:v", f"{bufsize}k"]
            else:
                cmd += ["-cq", str(int(cq if cq is not None else 18))]
            if first_meta.color is not None:
                cmd += first_meta.color.ffmpeg_args()
            cmd += encode_config.build_final_ffmpeg_reencode_tail_args(first_meta.source_fps)
            cmd += ["-an"]
        if copy_mode:
            size_hint = sum(path.stat().st_size for path in paths if path.exists())
        else:
            size_hint = int(bitrate_bps * max(0.0, sum(float(e.end_s) - float(e.start_s) for e in entries)) / 8) if bitrate_bps else None
        cmd += mux.faststart_args(size_hint)
        cmd += [str(concat_output)]
        _run(cmd, log_callback=log_callback, process_callback=process_callback, label="concat")

        if audio_source is not None:
            _mux_mp4_video_with_source_audio(
                concat_output,
                output,
                audio_source,
                log_callback=log_callback,
                process_callback=process_callback,
            )
    finally:
        try:
            list_file.unlink()
        except OSError:
            pass
        if temp_video is not None:
            try:
                temp_video.unlink()
            except OSError:
                pass


def _mux_mp4_video_with_source_audio(
    video_mp4: str | Path,
    output: str | Path,
    audio_source: str | Path,
    *,
    log_callback=None,
    process_callback=None,
) -> None:
    # Do not use -shortest. In -c copy mode, if the concatenated video container
    # header duration differs from the actual packets, -shortest can misclassify
    # audio as too long and cut it, appearing as an output with no audio. Video
    # and audio come from the same source and same duration, so let streams end naturally.
    # Do not use optional -map 1:a:0? either: source audio is mandatory, and missing
    # audio should fail loudly instead of being silently dropped.
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    size_hint = Path(video_mp4).stat().st_size if Path(video_mp4).exists() else None
    cmd = [
        ffmpeg,
        "-hide_banner", "-loglevel", "error", "-stats", "-y",
        "-i", str(video_mp4),
        "-i", str(audio_source),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "copy",
    ]
    cmd += mux.faststart_args(size_hint)
    cmd += [str(output)]
    _run(cmd, log_callback=log_callback, process_callback=process_callback, label="audio mux")
