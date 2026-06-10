import subprocess
import os
import shutil
import glob
import json
import queue
import time
import sys
import threading
from pathlib import Path

# Import engine layer and helper methods.
try:
    from utils import engine_runner, app_config
    from utils.ffmpeg_checker import get_startupinfo, get_video_bitrate
    from gpu_engine.fallback import OperationCancelled
except ImportError:
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from utils import engine_runner, app_config
    from utils.ffmpeg_checker import get_startupinfo, get_video_bitrate
    from gpu_engine.fallback import OperationCancelled

def check_dependencies():
    missing = []
    tools = ["ffmpeg", "ffprobe"]
    if engine_runner.is_native_engine():
        # Opening the page should not import CuPy or warm up the GPU stack.
        # Full runtime availability is checked when a native_gpu task actually starts.
        try:
            from gpu_engine import native_mosaic
            reason = native_mosaic.unavailable_reason(runtime_check=False)
            if reason:
                missing.append(f"内置引擎: {reason}")
        except Exception as e:
            missing.append(f"内置引擎(torch CUDA / 模型文件): {e}")
    else:
        engine_cli = engine_runner.get_engine_executable()
        if engine_cli:
            tools.append(engine_cli)
    for tool in tools:
        if not shutil.which(tool):
            missing.append(tool)
    return missing


def _time_to_sec(t):
    """Convert 'HH:MM:SS' / 'MM:SS' / 'SS' / None to seconds as float or None."""
    if not t:
        return None
    try:
        parts = [float(x) for x in str(t).split(':')]
    except ValueError:
        return None
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return None


# crop_filter string -> gpu_engine crop mode.
_CROP_MODE = {
    "crop=iw/2:ih:0:0": "left",
    "crop=iw/2:ih:iw/2:0": "right",
    "crop=iw:ih/2:0:0": "top",
    "crop=iw:ih/2:0:ih/2": "bottom",
}


def _single_eye_split_vbr_bps(source_bitrate_bps: int | None) -> tuple[int | None, int | None]:
    """Return one-eye split VBR target/max as 0.75x/1.0x source bitrate."""
    try:
        source = int(source_bitrate_bps or 0)
    except (TypeError, ValueError):
        return None, None
    if source <= 0:
        return None, None
    return max(1, int(source * 0.75)), source


def _area_scaled_bitrate_bps(source_bitrate_bps: int | None,
                             source_w: int | None,
                             source_h: int | None,
                             out_w: int | None,
                             out_h: int | None,
                             expansion: float = 2.0) -> int | None:
    """Scale source bitrate by output/source area with a quality expansion factor."""
    try:
        source = int(source_bitrate_bps or 0)
        src_area = int(source_w or 0) * int(source_h or 0)
        out_area = int(out_w or 0) * int(out_h or 0)
        factor = float(expansion or 1.0)
    except (TypeError, ValueError):
        return None
    if source <= 0 or src_area <= 0 or out_area <= 0 or factor <= 0:
        return None
    return max(1, int(source * (out_area / src_area) * factor))


def _bitrate_bps_to_kbps(bitrate_bps: int | None) -> int | None:
    if not bitrate_bps or bitrate_bps <= 0:
        return None
    return max(1, int(bitrate_bps) // 1000)


def _pipeline_baseline_bitrate_bps(out_w: int | None, out_h: int | None, fps: float | None) -> int | None:
    """Conservative floor for very low-bitrate sources, about 0.015 bit/px/frame."""
    try:
        px = int(out_w or 0) * int(out_h or 0)
        rate = float(fps or 30.0)
    except (TypeError, ValueError):
        return None
    if px <= 0:
        return None
    return max(1, int(px * max(1.0, rate) * 0.015))


def _config_float(key: str, default: float) -> float:
    try:
        return float(app_config.get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)


def _resolve_pipeline_bitrate(stage: str,
                              out_w: int | None,
                              out_h: int | None,
                              fps: float | None,
                              source_bps: int | None,
                              keep_original: bool = False,
                              *,
                              source_w: int | None = None,
                              source_h: int | None = None,
                              log_callback=None) -> int | None:
    """Resolve OneClick target bitrate for intermediate and final NVENC outputs."""
    stage_key = str(stage or "").strip().lower()
    if stage_key not in {"intermediate", "final"}:
        raise ValueError(f"unknown bitrate stage: {stage}")
    # Intermediate stages are repeatedly decoded/re-encoded downstream, so we
    # always reserve quality headroom for high-detail regions regardless of the
    # keep-original toggle (which only governs the final-output convergence).
    if stage_key == "intermediate":
        multiplier = _config_float("gpu_bitrate_multiplier", 2.0)
    else:
        multiplier = 1.0 if keep_original else _config_float("gpu_bitrate_final_multiplier", 1.0)
    try:
        source = int(source_bps or 0)
    except (TypeError, ValueError):
        source = 0
    try:
        src_area = int(source_w or 0) * int(source_h or 0)
        out_area = int(out_w or 0) * int(out_h or 0)
    except (TypeError, ValueError):
        src_area = 0
        out_area = 0
    area_scale = (out_area / src_area) if src_area > 0 and out_area > 0 else 1.0
    baseline = _pipeline_baseline_bitrate_bps(out_w, out_h, fps)
    target = int(source * area_scale * multiplier) if source > 0 else 0
    pre_baseline_target = target
    # Intermediate stages always honour the baseline floor (they feed downstream
    # re-encodes and need quality headroom even when the user kept the original
    # bitrate). Final-stage with keep_original strictly trusts the source so the
    # output convergence promise holds.
    skip_baseline = stage_key == "final" and keep_original and source > 0
    if baseline and not skip_baseline:
        target = max(target, baseline)
    if target <= 0:
        return None
    if log_callback:
        source_label = f"{source // 1000}kbps" if source > 0 else "unknown"
        target_label = f"{target // 1000}kbps"
        log_callback(
            f"[bitrate] stage={stage_key} source={source_label} "
            f"keep_original={bool(keep_original)} multiplier={multiplier:.2f} -> target={target_label}"
        )
        if baseline and target > pre_baseline_target:
            before_label = f"{pre_baseline_target // 1000}kbps" if pre_baseline_target > 0 else "unknown"
            log_callback(
                f"[bitrate] baseline applied: baseline={baseline // 1000}kbps "
                f"{before_label} -> target={target_label}"
            )
    return target


def _resolve_single_eye_final_bitrate(input_file: str,
                                      original_bitrate: int | None,
                                      keep_original: bool,
                                      log_callback=None) -> int | None:
    try:
        from gpu_engine import probe as gpu_probe

        meta = gpu_probe.probe_video(input_file)
        return _resolve_pipeline_bitrate(
            "final",
            max(1, int(meta.width // 2)),
            meta.height,
            meta.source_fps or 30.0,
            int(original_bitrate / 2) if original_bitrate else None,
            keep_original,
            log_callback=log_callback,
        )
    except Exception:
        return int(original_bitrate / 2) if (keep_original and original_bitrate) else None


def _resolve_sbs_eye_intermediate_bitrate(input_file: str,
                                          original_bitrate: int | None,
                                          keep_original: bool,
                                          log_callback=None) -> int | None:
    try:
        from gpu_engine import probe as gpu_probe

        meta = gpu_probe.probe_video(input_file)
        return _resolve_pipeline_bitrate(
            "intermediate",
            max(1, int(meta.width // 2)),
            meta.height,
            meta.source_fps or 30.0,
            int(original_bitrate / 2) if original_bitrate else None,
            keep_original,
            log_callback=log_callback,
        )
    except Exception:
        return None


def _log_final_bitrate_summary(final_output: str | Path,
                               source_bps: int | None,
                               log_callback=None) -> None:
    """Log final mp4 average bitrate and warn when it drifts far above source."""
    if not log_callback:
        return
    try:
        path = Path(final_output)
        if not path.exists():
            log_callback(f"[bitrate] final mp4 self-check skipped: missing output {path}")
            return
        from gpu_engine import probe as gpu_probe

        meta = gpu_probe.probe_video(path)
        duration = float(meta.duration or 0.0)
        if duration <= 0.0:
            log_callback(f"[bitrate] final mp4 self-check skipped: unknown duration for {path}")
            return
        avg_bps = int(path.stat().st_size * 8 / duration)
        try:
            source = int(source_bps or 0)
        except (TypeError, ValueError):
            source = 0
        if source > 0:
            ratio = avg_bps / float(source)
            log_callback(
                f"[bitrate] final mp4: {avg_bps // 1000} kbps avg "
                f"(source {source // 1000} kbps, ratio {ratio:.3f}x)"
            )
            if ratio > 1.20:
                log_callback(
                    f"[bitrate] WARNING final mp4 ratio {ratio:.3f}x exceeds 1.20x; "
                    "check final NVENC bitrate contract"
                )
        else:
            log_callback(f"[bitrate] final mp4: {avg_bps // 1000} kbps avg (source unknown)")
    except Exception as exc:
        log_callback(f"[bitrate] final mp4 self-check failed: {type(exc).__name__}: {exc}")


def _nvenc_vbr_or_cq_args(target_kbps: int | None, max_kbps: int | None = None) -> list[str]:
    if target_kbps and target_kbps > 0:
        maxrate = max(int(target_kbps), int(max_kbps if max_kbps and max_kbps > 0 else target_kbps * 1.2))
        bufsize = max(int(target_kbps * 2), int(maxrate * 2))
        return [
            "-c:v", "hevc_nvenc",
            "-preset", "p7",
            "-rc", "vbr",
            "-b:v", f"{int(target_kbps)}k",
            "-maxrate:v", f"{maxrate}k",
            "-bufsize:v", f"{bufsize}k",
        ]
    return ["-c:v", "hevc_nvenc", "-preset", "p7", "-cq", "18"]


class _ProcessFileLogger:
    def __init__(self, input_file: str, ui_callback=None):
        self.ui_callback = ui_callback
        self.path = self._path_for(input_file)
        self._handle = None
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self._handle = open(self.path, "w", encoding="utf-8-sig", buffering=1)
            self(f"=== Process log started {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
            self(f"[log] file: {self.path}")
            self(f"[log] input: {os.path.abspath(input_file)}")
            self(f"[log] cwd: {os.getcwd()}")
        except Exception as exc:
            self._handle = None
            if self.ui_callback:
                self.ui_callback(f"[log] failed to open process log: {exc}")

    @staticmethod
    def _path_for(input_file: str) -> str:
        directory = os.path.dirname(os.path.abspath(input_file))
        stem = os.path.splitext(os.path.basename(input_file))[0]
        return os.path.join(directory, f"{stem}_process.log")

    def __call__(self, message):
        text = str(message)
        if self.ui_callback:
            self.ui_callback(text)
        if self._handle:
            try:
                stamp = time.strftime("%Y-%m-%d %H:%M:%S")
                self._handle.write(f"[{stamp}] {text}\n")
            except Exception:
                pass

    def close(self):
        if not self._handle:
            return
        try:
            self(f"=== Process log ended {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
            self._handle.close()
        except Exception:
            pass
        self._handle = None


def _remove_file_quiet(path: str | Path | None, log_callback=None) -> None:
    if not path:
        return
    try:
        p = Path(path)
        if p.exists():
            p.unlink()
            if log_callback:
                log_callback(f"[cleanup] removed: {p}")
    except OSError:
        pass


def _cut_subrange_keyframe(input_path: str, output_path: str,
                           start_sec: float | None, end_sec: float | None,
                           log_callback=None, process_callback=None) -> None:
    """Fast keyframe-aligned subrange cut preserving both video and audio.

    Uses input-side ``-ss`` plus ``-c copy`` so this is essentially I/O bound:
    a one-minute 4K HEVC clip finishes in a few seconds, no NVENC/NVDEC load.
    The trade-off is that the output aligns to the nearest preceding keyframe
    of the source, so the actual start/duration can drift a few seconds from
    what the user requested. The UI surfaces this to the user next to the
    end-time input so the drift is not mistaken for a bug.

    Both video and audio streams are copied unchanged, so the pre-clip
    carries the full audio track that the rest of the OneClick pipeline
    relies on. This pre-cut is also why audio survives all downstream mux
    hops in source-scan / native-stream / legacy fallback.
    """
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-stats", "-y"]
    # -ss before -i: input-side seek, snaps to nearest preceding keyframe.
    if start_sec is not None and start_sec > 0.0:
        cmd += ["-ss", f"{float(start_sec):.6f}"]
    cmd += ["-i", str(input_path)]
    if end_sec is not None:
        duration = max(0.001, float(end_sec) - float(start_sec or 0.0))
        cmd += ["-t", f"{duration:.6f}"]
    cmd += [
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(output_path),
    ]
    if log_callback:
        log_callback(
            f"[preclip] keyframe cut "
            f"{start_sec if start_sec is not None else 'START'}->"
            f"{end_sec if end_sec is not None else 'END'} -> {output_path}"
        )
    run_process(cmd, log_callback, process_callback)


def _prepare_subrange_preclip(input_file: str, output_dir: str,
                              start_time, end_time,
                              log_callback=None, process_callback=None) -> tuple[str, str | None]:
    """If start/end are set, pre-cut the source once with audio preserved and
    return (new_input_file, preclip_path_for_cleanup). Otherwise return the
    original input and None.

    The pre-clip is named ``<source_stem>_S<ss>_E<to><source_ext>`` and placed
    in ``output_dir`` (same folder as the final output). Using the source's
    original extension keeps ``-c copy`` always valid. The rest of the
    OneClick pipeline (source-scan / native-stream / legacy fallback) then
    runs against this clipped file as if no start/end had been requested,
    eliminating the legacy ``start_time/end_time`` code path that used to
    silently drop audio across split/lada/merge mux hops.
    """
    if not (start_time or end_time):
        return input_file, None
    start_sec = _time_to_sec(start_time)
    end_sec = _time_to_sec(end_time)
    src_stem, src_ext = os.path.splitext(os.path.basename(input_file))
    src_ext = src_ext or ".mp4"
    ss_part = start_time.replace(":", "") if start_time else "START"
    to_part = end_time.replace(":", "") if end_time else "END"
    suffix = f"_S{ss_part}_E{to_part}"
    os.makedirs(output_dir, exist_ok=True)
    preclip_path = os.path.join(output_dir, f"{src_stem}{suffix}{src_ext}")
    if os.path.exists(preclip_path):
        if log_callback:
            log_callback(f"[preclip] reusing existing preclip: {preclip_path}")
    else:
        _cut_subrange_keyframe(
            input_file, preclip_path, start_sec, end_sec,
            log_callback=log_callback, process_callback=process_callback,
        )
        if log_callback:
            try:
                size_mb = os.path.getsize(preclip_path) / (1024 * 1024)
                log_callback(f"[preclip] done: {preclip_path} ({size_mb:.1f} MB)")
            except OSError:
                log_callback(f"[preclip] done: {preclip_path}")
    return preclip_path, preclip_path


def _cleanup_run_artifacts(input_file: str, final_output: str | None = None,
                           process_log_path: str | None = None,
                           log_callback=None) -> None:
    """Remove non-video process artifacts when intermediate files are not kept."""
    input_path = Path(input_file)
    candidates: list[Path] = [
        input_path.with_name(f"{input_path.stem}.detections.jsonl"),
    ]
    if final_output:
        final_path = Path(final_output)
        candidates.append(final_path.with_name(f"{final_path.stem}.source_intervals.json"))
    if process_log_path:
        candidates.append(Path(process_log_path))
    seen: set[str] = set()
    for path in candidates:
        key = os.path.normcase(os.path.abspath(str(path)))
        if key in seen:
            continue
        seen.add(key)
        _remove_file_quiet(path, log_callback=log_callback)


def get_video_info(file_path):
    try:
        # Get Duration
        cmd_dur = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path]
        print(f"Executing: {' '.join(cmd_dur)}")
        duration_str = subprocess.check_output(cmd_dur, startupinfo=get_startupinfo(), text=True, encoding='utf-8', errors='replace').strip()
        duration = float(duration_str)

        # Get Resolution (Width/Height)
        cmd_res = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", file_path]
        print(f"Executing: {' '.join(cmd_res)}")
        res_str = subprocess.check_output(cmd_res, startupinfo=get_startupinfo(), text=True, encoding='utf-8', errors='replace').strip()
        width, height = map(int, res_str.split(','))

        # Get Codec
        cmd_codec = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0", file_path]
        print(f"Executing: {' '.join(cmd_codec)}")
        codec = subprocess.check_output(cmd_codec, startupinfo=get_startupinfo(), text=True, encoding='utf-8', errors='replace').strip()

        return {"duration": duration, "width": width, "height": height, "codec": codec}
    except Exception as e:
        print(f"Error getting video info: {e}")
        return None

def run_process(cmd, log_callback, process_callback=None):
    if log_callback:
        log_callback(f"Executing: {' '.join(cmd)}")
    else:
        print(f"Executing: {' '.join(cmd)}")
        
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        encoding='utf-8',
        errors='replace',
        startupinfo=get_startupinfo(),
    )
    if process_callback: process_callback(process)

    try:
        for line in process.stdout:
            if log_callback: log_callback(line.strip())
    finally:
        try:
            if process.stdout:
                process.stdout.close()
        except Exception:
            pass
        process.wait()
    if process.returncode != 0:
        err_msg = f"Command failed with code {process.returncode}"
        try:
            checker_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if checker_path not in sys.path:
                sys.path.append(checker_path)
            from utils import ffmpeg_checker
            ffmpeg_checker.handle_ffmpeg_error(cmd, err_msg, log_callback)
        except Exception as e:
            if log_callback: log_callback(f"Checker error: {e}")
            pass
        raise Exception(err_msg)

# --- Core Actions ---

def split_video(input_file, output_file, crop_filter, start_time=None, end_time=None, log_callback=None, process_callback=None, final_bitrate_kbps=None, max_bitrate_kbps=None, keep_audio=True):
    """Single-eye crop with optional time range and bitrate. Prefer GPU and fall back to ffmpeg on failure."""
    crop_mode = _CROP_MODE.get(crop_filter)
    try:
        from gpu_engine import probe as gpu_probe, fallback as gpu_fallback, files as gpu_files
        meta, decision = gpu_probe.route(input_file)
        auto_bitrate_bps, auto_max_bps = _single_eye_split_vbr_bps(meta.bitrate_bps)
        target_bps = int(final_bitrate_kbps * 1000) if final_bitrate_kbps else auto_bitrate_bps
        max_bps = int(max_bitrate_kbps * 1000) if max_bitrate_kbps else (auto_max_bps if not final_bitrate_kbps else None)
        target_kbps = _bitrate_bps_to_kbps(target_bps)
        max_kbps = _bitrate_bps_to_kbps(max_bps)

        def _gpu_fn():
            token = gpu_files.CancelToken()
            if process_callback:
                process_callback(token)
            gpu_files.extract_clip(
                input_file, output_file, crop_mode=crop_mode,
                start_sec=_time_to_sec(start_time), end_sec=_time_to_sec(end_time),
                cq=None if target_bps else 18,
                bitrate_bps=target_bps,
                max_bitrate_bps=max_bps,
                keep_audio=keep_audio, log_callback=log_callback, cancel_token=token,
            )
            return True

        def _ffmpeg_fn():
            return _split_video_ffmpeg(input_file, output_file, crop_filter, start_time, end_time, log_callback, process_callback, target_kbps, max_kbps, keep_audio=keep_audio)

        if log_callback: log_callback(f"Splitting: {input_file} -> {output_file}")
        return gpu_fallback.run_with_fallback(
            _gpu_fn, _ffmpeg_fn, gpu_eligible=(decision.is_gpu and crop_mode is not None),
            log_callback=log_callback, label="split",
        )
    except Exception as e:
        if log_callback: log_callback(f"Split error: {e}")
        raise


def split_video_fisheye(input_file, output_file, crop_filter, start_time=None, end_time=None, log_callback=None, process_callback=None, final_bitrate_kbps=None, max_bitrate_kbps=None, keep_audio=True):
    """Single-eye path: crop + hequirect->fisheye in one decode/encode, avoiding one transcode pass.

    Prefer GPU through gpu_files.extract_clip(crop+to_fisheye), falling back to
    ffmpeg with crop and v360 in one command.
    """
    crop_mode = _CROP_MODE.get(crop_filter)
    try:
        from gpu_engine import probe as gpu_probe, fallback as gpu_fallback, files as gpu_files
        meta, decision = gpu_probe.route(input_file)
        auto_bitrate_bps, auto_max_bps = _single_eye_split_vbr_bps(meta.bitrate_bps)
        target_bps = int(final_bitrate_kbps * 1000) if final_bitrate_kbps else auto_bitrate_bps
        max_bps = int(max_bitrate_kbps * 1000) if max_bitrate_kbps else (auto_max_bps if not final_bitrate_kbps else None)
        target_kbps = _bitrate_bps_to_kbps(target_bps)
        max_kbps = _bitrate_bps_to_kbps(max_bps)

        def _gpu_fn():
            token = gpu_files.CancelToken()
            if process_callback:
                process_callback(token)
            gpu_files.extract_clip(
                input_file, output_file, crop_mode=crop_mode, to_fisheye=True,
                start_sec=_time_to_sec(start_time), end_sec=_time_to_sec(end_time),
                cq=None if target_bps else 18,
                bitrate_bps=target_bps,
                max_bitrate_bps=max_bps,
                keep_audio=keep_audio, log_callback=log_callback, cancel_token=token,
            )
            return True

        def _ffmpeg_fn():
            return _split_video_fisheye_ffmpeg(input_file, output_file, crop_filter, start_time, end_time, log_callback, process_callback, target_kbps, max_kbps, keep_audio=keep_audio)

        if log_callback: log_callback(f"Splitting + VR->Fisheye: {input_file} -> {output_file}")
        return gpu_fallback.run_with_fallback(
            _gpu_fn, _ffmpeg_fn, gpu_eligible=(decision.is_gpu and crop_mode is not None),
            log_callback=log_callback, label="split+fisheye",
        )
    except Exception as e:
        if log_callback: log_callback(f"Split+fisheye error: {e}")
        raise


def _split_video_fisheye_ffmpeg(input_file, output_file, crop_filter, start_time=None, end_time=None, log_callback=None, process_callback=None, final_bitrate_kbps=None, max_bitrate_kbps=None, keep_audio=True):
    """ffmpeg fallback: crop + v360=hequirect:fisheye in one command."""
    info = get_video_info(input_file)
    codec = info['codec'] if info else 'hevc'
    decoder_opt = ["-hwaccel", "cuda", "-c:v", "hevc_cuvid"]
    if codec == 'h264':
        decoder_opt = ["-hwaccel", "cuda", "-c:v", "h264_cuvid"]
    cmd = ["ffmpeg"]
    if start_time: cmd.extend(["-ss", start_time])
    if end_time: cmd.extend(["-to", end_time])
    cmd.extend(["-hide_banner", "-loglevel", "error", "-stats"])
    cmd.extend(decoder_opt)
    cmd.extend(["-i", input_file])
    cmd.extend(["-vf", f"{crop_filter},v360=hequirect:fisheye"])
    cmd.extend(["-c:a", "copy"] if keep_audio else ["-an"])
    cmd.extend(_nvenc_vbr_or_cq_args(final_bitrate_kbps, max_bitrate_kbps))
    cmd.extend([output_file, "-y"])
    run_process(cmd, log_callback, process_callback)


def _split_video_ffmpeg(input_file, output_file, crop_filter, start_time=None, end_time=None, log_callback=None, process_callback=None, final_bitrate_kbps=None, max_bitrate_kbps=None, keep_audio=True):
    # Detect codec for hardware acceleration
    info = get_video_info(input_file)
    codec = info['codec'] if info else 'hevc'
    
    decoder_opt = ["-hwaccel", "cuda", "-c:v", "hevc_cuvid"]
    if codec == 'h264':
        decoder_opt = ["-hwaccel", "cuda", "-c:v", "h264_cuvid"]
    
    cmd = ["ffmpeg"]
    if start_time: cmd.extend(["-ss", start_time])
    if end_time: cmd.extend(["-to", end_time])
    cmd.extend([ "-hide_banner", "-loglevel", "error","-stats"])
    cmd.extend(decoder_opt)
    cmd.extend(["-i", input_file])
    cmd.extend(["-vf", crop_filter])
    cmd.extend(["-c:a", "copy"] if keep_audio else ["-an"])
    
    cmd.extend(_nvenc_vbr_or_cq_args(final_bitrate_kbps, max_bitrate_kbps))
    cmd.extend([output_file, "-y"])
    
    if log_callback: log_callback(f"Splitting: {input_file} -> {output_file}")
    run_process(cmd, log_callback, process_callback)

def split_video_dual(input_file, output_left, output_right, start_time=None, end_time=None, log_callback=None, process_callback=None, keep_audio=True):
    """Split video into left and right halves. Prefer GPU and fall back to ffmpeg on failure."""
    try:
        from gpu_engine import probe as gpu_probe, fallback as gpu_fallback, files as gpu_files
        meta, decision = gpu_probe.route(input_file)
        target_bps, max_bps = _single_eye_split_vbr_bps(meta.bitrate_bps)
        target_kbps = _bitrate_bps_to_kbps(target_bps)
        max_kbps = _bitrate_bps_to_kbps(max_bps)

        def _gpu_fn():
            token = gpu_files.CancelToken()
            if process_callback:
                process_callback(token)
            gpu_files.split_video(
                input_file, {"left": output_left, "right": output_right},
                to_fisheye=False, cq=None if target_bps else 18,
                bitrate_bps=target_bps,
                max_bitrate_bps=max_bps,
                start_sec=_time_to_sec(start_time), end_sec=_time_to_sec(end_time),
                keep_audio=keep_audio, log_callback=log_callback, cancel_token=token,
            )
            return True

        def _ffmpeg_fn():
            return _split_video_dual_ffmpeg(input_file, output_left, output_right, start_time, end_time, log_callback, process_callback, target_kbps, max_kbps, keep_audio=keep_audio)

        if log_callback: log_callback(f"Splitting (dual): {input_file} -> {output_left}, {output_right}")
        return gpu_fallback.run_with_fallback(
            _gpu_fn, _ffmpeg_fn, gpu_eligible=decision.is_gpu,
            log_callback=log_callback, label="split_dual",
        )
    except Exception as e:
        if log_callback: log_callback(f"Split dual error: {e}")
        raise


def _split_video_dual_ffmpeg(input_file, output_left, output_right, start_time=None, end_time=None, log_callback=None, process_callback=None, final_bitrate_kbps=None, max_bitrate_kbps=None, keep_audio=True):
    """Split video into left and right halves in a single ffmpeg call."""
    # Detect codec for hardware acceleration
    info = get_video_info(input_file)
    codec = info['codec'] if info else 'hevc'
    
    decoder_opt = ["-hwaccel", "cuda", "-c:v", "hevc_cuvid"]
    if codec == 'h264':
        decoder_opt = ["-hwaccel", "cuda", "-c:v", "h264_cuvid"]
    
    cmd = ["ffmpeg"]
    if start_time: cmd.extend(["-ss", start_time])
    if end_time: cmd.extend(["-to", end_time])
    cmd.extend(["-hide_banner", "-loglevel", "error", "-stats"])
    cmd.extend(decoder_opt)
    cmd.extend(["-i", input_file])
    
    # Use filter_complex to output both files in one pass
    filter_complex = "[0:v]crop=iw/2:ih:0:0[left];[0:v]crop=iw/2:ih:iw/2:0[right]"
    cmd.extend(["-filter_complex", filter_complex])
    
    # Left output
    cmd.extend(["-map", "[left]"])
    cmd.extend(["-map", "0:a?", "-c:a", "copy"] if keep_audio else ["-an"])
    cmd.extend(_nvenc_vbr_or_cq_args(final_bitrate_kbps, max_bitrate_kbps))
    cmd.extend([output_left, "-y"])
    
    # Right output
    cmd.extend(["-map", "[right]"])
    cmd.extend(["-map", "0:a?", "-c:a", "copy"] if keep_audio else ["-an"])
    cmd.extend(_nvenc_vbr_or_cq_args(final_bitrate_kbps, max_bitrate_kbps))
    cmd.extend([output_right, "-y"])
    
    if log_callback: log_callback(f"Splitting (dual): {input_file} -> {output_left}, {output_right}")
    run_process(cmd, log_callback, process_callback)

def split_video_dual_fisheye(input_file, output_left_fisheye, output_right_fisheye, start_time=None, end_time=None, log_callback=None, process_callback=None, keep_audio=True):
    """Split + VR->fisheye dual output. Prefer GPU and fall back to ffmpeg on failure."""
    try:
        from gpu_engine import probe as gpu_probe, fallback as gpu_fallback, files as gpu_files
        meta, decision = gpu_probe.route(input_file)
        target_bps, max_bps = _single_eye_split_vbr_bps(meta.bitrate_bps)
        target_kbps = _bitrate_bps_to_kbps(target_bps)
        max_kbps = _bitrate_bps_to_kbps(max_bps)

        def _gpu_fn():
            token = gpu_files.CancelToken()
            if process_callback:
                process_callback(token)
            gpu_files.split_video(
                input_file, {"left": output_left_fisheye, "right": output_right_fisheye},
                to_fisheye=True, cq=None if target_bps else 18,
                bitrate_bps=target_bps,
                max_bitrate_bps=max_bps,
                start_sec=_time_to_sec(start_time), end_sec=_time_to_sec(end_time),
                keep_audio=keep_audio, log_callback=log_callback, cancel_token=token,
            )
            return True

        def _ffmpeg_fn():
            return _split_video_dual_fisheye_ffmpeg(input_file, output_left_fisheye, output_right_fisheye, start_time, end_time, log_callback, process_callback, target_kbps, max_kbps, keep_audio=keep_audio)

        if log_callback: log_callback(f"Splitting + VR->Fisheye: {input_file} -> {output_left_fisheye}, {output_right_fisheye}")
        return gpu_fallback.run_with_fallback(
            _gpu_fn, _ffmpeg_fn, gpu_eligible=decision.is_gpu,
            log_callback=log_callback, label="split_dual_fisheye",
        )
    except Exception as e:
        if log_callback: log_callback(f"Split dual fisheye error: {e}")
        raise


def _split_video_dual_fisheye_ffmpeg(input_file, output_left_fisheye, output_right_fisheye, start_time=None, end_time=None, log_callback=None, process_callback=None, final_bitrate_kbps=None, max_bitrate_kbps=None, keep_audio=True):
    """Split video into left and right halves and convert to fisheye in a single ffmpeg call."""
    # Detect codec for hardware acceleration
    info = get_video_info(input_file)
    codec = info['codec'] if info else 'hevc'
    
    decoder_opt = ["-hwaccel", "cuda", "-c:v", "hevc_cuvid"]
    if codec == 'h264':
        decoder_opt = ["-hwaccel", "cuda", "-c:v", "h264_cuvid"]
    
    cmd = ["ffmpeg"]
    if start_time: cmd.extend(["-ss", start_time])
    if end_time: cmd.extend(["-to", end_time])
    cmd.extend(["-hide_banner", "-loglevel", "error", "-stats"])
    cmd.extend(decoder_opt)
    cmd.extend(["-i", input_file])
    
    # Use filter_complex to split and convert to fisheye in one pass
    filter_complex = "[0:v]crop=iw/2:ih:0:0,v360=hequirect:fisheye[left];[0:v]crop=iw/2:ih:iw/2:0,v360=hequirect:fisheye[right]"
    cmd.extend(["-filter_complex", filter_complex])
    
    # Left output
    cmd.extend(["-map", "[left]"])
    cmd.extend(["-map", "0:a?", "-c:a", "copy"] if keep_audio else ["-an"])
    cmd.extend(_nvenc_vbr_or_cq_args(final_bitrate_kbps, max_bitrate_kbps))
    cmd.extend([output_left_fisheye, "-y"])
    
    # Right output
    cmd.extend(["-map", "[right]"])
    cmd.extend(["-map", "0:a?", "-c:a", "copy"] if keep_audio else ["-an"])
    cmd.extend(_nvenc_vbr_or_cq_args(final_bitrate_kbps, max_bitrate_kbps))
    cmd.extend([output_right_fisheye, "-y"])
    
    if log_callback: log_callback(f"Splitting + VR->Fisheye: {input_file} -> {output_left_fisheye}, {output_right_fisheye}")
    run_process(cmd, log_callback, process_callback)


def process_lada(input_file, output_file, log_callback=None, process_callback=None,
                 bitrate_bps: int | None = None, produce_mp4: bool = True,
                 sidecar_metadata: dict | None = None) -> str:
    """Remove mosaics. engine='native_gpu' uses the in-process built-in engine; otherwise use lada/jasna CLI."""
    tool_name = engine_runner.get_mosaic_tool_name()
    if engine_runner.is_native_engine():
        from gpu_engine import native_mosaic
        from gpu_engine import restored_sidecar
        from gpu_engine.files import CancelToken
        token = CancelToken()
        if process_callback:
            process_callback(token)
        if log_callback: log_callback(f"{tool_name} Processing: {input_file} -> {output_file}")
        ok = native_mosaic.restore_file(
            input_file,
            output_file,
            bitrate_bps=bitrate_bps,
            log_callback=log_callback,
            cancel_token=token,
            produce_mp4=produce_mp4,
            sidecar_metadata=sidecar_metadata,
        )
        if not ok:
            raise Exception("native_gpu restore failed or was cancelled")
        raw_path = restored_sidecar.raw_path_for_output(output_file)
        if not produce_mp4 and _file_nonempty(raw_path):
            return str(raw_path)
        return str(output_file)

    opts = engine_runner.build_lada_encoder_options(cq=18)
    cmd = engine_runner.build_engine_cmd(
        input_file=input_file,
        output_file=output_file,
        encoder_options=opts,
    )
    if log_callback: log_callback(f"{tool_name} Processing: {input_file} -> {output_file}")
    run_process(cmd, log_callback, process_callback)
    return str(output_file)


class PreExtractResult:
    OK = "ok"
    NO_MOSAIC = "no_mosaic"
    SCAN_FAILED = "scan_failed"
    CANCELLED = "cancelled"


def _pre_extract_supported(pre_extract, log_callback=None) -> bool:
    if not pre_extract:
        return False
    return True


def _resolve_fine_conf(fine_conf=None) -> float:
    value = fine_conf if fine_conf is not None else app_config.get("pre_extract_fine_yolo_conf", 0.50)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.50


def _release_pre_extract_detector_if_needed(pre_extract_enabled, log_callback=None) -> None:
    if not pre_extract_enabled:
        return
    try:
        from utils.mosaic_prescan import release_detector

        release_detector(log_callback=log_callback)
    except Exception:
        pass


def _run_pre_extract_branch(base_path, restored_path, keep_intermediate=False,
                            log_callback=None, process_callback=None,
                            fine_conf=None, output_bitrate_bps: int | None = None) -> str:
    """Run detect/cut/restore/paste for one prepared base video.

    SCAN_FAILED means the detector failed and callers should fall back to full
    lada/jasna. NO_MOSAIC is a valid scan result and is copied through here.
    """
    from gpu_engine import probe as gpu_probe
    from gpu_engine.files import CancelToken
    from utils.keyframe_cutter import align_segments, cut_segment, list_keyframes, segment_file_matches_rect
    from utils.mosaic_prescan import save_segments_json, scan_segments
    from utils.segment_paster import build_paste_segments, paste_segments_gpu_or_fallback

    base_path = os.path.abspath(base_path)
    restored_path = os.path.abspath(restored_path)
    base_dir = os.path.dirname(base_path)
    stem = os.path.splitext(os.path.basename(base_path))[0]
    segments_json = os.path.join(base_dir, f"{stem}.segments.json")
    detections_jsonl = os.path.join(base_dir, f"{stem}.detections.jsonl")
    keep_segments = bool(app_config.get("pre_extract_keep_segments", False)) or bool(keep_intermediate)

    if log_callback:
        log_callback(f"[pre-extract] scanning {base_path}")
    scan_token = CancelToken()
    if process_callback:
        process_callback(scan_token)
    try:
        fine_conf = _resolve_fine_conf(fine_conf)
        if log_callback:
            log_callback(f"[pre-extract] fine detection conf filter: {fine_conf:.2f}")
        segments = scan_segments(base_path, log_callback=log_callback, cancel_token=scan_token, min_conf=fine_conf)
    except OperationCancelled as exc:
        if log_callback:
            log_callback(f"[pre-extract] scan cancelled: {exc}")
        return PreExtractResult.CANCELLED
    except Exception as exc:
        if scan_token.cancelled:
            if log_callback:
                log_callback("[pre-extract] scan cancelled")
            return PreExtractResult.CANCELLED
        if log_callback:
            log_callback(f"[pre-extract] scan failed: {type(exc).__name__}: {exc}")
        return PreExtractResult.SCAN_FAILED
    if not segments:
        save_segments_json([], segments_json, source=base_path)
        if log_callback:
            log_callback("[pre-extract] detector found no mosaic; copying base video and skipping lada/jasna")
        if os.path.abspath(base_path) != os.path.abspath(restored_path):
            shutil.copy2(base_path, restored_path)
        if not keep_segments:
            _remove_file_quiet(segments_json, log_callback=log_callback)
            _remove_file_quiet(detections_jsonl, log_callback=log_callback)
        return PreExtractResult.NO_MOSAIC

    meta = gpu_probe.probe_video(base_path)
    keyframes = list_keyframes(base_path)
    if log_callback:
        log_callback(f"[pre-extract] keyframes found: {len(keyframes)}")
    if keyframes:
        try:
            inject_mode = str(app_config.get("pre_extract_inject_keyframes", "auto") or "auto").lower()
            gop_sec = float(app_config.get("pre_extract_inject_gop_sec", 2.0) or 2.0)
            max_gap = max((b - a) for a, b in zip(keyframes, keyframes[1:])) if len(keyframes) > 1 else 0.0
            if inject_mode in {"auto", "always"} and max_gap > max(5.0, gop_sec * 2.5) and log_callback:
                log_callback(
                    f"[pre-extract] large GOP detected (max keyframe gap {max_gap:.1f}s); "
                    "dense keyframe injection is not implemented yet, so aligned segments may be wider"
                )
        except Exception:
            pass
    segments = align_segments(segments, keyframes, duration=meta.duration)
    save_segments_json(segments, segments_json, source=base_path, fps=meta.source_fps)
    if log_callback:
        log_callback(f"[pre-extract] saved metadata: {segments_json}")

    restored_segments = []
    for seg in segments:
        seg_in = os.path.join(base_dir, f"{stem}.seg{seg.seg_id:03d}.mp4")
        seg_out = os.path.join(base_dir, f"{stem}.seg{seg.seg_id:03d}.restored.mp4")
        if log_callback:
            log_callback(
                f"[pre-extract] segment {seg.seg_id}: "
                f"{seg.start_s_kf:.3f}-{seg.end_s_kf:.3f}s rect={seg.x},{seg.y},{seg.w}x{seg.h}"
            )
        if os.path.exists(seg_out) and not segment_file_matches_rect(seg_out, seg):
            if log_callback:
                log_callback(f"[pre-extract] existing restored segment has wrong size, reprocessing: {seg_out}")
            try:
                os.remove(seg_out)
            except OSError:
                pass
        if not os.path.exists(seg_out):
            if os.path.exists(seg_in) and not segment_file_matches_rect(seg_in, seg):
                if log_callback:
                    log_callback(f"[pre-extract] existing segment has wrong size, recutting: {seg_in}")
                try:
                    os.remove(seg_in)
                except OSError:
                    pass
            if not os.path.exists(seg_in):
                cut_segment(base_path, seg_in, seg, log_callback=log_callback, process_callback=process_callback)
            else:
                if log_callback:
                    log_callback(f"[pre-extract] segment input exists, skipping cut: {seg_in}")
            process_lada(seg_in, seg_out, log_callback=log_callback, process_callback=process_callback)
        else:
            if log_callback:
                log_callback(f"[pre-extract] restored segment exists, skipping: {seg_out}")
        restored_segments.append(seg_out)

    if log_callback:
        log_callback("[pre-extract] pasting restored segments back")
    paste_segments_gpu_or_fallback(
        base_path,
        restored_path,
        segments,
        restored_segments,
        log_callback=log_callback,
        process_callback=process_callback,
        bitrate_bps=output_bitrate_bps,
    )

    if not keep_segments:
        for seg in segments:
            for suffix in (f".seg{seg.seg_id:03d}.mp4", f".seg{seg.seg_id:03d}.restored.mp4"):
                path = os.path.join(base_dir, f"{stem}{suffix}")
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass
        _remove_file_quiet(segments_json, log_callback=log_callback)
        _remove_file_quiet(detections_jsonl, log_callback=log_callback)
    return PreExtractResult.OK


def _process_pre_extract_or_lada(base_path, restored_path, pre_extract_enabled,
                                 keep_intermediate=False, log_callback=None,
                                 process_callback=None, fine_conf=None,
                                 output_bitrate_bps: int | None = None) -> str:
    if not pre_extract_enabled:
        process_lada(
            base_path,
            restored_path,
            log_callback=log_callback,
            process_callback=process_callback,
            bitrate_bps=output_bitrate_bps,
        )
        return PreExtractResult.OK
    result = _run_pre_extract_branch(
        base_path,
        restored_path,
        keep_intermediate=keep_intermediate,
        log_callback=log_callback,
        process_callback=process_callback,
        fine_conf=fine_conf,
        output_bitrate_bps=output_bitrate_bps,
    )
    if result == PreExtractResult.CANCELLED:
        raise OperationCancelled("cancelled by user")
    if result == PreExtractResult.SCAN_FAILED:
        if log_callback:
            log_callback("[pre-extract] falling back to full-video lada/jasna after scan failure")
        process_lada(
            base_path,
            restored_path,
            log_callback=log_callback,
            process_callback=process_callback,
            bitrate_bps=output_bitrate_bps,
        )
        return PreExtractResult.OK
    return result


def _source_scan_supported(source_scan, log_callback=None) -> bool:
    if not source_scan:
        return False
    return True


def _clone_segment(seg, *, seg_id: int, start_s: float | None = None,
                   end_s: float | None = None, x_offset: int = 0):
    from utils.mosaic_prescan import MosaicSegment

    start = float(seg.start_s if start_s is None else start_s)
    end = float(seg.end_s if end_s is None else end_s)
    return MosaicSegment(
        seg_id=seg_id,
        start_s=start,
        end_s=end,
        start_s_kf=start,
        end_s_kf=end,
        x=int(seg.x) + int(x_offset),
        y=int(seg.y),
        w=int(seg.w),
        h=int(seg.h),
        conf_max=float(seg.conf_max),
    )


def _segment_time_overlap(a, b) -> tuple[float, float, float]:
    start = max(float(a.start_s), float(b.start_s))
    end = min(float(a.end_s), float(b.end_s))
    return start, end, max(0.0, end - start)


def _segment_spatial_overlap_ratio(a, b) -> float:
    ax1 = float(a.x)
    ay1 = float(a.y)
    ax2 = ax1 + float(a.w)
    ay2 = ay1 + float(a.h)
    bx1 = float(b.x)
    by1 = float(b.y)
    bx2 = bx1 + float(b.w)
    by2 = by1 + float(b.h)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(1.0, float(a.w) * float(a.h))
    area_b = max(1.0, float(b.w) * float(b.h))
    return inter / max(1.0, min(area_a, area_b))


def _segment_frame_bounds(seg, fps: float) -> tuple[int, int]:
    fps = float(fps or 30.0)
    start = max(0, int(round(float(seg.start_s) * fps)))
    end = max(start + 1, int(round(float(seg.end_s) * fps)))
    return start, end


def _paired_segment_cache_key(seg, fps: float) -> str:
    start_frame, end_frame = _segment_frame_bounds(seg, fps)
    return (
        f"f{start_frame:08d}-{end_frame:08d}."
        f"r{int(seg.x)}_{int(seg.y)}_{int(seg.w)}x{int(seg.h)}"
    )


def _paired_segment_paths(base_dir: str, stem: str, side_name: str, fish: str, seg, fps: float) -> tuple[str, str]:
    key = _paired_segment_cache_key(seg, fps)
    base = os.path.join(base_dir, f"{stem}_{side_name}{fish}.{key}")
    return f"{base}.mp4", f"{base}.restored.mp4"


def _paired_restored_related_paths(restored_mp4: str | Path) -> list[str]:
    from gpu_engine import restored_sidecar

    raw = restored_sidecar.raw_path_for_output(restored_mp4)
    return [str(restored_mp4), str(raw), str(restored_sidecar.sidecar_path_for(raw))]


def _file_nonempty(path: str | Path) -> bool:
    try:
        return os.path.exists(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def _restored_raw_valid(path: str | Path) -> bool:
    if not _file_nonempty(path):
        return False
    try:
        from gpu_engine import restored_sidecar

        return restored_sidecar.sidecar_path_for(path).exists()
    except Exception:
        return False


def _cleanup_orphan_paired_segment_files(base_dir: str, stem: str, fish: str,
                                         valid_paths: set[str], log_callback=None) -> None:
    valid = {os.path.abspath(path) for path in valid_paths}
    patterns = []
    for side_name in ("L", "R"):
        patterns.append(os.path.join(base_dir, f"{stem}_{side_name}{fish}.seg*.mp4"))
        patterns.append(os.path.join(base_dir, f"{stem}_{side_name}{fish}.f*.r*.mp4"))
        patterns.append(os.path.join(base_dir, f"{stem}_{side_name}{fish}.f*.r*.hevc"))
        patterns.append(os.path.join(base_dir, f"{stem}_{side_name}{fish}.f*.r*.json"))
    for pattern in patterns:
        for path in glob.glob(pattern):
            abs_path = os.path.abspath(path)
            if abs_path in valid:
                continue
            try:
                os.remove(abs_path)
                if log_callback:
                    log_callback(f"[source-scan] removed orphan fine segment cache: {abs_path}")
            except OSError:
                pass


def _pair_eye_segments_by_time(left_segments, right_segments, log_callback=None):
    min_overlap_s = max(0.0, float(app_config.get("pre_extract_pair_min_overlap_s", 0.25) or 0.25))
    min_spatial = max(0.0, float(app_config.get("pre_extract_pair_min_spatial_overlap", 0.05) or 0.05))
    keep_unmatched_conf = max(0.0, float(app_config.get("pre_extract_pair_keep_unmatched_conf", 0.60) or 0.60))
    candidates = []
    time_rejected = 0
    spatial_rejected = 0
    for li, left in enumerate(left_segments):
        for ri, right in enumerate(right_segments):
            start, end, overlap_s = _segment_time_overlap(left, right)
            if overlap_s < min_overlap_s:
                time_rejected += 1
                continue
            spatial = _segment_spatial_overlap_ratio(left, right)
            if spatial < min_spatial:
                spatial_rejected += 1
                continue
            candidates.append({
                "li": li,
                "ri": ri,
                "left": left,
                "right": right,
                "start": start,
                "end": end,
                "overlap_s": overlap_s,
                "spatial": spatial,
            })

    paired_left_items = []
    paired_right_items = []
    paired_items = []
    used_left_windows: dict[int, list[tuple[float, float]]] = {}
    used_right_windows: dict[int, list[tuple[float, float]]] = {}
    conflict_rejected = 0

    def _has_time_conflict(windows: dict[int, list[tuple[float, float]]],
                           idx: int, start: float, end: float) -> bool:
        for used_start, used_end in windows.get(idx, []):
            if max(float(used_start), float(start)) < min(float(used_end), float(end)) - 1e-6:
                return True
        return False

    for item in sorted(candidates, key=lambda c: (c["spatial"], c["overlap_s"]), reverse=True):
        if (
            _has_time_conflict(used_left_windows, item["li"], item["start"], item["end"])
            or _has_time_conflict(used_right_windows, item["ri"], item["start"], item["end"])
        ):
            conflict_rejected += 1
            continue
        left = item["left"]
        right = item["right"]
        used_left_windows.setdefault(item["li"], []).append((item["start"], item["end"]))
        used_right_windows.setdefault(item["ri"], []).append((item["start"], item["end"]))
        paired_items.append(item)

    paired_items.sort(key=lambda item: (
        float(item["start"]),
        float(item["end"]),
        int(item["left"].y),
        int(item["left"].x),
        int(item["right"].y),
        int(item["right"].x),
    ))
    for item in paired_items:
        paired_left_items.append(item)
        paired_right_items.append(item)
        if log_callback:
            log_callback(
                f"[source-scan] pair fine segment {len(paired_left_items) - 1}: "
                f"L{item['left'].seg_id}<->R{item['right'].seg_id}, "
                f"{item['start']:.3f}-{item['end']:.3f}s, spatial_overlap={item['spatial']:.2f}"
            )

    used_left = set(used_left_windows)
    used_right = set(used_right_windows)

    def _uncovered_windows(seg, used_windows: list[tuple[float, float]]) -> list[tuple[float, float]]:
        start = float(seg.start_s)
        end = float(seg.end_s)
        if end <= start:
            return []
        clipped = []
        for used_start, used_end in used_windows:
            s = max(start, float(used_start))
            e = min(end, float(used_end))
            if e > s:
                clipped.append((s, e))
        clipped.sort()
        out = []
        cursor = start
        for used_start, used_end in clipped:
            if used_start - cursor >= min_overlap_s:
                out.append((cursor, used_start))
            cursor = max(cursor, used_end)
        if end - cursor >= min_overlap_s:
            out.append((cursor, end))
        return out

    def _append_unmatched(side_name: str, source_segments, used_windows_by_idx, out_items) -> tuple[int, int]:
        kept = 0
        skipped_low_conf = 0
        for idx, seg in enumerate(source_segments):
            conf = float(getattr(seg, "conf_max", 0.0))
            windows = _uncovered_windows(seg, used_windows_by_idx.get(idx, []))
            if not windows:
                continue
            if conf < keep_unmatched_conf:
                skipped_low_conf += 1
                continue
            for start, end in windows:
                out_items.append({"segment": seg, "start": start, "end": end, "unmatched": True})
                kept += 1
                if log_callback:
                    log_callback(
                        f"[source-scan] keep unmatched {side_name} fine segment {seg.seg_id}: "
                        f"{start:.3f}-{end:.3f}s rect={seg.x},{seg.y},{seg.w}x{seg.h} "
                        f"conf={conf:.3f}>={keep_unmatched_conf:.2f}"
                    )
        return kept, skipped_low_conf

    kept_unmatched_left, low_unmatched_left = _append_unmatched(
        "left", left_segments, used_left_windows, paired_left_items
    )
    kept_unmatched_right, low_unmatched_right = _append_unmatched(
        "right", right_segments, used_right_windows, paired_right_items
    )

    def _materialize(items: list[dict], side: str):
        items.sort(key=lambda item: (
            float(item["start"]),
            float(item["end"]),
            int(item["segment"].y if item.get("unmatched") else item[side].y),
            int(item["segment"].x if item.get("unmatched") else item[side].x),
        ))
        out = []
        for item in items:
            seg = item["segment"] if item.get("unmatched") else item[side]
            out.append(_clone_segment(seg, seg_id=len(out), start_s=item["start"], end_s=item["end"]))
        merged = []
        for seg in out:
            if (
                merged
                and int(merged[-1].x) == int(seg.x)
                and int(merged[-1].y) == int(seg.y)
                and int(merged[-1].w) == int(seg.w)
                and int(merged[-1].h) == int(seg.h)
                and float(seg.start_s) <= float(merged[-1].end_s) + 1e-6
            ):
                merged[-1].end_s = max(float(merged[-1].end_s), float(seg.end_s))
                merged[-1].end_s_kf = max(float(merged[-1].end_s_kf), float(seg.end_s_kf))
                merged[-1].conf_max = max(float(merged[-1].conf_max), float(seg.conf_max))
            else:
                seg.seg_id = len(merged)
                merged.append(seg)
        for idx, seg in enumerate(merged):
            seg.seg_id = idx
        return merged

    paired_left = _materialize(paired_left_items, "left")
    paired_right = _materialize(paired_right_items, "right")

    skipped_left = len(left_segments) - len(used_left)
    skipped_right = len(right_segments) - len(used_right)
    if log_callback:
        for idx, seg in enumerate(left_segments):
            if idx not in used_left and float(getattr(seg, "conf_max", 0.0)) < keep_unmatched_conf:
                log_callback(
                    f"[source-scan] skip unmatched left fine segment {seg.seg_id}: "
                    f"{seg.start_s:.3f}-{seg.end_s:.3f}s rect={seg.x},{seg.y},{seg.w}x{seg.h} "
                    f"conf={float(getattr(seg, 'conf_max', 0.0)):.3f}<{keep_unmatched_conf:.2f}"
                )
        for idx, seg in enumerate(right_segments):
            if idx not in used_right and float(getattr(seg, "conf_max", 0.0)) < keep_unmatched_conf:
                log_callback(
                    f"[source-scan] skip unmatched right fine segment {seg.seg_id}: "
                    f"{seg.start_s:.3f}-{seg.end_s:.3f}s rect={seg.x},{seg.y},{seg.w}x{seg.h} "
                    f"conf={float(getattr(seg, 'conf_max', 0.0)):.3f}<{keep_unmatched_conf:.2f}"
                )
    if log_callback:
        log_callback(
            f"[source-scan] paired fine segments: pairs={len(paired_items)}, "
            f"left_paired={len(used_left)}/{len(left_segments)}, right_paired={len(used_right)}/{len(right_segments)}, "
            f"unmatched_left={skipped_left}, unmatched_right={skipped_right}, "
            f"output_left={len(paired_left)}, output_right={len(paired_right)}, "
            f"rejected_time={time_rejected}, rejected_spatial={spatial_rejected}, "
            f"rejected_conflict={conflict_rejected}, "
            f"kept_unmatched_left={kept_unmatched_left}, kept_unmatched_right={kept_unmatched_right}, "
            f"low_conf_unmatched_left={low_unmatched_left}, low_conf_unmatched_right={low_unmatched_right}"
        )
    return paired_left, paired_right


def _process_sbs_paired_pre_extract_clip(base_clip, output_file, *, use_fisheye: bool,
                                         keep_intermediate: bool,
                                         original_bitrate: int | None,
                                         keep_original_bitrate: bool,
                                         log_callback=None,
                                         process_callback=None,
                                         fine_conf=None) -> str:
    from gpu_engine import probe as gpu_probe
    from gpu_engine import files as gpu_files
    from gpu_engine import restored_sidecar
    from utils.mosaic_prescan import save_segments_json, scan_segments_gpu_transform
    from utils.segment_paster import build_paste_segments, paste_segments_gpu_or_fallback

    base_clip = os.path.abspath(base_clip)
    output_file = os.path.abspath(output_file)
    base_dir = os.path.dirname(base_clip)
    stem = os.path.splitext(os.path.basename(base_clip))[0]
    fine_conf = _resolve_fine_conf(fine_conf)
    if log_callback:
        log_callback(
            f"[source-scan] Stage 3 paired fine pre-extract: base={base_clip}, "
            f"use_fisheye={use_fisheye}, fine_conf={fine_conf:.2f}"
        )

    scan_token = gpu_files.CancelToken()
    if process_callback:
        process_callback(scan_token)
    try:
        left_segments = scan_segments_gpu_transform(
            base_clip,
            crop_mode="left",
            to_fisheye=use_fisheye,
            log_callback=log_callback,
            cancel_token=scan_token,
            min_conf=fine_conf,
        )
        right_segments = scan_segments_gpu_transform(
            base_clip,
            crop_mode="right",
            to_fisheye=use_fisheye,
            log_callback=log_callback,
            cancel_token=scan_token,
            min_conf=fine_conf,
        )
    except OperationCancelled as exc:
        if log_callback:
            log_callback(f"[source-scan] paired fine scan cancelled: {exc}")
        return PreExtractResult.CANCELLED
    except Exception as exc:
        if scan_token.cancelled:
            if log_callback:
                log_callback("[source-scan] paired fine scan cancelled")
            return PreExtractResult.CANCELLED
        if log_callback:
            log_callback(f"[source-scan] paired fine scan failed: {type(exc).__name__}: {exc}")
        return PreExtractResult.SCAN_FAILED

    left_segments, right_segments = _pair_eye_segments_by_time(left_segments, right_segments, log_callback=log_callback)
    if not left_segments and not right_segments:
        if log_callback:
            log_callback("[source-scan] no fine segments to process; keeping interval unchanged")
        shutil.copy2(base_clip, output_file)
        return PreExtractResult.NO_MOSAIC

    restored_paths = []
    segment_input_paths = []
    paste_segments = []
    meta = gpu_probe.probe_video(base_clip)
    fps = meta.source_fps or 30.0

    save_segments_json(
        left_segments,
        os.path.join(base_dir, f"{stem}_L{'_fisheye' if use_fisheye else ''}.segments.json"),
        source=base_clip,
        fps=fps,
    )
    save_segments_json(
        right_segments,
        os.path.join(base_dir, f"{stem}_R{'_fisheye' if use_fisheye else ''}.segments.json"),
        source=base_clip,
        fps=fps,
    )
    eye_w = int(meta.width // 2)
    rect_source_bitrate = int(original_bitrate or meta.bitrate_bps or 0)
    final_source_bitrate = int(original_bitrate or meta.bitrate_bps or 0)
    bitrate_bps = _resolve_pipeline_bitrate(
        "final",
        meta.width,
        meta.height,
        fps,
        final_source_bitrate,
        keep_original_bitrate,
        log_callback=log_callback,
    )
    fish = "_fisheye" if use_fisheye else ""
    keep_segments = bool(app_config.get("pre_extract_keep_segments", False)) or bool(keep_intermediate)
    raw_restore_enabled = (not keep_segments) and not use_fisheye
    expected_cache_paths = set()
    for side_name, segments in (("L", left_segments), ("R", right_segments)):
        for seg in segments:
            seg_in, seg_out = _paired_segment_paths(base_dir, stem, side_name, fish, seg, fps)
            expected_cache_paths.add(seg_in)
            expected_cache_paths.update(_paired_restored_related_paths(seg_out))
    _cleanup_orphan_paired_segment_files(base_dir, stem, fish, expected_cache_paths, log_callback=log_callback)

    extract_tasks = []

    def _add_extract_tasks(side: str, segments, x_offset: int):
        for seg in segments:
            side_name = "L" if side == "left" else "R"
            seg_in, seg_out = _paired_segment_paths(base_dir, stem, side_name, fish, seg, fps)
            cache_key = _paired_segment_cache_key(seg, fps)
            start_frame, end_frame = _segment_frame_bounds(seg, fps)
            rect_bitrate_bps = _resolve_pipeline_bitrate(
                "intermediate",
                seg.w,
                seg.h,
                fps,
                rect_source_bitrate,
                keep_original_bitrate,
                source_w=meta.width,
                source_h=meta.height,
                log_callback=log_callback,
            )
            rect_bitrate_kbps = _bitrate_bps_to_kbps(rect_bitrate_bps)
            rect_bitrate_label = f"{rect_bitrate_kbps}kbps" if rect_bitrate_kbps else "auto"
            if log_callback:
                log_callback(
                    f"[source-scan] fine {side_name} segment {seg.seg_id}: "
                    f"{seg.start_s:.3f}-{seg.end_s:.3f}s rect={seg.x},{seg.y},{seg.w}x{seg.h} "
                    f"key={cache_key} bitrate={rect_bitrate_label}"
                )
            extract_tasks.append({
                "order": len(extract_tasks),
                "side": side,
                "side_name": side_name,
                "seg": seg,
                "seg_in": seg_in,
                "seg_out": seg_out,
                "raw_seg_out": str(restored_sidecar.raw_path_for_output(seg_out)),
                "x_offset": int(x_offset),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "bitrate_bps": rect_bitrate_bps,
                "cache_key": cache_key,
                "cached": _file_nonempty(seg_in),
            })

    _add_extract_tasks("left", left_segments, 0)
    _add_extract_tasks("right", right_segments, eye_w)

    extract_group_max = max(1, int(app_config.get("pre_extract_extract_group_max", 8) or 8))
    pipeline_enabled = bool(app_config.get("pre_extract_pipeline_enabled", True))
    groups: dict[tuple[int, int], list[dict]] = {}
    for task in extract_tasks:
        groups.setdefault((int(task["start_frame"]), int(task["end_frame"])), []).append(task)
    group_items = [
        (group_idx, int(start_frame), int(end_frame), group_tasks)
        for group_idx, ((start_frame, end_frame), group_tasks) in enumerate(sorted(groups.items()), start=1)
    ]

    class _PipelineCancelGroup:
        def __init__(self):
            self._lock = threading.Lock()
            self._children = []
            self._cancelled = False

        def add(self, child) -> None:
            should_cancel = False
            with self._lock:
                if child is not None and not any(existing is child for existing in self._children):
                    self._children.append(child)
                should_cancel = self._cancelled
            if should_cancel:
                self._kill_child(child)

        def discard(self, child) -> None:
            with self._lock:
                self._children = [existing for existing in self._children if existing is not child]

        def kill(self) -> None:
            with self._lock:
                self._cancelled = True
                children = list(self._children)
            for child in children:
                self._kill_child(child)

        def terminate(self) -> None:
            self.kill()

        @property
        def cancelled(self) -> bool:
            with self._lock:
                return self._cancelled

        @staticmethod
        def _kill_child(child) -> None:
            if child is None:
                return
            try:
                child.kill()
            except Exception:
                try:
                    child.terminate()
                except Exception:
                    pass

    pipeline_cancel = _PipelineCancelGroup()

    def _register_pipeline_child(child) -> None:
        pipeline_cancel.add(child)
        if process_callback:
            process_callback(pipeline_cancel)

    def _release_pipeline_child(child) -> None:
        pipeline_cancel.discard(child)

    def _new_extract_token():
        token = gpu_files.CancelToken()
        _register_pipeline_child(token)
        return token

    def _raise_if_pipeline_cancelled() -> None:
        if pipeline_cancel.cancelled:
            raise OperationCancelled("cancelled by user")

    def _run_extract_group(group_idx: int, start_frame: int, end_frame: int, group_tasks: list[dict]) -> None:
        pending_tasks = [task for task in group_tasks if not _file_nonempty(task["seg_in"])]
        if not pending_tasks:
            return
        _raise_if_pipeline_cancelled()
        if len(pending_tasks) > extract_group_max:
            if log_callback:
                log_callback(
                    f"[source-scan] fine extract group {group_idx}: tasks={len(pending_tasks)} "
                    f"exceeds max={extract_group_max}; using per-rect extract"
                )
        elif len(pending_tasks) > 1:
            token = _new_extract_token()
            if log_callback:
                log_callback(
                    f"[source-scan] fine extract group {group_idx}: "
                    f"frames={start_frame}-{end_frame}, tasks={len(pending_tasks)}"
                )
            try:
                gpu_files.extract_multi_rect_clip(
                    base_clip,
                    [
                        {
                            "dst": task["seg_in"],
                            "crop_mode": task["side"],
                            "rect": (task["seg"].x, task["seg"].y, task["seg"].w, task["seg"].h),
                            "bitrate_bps": task["bitrate_bps"],
                            "label": f"{task['side_name']}:{task['cache_key']}",
                        }
                        for task in pending_tasks
                    ],
                    to_fisheye=use_fisheye,
                    start_sec=float(start_frame) / float(fps),
                    end_sec=float(end_frame) / float(fps),
                    keep_audio=False,
                    log_callback=log_callback,
                    cancel_token=token,
                )
            finally:
                _release_pipeline_child(token)
            return

        for task in pending_tasks:
            _raise_if_pipeline_cancelled()
            if not _file_nonempty(task["seg_in"]):
                token = _new_extract_token()
                try:
                    gpu_files.extract_transformed_rect_clip(
                        base_clip,
                        task["seg_in"],
                        crop_mode=task["side"],
                        rect=(task["seg"].x, task["seg"].y, task["seg"].w, task["seg"].h),
                        to_fisheye=use_fisheye,
                        start_sec=task["seg"].start_s,
                        end_sec=task["seg"].end_s,
                        cq=None if task["bitrate_bps"] else 18,
                        bitrate_bps=task["bitrate_bps"],
                        keep_audio=False,
                        log_callback=log_callback,
                        cancel_token=token,
                    )
                finally:
                    _release_pipeline_child(token)

    restore_results = [None] * len(extract_tasks)

    def _run_restore_task(task: dict) -> str:
        _raise_if_pipeline_cancelled()
        if not _file_nonempty(task["seg_in"]):
            raise RuntimeError(f"fine segment extract produced no output: {task['seg_in']}")
        if raw_restore_enabled and _restored_raw_valid(task["raw_seg_out"]):
            return task["raw_seg_out"]
        elif _file_nonempty(task["seg_out"]):
            return task["seg_out"]
        sidecar_metadata = {
            "rect": {
                "x": int(task["seg"].x),
                "y": int(task["seg"].y),
                "w": int(task["seg"].w),
                "h": int(task["seg"].h),
            },
            "time": {
                "start_s": float(task["seg"].start_s),
                "end_s": float(task["seg"].end_s),
                "start_frame": int(task["start_frame"]),
                "end_frame": int(task["end_frame"]),
            },
        }
        child_handles = []

        def _restore_process_callback(child) -> None:
            child_handles.append(child)
            _register_pipeline_child(child)

        try:
            return process_lada(
                task["seg_in"],
                task["seg_out"],
                log_callback=log_callback,
                process_callback=_restore_process_callback,
                produce_mp4=not raw_restore_enabled,
                sidecar_metadata=sidecar_metadata,
            )
        finally:
            for child in child_handles:
                _release_pipeline_child(child)

    def _store_restored_task(task: dict) -> None:
        restored_actual = _run_restore_task(task)
        restore_results[int(task["order"])] = (
            task["seg_in"],
            _clone_segment(task["seg"], seg_id=int(task["order"]), x_offset=task["x_offset"]),
            restored_actual,
        )

    def _run_extract_restore_sequential() -> None:
        for group_idx, start_frame, end_frame, group_tasks in group_items:
            _run_extract_group(group_idx, start_frame, end_frame, group_tasks)
        for task in extract_tasks:
            _store_restored_task(task)

    def _run_extract_restore_pipeline() -> None:
        if log_callback:
            log_callback(f"[source-scan] fine extract/restore pipeline enabled: groups={len(group_items)}, depth=1")
        ready_queue: queue.Queue = queue.Queue(maxsize=1)
        sentinel = object()
        producer_errors = []

        def _producer_put(item) -> bool:
            while not pipeline_cancel.cancelled:
                try:
                    ready_queue.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def _extract_producer() -> None:
            try:
                for item in group_items:
                    if pipeline_cancel.cancelled:
                        break
                    group_idx, start_frame, end_frame, group_tasks = item
                    _run_extract_group(group_idx, start_frame, end_frame, group_tasks)
                    if not _producer_put(item):
                        break
            except BaseException as exc:
                producer_errors.append(exc)
                pipeline_cancel.kill()
            finally:
                if not pipeline_cancel.cancelled:
                    while True:
                        try:
                            ready_queue.put(sentinel, timeout=0.1)
                            break
                        except queue.Full:
                            if pipeline_cancel.cancelled:
                                break

        producer = threading.Thread(target=_extract_producer, name="paired-pre-extract-producer", daemon=True)
        producer.start()
        try:
            while True:
                try:
                    item = ready_queue.get(timeout=0.1)
                except queue.Empty:
                    if producer_errors:
                        raise producer_errors[0]
                    if pipeline_cancel.cancelled:
                        raise OperationCancelled("cancelled by user")
                    if not producer.is_alive():
                        if producer_errors:
                            raise producer_errors[0]
                        if pipeline_cancel.cancelled:
                            raise OperationCancelled("cancelled by user")
                        break
                    continue
                if item is sentinel:
                    if producer_errors:
                        raise producer_errors[0]
                    break
                _group_idx, _start_frame, _end_frame, group_tasks = item
                for task in group_tasks:
                    try:
                        _store_restored_task(task)
                    except OperationCancelled:
                        if producer_errors:
                            raise producer_errors[0]
                        raise
                if producer_errors:
                    raise producer_errors[0]
        except BaseException:
            pipeline_cancel.kill()
            raise
        finally:
            producer.join(timeout=30.0)

    if pipeline_enabled and len(group_items) > 1:
        _run_extract_restore_pipeline()
    else:
        _run_extract_restore_sequential()

    for restored in restore_results:
        if restored is None:
            if pipeline_cancel.cancelled:
                raise OperationCancelled("cancelled by user")
            raise RuntimeError("paired fine segment restore did not complete")
        seg_in, paste_seg, restored_actual = restored
        segment_input_paths.append(seg_in)
        paste_seg.seg_id = len(paste_segments)
        paste_segments.append(paste_seg)
        restored_paths.append(restored_actual)

    if use_fisheye:
        if log_callback:
            log_callback("[source-scan] Stage 3: in-memory fisheye rect patch onto interval")
        token = gpu_files.CancelToken()
        if process_callback:
            process_callback(token)
        gpu_files.paste_fisheye_eye_rects_to_sbs_gpu(
            base_clip,
            output_file,
            build_paste_segments(base_clip, paste_segments, restored_paths),
            cq=None if bitrate_bps else 18,
            bitrate_bps=bitrate_bps,
            keep_audio=False,
            log_callback=log_callback,
            cancel_token=token,
        )
        cleanup_extra = []
    else:
        if log_callback:
            log_callback("[source-scan] Stage 3: paste paired rects onto interval")
        paste_segments_gpu_or_fallback(
            base_clip,
            output_file,
            paste_segments,
            restored_paths,
            keep_audio=False,
            log_callback=log_callback,
            process_callback=process_callback,
            bitrate_bps=bitrate_bps,
        )
        cleanup_extra = []

    if not keep_segments:
        for p in segment_input_paths + restored_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass
            try:
                if str(p).lower().endswith(".hevc"):
                    from gpu_engine import restored_sidecar

                    sidecar = restored_sidecar.sidecar_path_for(p)
                    if sidecar.exists():
                        sidecar.unlink()
            except OSError:
                pass
        for p in cleanup_extra:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass
    return PreExtractResult.OK


def _process_sbs_clip_to_output(input_file, output_file, *, use_fisheye: bool,
                                pre_extract_inner: bool,
                                keep_intermediate: bool,
                                original_bitrate: int | None,
                                keep_original_bitrate: bool,
                                start_time=None,
                                end_time=None,
                                work_dir: str | None = None,
                                work_stem: str | None = None,
                                split_keep_audio: bool = True,
                                log_callback=None,
                                process_callback=None,
                                fine_conf=None) -> None:
    directory = os.path.abspath(work_dir) if work_dir else os.path.dirname(os.path.abspath(input_file))
    os.makedirs(directory, exist_ok=True)
    stem = work_stem or os.path.splitext(os.path.basename(input_file))[0]
    file_l = os.path.join(directory, f"{stem}_L.mp4")
    file_r = os.path.join(directory, f"{stem}_R.mp4")
    file_l_restored = os.path.join(directory, f"{stem}_L.restored.mp4")
    file_r_restored = os.path.join(directory, f"{stem}_R.restored.mp4")
    file_l_fish = os.path.join(directory, f"{stem}_L_fisheye.mp4")
    file_r_fish = os.path.join(directory, f"{stem}_R_fisheye.mp4")
    file_l_fish_restored = os.path.join(directory, f"{stem}_L_fisheye.restored.mp4")
    file_r_fish_restored = os.path.join(directory, f"{stem}_R_fisheye.restored.mp4")
    pre_extract_enabled = _pre_extract_supported(pre_extract_inner, log_callback)
    eye_intermediate_bitrate = None
    try:
        from gpu_engine import probe as gpu_probe

        src_meta = gpu_probe.probe_video(input_file)
        eye_intermediate_bitrate = _resolve_pipeline_bitrate(
            "intermediate",
            max(1, int(src_meta.width // 2)),
            src_meta.height,
            src_meta.source_fps or 30.0,
            int(original_bitrate / 2) if original_bitrate else None,
            keep_original_bitrate,
            log_callback=log_callback,
        )
    except Exception:
        eye_intermediate_bitrate = None

    if use_fisheye:
        if log_callback:
            log_callback(
                f"[source-scan] Stage 3: split source interval + VR->Fisheye "
                f"{start_time or 'START'}-{end_time or 'END'}"
            )
        split_video_dual_fisheye(
            input_file, file_l_fish, file_r_fish,
            start_time, end_time, log_callback, process_callback,
            keep_audio=split_keep_audio,
        )
        if log_callback:
            log_callback("[source-scan] Stage 3: restore fisheye eyes")
        _process_pre_extract_or_lada(file_l_fish, file_l_fish_restored, pre_extract_enabled, keep_intermediate, log_callback, process_callback, fine_conf=fine_conf, output_bitrate_bps=eye_intermediate_bitrate)
        _process_pre_extract_or_lada(file_r_fish, file_r_fish_restored, pre_extract_enabled, keep_intermediate, log_callback, process_callback, fine_conf=fine_conf, output_bitrate_bps=eye_intermediate_bitrate)
        if log_callback:
            log_callback("[source-scan] Stage 3: Fisheye->VR + merge")
        merge_videos_fisheye(file_l_fish_restored, file_r_fish_restored, output_file, original_bitrate, keep_original_bitrate, log_callback, process_callback)
        cleanup = [file_l_fish, file_r_fish, file_l_fish_restored, file_r_fish_restored]
    else:
        if log_callback:
            log_callback(
                f"[source-scan] Stage 3: split source interval "
                f"{start_time or 'START'}-{end_time or 'END'}"
            )
        split_video_dual(
            input_file, file_l, file_r,
            start_time, end_time, log_callback, process_callback,
            keep_audio=split_keep_audio,
        )
        if log_callback:
            log_callback("[source-scan] Stage 3: restore eyes")
        _process_pre_extract_or_lada(file_l, file_l_restored, pre_extract_enabled, keep_intermediate, log_callback, process_callback, fine_conf=fine_conf, output_bitrate_bps=eye_intermediate_bitrate)
        _process_pre_extract_or_lada(file_r, file_r_restored, pre_extract_enabled, keep_intermediate, log_callback, process_callback, fine_conf=fine_conf, output_bitrate_bps=eye_intermediate_bitrate)
        if log_callback:
            log_callback("[source-scan] Stage 3: merge SBS")
        merge_videos(file_l_restored, file_r_restored, output_file, original_bitrate, keep_original_bitrate, log_callback, process_callback)
        cleanup = [file_l, file_r, file_l_restored, file_r_restored]

    if not keep_intermediate:
        for path in cleanup:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass


def _process_single_eye_clip_to_output(input_file, output_file, *, eye_mode: int,
                                       use_fisheye: bool,
                                       pre_extract_inner: bool,
                                       keep_intermediate: bool,
                                       final_bitrate_kbps: int | None,
                                       start_time=None,
                                       end_time=None,
                                       work_dir: str | None = None,
                                       work_stem: str | None = None,
                                       split_keep_audio: bool = True,
                                       log_callback=None,
                                       process_callback=None,
                                       fine_conf=None) -> None:
    directory = os.path.abspath(work_dir) if work_dir else os.path.dirname(os.path.abspath(input_file))
    os.makedirs(directory, exist_ok=True)
    stem = work_stem or os.path.splitext(os.path.basename(input_file))[0]
    side_suffix = "_L" if eye_mode == 1 else "_R"
    crop_filter = "crop=iw/2:ih:0:0" if eye_mode == 1 else "crop=iw/2:ih:iw/2:0"
    file_cut = os.path.join(directory, f"{stem}{side_suffix}.mp4")
    file_cut_fish = os.path.join(directory, f"{stem}{side_suffix}_fisheye.mp4")
    file_cut_fish_restored = os.path.join(directory, f"{stem}{side_suffix}_fisheye.restored.mp4")
    pre_extract_enabled = _pre_extract_supported(pre_extract_inner, log_callback)
    final_bitrate_bps = int(final_bitrate_kbps * 1000) if final_bitrate_kbps else None

    if use_fisheye:
        if log_callback:
            log_callback(
                f"[source-scan] Stage 3: split single-eye source interval + VR->Fisheye "
                f"{start_time or 'START'}-{end_time or 'END'}"
            )
        split_video_fisheye(
            input_file, file_cut_fish, crop_filter,
            start_time, end_time, log_callback, process_callback,
            keep_audio=split_keep_audio,
        )
        _process_pre_extract_or_lada(file_cut_fish, file_cut_fish_restored, pre_extract_enabled, keep_intermediate, log_callback, process_callback, fine_conf=fine_conf)
        convert_projection(file_cut_fish_restored, output_file, "fisheye:hequirect", final_bitrate_kbps=final_bitrate_kbps, log_callback=log_callback, process_callback=process_callback)
        cleanup = [file_cut_fish, file_cut_fish_restored]
    else:
        if log_callback:
            log_callback(
                f"[source-scan] Stage 3: split single-eye source interval "
                f"{start_time or 'START'}-{end_time or 'END'}"
            )
        split_video(
            input_file, file_cut, crop_filter,
            start_time, end_time, log_callback, process_callback,
            keep_audio=split_keep_audio,
        )
        _process_pre_extract_or_lada(
            file_cut,
            output_file,
            pre_extract_enabled,
            keep_intermediate,
            log_callback,
            process_callback,
            fine_conf=fine_conf,
            output_bitrate_bps=final_bitrate_bps,
        )
        cleanup = [file_cut]

    if not keep_intermediate:
        for path in cleanup:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass


def _run_source_scan_branch(input_file, final_output, *, use_fisheye: bool,
                            pre_extract_inner: bool,
                            keep_intermediate: bool,
                            keep_original_bitrate: bool,
                            start_time=None,
                            end_time=None,
                            mode: str = "sbs",
                            eye_mode: int | None = None,
                            log_callback=None,
                            process_callback=None,
                            fine_conf=None) -> str:
    from gpu_engine import probe as gpu_probe
    from gpu_engine.files import CancelToken
    from utils.keyframe_cutter import cut_source_by_intervals
    from utils.sbs_concat import concat_timeline
    from utils.source_time_scanner import save_source_intervals_json, scan_source_time_segments

    if start_time or end_time:
        if log_callback:
            log_callback("[source-scan] start/end subranges are not wired yet; falling back to the normal path")
        return PreExtractResult.SCAN_FAILED

    meta = gpu_probe.probe_video(input_file)
    if meta.is_hdr or meta.is_bt2020:
        if log_callback:
            log_callback("[source-scan] HDR/bt2020 source is not enabled for source-scan; falling back to the normal path")
        return PreExtractResult.SCAN_FAILED

    final_path = os.path.abspath(final_output)
    tmp_dir = os.path.join(os.path.dirname(final_path), f"{os.path.splitext(os.path.basename(final_path))[0]}_scan_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    scan_input = os.path.abspath(input_file)
    keep_segments = bool(app_config.get("source_scan_keep_segments", False)) or bool(keep_intermediate)
    intervals_json = os.path.splitext(final_path)[0] + ".source_intervals.json"
    scan_stem = os.path.splitext(os.path.basename(scan_input))[0]
    source_detections_jsonl = os.path.join(os.path.dirname(scan_input), f"{scan_stem}.detections.jsonl")

    try:
        if log_callback:
            log_callback(f"[source-scan] Stage 1 scanning source: {scan_input}")
            log_callback(
                f"[source-scan] mode={mode}, use_fisheye={use_fisheye}, "
                f"pre_extract_inner={pre_extract_inner}, keep_intermediate={keep_intermediate}"
            )
            log_callback(f"[source-scan] temp dir: {tmp_dir}")
            log_callback(f"[source-scan] final output: {final_path}")
        scan_token = CancelToken()
        if process_callback:
            process_callback(scan_token)
        try:
            intervals = scan_source_time_segments(scan_input, log_callback=log_callback, cancel_token=scan_token)
        except OperationCancelled as exc:
            if log_callback:
                log_callback(f"[source-scan] scan cancelled: {exc}")
            return PreExtractResult.CANCELLED
        except Exception as exc:
            if scan_token.cancelled:
                if log_callback:
                    log_callback("[source-scan] scan cancelled")
                return PreExtractResult.CANCELLED
            if log_callback:
                log_callback(f"[source-scan] scan failed: {type(exc).__name__}: {exc}")
            return PreExtractResult.SCAN_FAILED

        save_source_intervals_json(intervals, intervals_json, source=scan_input)
        if log_callback:
            log_callback(f"[source-scan] saved intervals: {intervals_json}")
            log_callback(f"[source-scan] interval count: {len(intervals)}")
            for idx, interval in enumerate(intervals[:20]):
                log_callback(
                    f"[source-scan] interval {idx}: "
                    f"{float(interval.start_s):.3f}-{float(interval.end_s):.3f}s "
                    f"duration={float(interval.duration_s):.3f}s conf={float(interval.conf_max):.3f}"
                )
            if len(intervals) > 20:
                log_callback(f"[source-scan] ... {len(intervals) - 20} more intervals")
        if not intervals:
            if log_callback:
                log_callback(f"[source-scan] no mosaic in entire video, skipping: {input_file}")
            return PreExtractResult.NO_MOSAIC

        # Decide whether to materialize gaps by mode:
        #   SBS: concat_timeline_hevc_fast can seek the same source file a second
        #        time with -ss. If IDR PTS drifts slightly from ffprobe's reported
        #        values, AnnexB extraction can include non-IDR starting frames,
        #        causing players to stall at the second gap in the latter half.
        #        Materializing gaps as standalone mp4 files avoids later AnnexB
        #        extraction with -ss. Cost: about 3-5GB of temporary files, cleaned
        #        after processing.
        #   single_eye: virtual gaps are later cropped again to the selected eye by
        #        extract_clip, so materialization is not useful.
        materialize_gaps = (mode == "sbs")
        timeline = cut_source_by_intervals(
            scan_input,
            intervals,
            tmp_dir,
            None,
            log_callback=log_callback,
            process_callback=process_callback,
            materialize_gaps=materialize_gaps,
            materialize_mosaic=True,
        )
        timeline_json = os.path.join(tmp_dir, "timeline.json")
        with open(timeline_json, "w", encoding="utf-8") as f:
            json.dump({"entries": [entry.__dict__ | {"path": str(entry.path)} for entry in timeline]}, f, ensure_ascii=False, indent=2)
        if log_callback:
            mosaic_count = sum(1 for entry in timeline if entry.kind == "mosaic")
            gap_count = sum(1 for entry in timeline if entry.kind == "gap")
            log_callback(f"[source-scan] saved timeline: {timeline_json}")
            log_callback(f"[source-scan] timeline entries: mosaic={mosaic_count}, gap={gap_count}")
            for idx, entry in enumerate(timeline[:30]):
                log_callback(
                    f"[source-scan] timeline {idx}: kind={entry.kind} "
                    f"{float(entry.start_s):.3f}-{float(entry.end_s):.3f}s path={entry.path}"
                )
            if len(timeline) > 30:
                log_callback(f"[source-scan] ... {len(timeline) - 30} more timeline entries")

        original_bitrate = get_video_bitrate(scan_input, log_callback)
        single_eye_final_bitrate = _resolve_pipeline_bitrate(
            "final",
            max(1, int(meta.width // 2)),
            meta.height,
            meta.source_fps or 30.0,
            int(original_bitrate / 2) if original_bitrate else None,
            keep_original_bitrate,
            log_callback=log_callback,
        )
        final_bitrate_kbps = _bitrate_bps_to_kbps(single_eye_final_bitrate)
        for entry in timeline:
            if entry.kind == "mosaic":
                restored = entry.path.with_name(f"{entry.path.stem}.restored{entry.path.suffix}")
                if mode == "single_eye":
                    _process_single_eye_clip_to_output(
                        str(entry.path),
                        str(restored),
                        eye_mode=int(eye_mode or 1),
                        use_fisheye=use_fisheye,
                        pre_extract_inner=pre_extract_inner,
                        keep_intermediate=keep_intermediate,
                        final_bitrate_kbps=final_bitrate_kbps,
                        log_callback=log_callback,
                        process_callback=process_callback,
                        fine_conf=fine_conf,
                    )
                else:
                    if pre_extract_inner:
                        paired_result = _process_sbs_paired_pre_extract_clip(
                            str(entry.path),
                            str(restored),
                            use_fisheye=use_fisheye,
                            keep_intermediate=keep_intermediate,
                            original_bitrate=original_bitrate,
                            keep_original_bitrate=keep_original_bitrate,
                            log_callback=log_callback,
                            process_callback=process_callback,
                            fine_conf=fine_conf,
                        )
                        if paired_result == PreExtractResult.CANCELLED:
                            raise OperationCancelled("cancelled by user")
                        if paired_result == PreExtractResult.SCAN_FAILED:
                            if log_callback:
                                log_callback("[source-scan] paired fine path failed; falling back to full-eye restore")
                            _process_sbs_clip_to_output(
                                str(entry.path),
                                str(restored),
                                use_fisheye=use_fisheye,
                                pre_extract_inner=False,
                                keep_intermediate=keep_intermediate,
                                original_bitrate=original_bitrate,
                                keep_original_bitrate=keep_original_bitrate,
                                log_callback=log_callback,
                                process_callback=process_callback,
                                fine_conf=fine_conf,
                            )
                    else:
                        _process_sbs_clip_to_output(
                            str(entry.path),
                            str(restored),
                            use_fisheye=use_fisheye,
                            pre_extract_inner=False,
                            keep_intermediate=keep_intermediate,
                            original_bitrate=original_bitrate,
                            keep_original_bitrate=keep_original_bitrate,
                            log_callback=log_callback,
                            process_callback=process_callback,
                            fine_conf=fine_conf,
                        )
                entry.path = restored
                entry.inpoint_s = None
                entry.outpoint_s = None

        if mode == "single_eye":
            from gpu_engine.files import extract_clip

            crop_mode = "left" if int(eye_mode or 1) == 1 else "right"
            gap_bitrate = single_eye_final_bitrate
            gap_idx = 0
            for idx, entry in enumerate(timeline):
                if entry.kind != "gap":
                    continue
                gap_out = Path(tmp_dir) / f"gap_seg{gap_idx:03d}_{crop_mode}{Path(scan_input).suffix or '.mp4'}"
                gap_idx += 1
                interval_start = getattr(entry, "inpoint_s", None)
                interval_end = getattr(entry, "outpoint_s", None)
                if log_callback:
                    log_callback(
                        f"[source-scan] Stage 4 prepare single-eye gap {idx}: "
                        f"{entry.path} {interval_start if interval_start is not None else 'START'}-"
                        f"{interval_end if interval_end is not None else 'END'} -> {gap_out.name}"
                    )
                gap_token = CancelToken()
                if process_callback:
                    process_callback(gap_token)
                extract_clip(
                    entry.path,
                    gap_out,
                    crop_mode=crop_mode,
                    to_fisheye=False,
                    start_sec=interval_start,
                    end_sec=interval_end,
                    cq=None if gap_bitrate else 18,
                    bitrate_bps=gap_bitrate,
                    keep_audio=False,
                    log_callback=log_callback,
                    cancel_token=gap_token,
                )
                entry.path = gap_out
                entry.inpoint_s = None
                entry.outpoint_s = None

        if log_callback:
            log_callback("[source-scan] Stage 4 merge timeline")
        if mode == "single_eye":
            concat_bitrate = single_eye_final_bitrate
        else:
            concat_bitrate = _resolve_pipeline_bitrate(
                "final",
                meta.width,
                meta.height,
                meta.source_fps or 30.0,
                original_bitrate or meta.bitrate_bps,
                keep_original_bitrate,
                log_callback=log_callback,
            )
        if mode == "sbs":
            from gpu_engine.files import replace_timeline_segments_gpu
            from utils.sbs_concat import concat_timeline_hevc_fast

            merge_mode = str(app_config.get("source_scan_final_merge_mode", "auto") or "auto").strip().lower()
            if merge_mode not in {"auto", "fast", "gpu"}:
                merge_mode = "auto"
            if merge_mode in {"auto", "fast"}:
                try:
                    concat_timeline_hevc_fast(
                        timeline,
                        final_path,
                        source_src=scan_input,
                        audio_source=scan_input,
                        log_callback=log_callback,
                        process_callback=process_callback,
                    )
                    _log_final_bitrate_summary(final_path, concat_bitrate, log_callback)
                    return PreExtractResult.OK
                except OperationCancelled:
                    raise
                except Exception as exc:
                    if log_callback:
                        log_callback(
                            f"[source-scan] fast HEVC merge failed: {type(exc).__name__}: {exc}"
                        )
                    _remove_file_quiet(final_path)
                    if merge_mode == "fast":
                        raise
                    if log_callback:
                        log_callback("[source-scan] falling back to GPU timeline merge")

            merge_token = CancelToken()
            if process_callback:
                process_callback(merge_token)
            replace_timeline_segments_gpu(
                scan_input,
                final_path,
                timeline,
                audio_source=scan_input,
                cq=None if concat_bitrate else 18,
                bitrate_bps=concat_bitrate,
                log_callback=log_callback,
                cancel_token=merge_token,
            )
        else:
            concat_timeline(
                timeline,
                final_path,
                audio_source=scan_input,
                log_callback=log_callback,
                process_callback=process_callback,
                reencode="auto",
                cq=None if concat_bitrate else 18,
                bitrate_bps=concat_bitrate,
            )
        _log_final_bitrate_summary(final_path, concat_bitrate, log_callback)
        return PreExtractResult.OK
    finally:
        if not keep_segments:
            try:
                shutil.rmtree(tmp_dir)
            except OSError:
                pass
            _remove_file_quiet(intervals_json, log_callback=log_callback)
            _remove_file_quiet(source_detections_jsonl, log_callback=log_callback)


def _resolve_merge_final_bitrate(left_file, right_file, original_bitrate, keep_original_bitrate,
                                 log_callback=None) -> int | None:
    try:
        from gpu_engine import probe as gpu_probe

        left_meta = gpu_probe.probe_video(left_file)
        right_meta = gpu_probe.probe_video(right_file)
        out_w = int(left_meta.width) + int(right_meta.width)
        out_h = max(int(left_meta.height), int(right_meta.height))
        fps = left_meta.source_fps or right_meta.source_fps or 30.0
        source_bps = int(original_bitrate or 0)
        if source_bps <= 0:
            source_bps = int(left_meta.bitrate_bps or 0) + int(right_meta.bitrate_bps or 0)
        return _resolve_pipeline_bitrate(
            "final",
            out_w,
            out_h,
            fps,
            source_bps,
            keep_original_bitrate,
            log_callback=log_callback,
        )
    except Exception:
        return int(original_bitrate) if (keep_original_bitrate and original_bitrate) else None


def merge_videos(left_file, right_file, output_file, original_bitrate, keep_original_bitrate=False, log_callback=None, process_callback=None):
    """Merge left and right eyes into SBS. Prefer GPU and fall back to ffmpeg on failure."""
    try:
        from gpu_engine import probe as gpu_probe, fallback as gpu_fallback, files as gpu_files
        _, d1 = gpu_probe.route(left_file)
        _, d2 = gpu_probe.route(right_file)
        target_bitrate_bps = _resolve_merge_final_bitrate(
            left_file,
            right_file,
            original_bitrate,
            keep_original_bitrate,
            log_callback=log_callback,
        )

        def _gpu_fn():
            token = gpu_files.CancelToken()
            if process_callback:
                process_callback(token)
            gpu_files.combine_video(
                left_file, right_file, output_file, "left_right", from_fisheye=False,
                cq=None if target_bitrate_bps else 18,
                bitrate_bps=target_bitrate_bps,
                keep_audio=True, log_callback=log_callback, cancel_token=token,
            )
            return True

        def _ffmpeg_fn():
            return _merge_videos_ffmpeg(
                left_file,
                right_file,
                output_file,
                original_bitrate,
                keep_original_bitrate,
                log_callback,
                process_callback,
                bitrate_bps=target_bitrate_bps,
            )

        if log_callback: log_callback(f"Merging: {left_file} + {right_file} -> {output_file}")
        return gpu_fallback.run_with_fallback(
            _gpu_fn, _ffmpeg_fn, gpu_eligible=(d1.is_gpu and d2.is_gpu),
            log_callback=log_callback, label="merge",
        )
    except Exception as e:
        if log_callback: log_callback(f"Merge error: {e}")
        raise


def _merge_videos_ffmpeg(left_file, right_file, output_file, original_bitrate, keep_original_bitrate=False, log_callback=None, process_callback=None, bitrate_bps: int | None = None):
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error","-stats",
        "-hwaccel", "cuda", "-c:v", "hevc_cuvid", "-i", left_file,
        "-hwaccel", "cuda", "-c:v", "hevc_cuvid", "-i", right_file,
        "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]",
        "-map", "[v]", "-map", "0:a?",
        "-c:a", "copy"
    ]
    
    if bitrate_bps and bitrate_bps > 0:
        target_kbps = max(1, int(bitrate_bps / 1000))
        target_bitrate = f"{target_kbps}k"
        max_rate = f"{int(target_kbps * 1.2)}k"
        buf_size = f"{int(target_kbps * 2)}k"
        cmd.extend([
            "-c:v", "hevc_nvenc", 
            "-preset", "p7", 
            "-rc", "vbr",
            "-b:v", target_bitrate,
            "-maxrate:v", max_rate,
            "-bufsize:v", buf_size,
            "-shortest", output_file, "-y"
        ])
    else:
        # Use CQ mode for quality control
        cmd.extend([
            "-c:v", "hevc_nvenc", 
            "-preset", "p7", 
            "-cq", "18",
            "-shortest", output_file, "-y"
        ])
    if log_callback: log_callback(f"Merging: {left_file} + {right_file} -> {output_file}")
    run_process(cmd, log_callback, process_callback)

def merge_videos_fisheye(left_fisheye_file, right_fisheye_file, output_file, original_bitrate, keep_original_bitrate=False, log_callback=None, process_callback=None):
    """Fisheye->VR + left/right merge. Prefer GPU and fall back to ffmpeg on failure."""
    try:
        from gpu_engine import probe as gpu_probe, fallback as gpu_fallback, files as gpu_files
        _, d1 = gpu_probe.route(left_fisheye_file)
        _, d2 = gpu_probe.route(right_fisheye_file)
        target_bitrate_bps = _resolve_merge_final_bitrate(
            left_fisheye_file,
            right_fisheye_file,
            original_bitrate,
            keep_original_bitrate,
            log_callback=log_callback,
        )

        def _gpu_fn():
            token = gpu_files.CancelToken()
            if process_callback:
                process_callback(token)
            gpu_files.combine_video(
                left_fisheye_file, right_fisheye_file, output_file, "left_right",
                from_fisheye=True,
                cq=None if target_bitrate_bps else 18,
                bitrate_bps=target_bitrate_bps,
                keep_audio=True, log_callback=log_callback, cancel_token=token,
            )
            return True

        def _ffmpeg_fn():
            return _merge_videos_fisheye_ffmpeg(
                left_fisheye_file,
                right_fisheye_file,
                output_file,
                original_bitrate,
                keep_original_bitrate,
                log_callback,
                process_callback,
                bitrate_bps=target_bitrate_bps,
            )

        if log_callback: log_callback(f"Fisheye->VR + Merging: {left_fisheye_file} + {right_fisheye_file} -> {output_file}")
        return gpu_fallback.run_with_fallback(
            _gpu_fn, _ffmpeg_fn, gpu_eligible=(d1.is_gpu and d2.is_gpu),
            log_callback=log_callback, label="merge_fisheye",
        )
    except Exception as e:
        if log_callback: log_callback(f"Merge fisheye error: {e}")
        raise


def _merge_videos_fisheye_ffmpeg(left_fisheye_file, right_fisheye_file, output_file, original_bitrate, keep_original_bitrate=False, log_callback=None, process_callback=None, bitrate_bps: int | None = None):
    """Convert fisheye to VR and merge left/right videos in a single ffmpeg call."""
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error", "-stats",
        "-hwaccel", "cuda", "-c:v", "hevc_cuvid", "-i", left_fisheye_file,
        "-hwaccel", "cuda", "-c:v", "hevc_cuvid", "-i", right_fisheye_file,
        "-filter_complex", "[0:v]v360=fisheye:hequirect[left];[1:v]v360=fisheye:hequirect[right];[left][right]hstack=inputs=2[v]",
        "-map", "[v]", "-map", "0:a?",
        "-c:a", "copy"
    ]
    
    if bitrate_bps and bitrate_bps > 0:
        target_kbps = max(1, int(bitrate_bps / 1000))
        target_bitrate = f"{target_kbps}k"
        max_rate = f"{int(target_kbps * 1.2)}k"
        buf_size = f"{int(target_kbps * 2)}k"
        cmd.extend([
            "-c:v", "hevc_nvenc", 
            "-preset", "p7", 
            "-rc", "vbr",
            "-b:v", target_bitrate,
            "-maxrate:v", max_rate,
            "-bufsize:v", buf_size,
            "-shortest", output_file, "-y"
        ])
    else:
        # Use CQ mode for quality control
        cmd.extend([
            "-c:v", "hevc_nvenc", 
            "-preset", "p7", 
            "-cq", "18",
            "-shortest", output_file, "-y"
        ])
    if log_callback: log_callback(f"Fisheye->VR + Merging: {left_fisheye_file} + {right_fisheye_file} -> {output_file}")
    run_process(cmd, log_callback, process_callback)

_PROJ_KIND = {"hequirect:fisheye": "heq2fisheye", "fisheye:hequirect": "fisheye2heq"}


def convert_projection(input_file, output_file, projection, log_callback=None, process_callback=None, final_bitrate_kbps=None):
    """Single-file projection conversion. Prefer GPU and fall back to ffmpeg on failure. projection: 'hequirect:fisheye'|'fisheye:hequirect'."""
    kind = _PROJ_KIND.get(projection)
    try:
        from gpu_engine import probe as gpu_probe, fallback as gpu_fallback, files as gpu_files
        meta, decision = gpu_probe.route(input_file)

        def _gpu_fn():
            token = gpu_files.CancelToken()
            if process_callback:
                process_callback(token)
            gpu_files.vr_projection(
                input_file, output_file, kind, dual_screen=False,
                cq=None if final_bitrate_kbps else 18,
                bitrate_bps=int(final_bitrate_kbps * 1000) if final_bitrate_kbps else None,
                keep_audio=True, log_callback=log_callback, cancel_token=token,
            )
            return True

        def _ffmpeg_fn():
            return _convert_projection_ffmpeg(input_file, output_file, projection, log_callback, process_callback, final_bitrate_kbps)

        if log_callback: log_callback(f"Converting Projection ({projection}): {os.path.basename(input_file)}")
        return gpu_fallback.run_with_fallback(
            _gpu_fn, _ffmpeg_fn, gpu_eligible=(decision.is_gpu and kind is not None),
            log_callback=log_callback, label=f"v360 {projection}",
        )
    except Exception as e:
        if log_callback: log_callback(f"Convert projection error: {e}")
        raise


def _convert_projection_ffmpeg(input_file, output_file, projection, log_callback=None, process_callback=None, final_bitrate_kbps=None):
    # projection example: "hequirect:fisheye" or "fisheye:hequirect"
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error", "-stats",
        "-hwaccel", "cuda", "-c:v", "hevc_cuvid", "-i", input_file,
        "-vf", f"v360={projection}",
        "-c:a", "copy"
    ]
    
    if final_bitrate_kbps:
        cmd.extend([
            "-c:v", "hevc_nvenc", 
            "-preset", "p7", 
            "-rc", "vbr",
            "-b:v", f"{final_bitrate_kbps}k",
            "-maxrate:v", f"{int(final_bitrate_kbps * 1.2)}k",
            "-bufsize:v", f"{int(final_bitrate_kbps * 2)}k",
        ])
    else:
        cmd.extend([
            "-c:v", "hevc_nvenc", 
            "-preset", "p7", 
            "-cq", "18",
        ])
    
    cmd.extend([output_file, "-y"])
    if log_callback: log_callback(f"Converting Projection ({projection}): {os.path.basename(input_file)}")
    run_process(cmd, log_callback, process_callback)

# --- Workflows ---

def _native_stream_allowed(keep_intermediate=False) -> bool:
    # Section 4.5 no-intermediate-file streaming path: disabled by default. Long
    # segment profiling showed it is about 4x slower than the file path: 240
    # frames at 0.69fps vs 2.68fps for file mode. The streaming frame source runs
    # CuPy geometry plus whole-device synchronization on the same GPU, strongly
    # contending with torch YOLO/BasicVSR++ inference and slowing model inference
    # by 7-9x. A short-segment 3.9fps result was warmup noise and is not reliable.
    # The file path, around 22 minutes, is fastest and close to baseline. Set
    # app_config native_stream_enabled=True to experiment. See prompt/HANDOVER_20260531.md.
    if keep_intermediate or not engine_runner.is_native_engine():
        return False
    try:
        from utils import app_config
        if not app_config.get("native_stream_enabled", False):
            return False
        from gpu_engine.fallback import get_backend_mode
        return get_backend_mode() != "ffmpeg"
    except Exception:
        return False


def _native_stream_failure_is_fatal() -> bool:
    try:
        from gpu_engine.fallback import get_backend_mode
        return get_backend_mode() == "gpu"
    except Exception:
        return False


def _run_native_sbs_stream(input_file, output_file, start_time, end_time, use_fisheye,
                           original_bitrate, keep_original_bitrate,
                           log_callback=None, process_callback=None) -> bool:
    """native_gpu one_click dual-eye streaming path. Return True on success; in auto mode return False so the legacy path can take over."""
    if not _native_stream_allowed(False):
        return False
    try:
        from gpu_engine import native_mosaic
        from gpu_engine.files import CancelToken
        from gpu_engine.fallback import OperationCancelled

        token = CancelToken()
        if process_callback:
            process_callback(token)
        try:
            from gpu_engine import probe as gpu_probe

            meta = gpu_probe.probe_video(input_file)
            bitrate_bps = _resolve_pipeline_bitrate(
                "final",
                meta.width,
                meta.height,
                meta.source_fps or 30.0,
                original_bitrate or meta.bitrate_bps,
                keep_original_bitrate,
                log_callback=log_callback,
            )
        except Exception:
            bitrate_bps = int(original_bitrate) if (keep_original_bitrate and original_bitrate) else None
        if log_callback:
            stage = "fisheye/process/defisheye" if use_fisheye else "process"
            log_callback(f"--- NativeGPU SBS fused stream: {stage} without intermediate files ---")
        native_mosaic.restore_sbs_stream(
            input_file, output_file,
            use_fisheye=use_fisheye,
            start_sec=_time_to_sec(start_time),
            end_sec=_time_to_sec(end_time),
            bitrate_bps=bitrate_bps,
            log_callback=log_callback,
            cancel_token=token,
        )
        return True
    except Exception as e:
        try:
            if isinstance(e, OperationCancelled):
                raise
        except UnboundLocalError:
            pass
        if _native_stream_failure_is_fatal():
            raise
        if log_callback:
            log_callback(f"[native-stream fallback] {type(e).__name__}: {e}; using legacy intermediate-file path")
        return False


def _run_native_single_eye_stream(input_file, output_file, eye_mode, start_time, end_time,
                                  use_fisheye, original_bitrate, keep_original_bitrate,
                                  log_callback=None, process_callback=None) -> bool:
    """native_gpu one_click single-eye streaming path. Return True on success; in auto mode return False on failure."""
    if not _native_stream_allowed(False):
        return False
    try:
        from gpu_engine import native_mosaic
        from gpu_engine.files import CancelToken
        from gpu_engine.fallback import OperationCancelled

        token = CancelToken()
        if process_callback:
            process_callback(token)
        side = "left" if eye_mode == 1 else "right"
        bitrate_bps = _resolve_single_eye_final_bitrate(
            input_file,
            original_bitrate,
            keep_original_bitrate,
            log_callback=log_callback,
        )
        if log_callback:
            stage = "fisheye/process" if use_fisheye else "process"
            log_callback(f"--- NativeGPU single-eye stream: {side}/{stage} without intermediate files ---")
        native_mosaic.restore_single_eye_stream(
            input_file, output_file,
            eye_mode=side,
            use_fisheye=use_fisheye,
            start_sec=_time_to_sec(start_time),
            end_sec=_time_to_sec(end_time),
            bitrate_bps=bitrate_bps,
            log_callback=log_callback,
            cancel_token=token,
        )
        return True
    except Exception as e:
        try:
            if isinstance(e, OperationCancelled):
                raise
        except UnboundLocalError:
            pass
        if _native_stream_failure_is_fatal():
            raise
        if log_callback:
            log_callback(f"[native-stream fallback] {type(e).__name__}: {e}; using legacy intermediate-file path")
        return False

def run_single_file_pipeline(input_file, start_time, end_time, use_fisheye, keep_intermediate=False, keep_original_bitrate=False, log_callback=None, process_callback=None, pre_extract=False, source_scan=True, fine_conf=None):
    pre_extract_enabled = False
    source_scan_enabled = False
    process_logger = _ProcessFileLogger(input_file, log_callback)
    log_callback = process_logger
    cleanup_success_artifacts = False
    cleanup_final_output = None
    preclip_path: str | None = None
    try:
        directory = os.path.dirname(input_file)
        filename = os.path.splitext(os.path.basename(input_file))[0]
        
        # Time suffix
        ss_part = start_time.replace(":", "") if start_time else "START"
        to_part = end_time.replace(":", "") if end_time else "END"
        suffix = f"_S{ss_part}_E{to_part}"
        
        file_l = os.path.join(directory, f"{filename}{suffix}_L.mp4")
        file_r = os.path.join(directory, f"{filename}{suffix}_R.mp4")
        file_l_restored = os.path.join(directory, f"{filename}{suffix}_L.restored.mp4")
        file_r_restored = os.path.join(directory, f"{filename}{suffix}_R.restored.mp4")
        file_final = os.path.join(directory, f"{filename}{suffix}_sbs.restored.mp4")
        
        # Fisheye intermediate files
        file_l_fish = os.path.join(directory, f"{filename}{suffix}_L_fisheye.mp4")
        file_r_fish = os.path.join(directory, f"{filename}{suffix}_R_fisheye.mp4")
        file_l_fish_restored = os.path.join(directory, f"{filename}{suffix}_L_fisheye.restored.mp4")
        file_r_fish_restored = os.path.join(directory, f"{filename}{suffix}_R_fisheye.restored.mp4")
        cleanup_final_output = file_final

        if os.path.exists(file_final):
            if log_callback: log_callback(f"Output file exists: {file_final}. Skipping.")
            cleanup_success_artifacts = True
            return

        original_bitrate = get_video_bitrate(input_file, log_callback)
        eye_intermediate_bitrate = _resolve_sbs_eye_intermediate_bitrate(
            input_file,
            original_bitrate,
            keep_original_bitrate,
            log_callback=log_callback,
        )
        pre_extract_enabled = _pre_extract_supported(pre_extract, log_callback)
        source_scan_enabled = _source_scan_supported(source_scan, log_callback)

        # When start/end is set, pre-cut once with -c copy so the whole
        # downstream pipeline (source-scan / native-stream / legacy fallback)
        # runs against a self-contained clipped file that already carries the
        # audio track. This bypasses the legacy start_time/end_time code path
        # in which audio was silently lost across split/lada/merge mux hops.
        input_file, preclip_path = _prepare_subrange_preclip(
            input_file, directory, start_time, end_time,
            log_callback=log_callback, process_callback=process_callback,
        )
        if preclip_path is not None:
            start_time = None
            end_time = None

        if source_scan_enabled:
            result = _run_source_scan_branch(
                input_file,
                file_final,
                use_fisheye=use_fisheye,
                pre_extract_inner=pre_extract,
                keep_intermediate=keep_intermediate,
                keep_original_bitrate=keep_original_bitrate,
                start_time=start_time,
                end_time=end_time,
                mode="sbs",
                log_callback=log_callback,
                process_callback=process_callback,
                fine_conf=fine_conf,
            )
            if result in {PreExtractResult.OK, PreExtractResult.NO_MOSAIC}:
                if log_callback and result == PreExtractResult.OK:
                    log_callback(f"Done! Output: {file_final}")
                _release_pre_extract_detector_if_needed(pre_extract_enabled or source_scan_enabled, log_callback)
                cleanup_success_artifacts = True
                return
            if result == PreExtractResult.CANCELLED:
                raise OperationCancelled("cancelled by user")
            if log_callback:
                log_callback("[source-scan] falling back to the normal OneClick path")

        if _native_stream_allowed(keep_intermediate) and _run_native_sbs_stream(
            input_file, file_final, start_time, end_time, use_fisheye,
            original_bitrate, keep_original_bitrate, log_callback, process_callback,
        ):
            _log_final_bitrate_summary(file_final, original_bitrate, log_callback)
            if log_callback: log_callback(f"Done! Output: {file_final}")
            cleanup_success_artifacts = True
            return
        
        if use_fisheye:
            # Optimized fisheye pipeline: 4 commands instead of 8
            # Step 1: Split + VR->Fisheye in one pass
            if log_callback: log_callback("--- Step 1/3: Splitting + VR->Fisheye ---")
            split_video_dual_fisheye(input_file, file_l_fish, file_r_fish, start_time, end_time, log_callback, process_callback)
            
            # Step 2: LADA processing (2 commands)
            if log_callback: log_callback("--- Step 2/3: Processing ---")
            _process_pre_extract_or_lada(file_l_fish, file_l_fish_restored, pre_extract_enabled, keep_intermediate, log_callback, process_callback, fine_conf=fine_conf, output_bitrate_bps=eye_intermediate_bitrate)
            _process_pre_extract_or_lada(file_r_fish, file_r_fish_restored, pre_extract_enabled, keep_intermediate, log_callback, process_callback, fine_conf=fine_conf, output_bitrate_bps=eye_intermediate_bitrate)
            
            # Step 3: Fisheye->VR + Merge in one pass
            if log_callback: log_callback("--- Step 3/3: Fisheye->VR + Merging ---")
            merge_videos_fisheye(file_l_fish_restored, file_r_fish_restored, file_final, original_bitrate, keep_original_bitrate, log_callback, process_callback)
            
            # Cleanup fisheye intermediate files
            if not keep_intermediate:
                if log_callback: log_callback("Cleaning up intermediate files...")
                cleanup_list = [file_l_fish, file_r_fish, file_l_fish_restored, file_r_fish_restored]
                for f in cleanup_list:
                    if os.path.exists(f): os.remove(f)
        else:
            # Non-fisheye pipeline: unchanged
            # Step 1: Split (dual output in one pass)
            if log_callback: log_callback("--- Step 1/3: Splitting ---")
            split_video_dual(input_file, file_l, file_r, start_time, end_time, log_callback, process_callback)
            
            # Step 2: Process
            if log_callback: log_callback("--- Step 2/3: Processing ---")
            _process_pre_extract_or_lada(file_l, file_l_restored, pre_extract_enabled, keep_intermediate, log_callback, process_callback, fine_conf=fine_conf, output_bitrate_bps=eye_intermediate_bitrate)
            _process_pre_extract_or_lada(file_r, file_r_restored, pre_extract_enabled, keep_intermediate, log_callback, process_callback, fine_conf=fine_conf, output_bitrate_bps=eye_intermediate_bitrate)
            
            # Step 3: Merge
            if log_callback: log_callback("--- Step 3/3: Merging ---")
            merge_videos(file_l_restored, file_r_restored, file_final, original_bitrate, keep_original_bitrate, log_callback, process_callback)
            
            # Cleanup
            if not keep_intermediate:
                if log_callback: log_callback("Cleaning up intermediate files...")
                cleanup_list = [file_l, file_r, file_l_restored, file_r_restored]
                for f in cleanup_list:
                    if os.path.exists(f): os.remove(f)
        
        _release_pre_extract_detector_if_needed(pre_extract_enabled or source_scan_enabled, log_callback)
        _log_final_bitrate_summary(file_final, original_bitrate, log_callback)
        if log_callback: log_callback(f"Done! Output: {file_final}")
        cleanup_success_artifacts = True
        
    except Exception as e:
        _release_pre_extract_detector_if_needed(pre_extract_enabled or source_scan_enabled, log_callback)
        if log_callback: log_callback(f"Error: {e}")
        raise e
    finally:
        if preclip_path is not None and not keep_intermediate:
            _remove_file_quiet(preclip_path, log_callback=log_callback)
        process_logger.close()
        if cleanup_success_artifacts and not keep_intermediate:
            _cleanup_run_artifacts(
                input_file,
                cleanup_final_output,
                process_logger.path,
                log_callback=process_logger.ui_callback,
            )

def run_single_eye_pipeline(input_file, eye_mode, start_time, end_time, use_fisheye, keep_intermediate=False, keep_original_bitrate=True, log_callback=None, process_callback=None, pre_extract=False, source_scan=True, fine_conf=None):
    # eye_mode: 1=Left, 2=Right
    pre_extract_enabled = False
    source_scan_enabled = False
    process_logger = _ProcessFileLogger(input_file, log_callback)
    log_callback = process_logger
    cleanup_success_artifacts = False
    cleanup_final_output = None
    preclip_path: str | None = None
    try:
        directory = os.path.dirname(input_file)
        filename = os.path.splitext(os.path.basename(input_file))[0]
        
        ss_part = start_time.replace(":", "") if start_time else "START"
        to_part = end_time.replace(":", "") if end_time else "END"
        suffix = f"_S{ss_part}_E{to_part}"
        
        side_suffix = "_L" if eye_mode == 1 else "_R"
        crop_filter = "crop=iw/2:ih:0:0" if eye_mode == 1 else "crop=iw/2:ih:iw/2:0"
        
        file_cut = os.path.join(directory, f"{filename}{suffix}{side_suffix}.mp4")
        file_final = os.path.join(directory, f"{filename}{suffix}{side_suffix}.restored.mp4")
        
        file_cut_fish = os.path.join(directory, f"{filename}{suffix}{side_suffix}_fisheye.mp4")
        file_cut_fish_restored = os.path.join(directory, f"{filename}{suffix}{side_suffix}_fisheye.restored.mp4")
        cleanup_final_output = file_final
        
        if os.path.exists(file_final):
            if log_callback: log_callback(f"Output file exists: {file_final}. Skipping.")
            cleanup_success_artifacts = True
            return

        original_bitrate = get_video_bitrate(input_file, log_callback)
        final_bitrate_bps = _resolve_single_eye_final_bitrate(
            input_file,
            original_bitrate,
            keep_original_bitrate,
            log_callback=log_callback,
        )
        final_bitrate_kbps = _bitrate_bps_to_kbps(final_bitrate_bps)
        split_bitrate_bps, split_max_bps = _single_eye_split_vbr_bps(original_bitrate)
        split_bitrate_kbps = _bitrate_bps_to_kbps(split_bitrate_bps)
        split_max_kbps = _bitrate_bps_to_kbps(split_max_bps)
        pre_extract_enabled = _pre_extract_supported(pre_extract, log_callback)
        source_scan_enabled = _source_scan_supported(source_scan, log_callback)

        # See run_single_file_pipeline: pre-cut once with -c copy so audio
        # survives all downstream mux hops.
        input_file, preclip_path = _prepare_subrange_preclip(
            input_file, directory, start_time, end_time,
            log_callback=log_callback, process_callback=process_callback,
        )
        if preclip_path is not None:
            start_time = None
            end_time = None

        if source_scan_enabled:
            result = _run_source_scan_branch(
                input_file,
                file_final,
                use_fisheye=use_fisheye,
                pre_extract_inner=pre_extract,
                keep_intermediate=keep_intermediate,
                keep_original_bitrate=keep_original_bitrate,
                start_time=start_time,
                end_time=end_time,
                mode="single_eye",
                eye_mode=eye_mode,
                log_callback=log_callback,
                process_callback=process_callback,
                fine_conf=fine_conf,
            )
            if result in {PreExtractResult.OK, PreExtractResult.NO_MOSAIC}:
                if log_callback and result == PreExtractResult.OK:
                    log_callback(f"Done! Output: {file_final}")
                _release_pre_extract_detector_if_needed(pre_extract_enabled or source_scan_enabled, log_callback)
                cleanup_success_artifacts = True
                return
            if result == PreExtractResult.CANCELLED:
                raise OperationCancelled("cancelled by user")
            if log_callback:
                log_callback("[source-scan] falling back to the normal OneClick path")

        if _native_stream_allowed(keep_intermediate) and _run_native_single_eye_stream(
            input_file, file_final, eye_mode, start_time, end_time, use_fisheye,
            original_bitrate, keep_original_bitrate, log_callback, process_callback,
        ):
            _log_final_bitrate_summary(file_final, final_bitrate_bps, log_callback)
            if log_callback: log_callback(f"Done! Output: {file_final}")
            cleanup_success_artifacts = True
            return

        if use_fisheye:
            # Step 1: crop + VR->Fisheye in one pass, avoiding one extra transcode.
            if log_callback: log_callback(f"--- Step 1/3: Splitting + VR->Fisheye ({side_suffix}) ---")
            if not os.path.exists(file_cut_fish):
                split_video_fisheye(
                    input_file, file_cut_fish, crop_filter, start_time, end_time,
                    log_callback, process_callback,
                    final_bitrate_kbps=split_bitrate_kbps,
                    max_bitrate_kbps=split_max_kbps,
                )
            else:
                if log_callback: log_callback(f"Intermediate file exists: {file_cut_fish}. Skipping.")
            # Step 2: Process (lada/jasna)
            if log_callback: log_callback("--- Step 2/3: Processing ---")
            _process_pre_extract_or_lada(file_cut_fish, file_cut_fish_restored, pre_extract_enabled, keep_intermediate, log_callback, process_callback, fine_conf=fine_conf)
            # Step 3: Fisheye->VR
            if log_callback: log_callback("--- Step 3/3: Fisheye->VR ---")
            convert_projection(file_cut_fish_restored, file_final, "fisheye:hequirect", final_bitrate_kbps=final_bitrate_kbps, log_callback=log_callback, process_callback=process_callback)
            cleanup_list = [file_cut_fish, file_cut_fish_restored]
        else:
            # Step 1: Split
            if log_callback: log_callback(f"--- Step 1/2: Splitting ({side_suffix}) ---")
            if not os.path.exists(file_cut):
                split_video(
                    input_file, file_cut, crop_filter, start_time, end_time,
                    log_callback, process_callback,
                    final_bitrate_kbps=split_bitrate_kbps,
                    max_bitrate_kbps=split_max_kbps,
                )
            else:
                if log_callback: log_callback(f"Intermediate file exists: {file_cut}. Skipping split.")
            # Step 2: Process (lada/jasna)
            if log_callback: log_callback("--- Step 2/2: Processing ---")
            _process_pre_extract_or_lada(
                file_cut,
                file_final,
                pre_extract_enabled,
                keep_intermediate,
                log_callback,
                process_callback,
                fine_conf=fine_conf,
                output_bitrate_bps=final_bitrate_bps,
            )
            cleanup_list = [file_cut]

        # Cleanup
        if not keep_intermediate:
            if log_callback: log_callback("Cleaning up intermediate files...")
            for f in cleanup_list:
                if os.path.exists(f): os.remove(f)

        _release_pre_extract_detector_if_needed(pre_extract_enabled or source_scan_enabled, log_callback)
        _log_final_bitrate_summary(file_final, final_bitrate_bps, log_callback)
        if log_callback: log_callback(f"Done! Output: {file_final}")
        cleanup_success_artifacts = True

    except Exception as e:
        _release_pre_extract_detector_if_needed(pre_extract_enabled or source_scan_enabled, log_callback)
        if log_callback: log_callback(f"Error: {e}")
        raise e
    finally:
        if preclip_path is not None and not keep_intermediate:
            _remove_file_quiet(preclip_path, log_callback=log_callback)
        process_logger.close()
        if cleanup_success_artifacts and not keep_intermediate:
            _cleanup_run_artifacts(
                input_file,
                cleanup_final_output,
                process_logger.path,
                log_callback=process_logger.ui_callback,
            )

def run_batch_pipeline(directory, use_fisheye, keep_original_bitrate=False, log_callback=None, process_callback=None, pre_extract=False, source_scan=True, fine_conf=None):
    ui_log_callback = log_callback
    mp4_files = glob.glob(os.path.join(directory, "*.mp4"))
    pre_extract_enabled = _pre_extract_supported(pre_extract, log_callback)
    source_scan_enabled = _source_scan_supported(source_scan, log_callback)
    for input_file in mp4_files:
        # Skip intermediate files
        if any(x in input_file for x in ["_L_", "_R_", "_L.", "_R.", ".restored", "_sbs", "fisheye"]):
            continue

        process_logger = _ProcessFileLogger(input_file, ui_log_callback)
        log_callback = process_logger
        cleanup_success_artifacts = False
        cleanup_final_output = None
        if log_callback: log_callback(f"\nProcessing: {os.path.basename(input_file)}")
        
        try:
            # Re-implement batch logic (Smart Resume)
            filename = os.path.splitext(os.path.basename(input_file))[0]
            file_l = os.path.join(directory, f"{filename}_L.mp4")
            file_r = os.path.join(directory, f"{filename}_R.mp4")
            
            file_l_fish = os.path.join(directory, f"{filename}_L_fisheye.mp4")
            file_r_fish = os.path.join(directory, f"{filename}_R_fisheye.mp4")
            file_l_fish_restored = os.path.join(directory, f"{filename}_L_fisheye.restored.mp4")
            file_r_fish_restored = os.path.join(directory, f"{filename}_R_fisheye.restored.mp4")

            file_l_restored = os.path.join(directory, f"{filename}_L.restored.mp4")
            file_r_restored = os.path.join(directory, f"{filename}_R.restored.mp4")
            file_final = os.path.join(directory, f"{filename}_sbs.restored.mp4")
            cleanup_final_output = file_final
            
            if os.path.exists(file_final):
                if log_callback: log_callback("Output exists. Skipping.")
                cleanup_success_artifacts = True
                continue
            
            original_bitrate = get_video_bitrate(input_file, log_callback)
            eye_intermediate_bitrate = _resolve_sbs_eye_intermediate_bitrate(
                input_file,
                original_bitrate,
                keep_original_bitrate,
                log_callback=log_callback,
            )

            if source_scan_enabled:
                result = _run_source_scan_branch(
                    input_file,
                    file_final,
                    use_fisheye=use_fisheye,
                    pre_extract_inner=pre_extract,
                    keep_intermediate=False,
                    keep_original_bitrate=keep_original_bitrate,
                    mode="sbs",
                    log_callback=log_callback,
                    process_callback=process_callback,
                    fine_conf=fine_conf,
                )
                if result in {PreExtractResult.OK, PreExtractResult.NO_MOSAIC}:
                    if log_callback and result == PreExtractResult.OK:
                        log_callback(f"Done! Output: {file_final}")
                    cleanup_success_artifacts = True
                    continue
                if result == PreExtractResult.CANCELLED:
                    raise OperationCancelled("cancelled by user")
                if log_callback:
                    log_callback("[source-scan] falling back to the normal OneClick path")

            if _native_stream_allowed(False) and _run_native_sbs_stream(
                input_file, file_final, None, None, use_fisheye,
                original_bitrate, keep_original_bitrate, log_callback, process_callback,
            ):
                _log_final_bitrate_summary(file_final, original_bitrate, log_callback)
                if log_callback: log_callback(f"Done! Output: {file_final}")
                cleanup_success_artifacts = True
                continue
            
            if use_fisheye:
                # Optimized fisheye pipeline: 4 commands instead of 8
                # Check existing fisheye restored files for smart resume
                skip_l = os.path.exists(file_l_fish_restored)
                skip_r = os.path.exists(file_r_fish_restored)
                
                # Step 1: Split + VR->Fisheye in one pass
                if not skip_l and not skip_r:
                    if not os.path.exists(file_l_fish) and not os.path.exists(file_r_fish):
                        split_video_dual_fisheye(input_file, file_l_fish, file_r_fish, None, None, log_callback, process_callback)
                
                # Step 2: LADA processing
                if not skip_l:
                    if not os.path.exists(file_l_fish):
                        # Fall back to individual split+convert if dual failed
                        split_video(input_file, file_l, "crop=iw/2:ih:0:0", None, None, log_callback, process_callback)
                        convert_projection(file_l, file_l_fish, "hequirect:fisheye", log_callback, process_callback)
                        if os.path.exists(file_l): os.remove(file_l)
                    _process_pre_extract_or_lada(file_l_fish, file_l_fish_restored, pre_extract_enabled, False, log_callback, process_callback, fine_conf=fine_conf, output_bitrate_bps=eye_intermediate_bitrate)
                
                if not skip_r:
                    if not os.path.exists(file_r_fish):
                        # Fall back to individual split+convert if dual failed
                        split_video(input_file, file_r, "crop=iw/2:ih:iw/2:0", None, None, log_callback, process_callback)
                        convert_projection(file_r, file_r_fish, "hequirect:fisheye", log_callback, process_callback)
                        if os.path.exists(file_r): os.remove(file_r)
                    _process_pre_extract_or_lada(file_r_fish, file_r_fish_restored, pre_extract_enabled, False, log_callback, process_callback, fine_conf=fine_conf, output_bitrate_bps=eye_intermediate_bitrate)
                
                # Step 3: Fisheye->VR + Merge in one pass
                merge_videos_fisheye(file_l_fish_restored, file_r_fish_restored, file_final, original_bitrate, keep_original_bitrate, log_callback, process_callback)
                
                # Cleanup
                cleanup_list = [file_l_fish, file_r_fish, file_l_fish_restored, file_r_fish_restored]
                for f in cleanup_list:
                    if os.path.exists(f): os.remove(f)
            else:
                # Non-fisheye pipeline
                # Check existing restored
                skip_l = os.path.exists(file_l_restored)
                skip_r = os.path.exists(file_r_restored)
                
                # Step 1: Split
                if not skip_l and not skip_r:
                    split_video_dual(input_file, file_l, file_r, None, None, log_callback, process_callback)
                else:
                    if not skip_l:
                        split_video(input_file, file_l, "crop=iw/2:ih:0:0", None, None, log_callback, process_callback)
                    if not skip_r:
                        split_video(input_file, file_r, "crop=iw/2:ih:iw/2:0", None, None, log_callback, process_callback)
                    
                # Step 2: Process
                if not skip_l:
                    _process_pre_extract_or_lada(file_l, file_l_restored, pre_extract_enabled, False, log_callback, process_callback, fine_conf=fine_conf, output_bitrate_bps=eye_intermediate_bitrate)
                if not skip_r:
                    _process_pre_extract_or_lada(file_r, file_r_restored, pre_extract_enabled, False, log_callback, process_callback, fine_conf=fine_conf, output_bitrate_bps=eye_intermediate_bitrate)
                    
                # Step 3: Merge
                merge_videos(file_l_restored, file_r_restored, file_final, original_bitrate, keep_original_bitrate, log_callback, process_callback)
                
                # Cleanup
                cleanup_list = [file_l, file_r, file_l_restored, file_r_restored]
                for f in cleanup_list:
                    if os.path.exists(f): os.remove(f)
            _log_final_bitrate_summary(file_final, original_bitrate, log_callback)
            cleanup_success_artifacts = True
                
        except OperationCancelled:
            if log_callback:
                log_callback("Cancelled by user")
            raise
        except Exception as e:
            if log_callback: log_callback(f"Error processing {os.path.basename(input_file)}: {e}")
        finally:
            process_logger.close()
            if cleanup_success_artifacts:
                _cleanup_run_artifacts(
                    input_file,
                    cleanup_final_output,
                    process_logger.path,
                    log_callback=ui_log_callback,
                )
            log_callback = ui_log_callback
    _release_pre_extract_detector_if_needed(pre_extract_enabled or source_scan_enabled, ui_log_callback)

def run_batch_eye_pipeline(directory, eye_mode, use_fisheye, keep_original_bitrate=True, log_callback=None, process_callback=None, pre_extract=False, source_scan=True, fine_conf=None):
    ui_log_callback = log_callback
    mp4_files = glob.glob(os.path.join(directory, "*.mp4"))
    side_suffix = "_L" if eye_mode == 1 else "_R"
    crop_filter = "crop=iw/2:ih:0:0" if eye_mode == 1 else "crop=iw/2:ih:iw/2:0"
    pre_extract_enabled = _pre_extract_supported(pre_extract, log_callback)
    source_scan_enabled = _source_scan_supported(source_scan, log_callback)
    
    for input_file in mp4_files:
        if any(x in input_file for x in ["_L_", "_R_", "_L.", "_R.", ".restored", "_sbs", "fisheye"]):
            continue

        process_logger = _ProcessFileLogger(input_file, ui_log_callback)
        log_callback = process_logger
        cleanup_success_artifacts = False
        cleanup_final_output = None
        if log_callback: log_callback(f"\nProcessing: {os.path.basename(input_file)}")
        
        try:
            filename = os.path.splitext(os.path.basename(input_file))[0]
            file_cut = os.path.join(directory, f"{filename}{side_suffix}.mp4")
            file_final = os.path.join(directory, f"{filename}{side_suffix}.restored.mp4")
            cleanup_final_output = file_final
            
            file_cut_fish = os.path.join(directory, f"{filename}{side_suffix}_fisheye.mp4")
            file_cut_fish_restored = os.path.join(directory, f"{filename}{side_suffix}_fisheye.restored.mp4")
            
            if os.path.exists(file_final):
                if log_callback: log_callback("Output exists. Skipping.")
                cleanup_success_artifacts = True
                continue
                
            original_bitrate = get_video_bitrate(input_file, log_callback)
            final_bitrate_bps = _resolve_single_eye_final_bitrate(
                input_file,
                original_bitrate,
                keep_original_bitrate,
                log_callback=log_callback,
            )
            final_bitrate_kbps = _bitrate_bps_to_kbps(final_bitrate_bps)
            split_bitrate_bps, split_max_bps = _single_eye_split_vbr_bps(original_bitrate)
            split_bitrate_kbps = _bitrate_bps_to_kbps(split_bitrate_bps)
            split_max_kbps = _bitrate_bps_to_kbps(split_max_bps)

            if source_scan_enabled:
                result = _run_source_scan_branch(
                    input_file,
                    file_final,
                    use_fisheye=use_fisheye,
                    pre_extract_inner=pre_extract,
                    keep_intermediate=False,
                    keep_original_bitrate=keep_original_bitrate,
                    mode="single_eye",
                    eye_mode=eye_mode,
                    log_callback=log_callback,
                    process_callback=process_callback,
                    fine_conf=fine_conf,
                )
                if result in {PreExtractResult.OK, PreExtractResult.NO_MOSAIC}:
                    if log_callback and result == PreExtractResult.OK:
                        log_callback(f"Done! Output: {file_final}")
                    cleanup_success_artifacts = True
                    continue
                if result == PreExtractResult.CANCELLED:
                    raise OperationCancelled("cancelled by user")
                if log_callback:
                    log_callback("[source-scan] falling back to the normal OneClick path")

            if _native_stream_allowed(False) and _run_native_single_eye_stream(
                input_file, file_final, eye_mode, None, None, use_fisheye,
                original_bitrate, keep_original_bitrate, log_callback, process_callback,
            ):
                _log_final_bitrate_summary(file_final, final_bitrate_bps, log_callback)
                if log_callback: log_callback(f"Done! Output: {file_final}")
                cleanup_success_artifacts = True
                continue

            if use_fisheye:
                # Crop + VR->Fisheye in one pass.
                if not os.path.exists(file_cut_fish):
                    split_video_fisheye(
                        input_file, file_cut_fish, crop_filter, None, None,
                        log_callback, process_callback,
                        final_bitrate_kbps=split_bitrate_kbps,
                        max_bitrate_kbps=split_max_kbps,
                    )
                _process_pre_extract_or_lada(file_cut_fish, file_cut_fish_restored, pre_extract_enabled, False, log_callback, process_callback, fine_conf=fine_conf)
                convert_projection(file_cut_fish_restored, file_final, "fisheye:hequirect", final_bitrate_kbps=final_bitrate_kbps, log_callback=log_callback, process_callback=process_callback)
                if os.path.exists(file_cut_fish): os.remove(file_cut_fish)
                if os.path.exists(file_cut_fish_restored): os.remove(file_cut_fish_restored)
            else:
                if not os.path.exists(file_cut):
                    split_video(
                        input_file, file_cut, crop_filter, None, None,
                        log_callback, process_callback,
                        final_bitrate_kbps=split_bitrate_kbps,
                        max_bitrate_kbps=split_max_kbps,
                    )
                _process_pre_extract_or_lada(
                    file_cut,
                    file_final,
                    pre_extract_enabled,
                    False,
                    log_callback,
                    process_callback,
                    fine_conf=fine_conf,
                    output_bitrate_bps=final_bitrate_bps,
                )
                if os.path.exists(file_cut): os.remove(file_cut)
            _log_final_bitrate_summary(file_final, final_bitrate_bps, log_callback)
            cleanup_success_artifacts = True

        except OperationCancelled:
            if log_callback:
                log_callback("Cancelled by user")
            raise
        except Exception as e:
            if log_callback: log_callback(f"Error processing {os.path.basename(input_file)}: {e}")
        finally:
            process_logger.close()
            if cleanup_success_artifacts:
                _cleanup_run_artifacts(
                    input_file,
                    cleanup_final_output,
                    process_logger.path,
                    log_callback=ui_log_callback,
                )
            log_callback = ui_log_callback
    _release_pre_extract_detector_if_needed(pre_extract_enabled or source_scan_enabled, ui_log_callback)

def run_merge_tool(left_file, right_file, keep_original_bitrate=True, log_callback=None, process_callback=None):
    try:
        directory = os.path.dirname(left_file)
        filename = os.path.splitext(os.path.basename(left_file))[0]
        # Try to replace _L or _l with _sbs
        final_name = filename.replace("_L", "_sbs").replace("_l", "_sbs")
        if final_name == filename: final_name += "_sbs"
        
        output_file = os.path.join(directory, final_name + ".mp4")
        
        if os.path.exists(output_file):
            if log_callback: log_callback(f"Output file exists: {output_file}. Skipping.")
            return
        
        target_original_bitrate = None
        if keep_original_bitrate:
            b_left = get_video_bitrate(left_file, log_callback)
            if b_left:
                target_original_bitrate = b_left * 2

        merge_videos(left_file, right_file, output_file, target_original_bitrate, keep_original_bitrate, log_callback, process_callback)
        _log_final_bitrate_summary(output_file, target_original_bitrate, log_callback)
        if log_callback: log_callback(f"Done! Output: {output_file}")
        
    except Exception as e:
        if log_callback: log_callback(f"Error: {e}")
        raise e
