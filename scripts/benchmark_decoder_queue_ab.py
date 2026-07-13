"""Isolated A/B benchmark for PyNv decoder queue depth.

The parent process prepares a short SBS sample, then launches every benchmark
case in a fresh Python process while nvidia-smi samples device memory. Production
code is not modified: the worker temporarily wraps PyNvThreadedSerialDecoder and
forces buffer_size=32 or 8 only for the selected combine/extract call.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _force_decoder_buffer(buffer_size: int) -> None:
    from gpu_engine import files

    original = files.PyNvThreadedSerialDecoder
    size = max(2, int(buffer_size))

    def create(*args, **kwargs):
        kwargs["buffer_size"] = size
        kwargs["batch_size"] = min(8, size)
        return original(*args, **kwargs)

    files.PyNvThreadedSerialDecoder = create


def _worker_prepare(args) -> None:
    from one_click import logic

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    left = out_dir / "combine_left.mp4"
    right = out_dir / "combine_right.mp4"
    for path in (left, right):
        path.unlink(missing_ok=True)
    logic.split_video_dual(
        str(Path(args.input).resolve()),
        str(left),
        str(right),
        "00:00:00",
        f"00:00:{float(args.duration):06.3f}",
        log_callback=print,
        keep_audio=False,
    )


def _worker_combine(args) -> None:
    from gpu_engine import files

    _force_decoder_buffer(args.buffer)
    out_dir = Path(args.output_dir)
    output = out_dir / f"combine_buffer{args.buffer}.mp4"
    output.unlink(missing_ok=True)
    files.combine_video(
        out_dir / "combine_left.mp4",
        out_dir / "combine_right.mp4",
        output,
        "left_right",
        keep_audio=False,
        log_callback=print,
    )


def _worker_extract(args) -> None:
    from gpu_engine import files

    _force_decoder_buffer(args.buffer)
    out_dir = Path(args.output_dir)
    left = out_dir / f"extract_left_buffer{args.buffer}.mp4"
    right = out_dir / f"extract_right_buffer{args.buffer}.mp4"
    for path in (left, right):
        path.unlink(missing_ok=True)
    files.extract_multi_rect_clip(
        Path(args.input).resolve(),
        [
            {
                "dst": left,
                "crop_mode": "left",
                "rect": (1536, 1536, 1024, 1024),
                "bitrate_bps": 2_000_000,
                "label": "left",
            },
            {
                "dst": right,
                "crop_mode": "right",
                "rect": (1536, 1536, 1024, 1024),
                "bitrate_bps": 2_000_000,
                "label": "right",
            },
        ],
        start_sec=0.0,
        end_sec=float(args.duration),
        keep_audio=False,
        log_callback=print,
    )


def _run_worker(args) -> None:
    if args.worker == "prepare":
        _worker_prepare(args)
    elif args.worker == "combine":
        _worker_combine(args)
    elif args.worker == "extract":
        _worker_extract(args)
    else:
        raise ValueError(f"unknown worker: {args.worker}")


def _hidden_kwargs() -> dict:
    if os.name != "nt":
        return {}
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}


def _parse_trace(path: Path) -> tuple[int, int, int]:
    values: list[int] = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            raw = row.get(" memory.used [MiB]", row.get("memory.used [MiB]", ""))
            match = re.search(r"(\d+)", raw or "")
            if match:
                values.append(int(match.group(1)))
    if not values:
        raise RuntimeError(f"no memory samples in {path}")
    baseline_samples = values[: min(6, len(values))]
    baseline = int(round(statistics.median(baseline_samples)))
    peak = max(values)
    return baseline, peak, peak - baseline


def _probe_output(path: Path) -> dict:
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    result = subprocess.run(
        [
            ffprobe,
            "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=width,height,pix_fmt,nb_read_frames:format=duration",
            "-of", "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
        **_hidden_kwargs(),
    )
    data = json.loads(result.stdout)
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}
    return {
        "path": str(path),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "pix_fmt": str(stream.get("pix_fmt") or ""),
        "frames": int(stream.get("nb_read_frames") or 0),
        "duration_s": float(fmt.get("duration") or 0.0),
        "size_bytes": path.stat().st_size,
    }


def _case_outputs(kind: str, buffer_size: int, out_dir: Path) -> list[Path]:
    if kind == "combine":
        return [out_dir / f"combine_buffer{buffer_size}.mp4"]
    return [
        out_dir / f"extract_left_buffer{buffer_size}.mp4",
        out_dir / f"extract_right_buffer{buffer_size}.mp4",
    ]


def _run_case(args, kind: str, buffer_size: int) -> dict:
    out_dir = Path(args.output_dir).resolve()
    trace = out_dir / f"{kind}_vram_buffer{buffer_size}.csv"
    log = out_dir / f"{kind}_buffer{buffer_size}.log"
    for path in (trace, log):
        path.unlink(missing_ok=True)

    nvidia_smi = shutil.which("nvidia-smi") or "nvidia-smi"
    with trace.open("w", encoding="utf-8", newline="") as trace_handle:
        sampler = subprocess.Popen(
            [
                nvidia_smi,
                "--query-gpu=timestamp,memory.used,memory.total",
                "--format=csv",
                "-lms", "500",
            ],
            stdout=trace_handle,
            stderr=subprocess.STDOUT,
            text=True,
            **_hidden_kwargs(),
        )
        try:
            time.sleep(3.0)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker", kind,
                "--input", str(Path(args.input).resolve()),
                "--output-dir", str(out_dir),
                "--duration", str(args.duration),
                "--buffer", str(buffer_size),
            ]
            started = time.perf_counter()
            with log.open("w", encoding="utf-8-sig", newline="") as log_handle:
                completed = subprocess.run(
                    cmd,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=str(ROOT),
                    **_hidden_kwargs(),
                )
            elapsed = time.perf_counter() - started
            if completed.returncode != 0:
                raise RuntimeError(f"{kind} buffer={buffer_size} failed; see {log}")
            time.sleep(1.0)
        finally:
            sampler.terminate()
            try:
                sampler.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                sampler.kill()
                sampler.wait(timeout=5.0)

    baseline, peak, delta = _parse_trace(trace)
    outputs = [_probe_output(path) for path in _case_outputs(kind, buffer_size, out_dir)]
    return {
        "kind": kind,
        "buffer_size": buffer_size,
        "elapsed_s": elapsed,
        "baseline_mib": baseline,
        "peak_mib": peak,
        "peak_above_baseline_mib": delta,
        "trace": str(trace),
        "log": str(log),
        "outputs": outputs,
    }


def _parent(args) -> None:
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prepare_log = out_dir / "prepare.log"
    prepare_cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker", "prepare",
        "--input", str(Path(args.input).resolve()),
        "--output-dir", str(out_dir),
        "--duration", str(args.duration),
    ]
    with prepare_log.open("w", encoding="utf-8-sig", newline="") as handle:
        subprocess.run(
            prepare_cmd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(ROOT),
            check=True,
            **_hidden_kwargs(),
        )

    results = []
    for kind in ("combine", "extract"):
        for buffer_size in (32, 8):
            print(f"Running {kind} buffer={buffer_size}...")
            result = _run_case(args, kind, buffer_size)
            results.append(result)
            print(
                f"  elapsed={result['elapsed_s']:.2f}s "
                f"peak_delta={result['peak_above_baseline_mib']}MiB"
            )

    summary = {
        "input": str(Path(args.input).resolve()),
        "duration_s": float(args.duration),
        "results": results,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    print(f"Summary: {summary_path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "videos" / "2_2.mp4"))
    parser.add_argument("--output-dir", default=str(ROOT / "debug_output" / "decoder_queue_ab_2_2"))
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--buffer", type=int, default=32)
    parser.add_argument("--worker", choices=("prepare", "combine", "extract"))
    return parser


if __name__ == "__main__":
    parsed = _parser().parse_args()
    if parsed.worker:
        _run_worker(parsed)
    else:
        _parent(parsed)
