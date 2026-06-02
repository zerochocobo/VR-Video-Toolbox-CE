"""Phase 1 benchmark: GPU v360 pipeline vs the current ffmpeg path.

Measures end-to-end wall-clock time and output PSNR. Uses a 4K SBS clip by
default, generating one if it does not exist.
Usage: uv run python tests/bench_gpu_vs_ffmpeg.py [--res 3840x1920] [--dur 5]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gpu_engine  # noqa: E402
from gpu_engine import files, probe, runtime  # noqa: E402

TMP = Path(tempfile.gettempdir())


def ensure_bench_fixture(res: str, dur: int, bit10: bool) -> Path:
    w, h = res.split("x")
    tag = "10bit" if bit10 else "8bit"
    out = TMP / f"bench_sbs_{res}_{tag}.mp4"
    if out.exists():
        return out
    print(f"[bench] generating {out.name} ...")
    pix = "p010le" if bit10 else "yuv420p"
    prof = ["-profile:v", "main10"] if bit10 else []
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate=30:duration={dur}",
        "-c:v", "libx265", "-pix_fmt", pix, *prof,
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-color_range", "tv", "-tag:v", "hvc1", str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


def ffmpeg_v360(src: Path, dst: Path, mode: str, dual: bool, bit10: bool) -> float:
    """Reproduce the tool_v360_trans ffmpeg path and return elapsed seconds."""
    proj = "hequirect:fisheye" if mode == "heq2fisheye" else "fisheye:hequirect"
    if dual:
        if mode == "heq2fisheye":
            fc = ("[0:v]split=2[l][r];"
                  "[l]crop=iw/2:ih:0:0,v360=hequirect:fisheye[left];"
                  "[r]crop=iw/2:ih:iw/2:0,v360=hequirect:fisheye[right];"
                  "[left][right]hstack=inputs=2[v]")
        else:
            fc = ("[0:v]split=2[l][r];"
                  "[l]crop=iw/2:ih:0:0,v360=fisheye:hequirect[left];"
                  "[r]crop=iw/2:ih:iw/2:0,v360=fisheye:hequirect[right];"
                  "[left][right]hstack=inputs=2[v]")
        vargs = ["-filter_complex", fc, "-map", "[v]", "-map", "0:a?"]
    else:
        vargs = ["-vf", f"v360={proj}"]
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-hwaccel", "cuda", "-c:v", "hevc_cuvid", "-i", str(src),
        *vargs, "-c:a", "copy",
        "-c:v", "hevc_nvenc", "-preset", "p7", "-cq", "18",
        str(dst),
    ]
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    return time.perf_counter() - t0


def gpu_v360(src: Path, dst: Path, mode: str, dual: bool) -> float:
    t0 = time.perf_counter()
    files.vr_projection(src, dst, mode, dual_screen=dual, cq=18, keep_audio=False)
    return time.perf_counter() - t0


# Geometry and visual-quality correctness are covered by tests/verify_lut_geometry.py
# with raw pixels and ThreadedDecoder alignment, verified at roughly 79 dB.
# This script measures only end-to-end speed; directly computing PSNR on encoded
# output from different encoders (pynv vs nvenc) is unreliable because different
# GOP/PTS/B-frame structures can misalign paired frames.


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="3840x1920")
    ap.add_argument("--dur", type=int, default=5)
    ap.add_argument("--bit10", action="store_true", default=True)
    ap.add_argument("--bit8", dest="bit10", action="store_false")
    args = ap.parse_args()

    if not runtime.warmup(verbose=True).available:
        print("GPU unavailable"); return 2

    src = ensure_bench_fixture(args.res, args.dur, args.bit10)
    meta = probe.probe_video(src)
    print(f"\n[bench] src {meta.width}x{meta.height} {meta.bit_depth}bit "
          f"{meta.codec_name} {meta.source_fps:.2f}fps dur~{args.dur}s\n")

    cases = [
        ("heq2fisheye", True, "SBS hequirect->fisheye"),
        ("fisheye2heq", True, "SBS fisheye->hequirect"),
    ]
    print(f"{'case':<26}{'ffmpeg(s)':>11}{'gpu(s)':>9}{'speedup':>9}")
    print("-" * 56)
    for mode, dual, label in cases:
        ff_out = TMP / f"bench_ff_{mode}.mp4"
        gp_out = TMP / f"bench_gpu_{mode}.mp4"
        t_ff = ffmpeg_v360(src, ff_out, mode, dual, args.bit10)
        t_gp = gpu_v360(src, gp_out, mode, dual)
        speedup = t_ff / t_gp if t_gp > 0 else 0
        print(f"{label:<26}{t_ff:>11.2f}{t_gp:>9.2f}{speedup:>8.2f}x")
    print("\n几何正确性见 tests/verify_lut_geometry.py（裸像素 ~79dB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
