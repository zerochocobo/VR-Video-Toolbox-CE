"""LUT geometry regression test: GPU remap vs ffmpeg v360 at raw-pixel level.

Key points:
  - The GPU side uses ThreadedDecoder starting from the first displayed frame
    (PTS 0), aligned with ffmpeg display-frame order. SimpleDecoder emits two
    extra encoder-delay lead frames (PTS<0), causing a two-frame index offset.
  - Compare raw pixels before encoding to avoid GOP/PTS differences between
    different encoders (pynv vs nvenc).
  - Cover full-frame and SBS-dual, multiple frames, 8/10-bit, and both directions.

Usage: uv run python tests/verify_lut_geometry.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gpu_engine  # noqa: E402
from gpu_engine import v360_lut, nv12_kernels  # noqa: E402
from gpu_engine.pynv_io import PyNvThreadedSerialDecoder  # noqa: E402

FIX = ROOT / "tests" / "fixtures"
NFRAMES = 5
# Thresholds: the plan requires 8bit Y/UV>=42/38 and 10bit Y/UV>=45/40.
GATE = {8: (42.0, 38.0), 10: (45.0, 40.0)}


def psnr(a, b, peak):
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    return 99.0 if mse == 0 else 10.0 * np.log10(peak * peak / mse)


def ffmpeg_raw(src: Path, mode: str, dual: bool, pix_fmt: str, n: int) -> bytes:
    proj = "hequirect:fisheye" if mode == "heq2fisheye" else "fisheye:hequirect"
    if dual:
        v = "hequirect:fisheye" if mode == "heq2fisheye" else "fisheye:hequirect"
        fc = (f"[0:v]split=2[l][r];"
              f"[l]crop=iw/2:ih:0:0,v360={v}[a];"
              f"[r]crop=iw/2:ih:iw/2:0,v360={v}[b];"
              f"[a][b]hstack=inputs=2[v]")
        args = ["-filter_complex", fc, "-map", "[v]"]
    else:
        args = ["-vf", f"v360={proj}"]
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(src),
           *args, "-frames:v", str(n), "-pix_fmt", pix_fmt, "-f", "rawvideo", "-"]
    return subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout


def gpu_remap(y, uv, mode, dual):
    import cupy as cp
    h, w = y.shape
    ch, cw = uv.shape[0], uv.shape[1]
    if not dual:
        ly = v360_lut.make_lut(mode, w, h); lc = v360_lut.make_lut(mode, cw, ch)
        oy = nv12_kernels.remap_y(y, ly, w, h)
        ouv = nv12_kernels.remap_uv(uv, lc, cw, ch)
        return cp.asnumpy(oy), cp.asnumpy(ouv)
    hw = w // 2; chw = cw // 2
    ly = v360_lut.make_lut(mode, hw, h); lc = v360_lut.make_lut(mode, chw, ch)
    oly = nv12_kernels.remap_y(y[:, :hw], ly, hw, h)
    ory = nv12_kernels.remap_y(y[:, hw:], ly, hw, h)
    oy = cp.concatenate([oly, ory], axis=1)
    oluv = nv12_kernels.remap_uv(uv[:, :chw, :], lc, chw, ch)
    oruv = nv12_kernels.remap_uv(uv[:, chw:, :], lc, chw, ch)
    ouv = cp.concatenate([oluv, oruv], axis=1)
    return cp.asnumpy(oy), cp.asnumpy(ouv)


def run_case(name, mode, dual, bit_depth) -> tuple[float, float, bool]:
    import cupy as cp
    src = FIX / name
    pix_fmt = "p010le" if bit_depth > 8 else "nv12"
    peak = 65535.0 if bit_depth > 8 else 255.0
    dt = "<u2" if bit_depth > 8 else "u1"

    td = PyNvThreadedSerialDecoder(src, bit_depth=bit_depth)
    w, h = td.info.width, td.info.height
    gpu_frames = []
    for i in range(NFRAMES):
        f = td.frame_at(i); cp.cuda.Device().synchronize()
        y, uv = f.y_uv_cupy()
        gpu_frames.append(gpu_remap(y, uv, mode, dual))
    td.stop()

    raw = np.frombuffer(ffmpeg_raw(src, mode, dual, pix_fmt, NFRAMES), dtype=dt)
    per = w * h * 3 // 2
    ymin = puvmin = 99.0
    for i in range(NFRAMES):
        base = i * per
        fy = raw[base:base + w * h].reshape(h, w)
        fuv = raw[base + w * h:base + per].reshape(h // 2, w // 2, 2)
        gy, guv = gpu_frames[i]
        ymin = min(ymin, psnr(gy, fy, peak))
        puvmin = min(puvmin, psnr(guv, fuv, peak))
    gy_gate, guv_gate = GATE[bit_depth]
    ok = ymin >= gy_gate and puvmin >= guv_gate
    tag = "dual" if dual else "full"
    print(f"  [{'OK ' if ok else 'XX '}] {name} {mode} {tag} {bit_depth}bit: "
          f"Y={ymin:.1f} UV={puvmin:.1f} (gate {gy_gate}/{guv_gate})")
    return ymin, puvmin, ok


def main() -> int:
    from gpu_engine import runtime
    if not runtime.warmup().available:
        print("GPU unavailable"); return 2
    print("=== LUT 几何回归 (GPU raw vs ffmpeg v360 raw, ThreadedDecoder 对齐) ===")
    cases = [
        ("sbs_hevc_8bit_bt709.mp4", "heq2fisheye", False, 8),
        ("sbs_hevc_8bit_bt709.mp4", "fisheye2heq", False, 8),
        ("sbs_hevc_8bit_bt709.mp4", "heq2fisheye", True, 8),
        ("sbs_hevc_10bit_bt709.mp4", "heq2fisheye", False, 10),
        ("sbs_hevc_10bit_bt709.mp4", "fisheye2heq", False, 10),
        ("sbs_hevc_10bit_bt709.mp4", "heq2fisheye", True, 10),
    ]
    results = [run_case(*c) for c in cases]
    allok = all(r[2] for r in results)
    print(f"\n  {'ALL PASS' if allok else 'SOME FAILED'}  (min Y={min(r[0] for r in results):.1f}dB)")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
