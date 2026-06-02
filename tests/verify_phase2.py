"""Phase 2 verification: tool_v360_trans.convert_projection end-to-end, routing, and fallback.

Usage: uv run python tests/verify_phase2.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gpu_engine  # noqa: E402
from gpu_engine import probe, runtime  # noqa: E402
from utils import app_config  # noqa: E402
from tool_v360_trans import logic  # noqa: E402

FIX = ROOT / "tests" / "fixtures"
OUT = Path(tempfile.gettempdir()) / "phase2_out"
OUT.mkdir(exist_ok=True)
logs: list[str] = []


def log(m):
    logs.append(str(m))


def set_backend(mode):
    app_config.set("transcode_backend", mode)


def clean():
    for f in OUT.glob("*"):
        try:
            f.unlink()
        except OSError:
            pass
    logs.clear()


def check(name, cond):
    print(f"  [{'OK ' if cond else 'XX '}] {name}")
    return cond


def main() -> int:
    if not runtime.warmup().available:
        print("GPU unavailable"); return 2
    ok = True

    # 1) backend=auto, 10bit SBS dual, both directions
    print("=== 1) auto / 10bit SBS dual ===")
    set_backend("auto")
    for mode, expect_suffix in [("hequirect2fisheye", "_fisheye"),
                                ("fisheye2hequirect", "_hequirect")]:
        clean()
        logic.convert_projection(str(FIX / "sbs_hevc_10bit_bt709.mp4"), str(OUT), mode,
                                 dual_screen=True, keep_original_bitrate=False,
                                 log_callback=log)
        outs = list(OUT.glob("*.mp4"))
        good = len(outs) == 1
        if good:
            m = probe.probe_video(outs[0])
            good = m.width == 7680 // 2 * 2 if False else True  # dims preserved
            m0 = probe.probe_video(FIX / "sbs_hevc_10bit_bt709.mp4")
            good = (m.width == m0.width and m.height == m0.height and m.bit_depth == 10)
        ok &= check(f"{mode}: 1 output, 10bit, dims preserved", good)
        # Confirm the GPU path was used: logs include gpu encoded and no fallback.
        used_gpu = any("[gpu]" in l for l in logs) and not any("fallback" in l for l in logs)
        ok &= check(f"{mode}: used GPU path", used_gpu)

    # 2) backend=auto, 8bit single
    print("=== 2) auto / 8bit single ===")
    clean()
    set_backend("auto")
    logic.convert_projection(str(FIX / "sbs_hevc_8bit_bt709.mp4"), str(OUT),
                             "hequirect2fisheye", dual_screen=False, log_callback=log)
    outs = list(OUT.glob("*.mp4"))
    good = len(outs) == 1 and probe.probe_video(outs[0]).bit_depth == 8
    ok &= check("8bit single output, 8bit preserved", good)

    # 3) backend=auto, HDR10 -> automatically use ffmpeg
    print("=== 3) auto / HDR10 -> ffmpeg fallback ===")
    clean()
    set_backend("auto")
    logic.convert_projection(str(FIX / "sbs_hevc_10bit_hdr10.mp4"), str(OUT),
                             "hequirect2fisheye", dual_screen=True, log_callback=log)
    outs = list(OUT.glob("*.mp4"))
    routed_ffmpeg = any("ffmpeg" in l.lower() for l in logs)
    ok &= check("HDR10 output produced", len(outs) == 1)
    ok &= check("HDR10 routed to ffmpeg", routed_ffmpeg)

    # 4) backend=gpu, HDR10 -> no silent fallback; expect no output or an error log.
    print("=== 4) gpu(forced) / HDR10 -> error, no silent fallback ===")
    clean()
    set_backend("gpu")
    logic.convert_projection(str(FIX / "sbs_hevc_10bit_hdr10.mp4"), str(OUT),
                             "hequirect2fisheye", dual_screen=True, log_callback=log)
    outs = list(OUT.glob("*.mp4"))
    ok &= check("forced GPU on HDR10 produced no output", len(outs) == 0)

    # 5) backend=ffmpeg, 10bit -> force ffmpeg
    print("=== 5) ffmpeg(forced) / 10bit ===")
    clean()
    set_backend("ffmpeg")
    logic.convert_projection(str(FIX / "sbs_hevc_10bit_bt709.mp4"), str(OUT),
                             "hequirect2fisheye", dual_screen=True, log_callback=log)
    outs = list(OUT.glob("*.mp4"))
    forced = any("backend=ffmpeg" in l for l in logs)
    ok &= check("forced ffmpeg produced output", len(outs) == 1)
    ok &= check("log shows forced ffmpeg", forced)

    set_backend("auto")  # Restore default.
    print(f"\n  {'ALL PASS' if ok else 'SOME FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
