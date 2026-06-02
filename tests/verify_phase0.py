"""Phase 0 verification: routing plus decode->encode->mux round trip.

No geometry transform is applied, so the result should be near-lossless.
Usage: uv run python tests/verify_phase0.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gpu_engine  # noqa: E402  Trigger CUDA environment configuration.
from gpu_engine import probe, runtime, mux  # noqa: E402
from gpu_engine.pynv_io import (  # noqa: E402
    PyNvSimpleDecoder, GpuNv12AppFrame, GpuP016AppFrame, PyNvEncoderSession,
)

FIX = ROOT / "tests" / "fixtures"

EXPECT = {
    "sbs_h264_8bit_bt709.mp4": "gpu_nv12",
    "sbs_hevc_8bit_bt709.mp4": "gpu_nv12",
    "sbs_hevc_10bit_bt709.mp4": "gpu_p016",
    "sbs_hevc_10bit_hdr10.mp4": "ffmpeg_fallback",
}


def test_routing() -> bool:
    print("\n=== 路由决策 ===")
    ok = True
    for name, expected in EXPECT.items():
        meta, dec = probe.route(FIX / name)
        mark = "OK " if dec.backend == expected else "XX "
        if dec.backend != expected:
            ok = False
        print(f"  [{mark}] {name}: {dec.backend} (expect {expected}) "
              f"| {meta.codec_name} {meta.pix_fmt} {meta.bit_depth}bit "
              f"prim={meta.color.color_primaries} trc={meta.color.color_transfer} | {dec.reason}")
    return ok


def _pack_appframe(frame, bit_depth):
    """Copy a decoded frame into a contiguous packed buffer and wrap it as AppFrame."""
    import cupy as cp

    y, uv = frame.y_uv_cupy()
    h, w = frame.height, frame.width
    if bit_depth > 8:
        packed = cp.empty((h * 3 // 2, w), dtype=cp.uint16)
    else:
        packed = cp.empty((h * 3 // 2, w), dtype=cp.uint8)
    packed[:h, :] = y
    packed[h:, :] = uv.reshape(h // 2, w)
    if bit_depth > 8:
        return GpuP016AppFrame(packed, w, h)
    return GpuNv12AppFrame(packed, w, h)


def test_roundtrip(name: str, bit_depth: int) -> bool:
    print(f"\n=== 环回 {name} ({bit_depth}bit) ===")
    src = FIX / name
    meta = probe.probe_video(src)
    dec = PyNvSimpleDecoder(src, bit_depth=bit_depth)
    info = dec.info
    print(f"  src {info.width}x{info.height} fps={info.fps:.3f} frames={len(dec)} codec={info.codec_name}")

    enc = PyNvEncoderSession(
        info.width, info.height, bit_depth=bit_depth,
        codec="hevc", fps=f"{meta.source_fps:.6f}", gop="30", bf="0",
        bitrate="20000000", tuning_info="high_quality",
    )
    raw = Path(tempfile.gettempdir()) / f"phase0_{name}.hevc"
    out = Path(tempfile.gettempdir()) / f"phase0_{name}_rt.mp4"
    n = 0
    with open(raw, "wb") as f:
        for i in range(len(dec)):
            frame = dec.frame_at(i)
            app = _pack_appframe(frame, bit_depth)
            data = enc.encode(app, force_idr=(i == 0))
            if data:
                f.write(data)
            n += 1
        tail = enc.flush()
        if tail:
            f.write(tail)
    dec.stop()
    runtime.free_memory_pool()
    print(f"  encoded {n} frames -> {raw.stat().st_size} bytes raw hevc")

    mux.mux_hevc_with_audio(
        raw, out, fps=meta.source_fps, color=meta.color, audio_source=None,
    )
    out_meta = probe.probe_video(out)
    print(f"  muxed -> {out} | {out_meta.codec_name} {out_meta.pix_fmt} {out_meta.bit_depth}bit "
          f"{out_meta.width}x{out_meta.height}")
    ok = out_meta.width == info.width and out_meta.height == info.height
    ok = ok and out_meta.bit_depth == bit_depth
    print(f"  result: {'OK' if ok else 'XX dimension/bitdepth mismatch'}")
    return ok


def main() -> int:
    st = runtime.warmup(verbose=True)
    if not st.available:
        print("GPU unavailable; cannot run phase0 GPU verification:", st.reason)
        return 2

    results = []
    results.append(("routing", test_routing()))
    results.append(("roundtrip_8bit", test_roundtrip("sbs_hevc_8bit_bt709.mp4", 8)))
    results.append(("roundtrip_10bit", test_roundtrip("sbs_hevc_10bit_bt709.mp4", 10)))

    print("\n=== 汇总 ===")
    all_ok = True
    for name, ok in results:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
        all_ok = all_ok and ok
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
