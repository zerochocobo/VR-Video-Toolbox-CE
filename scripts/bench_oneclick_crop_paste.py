"""Benchmark the current OneClick pre-extract crop/paste primitives.

Default case:
  python scripts/bench_oneclick_crop_paste.py

This isolates the non-fisheye OneClick rect path:
  crop  : GPU NVDEC -> CuPy crop -> NVENC rect clip
  paste : GPU NVDEC base + GPU NVDEC rect clip -> CuPy patch -> NVENC output

The cropped rect clip is reused as the "restored" clip so the benchmark measures
crop/paste mechanics without LADA restoration cost.

By default the requested source time range is first stream-copied into a
temporary base clip, matching OneClick source-scan Stage 2 -> Stage 3 shape.
Use --base-mode direct to include original-file seek/keyframe-discovery cost.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_engine import files, probe, runtime  # noqa: E402
from gpu_engine.pynv_io import PyNvThreadedSerialDecoder  # noqa: E402
from utils import app_config, keyframe_cutter  # noqa: E402
from utils.mosaic_prescan import MosaicSegment  # noqa: E402
from utils.segment_paster import build_paste_segments  # noqa: E402


def parse_time(value: str) -> float:
    text = str(value).strip()
    if not text:
        raise argparse.ArgumentTypeError("empty time")
    if ":" not in text:
        return float(text)
    parts = [float(part) for part in text.split(":")]
    if len(parts) == 2:
        mm, ss = parts
        return mm * 60.0 + ss
    if len(parts) == 3:
        hh, mm, ss = parts
        return hh * 3600.0 + mm * 60.0 + ss
    raise argparse.ArgumentTypeError(f"unsupported time format: {value!r}")


def parse_rect(value: str) -> tuple[int, int, int, int]:
    try:
        parts = [int(part.strip()) for part in str(value).split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid rect: {value!r}") from exc
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("rect must be x,y,w,h")
    x, y, w, h = parts
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("rect width/height must be positive")
    return x, y, w, h


def _crop_region(mode: str | None, width: int, height: int) -> tuple[int, int, int, int]:
    mode = (mode or "").strip().lower()
    if not mode or mode == "none":
        return 0, 0, int(width), int(height)
    if mode == "left":
        return 0, 0, int(width // 2), int(height)
    if mode == "right":
        return int(width // 2), 0, int(width // 2), int(height)
    if mode == "top":
        return 0, 0, int(width), int(height // 2)
    if mode == "bottom":
        return 0, int(height // 2), int(width), int(height // 2)
    raise ValueError(f"unsupported crop_mode: {mode}")


def _clamp_even_rect(rect: tuple[int, int, int, int],
                     region_w: int,
                     region_h: int) -> tuple[int, int, int, int]:
    rx, ry, rw, rh = [int(v) for v in rect]
    rx -= rx % 2
    ry -= ry % 2
    rw -= rw % 2
    rh -= rh % 2
    rx = max(0, min(rx, max(0, int(region_w) - 2)))
    ry = max(0, min(ry, max(0, int(region_h) - 2)))
    rw = max(2, min(rw, int(region_w) - rx))
    rh = max(2, min(rh, int(region_h) - ry))
    rw -= rw % 2
    rh -= rh % 2
    return rx, ry, rw, rh


class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w", encoding="utf-8")

    def __call__(self, message: str) -> None:
        text = str(message)
        print(text, flush=True)
        self._f.write(text + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def _path_stats(path: Path) -> dict:
    meta = probe.probe_video(path)
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size) if path.exists() else 0,
        "codec": meta.codec_name,
        "pix_fmt": meta.pix_fmt,
        "bit_depth": int(meta.bit_depth),
        "width": int(meta.width),
        "height": int(meta.height),
        "fps": float(meta.source_fps or 0.0),
        "duration_s": float(meta.duration or 0.0),
        "nb_frames": int(meta.nb_frames or 0),
        "bitrate_bps": int(meta.bitrate_bps or 0),
    }


def _config_float(key: str, default: float) -> float:
    try:
        return float(app_config.get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)


def _pipeline_baseline_bitrate_bps(out_w: int | None, out_h: int | None, fps: float | None) -> int | None:
    try:
        px = int(out_w or 0) * int(out_h or 0)
        rate = float(fps or 30.0)
    except (TypeError, ValueError):
        return None
    if px <= 0:
        return None
    return max(1, int(px * max(1.0, rate) * 0.015))


def _oneclick_pipeline_bitrate(
    stage: str,
    out_w: int,
    out_h: int,
    fps: float,
    source_bps: int,
    *,
    keep_original: bool = False,
    source_w: int | None = None,
    source_h: int | None = None,
) -> int | None:
    stage_key = str(stage or "").strip().lower()
    if stage_key == "intermediate":
        multiplier = _config_float("gpu_bitrate_multiplier", 2.0)
    elif stage_key == "final":
        multiplier = 1.0 if keep_original else _config_float("gpu_bitrate_final_multiplier", 1.0)
    else:
        raise ValueError(f"unknown bitrate stage: {stage}")

    source = int(source_bps or 0)
    src_area = int(source_w or 0) * int(source_h or 0)
    out_area = int(out_w or 0) * int(out_h or 0)
    area_scale = (out_area / src_area) if source > 0 and src_area > 0 and out_area > 0 else 1.0
    target = int(source * area_scale * multiplier) if source > 0 else 0
    baseline = _pipeline_baseline_bitrate_bps(out_w, out_h, fps)
    skip_baseline = stage_key == "final" and keep_original and source > 0
    if baseline and not skip_baseline:
        target = max(target, baseline)
    return target if target > 0 else None


def _step(label: str, fn):
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    return result, elapsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark current GPU rect crop and paste used by OneClick pre-extract."
    )
    parser.add_argument("--src", default=str(ROOT / "videos" / "SI_TEST_2.mp4"))
    parser.add_argument("--start", type=parse_time, default=parse_time("00:01:00"))
    parser.add_argument("--end", type=parse_time, default=parse_time("00:05:00"))
    parser.add_argument(
        "--rect",
        type=parse_rect,
        default=parse_rect("1024,1024,1024,1024"),
        help="x,y,w,h in the selected crop-mode region; default is left-eye local.",
    )
    parser.add_argument(
        "--crop-mode",
        default="left",
        choices=["none", "left", "right", "top", "bottom"],
        help="Region used before rect crop. OneClick SBS rects are usually left/right eye local.",
    )
    parser.add_argument("--out-dir", default=str(ROOT / "debug_output" / "crop_paste_baseline"))
    parser.add_argument(
        "--base-mode",
        default="preclip-copy",
        choices=["preclip-copy", "direct"],
        help=(
            "preclip-copy first stream-copies the requested range like OneClick source-scan; "
            "direct crops/pastes from the original file with start/end frames."
        ),
    )
    parser.add_argument("--bitrate-bps", type=int, default=None)
    parser.add_argument(
        "--bitrate-mode",
        default="oneclick",
        choices=["oneclick", "auto"],
        help="oneclick applies OneClick pipeline bitrate policy; auto lets gpu_engine infer from base metadata.",
    )
    parser.add_argument(
        "--keep-original-bitrate",
        action="store_true",
        help="Apply OneClick final-stage keep-original bitrate behavior.",
    )
    parser.add_argument("--skip-crop", action="store_true", help="Reuse --restored instead of running crop.")
    parser.add_argument("--skip-paste", action="store_true")
    parser.add_argument("--restored", default="", help="Existing rect clip to paste when --skip-crop is used.")
    # --- paste-stage encoder overrides (experiment: faster preset + AQ to
    #     protect the patched region instead of dropping global quality) ---
    parser.add_argument(
        "--paste-preset",
        default="",
        help="Override NVENC preset for the paste stage only (e.g. P2). Empty = use gpu_encode_preset config.",
    )
    parser.add_argument(
        "--paste-aq",
        action="store_true",
        help="Enable spatial adaptive quantization (aq=1) on the paste encode; biases bits to high-detail (restored) regions.",
    )
    parser.add_argument(
        "--paste-temporal-aq",
        action="store_true",
        help="Enable temporal adaptive quantization (temporalaq=1) on the paste encode.",
    )
    parser.add_argument(
        "--paste-enc-extra",
        default="",
        help="Extra paste encoder kwargs as comma-separated key=value (e.g. 'multipass=fullres,lookahead=8').",
    )
    parser.add_argument(
        "--measure-quality",
        dest="measure_quality",
        action="store_true",
        default=True,
        help="Measure restored-region PSNR of the paste output vs the restored clip (default on).",
    )
    parser.add_argument(
        "--no-measure-quality",
        dest="measure_quality",
        action="store_false",
        help="Disable restored-region PSNR measurement.",
    )
    return parser


def _parse_paste_overrides(args) -> dict:
    """Build NVENC kwarg overrides applied only to the paste stage."""
    overrides: dict[str, str] = {}
    if args.paste_preset:
        preset = str(args.paste_preset).strip().upper()
        if preset not in {f"P{i}" for i in range(1, 8)}:
            raise ValueError(f"--paste-preset must be P1..P7, got {args.paste_preset!r}")
        overrides["preset"] = preset
    if args.paste_aq:
        overrides["aq"] = "1"
    if args.paste_temporal_aq:
        overrides["temporalaq"] = "1"
    for item in str(args.paste_enc_extra or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"--paste-enc-extra entry must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key.strip()] = value.strip()
    return overrides


def _measure_rect_psnr(
    reference_clip: Path,
    paste_clip: Path,
    *,
    rect_source: tuple[int, int, int, int],
    guard_px: int,
    max_frames: int,
    log,
) -> dict:
    """Mean Y/U/V PSNR over the restored rect interior (paste output vs restored clip).

    The reference clip frames *are* the rect (rw x rh); the paste output is the full
    frame with the rect pasted at (px, py). A ``guard_px`` border is excluded so the
    feathered alpha edge (identical across presets) does not dilute the signal that we
    care about: how much NVENC damaged the restored pixels.
    """
    import cupy as cp

    px, py, pw, ph = (int(v) for v in rect_source)
    m = max(0, int(guard_px))
    iw = pw - 2 * m
    ih = ph - 2 * m
    if iw < 2 or ih < 2:
        m = 0
        iw, ih = pw, ph
    # Even-align chroma sub-window.
    ix = m - (m % 2)
    iy = m - (m % 2)
    cx, cy, cw, ch = ix // 2, iy // 2, iw // 2, ih // 2

    ref_meta = probe.probe_video(reference_clip)
    test_meta = probe.probe_video(paste_clip)
    ref_bd = 10 if ref_meta.bit_depth > 8 else 8
    test_bd = 10 if test_meta.bit_depth > 8 else 8

    ref_dec = PyNvThreadedSerialDecoder(reference_clip, bit_depth=ref_bd)
    test_dec = PyNvThreadedSerialDecoder(paste_clip, bit_depth=test_bd)
    n = min(len(ref_dec), len(test_dec), int(max_frames))

    def _scale(bd: int) -> float:
        # P010 stores 10 valid bits in the high bits (value << 6); normalize to 0..1023.
        return 64.0 if bd > 8 else 1.0

    peak = 1023.0 if (ref_bd > 8 or test_bd > 8) else 255.0
    rs = _scale(ref_bd)
    ts = _scale(test_bd)

    se_y = se_u = se_v = 0.0
    cnt_y = cnt_c = 0
    try:
        for i in range(n):
            rf = ref_dec.frame_at(i)
            tf = test_dec.frame_at(i)
            cp.cuda.Device().synchronize()
            ry, ruv = rf.y_uv_cupy()
            ty, tuv = tf.y_uv_cupy()

            r_yi = ry[iy:iy + ih, ix:ix + iw].astype(cp.float32) / rs
            t_yi = ty[py + iy:py + iy + ih, px + ix:px + ix + iw].astype(cp.float32) / ts
            se_y += float(cp.sum((r_yi - t_yi) ** 2))
            cnt_y += r_yi.size

            r_ci = ruv[cy:cy + ch, cx:cx + cw, :].astype(cp.float32) / rs
            t_ci = tuv[(py // 2) + cy:(py // 2) + cy + ch, (px // 2) + cx:(px // 2) + cx + cw, :].astype(cp.float32) / ts
            se_u += float(cp.sum((r_ci[..., 0] - t_ci[..., 0]) ** 2))
            se_v += float(cp.sum((r_ci[..., 1] - t_ci[..., 1]) ** 2))
            cnt_c += r_ci[..., 0].size
    finally:
        ref_dec.stop()
        test_dec.stop()

    def _psnr(se: float, cnt: int) -> float:
        if cnt <= 0:
            return 0.0
        mse = se / cnt
        if mse <= 1e-9:
            return 99.0
        import math
        return 10.0 * math.log10((peak * peak) / mse)

    psnr_y = _psnr(se_y, cnt_y)
    psnr_u = _psnr(se_u, cnt_c)
    psnr_v = _psnr(se_v, cnt_c)
    result = {
        "frames": int(n),
        "interior": {"x": ix, "y": iy, "w": iw, "h": ih, "guard_px": int(m)},
        "peak": peak,
        "psnr_y_db": round(psnr_y, 3),
        "psnr_u_db": round(psnr_u, 3),
        "psnr_v_db": round(psnr_v, 3),
        "psnr_yuv_db": round((6 * psnr_y + psnr_u + psnr_v) / 8.0, 3),
    }
    if log:
        log(
            f"[bench] rect quality (interior {iw}x{ih}, {n} frames): "
            f"Y={psnr_y:.2f}dB U={psnr_u:.2f}dB V={psnr_v:.2f}dB "
            f"YUV={result['psnr_yuv_db']:.2f}dB"
        )
    return result


def _measure_frame_psnr(
    base_clip: Path,
    paste_clip: Path,
    *,
    rect_source: tuple[int, int, int, int],
    start_frame: int,
    max_frames: int,
    log,
) -> dict:
    """Full-frame and background-only Y-PSNR of paste output vs the base input.

    This quantifies re-encode generation loss on the unchanged 8K background, and
    whether AQ steals bits from the flat background to feed the patch. "background"
    excludes the pasted rect so it is unaffected by the restoration content itself.
    """
    import cupy as cp

    px, py, pw, ph = (int(v) for v in rect_source)

    base_meta = probe.probe_video(base_clip)
    test_meta = probe.probe_video(paste_clip)
    base_bd = 10 if base_meta.bit_depth > 8 else 8
    test_bd = 10 if test_meta.bit_depth > 8 else 8

    base_dec = PyNvThreadedSerialDecoder(base_clip, bit_depth=base_bd, start_frame=int(start_frame))
    test_dec = PyNvThreadedSerialDecoder(paste_clip, bit_depth=test_bd)
    n = min(len(base_dec) - int(start_frame), len(test_dec), int(max_frames))

    bs = 64.0 if base_bd > 8 else 1.0
    ts = 64.0 if test_bd > 8 else 1.0
    peak = 1023.0 if (base_bd > 8 or test_bd > 8) else 255.0

    se_full = 0.0
    se_rect = 0.0
    cnt_full = 0
    cnt_rect = 0
    try:
        for i in range(n):
            bf = base_dec.frame_at(int(start_frame) + i)
            tf = test_dec.frame_at(i)
            cp.cuda.Device().synchronize()
            by, _ = bf.y_uv_cupy()
            ty, _ = tf.y_uv_cupy()
            b = by.astype(cp.float32) / bs
            t = ty.astype(cp.float32) / ts
            d2 = (b - t) ** 2
            se_full += float(cp.sum(d2))
            cnt_full += b.size
            rd2 = d2[py:py + ph, px:px + pw]
            se_rect += float(cp.sum(rd2))
            cnt_rect += rd2.size
    finally:
        base_dec.stop()
        test_dec.stop()

    import math

    def _psnr(se: float, cnt: int) -> float:
        if cnt <= 0:
            return 0.0
        mse = se / cnt
        if mse <= 1e-9:
            return 99.0
        return 10.0 * math.log10((peak * peak) / mse)

    se_bg = max(0.0, se_full - se_rect)
    cnt_bg = max(0, cnt_full - cnt_rect)
    psnr_full = _psnr(se_full, cnt_full)
    psnr_bg = _psnr(se_bg, cnt_bg)
    result = {
        "frames": int(n),
        "peak": peak,
        "psnr_full_y_db": round(psnr_full, 3),
        "psnr_background_y_db": round(psnr_bg, 3),
    }
    if log:
        log(
            f"[bench] frame quality ({n} frames): full Y={psnr_full:.2f}dB "
            f"background(excl rect) Y={psnr_bg:.2f}dB"
        )
    return result


def main() -> int:
    args = build_parser().parse_args()
    src = Path(args.src)
    if not src.exists():
        print(f"source not found: {src}", file=sys.stderr)
        return 2
    if args.end <= args.start:
        print("--end must be greater than --start", file=sys.stderr)
        return 2
    if args.skip_crop and not args.restored:
        print("--skip-crop requires --restored", file=sys.stderr)
        return 2

    state = runtime.warmup(verbose=True)
    if not state.available:
        print(f"GPU unavailable: {state.reason}", file=sys.stderr)
        return 2

    source_meta = probe.probe_video(src)
    region_x, region_y, region_w, region_h = _crop_region(
        None if args.crop_mode == "none" else args.crop_mode,
        source_meta.width,
        source_meta.height,
    )
    rect = _clamp_even_rect(args.rect, region_w, region_h)
    rx, ry, rw, rh = rect
    paste_x = int(region_x + rx)
    paste_y = int(region_y + ry)
    fps = float(source_meta.source_fps or 30.0)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLogger(run_dir / "run.log")

    crop_mode = None if args.crop_mode == "none" else args.crop_mode
    base_clip = run_dir / "base_preclip.mp4"
    crop_out = run_dir / "rect_crop.mp4"
    paste_out = run_dir / "rect_paste.mp4"
    restored_path = Path(args.restored) if args.restored else crop_out

    if args.base_mode == "preclip-copy":
        base_src = base_clip
        bench_start_s = 0.0
        requested_duration_s = float(args.end) - float(args.start)
        bench_end_s = requested_duration_s
    else:
        base_src = src
        bench_start_s = float(args.start)
        bench_end_s = float(args.end)

    start_frame = int(round(float(bench_start_s) * fps))
    end_frame = int(round(float(bench_end_s) * fps))
    frame_count = max(0, end_frame - start_frame)

    metrics = {
        "source": _path_stats(src),
        "gpu": {
            "name": state.name,
            "compute_capability": state.compute_capability,
            "summary": state.summary,
        },
        "range": {
            "source_start_s": float(args.start),
            "source_end_s": float(args.end),
            "base_mode": args.base_mode,
            "base_start_s": bench_start_s,
            "base_end_s": bench_end_s,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frame_count": frame_count,
        },
        "crop": {
            "crop_mode": args.crop_mode,
            "region": {"x": region_x, "y": region_y, "w": region_w, "h": region_h},
            "rect_local": {"x": rx, "y": ry, "w": rw, "h": rh},
        },
        "paste": {
            "rect_source": {"x": paste_x, "y": paste_y, "w": rw, "h": rh},
        },
    }

    try:
        log(f"[bench] source={src}")
        log(
            f"[bench] source meta: {source_meta.width}x{source_meta.height} "
            f"{source_meta.bit_depth}bit {source_meta.codec_name}/{source_meta.pix_fmt} "
            f"{fps:.3f}fps duration={source_meta.duration:.3f}s"
        )
        log(f"[bench] base_mode={args.base_mode}")
        if args.base_mode == "preclip-copy":
            log(
                f"[bench] preclip copy source range={args.start:.3f}-{args.end:.3f}s "
                f"-> {base_clip}"
            )
            _, elapsed = _step(
                "preclip_copy",
                lambda: keyframe_cutter._cut_copy(src, base_clip, float(args.start), float(args.end), log_callback=log),
            )
            base_meta = probe.probe_video(base_clip)
            bench_end_s = min(float(base_meta.duration or requested_duration_s), requested_duration_s)
            fps = float(base_meta.source_fps or fps or 30.0)
            start_frame = 0
            end_frame = int(round(float(bench_end_s) * fps))
            frame_count = max(0, end_frame - start_frame)
            metrics["preclip_copy"] = {
                "elapsed_s": elapsed,
                "output": _path_stats(base_clip),
            }
            metrics["range"].update({
                "base_start_s": bench_start_s,
                "base_end_s": bench_end_s,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "frame_count": frame_count,
            })
            log(f"[bench] preclip elapsed={elapsed:.3f}s duration={base_meta.duration:.3f}s")
        else:
            log(
                "[bench] direct mode includes original-file nonzero-start decoder "
                "setup/keyframe-discovery cost."
            )
            metrics["range"].update({
                "base_start_s": bench_start_s,
                "base_end_s": bench_end_s,
            })
        metrics["base"] = _path_stats(base_src)
        if args.bitrate_bps is not None:
            crop_bitrate_bps = int(args.bitrate_bps)
            paste_bitrate_bps = int(args.bitrate_bps)
            bitrate_mode = "explicit"
        elif args.bitrate_mode == "oneclick":
            crop_bitrate_bps = _oneclick_pipeline_bitrate(
                "intermediate",
                rw,
                rh,
                fps,
                int(source_meta.bitrate_bps or 0),
                keep_original=args.keep_original_bitrate,
                source_w=int(source_meta.width),
                source_h=int(source_meta.height),
            )
            paste_bitrate_bps = _oneclick_pipeline_bitrate(
                "final",
                int(metrics["base"]["width"]),
                int(metrics["base"]["height"]),
                fps,
                int(source_meta.bitrate_bps or 0),
                keep_original=args.keep_original_bitrate,
            )
            bitrate_mode = "oneclick"
        else:
            crop_bitrate_bps = None
            paste_bitrate_bps = None
            bitrate_mode = "gpu_engine_auto"
        metrics["bitrate"] = {
            "mode": bitrate_mode,
            "crop_bitrate_bps": crop_bitrate_bps,
            "paste_bitrate_bps": paste_bitrate_bps,
            "source_bitrate_bps": int(source_meta.bitrate_bps or 0),
            "keep_original_bitrate": bool(args.keep_original_bitrate),
        }
        log(
            f"[bench] bitrate mode={bitrate_mode}, "
            f"crop={crop_bitrate_bps or 'auto'}bps, paste={paste_bitrate_bps or 'auto'}bps"
        )
        log(
            f"[bench] measured base range={bench_start_s:.3f}-{bench_end_s:.3f}s "
            f"frames={start_frame}-{end_frame} ({frame_count}) base={base_src}"
        )
        log(
            f"[bench] crop_mode={args.crop_mode}, rect_local={rx},{ry},{rw}x{rh}, "
            f"paste_rect_source={paste_x},{paste_y},{rw}x{rh}"
        )
        log(f"[bench] run_dir={run_dir}")

        if not args.skip_crop:
            _, elapsed = _step(
                "crop",
                lambda: files.extract_transformed_rect_clip(
                    base_src,
                    crop_out,
                    crop_mode=crop_mode or "",
                    rect=rect,
                    to_fisheye=False,
                    start_sec=float(bench_start_s),
                    end_sec=float(bench_end_s),
                    keep_audio=False,
                    bitrate_bps=crop_bitrate_bps,
                    log_callback=log,
                ),
            )
            metrics["crop"]["elapsed_s"] = elapsed
            metrics["crop"]["fps"] = frame_count / elapsed if elapsed > 0 else 0.0
            metrics["crop"]["output"] = _path_stats(crop_out)
            log(f"[bench] crop elapsed={elapsed:.3f}s throughput={metrics['crop']['fps']:.2f}fps")
        else:
            if not restored_path.exists():
                raise FileNotFoundError(restored_path)
            metrics["crop"]["skipped"] = True
            metrics["crop"]["output"] = _path_stats(restored_path)
            log(f"[bench] crop skipped, restored={restored_path}")

        if not args.skip_paste:
            seg = MosaicSegment(
                seg_id=0,
                start_s=float(bench_start_s),
                end_s=float(bench_end_s),
                start_s_kf=float(bench_start_s),
                end_s_kf=float(bench_end_s),
                x=paste_x,
                y=paste_y,
                w=rw,
                h=rh,
                conf_max=1.0,
            )
            paste_segments = build_paste_segments(base_src, [seg], [restored_path])

            paste_overrides = _parse_paste_overrides(args)
            metrics["paste"]["encoder_overrides"] = dict(paste_overrides)
            if paste_overrides:
                log(f"[bench] paste encoder overrides: {paste_overrides}")
            orig_enc_kwargs = files._encoder_kwargs

            def _patched_enc_kwargs(meta, bitrate_bps, **kw):
                base = orig_enc_kwargs(meta, bitrate_bps, **kw)
                base.update(paste_overrides)
                return base

            files._encoder_kwargs = _patched_enc_kwargs
            try:
                _, elapsed = _step(
                    "paste",
                    lambda: files.paste_segments_gpu(
                        base_src,
                        paste_out,
                        paste_segments,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        keep_audio=False,
                        bitrate_bps=paste_bitrate_bps,
                        log_callback=log,
                    ),
                )
            finally:
                files._encoder_kwargs = orig_enc_kwargs
            metrics["paste"]["elapsed_s"] = elapsed
            metrics["paste"]["fps"] = frame_count / elapsed if elapsed > 0 else 0.0
            metrics["paste"]["output"] = _path_stats(paste_out)
            log(f"[bench] paste elapsed={elapsed:.3f}s throughput={metrics['paste']['fps']:.2f}fps")

            if args.measure_quality:
                feather_px = int(files._cfg("pre_extract_feather_px", 12) or 12)
                guard = max(feather_px + 2, 4)
                try:
                    metrics["paste"]["quality"] = _measure_rect_psnr(
                        restored_path,
                        paste_out,
                        rect_source=(paste_x, paste_y, rw, rh),
                        guard_px=guard,
                        max_frames=frame_count,
                        log=log,
                    )
                except Exception as exc:  # measurement must never fail the bench
                    log(f"[bench] rect quality measurement failed: {exc!r}")
                    metrics["paste"]["quality"] = {"error": repr(exc)}
                try:
                    metrics["paste"]["frame_quality"] = _measure_frame_psnr(
                        base_src,
                        paste_out,
                        rect_source=(paste_x, paste_y, rw, rh),
                        start_frame=start_frame,
                        max_frames=frame_count,
                        log=log,
                    )
                except Exception as exc:
                    log(f"[bench] frame quality measurement failed: {exc!r}")
                    metrics["paste"]["frame_quality"] = {"error": repr(exc)}
        else:
            metrics["paste"]["skipped"] = True
            log("[bench] paste skipped")

        total_elapsed = float(metrics.get("crop", {}).get("elapsed_s", 0.0)) + float(
            metrics.get("paste", {}).get("elapsed_s", 0.0)
        )
        metrics["total_elapsed_s"] = total_elapsed
        metrics["total_fps_if_sequential"] = frame_count / total_elapsed if total_elapsed > 0 else 0.0
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log(
            f"[bench] total measured stages={total_elapsed:.3f}s "
            f"sequential throughput={metrics['total_fps_if_sequential']:.2f}fps"
        )
        log(f"[bench] metrics={run_dir / 'metrics.json'}")
        return 0
    finally:
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
