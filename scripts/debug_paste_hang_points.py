# -*- coding: utf-8 -*-
"""Locate local GPU/CuPy/PyNv hang points in the paste path.

This is a diagnostic script, not a benchmark and not production code.

Why this exists:
  During local uv-environment testing, some CuPy JIT-backed operations appeared
  to hang even though the real OneClick paste path does not hang on the user's
  machine. We should not change the production paste implementation based on
  that local symptom. Instead, use this script to mark exactly which low-level
  operation blocks so another engineer can inspect the environment/driver/CuPy
  interaction.

How to read output:
  Every risky operation prints a line before it starts and another after it
  returns/synchronizes. If the process hangs, the last printed
  "[HANG-CHECK ...] BEGIN ..." line identifies the operation that did not
  return, and the last "BEFORE CUDA SYNC" line means the kernel launch returned
  but synchronization did not complete.

中文说明：
  这个脚本只用于定位“测试环境在哪里卡住”，不要把这里的结果直接当成
  OneClick 实机生产流程的优化依据。实机 paste 当前没有卡死现象，所以这里
  的目标是把本地 uv/CuPy/PyNv 的卡点标出来，交给熟悉驱动、CUDA、CuPy
  JIT 或 PyNvVideoCodec 的同事继续分析。

  日志中的每一行都有序号和脚本启动后的累计秒数。进程卡住时，不需要猜：
  直接看最后一条已经 flush 出来的 [HANG-CHECK ...]。

  - 最后一行是 "BEGIN operation"：
    Python 调用本身没有返回，可能卡在 import、对象构造、PyNv 解码调用、
    CuPy kernel 编译、CuPy 数组创建等同步/半同步入口。
  - 最后一行是 "RETURNED ..." 但没有 "AFTER CUDA SYNC"：
    Python 调用已经返回，但 CUDA 队列中的工作没有完成，通常要看 kernel
    launch、stream 同步、驱动、设备状态或上游异步错误。
  - 某一步耗时很长但能继续：
    这不是“卡死”，而是热点。需要和 paste 主流程 FPS profile 分开分析。

Typical commands:
  uv run python scripts/debug_paste_hang_points.py --case env
  uv run python scripts/debug_paste_hang_points.py --case cupy-arange
  uv run python scripts/debug_paste_hang_points.py --case rawkernel

  uv run python scripts/debug_paste_hang_points.py ^
    --case paste-alpha ^
    --base debug_output\\paste_research_2_1_current\\20260626_154419\\base_preclip.mp4 ^
    --restored debug_output\\paste_research_2_1_prep\\20260626_145558\\rect_crop.mp4 ^
    --rect 1680,1696,1360,1552

Notes:
  - By default the script imports gpu_engine before importing CuPy, so it uses
    the same CUDA/CuPy environment setup as production. Use "--raw-env" only
    when you intentionally want to reproduce unconfigured CuPy defaults.
  - "paste-alpha" intentionally uses the original production CuPy alpha formula.
    By default it runs in "split" mode so every sub-expression has its own
    checkpoint. Use "--alpha-eval prod-expression" to reproduce the original
    one-line expression more closely.
  - "paste-direct-copy" tests strided CuPy view assignment for both Y and UV
    planes in the 8K packed encoder buffer.
  - "alpha-source=prod" uses gpu_engine.files._make_alpha_mask(), which itself
    uses CuPy arange/minimum. If that hangs, rerun with "--alpha-source cpu" to
    bypass mask generation and isolate the paste formula.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SCRIPT_T0 = time.perf_counter()
_SEQ = 0


CASE_GUIDE: dict[str, str] = {
    "env": (
        "只做环境信息采集。用于确认 uv 解释器、CuPy、CUDA runtime/driver、"
        "Torch CUDA 看到的是不是同一套设备。"
    ),
    "cupy-copy": (
        "只测试 numpy -> CuPy 的 host-to-device 上传。这个路径一般不触发 "
        "CuPy elementwise/JIT，适合作为 GPU 拷贝是否正常的对照组。"
    ),
    "cupy-arange": (
        "测试 cp.arange。生产 alpha mask 会走类似 CuPy elementwise 路径；"
        "如果这里卡，优先怀疑本地 CuPy JIT/kernel 生成或驱动交互。"
    ),
    "rawkernel": (
        "测试最小 RawKernel 编译和 launch。如果 arange 卡、RawKernel 也卡，"
        "说明问题可能比 paste 代码更底层。"
    ),
    "pynv-read": (
        "只打开 PyNv decoder 并读取第一帧，再把 Y/UV view 和 packed buffer "
        "准备出来。用于区分 PyNv 解码/plane view/packed copy 是否卡住。"
    ),
    "paste-direct-copy": (
        "不做 alpha 公式，只把 restored Y/UV 直接写进 packed base 的 rect "
        "view。用于定位 CuPy strided view assignment 是否异常。"
    ),
    "paste-alpha": (
        "完整准备 base/restored/alpha mask，然后执行 alpha paste。默认 split "
        "模式会把 astype、乘法、加法、rint、cast、写回逐个同步标记。"
    ),
    "encode-first-frame": (
        "不做 paste，只把第一帧 packed buffer 交给 NVENC。用于分离 encoder "
        "创建和首帧 encode 是否卡住。"
    ),
}


def _now() -> str:
    return time.strftime("%H:%M:%S")


def mark(label: str, message: str) -> None:
    """Print a flushed checkpoint line.

    Keep this tiny and dependency-free: when debugging hangs, stdout buffering
    can hide the last useful line unless every checkpoint flushes immediately.

    这里故意不用 logging 模块：logging 的 handler/format/buffering 会增加排查
    变量。卡死排查时，最重要的是“最后一行一定已经写到控制台”。
    """
    global _SEQ
    _SEQ += 1
    elapsed = time.perf_counter() - _SCRIPT_T0
    print(
        f"[{_now()}] [HANG-CHECK #{_SEQ:04d} +{elapsed:9.3f}s pid={os.getpid()} {label}] "
        f"{message}",
        flush=True,
    )


def timed(label: str, fn: Callable, *, sync: Callable | None = None):
    """Run one operation with explicit before/after/sync markers.

    CuPy/PyNv operations are often asynchronous. A Python call returning quickly
    does not prove the GPU work has finished, so each suspicious operation can
    provide a sync callback. If the last line is BEFORE CUDA SYNC, the enqueue
    returned and the wait is what blocked.
    """
    mark(label, "BEGIN operation")
    t0 = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:
        mark(label, f"RAISED {type(exc).__name__}: {exc}")
        raise
    mark(label, f"RETURNED in {time.perf_counter() - t0:.6f}s")
    if sync is not None:
        mark(label, "BEFORE CUDA SYNC")
        t1 = time.perf_counter()
        try:
            sync()
        except Exception as exc:
            mark(label, f"CUDA SYNC RAISED {type(exc).__name__}: {exc}")
            raise
        mark(label, f"AFTER CUDA SYNC in {time.perf_counter() - t1:.6f}s")
    return result


def explain_case(name: str) -> None:
    """Print the diagnostic intent before a case starts.

    这个说明是给拿到日志的人看的。因为一旦进程卡住，后续说明不会再打印，
    所以必须在真正进入风险调用前先输出。
    """
    guide = CASE_GUIDE.get(name)
    if guide:
        mark("GUIDE", f"{name}: {guide}")
    mark(
        "GUIDE",
        "判读规则：最后一条 BEGIN 表示调用未返回；最后一条 BEFORE CUDA SYNC "
        "表示调用返回了但 GPU/driver 同步没有完成。",
    )


def _path_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _describe_path(path: Path, *, write_probe: bool) -> str:
    """Return existence/writeability diagnostics for a path.

    The script may be run inside restricted workspaces. To avoid touching user
    profile directories during normal diagnostics, write probes are only done
    for repo-local paths unless the caller explicitly asks for a probe.
    """
    exists = path.exists()
    is_dir = path.is_dir()
    if not write_probe:
        return f"{path} exists={exists} is_dir={is_dir} write_probe=skipped"
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return f"{path} exists={path.exists()} is_dir={path.is_dir()} write_probe=ok"
    except Exception as exc:
        return (
            f"{path} exists={path.exists()} is_dir={path.is_dir()} "
            f"write_probe={type(exc).__name__}: {exc}"
        )


def configure_runtime_env(args) -> None:
    """Mirror production CUDA/CuPy environment setup before any CuPy import."""
    if args.raw_env:
        mark("CUDA-ENV", "raw env requested; skipping gpu_engine._cuda_env.configure()")
        return
    mark("CUDA-ENV", "BEGIN import gpu_engine to configure CUDA/CuPy environment")
    t0 = time.perf_counter()
    import gpu_engine  # noqa: F401  # package import runs _cuda_env.configure()
    mark("CUDA-ENV", f"RETURNED in {time.perf_counter() - t0:.6f}s")


def parse_rect(value: str) -> tuple[int, int, int, int]:
    parts = [int(part.strip()) for part in str(value).split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("rect must be x,y,w,h")
    x, y, w, h = parts
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("rect width/height must be positive")
    return x, y, w, h


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print checkpoint markers around paste-path GPU operations."
    )
    parser.add_argument(
        "--case",
        default="env",
        choices=[
            "env",
            "cupy-copy",
            "cupy-arange",
            "rawkernel",
            "pynv-read",
            "paste-direct-copy",
            "paste-alpha",
            "encode-first-frame",
            "all",
        ],
        help=(
            "Diagnostic case to run. Use individual cases because a hang stops "
            "the process before later cases can print their markers."
        ),
    )
    parser.add_argument("--base", default="", help="Base 8K clip for PyNv/paste cases.")
    parser.add_argument("--restored", default="", help="Restored rect clip for paste cases.")
    parser.add_argument("--rect", type=parse_rect, default=parse_rect("1680,1696,1360,1552"))
    parser.add_argument("--bit-depth", type=int, default=10, choices=[8, 10])
    parser.add_argument("--arange-size", type=int, default=1360)
    parser.add_argument(
        "--alpha-source",
        default="prod",
        choices=["prod", "cpu"],
        help=(
            "prod uses gpu_engine.files._make_alpha_mask(); cpu builds the same "
            "mask with numpy and uploads it with cp.asarray to isolate paste."
        ),
    )
    parser.add_argument(
        "--alpha-eval",
        default="split",
        choices=["split", "prod-expression"],
        help=(
            "paste-alpha evaluation mode. split adds checkpoints after every "
            "sub-expression; prod-expression uses the original one-line formula."
        ),
    )
    parser.add_argument(
        "--show-cupy-config",
        action="store_true",
        help="Print cupy.show_config(); useful but verbose.",
    )
    parser.add_argument(
        "--raw-env",
        action="store_true",
        help="Do not import gpu_engine before CuPy; useful only to reproduce unconfigured defaults.",
    )
    parser.add_argument(
        "--probe-external-cache-write",
        action="store_true",
        help="Also write-probe cache dirs outside the repo. Use only when debugging permissions.",
    )
    return parser


def case_env(args) -> None:
    mark("ENV", f"python={sys.executable}")
    mark("ENV", f"cwd={Path.cwd()}")
    for key in (
        "CUDA_PATH",
        "CUDA_HOME",
        "CUDA_PATH_ORIGINAL_VRTB",
        "CUDA_HOME_ORIGINAL_VRTB",
        "CUPY_COMPILE_WITH_PTX",
        "CUPY_CACHE_DIR",
        "CUDA_CACHE_PATH",
        "CUDA_CACHE_DISABLE",
        "CUDA_CACHE_MAXSIZE",
        "LOCALAPPDATA",
        "APPDATA",
        "TEMP",
        "TMP",
    ):
        mark("ENV", f"{key}={os.environ.get(key)!r}")

    runtime_cache = ROOT / "runtime_cache"
    cupy_cache = Path(os.environ.get("CUPY_CACHE_DIR") or (Path.home() / ".cupy" / "kernel_cache"))
    cuda_cache = Path(os.environ.get("CUDA_CACHE_PATH") or (Path(os.environ.get("LOCALAPPDATA", "")) / "NVIDIA" / "ComputeCache"))
    user_cupy_root = Path.home() / ".cupy"
    for label, path in (
        ("RUNTIME-CACHE", runtime_cache),
        ("ACTIVE-CUPY-CACHE", cupy_cache),
        ("ACTIVE-CUDA-CACHE", cuda_cache),
        ("DEFAULT-USER-CUPY-ROOT", user_cupy_root),
    ):
        allow_write = _path_under(path, ROOT) or bool(args.probe_external_cache_write)
        mark("CACHE", f"{label}: {_describe_path(path, write_probe=allow_write)}")

    try:
        import cupy as cp

        mark("ENV", f"cupy={cp.__version__}")
        mark("ENV", f"runtime={cp.cuda.runtime.runtimeGetVersion()}")
        mark("ENV", f"driver={cp.cuda.runtime.driverGetVersion()}")
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        mark(
            "ENV",
            "device="
            f"{props.get('name', b'').decode(errors='ignore')} "
            f"cc={props.get('major')}.{props.get('minor')}",
        )
        if args.show_cupy_config:
            mark("ENV", "BEGIN cupy.show_config()")
            cp.show_config()
            mark("ENV", "END cupy.show_config()")
    except Exception as exc:
        mark("ENV", f"cupy import/config failed: {type(exc).__name__}: {exc}")

    try:
        import torch

        mark("ENV", f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            mark("ENV", f"torch_device={torch.cuda.get_device_name(0)}")
    except Exception as exc:
        mark("ENV", f"torch import/config failed: {type(exc).__name__}: {exc}")


def case_cupy_copy(args) -> None:
    import numpy as np
    import cupy as cp

    # Control case: this uses host-to-device copy and usually does not require
    # CuPy elementwise/JIT kernel generation.
    # 中文：这是对照组。如果这里也卡住，问题大概率不是 paste 公式，而是
    # 基础 GPU 内存上传、CUDA context 初始化或驱动状态。
    timed(
        "CUPY-COPY",
        lambda: cp.asarray(np.ones((1552, 1360), dtype=np.float32)),
        sync=lambda: cp.cuda.Device().synchronize(),
    )


def case_cupy_arange(args) -> None:
    import cupy as cp

    # Suspect case in the local uv environment: if this hangs before RETURNED,
    # CuPy's arange/elementwise path is blocked before we even reach paste.
    # 中文：生产 _make_alpha_mask() 会用到 cp.arange/minimum 这类 CuPy
    # elementwise 路径。这里单独跑，是为了把“mask 生成卡住”和“paste 卡住”
    # 分开。
    timed(
        "CUPY-ARANGE",
        lambda: cp.arange(int(args.arange_size), dtype=cp.float32),
        sync=lambda: cp.cuda.Device().synchronize(),
    )


def case_rawkernel(args) -> None:
    import numpy as np
    import cupy as cp

    # 最小 CUDA kernel。它不依赖视频输入，也不依赖生产 paste 代码。
    # 如果这个 case 都卡在 launch/sync，同事应优先查 CUDA driver、CuPy cache、
    # 编译器/toolkit 兼容性，而不是继续看 OneClick paste 逻辑。
    code = r'''
extern "C" __global__
void fill(unsigned short* dst, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) dst[i] = (unsigned short)7;
}
'''
    kernel = timed("RAWKERNEL-COMPILE", lambda: cp.RawKernel(code, "fill"))
    arr = timed(
        "RAWKERNEL-UPLOAD",
        lambda: cp.asarray(np.zeros((32,), dtype=np.uint16)),
        sync=lambda: cp.cuda.Device().synchronize(),
    )
    mark("RAWKERNEL-LAUNCH", "BEGIN kernel((1,), (32,), ...)")
    t0 = time.perf_counter()
    kernel((1,), (32,), (arr, 32))
    mark("RAWKERNEL-LAUNCH", f"RETURNED in {time.perf_counter() - t0:.6f}s")
    mark("RAWKERNEL-LAUNCH", "BEFORE CUDA SYNC")
    t1 = time.perf_counter()
    cp.cuda.Device().synchronize()
    mark("RAWKERNEL-LAUNCH", f"AFTER CUDA SYNC in {time.perf_counter() - t1:.6f}s")


@dataclass
class PasteState:
    cp: object
    files: object
    base_dec: object
    restored_dec: object | None
    packed: object
    y: object
    uv: object
    sy: object | None
    suv: object | None
    alpha_y: object | None
    alpha_c: object | None
    rect: tuple[int, int, int, int]


def _require_file(path: str, label: str) -> Path:
    if not path:
        raise SystemExit(f"--{label} is required for this case")
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"--{label} does not exist: {p}")
    return p


def _alpha_mask_cpu_upload(cp, w: int, h: int, px: int):
    import numpy as np

    px = max(0, min(int(px), max(0, min(w, h) // 2)))
    if px <= 0:
        return cp.asarray(np.ones((h, w), dtype=np.float32))
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    ax = np.minimum(np.minimum(xs, w - 1 - xs), px) / float(px)
    ay = np.minimum(np.minimum(ys, h - 1 - ys), px) / float(px)
    return cp.asarray((ay[:, None] * ax[None, :]).astype(np.float32))


def prepare_paste_state(
    args,
    *,
    need_restored: bool,
    need_alpha: bool,
) -> PasteState:
    import cupy as cp
    from gpu_engine import files
    from gpu_engine.pynv_io import PyNvThreadedSerialDecoder

    base = _require_file(args.base, "base")
    restored = _require_file(args.restored, "restored") if need_restored else None
    x, y0, w, h = args.rect
    bd = int(args.bit_depth)

    mark(
        "PASTE-ARGS",
        f"base={base} restored={restored} rect=(x={x}, y={y0}, w={w}, h={h}) "
        f"bit_depth={bd} alpha_source={args.alpha_source} alpha_eval={args.alpha_eval}",
    )

    # Decoder open can initialize CUDA/NVDEC resources. Mark it separately from
    # the first frame read so we know whether the block happens during resource
    # construction or during decode.
    base_dec = timed(
        "PYNV-BASE-OPEN",
        lambda: PyNvThreadedSerialDecoder(base, bit_depth=bd),
    )
    restored_dec = None
    if restored is not None:
        restored_dec = timed(
            "PYNV-RESTORED-OPEN",
            lambda: PyNvThreadedSerialDecoder(restored, bit_depth=bd),
        )

    # frame_at(0) is the first real decode request. The explicit CUDA sync after
    # it tells us whether PyNv returned a frame object while GPU decode/copy work
    # was still pending.
    base_frame = timed(
        "PYNV-BASE-FRAME0",
        lambda: base_dec.frame_at(0),
        sync=lambda: cp.cuda.Device().synchronize(),
    )
    mark("PYNV-BASE-VIEWS", "BEGIN frame.y_uv_cupy()")
    base_y, base_uv = base_frame.y_uv_cupy()
    mark("PYNV-BASE-VIEWS", f"RETURNED y={base_y.shape} uv={base_uv.shape}")

    restored_y = restored_uv = None
    if restored_dec is not None:
        restored_frame = timed(
            "PYNV-RESTORED-FRAME0",
            lambda: restored_dec.frame_at(0),
            sync=lambda: cp.cuda.Device().synchronize(),
        )
        mark("PYNV-RESTORED-VIEWS", "BEGIN frame.y_uv_cupy()")
        restored_y, restored_uv = restored_frame.y_uv_cupy()
        mark("PYNV-RESTORED-VIEWS", f"RETURNED y={restored_y.shape} uv={restored_uv.shape}")

    # Production encode path wants one packed buffer. This copy is a real
    # per-frame cost in paste, so it is both a possible hang point and a possible
    # FPS hotspot. The sync makes its cost visible in this diagnostic script.
    packed, packed_y, packed_uv = timed(
        "PACKED-COPY",
        lambda: files._copy_planes_to_packed_views(base_y, base_uv, bd),
        sync=lambda: cp.cuda.Device().synchronize(),
    )

    alpha_y = alpha_c = None
    if need_alpha:
        chroma_feather = 0 if 12 <= 0 else max(1, 12 // 2)
        if args.alpha_source == "prod":
            # This is the production alpha-mask path. It may trigger CuPy
            # elementwise/JIT work through cp.arange/minimum. If it hangs here,
            # rerun --case cupy-arange and then rerun paste-alpha with
            # --alpha-source cpu to bypass only mask generation.
            alpha_y = timed(
                "ALPHA-Y-PROD",
                lambda: files._make_alpha_mask(w, h, 12),
                sync=lambda: cp.cuda.Device().synchronize(),
            )
            alpha_c = timed(
                "ALPHA-C-PROD",
                lambda: files._make_alpha_mask(w // 2, h // 2, chroma_feather),
                sync=lambda: cp.cuda.Device().synchronize(),
            )
        else:
            # CPU mask upload is only an isolation tool. It is not proposed as a
            # production change. It answers: "if mask generation is not using
            # CuPy elementwise kernels, does the actual paste formula still
            # block?"
            alpha_y = timed(
                "ALPHA-Y-CPU-UPLOAD",
                lambda: _alpha_mask_cpu_upload(cp, w, h, 12),
                sync=lambda: cp.cuda.Device().synchronize(),
            )
            alpha_c = timed(
                "ALPHA-C-CPU-UPLOAD",
                lambda: _alpha_mask_cpu_upload(cp, w // 2, h // 2, chroma_feather),
                sync=lambda: cp.cuda.Device().synchronize(),
            )

    return PasteState(
        cp=cp,
        files=files,
        base_dec=base_dec,
        restored_dec=restored_dec,
        packed=packed,
        y=packed_y,
        uv=packed_uv,
        sy=restored_y,
        suv=restored_uv,
        alpha_y=alpha_y,
        alpha_c=alpha_c,
        rect=args.rect,
    )


def close_state(state: PasteState) -> None:
    for dec in (state.restored_dec, state.base_dec):
        if dec is not None:
            try:
                dec.stop()
            except Exception:
                pass


def case_pynv_read(args) -> None:
    state = prepare_paste_state(args, need_restored=bool(args.restored), need_alpha=False)
    close_state(state)


def case_paste_direct_copy(args) -> None:
    state = prepare_paste_state(args, need_restored=True, need_alpha=False)
    cp = state.cp
    x, y0, w, h = state.rect
    try:
        # Direct copy skips alpha math entirely. It still uses the same packed
        # base view and restored frame view as production paste, so it isolates
        # strided CuPy assignment and synchronization.
        rect_y = state.y[y0:y0 + h, x:x + w]
        mark("PASTE-DIRECT-Y", "BEGIN rect_y[...] = restored_y")
        t0 = time.perf_counter()
        rect_y[...] = state.sy
        mark("PASTE-DIRECT-Y", f"ASSIGNMENT RETURNED in {time.perf_counter() - t0:.6f}s")
        mark("PASTE-DIRECT-Y", "BEFORE CUDA SYNC")
        cp.cuda.Device().synchronize()
        mark("PASTE-DIRECT-Y", "AFTER CUDA SYNC")

        # UV is half resolution for 4:2:0. Some shape/stride issues only appear
        # on the chroma plane, so mark it independently from Y.
        rect_uv = state.uv[y0 // 2:(y0 + h) // 2, x // 2:(x + w) // 2]
        mark("PASTE-DIRECT-UV", "BEGIN rect_uv[...] = restored_uv")
        t1 = time.perf_counter()
        rect_uv[...] = state.suv
        mark("PASTE-DIRECT-UV", f"ASSIGNMENT RETURNED in {time.perf_counter() - t1:.6f}s")
        mark("PASTE-DIRECT-UV", "BEFORE CUDA SYNC")
        cp.cuda.Device().synchronize()
        mark("PASTE-DIRECT-UV", "AFTER CUDA SYNC")
    finally:
        close_state(state)


def _paste_alpha_prod_expression(state: PasteState, rect_y) -> None:
    """Run the original one-line luma formula with only coarse markers.

    用途：尽量复现生产路径的单表达式行为。如果这里卡住，只能知道“一整条
    alpha 表达式或写回”卡了；下一步应切到 split 模式看具体子步骤。
    """
    cp = state.cp
    alpha_y = state.alpha_y
    mark("PASTE-ALPHA-Y-PROD", "BEGIN original one-line production alpha formula")
    t0 = time.perf_counter()
    rect_y[:] = cp.rint(
        alpha_y * state.sy.astype(cp.float32)
        + (1.0 - alpha_y) * rect_y.astype(cp.float32)
    ).astype(state.y.dtype)
    mark("PASTE-ALPHA-Y-PROD", f"ASSIGNMENT RETURNED in {time.perf_counter() - t0:.6f}s")
    mark("PASTE-ALPHA-Y-PROD", "BEFORE CUDA SYNC")
    cp.cuda.Device().synchronize()
    mark("PASTE-ALPHA-Y-PROD", "AFTER CUDA SYNC")


def _paste_alpha_split_luma(state: PasteState, rect_y) -> None:
    """Run the luma alpha formula as small synchronized steps.

    This intentionally changes scheduling and adds many synchronizations. It is
    not an FPS benchmark and should not be compared with production speed.

    中文：split 模式只用于定位卡点。它把生产公式：

        round(alpha * restored + (1 - alpha) * base)

    拆成 astype、乘法、加法、rint、cast、写回，每一步后面都 sync 一次。
    这样同事拿到日志时，可以直接判断卡在 CuPy 类型转换、elementwise 乘法、
    加法、round、写回，还是 CUDA 同步。
    """
    cp = state.cp
    alpha_y = state.alpha_y

    sy_f = timed(
        "ALPHA-SPLIT-01-RESTORED-ASTYPE",
        lambda: state.sy.astype(cp.float32),
        sync=lambda: cp.cuda.Device().synchronize(),
    )
    base_f = timed(
        "ALPHA-SPLIT-02-BASE-ASTYPE",
        lambda: rect_y.astype(cp.float32),
        sync=lambda: cp.cuda.Device().synchronize(),
    )
    restored_weighted = timed(
        "ALPHA-SPLIT-03-RESTORED-WEIGHT",
        lambda: alpha_y * sy_f,
        sync=lambda: cp.cuda.Device().synchronize(),
    )
    inv_alpha = timed(
        "ALPHA-SPLIT-04-INVERT-ALPHA",
        lambda: 1.0 - alpha_y,
        sync=lambda: cp.cuda.Device().synchronize(),
    )
    base_weighted = timed(
        "ALPHA-SPLIT-05-BASE-WEIGHT",
        lambda: inv_alpha * base_f,
        sync=lambda: cp.cuda.Device().synchronize(),
    )
    blended = timed(
        "ALPHA-SPLIT-06-ADD",
        lambda: restored_weighted + base_weighted,
        sync=lambda: cp.cuda.Device().synchronize(),
    )
    rounded = timed(
        "ALPHA-SPLIT-07-RINT",
        lambda: cp.rint(blended),
        sync=lambda: cp.cuda.Device().synchronize(),
    )
    casted = timed(
        "ALPHA-SPLIT-08-CAST-TO-OUTPUT",
        lambda: rounded.astype(state.y.dtype),
        sync=lambda: cp.cuda.Device().synchronize(),
    )

    mark("ALPHA-SPLIT-09-WRITEBACK", "BEGIN rect_y[...] = casted")
    t0 = time.perf_counter()
    rect_y[...] = casted
    mark("ALPHA-SPLIT-09-WRITEBACK", f"ASSIGNMENT RETURNED in {time.perf_counter() - t0:.6f}s")
    mark("ALPHA-SPLIT-09-WRITEBACK", "BEFORE CUDA SYNC")
    cp.cuda.Device().synchronize()
    mark("ALPHA-SPLIT-09-WRITEBACK", "AFTER CUDA SYNC")


def case_paste_alpha(args) -> None:
    state = prepare_paste_state(args, need_restored=True, need_alpha=True)
    x, y0, w, h = state.rect
    try:
        rect_y = state.y[y0:y0 + h, x:x + w]
        if args.alpha_eval == "prod-expression":
            _paste_alpha_prod_expression(state, rect_y)
        else:
            _paste_alpha_split_luma(state, rect_y)
    finally:
        close_state(state)


def case_encode_first_frame(args) -> None:
    import cupy as cp
    from gpu_engine import files, probe
    from gpu_engine.pynv_io import PyNvEncoderSession

    # Encoder case intentionally skips restored decode and paste. It answers:
    # can the already-packed 8K frame be accepted by NVENC, and is first-frame
    # encode/flush where the local environment blocks?
    state = prepare_paste_state(args, need_restored=False, need_alpha=False)
    base = _require_file(args.base, "base")
    try:
        meta = timed("PROBE-BASE", lambda: probe.probe_video(base))
        bitrate_bps = int(meta.bitrate_bps or 30_000_000)
        enc_kwargs = timed("ENCODER-KWARGS", lambda: files._encoder_kwargs(meta, bitrate_bps))
        enc = timed(
            "NVENC-CREATE",
            lambda: PyNvEncoderSession(
                int(meta.width),
                int(meta.height),
                bit_depth=int(args.bit_depth),
                codec="hevc",
                **enc_kwargs,
            ),
        )
        app = timed(
            "APP-FRAME",
            lambda: files._app_frame_from_packed(
                state.packed,
                int(meta.width),
                int(meta.height),
                int(args.bit_depth),
            ),
        )
        mark("NVENC-ENCODE", "BEFORE CUDA SYNC")
        cp.cuda.Device().synchronize()
        mark("NVENC-ENCODE", "AFTER CUDA SYNC; BEGIN enc.encode(force_idr=True)")
        t0 = time.perf_counter()
        data = enc.encode(app, force_idr=True)
        mark("NVENC-ENCODE", f"RETURNED in {time.perf_counter() - t0:.6f}s bytes={len(data) if data else 0}")
        timed("NVENC-FLUSH", lambda: enc.flush())
    finally:
        close_state(state)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_runtime_env(args)
    cases: dict[str, Callable] = {
        "env": case_env,
        "cupy-copy": case_cupy_copy,
        "cupy-arange": case_cupy_arange,
        "rawkernel": case_rawkernel,
        "pynv-read": case_pynv_read,
        "paste-direct-copy": case_paste_direct_copy,
        "paste-alpha": case_paste_alpha,
        "encode-first-frame": case_encode_first_frame,
    }
    if args.case == "all":
        # "all" is convenient only when nothing hangs. If one case blocks, run
        # the next case separately after collecting the last checkpoint line.
        for name in [
            "env",
            "cupy-copy",
            "cupy-arange",
            "rawkernel",
            "pynv-read",
            "paste-direct-copy",
            "paste-alpha",
            "encode-first-frame",
        ]:
            explain_case(name)
            mark("CASE", f"BEGIN {name}")
            cases[name](args)
            mark("CASE", f"END {name}")
        return 0
    explain_case(args.case)
    cases[args.case](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
