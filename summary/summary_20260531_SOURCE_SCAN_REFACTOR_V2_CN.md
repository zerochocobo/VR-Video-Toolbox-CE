# 修订方案：源级筛 V2 — 砍 Stage 0、移除 ffmpeg 处理、源级粗合并

- 日期：2026-05-31
- 范围：源级时间筛（source_scan）的 Stage 0/2/4 + 源级聚合参数
- 前置：[summary_20260531_SOURCE_TIME_FILTER_PLAN_CN.md](summary_20260531_SOURCE_TIME_FILTER_PLAN_CN.md) 已完成的实现
- 触发：实测发现"问题非常大"，三点修订

---

## 0. 实测反馈的三个核心问题

1. **`ensure_dense_keyframes` 是反向优化**：为了"-c copy 切割对齐 I 帧"，先把 30 分钟 8K 源全片重编（~3 min + 几个 GB 临时文件），换来后面切割从 ~30s 降到 ~10s，**净亏**。粗剪本来就是粗的，细剪在 Stage 3 内层做。
2. **ffmpeg 子进程做切+concat 是性能洼地**：
   - 每个 segment 一次 ffmpeg subprocess（启动开销 + 进程间 IO）；
   - `concat_timeline` 的 reencode 兜底路径还是 ffmpeg；
   - 项目已经有成熟的 `gpu_engine`（NVDEC + CuPy + NVENC + `_EncodeSink`），应该用它。
3. **源级时间聚合太细**：当前复用 `pre_extract_*` 配置（`merge_gap_s=1.5s`、`min_segment_s=1.5s`、`head_tail_pad_s=2s`），出来的 mosaic_seg 普遍 5-15s，**每段 lada CLI 启动 ~6-8s** 等于纯启动开销吃掉 50%+。源级该用更粗的阈值。

---

## 1. 目标架构（V2）

```
源 SBS 8K
  │
  ├─[Stage 1] 源级时间扫描 (mosaic_prescan 直接喂源)
  │     +
  │     源级粗合并（新参数，独立于 pre_extract_*）
  │     输出 TimeInterval[]
  │     IF empty → skip 整个视频
  │
  ├─[Stage 2] GPU NVDEC→NVENC 切 mosaic_seg（仅切有马赛克段）
  │     gpu_engine.files.extract_clip(source, mosaic_seg{NN},
  │         start_sec=interval.start_s, end_sec=interval.end_s)
  │     不切 gap，gap 只是时间区间元数据
  │     ★ Stage 0 ensure_dense_keyframes 彻底删除
  │
  ├─[Stage 3] 对每个 mosaic_seg 跑完整短段管线（不变，复用 V1）
  │     split + [fisheye] + 内层 pre_extract + lada + paste + merge
  │     → mosaic_seg{NN}.restored.mp4
  │
  └─[Stage 4] GPU 单遍流式时间替换（新模块 replace_segments_gpu）
        decode 源 → 逐帧判断 →
            mosaic 时段：用对应 restored_seg 的帧
            gap 时段：用源帧
        → NVENC encode → 加回源音轨
        ★ 不用 ffmpeg concat，gap 完全不落盘
```

### 1.1 与 V1 的差异速查

| 项 | V1（已实现） | V2（本次修订） |
|---|---|---|
| Stage 0 dense GOP | ✓ 实现且默认 auto 启用 | **删除** |
| Stage 2 切割 | ffmpeg `-ss/-t -c copy` per-seg | `gpu_engine.files.extract_clip` 进程内 |
| Gap 段落盘 | ✓ 切成独立 gap_seg{MM}.mp4 | **不落盘**，只记元数据 |
| Stage 4 concat | ffmpeg concat demuxer（auto 重编 fallback） | 新模块 `replace_segments_gpu` 单遍 GPU 流式 |
| 源级聚合阈值 | 复用 `pre_extract_*`（1.5s 级别） | 新增 `source_scan_*` 独立参数（30-60s 级别） |

---

## 2. 改动清单

### 2.1 改动 A：删 Stage 0

**删除**：
- `utils/keyframe_cutter.py` 的 `ensure_dense_keyframes` + 辅助函数 `_video_file_is_usable`（无其它调用方）
- `utils/app_config.py` 的 `source_scan_inject_keyframes` / `source_scan_inject_gop_sec`
- `one_click/logic.py` `_run_source_scan_branch` 中 `inject_mode = str(app_config.get("source_scan_inject_keyframes", ...))` 整块（约 9 行）
- 相应单元测试

工作量小。注意：keyframe_cutter.py 还有 `list_keyframes` / `_floor_kf` / `_ceil_kf` 给内层 pre_extract 用，**不要删**。

### 2.2 改动 B：源级粗合并参数

**新增 `utils/app_config.py`**（独立于 `pre_extract_*`）：

```python
# 源级聚合（粗合并，专给 source_scan）
'source_scan_merge_gap_s': 30.0,         # 两段相距 ≤30s 合并
'source_scan_min_segment_s': 30.0,       # 段时长 <30s 自动 pad 到 30s
'source_scan_head_tail_pad_s': 5.0,      # 每段首尾各 pad 5s
'source_scan_max_segment_s': 0.0,        # >0 时段时长上限，超出强行不再合并；0 = 无上限
```

**修改 `utils/source_time_scanner.py` `scan_source_time_segments`**：

`mosaic_prescan.scan_segments` 内部用 `pre_extract_*` 配置（保留不动，给内层 pre_extract 用）。在 `scan_source_time_segments` 拿到 segments 后，**再跑一轮源级粗合并**：

```python
def _coarse_merge(intervals: list[TimeInterval]) -> list[TimeInterval]:
    if not intervals:
        return []
    merge_gap = float(app_config.get("source_scan_merge_gap_s", 30.0) or 30.0)
    min_seg = float(app_config.get("source_scan_min_segment_s", 30.0) or 30.0)
    pad = float(app_config.get("source_scan_head_tail_pad_s", 5.0) or 5.0)
    max_seg = float(app_config.get("source_scan_max_segment_s", 0.0) or 0.0)

    # 1) 先 head/tail pad (放大每段)
    padded = []
    for itv in intervals:
        padded.append(TimeInterval(
            start_s=max(0.0, itv.start_s - pad),
            end_s=itv.end_s + pad,
            conf_max=itv.conf_max,
        ))
    # 2) 大间距合并（pad 后再合并以处理 pad 引起的新重叠）
    padded.sort(key=lambda x: x.start_s)
    merged = []
    for itv in padded:
        if merged and itv.start_s - merged[-1].end_s <= merge_gap:
            # 若有 max_seg 上限，超过则不合并
            if max_seg > 0 and (max(merged[-1].end_s, itv.end_s) - merged[-1].start_s) > max_seg:
                merged.append(itv)
            else:
                merged[-1].end_s = max(merged[-1].end_s, itv.end_s)
                merged[-1].conf_max = max(merged[-1].conf_max, itv.conf_max)
        else:
            merged.append(itv)
    # 3) min_seg 强制扩段（对称扩，扩完可能再次重叠 → 再做一遍合并）
    extended = []
    grew = False
    for itv in merged:
        dur = itv.end_s - itv.start_s
        if dur < min_seg:
            grow = (min_seg - dur) / 2
            extended.append(TimeInterval(
                start_s=max(0.0, itv.start_s - grow),
                end_s=itv.end_s + grow,
                conf_max=itv.conf_max,
            ))
            grew = True
        else:
            extended.append(itv)
    if grew:
        return _coarse_merge_no_pad(extended, merge_gap, max_seg)
    return extended

def _coarse_merge_no_pad(intervals, merge_gap, max_seg):
    intervals.sort(key=lambda x: x.start_s)
    out = []
    for itv in intervals:
        if out and itv.start_s - out[-1].end_s <= merge_gap:
            if max_seg > 0 and (max(out[-1].end_s, itv.end_s) - out[-1].start_s) > max_seg:
                out.append(itv)
            else:
                out[-1].end_s = max(out[-1].end_s, itv.end_s)
                out[-1].conf_max = max(out[-1].conf_max, itv.conf_max)
        else:
            out.append(itv)
    return out

def scan_source_time_segments(source_sbs, log_callback=None, cancel_token=None):
    from utils import mosaic_prescan, app_config
    segments = mosaic_prescan.scan_segments(source_sbs, ...)
    intervals = [TimeInterval(...) for seg in segments if ...]
    # 先做现有的 0.05s 相邻合并（已存在 _merge_intervals）
    intervals = _merge_intervals(intervals, gap_s=0.05)
    # 再做源级粗合并（新）
    intervals = _coarse_merge(intervals)
    return intervals
```

**效果**：对于一个 30 min 视频，原来可能产 30 个 5-15s 小段；新逻辑下产 3-8 个 60-120s 大段。**lada 段数 ↓ 5-10x，启动开销 ↓ 同比例**。

### 2.3 改动 C：Stage 2 切割改 GPU

**删除**：
- `utils/keyframe_cutter.py` 的 `_cut_copy`（ffmpeg `-c copy` 切）
- `cut_source_by_intervals` 里 ffmpeg 调用部分

**新增 `utils/keyframe_cutter.py`**（保留函数签名兼容外层调用）：

```python
def cut_source_by_intervals(src, intervals, out_dir,
                            keyframes=None,  # 接受但不再使用
                            log_callback=None, process_callback=None
                            ) -> list[TimelineEntry]:
    """GPU 切源 SBS 成 mosaic_seg。不切 gap（gap 只是元数据）。"""
    from gpu_engine import probe, files as gpu_files

    src = Path(src); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    meta = probe.probe_video(src)
    duration = float(meta.duration or 0.0)

    # 准备 mosaic interval（已粗合并的输入）
    mosaic_specs = []
    for interval in intervals:
        start = max(0.0, _interval_value(interval, "start_s"))
        end = min(duration if duration > 0 else float("inf"),
                  _interval_value(interval, "end_s"))
        conf = _interval_value(interval, "conf_max")
        if end - start > 0.05:
            mosaic_specs.append((start, end, conf))
    mosaic_specs.sort()

    # GPU 切 mosaic_seg
    mosaic_entries = []
    for idx, (start, end, conf) in enumerate(mosaic_specs):
        path = out_dir / f"mosaic_seg{idx:03d}.mp4"
        if log_callback:
            log_callback(f"[source-scan] GPU extract mosaic seg {idx}: {start:.3f}-{end:.3f}s -> {path.name}")
        token = gpu_files.CancelToken()
        if process_callback:
            process_callback(token)
        gpu_files.extract_clip(
            src, path,
            crop_mode=None,           # 不裁空间, 保留全帧
            to_fisheye=False,
            start_sec=start, end_sec=end,
            cq=18,                    # 与现有 GPU 管线一致
            keep_audio=True,          # 音频按 ffmpeg `-c copy` 走（已在 mux 内）
            log_callback=log_callback,
            cancel_token=token,
        )
        mosaic_entries.append(TimelineEntry(
            start_s=start, end_s=end, path=path, kind="mosaic", conf_max=conf,
        ))

    # gap 不切, 只记元数据
    gap_entries = []
    cursor = 0.0
    for entry in mosaic_entries:
        if entry.start_s - cursor > 0.05:
            gap_entries.append(TimelineEntry(
                start_s=cursor, end_s=entry.start_s,
                path=Path(src),  # 引用源, 不生成新文件
                kind="gap", conf_max=0.0,
            ))
        cursor = max(cursor, entry.end_s)
    if duration > 0 and duration - cursor > 0.05:
        gap_entries.append(TimelineEntry(
            start_s=cursor, end_s=duration, path=Path(src),
            kind="gap", conf_max=0.0,
        ))

    timeline = sorted(mosaic_entries + gap_entries, key=lambda e: e.start_s)
    return timeline
```

**注意**：
- `extract_clip` 已支持 `start_sec/end_sec`（[gpu_engine/files.py:709-711](gpu_engine/files.py)），re-encode 时帧精确，**不依赖关键帧**——这就是为什么不需要 Stage 0。
- 音频用 mux_hevc_with_audio 自动从源段（extract_clip 内已处理）取，无额外工作。
- gap 不落盘 → 节省磁盘 IO、节省一次编码 + 一次解码。

### 2.4 改动 D：Stage 4 改 GPU 单遍流式

**删除**：
- `utils/sbs_concat.py` 整个文件（基于 ffmpeg concat）

**新增 `gpu_engine/files.py` `replace_segments_gpu`**：

```python
def replace_segments_gpu(
    source: str | Path,
    dst: str | Path,
    segments: list,  # TimelineEntry-like, only those with kind=="mosaic"
    *,
    cq: int | None = 18,
    bitrate_bps: int | None = None,
    keep_audio: bool = True,
    log_callback=None,
    cancel_token: "CancelToken | None" = None,
) -> Path:
    """单遍 GPU 流式：decode 源, 在 mosaic 时段用对应 restored_seg 的帧, 其余直通源.

    segments 每项要求:
        path           : 已恢复的 mosaic_seg 文件
        base_frame_start : 在源 timeline 上的起始帧号（含）
        base_frame_end   : 在源 timeline 上的结束帧号（不含）
    segments 不应有时间重叠（Stage 1 已粗合并）。
    """
    import cupy as cp

    source = Path(source); dst = Path(dst)
    meta, decision = probe.route(source)
    if not decision.is_gpu:
        raise RuntimeError(f"source not GPU-eligible: {decision.reason}")
    bd = 10 if meta.bit_depth > 8 else 8
    src_dec = PyNvThreadedSerialDecoder(source, bit_depth=bd)
    info = src_dec.info
    fps = meta.source_fps or info.fps or 30.0
    total = len(src_dec)

    segs = sorted(segments, key=lambda s: int(s.base_frame_start))
    bitrate_bps = _resolve_bitrate(info.width, info.height, fps,
                                    bitrate_bps, meta.bitrate_bps)
    enc = PyNvEncoderSession(info.width, info.height, bit_depth=bd, codec="hevc",
                              **_encoder_kwargs(meta, bitrate_bps))
    raw = Path(tempfile.gettempdir()) / f"{dst.stem}.replace.raw.hevc"

    next_idx = 0
    active = None
    done = 0
    cancelled = False
    prog = _Progress(total, log_callback)
    try:
        with open(raw, "wb") as f:
            sink = _EncodeSink(enc, f)
            for i in range(total):
                if cancel_token is not None and cancel_token.cancelled:
                    cancelled = True; break

                # 段活跃管理（最多 1 个 active, 因不重叠）
                if active is not None and i >= int(active.base_frame_end):
                    active["dec"].stop(); active = None
                if active is None and next_idx < len(segs) \
                        and int(segs[next_idx].base_frame_start) <= i:
                    seg = segs[next_idx]; next_idx += 1
                    if int(seg.base_frame_end) <= i:
                        continue
                    seg_meta = probe.probe_video(seg.path)
                    seg_bd = 10 if seg_meta.bit_depth > 8 else 8
                    seg_dec = PyNvThreadedSerialDecoder(seg.path, bit_depth=seg_bd)
                    active = {"seg": seg, "dec": seg_dec, "bd": seg_bd,
                              "frames": len(seg_dec)}

                # 源永远 advance（保持 NVDEC 顺序解码）
                src_frame = src_dec.frame_at(i)
                cp.cuda.Device().synchronize()

                if active is not None:
                    sidx = i - int(active["seg"].base_frame_start)
                    if 0 <= sidx < active["frames"]:
                        sf = active["dec"].frame_at(sidx)
                        sy, suv = sf.y_uv_cupy()
                        # 跨位深处理（同 paste_segments_gpu 的 _match_depth）
                        if active["bd"] != bd:
                            sy = _match_depth(sy, active["bd"], bd)
                            suv = _match_depth(suv, active["bd"], bd)
                        y, uv = sy, suv
                    else:
                        # 段已耗尽但还在 base 范围内 → 用源兜底
                        y, uv = src_frame.y_uv_cupy()
                else:
                    y, uv = src_frame.y_uv_cupy()

                app = _pack_planes(y, uv, bd)
                sink.feed(app, force_idr=(i == 0))
                done += 1
                prog.update(done)
            if not cancelled:
                prog.finish(done); sink.flush()
    finally:
        if active is not None:
            try: active["dec"].stop()
            except Exception: pass
        src_dec.stop()
        runtime.free_memory_pool()

    if cancelled:
        try: raw.unlink()
        except OSError: pass
        raise OperationCancelled("cancelled by user")

    mux.mux_hevc_with_audio(
        raw, dst, fps=fps, color=meta.color,
        audio_source=str(source) if keep_audio else None,
        log_callback=log_callback,
    )
    try: raw.unlink()
    except OSError: pass
    return dst


def _match_depth(arr, src_bd: int, dst_bd: int):
    """与 paste_segments_gpu 同款（复用现有函数, 或抽到模块 top）"""
    import cupy as cp
    if src_bd == dst_bd: return arr
    if dst_bd > 8 and src_bd <= 8: return arr.astype(cp.uint16) * cp.uint16(257)
    if dst_bd <= 8 and src_bd > 8:
        return cp.rint(arr.astype(cp.float32) * (255.0 / 65535.0)).astype(cp.uint8)
    return arr
```

**实现要点**：
- 跟 `paste_segments_gpu` 同栈（PyNv decode/encode + `_EncodeSink` + `mux_hevc_with_audio`），引擎层已经处理好 NVENC 输入生命周期、码率、色彩等。
- 单段 active（不重叠），逻辑比 paste_segments_gpu 更简单（无 alpha blend、无 rect 切片）。
- 源 decoder 始终 advance（避免 NVDEC 状态错乱），即使该帧不用也要拉。NVDEC 解码本身廉价，编码不付那帧的代价。
- 8↔10 bit 兼容（mosaic_seg 经 lada/jasna 后可能位深变化，沿用 paste 的 `_match_depth`）。

**外层调用**：

`one_click/logic.py` `_run_source_scan_branch` 改：

```python
# 删除 from utils.sbs_concat import concat_timeline
from gpu_engine.files import replace_segments_gpu

# ... Stage 3 已完成, restored 文件路径在 entry.path 上 ...

# Stage 4: GPU 单遍流式时间替换
if log_callback:
    log_callback("[source-scan] Stage 4: GPU stream replace")

# 准备 segments: 仅 mosaic kind, 转 base_frame_start/end
from gpu_engine import probe as gpu_probe
src_meta = gpu_probe.probe_video(scan_input)
fps = src_meta.source_fps or 30.0
seg_specs = []
for entry in timeline:
    if entry.kind != "mosaic": continue
    seg_specs.append(SimpleNamespace(
        path=entry.path,
        base_frame_start=int(round(entry.start_s * fps)),
        base_frame_end=int(round(entry.end_s * fps)),
    ))

token = CancelToken()
if process_callback: process_callback(token)
replace_segments_gpu(
    source=scan_input,
    dst=final_path,
    segments=seg_specs,
    cq=None if (keep_original_bitrate and original_bitrate) else 18,
    bitrate_bps=int(original_bitrate) if (keep_original_bitrate and original_bitrate) else None,
    keep_audio=True,
    log_callback=log_callback,
    cancel_token=token,
)
```

### 2.5 改动 E：清理已废弃配置和代码

- 删除 `source_scan_concat_reencode` 配置（无 concat 步骤）
- 删除 `source_scan_full_coverage_ratio` / `source_scan_full_coverage_gap_s`（"全片覆盖直通"逻辑可保留，但因为 Stage 4 现在是单遍 GPU 直通，此逻辑已无收益，建议删除）
- 删除 `_run_source_scan_branch` 中 `_intervals_cover_full_duration` 分支（同上）

简化后的 `_run_source_scan_branch` 流程：

```python
def _run_source_scan_branch(input_file, final_output, *, use_fisheye, pre_extract_inner,
                            keep_intermediate, keep_original_bitrate,
                            mode="sbs", eye_mode=None,
                            log_callback=None, process_callback=None) -> str:
    # 检查 HDR 等约束
    meta = gpu_probe.probe_video(input_file)
    if meta.is_hdr or meta.is_bt2020:
        return PreExtractResult.SCAN_FAILED

    tmp_dir = ...; os.makedirs(tmp_dir, exist_ok=True)
    keep_segments = bool(app_config.get("source_scan_keep_segments", False)) or keep_intermediate
    try:
        # Stage 1
        intervals = scan_source_time_segments(input_file, ...)
        save_source_intervals_json(intervals, ...)
        if not intervals:
            log_callback(f"[source-scan] no mosaic, skipping: {input_file}")
            return PreExtractResult.NO_MOSAIC

        # Stage 2: GPU 切 mosaic_seg
        timeline = cut_source_by_intervals(input_file, intervals, tmp_dir, ...)
        save_timeline_json(timeline, ...)

        # Stage 3: 每个 mosaic_seg 跑短段管线
        for entry in timeline:
            if entry.kind != "mosaic": continue
            restored = entry.path.with_name(f"{entry.path.stem}.restored.mp4")
            if mode == "single_eye":
                _process_single_eye_clip_to_output(entry.path, restored, ...)
            else:
                _process_sbs_clip_to_output(entry.path, restored, ...)
            entry.path = restored

        # Stage 4: GPU 单遍流式替换
        seg_specs = [...]  # 见 2.4
        replace_segments_gpu(input_file, final_path, seg_specs, ...)
        return PreExtractResult.OK
    finally:
        if not keep_segments:
            shutil.rmtree(tmp_dir, ignore_errors=True)
```

---

## 3. 时间预估（8K SBS 30min @ 25% 马赛克密度）

| 阶段 | V1 (已实现) | V2 (本次修订) |
|---|---|---|
| Stage 0 dense GOP | ~3 min (全片重编) | **0** (删除) |
| Stage 1 源扫 | ~2 min | ~2 min |
| Stage 2 切 mosaic_seg | <30s (`-c copy`) | ~2-3 min (GPU re-encode 7.5min content) |
| Stage 2 切 gap_seg | <30s | **0** (不落盘) |
| Stage 3 lada/jasna | 15-25 min | **10-18 min** ↓（粗合并使段数 ↓ → 启动开销 ↓） |
| Stage 4 concat | ~30s (-c copy) 或 ~3 min (reencode) | ~6-8 min (GPU 单遍 decode 30min + encode 30min @ 8K) |
| **合计** | **~21-34 min** | **~20-31 min** |

实测时间预计与 V1 持平或略优，但**几个关键质性改进**：
1. **磁盘 IO ↓**：不再有 dense_gop temp 文件（~4-8 GB）+ 不再有 gap_seg 文件
2. **段数 ↓**：lada 调用次数 ~5x 减少，CLI 启动开销摊销大幅改善
3. **代码统一**：cut/concat 都走 gpu_engine，与项目主线一致
4. **无 keyframe 漂移风险**：GPU 切是帧精确，不依赖源 GOP 结构

**对边界场景**：
- 无马赛克：Stage 1 扫完空 → 直接返回 NO_MOSAIC（不变）
- 高密度（>80%）：Stage 2 切的就是几乎整片 → Stage 4 几乎全是 mosaic 路径 → 几乎退化为现有 pre-extract 行为（合理）
- 短视频（<5min）：粗合并后可能只有 1 个 mosaic_seg = 整片 → Stage 4 单段全替换 ≈ 直接走 pre-extract（合理）

---

## 4. 兼容性 / 风险

| # | 风险 | 缓解 |
|---|---|---|
| R1 | Stage 4 单遍流式与现有 paste_segments_gpu 共享 PyNv 同时多 decoder 的资源 | mosaic_seg 之间不重叠，最多 1 段 active；NVDEC 一次只用 1 路 |
| R2 | source 与 mosaic_seg 位深不一致（lada 输出 8-bit, 源 10-bit）| 已用 `_match_depth`（沿用 paste_segments_gpu） |
| R3 | Stage 2 GPU 切的 mosaic_seg 编码参数与源不齐 → Stage 4 拼接颜色突变 | Stage 4 是统一 NVENC 编码（一次性输出全片），不存在拼接突变；只要 source/mosaic_seg 解码后**色彩一致**即可，extract_clip 已保留色彩元数据 |
| R4 | replace_segments_gpu 是新代码 | 跟 paste_segments_gpu 同栈，单元测试用全 mosaic 段（覆盖整片）验证 PSNR vs paste_segments_gpu 等价 |
| R5 | 删 Stage 0 后碰到极端大 GOP (60s+) 源 | 切 mosaic_seg 是 GPU re-encode，与 GOP 无关，无影响 |
| R6 | 源级粗合并使 lada 处理某些"无马赛克"内容（被合并到大段内的间隙） | 段内的 Stage 3 内层 pre_extract 会再做细筛，只 lada 真有马赛克的小区域，**总 lada 计算量不变**，只是少跑了几次启动 |
| R7 | gap 不落盘 → Stage 4 失败时无法续跑 mosaic_seg 已处理结果 | restored mosaic_seg 在 tmp_dir, keep_intermediate=True 时仍保留, 重跑 Stage 4 不重新跑 Stage 3 |

---

## 5. 配置默认值表（决策依据）

| 参数 | 默认 | 理由 |
|---|---|---|
| `source_scan_merge_gap_s` | 30.0 | 30s 内的两个马赛克段合并；lada CLI 启动 ~8s, 合并比"分跑两次"快 |
| `source_scan_min_segment_s` | 30.0 | 单段 ≥30s, 让 lada 跑 ≥150 帧才有意义（启动开销摊得开） |
| `source_scan_head_tail_pad_s` | 5.0 | 给 Stage 3 内层重扫留富余, 避免边缘漏检 |
| `source_scan_max_segment_s` | 0.0 | 无上限；超大段（>10min）也允许，依赖 Stage 3 内层 rect 切 |
| `source_scan_keep_segments` | False | 默认清 tmp_dir; 调试时改 True |

---

## 6. 实施任务分解（建议 commit 顺序）

按"小步快跑、独立验证"原则拆：

1. **任务 A — 删 Stage 0**
   - 删 `ensure_dense_keyframes` + 相关配置 + 调用点
   - 跑现有 source_scan 全片测试，确认正常完成（速度会变慢，因为 ffmpeg `-c copy` 切对齐 pad 大；后续任务 C 修）

2. **任务 B — 源级粗合并参数**
   - `source_time_scanner` 加 `_coarse_merge`
   - `app_config` 加 4 个新配置
   - 拿同一个视频测：observe `source_intervals.json` 段数应 ↓ 5-10x

3. **任务 C — Stage 2 GPU 切**
   - 重写 `cut_source_by_intervals`：mosaic_seg 走 `gpu_engine.files.extract_clip`，gap 不落盘
   - `TimelineEntry` 含义微调（gap 的 path 指向源, kind="gap"）
   - 端到端测试: 不带 Stage 4 改的情况下用现有 ffmpeg concat 应仍可拼回（兼容过渡）

4. **任务 D — Stage 4 GPU 替换**
   - `gpu_engine/files.py` 加 `replace_segments_gpu` + `_match_depth`（如未抽出）
   - `_run_source_scan_branch` 替换 Stage 4 调用
   - 删 `utils/sbs_concat.py`
   - 单元测试: 全 mosaic 覆盖测试（PSNR vs 现有 pre_extract 等价 ≥60dB）

5. **任务 E — 配置/UI 清理**
   - 删 `source_scan_inject_keyframes` / `source_scan_inject_gop_sec` / `source_scan_concat_reencode` / `source_scan_full_coverage_*`
   - 清理 `_intervals_cover_full_duration` 分支
   - 更新 i18n 说明（如有提到 dense keyframe 的）

6. **任务 F — 实测对比**
   - 8K SBS 30min × {马赛克密度 10% / 30% / 60%} × {V1 / V2}
   - 记录 wall time、磁盘峰值用量、lada 调用次数、PSNR
   - 写实测报告 `summary/summary_..._SOURCE_SCAN_V2_BENCH_CN.md`

每个任务独立可验证；A/B 先 commit 即可见效（A 节省 ~3 min Stage 0，B 节省 lada 启动开销）；C/D/E 是 GPU 化的完整切换。

---

## 7. 待确认的细节（开发可按现实调整）

1. **gap 段的色彩元数据透传**：Stage 4 输出的 NVENC 编码参数应与源一致（色彩、profile）。当前 `_encoder_kwargs` 已基于源 meta 派生，理论 OK；实测确认。
2. **音频处理**：mosaic_seg 通过 extract_clip 已带音频；gap 时段不落盘但 Stage 4 用 `mux_hevc_with_audio(audio_source=源)` 取整段音频，对齐到完整时间轴。一次性 -c copy 音频流，无切换问题。
3. **`pre_extract_keep_segments=False` 时**：Stage 3 内层会清掉自己的 seg；source_scan 的 tmp_dir 由 source_scan_keep_segments 控制独立清理。两层 keep flag 是否要联动？建议独立（current behavior, keep）。
4. **Cancel 透传**：每个 Stage 都有自己的 CancelToken；只要每段开始时 `process_callback(token)` 更新到 UI，stop 按钮就能在任何阶段中断。当前实现已对齐，无需改。
