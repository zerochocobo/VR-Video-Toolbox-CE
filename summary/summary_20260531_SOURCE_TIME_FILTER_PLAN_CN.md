# 方案：源级时间筛 + Multi-region Pre-extract + Skip-on-empty

- 日期：2026-05-31
- 范围：`one_click/` 的四个 Tab（单文件/单眼/批量/批量单眼）
- 适用引擎：**Phase 1 仅 `lada` / `jasna` CLI**（`native_gpu` 留 Phase 2）
- 前置：基于已完成的 pre-extract 方案（[summary_20260530_PRE_EXTRACT_MOSAIC_REGION_PLAN_CN.md](summary_20260530_PRE_EXTRACT_MOSAIC_REGION_PLAN_CN.md)）
- 关联文档：[[gpu-engine-architecture]]、`summary_20260531_NATIVE_BOTTLENECK_PROFILE_CN.md`

---

## 1. 用户需求与本期目标

### 1.1 三点修订（来自用户最新反馈）

1. **`lada_vr_mosaic_detection_model_v2_fast.pt` 在非鱼眼模式下也能识别**。因此可以**直接在原始 SBS VR 视频上**做"是否有马赛克"的判断 + 时间区间切割。**这一步只切时间，不切空间**。
2. **如果整片扫描检测不到任何马赛克 → 直接跳过整个视频的后续处理**（不输出、不跑 lada/jasna）。Batch 场景下收益最大。
3. **lada/jasna 之前的检测（即现有 pre-extract 的内层）需要支持多个空间马赛克区域**。当前 `_aggregate_hits` 把段内所有 box 取并集，碰到"两眼各一个马赛克"或"同帧两人各一处"时会产出**横跨大半屏的巨型 rect**，等于让 lada 吃几乎全帧——本期一并修复。

### 1.2 性能目标

8K SBS 30min @ 25% 马赛克密度：当前 pre-extract ~45–60 min → 目标 ~22–35 min（节省 35–50%）。
**对无马赛克视频**：当前 ~45–60 min → 目标 ~2 min（扫完空 → 整片跳过）。Batch 隔夜跑收益巨大。

---

## 2. 最终架构（三层）

```
源 SBS 8K
  │
  ├─[Stage 0] 可选：大 GOP 兜底
  │     若源 GOP > 5s → GPU 重编一遍注入 2s 密集 I 帧（仅一次性 ~3 min）
  │     输出 → 作为 Stage 1 的输入
  │
  ├─[Stage 1] 源级时间筛
  │     mosaic_prescan.scan_segments(source_sbs) ← 直接喂源，无投影
  │     **只保留时间区间，丢弃 rect**
  │     IF 空 → log "无马赛克，跳过" → return / batch continue
  │
  ├─[Stage 2] -c copy 按 kf 切源 SBS
  │     按 Stage 1 的时间区间 + 间隙生成完整时间轴：
  │       → mosaic_seg{NN}.mp4 (SBS 全帧, 有马赛克时段, 进 Stage 3)
  │       → gap_seg{MM}.mp4    (SBS 全帧, 间隙,       直通 Stage 4)
  │
  ├─[Stage 3] 对每个 mosaic_seg 跑完整短段管线
  │     split L/R [+ VR→Fisheye 若 use_fisheye=True]
  │       │
  │       └─ 内层 pre_extract (NEW: 多区域支持)
  │            ├─ 在 L_fish (或 L) 上扫描 → N 个空间 cluster rects
  │            ├─ 每个 rect 一次 cut + lada/jasna
  │            └─ paste 回 L_fish
  │       └─ 右眼同理
  │       └─ [Fisheye→VR] + merge → restored_seg{NN}.mp4 (SBS)
  │
  └─[Stage 4] concat: gap_seg + restored_seg 按时间顺序拼回
        默认 -c copy（参数对齐时）；不齐时一遍 NVENC 重编兜底
```

### 2.1 核心不变量（实施必须严守）

| # | 不变量 | 影响 |
|---|---|---|
| I1 | Stage 1 **只产时间**，Stage 3 **只产 rect** | 两层检测语义独立；singleton detector 共用 |
| I2 | 所有 -c copy 切割必须 **kf 对齐** | 拼回不出现时间漂移 / 帧损失；大 GOP 必须 Stage 0 兜底 |
| I3 | concat 前所有段 codec/pix_fmt/profile/resolution/color **必须一致** | Stage 3 输出参数对齐源；不齐自动重编 |
| I4 | `PreExtractResult` 三态严格区分 | `NO_MOSAIC`=设计预期跳过；`SCAN_FAILED`=异常回退；`OK`=正常 |
| I5 | Stage 1 与 Stage 3 detector **同一个 singleton** | 模型只加载一次，跨阶段复用 |

### 2.2 概念图：两层检测的语义差异

```
Stage 1 (源级):                Stage 3 (短段 fisheye 级):
  问: 有没有 + 什么时候有       问: 哪里有 + 多大
  输入: 源 SBS (整片)           输入: L_fish / R_fish (短段)
  输出: [t_start, t_end]        输出: [x, y, w, h] × N
  用途: 切时间区间, 决定跳过     用途: cut rect → lada → paste
  rect: 丢弃                    multi-region: 启用
```

---

## 3. 改动清单

### 3.1 改动 A：multi-region 聚类（修现有 pre-extract 的 bug）

#### A1. `utils/mosaic_prescan.py` 新增 `_spatial_cluster`

```python
def _spatial_cluster(boxes, frame_w, frame_h):
    """段内所有 box 按空间邻近度聚类。
    两 box overlap 或 inflate(gap_px) 后相交 → 同簇。
    返回 list[list[box]]，每簇一个 list。
    """
    if len(boxes) <= 1:
        return [boxes] if boxes else []
    gap_ratio = float(_cfg("pre_extract_cluster_gap_ratio", 0.03) or 0.03)
    gap_px = max(20, int(min(frame_w, frame_h) * gap_ratio))
    n = len(boxes)
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            ax1, ay1, ax2, ay2 = boxes[i][:4]
            bx1, by1, bx2, by2 = boxes[j][:4]
            if (ax2 + gap_px >= bx1 and bx2 + gap_px >= ax1
                and ay2 + gap_px >= by1 and by2 + gap_px >= ay1):
                parent[find(i)] = find(j)
    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(boxes[i])
    return list(clusters.values())
```

#### A2. `_aggregate_hits` 末尾：每时段输出 N 个 cluster segment

```python
# 替换原本只产一个 segment 的循环
for group in groups:
    if group["end"] - group["start"] < min_segment_s:
        continue
    for cluster_boxes in _spatial_cluster(group["boxes"], meta.width, meta.height):
        x, y, w, h, conf = _expanded_rect(cluster_boxes, meta.width, meta.height)
        if w <= 0 or h <= 0:
            continue
        segments.append(MosaicSegment(
            seg_id=len(segments),
            start_s=group["start"], end_s=group["end"],
            start_s_kf=group["start"], end_s_kf=group["end"],
            x=x, y=y, w=w, h=h, conf_max=conf,
        ))
```

#### A3. `utils/keyframe_cutter.align_segments`：合并条件改"时间 AND rect 都重叠"

当前是"时间重叠即 `_merge_rect`"——会把刚分出来的多簇又合回去。改：

```python
def _rects_overlap_with_gap(a, b, gap=16):
    return (a.x + a.w + gap >= b.x and b.x + b.w + gap >= a.x
            and a.y + a.h + gap >= b.y and b.y + b.h + gap >= a.y)

# align_segments 内部循环改：
if merged:
    prev = merged[-1]
    time_overlap = seg.start_s_kf <= prev.end_s_kf + 1e-3
    rect_overlap = _rects_overlap_with_gap(prev, seg)
    if time_overlap and rect_overlap:
        # 现有的 _merge_rect 路径
        prev.start_s = min(prev.start_s, seg.start_s)
        prev.end_s = max(prev.end_s, seg.end_s)
        prev.start_s_kf = min(prev.start_s_kf, seg.start_s_kf)
        prev.end_s_kf = max(prev.end_s_kf, seg.end_s_kf)
        prev.x, prev.y, prev.w, prev.h = _merge_rect(prev, seg)
        prev.conf_max = max(prev.conf_max, seg.conf_max)
        continue
merged.append(seg)
```

#### A4. `paste_segments_gpu`：**不需要改动**

已支持 active list 多段并发贴回；同时段不同 rect 的多个 PasteSeg 会同时进入 active，各贴各的位置。

唯一隐患：NVDEC 并发段上限。RTX 5060 Ti 实测 ≤4 路并发安全。Phase 1 假设 multi-region 实际 cluster 数 ≤ 4；超出时打 warning 并分批贴回（用上次方案的 batch-as-base 思路），但**本期可不实现**——典型素材 cluster 数都很小。

### 3.2 改动 B：PreExtractResult 三态 + Skip-on-empty

#### B1. `one_click/logic.py` 引入 enum

```python
class PreExtractResult:
    OK = "ok"
    NO_MOSAIC = "no_mosaic"     # 设计预期：扫完就是没找到马赛克
    SCAN_FAILED = "scan_failed" # 异常：模型/IO 错误
```

#### B2. `_run_pre_extract_branch` 返回值改 enum

```python
def _run_pre_extract_branch(base_path, restored_path, ...) -> str:
    ...
    try:
        segments = scan_segments(base_path, log_callback, cancel_token=scan_token)
    except Exception as e:
        log_callback(f"[pre-extract] scan failed: {type(e).__name__}: {e}")
        return PreExtractResult.SCAN_FAILED
    if not segments:
        log_callback("[pre-extract] no mosaic detected after scan")
        return PreExtractResult.NO_MOSAIC
    # ... 现有 cut / lada / paste 逻辑 ...
    return PreExtractResult.OK
```

#### B3. 所有调用点判断三态

```python
# 内层 pre_extract (Stage 3 内, mosaic_seg 上跑)
result = _run_pre_extract_branch(L_fish, L_fish_restored, ...)
if result == PreExtractResult.NO_MOSAIC:
    # Stage 1 报有马赛克但内层 fisheye 上没找到 (conf 阈值差异 / 投影后失真)
    # 不跑 lada, 直接复制原 L_fish 当作"已恢复"
    shutil.copy2(L_fish, L_fish_restored)
elif result == PreExtractResult.SCAN_FAILED:
    # 真扫描出错 → 保守回退全段 lada
    process_lada(L_fish, L_fish_restored, ...)
# OK 正常
```

#### B4. Stage 1 的 skip 语义

```python
# _run_source_scan_branch 顶部
intervals = scan_source_time_segments(input_file, log_callback, scan_token)
if not intervals:
    log_callback(f"[source-scan] No mosaic in entire video, skipping: {input_file}")
    return  # 单文件 → 直接 return；batch 调用方 → continue 下个文件
```

### 3.3 改动 C：Stage 0 大 GOP 兜底（之前欠的）

```python
# utils/keyframe_cutter.py 新增
def ensure_dense_keyframes(src: str | Path, *, gop_sec: float = 2.0,
                           threshold_sec: float = 5.0,
                           log_callback=None) -> Path:
    """若源 GOP > threshold_sec，GPU 重编一遍注入密集 I 帧。
    返回新文件路径（src 改名为 .original_gop.mp4 备份）；GOP 已足够小则原样返回 src。
    """
    keyframes = list_keyframes(src)
    if len(keyframes) < 2:
        return Path(src)
    max_gap = max(b - a for a, b in zip(keyframes, keyframes[1:]))
    if max_gap <= threshold_sec:
        return Path(src)
    # GPU NVENC 重编, -g <fps*gop_sec> 强制密集 GOP
    ...
```

调用时机：Stage 1 之前。仅当 `source_scan_inject_keyframes='auto'/'always'` 时启用。

### 3.4 改动 D：Stage 1 源级扫描（新模块）

```python
# utils/source_time_scanner.py 新建
from dataclasses import dataclass

@dataclass
class TimeInterval:
    start_s: float
    end_s: float
    conf_max: float

def scan_source_time_segments(source_sbs, log_callback=None,
                              cancel_token=None) -> list[TimeInterval]:
    """直接扫源 SBS 找时间区间。复用 mosaic_prescan, 丢 rect。"""
    from utils import mosaic_prescan
    segments = mosaic_prescan.scan_segments(
        source_sbs, log_callback=log_callback, cancel_token=cancel_token
    )
    return [TimeInterval(start_s=s.start_s, end_s=s.end_s, conf_max=s.conf_max)
            for s in segments]
```

实现工作量极小：直接复用现有 `mosaic_prescan.scan_segments` 与 singleton detector。conf 阈值用 0.70 同 pre_extract（已验证）。

### 3.5 改动 E：Stage 2 切源 + Stage 4 concat

#### E1. `utils/keyframe_cutter.cut_source_by_intervals`

```python
@dataclass
class TimelineEntry:
    start_s: float
    end_s: float
    path: Path
    kind: str  # "mosaic" or "gap"

def cut_source_by_intervals(src, intervals, out_dir, keyframes,
                            log_callback=None, process_callback=None
                            ) -> list[TimelineEntry]:
    """把源 SBS 按 mosaic 时间区间切成 mosaic_seg + gap_seg。
    
    所有切割 -c copy. mosaic 区间向外扩到 kf 边界。
    gap 区间 = 视频头 / mosaic 之间 / 视频尾 (扣掉 mosaic 占用后的剩余)。
    返回完整时间轴 list, 按 start_s 排序.
    """
    # 1. 对齐每个 interval 到 [floor_kf(start), ceil_kf(end)]
    # 2. 合并相邻/重叠的对齐后 interval (避免 gap 为负)
    # 3. 枚举 gap: [0, first.start), [prev.end, next.start), [last.end, duration)
    # 4. ffmpeg -ss <a> -to <b> -c copy 切每一段
    # 5. 按时间顺序返回
```

#### E2. `utils/sbs_concat.py` 新建

```python
def concat_timeline(timeline: list[TimelineEntry], output: str | Path,
                    log_callback=None, reencode: str = "auto") -> None:
    """concat 时间轴上所有段到 output。
    
    reencode: "auto" / "never" / "always"
    auto: ffprobe 检查参数一致性, 一致 -c copy, 否则 NVENC 一遍重编
    """
    if reencode == "auto":
        reencode = not _all_params_match([e.path for e in timeline])
    if not reencode:
        # ffmpeg concat demuxer + -c copy (<30s on 8K 30min)
        list_file = _write_concat_list(timeline)
        run(["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-f", "concat", "-safe", "0", "-i", list_file,
             "-c", "copy", "-y", str(output)])
    else:
        # filter concat + NVENC (~3 min on 8K 30min, 兜底)
        _concat_with_reencode(timeline, output, log_callback)


def _all_params_match(paths: list[Path]) -> bool:
    """ffprobe 检查 codec/pix_fmt/profile/width/height/color 是否全一致。"""
    ...
```

### 3.6 改动 F：logic.py 外壳

#### F1. 引入 `_run_source_scan_branch`

```python
def _run_source_scan_branch(input_file, final_output, *,
                            use_fisheye: bool, pre_extract_inner: bool,
                            keep_intermediate: bool,
                            log_callback=None, process_callback=None) -> str:
    """Stage 1-4 外层。返回 PreExtractResult.NO_MOSAIC 时调用方跳过整个文件。"""
    from utils.source_time_scanner import scan_source_time_segments
    from utils.keyframe_cutter import (
        ensure_dense_keyframes, cut_source_by_intervals, list_keyframes
    )
    from utils.sbs_concat import concat_timeline
    
    # Stage 0
    if app_config.get("source_scan_inject_keyframes", "auto") != "never":
        input_file = str(ensure_dense_keyframes(input_file, ...))
    
    # Stage 1
    intervals = scan_source_time_segments(input_file, log_callback, ...)
    if not intervals:
        log_callback(f"[source-scan] no mosaic in entire video, skipping")
        return PreExtractResult.NO_MOSAIC
    
    # Stage 2
    keyframes = list_keyframes(input_file)
    timeline = cut_source_by_intervals(input_file, intervals,
                                       out_dir=..., keyframes=keyframes, ...)
    
    # Stage 3: 对每个 mosaic_seg 跑现有完整管线
    for entry in timeline:
        if entry.kind != "mosaic":
            continue
        restored_path = entry.path.with_suffix(".restored.mp4")
        # 调用 _process_one_input (从 run_single_file_pipeline 抽出来的核心)
        # 内部跑 split + [fisheye] + pre_extract + [defisheye] + merge
        _process_one_input(
            entry.path, restored_path,
            use_fisheye=use_fisheye,
            pre_extract=pre_extract_inner,  # 通常 True
            keep_intermediate=False,
            log_callback=log_callback, process_callback=process_callback,
        )
        entry.path = restored_path  # 替换 timeline 该项指向 restored
    
    # Stage 4
    concat_timeline(timeline, final_output, log_callback, 
                    reencode=app_config.get("source_scan_concat_reencode", "auto"))
    
    # Cleanup
    if not keep_intermediate:
        for entry in timeline:
            try: entry.path.unlink()
            except OSError: pass
    return PreExtractResult.OK
```

#### F2. 重构核心：抽出 `_process_one_input`

把当前 `run_single_file_pipeline` 的 split + pre_extract + merge 核心抽到一个独立函数 `_process_one_input(input, output, use_fisheye, pre_extract, ...)`，让两个调用方共用：
- `run_single_file_pipeline` 当 `source_scan=False` 时直接调
- `_run_source_scan_branch` 在 Stage 3 对每个 mosaic_seg 调

避免递归 + 行为复制。其它 3 个 run_*_pipeline 同理重构。

#### F3. 4 个 run_*_pipeline 入口加 source_scan 分支

```python
def run_single_file_pipeline(input_file, start_time, end_time, use_fisheye,
                             keep_intermediate=False, keep_original_bitrate=False,
                             log_callback=None, process_callback=None,
                             pre_extract=False, source_scan=False):
    ...
    if source_scan and not engine_runner.is_native_engine():
        result = _run_source_scan_branch(
            input_file, file_final,
            use_fisheye=use_fisheye,
            pre_extract_inner=pre_extract,  # 可叠加
            keep_intermediate=keep_intermediate,
            log_callback=log_callback, process_callback=process_callback,
        )
        if result == PreExtractResult.NO_MOSAIC:
            return  # 整片跳过
        if result == PreExtractResult.OK:
            return  # 已完成
        # SCAN_FAILED 才落到下面的正常路径
    
    # 既有正常路径 (或带 pre_extract 单层)
    _process_one_input(input_file, file_final, use_fisheye, pre_extract, ...)
```

Batch 路径同理（NO_MOSAIC → continue 下个文件）。

### 3.7 改动 G：UI & i18n & 配置

#### G1. 4 个 Tab 各加 Checkbutton

```python
# one_click/main.py 在 _grid_pre_extract_check 模式下加一个并列的
self.s_auto_source_scan = tk.BooleanVar(value=False)
self._grid_source_scan_check(tab, self.s_auto_source_scan, row)
```

互锁规则：
- 与 `engine != native_gpu` 互锁（同 pre_extract）
- 与 `opt_fisheye` **完全独立**
- 与 `opt_pre_extract` **可叠加**（推荐都开）

#### G2. i18n

```json
// zh.json
"opt_source_time_filter": "扫描源视频先剔除无马赛克片段（最大化加速；无马赛克视频整片跳过）"
// en.json
"opt_source_time_filter": "Pre-scan source to skip non-mosaic spans (max speed; whole-video skip when empty)"
// ja.json
"opt_source_time_filter": "ソースを事前スキャンしモザイク無し区間をスキップ（最大化速、無モザイク動画は丸ごとスキップ）"
```

#### G3. `utils/app_config.py` 新增

```python
# Stage 1: 源级时间筛
'source_scan_enabled': False,  # 默认关，先验证再推广
'source_scan_inject_keyframes': 'auto',  # auto / never / always
'source_scan_inject_gop_sec': 2.0,

# Stage 3 multi-region (修 bug 的)
'pre_extract_cluster_gap_ratio': 0.03,  # 短边的 3% 作聚类间距

# Stage 4 concat
'source_scan_concat_reencode': 'auto',  # auto / never / always
'source_scan_keep_segments': False,
```

`mosaic_prescan` 与 pre_extract 配置全部沿用，包括 `pre_extract_yolo_conf=0.70`。

---

## 4. 文件命名约定

```
源:                              xxx.mp4
Stage 0 (若注入 kf):              xxx.dense_gop.mp4 (重编, 用作 Stage 1+ 输入)
                                  原 xxx.mp4 备份为 xxx.original_gop.mp4

Stage 2 切割中间文件 (放 _scan_tmp/ 子目录):
  mosaic_seg{NN}.mp4              SBS 全帧, 有马赛克时段
  gap_seg{MM}.mp4                 SBS 全帧, 无马赛克时段
  mosaic_seg{NN}.restored.mp4     Stage 3 处理完
  timeline.json                   timeline 元数据

Stage 3 内层 pre-extract 文件 (在 _scan_tmp/<seg_dir>/ 下):
  与现有 pre-extract 命名约定一致 (BASE.seg{KK}.mp4 / .restored.mp4 / .segments.json)

Stage 4 最终输出:
  xxx_S00..._E00..._sbs.restored.mp4 (与现有命名一致)

调试输出:
  xxx.source_intervals.json       Stage 1 时间区间元数据
  xxx.detections.jsonl            Stage 1 detector 调试 (复用 mosaic_prescan)
```

`source_scan_keep_segments=False`（默认）→ 清掉 `_scan_tmp/`；`timeline.json` 与 `source_intervals.json` 默认保留。

---

## 5. 时间预估（8K SBS 30min @ 25% 马赛克密度）

| 阶段 | 当前 pre-extract | 新架构 |
|---|---|---|
| Stage 0 (kf 注入, 仅大 GOP) | — | 0 或 ~3 min |
| Stage 1 源扫 | — | ~2 min |
| 全长 split+fisheye | 3 min | — |
| Stage 2 -c copy 切 | — | <30s |
| Stage 3 内层 pre_extract (短段, multi-region) | 4 min (双眼, 全长) | ~1.5 min (短段) |
| lada/jasna | 25-35 min | **15-25 min** ⬇ (multi-region 让总 rect 面积更小) |
| 全长 paste | 8-12 min (双眼) | — |
| 全长 Fisheye→VR + merge | 3 min | ~1 min (只 merge mosaic_seg) |
| Stage 4 concat | — | <30s (-c copy) 或 ~3 min (重编兜底) |
| **合计 (典型)** | **~45-60 min** | **~22-35 min** |

**对照场景**：
- 无马赛克视频：~45-60 min → **~2 min**（Stage 1 扫完空 → 整片跳过）
- 高密度（80% 时长）：~45-60 min → ~30-40 min（Stage 0/2/4 收益小, Stage 3 multi-region 仍有效）
- Batch 50 个文件，20 个无马赛克：**节省 ~14 小时**

---

## 6. 风险与缓解

| # | 风险 | 缓解 |
|---|---|---|
| R1 | Stage 1 漏检整段 → 整片跳过, 漏处理 | conf 0.70 已经较严, pad ≥3s; 用户可在 UI 关掉 source_scan, 走纯 pre_extract |
| R2 | Stage 1 与 Stage 3 conf 不一致, Stage 1 报有 Stage 3 没找到 | Stage 3 内层 NO_MOSAIC → 直接 copy 原段做 restored (B3 已处理), 不跑 lada |
| R3 | 大 GOP 源 → kf 对齐 pad 过大, 切的 mosaic_seg 几乎全片 | Stage 0 ensure_dense_keyframes 一次性 ~3 min 重编兜底 |
| R4 | concat seam 编码参数不齐 | reencode='auto' 自动检测, 不齐一遍 NVENC 重编 (~3 min) |
| R5 | multi-region 聚类阈值选不好 → 应分的合, 应合的分 | 配置可调 (`pre_extract_cluster_gap_ratio`); detections.jsonl 已能事后诊断 |
| R6 | NVDEC 并发段 > 4 时 paste 慢 | Phase 1 实测; 超出再分批 (现方案有伏笔) |
| R7 | Stage 3 重构 `_process_one_input` 抽出可能引入回归 | 单独 commit, 每改一个 run_*_pipeline 跑既有完整管线对比 |

---

## 7. 实施任务分解（建议 commit 顺序）

### Phase 1A：独立 bug 修 + 小优化（先 commit, 即可见效）

1. **任务 A — multi-region 修 bug**
   - `mosaic_prescan._spatial_cluster` + `_aggregate_hits` 末尾改
   - `keyframe_cutter.align_segments` 合并条件改"时间 AND rect 重叠"
   - 配置 `pre_extract_cluster_gap_ratio`
   - 自测：拿一段 SBS 多马赛克视频跑现有 pre_extract（无需 source_scan），`segments.json` 应出现多个**同时段不同 rect** 的段；lada 段总输入面积应明显减少

2. **任务 B — PreExtractResult 三态 + skip-on-empty**
   - 引入 `PreExtractResult` 枚举
   - `_run_pre_extract_branch` 返回值改 enum
   - 4 个 Tab 所有 `if not _run_pre_extract_branch(...)` 改判断三态
   - 与 source_scan 解耦：当前 pre_extract 单层时 NO_MOSAIC 仍可选择"跳过 vs 回退全段 lada"（保守默认回退；source_scan=True 时强制跳过）

3. **任务 C — Stage 0 大 GOP 兜底**
   - `keyframe_cutter.ensure_dense_keyframes(src, gop_sec, threshold_sec)`
   - 单独单元测试: 拿一个 GOP=10s 的视频，应被重编;  GOP=2s 的视频应直接原样返回

### Phase 1B：源级筛完整接入

4. **任务 D — Stage 1 源扫**
   - `utils/source_time_scanner.py` 复用 mosaic_prescan

5. **任务 E — Stage 2 + Stage 4**
   - `keyframe_cutter.cut_source_by_intervals`
   - `utils/sbs_concat.py` (含参数一致性检查 + 重编兜底)

6. **任务 F — logic.py 重构与接入**
   - F1: 抽 `_process_one_input` 核心函数（4 个 run_*_pipeline 共用）
   - F2: `_run_source_scan_branch` 外壳
   - F3: 4 个 run_*_pipeline 入口加 source_scan 分支
   - 每改完一条管线跑一遍既有 pre_extract 回归对比

7. **任务 G — UI / i18n / 配置**
   - 4 Tab 加 `opt_source_time_filter` Checkbutton（紧邻 `opt_pre_extract` 下方）
   - i18n 中/英/日三语
   - `app_config` 新配置项

### Phase 1C：实测与上线

8. **任务 H — 端到端实测**
   - 矩阵：8K SBS × {30min} × {0% / 25% / 60% 马赛克密度} × {single eye / fisheye / batch} × {pre_extract on/off + source_scan on/off}
   - 记录：wall time / segment 数 / concat seam 检查（PSNR > 50dB）/ 输出体积
   - 写实测报告 `summary/summary_..._SOURCE_TIME_FILTER_BENCH_CN.md`

### 提交建议

- 任务 A / B / C 单独 commit，互相独立可验证，**先合可立即拿到 multi-region 加速 + 无马赛克跳过收益**（不依赖 source_scan 完整实现）
- D / E 单独 commit
- F1（重构）单独 commit, 配合回归测试
- F2 / F3 一起 commit
- G 一个 commit
- H 实测报告作为收尾 commit

---

## 8. 与上一版 pre-extract 方案的关系

- **共存不替换**：现有 pre-extract 仍然有效，本期新增的 source_scan 是**外壳**，与 pre_extract 可任意组合
- 现有 pre-extract 的 multi-region bug 由本期任务 A **顺带修复**
- 现有 pre-extract 的 SCAN_FAILED / NO_MOSAIC 三态由本期任务 B **顺带规范**
- 大 GOP 兜底（之前方案欠的）由本期任务 C **统一还上**

四种组合矩阵：

| `source_scan` | `pre_extract` | 行为 | 适用 |
|---|---|---|---|
| ✗ | ✗ | 当前全片 lada/jasna 管线 | 极短视频 / 调试 |
| ✗ | ✓ | 当前 pre_extract (现状, 本期带 multi-region 修复) | 默认推荐, 兼容好 |
| ✓ | ✗ | 源级时间筛后整段送 lada/jasna | 实现简单, lada 段是全帧 |
| ✓ | ✓ | **双层筛, 最大加速** | 8K + 低密度 (<30%) + batch 场景, 推荐 |

---

## 9. 备注

- 本方案完全在 lada/jasna CLI 引擎下落地，**Phase 1 不动 native_gpu 路径**。native_gpu Phase 2 再考虑（多半架构可大段复用, multi-region 可直接 port）。
- HDR / 10-bit 路径走老流程（现有 pre_extract 早已有同样限制）。Stage 1 source_scan 在 HDR 源上 conf 表现未验证, 默认对 HDR 不启用 source_scan（在 `_run_source_scan_branch` 顶部检测 meta.is_hdr 直接 fallback）。
- AGPL 等许可证状态不变（沿用 vendored lada 的约束）。
