# 源级筛 V3 — Stage 4 回退 concat、Stage 3 启用 native_gpu

- 日期：2026-05-31
- 触发：savr00792_2_8k_1 实测 1h37m，profile 显示 Stage 3 76% / Stage 4 22%
- 修正：V2 方案中 `replace_segments_gpu` 设计错误（结构性硬上限 = NVENC 8K 单卡吞吐），导致 Stage 4 21m
- 状态：覆盖 [summary_20260531_SOURCE_SCAN_REFACTOR_V2_CN.md](summary_20260531_SOURCE_SCAN_REFACTOR_V2_CN.md) 中 Stage 4 部分，其余（删 Stage 0、源级粗合并独立参数）保留

---

## 0. 实测瓶颈数据（savr00792_2_8k_1, 13min 8K SBS, 总耗时 1h37m06s）

| 阶段 | 耗时 | 占比 |
|---|---|---|
| Stage 1 源扫 | 1m14s | 1.3% |
| Stage 2 ffmpeg -c copy 切 | <1s | ~0% |
| Stage 3 mosaic_seg000（149s 内容）| 31m03s | 32% |
| Stage 3 mosaic_seg001（175s 内容）| 43m13s | 45% |
| Stage 4 GPU replace 全片重编 | **21m02s** | **22%** |

**Stage 3 子分布（典型）**：split+fisheye 4m + 扫描 2m + lada 子调用 N×（cut + lada-cli + paste）+ merge 4m，其中 lada-cli 调用是变量主导。mosaic_seg001 内层 pre_extract 出"1 段 100%"（rect=1776×1264，整段 175s 送 lada），左眼 lada 9m26s + 右眼 9m03s = 18m29s 仅这一段 lada。

---

## 1. V2 Stage 4 设计错误的复盘

V2 提出的 `replace_segments_gpu` 让源 SBS 走 NVDEC → 逐帧判断（mosaic 段读 restored_seg；gap 段读源）→ NVENC 编码 → mux。

**结构性问题**：NVENC 8K 单卡吞吐 = 37.2 fps（与本机算力绑死，无优化空间）。对 46756 帧硬性付 21 分钟。但实际只有 41.6% 帧需要替换，**剩下 58.4% 的 gap 帧被白白重编**——这是设计层面的浪费，不是实现质量问题。

**根因**：V2 误把"不要 ffmpeg 来处理"读成"任何 ffmpeg 子进程都不行"，所以舍弃了 V1 的 `ffmpeg concat demuxer + -c copy`。但：
- `ffmpeg concat -c copy` 与 `gpu_engine/mux.py` 已经用的 `ffmpeg -c copy` mux 同类，本质都是 **bitstream 拷贝 + container 索引写入**，不"处理"任何像素
- 这类操作是 GPU 流水线必备的工具组件，不应在禁用范围内

---

## 2. P0 — Stage 4 回退 ffmpeg concat demuxer（必做，省 ~20 分钟）

### 2.1 目标

```
21m02s (V2 整片 NVENC 重编) → ~30s (concat demuxer -c copy, 纯 IO + container mux)
```

### 2.2 实现要点

**复活 `utils/sbs_concat.py`**（V1 dev 写过的代码可直接复用）：

```python
def concat_timeline(timeline: list[TimelineEntry], output: str | Path,
                    log_callback=None, process_callback=None,
                    reencode: str = "auto") -> None:
    """timeline 已按 start_s 排序, 包含 mosaic.restored 与 gap_seg 两类.
    
    reencode='auto': 所有段 codec/pix_fmt/profile/分辨率/色彩 一致 → -c copy
                     不一致 → NVENC 一遍重编兜底
    """
    paths = [Path(e.path) for e in sorted(timeline, key=lambda x: x.start_s)]
    mode = "never" if (reencode == "auto" and _all_params_match(paths)) else \
           ("never" if reencode == "never" else "always")
    
    list_file = _write_concat_list(paths)
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-stats", "-y",
           "-f", "concat", "-safe", "0", "-i", str(list_file)]
    if mode == "never":
        cmd += ["-c", "copy", "-movflags", "+faststart", str(output)]
    else:
        cmd += [...NVENC reencode 参数, 沿用 V1...]
    _run(cmd, ...)
```

**配合 `utils/keyframe_cutter.py` 的 `cut_source_by_intervals`**：
- gap 必须**物化**为独立的 `gap_seg{NN}.mp4`（`-ss <kf> -t <dur> -c copy`），不再像 V2 当前那样 path 指向源
- gap 切割时延暂时多占用 ~3-4GB 临时磁盘，处理完清掉
- 与 mosaic_seg 切割是同一个 ffmpeg `-c copy` 模板

**`utils/sbs_concat.py` 兼容性自检** `_all_params_match`：probe.probe_video 比较 (codec, profile, pix_fmt, width, height, bit_depth, color_*) 元组。我们 GPU NVENC + lada-cli/native_gpu 都用 hevc + bt709 + 8-bit，预期都一致 → 走 `-c copy` 分支。

### 2.3 删除

- `gpu_engine/files.py` 的 `replace_segments_gpu`（V2 产物，确认无其它调用方）
- `one_click/logic.py` `_run_source_scan_branch` 中调 `replace_segments_gpu` 的整块

### 2.4 风险

| 风险 | 缓解 |
|---|---|
| 段间 IDR 不齐 → concat 解码异常 | mosaic_seg.restored 由 NVENC `force_idr=True` 起头（已保证）；gap_seg 由 `-ss <kf> -c copy` 切，起头就是 IDR；理论上一定齐 |
| 不同段 SPS/PPS 微差 → 播放器兼容性问题 | `-c copy` 输出含两组 SPS/PPS 是 HEVC 标准允许；主流播放器（VLC/MPV/headset）均兼容；测试覆盖即可 |
| 极个别参数实际不齐 | reencode='auto' 自动降级 NVENC 一遍重编（~3 min 8K 30min，可接受）|

### 2.5 验收

- 8K SBS 13min 实测 Stage 4 < 60s（含两次 mux 与 list 文件写入）
- 输出 mp4 与现有 V2 输出做 PSNR 对比 > 50dB（mosaic 段 100% 相同, gap 段 bitstream 等价）
- 各播放器抽帧检查段交界帧无马赛克残留 / 无解码错位

---

## 3. P1 — 内层 pre_extract 时间合并阈值复测（条件做，省 0–15 分钟）

### 3.1 触发条件

仅在 mosaic_seg001 类型情况（内层 pre_extract 输出 `aggregated 1 segments, 100%`）下尝试。从 detections.jsonl 验证：

```bash
# 看 mosaic_seg001_L_fisheye.detections.jsonl 中 accepted=true 的帧分布
# 若帧序列基本连续 → 内容真的连续, P1 无效
# 若有显著时间间隙（≥0.5s 多次）→ 内层 merge_gap_s 把它们合一了, P1 可生效
```

### 3.2 配置调整（仅当可生效）

```python
# utils/app_config.py
'pre_extract_merge_gap_s': 0.5,   # 1.5 → 0.5, 让间断的连续段拆开
# 不动 min_segment_s（保持 1.5s 下限，防短段碎片）
```

### 3.3 预期

- 若 mosaic 间断 → seg001 拆成 3-5 个内层段, rect 各自更紧, 可能省 5-15m
- 若 mosaic 连续 → 配置改了也只出 1 段, 不影响

低风险参数调整，A/B 跑同一视频对比即可。

---

## 4. P3 — 启用 native_gpu 引擎进入 source_scan 流程（必做，省 ~10-15 分钟）

### 4.1 当前状态

`one_click/logic.py`：

```python
def _pre_extract_supported(pre_extract, log_callback=None) -> bool:
    if not pre_extract:
        return False
    if engine_runner.is_native_engine():       # ← Phase 1 锁
        log_callback("[pre-extract] native_gpu is not supported in Phase 1; ...")
        return False
    return True

def _source_scan_supported(source_scan, log_callback=None) -> bool:
    if not source_scan:
        return False
    if engine_runner.is_native_engine():       # ← Phase 1 锁
        log_callback("[source-scan] native_gpu is not supported in Phase 1; ...")
        return False
    return True
```

这两道闸是 pre_extract 方案首期写下的保守约束。当时考虑 native_gpu 流水线复杂、与 pre_extract 切片逻辑未验证，先用 lada-cli 跑通。**现在内层 pre_extract 已经稳定，可以放开**。

### 4.2 关键观察

`one_click/logic.py:process_lada` 早已支持 native_gpu 路由：

```python
def process_lada(input_file, output_file, log_callback=None, process_callback=None):
    if engine_runner.is_native_engine():
        from gpu_engine import native_mosaic
        ok = native_mosaic.restore_file(input_file, output_file, ...)
        ...
        return
    # 否则 lada/jasna CLI
    ...
```

只要解除上层 `_pre_extract_supported` / `_source_scan_supported` 的 native_gpu 闸，内层 N 次 `process_lada` 调用就自动改走 `native_mosaic.restore_file`（单例模型，零启动开销）。

### 4.3 预期收益（基于 [[gpu-engine-architecture]] 记录的实测）

- 全单眼 8K fisheye 管线：lada-cli **71.6s** → native_gpu **34.7s（2.06×）**
- 单次 lada 子调用启动开销：CLI **~6-7s** → native_gpu **0**（模型常驻）

应用到本次实测：

| 项 | lada-cli (当前) | native_gpu (P3) | 省 |
|---|---|---|---|
| mosaic_seg001 单段 L lada (rect 1776×1264, 175s) | 9m26s | ~4m43s | 4m43s |
| mosaic_seg001 单段 R lada (rect 1200×2832, 175s) | 9m03s | ~4m32s | 4m31s |
| mosaic_seg000 L 共 5 段（含 5×7s 启动）| ~10m + 35s 启动 | ~5m + 0 启动 | ~5m35s |
| mosaic_seg000 R 共 N 段 | 类似 | 类似 | 类似 |
| **小计** | | | **~15-18m** |

### 4.4 实施

**最小改动版**：直接删两条 native_gpu 守卫即可：

```diff
 def _pre_extract_supported(pre_extract, log_callback=None) -> bool:
     if not pre_extract:
         return False
-    if engine_runner.is_native_engine():
-        if log_callback:
-            log_callback("[pre-extract] native_gpu is not supported in Phase 1; ...")
-        return False
     return True

 def _source_scan_supported(source_scan, log_callback=None) -> bool:
     if not source_scan:
         return False
-    if engine_runner.is_native_engine():
-        if log_callback:
-            log_callback("[source-scan] native_gpu is not supported in Phase 1; ...")
-        return False
     return True
```

**UI 互锁同步放开**：`one_click/main.py` 里 `_grid_pre_extract_check` / source_scan 类似函数中"engine=='native_gpu' 时禁用"逻辑同时去除。

### 4.5 与现有 native_stream 路径的关系

`one_click/logic.py` 已有 `_run_native_sbs_stream` / `_run_native_single_eye_stream` 的快速路径（M1 流式融合管线）。当前流程顺序：

```
source_scan_enabled? → 走 source_scan (Stage 1-4)
   ↓ no
native_stream_allowed?  → 走融合流式
   ↓ no
fall through 标准管线
```

**不需要改顺序**：source_scan 在最外层判断。当 source_scan + native_gpu 同时启用时，外层走 source_scan，内层每段 mosaic_seg 走 split + pre_extract（多次 native_mosaic.restore_file 调用）+ merge。**不会**也不应该掉进 native_stream（它是另一条无 pre_extract 的快速路径）。

### 4.6 风险

| 风险 | 缓解 |
|---|---|
| native_mosaic 与 pre_extract 同时持有 YOLO 检测器实例 → VRAM 占用 ↑（2 个检测器各 ~200MB）| RTX 5060 Ti 16GB 远超需求，实测不会瓶颈；长期可让两者共享单例 |
| native_mosaic 内部 NVENC + 外层 pre_extract paste_segments_gpu NVENC 同时活跃 | PyNv NVENC 支持多 session；现有 native_gpu 单文件管线就有 2-3 个 NVENC session 同时活跃，已验证 |
| native_mosaic 在 rect-cropped 输入上未验证 | restore_file 接受任意 mp4，内部 YOLO 自适应 imgsz；rect crop 仅改变输入分辨率，无算法变化；首次跑实测验证即可 |
| 小 rect (<256px) 上 BasicVSR++ 恢复质量下降 | pre_extract 已有 `pre_extract_rect_min_px=512` 兜底，rect 总会被 padding 到 ≥512 |

### 4.7 验收

- 8K SBS 13min 用例 Stage 3 总耗时 < 50m（vs 当前 74m16s）
- 输出与 lada-cli 版本视觉对比无可感差异（lada 与 native_mosaic 跑的是同一个 v2 检测 + 同一恢复模型，结果应该几乎相同）
- VRAM 峰值 < 12GB
- UI 选择"内置(GPU)"+ source_scan 不再被禁用

---

## 5. 综合时间预估（叠加 P0 + P1 + P3）

基于本次 savr00792_2_8k_1 实测（1h37m06s）：

| 项 | 现状 | P0 (Stage 4 concat) | + P1 (若适用) | + P3 (native_gpu) |
|---|---|---|---|---|
| Stage 1+2 | 1m14s | 同 | 同 | 同 |
| Stage 3 seg000 | 31m03s | 同 | 略降 | ~18-22m ↓ |
| Stage 3 seg001 | 43m13s | 同 | ~28-35m ↓ (条件) | **~22-28m** ↓↓ |
| Stage 4 | 21m02s | **~30-60s** ↓↓↓ | 同 | 同 |
| **合计** | **1h37m06s** | **~1h17m** | **~1h05m** | **~50m–1h** |

**节省**：P0 单做 ~20%，P0+P3 ~38-45%，再加 P1（如适用）~48-50%。

---

## 6. 实施顺序与 commit 建议

1. **任务 A（P0 必做）**：
   - 复活 `utils/sbs_concat.py`（恢复 V1 dev 写过的实现）
   - `utils/keyframe_cutter.cut_source_by_intervals` 让 gap 也物化为 -c copy 切片（不再 path 指向源）
   - `one_click/logic.py` `_run_source_scan_branch` 中调 `replace_segments_gpu` 改为调 `concat_timeline`
   - 删 `gpu_engine/files.replace_segments_gpu`
   - 实测 savr00792_2_8k_1，确认 Stage 4 < 60s

2. **任务 B（P3 必做）**：
   - 删 `_pre_extract_supported` / `_source_scan_supported` 中两条 native_gpu 守卫
   - `one_click/main.py` UI 互锁中对应放开
   - 实测 savr00792_2_8k_1 + 引擎切到"内置(GPU)"，对比时间与画质

3. **任务 C（P1 条件做）**：
   - 先看 mosaic_seg001 detections.jsonl 帧分布
   - 若有间隙 → 改 `pre_extract_merge_gap_s` 0.5s 试一次
   - 若无收益或质量下降 → 还原

4. **任务 D（实测验证 + summary）**：
   - 矩阵：本次素材 + 1-2 个不同密度素材 × {lada-cli / native_gpu} × {source_scan on/off}
   - 记录 wall time / VRAM / 输出 PSNR
   - 写 `summary_2026XXXX_SOURCE_SCAN_V3_BENCH_CN.md`

任务 A 与任务 B **互相独立**，可并行 commit。先合任一即可拿到对应收益。

---

## 7. 哪些 V2 决策保留

- ✓ 删 Stage 0 `ensure_dense_keyframes`（实测 Stage 2 -c copy 切只用 0.3s, dense kf 注入完全没必要）
- ✓ 源级粗合并参数独立（`source_scan_merge_gap_s` / `source_scan_min_segment_s` / `source_scan_head_tail_pad_s`）
- ✓ Multi-region 空间聚类（mosaic_seg000 实测产 5 个不同 rect, 起作用了）
- ✓ PreExtractResult 三态 + skip-on-empty

## 8. 哪些 V2 决策推翻

- ✗ ~~Stage 4 `replace_segments_gpu` 单遍整片~~ → **回退 V1 的 ffmpeg concat demuxer -c copy**
- ✗ ~~native_gpu Phase 1 不支持~~ → **本期同时启用**
- ✗ ~~内层 pre_extract 在 hequirect 跑（P2 设想）~~ → 撤销，用户澄清"模型在源 SBS 能跑"是指 Stage 1 源级扫描，内层保持 fisheye 不变

---

## 9. 备注

- "不要再用 ffmpeg 来处理"的合理边界：**不要让 ffmpeg 做需要解码/编码像素的重处理**（NVENC 全片重编、复杂滤镜图）。**允许**：bitstream `-c copy` 切割、concat demuxer 拼接、mux 封容器——这些与 `gpu_engine/mux.py` 现有用法同类，是流水线的"打包工具"。
- native_gpu 启用后，pre_extract 内层每个 rect 的 process_lada 走 `native_mosaic.restore_file`，**单例模型零启动**，是 P3 收益的根源。
- P0 + P3 是**结构性修复**（V2 设计错误 + V1 保守约束），P1 是参数调优。前两者强烈建议先合，P1 看条件再做。
