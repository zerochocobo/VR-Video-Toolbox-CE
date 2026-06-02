# Source Scan 成对细分 Rect 优化计划

- 日期：2026-06-01
- 背景：source-scan 阈值降到 0.5 后能提升召回，但会带来单眼误判。用户要求 coarse 可以低阈值，fine 阶段提高阈值，并且左右眼 fine 判断不一致的时间片段直接放弃。

## 1. 最终目标

source-scan 分两层：

```text
Stage 1 coarse source scan:
  源 SBS 低阈值检测，只决定哪些大时间区间可能有马赛克

Stage 2 interval materialization:
  source.mp4 -> mosaic_segNNN.mp4

Stage 3 paired fine rect:
  在 mosaic_segNNN.mp4 上分别扫描左/右眼
  fine 阈值提高到 0.6
  只有左右眼同一时间组都检测到时才保留
  直接输出 L/R 的 rect seg 文件
  lada/jasna/native_gpu 处理 rect seg
  GPU 贴回到 interval 基底，生成 mosaic_segNNN.restored.mp4

Stage 4 concat:
  gap 仍用源视频 inpoint/outpoint 虚拟引用
  mosaic 用 mosaic_segNNN.restored.mp4
  video-only concat
  最后从源视频统一 mux 音频
```

## 2. 阈值策略

- `pre_extract_yolo_conf = 0.50`：作为 detector 构建阈值，保证 source-scan coarse 召回。
- `pre_extract_fine_yolo_conf = 0.60`：fine 阶段额外后处理过滤，低于 0.6 的 box 标记为 `low_conf` 并丢弃。

这样 detector 不需要在 source/fine 之间反复重建，fine 仍能提高有效阈值。

## 3. UI 分支匹配

### SBS + 不转鱼眼

```text
mosaic_seg000.mp4
  -> GPU scan left/right
  -> mosaic_seg000_L.seg000.mp4 / mosaic_seg000_R.seg000.mp4 ...
  -> restored seg
  -> GPU paste 到 mosaic_seg000.mp4
  -> mosaic_seg000.restored.mp4
```

### SBS + 转鱼眼

```text
mosaic_seg000.mp4
  -> GPU scan left/right after heq->fisheye
  -> mosaic_seg000_L_fisheye.seg000.mp4 / mosaic_seg000_R_fisheye.seg000.mp4 ...
  -> restored seg
  -> GPU 内存中逐帧 heq->fisheye
  -> GPU 内存中 paste restored fisheye rect
  -> GPU 内存中 fisheye->hequirect
  -> mosaic_seg000.restored.mp4
```

说明：fisheye rect 不能几何正确地直接贴到 hequirect 基底，但不需要落盘整段 fisheye SBS 基底。现在由 `gpu_engine.files.paste_fisheye_eye_rects_to_sbs_gpu()` 在 GPU 内存中完成投影、贴回、反投影和编码。

### Single-eye

本轮保持现有单眼路径。single-eye 没有左右眼一致性校验对象，仍按 fine 阈值跑 pre-extract。

## 4. 一致性规则

fine 阶段把左右眼检测结果按时间合组：

- 同一时间组左右眼都至少有一个 detection：保留该时间组。
- 只有左眼或只有右眼检测到：跳过该时间组，不切 rect，不送 lada/jasna。
- 保留组内左右眼各自 rect，不要求空间位置一致，因为左右眼视差会导致坐标不同。

## 5. 文件产物

必要产物：

- `mosaic_segNNN.mp4`
- `mosaic_segNNN_L[_fisheye].segMMM.mp4`
- `mosaic_segNNN_R[_fisheye].segMMM.mp4`
- 对应 `.restored.mp4`
- `mosaic_segNNN.restored.mp4`

鱼眼分支临时产物：

- 无整段 fisheye SBS 临时文件。只保留 rect seg 和对应 restored seg。

`keep_intermediate=False` 时清理 fine rect；source-scan 的 tmp 目录仍按现有 `source_scan_keep_segments` 控制。

## 6. 验收点

- coarse scan 仍使用低阈值召回。
- fine scan 日志显示 `fine_conf=0.60`。
- 左右眼不一致时间组被日志标记并跳过。
- SBS + 鱼眼/非鱼眼均走 paired fine path。
- Stage 4 仍然是 video-only concat + source audio mux。
- 单元测试覆盖 fine conf 过滤、左右眼配对跳过、source-scan SBS paired path。
