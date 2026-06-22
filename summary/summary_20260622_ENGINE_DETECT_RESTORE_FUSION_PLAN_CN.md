# Track B：内置 GPU 引擎内 detector+restoration 全融合（单次检测 + 时间门控）

日期：2026-06-22
范围：**仅内置 GPU 引擎 `gpu_engine/native_mosaic`**，不改 one_click 编排（后续它只需改调新入口）。
目标：消除 pre-extract 的视频处理中间文件，让检测与修复在引擎内一趟（逻辑上）完成。

## 1. 现状与中间文件成本

两条路径：
- **`restore_file`（已融合、无中间文件）**：NVDEC→YOLO 检测→256 clip→BasicVSR++→blend→NVENC，单输出。检测与修复在 `FrameRestorer` 内耦合，clean 帧透传。**但解码全片**。
- **pre-extract（one_click，中间文件来源）**：独立 prescan(`scan_segments`)→`segments.json`/`detections.jsonl`；每段 cut `seg.mp4`→修复 `seg.restored.mp4`（内部又跑一遍 YOLO）→`segment_paster` 贴回再编码。

成本：检测**算两遍**（prescan 用 ultralytics 直调，引擎用 `Yolo11SegmentationModel`，两套路径）；每个马赛克区 **3+ 代编解码**；大量 seg/restored/json 落盘。

## 2. 对接面（已确认）

- 引擎 `mosaic_detector.py`：`Scene`(file_path, frames/masks/boxes, frame_start/end) 时序跟踪；`Clip` 携带 frame_start/end + 裁剪框 + masks。
- prescan `mosaic_prescan.py`：`MosaicSegment`(seg_id, start_s/end_s, start_s_kf/end_s_kf, x/y/w/h, conf_max)；含 `_expanded_rect`(rect 扩张/对齐) 与 `_merge_overlapping_segments`(时空重叠合并) —— **可复用**。
- 结论：引擎 Scene 聚合即可生成 MosaicSegment，无需第二套检测。

## 3. 全融合架构（单次检测 + 时间门控）

两趟、皆在引擎内、全程显存、零文件：

**Pass 1 — 单次检测定位（取代 prescan + 第二遍检测）**
- 用引擎已加载的检测模型，低成本扫描全片（可低分辨率/抽帧加速），Scene 跟踪 →聚合为内存 `MosaicSegment` 列表（复用 `_expanded_rect`/`_merge_overlapping_segments`）。
- **关键：同时缓存每个马赛克帧的 mask+box（紧凑形式：cropped 256 mask 或 RLE/bbox，必要时 vram_offload）**，供 Pass 2 直接复用 → 实现"真正单次检测"，Pass 2 不再跑 YOLO。
- 产出：内存 segments（供 UI/QA，不落盘）+ 每段所需 mask/box。

**Pass 2 — 时间门控修复 + 透传**
- 只对马赛克时间段 NVDEC seek 解码 → 用 Pass 1 的 mask/box 裁剪 256 clip → BasicVSR++（A1 的 native DCN + CUDA Graph）→ blend → NVENC。
- clean 段走 stream-copy/passthrough（参考现有 `segment_paster` 的 passthrough 思路，但无文件、引擎内直接拼流）。
- 单输出，NVENC IDR/关键帧对齐处理接缝。

## 4. 分阶段实施（每阶段独立可测）

- **Stage 1（基础，低风险）**：引擎内 `detect_segments(input)->list[MosaicSegment]`，用引擎检测模型一趟产出内存 segments；与现有 `scan_segments` JSON 做等价性对拍（时间段/rect 容差内一致）。**统一到一套检测器**。
- **Stage 2**：Pass 1 扩展为同时缓存 mask/box；新增门控修复入口 `restore_file_fused(input, output, *, return_segments=False)`，对马赛克段修复、clean 段 passthrough，单输出。先用 restore_file 全片结果做质量对拍（PSNR）。
- **Stage 3**：NVENC 接缝/关键帧对齐打磨；4K/8K 验证；mask 内存/offload 调优；one_click 切换到新入口（独立、引擎外）。

## 5. 正确性/风险

- **关键帧对齐**：Pass 2 的修复段与 passthrough 段边界须落在 IDR/关键帧，否则接缝或重编码。复用 `keyframe_cutter` 逻辑但无文件版。
- **mask 复用 vs 重检测**：若 mask 缓存代价过大，可退化为 Pass 2 仅在马赛克段重跑 YOLO（检测≈full-low-res + ranges，仍远小于两遍全片）。先按 mask 复用做，过大再退化。
- **时序一致性**：BasicVSR++ 双向递归对段边界敏感；门控分段须保证每个马赛克 Scene 完整落在一个修复段内（不可从中间切断），否则修复质量降。
- 不引入任何 `.engine`/Triton 依赖；修复仍走 A1（native DCN + CUDA Graph）。

## 6. 关联
- 加速地基见 `summary_20260622_RESTORATION_CUDA_GRAPH_ACCEL_CN.md`（A1）。
- 内存：项目记忆 `restoration-accel-plan.md`、`gpu-engine-architecture.md`。
