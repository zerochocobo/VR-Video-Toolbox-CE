# Changelog

## English

### 2026-06-14

- New: Added Clone Translation Dubbing documentation and workflow notes, covering per-speaker voice cloning, SI remix output `_SI.mp4`, and dubbing output `_DUB.mp4`.
- Major update: Improved clonevoice loudness matching by aligning synthesized speech to the source sentence RMS, with bounded gain and a UI toggle.
- Major update: Completed dubbing remix behavior: SI-only controls are hidden in dubbing mode, ducking is disabled, and DUB audio can optionally be added as an independent track.
- Fix: Batch scans now ignore generated `_SI.mp4` and `_DUB.mp4` outputs so they are not reprocessed as source videos.

### 2026-06-05

- New: Added the grouped OneClick paired pre-extract pipeline. Rects sharing the same frame window can now share one GPU decode, and restored rects can use raw HEVC plus sidecar metadata instead of temporary MP4 muxes.
- Major optimization: Added group-level extract/restore pipelining and HEVC passthrough for inactive paste gaps, reducing idle time, temporary storage, and unnecessary re-encoding.
- Major fix: Fixed paired fine matching that could send detected mosaics into passthrough. Non-overlapping windows can reuse the same raw segment, high-confidence unmatched one-eye segments are retained with `pre_extract_pair_keep_unmatched_conf=0.60`, and adjacent same-rect outputs are merged.
- Major fix: Added raw HEVC paste fallback and stronger pipeline error propagation. Failed direct raw paste now retries through a temporary MP4 wrapper before falling back, and producer errors are no longer reported as user cancellation.
- Optimization: Added an empty-result cache for fine scans so repeated no-hit scans can skip detector work.

### 2026-06-04

- Major optimization: Added GPU keyframe coarse scanning for OneClick source-scan, detector batch OOM split/retry, and fast HEVC concat/Annex-B merge mode for source-scan outputs.
- Major optimization: Reworked the OneClick bitrate contract. Intermediate outputs keep quality headroom, final outputs converge to source bitrate, the 8K baseline floor was corrected, and final MP4 bitrate self-check logs oversize warnings.
- Major fix: Improved cancellation and progress behavior. User cancellation now propagates cleanly instead of falling through into expensive fallback paths.
- Major fix: Improved the 2DVR E2FGVI path with black-line cleanup, mask-aware resize, FP16/GPU compositing, and OOM fallback.

### 2026-06-03

- New: Added the 2D to Depth VR tool using local Depth Anything 3 Small, with VR180 SBS output, fisheye/equirectangular projection, start/end trimming, eye-distance control, launcher integration, and i18n.
- Major fix: Repaired Kotoba Whisper CTranslate2 alignment-head configs and safely re-enabled Kotoba word timestamps.
- Major fix: Tuned OneClick source-scan recall by scanning the source left eye at native size, fixing detector cache keys, and aligning detection metadata/debug coordinates.

### 2026-06-01

- New: Added the OneClick source-scan and pre-extract workflow to process only detected mosaic intervals and paste restored rects back onto the base video.
- Major optimization: Added fast HEVC timeline concat for source-scan outputs, with original audio copied once and GPU timeline merge kept as a fallback.
- Major fix: Corrected source-scan final merge playback/seeking issues and final GPU timeline bitrate overshoot.

### 2026-05-29 to 2026-05-30

- New: Added the `gpu_engine` backend with PyNvVideoCodec/CuPy GPU decode, transform, encode, and automatic ffmpeg fallback for major VR geometry workflows.
- New: Added the built-in `native_gpu` mosaic engine, integrating vendored LADA in-process with CUDA dependency checks and UI engine selection.
- Major optimization: GPU-accelerated VR split/merge, fisheye/equirectangular conversion, VR-to-flat projection, and OneClick geometry stages.

## 中文

### 2026-06-14

- 新功能：补充克隆翻译配音文档与流程说明，覆盖按说话人音色克隆、同声传译输出 `_SI.mp4`、配音输出 `_DUB.mp4`。
- 重大更新：改进 clonevoice 响度匹配，合成语音会按原句 RMS 对齐，并加入增益保护和界面开关。
- 重大更新：完善配音回混行为：配音模式隐藏 SI 专用控件、禁用 ducking，并支持将 DUB 音频作为独立音轨加入。
- 修复：批量扫描会忽略已生成的 `_SI.mp4` / `_DUB.mp4`，避免把输出文件再次当作源视频处理。

### 2026-06-05

- 新功能：新增 OneClick paired pre-extract 分组流水线。同一帧窗口内的多个 rect 可共享一次 GPU 解码，恢复小段可使用 raw HEVC + sidecar 元数据，减少临时 MP4 mux。
- 重大优化：新增 group 级 extract/restore 流水线，以及 HEVC inactive gap passthrough，减少等待时间、临时空间占用和不必要的重编码。
- 重大修复：修复 paired fine 匹配把已检测到的马赛克误送入 passthrough 的问题。同一 raw segment 可在不重叠时间窗复用，高置信单眼 unmatched 段通过 `pre_extract_pair_keep_unmatched_conf=0.60` 保留，同侧相邻同 rect 输出会合并。
- 重大修复：增强 raw HEVC paste 兜底和 pipeline 异常传播。直接消费 raw HEVC 失败时会先临时包装成 MP4 重试，producer 普通异常不再被误报成用户取消。
- 优化：新增 fine scan 空结果缓存，重复无命中扫描可跳过 detector。

### 2026-06-04

- 重大优化：OneClick source-scan 新增 GPU keyframe 粗扫、detector batch OOM 自动拆批重试，以及 source-scan 输出的 HEVC Annex-B 快速合并模式。
- 重大优化：重构 OneClick 码率契约。中间产物保留画质裕度，最终输出默认收敛到源码率，修正 8K baseline 下限，并新增最终 MP4 平均码率自检和超限 warning。
- 重大修复：改进取消和进度行为。用户取消会干净向上传播，不再继续进入耗时 fallback。
- 重大修复：改进 2DVR E2FGVI 路径，加入黑线清理、mask-aware resize、FP16/GPU 合成和 OOM 兜底。

### 2026-06-03

- 新功能：新增 2D 转深度 VR 工具，使用本地 Depth Anything 3 Small，支持 VR180 SBS 输出、鱼眼/等距投影、起止时间、眼距控制、主界面入口和多语言文案。
- 重大修复：修复 Kotoba Whisper CTranslate2 alignment-head 配置，安全恢复 Kotoba word timestamps。
- 重大修复：调优 OneClick source-scan 召回率：改为原始尺寸左眼粗扫，修复 detector cache key，并对齐检测 metadata/debug 坐标。

### 2026-06-01

- 新功能：新增 OneClick source-scan / pre-extract 工作流，只处理检测到马赛克的时间区间，并把恢复后的 rect 贴回底片视频。
- 重大优化：新增 source-scan 输出的 HEVC timeline 快速拼接，最终音频只从原始源复制一次，并保留 GPU timeline merge 作为兜底。
- 重大修复：修复 source-scan 最终合并后的播放/seek 问题，以及最终 GPU timeline merge 的码率膨胀问题。

### 2026-05-29 至 2026-05-30

- 新功能：新增 `gpu_engine` 后端，基于 PyNvVideoCodec/CuPy 实现 GPU 解码、几何变换、编码，并为主要 VR 几何流程保留 ffmpeg 自动兜底。
- 新功能：新增内置 `native_gpu` 去马赛克引擎，进程内集成 vendored LADA，支持 CUDA 依赖检查和 UI 引擎选择。
- 重大优化：VR 分割/合并、鱼眼/等距转换、VR 转平面和 OneClick 几何阶段接入 GPU 加速。
