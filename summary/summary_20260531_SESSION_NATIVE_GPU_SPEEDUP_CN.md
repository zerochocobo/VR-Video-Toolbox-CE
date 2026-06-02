# 会话总结：内置(GPU)去马赛克提速（方案一落地 / 方案三验证否决）

- 日期：2026-05-31
- 范围：`native_gpu`（内置GPU）引擎的去马赛克性能优化
- 配套文档：`summary/summary_20260531_NATIVE_BOTTLENECK_PROFILE_CN.md`（详细 profile 数据）、`summary/summary_20260531_RAWKERNEL_HANG_TROUBLESHOOTING_CN.md`（卡死排查）
- 相关提交：`fa6204f`(方案一) `046e222`(进度条) `f6e1514`(方案三否决) `0be35d7`(默认引擎)

---

## 1. 起点与目标

内置(GPU)引擎（进程内 lada：YOLO11-seg 检测 + BasicVSR++ 恢复）此前最慢路径 ~1 小时（§4.5 流式 1fps），目标大幅提速、接近用户基线 lada-cli(<20min)。

## 2. 关键诊断（逐组件 profile，8K 单眼鱼眼 warm）

**去马赛克段瓶颈不是 AI 模型，而是输出编码。**

| 组件 | 占 wall |
|---|---|
| YOLO 检测 | 19% |
| BasicVSR++ 恢复 | 21% |
| → AI 模型合计 | **40%** |
| **lada VideoWriter 输出编码** | **59%（最大瓶颈）** |
| 解码(PyAV软解+swscale,×2线程) | 27%（重叠） |
| CPU blend | 22%（重叠） |

- lada VideoWriter 虽用 `hevc_nvenc`，但每帧 `GPU→CPU下载 + CPU swscale rgb24→yuv420p(4096²) + 回传NVENC` = **97ms/帧(10.3fps)**。慢的是 CPU 色彩转换 + 两次往返，不是 NVENC 硬件。
- **理论下限**：模型 40% → "other"清零时 ~6.7fps ≈ 9min（单卡、当前模型不变）。

## 3. 方案一（已落地）：GPU NVENC 直编替换 lada VideoWriter

- `engine.py` `restore_file` 拆为 `_restore_file_gpu_nvenc`(主) + `_restore_file_videowriter`(回退)。
- GPU 路径：恢复帧 BGR → 上传 GPU → `_prepare_restored_nv12(from_fisheye=False)` 用 M1 融合 kernel 做 BGR→NV12（**仅当前流同步，不与模型争用**）→ `_EncodeSink` NVENC 直编 → `mux_hevc_with_audio`。
- 仅对 PyNv 安全 8-bit SDR 启用；HDR/10-bit/bt2020 或编码器设置失败 → 抛 `_GpuEncodeSetupError` 廉价回退 VideoWriter；进入编码循环后的错误不回退（不白跑两遍去马赛克）。
- **实测**：去马赛克段 48.9s→25.5s(**1.92x**)；全单眼管线 71.6s→34.7s(**~2.06x**, 6.92fps)。输出正确（RGB 均值与输入一致、无绿块、mean|A−B|=1.2）。
- 日志确认：`[native] encoder=hevc_nvenc(gpu-resident) ...`。

## 4. 进度条修复：滚动窗口 fps

- 原 `_Progress` fps=done/总经过秒（累计平均），把首帧前 ~30s 启动开销长期摊进分母 → fps「一路上涨」、初始 ETA 冒出几十小时（用户误以为越跑越快）。
- 改为近 `window_sec`(默认60s)滚动速率：首个进度行就显示真实速度，ETA 稳定，并把 lada 每 180 帧(`max_clip_length`)一次的批量恢复 stall 平滑进去。
- 注：fps「上涨」纯属显示口径，真实瞬时速度从头就有（相邻行帧差÷秒差可验证）。

## 5. 方案三（已尝试·否决）：全 GPU 常驻单遍（融合）

**结论：单 GPU 下「融合」打不过「分段」。**

| 配置 | fps |
|---|---|
| GPU 帧源单独（解码+裁+鱼眼+颜色→torch） | **156**（极快）|
| 融合流式 原始（帧源每帧整设备同步） | 0.69 |
| 融合流式 去掉帧源整设备同步 | **3.05**（4.4x，输出正确）|
| **方案一 分段（默认）** | **6.92** |

- **根因证实**：帧源 156fps 不慢；融合 0.69fps 的元凶是帧源里 `cp.cuda.Device().synchronize()`（整设备同步）每帧会等另一线程 BasicVSR++ 整段 clip(10-14s)跑完 → 串行化。已去掉（`_iter_gpu_bgr_frames`/`_sbs`，靠 ThreadedDecoder 提前缓冲 + yield 前当前流同步保证就绪/复用安全）。
- **但仍比分段慢 2x**：再 profile 显示模型仍被并发的帧源/编码争用（尤其 `_EncodeSink.feed` 整设备同步每帧仍等模型，NVENC 防绿块必需）。
- **本质**：单卡总算力固定，融合让 几何+颜色+模型+编码 每帧并发抢一张卡互相拖慢；分段让每阶段独占整卡跑满反而快。融合只省中间 I/O，而那点用 NVENC/NVDEC 本就便宜。
- 流式 **保持默认关闭**；帧源去整设备同步的修复保留（对禁用路径 4.4x 改善 + 记录根因）。

## 6. 默认引擎

- `app_config` 默认 `engine`: `lada → native_gpu`；当前配置同步。打开 App 默认即「内置(GPU)」=方案一分段管线，可在界面随时切回 lada/jasna。

## 7. 最终状态与后续

- **现状**：内置(GPU)=方案一，单眼管线 ~7fps（warm），去马赛克段比改前快约一倍；进度/ETA 显示真实。
- **单卡已到甜点**：要再快只剩 ① 换更小/更快的恢复模型（画质权衡）；② 第二张 GPU（几何/编码与模型分卡，那时融合才会赢）。
- **不要做**：①别优化/换 AI 模型（只占 40%，下限 9min）；②别开流式/再追单卡融合（争用，慢 2-4x）；③检测模型 v2↔v4 对速度无影响（2.45 vs 2.47fps）。
- **测量纪律**：短片段 fps 不可靠（启动/cudnn/争用放大），真实时长跑全片或 ≥30-60s 长段。
