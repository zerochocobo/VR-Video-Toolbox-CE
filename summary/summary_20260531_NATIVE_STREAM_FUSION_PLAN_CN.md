# 内置引擎(native_gpu)流式融合 —— 性能发现总结 + Option A 实施计划

- 文档日期：2026-05-31
- 适用项目：VR_Video_Toolbox_NE
- 背景：`prompt/HANDOVER_20260530.md`（§4.5 流式）+ 本次性能排查
- 状态：**计划，供开发接手**。Option A = PyNv↔lada 单次解码、GPU 帧常驻、避免整设备同步。

---

## Part 1 — 本次性能排查发现（必读）

### 1.1 现象
用户实测 8K 单眼鱼眼，§4.5 无中间文件流式路径约 **1 fps（全片 ETA ~1 小时）**，比原 lada-cli 基线（<20 分钟）慢得多。

### 1.2 分层 profile（test_8k_0，眼睛鱼眼实际 4096×4096）
| 环节 | 速度 | 结论 |
|---|---|---|
| PyNv/NVDEC 解码 | ~28 fps（PyAV 软解 28，PyNv 更快） | 不慢 |
| YOLO 检测 inference | ~49 fps | 不慢 |
| BasicVSR++ 恢复 forward（256² 单区域，T=1..120）| ~48 fps | 不慢 |
| 我方 GPU 几何（crop+fisheye） | 帧源单线程 ~48 fps | 不慢 |
| **流式帧源颜色转换 `_planes_to_torch_bgr`** | 占帧源 **80%** | 慢（见 §2.3） |
| `restore_file`（文件路径 lada，**warm**） | **~8.5 fps** | 可接受 |
| 文件路径全链（split+fisheye→restore→defisheye，3 段串行） | ~3 fps（warm 估算） | ≈ 基线 ~19 分钟 |
| **§4.5 流式** | **~1 fps** | 严重倒退 |

> 注意：早期测到的 1.1 fps / 36 分钟是**冷启动**（cudnn autotune + VRAM 预热 + 每段 init）放大的假象；warm 后 lada 恢复 8.5 fps、文件全链 ≈ 基线。

### 1.3 根因（两个）
1. **整设备同步在多线程流水线里串行化**：§4.5 帧源里每帧 `cp.cuda.Device().synchronize()`（为等 PyNv 不透明解码流可见）。单线程没事（帧源 48 fps），但 lada 的 `FrameRestorer` 是**多线程**（检测 worker / clip 恢复 worker / 帧合成 worker 并行），帧源运行在检测线程里，每次**整设备**同步都会等待恢复线程的 BasicVSR++ 跑完 → 制造跨线程假依赖 → 把并行打回串行。
2. **2× 解码**：lada 的检测 worker 和帧合成 worker **各自读一遍帧源** → 8K 源被解码两遍 + 几何两遍 + 颜色转换两遍。

### 1.4 已止血（本次已改）
- `one_click/logic.py: _native_stream_allowed()` 改为**默认关闭** §4.5 流式（需 `app_config.native_stream_enabled=True` 才启用）。native_gpu 现走文件路径（≈ 基线 ~20 分钟，不再 1 小时）。

---

## Part 2 — Option A 实施计划：单次解码 + GPU 帧常驻 + 避免整设备同步

### 2.0 目标
8K 单眼/双眼鱼眼端到端从 ~20 分钟降到 **~8-10 分钟**（解码/检测/恢复/几何尽量重叠或单线程紧凑，无中间文件、无整设备同步串行化）。恢复段仍 8-bit（lada 固有）。

### 2.1 关键判断：放弃 lada 的多线程，改**单线程 GPU 管线**（推荐 A1）

PTMediaServer 的 `pipeline/pynv_stream.py` 是**单线程**走完 decode→处理→encode（用 `cp.cuda.get_current_stream().synchronize()` 当前流同步 + 编码输入 ring slot），正因如此它在 8K 能跑 30+ fps、不受跨线程整设备同步之苦。lada 的多线程在 CPU 帧时靠重叠提速，但在 GPU + 不透明解码流下，跨线程同步反而是毒药（§1.3）。

**A1（推荐）**：不复用 lada 的 `FrameRestorer` 线程架构，而是**单线程**驱动 lada 的检测/跟踪/恢复**组件**（复用 `MosaicDetector` 的 `Scene`/`Clip` 跟踪逻辑、`Yolo11SegmentationModel`、`BasicvsrppMosaicRestorer`），在一个循环里：
```
PyNv 解码一帧(GPU) → crop+fisheye(CuPy, GPU) → 检测(GPU torch) → 更新 Clip 跟踪
   → clip 满/结束时 BasicVSR++ 恢复(GPU) → blend 回该帧(GPU) → fisheye→VR(CuPy)
   → PyNv 编码(GPU, _EncodeSink ring) → 裸 HEVC
最后 ffmpeg mux 音频。
```
- **单次解码**（不再 2x）。
- 单线程 → 只用**当前流同步**，无跨线程整设备同步串行化。
- 全程 GPU 常驻，无中间文件。
- 估算：各阶段串行但都快（解码 14 + 检测 49 + 恢复 48 fps + 几何 ~50），harmonic ≈ 7-10 fps。

**A2（备选，更难）**：保留 lada 多线程，但把所有 `cp.cuda.Device().synchronize()` 换成**每流同步 + CUDA event 跨流依赖**，并让 CuPy/torch/PyNv 共享一个流；再把"检测/恢复 2x 读帧"改成**单次解码 + 帧 ring 共享**（检测线程把帧 push 给恢复线程，buffer 大小≈检测→恢复滞后≈clip 长度，4096² 太大需放 CPU pinned 内存按需上传）。A2 更接近 lada 原状但同步正确性极难调，风险高。**建议先做 A1。**

### 2.2 单次解码 + 帧共享（A1 里如何让检测与恢复共用一帧）
A1 单线程下，检测和 blend 用**同一帧**，天然单次解码、无需跨线程共享。唯一缓冲需求：BasicVSR++ 是**时序**的，一个 Clip（最长 `max_clip_length=180` 帧）要等区域跨帧攒齐才恢复，blend 回去时需要那些帧的原图。
- 方案：维护一个**帧 ring（GPU 或 CPU pinned）**，保留最近 ≤180 帧的鱼眼 NV12（GPU 25MB/帧×180≈4.5GB，16GB 卡偏紧；建议放 **CPU pinned** ~4.5GB，blend 时按需上传，PCIe ~5ms/帧可接受）。
- 或**限制 clip 长度**（如 60）换更小 buffer + 略降时序稳定性。
- 复用 lada `mosaic_detector.py` 的 `Scene`/`Clip` 与 `crop_to_box_v3`/resize/pad 逻辑（这些已较优），只把"线程 + VideoReader"换成我们的单线程 + GPU 帧源。

### 2.3 颜色转换必须重写（当前占帧源 80%）
`engine.py: _planes_to_torch_bgr` 用 `repeat_interleave(2).repeat_interleave(2)` 做色度上采样 + dlpack 同步 + 整幅 BGR 矩阵——是帧源 80% 成本。
- 改为**单个 CuPy RawKernel**（NV12/P010 → BGR，或直接 NV12↔我们处理域），融合上采样 + YUV→RGB，零额外同步。
- 注意：§4.5 当时避开 CuPy RawKernel 是因为"首次 JIT 卡住"——那是 **cuda13x 的坑，现已切 cuda12.8 全栈对齐解决**（见 memory cuda-env-blackwell-coexistence）。现在新增 RawKernel 安全。
- 更进一步：lada 检测/恢复要 **BGR uint8**；若能让 lada 直接吃我们的 GPU 平面（torch CUDA），可少一次颜色往返。但 lada 内部按 BGR HWC，改动大，**第一步先做对的 BGR RawKernel**。

### 2.4 同步纪律（贯穿 A1）
- **禁用 `cp.cuda.Device().synchronize()`**（整设备）。单线程下用 `cp.cuda.get_current_stream().synchronize()`，且尽量只在必须的边界同步（PyNv 解码→CuPy 读、CuPy→torch dlpack、torch→CuPy dlpack、编码前 _EncodeSink 已处理）。
- **CuPy ↔ torch 共享流**：用同一 CUDA stream 包住两边（torch `with torch.cuda.stream(s)` + CuPy `with cp.cuda.ExternalStream(s.cuda_stream)`），dlpack handoff 就不需要整设备同步，只需流内顺序。参考 PTMediaServer 的 CUDA_SHARED_STREAM 思路。
- PyNv ThreadedDecoder 解码流不透明：单线程下，解码→CuPy 读之间做一次当前流/设备同步是可接受的（不再跨线程，不会等别的阶段）；或评估 `SimpleDecoder` 顺序读是否够用。
- 编码输入生命周期：继续用 `gpu_engine.files._EncodeSink`（ring 延迟释放，已验证防绿块）。

### 2.5 里程碑（建议顺序，每步可独立验证/提速）
1. **M1 颜色 RawKernel**：把 `_planes_to_torch_bgr` / `_torch_bgr_to_nv12_cupy` 换成 CuPy RawKernel，单测正确性（PSNR vs 现实现）+ 帧源 fps（目标帧源 >150 fps）。低风险、立竿见影。
2. **M2 单线程 GPU 管线骨架**：写 `engine.restore_single_eye_stream_v2()`：PyNv 解码→crop+fisheye→检测→Clip 跟踪→恢复→blend→defisheye→编码，单线程、当前流同步、帧 ring（先 CPU pinned）。先**非鱼眼**单眼跑通，再加鱼眼。
3. **M3 共享流 + 去整设备同步**：CuPy/torch 共流，清掉所有 Device().synchronize()，确认无竞态（逐帧 PSNR vs 文件路径）。
4. **M4 双眼 SBS**：单解码整帧 SBS，左右半各几何，一个管线。
5. **M5 8K 全片实测**：对比文件路径 + lada-cli 基线（同检测模型！见 §2.7），出 fps/显存/画质表。
6. **M6 接线**：`_native_stream_allowed` 重新默认开启（仅当 v2 管线达标）；保留文件路径作回退。

### 2.6 度量方法（务必 warm）
- 每次先 warm 一遍（cudnn autotune + 模型 + JIT），再计时第二遍。
- 用 `videos/test_8k*.mp4`，先 2-4 秒段落，再拉长。
- 逐帧画质：解码输出与"文件路径"输出逐帧对齐 PSNR（注意 ThreadedDecoder vs SimpleDecoder 2 帧偏移，见 HANDOVER_20260529 §3.4）。
- 关键对照：**与 lada-cli 必须用同一检测模型**（见 §2.7）。

### 2.7 重要对照变量：检测模型
用户 "<20 分钟" 基线用的是 **`lada_mosaic_detection_model_v4_fast.pt`**（通用快模型），而内置引擎现默认 **`lada_vr_mosaic_detection_model_v2_accurate.pt`**（VR 准模型，90MB）。实测三种模型 restore_file 都 ~8.5 fps（检测 inference 不是瓶颈），但**检测出的区域数量会影响 MosaicDetector 编排成本与画质**。建议：把检测模型做成可配置（已支持 `app_config native_detection_model`），M5 对照时跑同模型；并向用户确认质量/速度取舍。

### 2.8 风险与回退
| 风险 | 缓解 |
|---|---|
| 单线程失去重叠，不够快 | 各阶段本就快（解码14/检测49/恢复48），harmonic ~8fps 已优于 baseline；不够再考虑 A2 双流重叠 |
| 帧 ring 显存/内存 | 用 CPU pinned；或限 clip 长度；4096² NV12 ~25MB/帧 |
| CuPy/torch 共享流的竞态 | 逐帧 PSNR 对照文件路径；先保守同步再逐步去 |
| 复用 lada Scene/Clip 跟踪逻辑耦合其线程假设 | 抽出纯逻辑（不含 queue/thread），单测 |
| 改坏现有文件路径 | 文件路径保持默认回退（`_native_stream_allowed` 默认 False，v2 达标才开） |

### 2.9 涉及文件
- 改：`gpu_engine/native_mosaic/engine.py`（新 v2 单线程管线 + 颜色 RawKernel）、`gpu_engine/nv12_kernels.py`（新增 NV12↔BGR RawKernel）、`one_click/logic.py`（达标后重开流式）。
- 复用/可能小改：`_vendor/lada/restorationpipeline/mosaic_detector.py`（抽出 Scene/Clip 跟踪逻辑供单线程调用）、`_vendor/lada/models/**`（检测/恢复模型，按现状）。
- 参考：`reference/PTMediaServer/pipeline/pynv_stream.py`（单线程 GPU 管线 + 当前流同步 + 编码 ring）、`pipeline/matting.py`（CuPy RawKernel + 共享流）。

---

## Part 3 — 给开发的 TL;DR
1. §4.5 流式 1fps 的根因 = **整设备同步在 lada 多线程里串行化** + **2x 解码**。已默认关闭止血。
2. Option A = **单线程 GPU 管线**（A1）：单次解码、GPU 帧常驻、只用当前流同步、CuPy/torch 共享流、颜色转换换 RawKernel、复用 lada 的 Clip 跟踪与模型。目标 ~8-10 分钟。
3. 先做 M1（颜色 RawKernel，低风险大收益）→ M2 骨架 → M3 去整设备同步 → 验证 → 重开。
4. 对照基线务必同检测模型（baseline 用 v4_fast，当前默认 v2_accurate）。
5. 全程 warm 计时，逐帧 PSNR 对照文件路径，文件路径保持回退。
