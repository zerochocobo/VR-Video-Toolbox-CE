# 内置(native_gpu)去马赛克——真实瓶颈剖析与提速方案

- 日期：2026-05-31
- 方法：逐组件 profile（monkeypatch 计时 YOLO/BasicVSR++/decode/encode/blend），8K 单眼鱼眼 4096×4096，warm，120–240 帧。
- 结论先行：**瓶颈不是 AI 模型，是输出编码（lada VideoWriter 的 CPU 色彩转换 + GPU↔CPU 往返）。流式(CuPy)方案因 GPU 争用反而慢 4 倍，已弃用。**

---

## 1. 文件路径逐段（run_single_eye_pipeline 鱼眼分支）

三段串行（8K 单眼 120 帧 warm）：

| 阶段 | 内容 | fps |
|---|---|---|
| A | split + to_fisheye（我方 GPU 几何） | 10.0 |
| B | **lada 去马赛克** | **2.45** |
| C | defisheye（我方 GPU 几何） | 11.2 |
| 合计 | 三段串行 | 1.68（~36min）|

→ 我方 GPU 几何（A/C，10–11fps）不是瓶颈。瓶颈在 B。

## 2. B（lada 去马赛克）内部拆解 —— 关键

instrument 计时（120 输出帧）：

| 组件 | 占 wall | 说明 |
|---|---|---|
| YOLO 检测 | 19.3% | 60 calls |
| **BasicVSR++ 恢复** | **20.8%** | 仅 2 clips / 240 clip-frames（整段就 2 个持续马赛克区）|
| **other（解码/编码/blend/IO）** | **59.8%** | ← 真正大头 |

**AI 模型合计只占 ~40%。** "other" 60% 才是瓶颈。

## 3. "other" 再拆解（最重要发现）

| 子项 | 占 wall | 实测 |
|---|---|---|
| **编码 VideoWriter.write** | **58.8%** | **10.3 fps / 97ms 每帧** @4096²（隔离实测）|
| 解码+swscale（PyAV 软解，×2 线程重叠） | 27.1% | 37fps（含 bgr24 转换），检测/恢复各解一遍 |
| blend（_restore_frame，CPU numpy） | 21.6% | 240 calls |

**★ 编码是单一最大瓶颈（59%）。** 注意：lada VideoWriter 其实已用 `hevc_nvenc`，但慢的不是 NVENC 硬件，而是它每帧做：
`restored GPU tensor → to_ndarray(GPU→CPU 下载) → from_ndarray(rgb24) → libav swscale rgb24→yuv420p（CPU，4096²）→ 上传 NVENC`。
CPU 色彩转换 + 两次 4096² 数据往返 = 97ms/帧。NVENC 真正编码很快（我方 gpu_engine 8K 解码受限也有 ~28fps）。

## 4. 流式(§4.5 CuPy)方案——验证为「陷阱」，已弃用

长段(240帧)实测流式 **0.69fps（~87min）**，比文件路径(2.65fps)慢 **~4 倍**。拆解：

| 组件 | 文件路径 | 流式 |
|---|---|---|
| YOLO/call | 145ms | **1304ms（9x 慢）** |
| BasicVSR++ | 25.7 cf/s | **3.5 cf/s（7x 慢）** |

根因：流式帧源在**同一张 GPU**上跑 CuPy 几何 + **整设备同步**（`cp.cuda.Device().synchronize()`/帧）+ CuPy↔torch dlpack 每帧 stream 同步，与 torch 的 YOLO/BasicVSR++ 推理**强争用 + 串行化**，把模型推理拖慢 7–9x。
（之前短段测出 3.9fps 是预热噪声，不可信。）→ `native_stream_enabled` 已改回**默认 False**。

## 5. 理论下限

文件路径 120 帧 44.8s 中，YOLO+BasicVSR = 18s。若 "other"→0：120/18 = **6.7fps ≈ 9min**（全片）。这是当前模型不变时的物理下限。当前 ~22min（文件路径 B 段 2.45fps，加 A/C 到 ~36min；纯 B 约 22min）。**可压缩空间：~22min → ~9–13min（约 2x）。**

---

## 6. 提速方案（按 性价比/风险 排序）

### ✅ 方案一（已实施并验证，2026-05-31）——用我方 GPU NVENC 替换 lada VideoWriter
**实测结果**（8K 单眼鱼眼，warm，同一 240 帧段）：
- 去马赛克段(B)：**48.9s → 25.5s（1.92x）**，编码 ~59% 开销基本消除。
- 全单眼管线(A+B+C)：**71.6s → 34.7s（~2.06x），6.92 fps**。
- 正确性：输出 RGB 均值 [105,84,78] ≈ 输入 [106,85,79]，无绿块/无偏色，std 55 内容正常，mean|A−B|=1.2（整帧忠实、仅马赛克区变化）。
- 实现：`engine.py` `restore_file` 拆为 `_restore_file_gpu_nvenc`（主）+ `_restore_file_videowriter`（HDR/10-bit/非 PyNv 安全源回退）。GPU 路径复用 `_prepare_restored_nv12(from_fisheye=False)` 做 BGR→NV12（仅当前流同步，不与模型争用）+ `_EncodeSink`/`_pack_planes`/`mux_hevc_with_audio`。设置阶段失败抛 `_GpuEncodeSetupError` 廉价回退；编码循环内错误不回退（避免去马赛克白跑两遍）。

#### 实现细节（原方案描述）——用我方 GPU NVENC 替换 lada VideoWriter
- 在 `restore_file` 输出端，把 lada VideoWriter 换成 gpu_engine 的 PyNv NVENC（`files._EncodeSink`）：恢复帧(CPU BGR) → 一次上传 GPU → **BGR→NV12 用 M1 融合 kernel（0.3ms）** → NVENC 从 GPU 直编，**省掉 CPU swscale rgb24→yuv420p + 两次 4096² 往返**。
- 预期：编码 97ms → ~5–10ms/帧，砍掉 59% 中的绝大部分 → 文件路径 **~22min → ~11–13min**，已优于 20min 基线。
- 风险低：只换输出 writer，不动 lada 线程/帧源，不引入 GPU 争用。我方 NVENC + _EncodeSink 已在 8K 实测无绿块、码率正常。
- 注意：engine.py 已在流式分支 import 了 `_EncodeSink/_pack_planes/_encoder_kwargs`，可直接复用到文件路径。

### 方案二（中收益·中风险）——单次解码共享
- lada 的 MosaicDetector 与 restoration loop **各解一遍**（27% 的 2x）。单次解码 + 帧共享 → 解码减半。需改 lada 线程（按检测→恢复滞后 buffer），较繁琐，收益中等。

### ❌ 方案三（已尝试并验证不划算，2026-05-31）——全 GPU 常驻单遍（融合）

**结论：单 GPU 下「融合」打不过「分段」，放弃。** 实测数据：

| 配置 | fps |
|---|---|
| GPU 帧源单独跑（解码+裁+鱼眼+NV12→torchBGR） | **156 fps**（6.4ms/帧，极快）|
| 融合流式 — 原始（帧源每帧整设备同步） | 0.69 fps |
| 融合流式 — 去掉帧源整设备同步（方案三修复） | **3.05 fps**（4.4x，输出正确无绿块）|
| **方案一 分段（现默认）** | **6.92 fps** |

- **根因证实**：帧源本身 156fps 不慢；融合流式 0.69fps 的元凶就是帧源里 `cp.cuda.Device().synchronize()`（整设备同步）——它每帧会等到另一线程 BasicVSR++ 整段 clip(10-14s)跑完才返回，把多线程打回串行。去掉后 4.4x→3.05fps。已去掉（`_iter_gpu_bgr_frames`/`_sbs`，靠 ThreadedDecoder 提前缓冲 + yield 前当前流同步保证就绪/复用安全）。
- **但仍比分段慢 2x**：再 profile 融合流式，YOLO/BasicVSR++ 仍比文件路径慢 ~7x（模型仍被并发的帧源/编码争用——尤其 `_EncodeSink.feed` 的整设备同步每帧仍在等模型，且这是 NVENC 防绿块所必需）。
- **本质**：单 GPU 上总算力固定，融合让 几何+颜色+模型+编码 每帧并发抢一张卡 → 互相拖慢；分段让每个阶段独占整卡跑满，反而更快。融合只省「中间编解码 I/O」，而那点用 NVENC/NVDEC 本就很便宜，省下的远抵不过争用损失。
- **决定**：流式仍**默认关闭**，方案一（分段）是单 GPU 甜点。帧源整设备同步的修复保留（对那条禁用路径是 4.4x 改善 + 记录根因），但不改默认。

#### 方案三原始设想（保留参考，未采用）——全 GPU 常驻单遍（重构 lada 流程）
- 解码(NVDEC)→几何→YOLO→BasicVSR++→blend→编码(NVENC)**全程 GPU、单遍、零中间文件**，逼近 ~9min 下限。
- **必须避开流式陷阱的根因**：① 几何改用 **torch**（`F.grid_sample` + 预算 grid，由 v360 LUT 生成）而非 CuPy，全程在 torch 自己的 stream，**不做整设备同步**；② 解码帧用 dlpack 包成 torch（不经 CuPy）；③ blend 已是 torch GPU（file 路径走 CPU 分支，需让 frame 在 GPU 上以触发 `_blend_gpu`）。
- 这是历来卡住开发的硬骨头，建议**先做方案一拿到确定收益**，再评估是否值得。

### 不要做
- **别优化/替换 AI 模型**（YOLO+BasicVSR 只占 40%，下限 ~9min；换 GAN 收益小、风险大）。检测模型 v2_accurate↔v4_fast 对速度无影响（实测 2.45 vs 2.47fps）。
- **别用现流式 CuPy 方案**（GPU 争用，慢 4x）。
- **别碰原 M2「杀 2x 解码」当主杠杆**（解码只占 27%，且方案一先解决更大的编码）。

---

## 7. 复现命令
```
# 逐段：A/B/C
# B 内部：patch BasicvsrppMosaicRestorer.restore + Yolo11SegmentationModel.inference 计时
# other：patch VideoReader.frames / VideoWriter.write / FrameRestorer._restore_frame 计时
# 编码隔离：vu.VideoWriter(...,'hevc_nvenc','-preset p5 -rc vbr -cq 20') 喂 120 帧 4096² → 10.3fps
```
（脚本见本次会话；务必 warm + ≥120 帧，短段 fps 不可信。）
