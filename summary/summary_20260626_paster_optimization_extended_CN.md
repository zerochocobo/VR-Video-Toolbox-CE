# Paster 性能优化：扩展研究方向

日期：2026-06-26

本文在 `summary_20260626_paster_performance_optimization_research_CN.md` 基础上，针对"NVENC encode 占 85%"这个已确认瓶颈，拓展出原文未覆盖的优化思路。

---

## 核心前提（来自已有研究）

- 瓶颈：8K P4/multipass=fullres/AQ NVENC encode，约 36ms/frame，实际吞吐约 25.8fps。
- alpha paste、decode 均不是瓶颈（合计 < 3ms/frame）。
- 约束：不改 P4，不降画质，不让测试代码影响生产热路径。

---

## 方向一：消除过宽的 Device Sync，解锁 NVENC ∥ decode+paste 并行

### 问题所在

`_EncodeSink.feed()` 里：

```python
cp.cuda.Device().synchronize()   # 全局设备 sync
data = self.enc.encode(app)      # 提交给 NVENC 硬件
```

`cp.cuda.Device().synchronize()` 会等待设备上 **所有** CUDA 操作完成——包括上一帧 NVENC encode 的硬件完成事件（如果 NVENC 内部使用 CUDA stream/event 上报完成）。这导致：

```
Frame N:   paste → [全局sync：等paste+上帧NVENC] → NVENC.encode(N) → [NVENC硬件跑]
Frame N+1: decode → paste → [全局sync：等paste+N帧NVENC] → NVENC.encode(N+1) → ...
```

结果是 NVENC 硬件从未与 CPU/CuPy 操作并行，完全串行。

### NVENC 硬件的并行能力

NVENC 是独立于 CUDA 核心的专用硬件引擎。理论上：
- CUDA Core 跑 paste 的时候，NVENC 可以同时 encode 上一帧
- 两者是真正物理并行的，不是时分复用

当前代码完全没有利用这个并行性。

### 正确的 sync 粒度

`feed()` 里做全局 sync 的真实目的是：**确保 packed buffer 写入（paste kernels）在 NVENC 读取前完成**。

更精确的做法：

```python
# 方案 A：stream-level sync（仅等 paste stream）
paste_stream.synchronize()
data = self.enc.encode(app)
# NVENC 开始跑 frame N，立即开始 frame N+1 的 decode/paste
```

但这要求：
1. paste kernels 固定跑在 `paste_stream` 上
2. NVENC 的 input 读取在 `paste_stream` 完成后才开始（通过 CUDA event 或 SDK 参数）

### PyNvVideoCodec 的 stream 支持

`nvc.CreateEncoder(w, h, fmt, False, **kwargs)` 第四个参数是 `useCudaGraph`（当前为 False）。  
VideoCodecSDK 的 `NvEncoderCuda` 支持：
- 指定输入 CUDA stream
- 在指定 stream 上等待 input buffer 就绪

如果 PyNvVideoCodec 暴露了相应 API（待查），可以：

```python
enc.Encode(app_frame, input_stream=paste_stream)
# 这让 NVENC 内部在 paste_stream 完成后才 DMA 输入
# CPU 可以立刻开始 frame N+1 的 decode，不用等 NVENC 完成
```

### ring buffer 的设计意图

`_EncodeSink.pending` 保留 4 个 AppFrame 引用，原本就是为了支持"NVENC 异步消费，不立刻释放"。但全局 sync 把这个设计的价值完全抵消了：sync 之后 NVENC 已经完成，ring buffer 实际上每次只有 1 帧在 NVENC 里。

**如果能换成 stream-level sync，ring buffer 就能真正发挥作用**——NVENC 最多同时持有 4 帧。

### 预期收益估算

假设 decode+paste 约 5ms，NVENC encode 约 36ms：

- 当前：串行，每帧 ~36ms
- 优化后：frame N+1 的 decode+paste (5ms) 与 frame N 的 NVENC encode (36ms) 完全重叠，瓶颈降至 max(36, 5) = 36ms 的 NVENC 硬件吞吐

实际上 NVENC 硬件本身就是瓶颈，吞吐提升可能有限。但 CPU 利用率和调度延迟会改善，可能有 5-15% 的实际帧率提升（减少调度空洞）。

### 研究任务

1. 查 PyNvVideoCodec Python API，确认是否暴露 `input_cuda_stream` 参数。
2. 用小样本验证：`paste_stream.synchronize()` 替代 `Device().synchronize()` 是否会复现绿块（buffer 生命周期问题）。
3. 如果复现绿块，说明 NVENC 读取不依赖 CUDA stream 同步，需要其他机制。

---

## ~~方向二：NVENC numSlicesPerFrame（单卡双引擎并行）~~ ❌ 已取消

**取消原因**：依赖 RTX 50 系列的双 NVENC 引擎，不适用于所有用户显卡。软件面向通用用户，不能针对特定硬件做默认配置。

---

## 方向三：用 CUDA Graph 消除 paste 内循环的 kernel launch 开销

### 背景

per-frame paste 内循环每帧做相同的 kernel 序列：

```
packed_copy (memcpy kernel) → alpha_luma (elementwise) → alpha_chroma (elementwise)
```

对于固定 rect 和固定 8K 分辨率，kernel 配置（block size、grid size）每帧完全相同，只有数据指针变化。

CUDA Graph 可以：
1. 第一帧"录制"kernel graph（capture mode）
2. 后续帧只做 pointer rebind + replay，CPU kernel launch overhead 接近零
3. GPU scheduler 可以更激进地流水线排列 kernel

### 当前已有的 CUDA Graph 入口

`nvc.CreateEncoder(w, h, fmt, False, **kwargs)` 第四个参数 `False` 是 `useCudaGraph`。  
如果改为 `True`，PyNvVideoCodec 内部会用 CUDA Graph 加速 encoder 的 input 处理路径。这是最低成本的尝试：**仅改一个参数**。

```python
# pynv_io.py :: PyNvEncoderSession.__init__
self._enc = nvc.CreateEncoder(self.width, self.height, self.fmt, True, **kwargs)
#                                                                 ^^^^
```

### 预期收益

- `useCudaGraph=True` 对 NVENC input DMA path 的提速：可能 1-3ms/frame
- CuPy 侧 alpha paste 的 CUDA Graph：需要手动实现，gains 约 0.5-1ms/frame
- 总体：边际收益，但实现成本低（改一个参数即可测试 encoder 侧）

### 研究任务

1. 将 `useCudaGraph=False` 改为 `True`，跑真实短基准对比
2. 如无负面效果（绿块、崩溃），可直接保留

---

## 方向四：更精细的 passthrough plan（减少重编码帧数）

原文 8.1 节已提到，但可以深挖以下两个子方向：

### 4a：Segment-end 尾部 passthrough

当前样本 `2_1` 中 rect 持续到末尾，尾部几乎无 passthrough 空间。但对于更多用户样本：

- 如果 restored rect 在 `frame_end - K` 帧结束，后面 K 帧只需 stream-copy
- 当前 `_try_paste_segments_gpu_passthrough()` 已有 passthrough 逻辑，但可能没有精细到"rect 消失后立刻切换 stream-copy"
- 值得 trace 一下 `passthrough_frames=602/10922` 的计划是否已包含尾部

### 4b：GOP 对齐的精细 passthrough

stream-copy 要求 keyframe 对齐。如果能在 encode 开始时控制 `gop` 参数，使输出 IDR 落在 rect 消失帧附近，则之后的 passthrough 段不需要等下一个自然 keyframe。

```python
# 在 _encoder_kwargs 里根据 rect end frame 动态设置 gop
# 让最后一个 IDR 正好落在 rect_end 帧
enc_kwargs["gop"] = str(rect_end_frame - frame_start)
```

这对"rect 只覆盖前半段"的场景收益显著，对"rect 覆盖全程"的样本无效。

---

## 方向五：multipass=twopass 替代 multipass=fullres（在质量等价前提下）

### 背景

当前注释中提到（`files.py:407-412`）：

> `multipass=fullres` 两路 rate control pass 均在全分辨率跑。multipass costs ~30% encode time。

NVENC 还有另一个模式：`multipass=twopass`（或 `multipass=qres`），第一路在 1/4 分辨率跑，第二路在全分辨率跑。

### 权衡

| 模式 | 速度 | 质量 |
|---|---|---|
| `multipass=disabled` | 最快 | 最低 |
| `multipass=qres` | 中 | 中 |
| `multipass=fullres` | -30% | 最高 |

如果 `qres` 的质量在视觉上与 `fullres` 无法区分（对于高码率 VBR 场景可能如此），则 `qres` 直接节省 ~30% encode 时间，折算到整体约 25% 提速。

### 约束分析

原文约束是"不调整 P4，不降低画质"。`multipass` 是独立参数，不是 preset。如果用户愿意用 VMAF/PSNR 对比验证 `qres` 和 `fullres` 的视觉等价性，这个方向可以解锁。

### 研究任务

1. 跑同一段视频：`multipass=fullres` vs `multipass=qres`，对比 FPS
2. 用 ffmpeg `libvmaf` filter 量化质量差异
3. 如 VMAF 差 < 0.5，可提案给用户确认是否接受

---

## 方向六：decode 侧 prefetch ring buffer（rect clip 预取）

### 现状

`PyNvThreadedSerialDecoder` 已经是 threaded serial decoder，应有内部 prefetch。但当 rect clip 有 GOP 边界时，`frame_at(seg_idx)` 可能需要等 NVDEC 完成 keyframe seek。

Profile 显示 `restored_decode_sync` 仅 0.2ms/frame，说明当前 decode 延迟已经很低，这个方向没有收益空间。

**结论：skip，不值得投入。**

---

## 方向七：NVENC 输入格式对齐检查

### 背景

`PyNvEncoderSession` 接受 `P010` 格式（10-bit），packed buffer 由 `_copy_planes_to_packed_views()` 生成。

`GpuP016AppFrame` 和 `GpuNv12AppFrame` 的内存布局是否与 NVENC 期望的 stride 精确对齐，直接影响是否有隐藏的 format conversion 开销。

检查点：
- `_copy_planes_to_packed_views()` 分配的 buffer stride 是否是 256 字节对齐（NVENC 偏好）
- 如果 stride 不对齐，NVENC 内部可能做一次额外 copy

### 研究任务

1. 打印 packed buffer 的 `strides` 属性
2. 对比 NVENC SDK 对 P010 格式的 alignment 要求（通常是 `widthInBytes` 的 256 字节倍数）
3. 如不对齐，在 `_copy_planes_to_packed_views()` 里 pad 到 256 对齐

---

## 优先级排序

| 方向 | 实施成本 | 潜在收益 | 风险 | 优先级 |
|---|---|---|---|---|
| 三：`useCudaGraph=True`（一行改动） | 极低 | 1-3ms/frame | 低 | **立即试** |
| ~~二：`numSlicesPerFrame=2`~~ | — | — | — | **已取消（硬件特定）** |
| 五：`multipass=qres` 质量等价验证 | 中 | ~25% | 需用户确认 | **优先研究** |
| 一：Stream-level sync 解锁 NVENC ∥ paste | 高 | 5-15%（调度改善） | 高（需防绿块） | **中期研究** |
| 四：passthrough 精细化 | 中 | 样本相关 | 低 | **中期** |
| 七：stride 对齐 | 低 | 未知 | 低 | **顺手查** |
| 六：decode prefetch | — | 几乎无 | — | **skip** |

---

## 最低成本实验序列（建议立刻跑）

```powershell
# 实验 A：useCudaGraph=True（仅改 pynv_io.py 一行，跑真实 FPS 对比）
# 修改：pynv_io.py::PyNvEncoderSession.__init__ 第四参数 False -> True
uv run python scripts\bench_oneclick_crop_paste.py `
  --src videos\test_8k_m426_2_demosaic.mp4 `
  --start 00:00:00 --end 00:00:10 `
  --rect 1680,1696,1360,1552 --crop-mode left `
  --skip-crop --restored <rect_crop.mp4> --no-measure-quality

# 实验 B：numSlicesPerFrame=2（在 _encoder_kwargs 末尾加一行）
# kwargs["numSlicesPerFrame"] = "2"
# 同上命令跑，对比 FPS

# 实验 C：multipass=qres（改 profile_kwargs 或 encode_config 里的 multipass 值）
# 同上命令跑，另外加 ffmpeg vmaf 对比
```

每个实验改动只需一行，风险极低，收益可能显著。

---

## 总结

在不改 P4 preset、不减 feather、不改 alpha paste 公式的前提下，仍有三个可研究的方向：

1. **`useCudaGraph=True`**：零成本实验，可能有小幅提速
2. **`multipass=qres` 等价性验证**：如画质等价则直接省 ~25% encode 时间

这三个方向都不触碰 alpha paste 逻辑，不引入新的 sync 风险，实验成本极低。

---

## 2026-06-26 实测结果补充

测试条件：

- 源：`videos/test_8k_m426_2_demosaic.mp4`
- 范围：`00:00:00-00:00:05`
- 帧数：`300`
- rect：`1680,1696,1360,1552`
- restored clip：`debug_output\paste_profile\20260626_171654\rect_crop.mp4`
- 编码基线：`P4`, `vbr`, `aq=1`, `bf=0`, `bitrate=30168kbps`, `maxbitrate=60337kbps`

为避免测试代码影响实机路径，先做了清理：

- 移除 `gpu_engine/files.py::paste_segments_gpu()` 默认路径中的 profile section wrapper。
- `VRVT_NVENC_CUDA_GRAPH=1` 作为 PyNv encoder 创建开关，默认关闭。
- `VRVT_PASTE_ENCODE_SYNC=stream` 作为 `_EncodeSink` 同步粒度开关，默认仍为 device sync；开关只在 sink 初始化时读取一次。

速度：

| 实验 | 环境 / 参数 | paste elapsed | FPS | 相对 fullres |
|---|---|---:|---:|---:|
| baseline | `multipass=fullres` | `13.404s` | `22.38` | - |
| CUDA Graph | `VRVT_NVENC_CUDA_GRAPH=1`, `multipass=fullres` | `13.331s` | `22.50` | `+0.5%` |
| fullres quality run | `multipass=fullres` | `13.321s` | `22.52` | - |
| qres | `multipass=qres` | `11.143s` | `26.92` | `+19.5%` |
| stream sync | `VRVT_PASTE_ENCODE_SYNC=stream`, `multipass=fullres` | `13.217s` | `22.70` | `+0.8%` |
| qres + graph + stream | `multipass=qres`, `VRVT_NVENC_CUDA_GRAPH=1`, `VRVT_PASTE_ENCODE_SYNC=stream` | `11.040s` | `27.17` | `+20.6%` |

质量：

| 实验 | Rect YUV PSNR | Full Y PSNR | Background Y PSNR | 输出大小 |
|---|---:|---:|---:|---:|
| fullres | `55.06 dB` | `48.53 dB` | `48.35 dB` | `35,418,881` bytes |
| qres | `55.10 dB` | `48.59 dB` | `48.41 dB` | `35,445,847` bytes |
| stream sync fullres | `55.06 dB` | `48.53 dB` | `48.35 dB` | `35,418,881` bytes |
| qres + graph + stream | `55.10 dB` | `48.59 dB` | `48.41 dB` | `35,445,847` bytes |

VMAF：

- qres 输出 vs fullres 输出，完整 `300` 帧：`96.210905`
- fullres 输出 vs 原始源，`n_subsample=5`：`86.033316`
- qres 输出 vs 原始源，`n_subsample=5`：`86.224242`

结论：

- `multipass=qres` 是当前最值得继续验证的方向；本样本速度提升约 `19.5%`，PSNR / VMAF 没有劣化。
- `useCudaGraph=True` 对 8K P4/fullres paste 的提升只有 `0.5%`，暂不值得默认开启。
- stream-level sync 短测未出绿块，质量指标一致，但收益只有 `0.8%`，风险收益比不够；保留实验开关，不建议默认。
- 组合实验的额外收益基本来自 `qres`，CUDA Graph 与 stream sync 叠加贡献不足 `1%`。
