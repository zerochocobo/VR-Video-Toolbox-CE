# 修复模型加速（Track A）：可捕获 DCNv2 + 整-clip CUDA Graph

日期：2026-06-22
环境：torch 2.8.0+cu128 / torchvision 0.23.0 / RTX 5060 Ti (sm_120) / Windows 10

## 1. 背景与问题

内置 GPU 引擎的马赛克修复模型是 BasicVSR++（GAN，推理只用 `generator_ema`），输入为 `(1, T, 3, 256, 256)` 的 clip，同尺寸残差输出，沿时间双向递归。

项目早已实现了一个"整 clip 一次捕获、后续 replay"的 CUDA Graph 包装器（`gpu_engine/native_mosaic/_cuda_graph_runner.py`），但**默认被禁用**：在 graph capture/replay 阶段触发 Windows 原生 fast-fail（`STATUS_STACK_BUFFER_OVERRUN 0xc0000409`），且该崩溃绕过 Python 解释器、`try/except` 无法兜住。

**根因定位**：崩溃源自单个算子 `torchvision.ops.deform_conv2d`（二阶可变形对齐 DCNv2 用到）。它是一个融合的原生 CUDA 算子，**不可被 CUDA Graph 捕获**，也无法被 ONNX/TensorRT 原生表达。这一个算子同时卡住了所有在线/离线加速路径。

## 2. 方案

### A0 — 可捕获的纯 PyTorch DCNv2

新增 `gpu_engine/native_mosaic/_deform_conv_native.py`，用标准、可捕获的 aten 算子（`arange` 构网格 + `grid_sample` 双线性采样 + `conv2d`）等价实现 modulated deformable convolution，签名对齐 `torchvision.ops.deform_conv2d`。

- 偏移/掩码通道布局与采样语义（zero padding、`align_corners`、调制 mask）严格对齐 torchvision。
- 在 `SecondOrderDeformableAlignment.forward`（vendored `basicvsr_plusplus_net.py`）接入，env 开关 `VRVT_NATIVE_DCN`（默认开，置 0 回退 torchvision）。

**数值校验**（全模型，真实权重）：
- fp32 PSNR **81.3 dB**、fp16 PSNR **64.7 dB**（视觉无差）；
- CUDA Graph replay 与 eager **位精确（max_abs=0）**。

### A1 — 重启整-clip CUDA Graph

- `_cuda_graph_runner.py`：默认策略改为"native DCN 开则自动开"（显式 `VRVT_CUDA_GRAPH` 始终可覆盖；native DCN 关时自动保持关闭以避免原生 fast-fail）。形状缓存 `max_entries` 2→4。
- 顺带修复 SPyNet `forward` 末尾的光流缩放：原写法 `flow[:, 0, :, :] *= ...`（原地切片）既不利于图导出、也曾在改写中暴露 `new_tensor([...])` 的 host 分配破坏捕获的坑。最终改为 `切片 * 标量 + cat` 的形式，**同时满足 capture 安全与导出友好**，数学完全等价。

**集成验证**（经 `BasicvsrppMosaicRestorer.restore()`，变长 clip）：
- 按形状捕获/缓存、`capture_failures=0`、输出对 eager **位精确**；
- 变长 T **不做 padding**（实测 padding 会扰动双向输出，stress 仅 25–31 dB），靠 per-shape 捕获 + LRU 缓存，未命中自动回退 eager。

## 3. 性能结果

T=30、fp16、RTX 5060 Ti：

| 路径 | 延迟 | 吞吐 |
|---|---|---|
| 旧生产默认（torchvision DCN，eager） | 631 ms/clip | 47.5 fps |
| native DCN（eager） | 674 ms/clip | 44.5 fps |
| **native DCN + 整-clip CUDA Graph** | **179 ms/clip** | **167.5 fps** |

**端到端提速 3.53x**，零中间文件、in-process、即开即用、位精确。

## 4. 其他路线评估

- **A2 / `torch.compile`（inductor）**：本环境 Windows 无可用 Triton（`TritonMissing`），inductor GPU 代码生成不可用 → 放弃。这也反向说明手写 CUDA Graph（不依赖 Triton/inductor）是本平台的稳健解。
- **A3 / 落盘 TensorRT**：做了完整可行性验证。模型换成 native DCN 后已是 TRT 原生可表达（无需自定义 DCNv2 plugin），in-memory dynamo 编译可跑通（需先修 SPyNet 原地算子，已修）。实测 T=6：TRT ~299 fps（相对 eager ~7.6x、相对 A1 约 1.8x），**但** 单形状编译 455 s、per-shape 重建、PSNR 仅 27 dB（需额外 fp32 累加调优）。多分钟级 per-T 构建与"即开即用 / 无文件 / 精确"目标冲突 → **不采纳**。

## 5. 结论与出货

**A1（native DCN + 整-clip CUDA Graph）为最终出货方案**：3.5x、零文件、位精确、即开即用、不依赖 Triton/TensorRT。默认行为已切换为 native DCN + CUDA Graph 开启。

逃生开关：`VRVT_NATIVE_DCN=0`（回 torchvision DCN）、`VRVT_CUDA_GRAPH=0`（关图）。

## 5b. 生产崩溃修复（默认开图后真实 one_click 复发 0xc0000409）

默认开图后，真实 one_click 跑 `videos/2_2.mp4`（pre-extract 分段经 `restore_file` 修复）在某分段中途崩 `0xc0000409`。隔离单线程测试不复现，必须走真实引擎。

**根因**：CUDA graph **运行时 capture** 与并发的 NVENC 编码 / YOLO 检测线程争用进程级 CUDA allocator 状态。`restore_file` 是流式设计——修复（捕获新 clip 形状的图）与编码/检测在不同线程并发。捕获默认 `capture_error_mode='global'`，任何跨线程 GPU 活动会让 CUDA `__fastfail` 整个进程（不可 try/except）。

**修复（三件套）**：
1. capture 加 `capture_error_mode="thread_local"`（把不可捕获的硬崩降级为可捕获的 `cudaErrorStreamCaptureUnsupported`）。
2. **运行时禁止 capture，只在单线程 warmup 时 capture**：`CudaGraphRunner._capture_allowed` 默认 False + `allow_capture()` 上下文（`warmup_graph` 内启用）；运行时未命中形状直接走 eager。这是真正的根因修复——杜绝并发期 capture。
3. `restore_file` 开头（线程未起、单线程）预热 `warmup_graph(max_clip_length)`，主导长度 clip 走 replay 拿 3.5x，其余 eager。启动期 warmup 降为小 T（仅校验，不占大图 VRAM）。

**验证**：崩溃的那个分段（1040x1472 / 2220 帧）修复后 `returned True`、无崩溃；集成测试确认 warmed 形状 replay 位精确、unwarmed eager 位精确、运行时 0 capture、0 failure。默认全局安全：任何路径运行时都不 capture（流式 SBS 路径暂为 eager-safe）。

## 6. 改动与验证清单

改动文件：
- 新增 `gpu_engine/native_mosaic/_deform_conv_native.py`
- 改 `gpu_engine/native_mosaic/_vendor/lada/models/basicvsrpp/mmagic/basicvsr_plusplus_net.py`（接入 native DCN；SPyNet 光流缩放 capture/export 友好化）
- 改 `gpu_engine/native_mosaic/_cuda_graph_runner.py`（默认耦合 native DCN；`max_entries` 4）

验证/基准脚本（`scripts/`）：
- `verify_native_dcn.py`（算子级 parity + CUDA Graph 捕获）
- `verify_native_dcn_fullmodel.py`（全模型 PSNR parity）
- `verify_cuda_graph_fullmodel.py`（全模型捕获/replay）
- `verify_clip_padding.py`（padding 扰动评估）
- `verify_restorer_integration.py`（restorer 端到端、变长 clip、位精确）
- `bench_restore_paths.py`（三路径基准）、`bench_a3_trt.py`（TRT 路线评估）

后续：Track B —— 将马赛克一键流程改为 in-VRAM 流式，消除视频处理中间文件（独立推进）。
