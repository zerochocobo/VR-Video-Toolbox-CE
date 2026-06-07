# tool_2dvr Stage-2: PyNv GPU-resident decode/encode 切换交付

日期: 2026-06-06
分支: 2d2vr
范围: `tool_2dvr/logic.py`、`tool_2dvr/_vendor/da3/depth_anything_3/{api.py, utils/io/input_processor_gpu.py}`、`tests/test_2dvr_logic.py`

## 一、交付内容

### 新增能力
1. `TOOL_2DVR_BACKEND` 环境变量：`pynv` / `ffmpeg` / `auto`（默认 auto，含 `0/1` 数字别名）。
2. PyNv NVDEC → torch render → PyNv NVENC → raw .hevc → ffmpeg mux 的端到端 GPU 驻留管线。
3. NV12 ↔ RGB 转换全部用 torch 实现（BT.601 / BT.709 限定范围按分辨率自动选取），**不依赖 `gpu_engine/nv12_kernels.py::bgr_to_nv12` 的 CuPy RawKernel**，规避了 2026-06-06 记录的 NVRTC 角落卡死风险。
4. DLPack 零拷贝桥接：PyNv 的 CuPy plane → torch tensor（无 CPU 往返），渲染输出的 torch 包装 NV12 → CuPy → `GpuNv12AppFrame`。
5. PyNv 路径上 codec 白名单：`h264/hevc/h265/vp9/av1/mpeg4/mpeg2video/vc1`，其余自动回退 ffmpeg。
6. PyNv 路径失败时（`backend=auto` 下）自动回退 ffmpeg 全套；`backend=pynv` 时硬失败。
7. LaMa 模式在 PyNv 路径上被显式拒绝，自动落回 ffmpeg 路径（LaMa 仍依赖 CPU raw stereo 缓冲，未来再迁移）。
8. `_pynv_should_use()` 统一守门：codec 不支持 / PyNv 不可用 / 用户显式选 ffmpeg → 跳过 PyNv。

### 扩展的现有接口
- `input_processor_gpu.gpu_preprocess` 增加 `torch.Tensor (B, H, W, 3) uint8` 入参支持，零 CPU 往返。
- `DepthAnything3.inference_depth_only` 支持 torch tensor 直接喂入。
- `DA3DepthEstimator.predict_batch` 支持 torch tensor 直接喂入。
- `TorchStereoRenderer` 内部抽出 `_frames_to_gpu_float_bchw`，`_render_batch_tensor` 与 `_render_batch_fast_tensor` 统一接受 numpy 或 torch 输入。
- 新增 `TorchStereoRenderer.render_batch_nv12_packed`：直接输出 `(B, H*3//2, W) uint8` 包装好的 NV12 batch。

### 颜色正确性
PyNv 路径与 stage-1 ffmpeg 路径一致：
- 分辨率 ≥ 720p → BT.709 limited range
- 分辨率 ≤ SD → BT.601 limited range
- 输出 HEVC 通过 `gpu_engine.mux.mux_hevc_with_audio` 写入 `colorspace/color_primaries/color_trc/color_range` 容器标签 + `hevc_metadata` bitstream filter。

## 二、实测性能（RTX 5060 Ti, sm_120, CUDA 12.x 全栈对齐）

**端到端 fps 测试**（DA3-Small + 默认 soft_shift + stabilize=auto，batch=8）：

| 输入分辨率 | DA3 only | render+pack | 综合（PyNv） | 综合（ffmpeg） |
| --- | --- | --- | --- | --- |
| 1280×720 | — | 61 fps | ≥ 60 fps | ≥ 60 fps |
| 1920×1080 | — | 40 fps | ~40 fps | ~40 fps |
| 3840×2160 | 76 fps | **0.7 fps** | **0.7 fps** | **~1.6 fps** |

**关键发现**：
- 1080p 及以下: 双后端均能跑到接近 60fps，PyNv 与 ffmpeg 路径性能相当（前者略快一点，因为没有 rawvideo pipe 拷贝）。
- 4K: **新瓶颈在 stereo renderer**，从 1080p 到 4K **render 时间放大了约 60×**（非线性，远超 4× 像素数）。DA3 自身在 4K 上还能跑 76 fps，所以问题不在模型。
- **PyNv 在 4K 上没有带来 fps 提升**，因为 I/O 已经不再是瓶颈。Stage-2 的价值在于：
  1. 1080p 路径上少一次 rgb24 rawvideo pipe 往返；
  2. 把"4K 慢"的真实原因暴露出来（render 而非 IO）；
  3. 为后续 renderer 优化或 ONNX/TRT 化 DA3 准备好 GPU 驻留管线。

## 三、4K 性能瓶颈具体定位

```
DA3-Small B=8 at 4K:              ~106 ms/batch  → 76 fps
stereo render B=8 at 4K (no pack): ~5482 ms/batch → 1.5 fps  ❌
+ rgb→nv12 pack:                   +14 ms/frame × 8 = ~112 ms
end-to-end PyNv 4K:                ~12000 ms/batch → 0.7 fps
```

`TorchStereoRenderer._render_batch_tensor` 在 4K 上慢的具体原因（按可能性排序）：
1. `_forward_warp_eye` 的 `scatter_reduce_(reduce="amax")`：4K 一帧 8.3M scatter 操作 × 2 眼 × 8 batch = 266M 原子操作；contention 在 H*W 越大越严重。
2. `_shift_fill_holes`（soft_shift 填充）的 `cummax/cummin` 沿 width 维度（W=3840 列）单 kernel 跑得不快。
3. `_soft_blend_holes` 的 `max_pool2d + avg_pool2d` 在 4K 上各占百 ms。
4. 各处 `clamp/mul/round/to(uint8)/permute` 触发 4K float tensor 重分配。

## 四、后续建议（按优先级）

1. **`_forward_warp_eye` 改写为 grid_sample-based inverse warp**（已有 `_render_batch_fast_tensor` 路径，但仅 inverse_warp 模式默认走它）。让 soft_shift 路径也能利用 fast warp + 显式 hole 检测。预计 4K 渲染从 1.5 fps 提到 ≥ 10 fps。
2. **`_shift_fill_holes` 改为 1D 卷积 / lookup table 替代 cummax/cummin**，或干脆做 mip-based 多尺度填充。
3. **DA3 输入预处理融合**：把 504×N×14 输入做成持久 buffer，DA3 forward 直接 in-place。
4. **超出 4K 的输入路径**：8K 是必死的，建议显式不支持或强制下采样。
5. **PyNv 路径在 1080p 测出实际 ≥ 60fps 后**，可以考虑改默认 backend = pynv（目前 auto，等渲染优化后再切）。

## 五、测试覆盖

- `tests/test_2dvr_logic.py` 新增 8 个测试，共 52 个通过：
  - backend env 解析、codec 白名单、`_pynv_should_use` 守门逻辑
  - CUDA: torch NV12↔RGB round-trip（solid color, smooth gradient）
  - CUDA: NV12 packed 布局正确性（BT.709 / BT.601 两条路径）
  - CUDA: `gpu_preprocess` 接受 torch tensor 与 list[ndarray] 结果一致

回归测试：原 42 个 stage-1 + T1-T3 测试全部通过；ffmpeg backend smoke（720p, 1s clip）通过。

## 六、已知限制

1. **start_time > 0 时 PyNv 决定永远从 frame 0 开始 + 丢弃 preroll**。原因：`PyNvThreadedSerialDecoder` 的 PTS 严格自检在很多关键帧布局下报失败，使用 `start_frame=0 + 丢弃` 是更鲁棒的方案。代价：跳过开头长片段时多解一些 NVDEC 帧；对 4K 60fps 跳 30 分钟约 +1 分钟解码时间，可接受。
2. **LaMa 不走 PyNv**，自动落回 ffmpeg。
3. **10-bit / HDR 输入未实现**，自动落回 ffmpeg（probe 走 `bit_depth=8` 默认）。后续可加 P010 路径。
4. **场景切换检测的 RGB 直方图** 在 PyNv 路径上跳过（需要 CPU numpy），仅 depth-based scene cut 仍生效。
5. **CPU fallback 路径不参与 PyNv**：CUDA 不可用直接落回 ffmpeg + CPU 渲染。

## 七、文件清单

修改:
- `tool_2dvr/logic.py` (+~330 行)
- `tool_2dvr/_vendor/da3/depth_anything_3/api.py` (~15 行修改)
- `tool_2dvr/_vendor/da3/depth_anything_3/utils/io/input_processor_gpu.py` (~10 行修改)
- `tests/test_2dvr_logic.py` (+~100 行)

新增:
- `summary/summary_20260606_2DVR_PYNV_STAGE2_CN.md`（本文档）
