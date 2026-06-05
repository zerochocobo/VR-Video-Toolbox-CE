# OneClick 解码阶段第一阶段优化计划（不含 TRT）

聚焦 oneclick 的 `process_lada` 路径，在不引入 TensorRT 的前提下，
消除当前 decode → 模型送入这一段的 CPU/拷贝/同步瓶颈。本阶段产出的 GPU
frame source 与 batch 张量布局，也是后续 TRT 化（第二阶段）的前置条件。

## 背景与现状

入口：[one_click/logic.py:process_lada](one_click/logic.py)，调用
[gpu_engine/native_mosaic/engine.py](gpu_engine/native_mosaic/engine.py)
的 `NativeMosaicEngine.restore_file`。

主路径目前的实际行为：

1. **解码**：走 vendored lada `VideoReader` = PyAV CPU 解码 + swscale 输出
   BGR ndarray。SBS / streaming 子路径才使用 `PyNvThreadedSerialDecoder` +
   `nv12_kernels` 走 GPU；oneclick 全片整体跑 lada 时未启用 GPU
   frame source factory。
2. **送入模型前**：`crop_to_box_v3` + `image_utils.resize`（cv2 INTER_LINEAR）
   + `pad_image` 全部在 CPU；随后 `torch.stack(...).to(device)` 每 clip 一次
   大 H2D 拷贝。
3. **NV12→BGR**：仅 GPU 子路径用 CuPy `nv12_kernels.nv12_to_bgr`；每帧之后
   `cp.cuda.get_current_stream().synchronize()`，迫使整条流串行。
4. **VRAM**：`runtime.free_memory_pool()` + `torch.cuda.empty_cache()`，
   无主动 D→H offload，`max_clip_length=180` 受显存限。

4K/8K 素材下，单帧成本主要被 CPU decode + swscale + numpy→torch H2D + cv2
resize 占据，模型本体反而吃不饱。

| Pri | 项目 | 预估工时 | 风险 |
|---|---|---|---|
| S1 | `restore_file` 主路径接入 GPU frame source factory | ~1.5 天 | 中 |
| S2 | vendored mosaic_detector 内 crop/resize 迁 GPU | ~1 天 | 中 |
| S3 | clip 级 batch resize/pad 张量化 | ~0.5 天 | 低 |
| S4 | 去除每帧全流 synchronize，引入 stream/event | ~0.5 天 | 中 |
| S5 | VRAM offloader（BlendBuffer D→H pinned） | ~1 天 | 中 |
| S6 | warmup + 端到端基线 profile | ~0.5 天 | 低 |

---

## S1：`restore_file` 主路径接入 GPU frame source factory

### 现状

`NativeMosaicEngine` 已经存在 `_make_gpu_bgr_frame_source_factory`
（SBS/streaming 用），但 oneclick 主路径 `restore_file` 把 `video_file` 一路
透传给 vendored `FrameRestorer`，其内部的 `_frame_feeder_worker` 自行 PyAV 打开。

### 改造

- 在 `restore_file` 里强制构造 GPU factory，落参数：
  - `crop_mode='passthrough'`（新增模式：不做 SBS 半切、不做 fisheye）
  - 输出形态固定为 GPU torch tensor (uint8, HWC BGR, device='cuda')，
    与下游 detect / restore 期望一致
- 修改 vendored `FrameRestorer` 接受 `frame_source_factory` 参数（已存在的
  hook 路径优先复用，缺失则按当前代码风格直接加一个可选 kwarg，None 时
  回退原行为，保证 SBS/streaming 子路径不受影响）。
- `_frame_feeder_worker` 中 PyAV 开流的分支用 `if frame_source is None`
  包起来；非 None 时直接 `for frame in frame_source:`，跳过所有
  `av.open` / `cv2.cvtColor` 逻辑。

### 验收

- 1 分钟 4K SBS 样片：解码线程 CPU 占用从 ~100% 降到 <20%；
  GPU 解码线程在 `nvidia-smi dmon` 上出现 dec 利用率。
- 输出 mp4 与改造前帧级 PSNR 完全一致（同种子）。

### 涉及文件

- [gpu_engine/native_mosaic/engine.py](gpu_engine/native_mosaic/engine.py)
  `restore_file`、`_make_gpu_bgr_frame_source_factory`
- `gpu_engine/native_mosaic/_vendor/lada/restorationpipeline/frame_restorer.py`
- `gpu_engine/native_mosaic/_vendor/lada/restorationpipeline/mosaic_detector.py`
  `_frame_feeder_worker`

---

## S2：vendored mosaic_detector 内 crop/resize 迁 GPU

### 现状

`mosaic_detector._frame_feeder_worker` 与 `Clip.__init__` 收到 frame 后仍
按 ndarray 处理：`crop_to_box_v3` 用 numpy 切片，`image_utils.resize`
(cv2 INTER_LINEAR) 缩到 256×256，`pad_image` 用 cv2 reflect。

### 改造

- 在 vendored utils 旁边新增 `gpu_engine/native_mosaic/_gpu_ops.py`
  （与 `nv12_kernels.py` 同风格，纯函数 + torch）：
  - `crop_to_box_gpu(tensor_hwc_bgr, box) -> tensor`：torch.narrow / 切片
  - `resize_bilinear_gpu(tensor, size) -> tensor`：`F.interpolate(mode='bilinear', align_corners=False)`
  - `pad_reflect_gpu(tensor, pad) -> tensor`：`F.pad(..., mode='reflect')`
- `mosaic_detector` 在判断输入是 torch.Tensor 且在 cuda 上时走新路径；
  ndarray 时回退老路径（保留兼容）。
- `Clip` 内 crop 列表元素从 ndarray 改为 GPU tensor view；
  原本最终 `np.stack` 改为 `torch.stack` 直接产 GPU batch。

### 验收

- 同一 segment 测试集：detect 阶段 wall time 至少减半。
- D2H 调用计数（用 torch profiler）从 N 帧降到 0。

### 涉及文件

- 新增 `gpu_engine/native_mosaic/_gpu_ops.py`
- `gpu_engine/native_mosaic/_vendor/lada/restorationpipeline/mosaic_detector.py`
- `gpu_engine/native_mosaic/_vendor/lada/utils/image_utils.py`（仅在
  vendored 上层调用点切换，不动 utils 本体以减少 vendor 漂移）

---

## S3：clip 级 batch resize/pad 张量化

承接 S2 的 GPU crop tensor 后，restore 端送入模型前的准备改成 **整 clip
一次性**：

- 收齐一个 clip 的 N 个 crop（已经是 GPU tensor）后：
  - `torch.stack` → (N, H, W, 3) uint8
  - `permute(0, 3, 1, 2).contiguous()` → (N, 3, H, W)
  - `F.interpolate` 整批缩到模型输入尺寸（替代逐帧 cv2 resize）
  - `.to(torch.float16).div_(255.0)` in-place
- pad_batch_with_last：若末尾不足，用 `tensor[-1:].expand(...)` 视图扩展，
  无额外显存拷贝。

### 验收

- 单 clip preprocess wall time 下降至原 1/3 以下。
- 显存峰值不增（用 view/expand，不实体复制）。

### 涉及文件

- `gpu_engine/native_mosaic/_vendor/lada/restorationpipeline/basicvsrpp_mosaic_restorer.py`
- 必要时在 `gpu_engine/native_mosaic/engine.py` 旁加一个轻量
  `_clip_preprocess.py` 集中这段逻辑，避免改 vendored 入口。

---

## S4：去除每帧全流 synchronize

### 现状

`engine.py:_iter_gpu_bgr_frames` 与 `_prepare_restored_nv12` 在 nv12→bgr
每帧后都 `cp.cuda.get_current_stream().synchronize()`，CuPy 与下游 torch
被强行串行。

### 改造

- 引入一条 `torch.cuda.Stream`（命名 `_decode_stream`）专用给 color-convert：
  - CuPy 提交 kernel 时用 `cp.cuda.ExternalStream(decode_stream.cuda_stream)`
  - 结束记录 `event = torch.cuda.Event(); event.record(decode_stream)`
- 下游消费者（detect / restore）在拿到 tensor 时 `event.wait(current_stream)`
  +（如有必要）`tensor.record_stream(current_stream)`，**不再 synchronize**。
- 每个 clip 边界仍可保留一次 `event.synchronize()` 兜底，定位问题更容易。

### 验收

- Nsight Systems 时间线上 decode/color-convert kernel 与 inference kernel
  在不同 stream 上有可见重叠。
- 4K SBS 端到端 FPS 提升（具体数据 S6 给）。

### 涉及文件

- [gpu_engine/native_mosaic/engine.py](gpu_engine/native_mosaic/engine.py)
- [gpu_engine/nv12_kernels.py](gpu_engine/nv12_kernels.py)

---

## S5：VRAM offloader（BlendBuffer D→H pinned）

### 现状

`BlendBuffer` 与 crop 池常驻 GPU，`max_clip_length=180` 显存吃紧；长片
时通过 `empty_cache` 救火。

### 改造

- 新增 `gpu_engine/vram_offload.py`：
  - 后台 daemon thread，按 `torch.cuda.mem_get_info()` 比例阈值
    （默认占用 >80% 触发，<60% 停手）
  - 把 BlendBuffer 内"已生成但下游未消费"的 NV12/BGR tensor `.to('cpu', non_blocking=True)`
    到 pinned host 池，消费时按需 H2D 回 GPU
  - 注册/反注册接口给 BlendBuffer & crop pool
- 关键约束：offload 与 detect/restore 不在同一 stream；H2D 回程预留 1 帧
  lookahead，避免编码线程等待。

### 验收

- 同样素材 `max_clip_length` 可从 180 提到 300 不 OOM。
- 单帧延迟不升高（用 `time.perf_counter` 测 P95）。

### 涉及文件

- 新增 `gpu_engine/vram_offload.py`
- [gpu_engine/native_mosaic/engine.py](gpu_engine/native_mosaic/engine.py)
  注册点

---

## S6：warmup + 端到端基线 profile

### 改造

- 在 `NativeMosaicEngine.__init__` 末尾跑一次 dummy clip（全零 256×256，
  N=clip_length），触发：CuPy kernel JIT、cudnn algo 选择、torch fp16
  buffer 分配。
- 加 `--profile-decode` 隐藏开关，写 `runtime_cache/profile_<ts>.json`：
  - 每段 wall time / GPU 时间 / D2H 次数 / sync 次数
- 出一份基线表（S1–S5 前 vs S1–S5 后），写进本 summary 末尾。

### 涉及文件

- [gpu_engine/native_mosaic/engine.py](gpu_engine/native_mosaic/engine.py)
- 新增 `gpu_engine/_profile.py`

---

## 落地顺序建议

1. **S6 基线先打**（半天）：没有数字后面没法判断收益。
2. **S1 → S2 → S3** 串行落，每步独立提 commit + 基线对比。
3. **S4** 在 S1 之后任意时机落（独立性高）。
4. **S5** 最后落，等 S1–S4 把延迟基线压下来后再做显存扩容更划算。

## 与后续 TRT 阶段（不在本计划内）的衔接

- S2/S3 产出的 GPU batch tensor (N,3,H,W) fp16 normalized 已经是 TRT
  preprocess 子图的天然入口，第二阶段无需再改前段。
- S4 的 stream/event 模型保留给 TRT runner 直接复用。
- S5 的 offloader 与 TRT engine 显存（workspace 自适应）正交，不冲突。
