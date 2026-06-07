# tool_2dvr 4K@60fps 性能改造计划

日期: 2026-06-05
作者: 研究阶段交付
目标: **DA3-Small + 默认 soft_shift 羽化，4K 输入达到 ≥60fps 端到端转换**
范围: `tool_2dvr/logic.py`、`tool_2dvr/_vendor/da3/depth_anything_3/`（必要时深改）
不在范围: 视频修补模式、`flat3d` 以外投影的 FOV 调优

---

## 一、现状链路与瓶颈

```
ffmpeg 解码 (CPU rgb24) ─pipe─▶ Python read
  └▶ DA3.inference(list[np.ndarray])
        ├ InputProcessor (PIL ↔ cv2 ↔ ToTensor ↔ Normalize, CPU sequential)
        ├ Backbone DinoV2-vits  (504×W, bf16 autocast) ✅
        ├ DPT head
        ├ cam_dec (相机位姿)                 ❌ 单帧深度用不到
        ├ sky mask + torch.quantile          ❌ 同步点
        └ OutputProcessor
  └▶ predict_batch: PIL.Image.resize depth 504→4K (CPU 单线程)   ❌
  └▶ TorchStereoRenderer (4K)
        ├ _normalize_near: Python for + quantile×batch  ❌ 同步
        ├ forward warp + soft_shift                    ✅
        └ SBS uint8 → .cpu().numpy()                   ❌ 7680×2160 来回
  └─pipe─▶ ffmpeg hevc_nvenc -preset p7 -cq 18   ❌ 4K60 p7 大概率扛不住
```

主要损耗点（预估贡献，单位 ms/帧，4K 输入，RTX 5060 Ti）：

| 模块 | 现状 | 备注 |
| --- | --- | --- |
| ffmpeg CPU 解码 4K HEVC | ~10–15 | 单实例接近极限 |
| InputProcessor (PIL pipeline) | ~30–60 | sequential, batch=8 串行 |
| DA3 forward (含 cam_dec/sky) | ~15–25 | cam_dec/sky 占其中 30–50% |
| Depth 504→4K PIL.resize | ~10–20 | CPU 单线程 |
| TorchStereoRenderer | ~10–15 | normalize_near 同步占大头 |
| SBS GPU→CPU + pipe 写 | ~10 | 单线程阻塞 IO |
| hevc_nvenc p7 cq 18 | ~15–25 | preset 过高 |
| **合计** | **~100–160ms** | **即 6–10fps**（与现场 log 吻合） |

---

## 二、改造分级（按性价比排序）

### P0 — 最小动静拿大头收益（目标：30–50fps）

**P0-1 DA3 推理路径裁剪**
- 文件：`tool_2dvr/_vendor/da3/depth_anything_3/api.py`
- 改动：新增 `inference_depth_only(images: list[np.ndarray]) -> torch.Tensor`
  - 跳过 `cam_dec` 整段（不构造 `pose_enc`、不解算 c2w/ixt）
  - 跳过 `_process_mono_sky_estimation`（去 `torch.quantile` 同步）
  - 跳过 `_align_to_input_extrinsics_intrinsics`（无 extrinsics 输入路径根本不需要）
  - 跳过 `OutputProcessor` 里 numpy 化和 `processed_images` 回传
  - 直接返回 backbone+head 的 `depth` tensor（GPU 上，shape `(B, H, W)`）
- 文件：`tool_2dvr/_vendor/da3/depth_anything_3/model/da3.py`
  - `forward` 增加 `skip_camera: bool = False`、`skip_sky: bool = False` 参数
  - 上游传 True 时直接 return depth-only Dict
- 调用点改动：`tool_2dvr/logic.py::DA3DepthEstimator.predict_batch` 改调 `inference_depth_only`

**P0-2 Depth 上采样移到 GPU**
- 文件：`tool_2dvr/logic.py`
- 改动：
  - `DA3DepthEstimator.predict_batch` 不再返回 numpy 上采样到 4K 的 depth；改为返回 `torch.Tensor` （GPU，504 分辨率，bf16/float32）
  - `TorchStereoRenderer.render_batch` 入参 `depths` 接受 GPU tensor；在内部用 `F.interpolate(..., mode='bilinear', align_corners=False)` 上采样到 `(src_h, src_w)`
  - CPU fallback 路径保留，但加 deprecation note
- 影响：去掉 PIL CPU 上采样 + numpy↔torch 来回

**P0-3 编码器 preset 降档**
- 文件：`tool_2dvr/logic.py::convert_2d_to_vr` 的 `encode_cmd`
- 改动：
  - `-preset p7` → `-preset p4`
  - `-cq 18` → `-cq 20`
  - 增加 `-tune hq -rc vbr -multipass fullres -spatial_aq 1 -temporal_aq 1`
  - 可加 `-b:v 0 -maxrate 80M -bufsize 160M` 防止码率失控
- 视觉差异在 4K60 输入上肉眼几乎不可见

**P0 验收**
- 4K H264 30fps 输入：端到端 ≥30fps（CPU 解码仍是瓶颈）
- 4K H265 60fps 输入：端到端 ≥40fps
- depth_eye_debug 图像不应有明显劣化
- 不破坏 `flat3d/hequirect/fisheye` 三种投影
- 不破坏 `hole_fill_mode` 的其它分支

---

### P1 — 中等改造（目标：≥60fps）

**P1-1 `_normalize_near` 去 Python 循环**
- 文件：`tool_2dvr/logic.py::TorchStereoRenderer._normalize_near`
- 改动：
  - `inv_depth.flatten(1)` 后用 `torch.quantile(flat, q=torch.tensor([0.05,0.95]), dim=1)` 一次性算 batch
  - 移除 `for idx in range(depth.shape[0])` 循环
  - 注意 `quantile` 对 batch 维 mask 不一致的处理：用 `torch.nan_to_num` 把无效值置 NaN，`quantile` 自动忽略；或者预先做 `valid` 加权后用 `kthvalue`
- 收益：去 batch-1 次 GPU 同步，整体掉 ~3–5ms/帧

**P1-2 NVDEC 硬解 + GPU 色彩转换**
- 文件：`tool_2dvr/logic.py::convert_2d_to_vr` 的 `decode_cmd`
- 改动：
  - 增加 `-hwaccel cuda -hwaccel_output_format cuda`
  - 用 `scale_cuda=format=yuv420p` 后 `hwdownload,format=nv12` 再 `format=rgb24`，避免 CPU 解码
  - 若 NVDEC 不支持当前编码（如 AV1 在老卡），自动 fallback 到 CPU 路径
- 检测：调用 `ffmpeg -hide_banner -hwaccels` 一次性缓存能力
- 收益：CPU 占用大幅下降，4K60 解码不再瓶颈

**P1-3 InputProcessor 旁路 / GPU 化**
- 文件：`tool_2dvr/_vendor/da3/depth_anything_3/utils/io/input_processor.py` 不动，新增 `tool_2dvr/_vendor/da3/depth_anything_3/utils/io/input_processor_gpu.py`
- 实现：
  ```
  def gpu_preprocess(frames_uint8_gpu: torch.Tensor, target_res: int = 504) -> torch.Tensor:
      # frames: (B, H, W, 3) uint8 on CUDA
      x = frames.permute(0,3,1,2).float().div_(255.0)
      x = F.interpolate(x, size=auto_size(H, W, target_res, patch=14),
                        mode='bilinear', align_corners=False)
      x.sub_(mean).div_(std)  # broadcast
      return x.unsqueeze(0)  # (1, B, 3, H, W) 符合 DA3 输入
  ```
- `DA3DepthEstimator.predict_batch` 直接传 GPU tensor，绕过 `InputProcessor.__call__`
- 仅在 `inference_depth_only` 路径启用；旧 `inference` 不动
- 收益：去 CPU PIL pipeline，batch=8 大约掉 30–60ms/帧

**P1 验收**
- 4K60 HEVC 输入：端到端 ≥60fps
- GPU 利用率（nvidia-smi）应 ≥85%
- CPU 占用应 <50%（一颗核心以下）

---

### P2 — 工程化（让 60fps 稳定不掉帧，可选）

**P2-1 三段流水线异步化**
- 文件：`tool_2dvr/logic.py::convert_2d_to_vr` 主循环
- 改动：
  - 引入 `reader_thread`：从 `decode_proc.stdout` 读 frame，push 到 `queue.Queue(maxsize=4)`
  - 主线程：从 reader queue pop，组 batch，跑 DA3+渲染，push SBS batch 到 `writer_queue`
  - `writer_thread`：从 writer queue pop，写 `encode_proc.stdin`
  - 任何线程异常立刻 `handle.kill()` 取消
- 收益：解决 pipe 阻塞导致的 GPU 空转

**P2-2 SBS 输出零拷贝**
- 当前 `.cpu().numpy()` 是阻塞同步拷贝。改为：
  - 用两个 pinned-memory CPU buffer 轮转
  - `tensor.to(cpu_buf, non_blocking=True)` 后用 CUDA event 标记
  - writer thread 检查 event 完成后再 `tobytes()` 写 pipe
- 收益：渲染和写编码重叠

**P2-3（可选）torch.compile / TensorRT export DA3-Small backbone**
- 仅 vits 主干 + DPT head，固定 504×N×14 输入
- 用 `torch.compile(mode='reduce-overhead', fullgraph=False)`
- 若仍不够，导 ONNX → trtexec → 用 `torch_tensorrt` 包装
- 收益：DA3 forward 再降 30–50%，封顶 90fps+

---

## 三、提交顺序与里程碑

| 里程碑 | 包含项 | 验收 | 预计工时 |
| --- | --- | --- | --- |
| M1 | P0-1, P0-2, P0-3 | 4K30 ≥30fps；视觉无可见劣化 | 1.5 天 |
| M2 | P1-1, P1-2 | 4K60 ≥45fps，CPU <60% | 1.5 天 |
| M3 | P1-3 | 4K60 ≥60fps | 1 天 |
| M4（可选）| P2-1, P2-2 | 60fps 稳定无毛刺 | 1 天 |
| M5（可选）| P2-3 | 4K60 ≥80fps | 2 天 |

---

## 四、风险与回归测试清单

- DA3 `inference_depth_only` 必须保证 depth 数值与 `inference` 完全一致（diff < 1e-4），用一段静态图做单元对比
- 三种投影 `flat3d / hequirect / fisheye` 各跑 5s clip 回归
- 所有 `hole_fill_mode`（soft_shift / shift_fill / background / inpaint / lama / none）回归
- `start_time/end_time` 裁剪、音频 `-map 0:a?` mux 回归
- CUDA OOM 回退路径（`_predict_depths_resilient`、`_render_with_depths` 的二分递归）必须保留
- NVDEC fallback 路径：故意喂一段 AV1 文件验证回退到 CPU 解码

---

## 五、不做什么（明确范围）

- 不替换 DA3 模型（仍是 Small）
- 不引入新的视频解码库（PyAV/VPF）—— 先用 ffmpeg `-hwaccel cuda` 走通
- 不改变默认 `hole_fill_mode=soft_shift` 行为
- 不动视频修补后处理路径
- 不调整 max_disparity_pixels 等视觉参数

---

## 六、关键代码位置速查

| 关注点 | 文件 | 行 |
| --- | --- | --- |
| 主转换流程 | `tool_2dvr/logic.py` | 1249 (`convert_2d_to_vr`) |
| Depth 估计 + 上采样 | `tool_2dvr/logic.py` | 395 (`predict_batch`) |
| Torch 渲染器 | `tool_2dvr/logic.py` | 959 (`TorchStereoRenderer`) |
| `_normalize_near` for-loop | `tool_2dvr/logic.py` | 995 |
| 编码命令 | `tool_2dvr/logic.py` | 1321 |
| DA3 inference 入口 | `_vendor/da3/depth_anything_3/api.py` | 101 |
| DA3 forward 主体 | `_vendor/da3/depth_anything_3/model/da3.py` | 100 |
| InputProcessor | `_vendor/da3/depth_anything_3/utils/io/input_processor.py` | 65 |
