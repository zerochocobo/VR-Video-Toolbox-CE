# OneClick 解码阶段第二阶段优化计划（A1 + A3，非 TRT）

承接 [summary_20260605_ONECLICK_DECODE_STAGE1_NOTRT_PLAN_CN.md](summary_20260605_ONECLICK_DECODE_STAGE1_NOTRT_PLAN_CN.md)
（已 commit `dd71d9a`）。本阶段聚焦 BasicVSR++ 推理本体在不引入 TensorRT
的前提下的两项调优。预计组合收益 **+20%–45%**，工作量合计 2–3 天，TRT
化所需要的"模型固定形状"前置条件也由本阶段顺带满足。

| Pri | 项目 | 工时 | 风险 |
|---|---|---|---|
| A3 | cudnn/TF32/channels_last/SDPA flash 开关 | ~0.5 天 | 低 |
| A1 | CUDA Graph 捕获 BasicVSR++ 固定形状子图 | ~1.5–2 天 | 中 |

A3 先做（零侵入、立竿见影），A1 在 A3 基础上做（channels_last 已生效时
graph 捕获更稳）。

> **HOTFIX 2026-06-06**：A1（CUDA Graph）与 A3 channels_last 触发了
> Windows native fast-fail（STATUS_STACK_BUFFER_OVERRUN, 0xc0000409），
> Python try/except 抓不到。已把 `VRVT_CUDA_GRAPH` 与 `VRVT_CHANNELS_LAST`
> 默认翻到 `0`、撤掉 `enable_math_sdp(False)` 保留 math fallback。代码本体
> 保留，需要 A/B 时显式 `VRVT_CUDA_GRAPH=1` / `VRVT_CHANNELS_LAST=1` 启用。
> 安全项（cudnn.benchmark / allow_tf32 / matmul TF32 /
> float32_matmul_precision / flash+mem_efficient SDPA）继续默认开启。

---

## A3：cudnn benchmark + TF32 + channels_last + SDPA flash

### 现状

[gpu_engine/native_mosaic/engine.py](gpu_engine/native_mosaic/engine.py)
启动期 `_warmup_native_pipeline` 已经触发 cudnn algo 选择，但没有显式打开
`cudnn.benchmark`、TF32 与 flash SDPA；`BasicvsrppMosaicRestorer.restore`
里张量是 `(B,T,C,H,W)` contiguous，未切换到 channels_last，Blackwell
（sm_120）Tensor Core 的 8×16 矩阵乘吃不满。

### 改造

新增 `gpu_engine/native_mosaic/_torch_tuning.py`（与 `_gpu_ops.py`、
`_profile.py` 同风格的轻量模块），在 `NativeMosaicEngine.__init__`
首句调用 `apply_inference_tuning()`：

```python
def apply_inference_tuning():
    import torch

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    # SDPA flash/mem-efficient kernels for any attention layers
    try:
        torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_mem_efficient=True, enable_math=False,
        )
    except Exception:
        pass
```

环境变量开关：`VRVT_INFERENCE_TUNING=0` 兜底关闭，定位回归用。

把 restoration model 与 detection model 切到 `channels_last`：

```python
# 在 NativeMosaicEngine.__init__ load_models 之后
if self.fp16 and hasattr(self.restoration_model, "model"):
    self.restoration_model.model.to(memory_format=torch.channels_last)
```

`BasicvsrppMosaicRestorer.restore` 内 `inference_view` 在送入 `self.model`
前 reshape 成 `(B*T, C, H, W)`（去掉 time 维），调一次
`.to(memory_format=torch.channels_last)` 再 reshape 回去；如果 model
forward 强依赖 `(B,T,C,H,W)` 原 layout，则只对最内层 conv 子模块走
channels_last（fallback 路径：仅设置全局 cudnn flag 即可拿到大半收益）。

### 验收

- 4K SBS 单 segment：`profile.sections["model.forward"].cuda_ms` 对比下降
  ≥10%。
- 输出帧 PSNR 与基线一致（cudnn benchmark / TF32 不改 fp16 精度）。
- 关 `VRVT_INFERENCE_TUNING=0` 行为回到当前 master。

### 涉及文件

- 新增 `gpu_engine/native_mosaic/_torch_tuning.py`
- [gpu_engine/native_mosaic/engine.py](gpu_engine/native_mosaic/engine.py)
  `__init__` 顶部 + `restoration_model` 切 memory_format
- `gpu_engine/native_mosaic/_vendor/lada/restorationpipeline/basicvsrpp_mosaic_restorer.py`
  `restore` 内 layout 切换（仅当 channels_last 在该 forward 上确实生效时）

---

## A1：CUDA Graph 捕获 BasicVSR++ 固定形状子图

### 现状

`BasicvsrppMosaicRestorer.restore` 每个 clip 一次 `self.model(inputs=...)`，
输入形状 `(1, T, 3, 256, 256)` fp16。`max_clip_length` 上限固定（默认 180）
但实际每 clip 通常正好是 `max_clip_length`，仅文件末尾 1 个 clip 是余数。

BasicVSR++ 内部 deform_align / SPyNet / propagate / upsample 是大量小算子
（>1000 kernel launch / clip），eager 模式下 launch overhead 占据可观比例。
当输入形状固定时，`torch.cuda.CUDAGraph` 一次 capture 后 replay 可以把这
~1000 个 launch 折合成 1 个 graph launch。

### 改造

新增 `gpu_engine/native_mosaic/_cuda_graph_runner.py`：

```python
class CudaGraphRunner:
    """Capture model forward once per input shape, replay on subsequent calls."""

    def __init__(self, model, device, *, warmup_iters: int = 3, enabled: bool = True):
        self.model = model
        self.device = device
        self.warmup_iters = warmup_iters
        self.enabled = enabled
        self._cache: dict[tuple, _Entry] = {}

    def __call__(self, inputs):
        if not self.enabled:
            return self.model(inputs=inputs)
        key = (tuple(inputs.shape), inputs.dtype, inputs.stride())
        entry = self._cache.get(key)
        if entry is None:
            entry = self._capture(inputs)
            self._cache[key] = entry
        entry.static_input.copy_(inputs, non_blocking=True)
        entry.graph.replay()
        return entry.static_output.clone()

    def _capture(self, inputs):
        import torch

        static_input = torch.empty_like(inputs)
        static_input.copy_(inputs)

        # Warmup on a side stream so capture sees stable cuDNN algo choices.
        s = torch.cuda.Stream(device=self.device)
        s.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(s):
            for _ in range(self.warmup_iters):
                _ = self.model(inputs=static_input)
        torch.cuda.current_stream(self.device).wait_stream(s)
        torch.cuda.synchronize(self.device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = self.model(inputs=static_input)
        return _Entry(graph=graph, static_input=static_input, static_output=static_output)
```

接入点在 `BasicvsrppMosaicRestorer`：

```python
def __init__(self, model, device, fp16):
    ...
    from gpu_engine.native_mosaic._cuda_graph_runner import CudaGraphRunner

    self._graph_runner = CudaGraphRunner(
        model, device,
        enabled=os.environ.get("VRVT_CUDA_GRAPH", "1").strip().lower() not in {"0", "false", "no", "off"},
    )

def restore(self, video, max_frames=-1):
    ...
    if max_frames > 0:
        for i in range(0, inference_view.shape[1], max_frames):
            output = self._graph_runner(inference_view[:, i:i + max_frames])
            result.append(output)
    else:
        result = self._graph_runner(inference_view)
```

#### 关键约束（开发务必落实）

1. **形状缓存**：cache key 必须包含 `shape + dtype + stride`，否则末尾 clip
   余数或 channels_last 切换会用错 graph。最多缓存 2 个 entry（`max_clip_length`
   主形状 + 余数形状），第三个出现时 evict 最早。
2. **stride 对齐**：`inference_view` 在 `restore` 内已经 `.contiguous()`，但
   channels_last 后 stride 变化，capture 前要再 contiguous 一次。
3. **side-stream warmup**：cudnn 在 capture 期间禁止 algo 选择，必须先 warmup
   3 次让 algo 固化（见 `_capture` 实现）。
4. **`static_output.clone()` 必须有**：graph replay 复用同一块输出显存，下一
   次 replay 会覆盖；外层消费者拿引用会读到错数据。
5. **回退路径**：capture 失败（OOM、deform_conv 不支持等）`__call__` 必须
   `try/except` 兜底直接 eager 调用，并把该形状从 cache 标记为
   `enabled=False`，下次同形状直接 eager。
6. **inference_mode 互斥**：`torch.cuda.graph` 上下文与 `torch.inference_mode()`
   嵌套有时报错，capture 时用 `torch.no_grad()` 替代；replay 时仍可以在
   `inference_mode` 内。
7. **首 clip 额外耗时**：capture 阶段 +3 次 warmup forward = 单 clip 推理时间
   的 ~4 倍，启动延迟会延长 0.5–2s。`_warmup_native_pipeline` 顺手用
   `(1, max_clip_length, 3, 256, 256)` 跑一次让首文件不付这个代价。

### 验收

- 4K SBS 同 segment 重跑：`profile.sections["restorer.start"]` 首 clip
  耗时上升符合预期；后续 clip `wall_ms` 下降 **≥15%**。
- 内存：缓存 ≤2 个 entry 时 VRAM 增量 ≤ `2 × clip 显存`，4K 下 <1.5GB。
- 关 `VRVT_CUDA_GRAPH=0` 行为回到 A3-only 基线。
- 抽 20 帧 PSNR / SSIM 与 eager 路径完全一致（capture/replay 不改语义）。
- 末尾余数 clip 与正常 clip 输出像素级一致（验证 cache key 正确）。

### 涉及文件

- 新增 `gpu_engine/native_mosaic/_cuda_graph_runner.py`
- `gpu_engine/native_mosaic/_vendor/lada/restorationpipeline/basicvsrpp_mosaic_restorer.py`
- [gpu_engine/native_mosaic/engine.py](gpu_engine/native_mosaic/engine.py)
  `_warmup_native_pipeline` 调一次 `max_clip_length` 形状

### 已知风险与对策

- **deform_conv2d 在 graph capture 模式下报错**：torchvision 部分版本不支持，
  退路是把 deform_align 子模块单独 graph、其余 propagate 走 eager；或直接
  禁用 graph（兜底已就位）。
- **多线程并发 capture 触发死锁**：`FrameRestorer` 当前 4 线程里只有
  `primary/secondary_restore_loop` 调 `restore`；CudaGraphRunner 需要在
  实例级加 `threading.Lock` 保护 capture 阶段（replay 不需要锁，static
  buffer 由 lock 期间分配后只读）。
- **TRT 化时如何衔接**：A1 的 CudaGraphRunner 是 model 的薄包装，TRT 化
  后只需把 inner model 替换成 TRT module，CudaGraphRunner 继续工作。

---

## 落地顺序与基线

1. 拉一份当前 master 的基线 profile（`--profile-decode` 跑 1 个 4K SBS
   segment），记下 `model.forward` cuda_ms / 整段 fps。
2. **A3** 提一个 commit，重跑同 segment，对比记录。
3. **A1** 再提一个 commit，对比记录。
4. 三组数据写回本 summary 末尾的"实测对比"小节。

## 与后续 TRT 阶段（不在本计划内）的衔接

- A3 的 cudnn benchmark + channels_last 在 TRT 化后仍然生效（TRT 内部
  也用 cudnn 选择 algo，但 cudnn 全局 flag 影响 fallback 路径）。
- A1 的 CudaGraphRunner 在 TRT 化后保留：TRT engine 同样从 cudagraph
  replay 受益（jasna 的 `use_python_runtime=False` 路径就是这种组合）。
- 本阶段验证过的"固定形状 + warmup → capture → replay"模式，可直接复用
  到 TRT runner 上。

---

## 实现状态（2026-06-06）

### 已落地

- A3：
  - 新增 `gpu_engine/native_mosaic/_torch_tuning.py`；
  - `NativeMosaicEngine.__init__` 入口调用 `apply_inference_tuning()`；
  - 支持 `VRVT_INFERENCE_TUNING=0` 关闭 cudnn benchmark / TF32 / SDPA / channels_last；
  - restoration model 与 detection model 尝试 `channels_last`，失败自动保留全局 flag 路径；
  - `BasicvsrppMosaicRestorer` 输入 tensor 支持 5D BTCHW 的 inner-frame channels_last layout。
- A1：
  - 新增 `gpu_engine/native_mosaic/_cuda_graph_runner.py`；
  - cache key 包含 `shape + dtype + stride + device`；
  - cache 最多 2 个 entry，LRU 淘汰；
  - capture 阶段使用 side stream warmup 3 次；
  - capture 内显式 `torch.inference_mode(False)` + `torch.no_grad()`；
  - replay 返回 `static_output.clone()`；
  - capture / replay 失败后同 key 标记 eager fallback；
  - `CudaGraphRunner` 使用实例 lock 保护 capture、static input copy、replay，避免多线程共用 static buffer；
  - `_warmup_native_pipeline` 调用 `warmup_graph()`，默认捕获 `(1,180,3,256,256)`，可用 `VRVT_CUDA_GRAPH_WARMUP_FRAMES` 调整；
  - `BasicvsrppMosaicRestorer.restore` 外层恢复为 `torch.inference_mode()`，只在 graph capture 内关闭 inference mode，避免 A3/A1 关闭时出现性能回退。
- profile：
  - `gpu_engine/_profile.py` 增加 active profile；
  - BasicVSR++ forward 包在 `profile.sections["model.forward"]`，可记录 wall/cuda time；
  - `restore_file` profile metadata 增加 `torch_tuning` 状态。

### 已验证

- `python -m py_compile gpu_engine/native_mosaic/_torch_tuning.py gpu_engine/native_mosaic/_cuda_graph_runner.py gpu_engine/native_mosaic/_vendor/lada/restorationpipeline/basicvsrpp_mosaic_restorer.py gpu_engine/native_mosaic/engine.py gpu_engine/_profile.py tests/test_native_mosaic_profile.py`：通过。
- `python -m pytest tests/test_native_mosaic_profile.py tests/test_engine_runner.py -q`：11 passed。
- 轻量 engine 初始化 smoke：
  - `VRVT_NATIVE_WARMUP=0 VRVT_INFERENCE_TUNING=0 VRVT_CUDA_GRAPH=0`：`NativeMosaicEngine True`。
  - `VRVT_NATIVE_WARMUP=1 VRVT_INFERENCE_TUNING=1 VRVT_CUDA_GRAPH=1 VRVT_CUDA_GRAPH_WARMUP_FRAMES=1`：`NativeMosaicEngine True None`。

### 实测对比

| 组别 | profile | model.forward cuda_ms | segment fps | PSNR/SSIM | 状态 |
|---|---|---:|---:|---|---|
| baseline | `VRVT_INFERENCE_TUNING=0 VRVT_CUDA_GRAPH=0` | 待测 | 待测 | 待测 | 本轮完整跑 `videos/2_2.mp4` 超过 15 分钟被中断，未生成有效 profile |
| A3 | `VRVT_INFERENCE_TUNING=1 VRVT_CUDA_GRAPH=0` | 待测 | 待测 | 待测 | 待同素材重跑 |
| A1 | `VRVT_INFERENCE_TUNING=1 VRVT_CUDA_GRAPH=1` | 待测 | 待测 | 待测 | 待同素材重跑 |

> 注意：本轮没有伪造性能数据。完整验收仍需在同一 4K SBS mosaic segment 上跑三组 profile，并抽 20 帧比较 eager / graph 输出 PSNR、SSIM。

---

## HOTFIX 后隔离 smoke（2026-06-06）

- 已确认 `e09c798`：
  - `VRVT_CUDA_GRAPH` 默认 `0`；
  - `VRVT_CHANNELS_LAST` 默认 `0`；
  - 保留 SDPA math fallback，不再 `enable_math_sdp(False)`。
- 默认路径：
  - engine warmup smoke 通过；
  - profile metadata 显示 `restoration_channels_last=False`、`detection_channels_last=False`、`_warmup_error=None`。
- 单项隔离：
  - `VRVT_CUDA_GRAPH=0 VRVT_CHANNELS_LAST=1`，8 帧 warmup 通过；
  - `VRVT_CUDA_GRAPH=1 VRVT_CHANNELS_LAST=0`，8 帧 graph warmup 通过；
  - `1×8×3×64×64` synthetic restore 被 BasicVSR++ 自身断言拒绝（内部下采样后低于 64），不是 native fast-fail；
  - `VRVT_CUDA_GRAPH=0 VRVT_CHANNELS_LAST=1`，10 次 `1×8×3×256×256` synthetic restore 通过；
  - `VRVT_CUDA_GRAPH=1 VRVT_CHANNELS_LAST=0`，10 次 `1×8×3×256×256` synthetic restore 通过；
  - `VRVT_CUDA_GRAPH=1 VRVT_CHANNELS_LAST=1`，10 次 `1×8×3×256×256` synthetic restore 通过。
- 当前结论：
  - 小规模 BasicVSR++ forward/replay 未复现 `0xc0000409`；
  - 不能据此翻默认值，仍需真实 decode segment 与更长循环验证；
  - 翻默认前建议至少跑 200 次 `1×8×3×256×256` synthetic clip，并分别跑真实 segment 的 channels_last-only / graph-only。
