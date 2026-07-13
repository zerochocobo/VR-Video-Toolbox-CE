# OneClick GPU 显存优化方案（专家评审稿）

## 1. 背景与问题判断

RTX 5070 12GB 用户在 OneClick 流程中发生进程闪退。此前已在最终合并前释放 Native restoration engine、YOLO detector、Torch/CuPy 缓存，但用户复测后仍然崩溃。

用户进一步修改后成功完成任务，其核心调整为：

1. coarse/fine 扫描完成后提前释放 YOLO detector。
2. GPU split 完全返回后、Jasna/Lada 启动前，再次清理 CuPy/Torch/Python 缓存。

结合当前代码生命周期，真正的显存峰值很可能出现在“扫描或拆分完成后启动恢复模型”这一阶段，而不只是最终文件合并：

- YOLO detector 作为全局对象保持缓存，约占用 2 GiB 显存。
- GPU split 结束时，函数内部最后一帧、编码输入环和局部 CuPy 数组可能仍有活动引用。
- 随后加载 Jasna/Lada/Native restoration engine，形成模型和编解码资源重叠。
- 原生 CUDA、PyNv 或恢复引擎在显存不足时可能直接终止进程，无法进入 Python fallback。

用户方案方向有效，但如果直接在多个业务函数中散布 `release_detector()`、`gc.collect()`、`torch.cuda.empty_cache()` 和 `runtime.free_memory_pool()`，会引入模型重复加载、无意初始化 CUDA context、并发释放以及异常分支遗漏等风险。

## 2. 优化目标

1. 在恢复、paste 和 merge 开始前降低显存峰值。
2. 不在纯 Jasna/FFmpeg 路径中因清理操作意外初始化 Torch CUDA context。
3. 不在扫描仍运行时释放 detector。
4. 避免同一连续扫描会话内反复卸载和加载 YOLO。
5. 正常、NO_MOSAIC、异常和取消路径使用一致的资源生命周期。
6. 保留现有最终 merge 前深度清理，作为最后一道显存保护。
7. 增加低开销诊断信息，以便确定真实峰值阶段。

## 3. 实施方案

### 3.1 建立统一的阶段边界显存清理接口

在 `one_click/logic.py` 中新增统一内部接口，例如：

```python
_cleanup_gpu_stage(
    release_detector=False,
    release_restore_engine=False,
    collect_python=True,
    log_callback=None,
    stage=None,
)
```

固定执行顺序：

1. 按需解除 detector 或 restoration engine 的全局/缓存引用。
2. 如果确实释放了对象或调用方明确要求，则执行 `gc.collect()`。
3. 清理 CuPy memory pool。
4. 仅当 Torch 已导入且 `torch.cuda.is_initialized()` 为真时调用 `torch.cuda.empty_cache()`。
5. 记录清理阶段、释放对象及清理前后的缓存摘要。

接口必须采用 best-effort 语义：清理或诊断失败时记录日志，但不能中断视频处理主流程。

### 3.2 避免清理动作初始化 Torch CUDA

不得为了执行 `empty_cache()` 无条件 `import torch` 并调用 `torch.cuda.is_available()`，因为后者在某些环境中可能触发 CUDA runtime/context 初始化。

建议逻辑：

```python
torch_mod = sys.modules.get("torch")
if torch_mod is not None:
    cuda = getattr(torch_mod, "cuda", None)
    if cuda is not None and cuda.is_initialized():
        cuda.empty_cache()
```

纯 Jasna/FFmpeg 路径如果此前没有加载 Torch，则完全跳过 Torch CUDA 清理。

### 3.3 为 detector 建立扫描会话生命周期

扫描期间继续复用 detector，扫描结果完全转换为 CPU 数据后再释放：

- coarse source scan：interval/detection 数据保存完成后释放。
- 普通 pre-extract scan：segment 对齐并保存后、cut/restore 开始前释放。
- paired fine scan：左右眼 fine segment 均生成并保存后、extract/restore pipeline 开始前释放。

不建议在左右眼属于同一连续扫描会话时中途释放，否则会产生重复加载。释放点应位于逻辑扫描会话的末端，而不是简单地位于每次 detector 函数调用之后。

需要用 `try/finally` 或显式会话控制覆盖：

- 正常发现 segment。
- NO_MOSAIC。
- 扫描异常。
- 用户取消。

如果未来扫描改为后台并发，释放动作必须在扫描线程 join/完成之后执行。不能仅将全局 `_DETECTOR` 设为 `None` 就假设显存已释放，因为活动线程的局部引用仍会持有模型。

### 3.4 在 GPU split 完全返回后执行阶段清理

当前 `gpu_engine.files` 已在内部 `finally` 中调用 `runtime.free_memory_pool()`，但该调用发生时函数局部变量、最后一帧、encoder state 或 encode ring 仍可能存在活动引用。函数完全返回后，这些引用才会自然解除。

因此应在 OneClick wrapper 的 split 调用返回后、Jasna/Lada 启动前，再执行一次阶段清理：

```text
GPU split 完全返回
    -> 解除已经离开作用域的临时对象
    -> gc.collect（按需）
    -> CuPy pool 清理
    -> 启动恢复引擎
```

覆盖路径：

- SBS `split_video_dual`。
- SBS `split_video_dual_fisheye`。
- 单眼 `split_video`。
- 单眼 `split_video_fisheye`。
- source-scan interval 的相同拆分路径。

该阶段默认不释放 restoration engine，因为恢复任务尚未开始；也不主动初始化 Torch。

### 3.5 在恢复启动前增加 detector 兜底释放

在进入 Jasna/Lada/Native restoration 前增加轻量保护：

- detector 已加载且当前没有扫描使用者时释放。
- detector 已释放时不重复执行昂贵的 `gc.collect()`。
- 如果检测到扫描任务仍然活动，记录警告并跳过强制释放。

这用于防止某个异常分支、未来重构或新增调用路径遗漏扫描后的显存释放。

为了可靠判断是否仍有扫描使用者，建议在 `utils/mosaic_prescan.py` 中维护受现有 detector lock 保护的活动扫描计数或扫描会话计数，而不是仅检查 `_DETECTOR is not None`。

### 3.6 保留并统一最终 paste/merge 前深度清理

现有 `_release_gpu_models_before_paste()` 迁移到统一接口，并在以下阶段继续执行深度清理：

- segment paste 前。
- paired rect paste 前。
- 普通 SBS merge 前。
- fisheye-to-VR merge 前。
- source-scan GPU timeline merge 前。

深度清理包括：

- detector。
- Native restoration engine。
- Python unreachable objects。
- CuPy memory pool。
- 已初始化的 Torch CUDA cache。

快速 HEVC bitstream concat 本身不依赖大量 GPU 显存，但在进入 Stage 4 时统一释放已无用途的恢复模型，可以降低 fallback 到 GPU timeline merge 时的风险。

### 3.7 避免重复释放与性能退化

统一接口需要依据真实状态决定是否执行重操作：

- detector 不存在时不重复触发 detector GC。
- Native engine 未缓存时不导入或初始化模型模块。
- Torch CUDA 未初始化时跳过。
- CuPy pool 已为空时避免不必要的同步或重复清理。
- 同一 stage 使用标识避免连续重复执行深度清理。

预计性能影响：

- 阶段边界可能增加少量 GC 停顿。
- source-scan coarse detector 释放后，后续 fine scan 可能重新加载一次 detector，这是用加载时间换取 12GB 显卡稳定性。
- 同一 paired fine scan 的左右眼检测应继续共享一次 detector 加载，避免每只眼分别重载。

## 4. 诊断与可观测性

增加低频显存阶段日志，建议记录：

- coarse/fine scan 完成后。
- split 返回后。
- restoration 启动前。
- paste/merge 启动前。

日志字段：

- stage 名称。
- detector 是否缓存、活动扫描计数。
- Native restoration engine 是否缓存。
- Torch allocated/reserved，仅在 Torch CUDA 已初始化时读取。
- CuPy pool used/total，仅在 CuPy 已加载时读取。
- 本次实际释放了哪些资源。

不得为了采集日志导入 Torch/CuPy 或初始化 CUDA。诊断只能读取已经加载模块的状态。

## 5. 风险控制

### 5.1 正确性风险

- 扫描运行期间释放 detector：通过活动扫描计数和完成后释放规避。
- 异常/取消路径泄漏：通过 `try/finally` 和统一会话生命周期规避。
- 清理异常中断处理：所有清理操作采用 best-effort 并单独捕获异常。

### 5.2 性能风险

- detector 重载：只在扫描会话末端释放，不在每个 detector batch 后释放。
- GC 停顿：只在释放大对象或阶段切换时调用，不逐 segment/逐帧调用。
- allocator 抖动：同一处理阶段内保留必要缓存，只在模型类型切换或编解码阶段切换时清理。

### 5.3 显存风险

- Torch 清理意外创建 context：只对已经初始化的 Torch CUDA 执行。
- `empty_cache()` 误认为可释放活动对象：先解除全局引用并 GC，再清 allocator cache。
- split 内部清理过早：保留内部清理，同时在 wrapper 完全返回后再清理一次已失去引用的缓存。

## 6. 测试计划

新增或调整测试：

1. 清理函数不会导入 Torch，也不会初始化未初始化的 Torch CUDA。
2. detector 正常扫描后释放。
3. NO_MOSAIC、扫描异常和取消路径均正确减少活动计数并释放。
4. paired fine scan 左右眼之间不会提前释放 detector。
5. 后台扫描未完成时拒绝强制释放。
6. GPU split wrapper 返回后才调用阶段清理。
7. Jasna、NativeGPU、普通 SBS、fisheye、单眼和 source-scan 路径均覆盖。
8. merge 前释放 Native engine 和 detector。
9. 清理函数任一子步骤异常时，主处理流程仍继续。
10. 现有 OneClick、source-scan、segment paste、SBS concat 和 GPU fallback 测试全部通过。

## 7. 建议实施顺序

### 第一阶段：生命周期与安全清理

1. 实现统一阶段清理接口。
2. 实现 detector 活动扫描计数/会话管理。
3. 将 coarse/fine scan、split 后、restore 前和 merge 前接入统一接口。
4. 补充异常与取消路径测试。

### 第二阶段：诊断与性能优化

1. 增加不触发 CUDA 初始化的显存阶段日志。
2. 根据日志识别重复清理和不必要的 detector 重载。
3. 调整阶段边界，使 12GB 显卡稳定性和处理速度取得平衡。

## 8. 评审重点

请专家重点评审：

1. detector 活动扫描计数是否足以应对当前及未来并发扫描。
2. `gpu_engine.files` 内部清理与 OneClick wrapper 返回后清理的职责边界。
3. 如何可靠判断 Native engine 是否已缓存，而不因检查动作触发模块初始化。
4. Windows 下 Torch、CuPy、PyNv 使用不同 CUDA context/allocator 时，清理顺序是否需要额外设备同步。
5. Jasna 子进程启动前，父进程是否还存在无法由 CuPy/Torch cache API释放的 NVDEC/NVENC context。
6. 是否需要为 12GB 及以下显卡提供更保守的自动显存策略，而不是对所有显卡采用相同清理频率。

## 9. 评审后实施范围调整

专家评审建议缩小首轮修改范围，并将诊断测量提前。最终采纳以下调整：

1. 不引入 detector 活动扫描计数。当前 OneClick detector 扫描均为同步顺序执行，后台 producer 仅负责 extract；计数器会引入异常漏减和永久拒绝释放的新风险。
2. 不迁移现有 paste/merge 前已经验证工作的清理调用点，保持较小的修改和回滚范围。
3. 不增加 stage 去重状态、CuPy 空池检查或按显存容量分级策略。
4. 所有显卡采用一致的清理行为，降低测试矩阵复杂度。
5. Torch、CuPy 和 Native engine 的诊断与清理只能读取 `sys.modules` 中已经加载的模块，不能为了日志或清理主动初始化 CUDA。
6. 保留两层 split 清理：`gpu_engine.files` 内部 `finally` 负责函数内部资源，OneClick wrapper 返回后负责回收函数退出后才解除引用的 PyNv/CuPy 对象。

实施顺序调整为：

- Phase 0：获取显存基线和阶段曲线。
- Phase 1：扫描结束后释放 detector，并在 GPU split wrapper 返回后执行安全清理。
- Phase 2：依据实测数据处理最终 paste 的主要显存峰值。

## 10. Phase 0 基线测量

测试条件：16GB GPU 使用独立压舱进程制造显存压力，`nvidia-smi` 约每 0.5 秒记录一次设备显存；测试片源为 8192×4096、P016/10bit、约 59.94fps 的 SBS 视频。

基线 process log 运行时间为 2026-07-13 12:21:16 至 12:48:49，总耗时 27分32秒。

测量结果：

- 压舱稳定基线约 7072 MiB。
- OneClick 运行峰值 15839/16311 MiB。
- 距离设备总显存仅约 472 MiB。
- coarse scan 日志约 8.1/15.9 GiB。
- paired fine scan 约 12.5–12.6/15.9 GiB。
- fine extract 约 13.8–13.9/15.9 GiB。
- 最终 8K GPU paste 长时间保持约 15.1–15.4/15.9 GiB，是全流程最高且持续时间最长的显存阶段。

日志同时确认：YOLO detector 在 paired fine scan 完成后仍持续驻留，跨越两个 extract group 和四次外部 `lada-cli` restoration，直到 paste 前才释放。因此“扫描模型与后续 extract/restoration 显存重叠”的判断被代码和运行日志共同证实。

## 11. Phase 1：最小生命周期修复

### 11.1 已实施修改

1. `_run_pre_extract_branch`：在同步 scan 的 `finally` 中释放 detector，覆盖成功、NO_MOSAIC、异常和取消路径。
2. `_process_sbs_paired_pre_extract_clip`：左右眼同步 fine scan 完成后立即释放 detector，使后续 extract/Lada 不再与 YOLO 生命周期重叠。
3. `_process_sbs_clip_to_output`：进入 split/restore 路径时释放可能由 coarse scan 遗留的 detector。
4. `split_video_dual` 和 `split_video_dual_fisheye` wrapper 完全返回后执行安全清理：
   - `gc.collect()`。
   - 清理已经加载的 CuPy memory pool。
   - 仅当 Torch CUDA 已初始化时调用 `empty_cache()`。
5. 新增 `[gpu-memory]` 阶段日志，仅记录已经加载的 Torch/CuPy allocator，不触发模块导入或 CUDA 初始化。
6. `release_detector()` 和原 paste/merge 清理中的 Torch 操作改为 `sys.modules` 与 `cuda.is_initialized()` 守卫。

### 11.2 Phase 1 实测结果

Phase 1 运行时间为 13:41:19 至 14:06:11，总耗时 24分52秒。

- 压舱基线约 5644 MiB。
- 运行峰值约 15833 MiB。
- 扣除基线后的峰值增量约 10189 MiB。
- detector 释放前后日志中的 Torch allocator 均为约 75/236 MiB allocated/reserved，CuPy 均为约 20/64 MiB used/pooled。

由于 Phase 0 和 Phase 1 的压舱基线相差约 1.4 GiB，不能将两轮绝对显存差直接归因于 Phase 1。Phase 1 修正了资源生命周期，并且有 RTX 5070 用户实机成功运行的支持，但本机数据不能证明它降低了全流程最高峰。

Phase 1 后最高峰仍然位于 8K paste，约 15.4/15.9 GiB。因此依据测量进入 Phase 2，而没有继续扩大 detector 生命周期改动。

## 12. Phase 2：降低 8K Paste NVDEC 队列显存

### 12.1 根因

`paste_segments_gpu()` 使用 `PyNvThreadedSerialDecoder`，其通用默认配置为：

```text
batch_size=8
buffer_size=32
```

8192×4096 P016 解码帧的 Y+UV 设备数据约为 96 MiB，不包含 pitch、对齐和驱动内部 surface。仅32帧解码队列理论上就可能占用约3 GiB。

Paste 实测速率约25.8fps，瓶颈在 patch/NVENC，而不是 NVDEC；32帧预取深度没有带来可见吞吐收益，却成为主要显存峰值来源。

NVENC `_EncodeSink` 的4帧输入保活环用于解决已经验证过的异步读取缓冲复用和绿块问题，本阶段明确不修改该安全机制。

### 12.2 已实施修改

1. 新增代码默认配置：

```text
gpu_paste_decoder_buffer_size=8
```

2. 新增 `_paste_decoder_kwargs()`，将 paste decoder 默认配置调整为：

```text
batch_size=8
buffer_size=8
```

3. 普通 `paste_segments_gpu()` 的8K base decoder 和所有活动 restored segment decoder 均使用受限队列。
4. fisheye rect paste 使用相同队列策略，避免同类显存峰值。
5. 新增运行日志：

```text
[gpu-memory] paste decoder queue: batch=8, buffer=8 frames
```

## 13. Phase 2 A/B 实测结果

Phase 2 运行时间为 14:24:48 至 14:49:35，总耗时 24分47秒。

| 指标 | Phase 1 | Phase 2 | 变化 |
|---|---:|---:|---:|
| 压舱基线 | 5644 MiB | 5403 MiB | 测试环境差异 |
| 运行峰值 | 15833 MiB | 13391 MiB | -2442 MiB |
| 扣除基线后的峰值增量 | 10189 MiB | 7988 MiB | **-2201 MiB** |
| 总耗时 | 24:52 | 24:47 | 无性能下降 |
| Paste 稳态速度 | 约25.8fps | 约25.8–25.9fps | 无吞吐下降 |

关键结果：

- 归一化峰值下降2201 MiB，约2.15 GiB。
- 实测下降量与8K P016 decoder 从32帧降至8帧的理论节省量一致。
- Paste 稳态显存从约15.4/15.9 GiB下降到约12.9–13.1/15.9 GiB。
- 日志确认 `batch=8, buffer=8` 配置生效。
- Paste fps没有下降，总处理时间没有增加。
- 最终输出通过 fast HEVC merge 的时长、分辨率检查，最终平均码率与此前一致。

## 14. 最终结论与停止条件

本轮优化已经达到目标：

1. 修正 detector 与后续恢复阶段不必要的生命周期重叠。
2. 修正 split 返回后仍可能存在的临时 PyNv/CuPy 对象回收边界。
3. 将全流程主要显存峰值降低约2.15 GiB。
4. 在降低显存的同时保持 paste 吞吐和总体处理时间。
5. 保留 NVENC 输入缓冲安全机制，没有用稳定性换取显存。

当前不建议继续将 decoder buffer 从8降低到4，也不建议缩小 NVENC 4帧保活环。继续修改可能造成 NVDEC 饥饿、吞吐波动或重新引入编码绿块，而当前数据没有显示进一步优化的必要性。

建议将当前版本交给 RTX 5070 12GB 用户使用相同问题文件复测。若仍然发生闪退，应依据新加入的阶段日志判断是否属于不同路径，例如 fisheye paste、GPU timeline fallback、普通左右眼 merge 或特定驱动错误，而不应继续无差别缩减全局缓冲。
