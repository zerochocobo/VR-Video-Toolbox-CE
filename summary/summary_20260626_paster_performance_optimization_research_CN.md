# Paster 性能优化研究进展

日期：2026-06-26

本文用于给其他开发人员同步 OneClick `paste paired rects onto interval` / paster 性能优化的最新进展。当前工作还没有完工，重点是说明已经确认的事实、测试入口、已否决方案，以及后续需要继续研究的方向。

## 1. 目标与约束

目标：

- 优化 OneClick source-scan Stage 3 的 `paste paired rects onto interval`。
- 场景是 Lada/Jasna/native restore 已经生成 restored rect clip，paster 需要把 restored rect 贴回原始 8K base 视频。
- 不调整 P4，不降低画质，不改变用户当前编码质量边界。

明确约束：

- 不基于本地 uv/CuPy 异常直接改生产 paste 算法。
- 性能测算代码不能影响实机生产路径性能。
- 任意 profile / debug sync 都只能作为研究工具，不能长期留在默认热路径里。

## 2. 关键代码路径

非 fisheye paste：

- `utils/segment_paster.py::paste_segments_gpu_or_fallback()`
- `utils/segment_paster.py::_try_paste_segments_gpu_passthrough()`
- `gpu_engine/files.py::paste_segments_gpu()`

fisheye paste：

- `gpu_engine/files.py::paste_fisheye_eye_rects_to_sbs_gpu()`

相关底层：

- `gpu_engine/files.py::_copy_planes_to_packed_views()`
- `gpu_engine/files.py::_make_alpha_mask()`
- `gpu_engine/files.py::_EncodeSink.feed()`
- `gpu_engine/pynv_io.py::PyNvThreadedSerialDecoder`
- `gpu_engine/pynv_io.py::PyNvEncoderSession`

非 fisheye paste 当前主流程：

1. NVDEC 解码 base full frame。
2. `cp.cuda.Device().synchronize()` 等待 base decode 完成。
3. 取 base `Y/UV` CuPy view。
4. `_copy_planes_to_packed_views()` 把 full frame 拷到 NVENC 需要的 packed buffer。
5. 对每个 active restored rect：
   - NVDEC 读取 restored rect frame。
   - `_match_depth()` 做 bit-depth 对齐。
   - 对 Y plane 做 feather alpha paste。
   - 对 UV plane 做 feather alpha paste。
6. `_app_frame_from_packed()` 包装 NVENC AppFrame。
7. `_EncodeSink.feed()`：
   - encode 前整设备 sync，确保前面 CuPy 写入完成。
   - 调用 `enc.encode(...)`。
   - 保留最近几个 AppFrame 引用，避免 NVENC 尚未读完输入 buffer 时 CuPy 内存池复用导致绿块。

## 3. 来自真实日志的瓶颈

参考日志：

- `videos/2_1_process_base.log`

关键事实：

- Stage 3 paste paired rects：
  - `10318` 帧
  - 约 `6m40s`
  - 稳定约 `25.8 fps`
- 编码设置：
  - `8192x4096`
  - `10bit`
  - `preset=P4`
  - `rc=vbr`
  - `multipass=fullres`
  - `aq=1`
  - `bf=0`
- passthrough：
  - `parts=2`
  - `passthrough_frames=602/10922`
  - 只有开头约 10 秒能 stream-copy，主要区间仍必须完整 8K 重编码。
- fine restored rect：
  - left rect：`1680,1696,1360x1552`
  - right rect：`5504,1664,1248x1600`
  - 两个 rect 基本持续到待处理区间尾部。

结论：

- 该样本 paste 慢不是因为 stage 2 copy 或 concat。
- 当前样本几乎没有尾部 passthrough 空间。
- 只要 restored rect 持续存在，就必须重编码完整 8K frame；这是主成本。

## 4. 测试与诊断代码

### 4.1 crop/paste 基准脚本

脚本：

```powershell
scripts\bench_oneclick_crop_paste.py
```

用途：

- 模拟 OneClick pre-extract 的 crop/paste 形态。
- 默认 `--base-mode preclip-copy`，先 stream-copy 出短 base clip，避免把原片非零 start 的 keyframe discovery/seek 成本误算成 paste 成本。
- 可通过 `--skip-crop --restored <rect_clip>` 复用已有 rect clip，只测 paste。

示例：

```powershell
uv run python scripts\bench_oneclick_crop_paste.py `
  --src videos\test_8k_m426_2_demosaic.mp4 `
  --start 00:00:00 `
  --end 00:00:03 `
  --rect 1680,1696,1360,1552 `
  --crop-mode left `
  --out-dir debug_output\paste_profile `
  --skip-crop `
  --restored debug_output\paste_profile\20260626_171654\rect_crop.mp4 `
  --no-measure-quality
```

### 4.2 本地卡点诊断脚本

脚本：

```powershell
scripts\debug_paste_hang_points.py
```

用途：

- 定位本地 uv/CuPy/PyNv 卡点。
- 默认先 `import gpu_engine`，使诊断脚本使用和生产一致的 CUDA/CuPy 环境配置。
- `--raw-env` 用于复现未配置 CuPy 默认环境。

重要 case：

- `--case env`
- `--case cupy-copy`
- `--case cupy-arange`
- `--case rawkernel`
- `--case pynv-read`
- `--case paste-direct-copy`
- `--case paste-alpha`
- `--case encode-first-frame`

`paste-alpha` 额外参数：

- `--alpha-source prod/cpu`
- `--alpha-eval split/prod-expression`

注意：

- `split` 会把 alpha paste 拆成多步并每步 sync，只用于定位卡点，不是性能基准。
- `prod-expression` 尽量复现原始一行 CuPy alpha 公式。

### 4.3 CuPy/CUDA cache 问题

已验证：

- 当前账号对 `C:\Users\dennis\.cupy` 存在写权限问题。
- 如果没有设置 `CUPY_CACHE_DIR` / `CUDA_CACHE_PATH`，CuPy 默认 cache 可能落到用户目录，导致首次 JIT / RawKernel 路径异常慢或卡住。

已处理：

- `gpu_engine/_cuda_env.py`
  - `CUPY_CACHE_DIR = <repo>/runtime_cache/cupy_kernel_cache`
  - `CUDA_CACHE_PATH = <repo>/runtime_cache/cuda_compute_cache`
- `packaging/runtime_hook_cuda.py`
  - frozen exe 下对应到 `<exe_dir>/runtime_cache/...`

验证：

- `cupy-arange` 正常完成。
- `rawkernel` 正常完成。
- `runtime.warmup` 第一次写新 cache 约 `27s`，第二次 cache 命中约 `1.2s`。

说明：

- `CUDA path could not be detected` 警告仍可能出现，因为项目故意清 `CUDA_PATH` 以使用 CUDA 12.8 wheel 头文件。
- 这个警告和 CuPy cache 权限问题不是同一件事。

## 5. Paste profile 最新结果

为了分清 base decode、packed copy、restored decode、alpha paste、NVENC encode 的占比，曾在 `gpu_engine/files.py::paste_segments_gpu()` 内加入研究用 `_PasteProfiler`。

profile 分桶：

- `open_segment_probe`
- `open_segment_decoder`
- `open_segment_alpha_masks`
- `base_frame_at`
- `base_decode_sync`
- `base_yuv_views`
- `packed_copy`
- `restored_frame_at`
- `restored_decode_sync`
- `restored_yuv_views`
- `match_depth_y`
- `match_depth_uv`
- `alpha_luma`
- `alpha_chroma`
- `app_frame`
- `encode_feed`
- `encode_flush`

开启方式：

```powershell
$env:VRVT_PASTE_PROFILE='1'
$env:VRVT_PASTE_PROFILE_SYNC='1'
$env:VRVT_PASTE_PROFILE_EVERY='0'
```

重要注意：

- profile 模式会插入额外 CUDA sync。
- profile 结果用于归因，不用于比较真实吞吐。
- 当前用户明确要求：性能测算代码不能影响实机性能。因此后续如果保留该 profiler，必须把它隔离到显式 profile 分支或 bench-only 代码路径，不能让默认生产路径每帧经过 profile wrapper。

### 5.1 误判修正

第一版 profile 显示：

- `base_decode_sync` 占大头。

后来确认这是误导：

- `_EncodeSink.feed()` 中 NVENC encode 异步返回。
- 下一帧开头的 `cp.cuda.Device().synchronize()` 会吃掉上一帧 encode 的等待。
- 所以等待被错误归到了 `base_decode_sync`。

修正方法：

- profile 模式下在 `encode_feed` section 后也做一次 sync。
- 这样上一帧 encode 等待会归到 `encode_feed`。

### 5.2 修正后的短基准结果

样本：

- `videos/test_8k_m426_2_demosaic.mp4`
- `8192x4096`
- `10bit`
- `59.940 fps`

短基准：

- 区间：`00:00:00-00:00:03`
- 帧数：`180`
- rect：`1680,1696,1360x1552`
- 单 rect，低于 `2_1` 实机双 rect 的 alpha 工作量。

profile 结果：

| section | total | 占比 | ms/frame |
| --- | ---: | ---: | ---: |
| `encode_feed` | `6.494s` | `85.1%` | `36.079` |
| `packed_copy` | `0.187s` | `2.4%` | `1.036` |
| `alpha_luma` | `0.187s` | `2.4%` | `1.038` |
| `alpha_chroma` | `0.064s` | `0.8%` | `0.356` |
| `base_decode_sync` | `0.044s` | `0.6%` | `0.246` |
| `restored_decode_sync` | `0.034s` | `0.4%` | `0.187` |

结论：

- 当前非 fisheye paste 的主瓶颈是 **8K 全帧 P4/fullres/AQ NVENC encode**。
- base decode、restored decode、packed copy、alpha paste 都不是主瓶颈。
- 对 `2_1` 双 rect case，alpha 成本大约会比单 rect 更高，但仍不可能超过 encode 主成本。

## 6. 已尝试但不应继续推进的方案

### 6.1 border-only alpha paste

思路：

- `_make_alpha_mask()` 生成的 feather 之外 alpha 精确为 `1.0`。
- rect interior 的 `round(1*src + 0*dst)` 等价于直接 copy restored。
- 只对 feather border 做 float blend，interior 直接 copy。

正确性验证：

- CuPy 小数组覆盖 Y/UV、feather/无 feather。
- 与原整块 alpha 公式逐像素一致。
- 最大差值：`0`。

真实 8K 短基准：

| 方案 | paste fps |
| --- | ---: |
| 默认整块 alpha | `21.07 fps` |
| border-only alpha | `17.72 fps` |

结论：

- 数学等价但更慢。
- 多个小 slice assignment / kernel launch 的调度成本超过了省掉的大块 float blend。
- 该方案已撤销，不应作为生产优化继续推进。

### 6.2 Torch/DLPack paste 替代 CuPy

背景：

- 本地 uv 环境曾出现 CuPy JIT / RawKernel / strided assignment 卡住。
- 一度尝试用 Torch/DLPack 绕过 CuPy elementwise。

结果：

- 这是本地环境问题，不是实机 paste 事实。
- 用户明确指出不能因为本地测试卡住就改实机生产路径。
- 该方向已撤销。

结论：

- 不应把 Torch/DLPack paste 作为默认生产优化。
- 若未来要研究，也必须基于实机 profile 和严格像素一致性验证。

### 6.3 CPU mask upload 替代 CuPy mask

用途：

- 只用于隔离 `_make_alpha_mask()` 是否卡在 CuPy `arange/minimum`。

结论：

- 不作为生产优化。
- cache 目录修正后，CuPy JIT 路径已经正常。

## 7. 当前代码状态与风险

当前状态：

- `gpu_engine/files.py` 中研究用 `_PasteProfiler`、`VRVT_PASTE_PROFILE` 和 `paste_segments_gpu()` 内的 profile section wrapper 已清理。
- 默认 `paste_segments_gpu()` 路径恢复为直线代码，不创建 profiler、不进入 no-op section、不额外计数、不额外插入 sync。
- `gpu_engine/files.py` 只保留 `_EncodeSink` 的实验开关 `VRVT_PASTE_ENCODE_SYNC=stream`，默认仍为原来的 device sync。
- `gpu_engine/pynv_io.py` 只保留 encoder 创建实验开关 `VRVT_NVENC_CUDA_GRAPH=1`，默认仍为 `useCudaGraph=False`。
- `fullres` 仍保留为默认 multipass 策略；`qres` 只作为实测有效的后续候选，不在本轮改默认。

风险：

- 后续若要重新做分桶 profile，不应把 profiling wrapper 重新包回 OneClick 默认热路径。
- profile 模式插入 sync 后得到的是归因数据，不是真实生产 FPS。

建议下一步：

1. 继续优化时以 `scripts/bench_oneclick_crop_paste.py` 的真实 paste elapsed/FPS 为主。
2. 若需要细粒度归因，优先用外部 profiler、独立 bench-only 代码或临时分支，不进入生产默认路径。
3. `multipass=fullres` 保留为默认；`multipass=qres` 需要更多素材和完整实机 case 验证后再讨论是否变成可配置或默认策略。

## 8. 后续可研究方向

### 8.1 减少必须重编码的帧数

这是不改 P4/画质时最有意义的方向。

可研究：

- 更精细的 passthrough plan。
- 对 active rect 结束后的尾部尽量 stream-copy。
- 对开头/结尾 keyframe 对齐策略做优化，减少无效重编码。

限制：

- `2_1` 日志中 rect 基本持续到尾部，只有开头 `602/10922` 帧可 passthrough。
- 因此该样本收益有限。

### 8.2 NVENC 输入 buffer 生命周期与复用

现状：

- `_EncodeSink` 保留最近几个 AppFrame 引用，避免 NVENC 异步读取时 buffer 被 CuPy 内存池复用。
- 每帧 `_copy_planes_to_packed_views()` 分配 packed buffer。

可研究：

- 固定 ring buffer 预分配，减少 `cp.empty()` / memory pool 调度。
- 但必须证明不会破坏 NVENC 输入生命周期。
- profile 显示 `packed_copy` 约 `1ms/frame`，即使完全消掉收益也有限。

### 8.3 同步策略细化

现状：

- 多处使用 `cp.cuda.Device().synchronize()`，安全但粗。

可研究：

- 当前 stream sync / CUDA event 替代部分整设备 sync。

风险：

- PyNv decoder 内部 stream 不透明。
- `_EncodeSink.feed()` 前的整设备 sync 是为了防止 NVENC 读取未完成的 packed buffer。
- 不能为速度牺牲稳定性，否则可能复现绿块/闪块问题。

### 8.4 编码策略

如果允许改质量边界，最有效的方向是 NVENC 参数：

- preset
- multipass
- AQ
- temporal AQ
- lookahead
- bitrate/maxrate

但当前用户约束是不调整 P4、不牺牲画质，所以这类方向暂不推进为默认。

## 9. 给接手开发者的建议

优先做：

1. 默认生产热路径已经清掉 profile wrapper；不要再把 `_PasteProfiler` 或 `VRVT_PASTE_PROFILE` 形式的代码包回 `paste_segments_gpu()`。
2. 用真实 `2_1` / m426 短片段继续做 A/B，重点看真实 paste elapsed/FPS，而不是 profile FPS。
3. `multipass=fullres` 保留为当前默认；`multipass=qres` 虽然本轮有约 19.5% 提速且指标未劣化，但仍需更多素材验证。
4. 若 encode 仍是主导，不要继续微调 alpha paste；优先研究 passthrough frame reduction 或编码策略的可配置化。

不要重复做：

- 不要再做 border-only alpha，已验证更慢。
- 不要因本地 CuPy JIT 卡顿改生产 paste 算法；先确认 cache 目录和 fresh process。
- 不要用 profile 模式的 FPS 当真实生产 FPS。
- 不要把 `CUDA path could not be detected` 警告当成 CuPy cache 问题。

推荐复现实验顺序：

```powershell
# 1. 检查环境/cache
uv run python scripts\debug_paste_hang_points.py --case env

# 2. 检查最小 CuPy JIT
uv run python scripts\debug_paste_hang_points.py --case cupy-arange
uv run python scripts\debug_paste_hang_points.py --case rawkernel

# 3. 跑真实短 paste baseline
uv run python scripts\bench_oneclick_crop_paste.py --src videos\test_8k_m426_2_demosaic.mp4 --start 00:00:00 --end 00:00:03 --rect 1680,1696,1360,1552 --crop-mode left --skip-crop --restored <rect_crop.mp4> --no-measure-quality

# 4. qres 候选验证（不改默认 fullres）
uv run python scripts\bench_oneclick_crop_paste.py --src videos\test_8k_m426_2_demosaic.mp4 --start 00:00:00 --end 00:00:05 --rect 1680,1696,1360,1552 --crop-mode left --skip-crop --restored <rect_crop.mp4> --paste-enc-extra multipass=qres
```

## 10. 目前结论

在当前约束下，paster 性能瓶颈不是 rect paste 公式本身，而是完整 8K P4/fullres/AQ NVENC 重编码。

如果继续坚持：

- 不改 P4
- 不改 multipass/AQ
- 不减少重编码帧数
- 不改变质量边界

那么代码层面的收益预计很小。真正能明显提速的方向，要么是减少必须重编码的帧数，要么是重新讨论 NVENC 编码策略。
