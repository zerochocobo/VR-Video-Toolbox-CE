# PyNv ThreadedDecoder 非关键帧启动导致 paste 时间错位问题审计摘要

日期：2026-06-05

## 背景

用户反馈：即使将 `pre_extract_pipeline_enabled` 设为 `False`，最终 paste 回原视频后仍出现严重时间错位。典型异常文件：

```text
videos\2_SSTART_EEND_sbs.restored_scan_tmp\mosaic_seg001_L.f00013620-00014762.r720_2848_2912x1248.restored.mp4
```

表象不是几帧误差，而是 restored/crop 区域内容相对底部完整画面提前约数秒。第一段目测正常，第二段异常。

## 关键结论

`pre_extract_pipeline_enabled=False` 只关闭 P9 的 extract/restore 生产者-消费者并发，不会关闭 PyNv ThreadedDecoder 在 extract 阶段的非关键帧 seek 行为。

本次错位的直接原因是：

1. 第二段细检测区间从 `frame 13620` / `227.229s` 开始。
2. 源 clip 的真实关键帧在 `222.255s` 和 `227.260s`。
3. `227.229s` 比后一个关键帧早约 2 帧，不是关键帧安全起点。
4. `PyNvVideoCodec.ThreadedDecoder(start_frame=13620)` 实际吐出了前一个关键帧附近的内容。
5. 旧包装层把返回的第一帧直接标记为 `13620`，没有丢弃关键帧预滚帧。
6. 因此生成的 crop 输入文件从约 `222.229s` 内容开始，但文件名和 paste 计划认为它从 `227.229s` 开始，最终出现约 5 秒错位。

这解释了为什么关闭 `pre_extract_pipeline_enabled` 没有效果：该开关关闭的是跨组并发污染风险，不处理 ThreadedDecoder 从非关键帧启动时的预滚帧问题。

## 排查证据

### 1. 当前配置确认

代码默认值和运行读取值均为：

```text
pre_extract_pipeline_enabled = False
```

因此本次不是配置没有生效。

### 2. 文件状态

实际异常文件在：

```text
videos\2_SSTART_EEND_sbs.restored_scan_tmp
```

而不是最初提到的 `videos\test_8k_unmasaic_SSTART_EEND_sbs.restored_scan_tmp` 当前目录状态。

相关文件时间戳显示：

```text
mosaic_seg001_L.f00013620-00014762...mp4          2026/6/5 18:45:40
mosaic_seg001_L.f00013620-00014762...restored.mp4 2026/6/5 19:00:49
mosaic_seg001.restored.mp4                         2026/6/5 19:25:13
```

不是早上旧缓存复用。

### 3. 帧数/时长匹配，排除简单 duration 漂移

`mosaic_seg001_L.f00013620-00014762...mp4`：

```text
width=2912
height=1248
duration=19.051987
nb_frames=1142
```

对应 `mosaic_seg001_L.segments.json`：

```json
{
  "start_s": 227.229,
  "end_s": 246.279367,
  "start_frame": 13620,
  "end_frame": 14762,
  "frame_count": 1142
}
```

帧数一致，说明不是 restored 文件少帧/多帧造成的线性漂移。

### 4. 抽帧对照定位到 extract 阶段

已生成调试图：

```text
debug_output\mosaic_seg001_timing_debug\base_L_t227p229_crop.png
debug_output\mosaic_seg001_timing_debug\input_L_f0.png
debug_output\mosaic_seg001_timing_debug\base_L_t222p229_crop_minus5s.png
debug_output\mosaic_seg001_timing_debug\base_R_t227p229_crop.png
debug_output\mosaic_seg001_timing_debug\input_R_f0.png
```

肉眼对比结果：

```text
input_L_f0 不匹配 base_L_t227p229_crop
input_L_f0 匹配 base_L_t222p229_crop_minus5s
```

右眼也呈现同类现象。结论：错误已经存在于 crop 输入文件，发生在 restore 和 paste 之前。

### 5. 关键帧验证

`ffprobe` 查到 `mosaic_seg001.mp4` 末段关键帧：

```text
217.250s
222.255s
227.260s
232.265s
237.270s
241.274s
246.279s
```

异常段开始时间：

```text
227.229s
```

它落在 `222.255s` 和 `227.260s` 之间，并且比 `227.260s` 关键帧早约 31ms，即约 2 帧。ThreadedDecoder 从这个非关键帧开始时，实际内容回退到了前一个 GOP 起点。

## 涉及代码路径

主要路径：

```text
one_click/logic.py
  _process_sbs_paired_pre_extract_clip()
  _run_extract_group()

gpu_engine/files.py
  extract_multi_rect_clip()
  extract_transformed_rect_clip()

gpu_engine/pynv_io.py
  PyNvThreadedSerialDecoder
```

重要细节：

- `pre_extract_pipeline_enabled=False` 后仍会执行 `_run_extract_restore_sequential()`。
- sequential 路径仍会调用 `_run_extract_group()`。
- `_run_extract_group()` 内部仍可能使用 `extract_multi_rect_clip()`。
- `extract_multi_rect_clip()` 和 `extract_transformed_rect_clip()` 都依赖 `PyNvThreadedSerialDecoder(start_frame=...)`。

所以关闭 P9 并不能规避非关键帧 seek 预滚问题。

## 修复方案

已在 `gpu_engine/pynv_io.py` 修改 `PyNvThreadedSerialDecoder`：

1. 保留调用方请求的 `start_frame` 作为业务目标帧。
2. 新增内部 `_decode_start_frame`。
3. 构造 decoder 时用 `ffprobe` 获取源文件关键帧列表。
4. 将 `_decode_start_frame` 设为目标帧之前最近的关键帧。
5. 真正创建 PyNv `ThreadedDecoder` 时传入 `_decode_start_frame`。
6. 初始 batch 会用 PTS 做一次校准：如果 ThreadedDecoder 实际第一帧不是理论关键帧，而是 keyframe 后若干帧，则把内部帧索引同步到该 PTS 对应的真实 frame index。
7. `frame_at(target_frame)` 仍按原目标帧请求，包装层会显式丢弃关键帧到目标帧之间的预滚帧。

效果：

```text
业务请求：frame_at(13620)
实际解码启动：previous_keyframe_frame，例如 13320
内部行为：丢弃 13320..13619
返回给调用方：13620 对应内容
```

这样细检测区间不需要扩大到关键帧边界，paste 仍可以保持精确区间。

## 追加错误与补强

首次修复后，运行 `videos\2_2.mp4` 时 PTS 校验拦截了另一个非静默错误：

```text
PyNvThreadedSerialDecoder NVDEC seek check failed:
start_frame=2400,
expected_pts=2402378,
got_pts=2404380,
delta=2002,
decode_start_frame=2102
```

该 delta 对应约 2 帧，说明 ThreadedDecoder 从前一个 keyframe 启动后，实际首帧并不一定是理论 keyframe 帧，可能受 B 帧/重排或 PyNv 内部 seek 语义影响，从 keyframe 后 2 帧开始吐出。

补强方式：

```text
构造期用 SimpleDecoder 探测 decode_start_frame 起若干帧的 PTS -> frame index 映射
ThreadedDecoder 第一个 batch 到达后读取 batch[0].getPTS()
若该 PTS 能映射到真实 frame index，则校准 _batch_start_idx / _next_source_idx
之后再继续丢 preroll 到业务 target_frame
```

因此后续遇到 `keyframe+2` 这类小重排偏差时会继续校准处理，而不是直接报错。

## 为什么不只把细检测区间扩大到关键帧

按关键帧切分是最安全的兜底方案，但会把 `227.229s` 前约 5 秒也送入 LADA/restore，并在 paste 时覆盖更大的时间范围。

当前修法更精确：

```text
底层解码按关键帧安全启动
业务输出仍按细检测原始帧边界开始
```

也就是兼顾关键帧安全和细检测精度。

## 测试

新增：

```text
tests/test_pynv_io_preroll.py
```

覆盖：

- 目标帧落在两个关键帧之间时，选择前一个关键帧作为 `_decode_start_frame`
- 目标帧本身是关键帧时，不额外回退
- 无关键帧列表时维持旧行为
- 用假的 `PyNvVideoCodec` 验证 `PyNvThreadedSerialDecoder(start_frame=13620)` 实际传给 ThreadedDecoder 的启动帧是前一个 keyframe frame
- 模拟 ThreadedDecoder 实际从 `keyframe+2` 开始吐帧时，初始 batch PTS 校准后 `frame_at(target)` 仍返回目标 PTS

已运行：

```text
python -m pytest tests/test_pynv_io_preroll.py -q
5 passed

python -m pytest tests/test_source_time_scanner.py tests/test_one_click_pre_extract.py tests/test_segment_paster.py -q
39 passed
```

## 剩余风险

1. 本地 shell 环境没有 `PyNvVideoCodec`，无法在当前终端直接跑真实 PyNv 解码探针；运行时验证需要在实际工具环境中完成。
2. 修复依赖 `ffprobe` 关键帧列表；如果 ffprobe 失败，会回退旧行为。
3. 对长 GOP 非关键帧起点，首次解码会多丢弃一个 GOP 的预滚帧，正确性优先，代价是少量性能开销。
4. 现有 `pre_extract_pipeline_enabled=False` 仍应保留，避免 P9 并发 decode/restore 的独立污染风险。

## 审核建议

审核重点建议看三处：

1. `gpu_engine/pynv_io.py` 的 `_threaded_decoder_preroll_frame()` 是否正确处理 keyframe 到 frame index 的映射。
2. `PyNvThreadedSerialDecoder.__init__()` 中 `_decode_start_frame` 与 `start_frame` 的分离是否覆盖所有 `frame_at()` 调用路径。
3. 对 paste passthrough / restored segment 局部 seek 的影响：该修复应同样改善局部 seek，不应改变输出时间轴长度。
