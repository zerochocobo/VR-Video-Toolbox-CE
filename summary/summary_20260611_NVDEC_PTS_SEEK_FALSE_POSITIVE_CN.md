# NVDEC PTS seek false positive 问题小结

日期：2026-06-11

## 现象

一键去马赛克流程在 GPU extract 阶段创建 `PyNvThreadedSerialDecoder(start_frame=1770)` 后报错：

```text
PyNvThreadedSerialDecoder NVDEC seek check failed:
start_frame=1770,
expected_pts=2657696,
got_pts=2660666,
delta=2970,
decode_start_frame=1554
```

该错误出现在 NVDEC seek 自检阶段，流程因此提前中断。

第一次修复 PTS 原点偏移后，又在后续 fine extract 片段遇到同类自检报错：

```text
start_frame=2700,
expected_pts=4054092,
got_pts=4057148,
delta=3056,
normalized_delta=86,
pts_origin_delta=2970,
decode_start_frame=2514
```

这里扣掉容器 PTS 原点后只剩 `86` ticks 残差。

## 关键结论

这次不是切片文件过大、不是未走 CUDA，也不是画质档位导致的解码性能问题。根因是同一段视频在 `SimpleDecoder` 与 `ThreadedDecoder` 两个 PyNv 解码入口中使用了不同的 PTS 原点，并且部分 B-frame/时间戳舍入场景会留下远小于一帧的 PTS 残差：

- `SimpleDecoder` 返回的目标帧 PTS：`2657696`
- `ThreadedDecoder` 返回的目标帧 PTS：`2660666`
- 差值：`2970`
- 容器首帧 PTS：`2970`

也就是说，`ThreadedDecoder` 的 PTS 保留了容器时间轴的首帧偏移，而 `SimpleDecoder` 的随机访问 PTS 已经以 0 为起点归一化。扣掉该偏移后，个别帧仍可能有小于 10% 帧间隔的时间戳残差。原有校验要求二者绝对相等，因此把正确帧误判成 seek 错位。

## 证据

真实探针和前置对比显示：

- 容器首帧 PTS 为 `2970`
- 报错中的 `got_pts - expected_pts` 也为 `2970`
- 同一目标帧通过两个解码入口取出的图像内容一致
- 像素差异为 `max=0`、`mean=0.0`
- 第二个失败点归一化后残差为 `86` ticks；相邻帧 PTS 步长约 `1501` ticks，残差约为 `5.7%` 帧间隔
- 第二个失败点 Simple/Threaded 图像内容同样一致，像素差异为 `max=0`、`mean=0.0`

因此这是 PTS 时间轴基准不同造成的 false positive，不是实际内容错帧。

## 修复

修改 `gpu_engine/pynv_io.py`：

1. 新增首帧容器 PTS 探测，用 `ffprobe` 读取视频第一帧的 `best_effort_timestamp` / `pts`。
2. 在 `PyNvThreadedSerialDecoder` 初始化时读取 `SimpleDecoder.frame_at(0).pts`，计算：

```text
pts_origin_delta = container_first_pts - simple_decoder_first_pts
```

3. 初始 preroll batch 校准时，先尝试用 `actual_pts - pts_origin_delta` 映射回 SimpleDecoder 的帧索引。
4. 用目标帧相邻帧 PTS 估算当前流的一帧 PTS 步长，并设置 `10%` 帧间隔以内的小残差容忍。
5. 首个目标帧 seek 校验时，允许以下情况通过：

```text
actual_pts == expected_pts
actual_pts - pts_origin_delta == expected_pts
abs((actual_pts - pts_origin_delta) - expected_pts) <= pts_tolerance
```

6. 如果归一化并套用小残差容忍后仍不相等，继续抛出 `NVDEC seek check failed`，并在错误中输出 `normalized_delta`、`pts_origin_delta` 与 `pts_tolerance`，方便区分真实错位和时间轴偏移。

没有把主路径直接改成 `SimpleDecoder`，原因是 extract 阶段是顺序连续读帧，`ThreadedDecoder` 是高速路径；`SimpleDecoder` 更适合随机取样，直接替换会显著拖慢长片段处理。当前策略是在保留高速路径的同时，让 seek 自检按同一 PTS 时间轴比较，并容忍远小于一帧的时间戳残差。

## 验证

单元与调用侧回归：

```text
python -m pytest tests/test_pynv_io_preroll.py -q
8 passed

.venv\Scripts\python.exe -m pytest tests/test_pynv_io_preroll.py tests/test_gpu_extract_multi.py tests/test_gpu_fisheye_patch.py tests/test_source_time_scanner.py tests/test_one_click_pre_extract.py tests/test_segment_paster.py -q
50 passed, 1 subtests passed
```

真实 PyNv/NVDEC 探针：

```text
idx=1770 ok frame_pts=2660666 pts_origin_delta=2970 normalized_delta=0 pts_tolerance=150 decode_start_frame=1554
idx=2700 ok frame_pts=4057148 pts_origin_delta=2970 normalized_delta=86 pts_tolerance=150 decode_start_frame=2514
```

同一复现场景在修复后可以通过首帧 seek 校验。另跑了失败起点附近的短窗口真实 extract smoke，实际 `extract_transformed_rect_clip` 路径可以完成编码和 mux。

## 剩余风险

- 修复依赖 `ffprobe` 能读取首帧 PTS；读取失败时会退回旧的严格行为。
- 该修复只容忍稳定的 PTS 原点偏移和远小于一帧的小残差，不会放过接近整帧或更大的真实 seek 错位。
- 系统 Python 环境缺少 `PyNvVideoCodec`，真实 NVDEC 探针需使用项目 `.venv`。
