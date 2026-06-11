# pre-extract 大面积区域局部绕过细剪开发计划

日期：2026-06-11

## 背景

用户启用“去马赛克前先提取有马赛克的时间和区域”后，source-scan 会先粗扫整片视频，切出含马赛克的时间段，再在每个时间段内做左右眼细扫、局部 crop、LADA restore、最后 paste 回原 SBS 时间段。

该策略对小面积马赛克有效：只恢复局部区域，减少 LADA 输入像素。但当细扫后需要 crop 的区域已经接近或超过单眼画面的较大比例时，局部细剪收益明显下降，反而会引入额外的 extract、restore、paste 管理成本。之前讨论过“把 paste rect 扩成整只眼”，但这仍然走 paste 逻辑，不能解决该场景下的主要效率问题。

## 目标

保留用户启用 pre-extract 的语义，但对每个 Stage 2 切出的 mosaic segment 单独判断：

- 小区域：继续现有 paired fine crop 路径。
- 大区域：仅当前这个 Stage 2 mosaic segment 绕过 crop 细剪，改走普通整眼 restore 路径。

不能把 `pre_extract_inner` 全局改成 `False`，否则会让其他小区域时间段也失去细剪收益。

## 判定位置

面积 guard 放在 Stage 2 切出不同时间段的 mosaic segment 之后、Stage 3 处理该 segment 时执行。

理由：

1. Stage 2 后的输入已经是实际 keyframe 对齐后的时间段。
2. Stage 3 paired fine scan 已经左右眼分开执行。
3. 使用的是用户设置的 `fine_conf`。
4. scan 结果坐标已经是单眼坐标。
5. `MosaicSegment.w/h` 已经包含细扫阶段实际会使用的 `pre_extract_rect_expand`、最小尺寸和对齐结果。

因此此处计算 `seg.w * seg.h / eye_area` 最接近真实 crop 面积。不要拿粗扫整 SBS 的 segment 再额外乘细扫放大倍数，否则可能重复放大。

## 实现方案

新增配置：

```text
pre_extract_bypass_crop_area_ratio = 0.333333
```

含义：如果任一细扫 segment 在某只眼中的 crop 面积比例超过该阈值，则当前 Stage 2 mosaic segment 不走局部 crop pre-extract。

实现步骤：

1. 在 `PreExtractResult` 增加：

```text
BYPASS_CROP = "bypass_crop"
```

2. 在 `_process_sbs_paired_pre_extract_clip()` 中：

```text
scan left/right
pair left/right
if no segment: copy through / NO_MOSAIC
compute max ratio per side
if ratio > threshold: return BYPASS_CROP
else continue existing extract -> restore -> paste
```

3. 在 `_run_source_scan_branch()` 中，仅对当前 timeline entry 处理该返回值：

```text
if paired_result == BYPASS_CROP:
    _process_sbs_clip_to_output(..., pre_extract_inner=False, ...)
```

这只影响当前 Stage 2 mosaic segment，不影响其他 timeline entries。

4. 日志需要明确说明局部绕过原因：

```text
[source-scan] paired fine crop bypassed for this segment:
side=L ratio=0.410 threshold=0.333; using full-eye restore path
```

## 测试计划

新增/更新 `tests/test_source_time_scanner.py`：

1. paired fine scan 返回小面积左右眼 segment 时，仍调用 `extract_multi_rect_clip()` 和 `paste_segments_gpu_or_fallback()`。
2. paired fine scan 返回超过阈值的大面积 segment 时，`_process_sbs_paired_pre_extract_clip()` 返回 `BYPASS_CROP`，不执行 extract/restore/paste。
3. source-scan 外层收到 `BYPASS_CROP` 时，只对当前 mosaic entry 调用 `_process_sbs_clip_to_output(... pre_extract_inner=False ...)`。
4. 可选覆盖阈值配置，确认阈值从 `app_config` 读取。

## 风险与边界

- 该方案是性能保护，不改变马赛克检测或恢复正确性。
- 二次细扫失败仍按原有 `SCAN_FAILED` 逻辑处理。
- 只绕过当前 Stage 2 mosaic segment，不禁用整个任务的 pre-extract。
- `use_fisheye=True` 也可用同一面积判断，因为判断发生在 paired fine scan 之后；但 fisheye 下面积语义是 fisheye 空间面积，后续如发现误判可单独增加开关。
