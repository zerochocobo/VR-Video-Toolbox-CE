# OneClick 二阶段细检测 Frame-Key 缓存修复总结

## 背景

在 OneClick source-scan 的二阶段 paired fine pre-extract 分支中，`mosaic_seg002.mp4` 处理后出现贴回画面错位。复盘 `debug_output/mosaic_fix_20260604` 后确认，问题集中在 paired 分支的细切缓存文件命名与复用策略。

## 根因

paired 分支此前使用重新编号后的 `seg_id` 生成缓存文件名：

```text
{stem}_{L|R}.seg000.mp4
{stem}_{L|R}.seg000.restored.mp4
```

但 `seg_id` 不是稳定内容标识。检测器结果会随阈值、排序、置信度抖动和配对顺序变化而改变，同一个 `seg001` 在不同 run 中可能对应不同的时间窗口或不同 rect。paired 分支又按文件存在性复用缓存，导致旧文件可能被当作新 segment 贴回，从而产生严重错位。

同一时间窗口多个 segment 会放大这个问题，因为它们天然帧数相同，仅靠时长/帧数无法证明缓存内容正确。

## 修复

采用 frame-key 内容命名，缓存文件名直接由实际处理帧区间和 rect 决定：

```text
{stem}_{L|R}.f{start_frame:08d}-{end_frame:08d}.r{x}_{y}_{w}x{h}.mp4
{stem}_{L|R}.f{start_frame:08d}-{end_frame:08d}.r{x}_{y}_{w}x{h}.restored.mp4
```

`start_frame/end_frame` 使用与 GPU extract/paste 完全一致的计算方式：

```python
round(start_s * fps)
round(end_s * fps)
```

这样排序和重新编号不会影响缓存命中；只有时间帧区间和 rect 完全一致时才会复用同一个缓存文件。

## 关键实现

- `one_click/logic.py`
  - 新增 `_segment_frame_bounds()`。
  - 新增 `_paired_segment_cache_key()`。
  - 新增 `_paired_segment_paths()`。
  - 新增 `_cleanup_orphan_paired_segment_files()`。
  - `_process_sbs_paired_pre_extract_clip()` 中，paired 细切文件名从 `segNNN` 改为 frame-key。
  - paired 分支入口清理旧 `segNNN` 文件和不属于本次 frame-key 集合的孤儿内容键文件。
  - cleanup 改为删除实际记录的 `segment_input_paths + restored_paths`，不再用字符串替换推导路径。

## 明确没有采用的方案

没有采用“同时间多 segment union 合并”。原因：

- 会显著放大 crop 区域，违背小块加速目标。
- 只能覆盖完全同时间的 case，不能覆盖时间部分重叠但空间不同的 case。
- paste 阶段本身支持同帧多个 active rect，根因不是多 rect 贴回能力，而是缓存文件名不稳定。

## 测试覆盖

- `test_pair_eye_segments_reindexes_by_start_time_after_score_matching`
  - 保留按时间排序后的 `seg_id` 可读性。
- `test_paired_fisheye_path_patches_in_memory_without_full_fisheye_base`
  - 验证 paired 细切文件名使用 frame-key。
- `test_paired_pre_extract_uses_frame_key_cache_and_removes_orphans`
  - 验证正确 frame-key 缓存可复用。
  - 验证旧 `segNNN` 缓存和不匹配内容键缓存会被清理。

## 验证结果

- `python -m compileall one_click\logic.py tests\test_source_time_scanner.py`
- `python -m pytest tests/test_source_time_scanner.py -q`
  - 15 passed
- `python -m pytest tests/test_source_time_scanner.py tests/test_one_click_pre_extract.py tests/test_mosaic_prescan.py tests/test_segment_paster.py tests/test_keyframe_cutter.py tests/test_app_config.py -q`
  - 43 passed
- `python -m pytest tests/test_engine_runner.py tests/test_source_time_scanner.py tests/test_sbs_concat.py tests/test_segment_paster.py tests/test_keyframe_cutter.py tests/test_mosaic_prescan.py tests/test_one_click_pre_extract.py tests/test_mux.py tests/test_app_config.py -q`
  - 50 passed

## 后续注意

frame-key 方案解决的是缓存语义错误。若未来需要更强的抗中断能力，可以再把 extract/restore 输出改成临时文件完成后原子 rename，避免异常中断留下半成品被复用。
