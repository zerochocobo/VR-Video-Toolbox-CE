# OneClick 切片输出无音轨修复（前置切片化）

## 用户反馈

一键去马赛克模式，输入填了起止时间（如 `00:00:00 → 00:01:00`），
处理完输出视频**没有任何音轨**（`ffprobe -show_streams -select_streams a`
返回空，不是 "audio 时长为 0" 而是流根本不存在）。换三个不同源、
切换最高/高/中三档画质都复现，日志里**没有** `fast HEVC merge failed`
字样。

## 根因

`one_click/logic.py` 里 `_run_source_scan_branch` 顶部有一条早退：

```python
if start_time or end_time:
    log_callback("[source-scan] start/end subranges are not wired yet; falling back to the normal path")
    return PreExtractResult.SCAN_FAILED
```

只要用户填了起止时间，**整条 source_scan 优化路径被直接跳过**，
回落到旧的"分眼 → 逐眼 lada → 合并"三步流水线
（`run_single_file_pipeline` 第 2724-2744 行附近）。

旧路径上音频要通过三道 mux 接力（split → lada restore → combine_video），
每一道都调 `gpu_engine.mux.mux_hevc_with_audio`，而该函数默认参数对
"切片场景"很脆弱：

1. `-map 1:a:0?` 是**可选映射**，找不到目标流 ffmpeg 直接静默丢弃；
2. 默认 `shortest=True`，raw HEVC 与源音频的时长亚秒级漂移在 `-c copy`
   下被误判为"音轨太长"，整条音轨被砍掉
   （`utils/sbs_concat.py:419-422` 的注释明确警告过这个坑，
   但 `replace_timeline_segments_gpu` 和 `combine_video` 没显式 `shortest=False`）。

三道 mux 任意一道丢音，下游就没源可取，最终输出彻底没有 audio stream。

`fast HEVC merge` 路径反而**没问题**——它走的是 `_mux_mp4_video_with_source_audio`
（必填 `-map 1:a:0`，无音轨会报错而不是静默丢）+ 显式 `shortest=False`。
这就是为什么日志里看不到 `fast HEVC merge failed`：根本没走到 source_scan。

## 修复方案

不改 `mux_hevc_with_audio` 各个调用点的 `shortest` 默认参数
（影响面太大且不一定都该改成 False）。改成**前置切片**：

> 用户填了起止时间 → 先用 ffmpeg `-c copy` 关键帧快切一段
> `<source_stem>_S<ss>_E<to>.<ext>`（左右眼俱全，自带音轨）→
> 把这个文件作为新的 input 走原本完整的 source_scan / native-stream /
> legacy 三种路径，**`start_time/end_time` 在内部置 None**，
> 不再触发那条旧的 SCAN_FAILED 早退。

这样切片场景跟无切片场景走的是同一条已经被充分优化的链路，音频也
自然通过 source_scan 的 `_mux_mp4_video_with_source_audio` 必填映射保证存在。

## 实际改动

只动 `one_click/logic.py` + `one_click/main.py` + 三个 i18n 文件。
未触碰 `gpu_engine/`、`utils/sbs_concat.py` 等下游模块。

### `one_click/logic.py`

紧跟 `_remove_file_quiet` 新增两个辅助函数：

- **`_cut_subrange_keyframe(input, output, start_sec, end_sec, ...)`**
  关键帧对齐流复制：
  ```
  ffmpeg -ss <start> -i <input> -t <dur>
         -map 0:v:0 -map 0:a?
         -c copy -avoid_negative_ts make_zero <output>
  ```
  - 不走 NVDEC/NVENC，磁盘 I/O 速度，一分钟 4K HEVC 片源秒级完成。
  - 起点对齐到最近的前置关键帧，实际时长可能比请求值偏前后几秒
    （这个误差通过 UI 提示告知用户）。
  - **曾试过 NVENC 重编码版**（输入端 `-ss` + 默认 `-accurate_seek`
    跳帧到精确 PTS），帧精度正确但实测只跑出 ~11 fps 太慢，
    用户否决，回到流复制方案。

- **`_prepare_subrange_preclip(input_file, output_dir, start_time, end_time, ...)`**
  - start/end 为空：直接 `return (input_file, None)`，旁路。
  - 有值：构造 preclip 路径 `<src_stem>_S<ss>_E<to><src_ext>` 放在
    `output_dir` 下，已存在则复用（断点续跑友好），否则调用上面的
    快切函数生成。
  - 命名方式按用户要求：用**源文件名** + 时间后缀（如
    `test_S000000_E000030.mp4`），**不带** `.preclip` 标记，
    跟最终输出 `test_S000000_E000030_sbs.restored.mp4` 前缀一致。

`run_single_file_pipeline` 和 `run_single_eye_pipeline` 的改造模板：

```python
preclip_path: str | None = None
try:
    # ... 算 directory/filename/file_final 等 ...
    if os.path.exists(file_final): ...  # 跳过逻辑保持

    # 在算完文件名、source_scan_enabled 已就绪之后，调用 source_scan 之前
    input_file, preclip_path = _prepare_subrange_preclip(
        input_file, directory, start_time, end_time,
        log_callback=log_callback, process_callback=process_callback,
    )
    if preclip_path is not None:
        start_time = None
        end_time = None

    # ... 后续 source_scan / native_stream / legacy 全部用新 input_file ...
finally:
    if preclip_path is not None and not keep_intermediate:
        _remove_file_quiet(preclip_path, log_callback=log_callback)
    # ... 原有 process_logger.close() 等 ...
```

关键点：
- `directory`/`filename`/`suffix`/`file_l`/`file_r`/`file_final` 都在
  preclip 之前 snapshot，所以替换 `input_file` 不会污染下游中间文件名。
- `cleanup_final_output` 仍然指向真正的最终 file_final，不受 preclip 影响。
- 不勾"保留中间文件"时 preclip 会在 finally 里清掉；勾上则保留供
  ffprobe 调试。

`_run_source_scan_branch` 顶部的 SCAN_FAILED 守卫**保留**，作为防御性
兜底——理论上调用方已经 strip 了 start/end，进不到这里。

### `one_click/main.py`

单文件 tab 和 单文件单眼 tab 的结束时间输入框右边各加一条灰色提示
标签。原本 `ttk.Entry` 直接 `grid` 在 `column=1`，现在改成包进
`ttk.Frame`，Frame 里用 `pack` 让 Entry 和 Label 并排：

```python
end_frame_s_auto = ttk.Frame(tab)
end_frame_s_auto.grid(row=2, column=1, sticky='w', padx=5)
ttk.Entry(end_frame_s_auto, textvariable=self.s_auto_end, ...).pack(side='left')
ttk.Label(end_frame_s_auto, text=get_text('lbl_end_hint'),
          foreground='gray').pack(side='left', padx=(8, 0))
```

其他组件的 grid 行号没变。

### i18n（`i18n/zh.json` / `en.json` / `ja.json`）

仅在 `one_click` 段加一条 `lbl_end_hint`：

- zh：`(用关键帧快切，前后误差几秒)`
- en：`(keyframe fast cut, a few seconds drift)`
- ja：`(キーフレーム高速カット、前後数秒の誤差)`

## 用户感知行为

跑 `test.mp4` + `00:00:00 → 00:00:30`：

1. 日志先出现：
   ```
   [preclip] keyframe cut 0.0->30.0 -> <dir>/test_S000000_E000030.mp4
   Executing: ffmpeg -ss 0.000000 -i ... -t 30.000000 -map 0:v:0 -map 0:a? -c copy ...
   [preclip] done: ... (XX MB)
   ```
2. 然后正常进入 `[source-scan] Stage 1 scanning source: ...test_S000000_E000030.mp4`
   走完整流水线。
3. 输出 `test_S000000_E000030_sbs.restored.mp4`，`ffprobe` 能看到 audio stream。
4. 默认清掉 preclip；勾"保留中间文件"则保留供调试。

## 已知 trade-off

- 流复制按 GOP 对齐，30 秒请求可能输出 28~32 秒区间内的任意时长。
  通过 UI 上的灰色提示文字告知用户，避免被误判成 bug。
- 如果以后真需要帧精度切片，可以加一个"精准切片"勾选项调用
  NVENC 重编码版本（代码已经写过又被回退，git 历史可参考）。

## 未涉及但仍存在的潜在隐患

旧路径触发概率被压到几乎为 0（除非 `_run_source_scan_branch` 因 HDR/
bt2020 等原因继续 SCAN_FAILED，回到 legacy 三步流水线），但
`mux_hevc_with_audio` 的 `-map 1:a:0?` + 默认 `shortest=True` 组合
仍然是潜在静默丢音风险点。如果未来遇到非切片场景下的丢音报告，
应该在 `gpu_engine/files.py` 的以下调用点显式补 `shortest=False` 并
把可选音轨映射改成必填（让丢音报错而不是静默）：

- `combine_video` (files.py:784-785)
- `replace_timeline_segments_gpu` (files.py:1457-1464)
- `paste_segments_gpu` (files.py:962-969)
- `paste_fisheye_eye_rects_to_sbs_gpu` (files.py:1264-1271)
- `process_video_multi` (files.py:619-625)
- `_run_native_sbs_stream` -> `native_mosaic/engine.py:1054-1063`

本次只做切片场景的根因修复，不大范围动 `mux_hevc_with_audio`
默认参数，避免影响面不可控。
