# SI Duck Key 开发计划

日期：2026-07-01

## 结论

基于字幕段生成独立 duck key 是合理方案，建议实施。

当前“自动降低原声音量”使用 ffmpeg `sidechaincompress`，但 key 来自实际混入输出的 SI.WAV。这样有两个限制：

- SI 语音短于原字幕段时，SI 没声音的尾部不会继续触发 duck，原声会漏出。
- “不可闻垫音”在当前结构下不可行，因为同一条 SI 轨既做 sidechain key 又混进输出；能触发 duck 的电平约为 `0.025`，本身已经可能被听见。

正确方向是把 sidechain key 和实际 SI 混音轨拆开：生成一条不混入输出的 mono duck key WAV，段内恒定电平，段外为 0。混音时第三路输入只给 `sidechaincompress` 使用。

## 当前实现

相关入口：

- `tool_clonevoice/gui.py`
  - “混合视频音轨”读取 UI 参数。
  - 调用 `tool_si.logic.mix_si_audio_track()` 或 `batch_mix_si_audio_tracks()`。
- `tool_si/logic.py`
  - `build_si_mix_filter()` 构造 ffmpeg filter。
  - `build_si_audio_mix_command()` 构造 ffmpeg 命令。
  - `mix_si_audio_track()` / `batch_mix_si_audio_tracks()` 执行混音。
  - 当前 duck 参数：
    - `SI_DUCK_THRESHOLD = "0.025"`
    - `SI_DUCK_RATIO = "5"`
    - `SI_DUCK_ATTACK_MS = "30"`
    - `SI_DUCK_RELEASE_MS = "600"`
    - `SI_DUCK_MAKEUP = "1"`

当前 duck filter 的核心问题：

```text
[1:a:0] ... adelay=..., volume=..., apad, asplit=2[si_key][si]
[orig_base][si_key]sidechaincompress=...[orig]
```

`si_key` 来自 SI.WAV 本身，并且 delay 会同时影响 key 和实际混音。

## 必须覆盖的两类 SI.WAV 生产源

不能只改 `tool_si.subtitle_to_audio()`，因为克隆语音工具生成 `.si.wav` 不走这个函数。

需要覆盖：

1. `tool_si.logic.subtitle_to_audio()`
   - 输入：SRT `SubtitleEntry(start, end, text)`。
   - 现有 timeline 已按字幕时间线写出 `.si.wav`。
   - 可同时写出 duck key。

2. `tool_clonevoice.omnivoice_backend.synthesize()`
   - 输入：manifest segments / merged units，包含原视频时间线 `start/end`。
   - 当前输出 `<video>.si.wav`。
   - 也应默认写出 `<video>.si.duck.wav`。

否则克隆语音工具的“混合视频音轨”仍无法利用字幕段级 duck。

## 输出文件约定

建议默认生成并持久化 duck key WAV。

路径规则：

- 对视频配对场景：
  - SI 音频：`movie.si.wav`
  - duck key：`movie.si.duck.wav`
- 对任意 SI 音频路径：
  - `default_si_duck_key_path(si_audio_or_video)` 返回同目录同 stem 的 `.si.duck.wav`。
  - 如果输入是 `movie.mp4`，返回 `movie.si.duck.wav`。
  - 如果输入是 `movie.si.wav`，返回 `movie.si.duck.wav`。

格式：

- mono WAV
- 采样率与 SI.WAV 一致，优先 24000 或最终 timeline 的 sample rate
- float timeline 写出为 16-bit PCM 即可
- 段内恒定电平，段外 0

建议 key 电平：

- 初始固定 `0.25`（约 -12 dBFS），远高于当前 threshold，能稳定触发 duck。
- key 不混入输出，所以不需要“不可闻”；只需避免溢出和确保 sidechain 稳定。

段边界：

- 默认按原字幕/manifest 的 `start/end`。
- 可选加 10-30ms 边缘 ramp，减少压缩器触发突变；不是 P0 必需。

## 混音 filter 改造

新增参数：

- `duck_key_path: str | Path | None = None`
- `duck_preset: str = "normal"` 或独立的 `duck_threshold/ratio/release` 参数

新逻辑：

1. `duck_original=False`
   - 不使用 duck key，保持现状。

2. `duck_original=True` 且 `duck_key_path` 存在
   - ffmpeg 命令增加第三路输入：

```text
-i video -i si.wav -i si.duck.wav
```

   - sidechain key 使用 `[2:a:0]`。
   - SI delay 只作用于实际 SI 混音轨，不作用于 duck key。

3. `duck_original=True` 但 duck key 不存在
   - 为兼容旧文件，回退到当前 SI.WAV-as-key 方案。
   - 日志提示：`duck key not found; falling back to SI waveform sidechain`。

both channel 新 filter 结构示意：

```text
[0:a:0]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,volume=... [orig_base];
[2:a:0]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=mono,apad [duck_key];
[1:a:0]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=mono,adelay=...,volume=...,apad [si_mono];
[orig_base][duck_key]sidechaincompress=... [orig];
[si_mono]aformat=channel_layouts=stereo [si];
[orig][si]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.95 [si_track]
```

left/right channel 同理：

- `[duck_key]` 压整条 stereo 原声。
- SI 仍只混入用户选择的 left/right/both。

## Duck 强度预设

建议做成 UI 可选项，但不要把它当作解决“漏声”的主方案。漏声由独立 duck key 解决，强度档只控制压低程度。

预设：

| 档位 | threshold | ratio | release |
|---|---:|---:|---:|
| 轻 | `0.03` | `2.5` | `400` |
| 普通 | `0.025` | `5` | `600` |
| 强 | `0.015` | `10` | `800` |

attack 可继续保持 `30ms`，makeup 继续 `1`。

UI 建议：

- 现有“自动降低原声音量” checkbox 保留。
- 勾选后显示或启用“压低强度：轻 / 普通 / 强”。
- 默认“普通”。

## 开发阶段

### Phase 1：duck key 生成基础设施

新增 helper：

- `default_si_duck_key_path(path)`
- `build_duck_key_timeline(spans, total_duration, sample_rate, level=0.25)`
- `write_duck_key_wav(path, spans, total_duration, sample_rate, level=0.25)`

span 数据结构：

```python
{"start": 10.0, "end": 14.0}
```

要求：

- clamp 到 `[0, total_duration]`
- 忽略 `end <= start` 的异常段
- 支持重叠段，重叠处仍为同一恒定 level
- 输出长度与 SI timeline 对齐

### Phase 2：两类 SI.WAV 生成时默认写 duck key

`tool_si.logic.subtitle_to_audio()`：

- 在写 `output_path` 之前或之后，根据最终 `entries` 和 `total_duration` 写 `default_si_duck_key_path(output_path)`。
- time window 模式需确认 entries 是否已经相对窗口对齐；duck key 必须和输出 `.si.wav` 的时间轴一致。
- `max_entries` 测试模式只覆盖实际转换的 entries。

`tool_clonevoice.omnivoice_backend.synthesize()`：

- 根据最终合成使用的 `units` 或 manifest segments 写 `<video>.si.duck.wav`。
- 优先使用原字幕/manifest 段的 `start/end`，而不是生成音频实际长度。
- 输出日志：`[synth] wrote duck key ...`。

### Phase 3：混音命令接入第三路 duck key

`tool_si.logic.SITrackMixTask`：

- 增加 `duck_key_path: Path | None`。

`collect_paired_si_mix_tasks()`：

- 为每个 `movie.si.wav` 查找 `movie.si.duck.wav`。
- 找到则填入 task；找不到则 `None`。

`mix_si_audio_track()` / `batch_mix_si_audio_tracks()`：

- 新增 `duck_key_path=None`。
- 单文件默认自动从 `si_audio_path` 推导 duck key。
- 批量使用 task 中的 key。

`build_si_audio_mix_command()`：

- 新增第三路输入和 filter 参数。
- 当 key 存在时使用 key；不存在时兼容旧 filter。

### Phase 4：duck 强度 UI

`tool_clonevoice/gui.py`：

- “混合视频音轨”增加“压低强度”下拉框。
- 默认普通。
- 传入 `duck_preset` 或实际参数。

`tool_si/gui.py`：

- 如果该独立工具仍保留同样混音功能，也同步 UI，避免两个入口行为不一致。

i18n：

- 中文：轻 / 普通 / 强，压低强度
- 英文、日文同步 key

### Phase 5：日志与兼容提示

混音日志建议：

- key 存在：
  - `Using subtitle duck key: movie.si.duck.wav`
- key 缺失：
  - `Duck key not found; falling back to SI waveform sidechain.`
- duck preset：
  - `Duck preset: normal (threshold=0.025, ratio=5, release=600ms)`

## 测试计划

单元测试：

1. `build_duck_key_timeline`
   - 段内为 level，段外为 0。
   - 重叠段不叠加超过 level。
   - 越界段会 clamp。

2. `subtitle_to_audio`
   - mock TTS 输出，确认 `.si.wav` 和 `.si.duck.wav` 都生成。
   - duck key 的非零区间覆盖字幕 `start/end`，不是生成音频长度。

3. `omnivoice_backend.synthesize`
   - fake model 输出，确认 `<video>.si.duck.wav` 生成。
   - key 覆盖 manifest 原时间段。

4. `build_si_audio_mix_command`
   - duck key 存在时命令包含第三路 `-i key.wav`。
   - filter 使用 `[2:a:0]` 作为 sidechain key。
   - `adelay` 只出现在 SI 混音轨，不出现在 duck key 轨。
   - key 缺失时回退旧 filter。

5. `collect_paired_si_mix_tasks`
   - 有 `movie.si.duck.wav` 时收集到 key。
   - 没有 key 时仍收集任务，保持旧文件可混音。

集成测试：

- 构造 4 秒原声 + 2 秒 SI + 4 秒 duck key。
- 混音后检查 2-4 秒区间原声仍被压低，证明不依赖 SI 实际尾部。

## 风险与注意点

- 如果只在 `subtitle_to_audio()` 写 key，克隆语音 `.si.wav` 不会受益；必须同步改 `omnivoice_backend.synthesize()`。
- key 的时间轴必须和最终 `.si.wav` 一致，尤其注意测试模式和截取时间窗。
- SI delay 不应影响 duck key，否则会重新产生“原声段覆盖不完整”的问题。
- duck key 缺失时必须保留旧行为，否则用户已有 `.si.wav` 无法使用自动降低原声音量。
- 强度档位不应过多，三档足够；真正的质量关键是 key 时间段正确。

## 建议默认行为

- 默认生成 `.si.duck.wav`。
- 默认混音时如果存在 `.si.duck.wav` 就使用它。
- 默认 duck 强度为“普通”。
- 没有 duck key 时自动回退当前 SI waveform sidechain，不中断用户流程。
