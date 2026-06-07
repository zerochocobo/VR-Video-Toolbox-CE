# Qwen3-TTS 字幕转音频慢与语言异常修复总结

## 背景

同声翻译工具的“字幕转音频”在真实字幕上暴露了两个问题：

1. **速度过慢**：`Hoppers...chs.srt` 共 1417 条字幕，总输出时长约 6198.448 秒。用户在处理到前 10 条时手动停止，系统耗时约 2 分 41 秒。
2. **输出语言不可辨识**：`videos/35.srt` 共 11 条字幕，批量 TTS 生成完成耗时约 55 秒，但 `videos/35.si.wav` 听感不像目标字幕语言，也不是可用的中文/日文语音。

日志中的关键线索：

- `FlashAttention2 is not installed; using eager attention.`
- 小样本批量生成后虽然速度有所改善，但输出内容质量异常。
- `videos/35.srt` 每条字幕同时包含中文和日文两行，旧解析逻辑没有按目标语言选行。

## 根因

### 1. TTS 推理路径慢

- 未安装 FlashAttention2 时，旧逻辑直接退到 eager attention，CUDA 上推理效率较低。
- 字幕逐条生成，1417 条字幕会产生大量小推理请求，GPU 利用率差。
- `max_new_tokens` 预算偏宽，短字幕也可能生成过长 token，浪费解码步数。
- 每条字幕适配目标时长时可能启动一次 ffmpeg 子进程，短句多时进程开销明显。

### 2. 多语言字幕选行错误

- `parse_srt()` 过去默认取第一条非空字幕行。
- `videos/35.srt` 这类双语字幕中，中文和日文同时存在；选择日语时仍可能读到中文，选择中文时也缺少明确规则。

### 3. 当前 in-process Qwen3-TTS 运行时质量异常

- 当前主环境使用 `transformers 5.9.0`。
- 本地 5.9 兼容 vendor 路径在实测中会生成“请订阅/感谢观看”等无关套话，ASR 无法识别为目标字幕内容。
- 对照官方 `qwen-tts 0.1.1` + `transformers 4.57.3` + `huggingface_hub 0.36.2` 后，同一句 `最近肩膀疼` 能被正确生成并被 ASR 识别。

## 解决方法

### 1. 提速字幕 TTS

已在 `tool_si/logic.py` 中调整：

- CUDA 下未安装 FlashAttention2 时，优先使用 PyTorch SDPA attention；加载失败再退回 eager。
- 将逐条字幕 TTS 改为小批量推理，默认 `VRTB_TTS_BATCH_SIZE=4`。
- 按 token 预算相近程度分组，避免极短句和长句混在同一批后被过度压缩。
- 新增 `VRTB_TTS_BATCH_TOKEN_SPREAD`，默认 `1.5`，最大限制 `10.0`。
- 收紧 `max_new_tokens`，按 12Hz codec 时长预算和文本长度下限估算，减少短字幕无谓生成。
- 默认使用内存内 `librosa.effects.time_stretch` 做时长适配，避免每条字幕启动 ffmpeg。
- 保留 `VRTB_TTS_TIME_FIT_MODE=ffmpeg`，需要旧路径时可显式切回。
- 清理 SRT/ASS 样式覆盖码，例如 `{\an8}`，避免送入 TTS。

### 2. 修复多语言字幕文本选择

已让 `parse_srt(path, language=...)` 支持按目标语言选择多行字幕：

- 中文优先选择纯汉字/中文行。
- 日语优先选择含假名的行。
- 韩语优先选择韩文行。
- 英语优先选择拉丁文本行。
- 未匹配时保留原来的第一条非空行兜底。

### 3. 修复生成内容异常

已新增官方 legacy worker 路径：

- 新增 `tool_si/_vendor/qwen_tts_legacy/`：基于官方 `qwen-tts 0.1.1`，只保留项目已有的 12Hz 懒导入补丁，避免不必要的 25Hz/torchaudio/sox 导入。
- 新增 `tool_si/qwen_tts_worker.py`：子进程加载 legacy 官方运行时，主进程通过 JSON line 发送 TTS 请求。
- 主进程优先扫描 `runtime_cache/uv_cache/archive-v0` 中的 `transformers-4.57.3.dist-info` 和 `huggingface_hub-0.36.2.dist-info`，可用时默认走官方 worker。
- 模型在 worker 中只加载一次，仍支持批量字幕请求。
- 可用 `VRTB_TTS_LEGACY_WORKER=0` 关闭 worker，回退到当前进程内运行时。
- `subtitle_to_audio()` / `batch_subtitle_to_audio()` 在任务结束或异常时关闭 worker，避免残留子进程。

## 验证结果

- 真实模型短句批量 smoke test：SDPA 路径可以成功生成。
- `videos/Hoppers.2026.BluRay.1080p.10Bit.x265.AAC(7.1).chs.srt` 前 10 条：模型加载后批量生成约 34.64 秒；用户原日志前 10 条到停止约 2 分 41 秒，虽然口径包含加载和停止等待，但改善量级明显。
- `videos/35.srt` 完整 11 条：通过 `subtitle_to_audio()` 生成后，ASR 能识别为字幕内容，不再是套话或不可辨识输出。
- 日语模式：`parse_srt('videos/35.srt', language='Japanese')` 选择 `最近肩が痛くて`；使用 `Ono_Anna` 生成单句后 ASR 可识别为目标日文。
- 编译验证通过：
  - `.venv\Scripts\python.exe -m py_compile tool_si\logic.py tool_si\qwen_tts_worker.py tests\test_tool_si.py tool_si\_vendor\qwen_tts\core\models\modeling_qwen3_tts.py`
- 测试通过：
  - `.venv\Scripts\python.exe -m pytest tests\test_tool_si.py tests\test_i18n.py -q`
  - 结果：`16 passed`

## 使用建议

- 默认保持 `VRTB_TTS_LEGACY_WORKER` 开启，优先保证生成内容正确。
- 显存充足时可尝试提高 `VRTB_TTS_BATCH_SIZE` 到 `6` 或 `8`，观察是否继续提速。
- 如果字幕长短差异大，可适度提高 `VRTB_TTS_BATCH_TOKEN_SPREAD`，但过高可能让短句和长句混批后影响时长适配。
- 如果后续安装 FlashAttention2，应重新跑一次同一批字幕基准，确认是否比 SDPA 更快。

## 剩余风险

- 已验证前 10 条真实字幕和 11 条完整小样本；1417 条完整长任务还需要用户机器跑一次全量基准确认最终耗时。
- batch size 增大可能带来更高显存占用，默认值保持保守。
- legacy worker 依赖 `runtime_cache/uv_cache/archive-v0` 中的官方依赖缓存；如果用户清理缓存，需要重新准备对应版本依赖，或临时关闭 `VRTB_TTS_LEGACY_WORKER`。
