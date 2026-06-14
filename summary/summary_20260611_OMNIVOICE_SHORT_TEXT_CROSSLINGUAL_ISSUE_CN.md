# OmniVoice：跨语言语音克隆对「短目标文本」产出乱码

**日期：** 2026-06-11
**来源：** VR-Video-Toolbox 的 `tool_clonevoice`（日语视频 → 中文配音，按说话人克隆音色）
**复现脚本：** `summary/omnivoice_short_text_repro.py`（+ `omnivoice_short_text_repro_result.json`）

---

## 一句话结论

用**一个固定的参考音频**（约 3.7s 的日语片段 + 其文本），`OmniVoice.generate()`：

- **长中文目标文本：克隆正确**；
- **短中文目标文本（几个字）：产出乱码 / 内容错误**。
- 与 `language` 参数（`"zh"` / `"Chinese"` / `None`）无关，与生成参数
  （`guidance_scale` / `position_temperature` / `class_temperature` / `num_step`）无关，
  与参考音频长度（3.7s vs 10s）无关。

这卡住了配音场景：配音里大量短句（如"厉害啊"/"嗯"），它们**必须用与该说话人长句相同的克隆音色**。
不能对短句换成别的/自动音色（声音/性别突变不可接受）。

**给专家的问题：** 是否有支持的方式克隆「短句」到指定音色？短目标文本是否是已知限制？
是否需要某种预处理、最小长度、padding 或参数设置？

---

## 环境

| 组件 | 版本 |
|---|---|
| omnivoice | 0.1.5 |
| 模型 | `k2-fsa/OmniVoice`（本地快照，自带 `audio_tokenizer/`）|
| torch | 2.8.0+cu128（CUDA 12.8）|
| transformers | 5.9.0 |
| accelerate | 1.13.0 |
| numpy | 2.0.2 |
| GPU | NVIDIA RTX 5060 Ti（16GB, sm_120）, fp16 |
| 回测 ASR | faster-whisper 1.2.1 + Whisper large-v3（ctranslate2 4.8.0）|

`model = OmniVoice.from_pretrained("models/OmniVoice", device_map="cuda", dtype=torch.float16)`

## 设置

- **参考（所有目标共用同一个）：** 3.70s 日语语音 + 其文本
  `ref_text = "仕事できない人に対してどう教えていいか分かんないんだよね"`（在推荐的 3–10s 内）。
- **目标：** 中文句子，长 → 短。
- 用 Whisper 把生成音频**回测转录**，客观判断（不靠耳朵）。OmniVoice 输出音量偏小，回测前峰值归一化到 0.6。

### 精确调用（每个目标完全相同）

```python
audio = model.generate(
    text=text,                 # 中文目标句
    ref_audio=REF_AUDIO,       # 固定的 3.7s 日语 wav
    ref_text=REF_TEXT,         # 其日语文本
    language="Chinese",        # 也试过 "zh" 和 None — 无差异
    num_step=32,
    guidance_scale=2.0,
)[0]
```

（也试过可复用 prompt 路径 `model.create_voice_clone_prompt(...)` + `voice_clone_prompt=` — 行为一致。）

## 结果（生成音频的 Whisper 回测）

| 目标（字数） | 生成时长 | Whisper 回测 | 判定 |
|---|---|---|---|
| `你好，今天天气真不错，我们一起出去散步聊聊天吧。`（24） | 3.72s | `你好今天天气真不错我们一起出去散步聊聊天吧` | **正确** |
| `后辈一下子多了好多。`（9） | 2.00s | `好歹一生是多了好多` | 部分/退化 |
| `加班多吧。`（5） | 1.60s | `整理&字幕志愿者 杨茜茜` | **乱码** |
| `厉害啊。`（4） | 1.44s | `很厲害` | 部分（丢"啊"）|

短目标配不同 `language` 参数 — 全错：

| 调用 | Whisper 回测 |
|---|---|
| `language="zh"` | `嗯好玩多了` |
| `language="Chinese"` | `嗯就這麼多吧` |
| `language=None` | `嗯好玩的` |

## 参数扫描（短目标 `加班多吧`，同一固定 ref）

扫了 `guidance_scale ∈ {2,3,4}`、`position_temperature ∈ {0,1,5}`、`class_temperature ∈ {0,0.5}`、
`num_step ∈ {32,64}`（10 组组合）。**没有一组**能正确生成短句；回测如 `嗯可惡多了`、`沒可不可以等嗎`、
`蛤蟒多嗎`、`太棒了`、`一二三`、`本歌曲来自…`。3 字的 `厉害啊` 偶尔出 `厲害`（丢"啊"），但 `加班多吧` 从未正确。

默认 `position_temperature=5.0`（gumbel 采样）下生成是**随机的**：同一短输入多次输出不同（仍然错）。

## 参考音频长度（3.7s vs 10s）— 没有帮助

把参考换成同一说话人的 10s 片段（推荐 3–10s 的上限）+ 其（更长的）日语文本，重跑相同目标：

| 目标 | 3.7s ref | 10s ref |
|---|---|---|
| 长句 | 正确 | 正确 |
| `后辈一下子多了好多` | `好歹一生是多了好多`（部分）| `小子天使`（**更差**）|
| `加班多吧` | 乱码 | `一 二 三`（乱码）|
| `厉害啊` | `很厲害`（部分）| ``（空，**更差**）|

更长的参考（因而更长的 `ref_text`）让短/中目标**更差**，而非更好——与 issue #50（长参考会退化）
和 3–10s 推荐一致。故障由**短目标长度**驱动，与参考长度无关。

## 交叉验证（排除是我们用法的问题）

- **无参考的自动音色**说同样的短句可以工作：`model.generate(text="加班多吧。", language="zh")`
  → Whisper 回测 `加班多嘛`（正确）。说明**文本和模型没问题**；是**参考 + 短目标**的组合崩。
- **同一参考 + 长目标**正确（见上表）。说明**参考没问题**；是**短目标**崩。
- 因此故障精确定位在「**有参考** 且 **目标文本短**」（跨语言：日语参考 → 中文目标）。

## 我们找到的 workaround（及其不足）

在短句后拼接一句载体 —— `text = "加班多吧。" + "我们改天再慢慢聊聊这件事情吧"` —— 能让**短句正确生成**
（Whisper 回测开头是正确的短句）。但随后必须**裁掉载体**，而在合成音频里检测目标/载体边界不可靠
（Whisper 对合成短片识别错误；繁简不一致；静音幻觉）。载体也会扰动目标的时长。所以这不是健壮方案。

## 给专家的问题

1. **短目标文本**（几个字）是否是 OmniVoice 跨语言克隆的已知限制？
2. 是否有**支持的方式**克隆短句到指定音色 —— 最小文本长度、必需标点、推荐的
   `num_step`/`guidance_scale`/`t_shift`/temperature，或不同的 API 路径？
3. 是否与短文本的**时长估计**（`RuleDurationEstimator` 的 `low_threshold` boost）产生的目标 token 太少有关？
   用 `duration`/`speed` 强制更多 token 是否是预期解法？（我们试过 `duration=2.5`/`4.0` — 仍乱码。）
4. 对**跨语言克隆**（参考语言 ≠ 目标语言）的短语句有无指导？

复现：`python summary/omnivoice_short_text_repro.py`（打印表格并写 JSON）。
