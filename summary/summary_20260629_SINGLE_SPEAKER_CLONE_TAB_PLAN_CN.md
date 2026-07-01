# tool_clonevoice「单人语音克隆」Tab 设计方案（外部评审版）

日期：2026-06-29  
状态：方案设计，尚未进入实现  
目标：在 `tool_clonevoice` 的「一键克隆」tab 之前新增「单人语音克隆」tab，让用户在单人语音场景下先细选最像的目标语言基准语音，再批量翻译并克隆输出 WAV。

---

## 1. 背景与用户目标

当前 `tool_clonevoice` 的「一键克隆」流程偏自动化：

```
选视频/目录
  -> 转录 + 说话人分离
  -> 自动抽取参考音频
  -> AI 翻译
  -> OmniVoice 克隆合成
  -> <视频名>.si.wav
```

需要先澄清：本 tab 不是新增一套独立克隆能力。现有 `omnivoice_backend._build_speaker_prompts()` 已经会把源语言参考音克隆成目标语言固定样句的 `work_ref`，并用 ECAPA 在多个 take 中选择最像原始参考音的一条。新 tab 的真正增量是把这条自动链路前移成用户可试听、可干预的显式步骤：

```
自动 refsel 选源音
  -> 自动生成目标语言 work_ref
  -> 自动 ECAPA 选 take

改为：

用户从多个源音候选中试听/选择
  -> 生成并试听目标语言 work_ref
  -> 人耳判断 + ECAPA 排序辅助
  -> 冻结为 SPEAKER1.wav/txt
```

这个流程适合多说话人或快速批量处理，但在“一个视频或一个目录里只有一个人的声音”时，用户希望：

- 先转录，不立即进入最终合成。
- 从原视频中挑出最像、最干净、最适合克隆的原音片段。
- 用目标语言固定样句生成试听样音。
- 用户自己听原音和目标语言样音，判断相似度。
- 也允许一键用 ECAPA 打分排序，辅助挑选。
- 也允许用户导入自己的 `WAV + 对应文本`。
- 也允许用 OmniVoice 的音色设计能力生成基准音色。
- 最终把基准语音和对应目标语言文本保存为类似：

```
SPEAKER1.wav
SPEAKER1.txt
```

然后再翻译、克隆成最终 `<视频名>.si.wav`。

一句话总结：最终 `SPEAKER1.wav` 是一段被冻结的目标语言 `work_ref`，`SPEAKER1.txt` 是这段目标语言样音对应的固定样句文本；manifest 通过 `skip_work_ref=true` 告诉合成阶段直接复用它，不再运行时重复生成 work_ref。

---

## 2. UI 总体方案

新增 tab 名称：

```
单人语音克隆
```

插入位置：

```
单人语音克隆 | 一键克隆 | 混合视频音轨
```

首行备注：

```
当单个视频或目录中只有一个人的语音，可以细选最像的。
```

采用分步式向导，而不是把全部设置堆在一个页面。

推荐步骤：

```
① 转录
② 选择基准语音
③ 确认 SPEAKER1
④ 翻译并克隆
```

底部统一按钮：

```
[上一步] [下一步] [执行当前步骤] [停止]
```

日志区始终保留，复用现有 `tool_clonevoice.gui.ClonevoiceToolsApp.log()` 风格。

---

## 3. 固定目标语言样句：复用现有 `_GENERIC_REF_TEXTS`

用户指出“目标语言样句已经有现成的固定样句，不用给三句”。已确认现有固定样句位置：

- 文件：`tool_clonevoice/omnivoice_backend.py`
- 常量：`_GENERIC_REF_TEXTS`
- 解析函数：`_resolve_generic_ref(language)`

当前已覆盖：

| key | 语言 |
|---|---|
| `chinese` | 中文 |
| `english` | 英文 |
| `korean` | 韩文 |
| `thai` | 泰文 |
| `german` | 德文 |
| `french` | 法文 |
| `spanish` | 西班牙文 |
| `portuguese` | 葡萄牙文 |
| `italian` | 意大利文 |
| `russian` | 俄文 |

设计结论：

- 不新增三句候选样句。
- 每个目标语言只使用 `_GENERIC_REF_TEXTS` 中现有固定样句。
- 每个原音候选最终只展示一条目标语言试听样音，但内部应沿用现有 `WORK_REF_TAKES` 思路：同一句固定样句可生成多个 take，再用 ECAPA 选最像的一条，避免手动流程质量低于现有自动流程。
- ECAPA 排序比较“原音候选”与“对应固定样句最终样音”的相似度；这不是全新风险，现有自动流程已经用同类跨语言 ECAPA 分数选择 work_ref take。
- 若目标语言不在 `_GENERIC_REF_TEXTS`，UI 需要提示该语言没有内置固定样句，并降级为：
  - 用户手动输入目标语言样句；或
  - 导入自己的 `SPEAKER1.wav/txt`；或
  - 暂不支持该语言的试听筛选。
- `_GENERIC_REF_DURATION` 当前仅中文/英文有固定时长；本功能进入目标语言试听阶段时，应补齐所有 `_GENERIC_REF_TEXTS` 覆盖语言的固定时长，避免自动估算导致样音语速不稳。

---

## 4. 现有代码复用地图

| 能力 | 现有位置 | 复用方式 |
|---|---|---|
| Notebook / tab 框架 | `tool_clonevoice/gui.py` | 新 tab 插入在 `tab_clone` 之前 |
| 转录 + manifest 写入 | `tool_clonevoice/logic.py::run_transcribe_diarize` | 强制 `diarize_backend="single"`、`num_speakers=1` |
| 批量视频扫描 | `ClonevoiceToolsApp._scan_clone_batch_videos` | 新 tab 复用 |
| 翻译 | `tool_clonevoice/logic.py::run_translate` | 最终阶段复用 |
| 合成输出 `.si.wav` | `tool_clonevoice/logic.py::run_synthesize` | 最终阶段复用，但需支持“目标语言基准音直通” |
| 参考音频候选算法 | `tool_clonevoice/refsel.py` | 扩展为返回 Top N 候选，不只是自动选一个 |
| OmniVoice 加载/生成 | `tool_clonevoice/omnivoice_backend.py` | 暴露固定样句试听生成接口 |
| ECAPA 模型 | `tool_clonevoice/diarize.py::_load_ecapa_model` | 用于相似度排序 |
| 模型状态和下载 UI | `tool_clonevoice/gui.py` | 新 tab 复用已有状态提示/下载按钮 |
| 翻译 API 配置 | `ClonevoiceToolsApp._open_translate_config` | 新 tab 复用 |

---

## 5. Step ①：转录

### UI 控件

- 输入模式：
  - 单个文件
  - 批量目录
- 输入路径：
  - 视频文件路径，或目录路径
- 源语言：
  - 自动检测 / 日语 / 英语 / 中文
- 识别模型：
  - `large-v3`
  - `large-v2`
  - `kotoba(日文)`
- 降噪：
  - 关闭 / 轻度 / 均衡 / 强力
- 目标语言：
  - 复用现有目标语言下拉
- 翻译 API 配置按钮：
  - 放在此步骤或第 ④ 步均可，建议此步骤可见，避免最后才发现未配置。

### 后端行为

单文件：

```python
logic.run_transcribe_diarize(
    video_path,
    model_key=model_key,
    language=source_language,
    diarize_backend="single",
    num_speakers=1,
    target_language=target_language,
    models_root=models_root,
    denoise=denoise,
)
```

批量目录：

- 扫描目录下所有可处理视频。
- 对每个视频生成 `<视频名>.clone/manifest.json` 和 `source.srt`。
- 所有视频都视为同一人声音，为后续共用 `SPEAKER1.wav/txt` 做准备。

### 产物

单文件：

```
<video>.clone/
  manifest.json
  audio16k.wav
  source.srt
```

批量目录：

```
<video1>.clone/...
<video2>.clone/...
...
```

---

## 6. Step ②：选择基准语音

此步骤是新 tab 的核心。

### 6.1 候选来源 A：从视频原音自动抽取

新增或扩展 `tool_clonevoice/refsel.py`：

```python
def collect_reference_candidates(
    video: str,
    manifest: dict,
    audio16k_path: str,
    clone_dir: Path,
    *,
    speaker: str = "SPEAKER_00",
    top_n: int = 12,
    log: LogCallback = print,
) -> list[dict]:
    ...
```

候选字段建议：

```json
{
  "id": "cand_001",
  "video": "...",
  "speaker": "SPEAKER_00",
  "start": 123.45,
  "end": 130.20,
  "dur": 6.75,
  "src_text": "...",
  "score": 0.873,
  "source": "turn|segment|turn_raw",
  "source_audio": "candidate_cand_001_src.wav",
  "target_sample_audio": "",
  "target_sample_text": "",
  "ecapa_similarity": null
}
```

裁剪出的原音候选建议保存到：

```
<video>.clone/single_candidates/
  cand_001_src.wav
  cand_002_src.wav
  ...
  candidates.json
```

批量目录模式下：

- 候选池来自所有视频。
- 每个视频取 Top M，再合并排序为目录级 Top N。
- UI 表格显示候选来自哪个视频。
- 最终选中的 `SPEAKER1.wav/txt` 供整个目录共用。

### 6.2 候选表 UI

表格列：

| 列 | 说明 |
|---|---|
| 排名 | 当前质量分或 ECAPA 排序 |
| 视频 | 批量目录模式显示 |
| 时间 | `start-end` |
| 时长 | 秒 |
| 原文 | ASR 文本，单行截断 |
| 质量分 | `refsel` 原有评分 |
| 相似度 | ECAPA 分数，可空 |
| 操作 | 播放原音 / 生成样句 / 播放样句 / 选为基准 |

试听方式：

- 首版建议用 `os.startfile(path)` 调用系统默认播放器。
- 不引入新的音频播放依赖。
- 后续如需要可再做内置播放器。

### 6.3 生成目标语言固定样句

新增 `tool_clonevoice/omnivoice_backend.py` 公开函数：

```python
def generic_ref_text(language: str) -> tuple[str, float]:
    return _resolve_generic_ref(language)
```

新增单样句生成函数：

```python
def generate_target_reference_sample(
    *,
    models_root: str,
    source_ref_audio: str,
    source_ref_text: str,
    target_language: str,
    output_wav: str,
    num_step: int = 32,
    guidance_scale: float = 2.0,
    instruct: str | None = None,
    log: LogCallback = print,
    stop_event: Event | None = None,
) -> tuple[str, str]:
    ...
```

行为：

- 从 `_GENERIC_REF_TEXTS` 取目标语言固定样句。
- 使用原音候选作为 `ref_audio`。
- 使用原音候选转录文本作为 `ref_text`。
- 内部生成 `takes` 条同一句目标语言试听音，默认沿用现有 `WORK_REF_TAKES = 3`；用 ECAPA 从这些 take 里选最像原始原音候选的一条。
- 对 UI 和最终文件只暴露一条最终目标语言试听音。
- 返回 `(output_wav, generic_text)`。

注意：

- 不生成三条不同样句。
- “多 take”指同一句固定样句的多次采样，不是多条文本样句。
- 若性能压力较大，可把 `takes` 做成隐藏高级参数，但默认不应低于现有自动流程的质量基线。

### 6.4 一键 ECAPA 排序

ECAPA 排序逻辑：

```
phase 1:
  load OmniVoice
  for selected/top candidates:
    generate target-language fixed-sentence sample(s)
    choose best take by ECAPA-compatible scoring data if available
  release OmniVoice

phase 2:
  load ECAPA
  for candidates with target_sample_audio:
    source_embedding = ECAPA(candidate.source_audio)
    target_embedding = ECAPA(candidate.target_sample_audio)
    similarity = cosine(source_embedding, target_embedding)
  release ECAPA
  sort desc
```

复用：

- `tool_clonevoice/diarize.py::_load_ecapa_model`
- `tool_clonevoice/omnivoice_backend.py::_ecapa_embed`
- `tool_clonevoice/omnivoice_backend.py::_ecapa_cosine`

需要把这些内部函数整理为可安全调用的公开/半公开函数，避免 GUI 直接调用私有实现。

显存纪律：

- 不要在候选循环中同时持有 OmniVoice 和 ECAPA。
- 必须沿用现有 `_build_speaker_prompts()` 的两阶段思路：先让 OmniVoice 批量生成样音并释放，再加载 ECAPA 评分并释放。
- “一键 ECAPA 排序”隐含要为多个候选生成目标语言样音，成本可能是数分钟级；UI 必须提前提示并允许限制候选数量，例如只对 Top 5 或用户勾选项执行。

外部专家需重点评估：

- 是否应把分数只作为排序辅助，而不自动选择。
- 是否需要加入质量分与 ECAPA 分数的组合排序，例如：

```
final_score = 0.35 * ref_quality + 0.65 * ecapa_similarity
```

建议首版：

- 默认只显示 ECAPA 分数和排序。
- 不自动覆盖用户选择。

---

## 7. 候选来源 B：导入用户自己的 WAV + 文本

UI：

- 选择 WAV。
- 输入或选择 TXT。
- 文本框可直接编辑。
- 按钮：
  - 播放 WAV
  - 保存为 SPEAKER1

约束：

- 用户导入的 WAV 必须是目标语言。
- 用户导入的 TXT 必须是该 WAV 的准确文本。
- UI 文案需要明确提示：最终基准语音文本必须和最终拼接语言一致。

后端：

- WAV 转为单声道、24kHz PCM，便于 OmniVoice prompt。
- 文本按 UTF-8 读取；如遇中文编码问题，按项目要求尝试 UTF-8 BOM。
- 保存为标准基准：

```
SPEAKER1.wav
SPEAKER1.txt
```

---

## 8. 候选来源 C：OmniVoice 音色设计

OmniVoice 支持 `instruct` 参数，无需参考音频。

依据：

- `reference/OmniVoice/docs/voice-design.md`
- `reference/OmniVoice/omnivoice/models/omnivoice.py::generate(..., instruct=...)`

UI 形态：

- 性别：
  - 不指定 / 男 / 女
- 年龄：
  - 不指定 / 儿童 / 少年 / 青年 / 中年 / 老年
- 音高：
  - 不指定 / 极低音调 / 低音调 / 中音调 / 高音调 / 极高音调
- 风格：
  - 不指定 / 耳语
- 英文口音：
  - 仅目标语言为英文时启用
- 中文方言：
  - 仅目标语言为中文时启用
- 自由 instruct：
  - 允许手动输入，和结构化控件合并。

行为：

- 用目标语言固定样句生成一条 `SPEAKER1.wav`。
- `SPEAKER1.txt` 保存同一条固定样句文本。
- 最终合成仍走“参考音克隆”路径，而不是每句都传 `instruct`。

这样做的好处：

- 所有来源最终统一为 `SPEAKER1.wav/txt`。
- 用户可以试听音色设计结果。
- 后续合成逻辑更简单，不需要在每个 segment 上同时管理 `instruct`。

专家讨论点：

- 音色设计生成的 `SPEAKER1.wav` 再作为参考音克隆，会不会二次漂移。
- 是否应该保留 `SPEAKER1.meta.json`，记录原始 instruct，未来可选择直接 instruct 合成。

建议首版保留：

```
SPEAKER1.meta.json
```

内容：

```json
{
  "source": "voice_design",
  "target_language": "Chinese",
  "generic_ref_text_key": "chinese",
  "instruct": "女，青年，高音调，四川话"
}
```

---

## 9. Step ③：确认 SPEAKER1

确认页显示：

- 基准 WAV 路径
- 基准 TXT 文本
- 来源：
  - 视频候选
  - 用户导入
  - 音色设计
- 目标语言
- 播放按钮
- 保存/重选按钮

必须写死的定义：

- 从视频候选生成时，`SPEAKER1.wav` 必须是“选中原音候选生成出的目标语言样音”，也就是被冻结的目标语言 work_ref。
- `SPEAKER1.txt` 必须是这段 `SPEAKER1.wav` 对应的目标语言固定样句文本。
- 不能把原始源语言候选片段直接保存成 `SPEAKER1.wav` 再配 `skip_work_ref=true`，否则会把日语/英语等源音当中文等目标语言参考音使用，正好落入 OmniVoice 跨语言短句克隆不稳定的风险区。
- 用户导入模式例外：用户导入的 WAV 本身就必须是目标语言，TXT 必须准确对应该 WAV。

### 文件落点

单文件模式：

```
<视频所在目录>/
  <视频名>.SPEAKER1.wav
  <视频名>.SPEAKER1.txt

<视频名>.clone/
  SPEAKER1.wav
  SPEAKER1.txt
  SPEAKER1.meta.json
```

批量目录模式：

```
<批量根目录>/
  SPEAKER1.wav
  SPEAKER1.txt

<video1>.clone/
  SPEAKER1.wav
  SPEAKER1.txt
  SPEAKER1.meta.json

<video2>.clone/
  SPEAKER1.wav
  SPEAKER1.txt
  SPEAKER1.meta.json
```

为什么同时保存两份：

- 视频目录/批量根目录下的用户可见副本给用户检查、替换、复用。
- 单文件模式使用 `<视频名>.SPEAKER1.wav/txt`，避免同一目录多个视频分别处理时互相覆盖。
- 批量模式才在批量根目录使用共享 `SPEAKER1.wav/txt`，表达“整目录共用一个基准音色”。
- 每个 `<视频名>.clone/` 里的副本给后端稳定读取，避免相对路径跨目录带来的问题。

### manifest 写入

每个视频 manifest 更新：

```json
{
  "speakers": {
    "SPEAKER_00": {
      "ref_audio": "SPEAKER1.wav",
      "ref_text": "...目标语言固定样句或用户文本...",
      "ref_language": "Chinese",
      "ref_kind": "target_language_basis",
      "skip_work_ref": true,
      "score": 1.0
    }
  }
}
```

说明：

- 每个 `<视频名>.clone/` 工作目录中固定使用 `SPEAKER1.wav/txt`，供 manifest 稳定引用。
- 单文件模式的用户可见副本使用 `<视频名>.SPEAKER1.wav/txt`，批量模式的用户可见共享副本使用 `SPEAKER1.wav/txt`。
- manifest 内仍使用现有 speaker id `SPEAKER_00`，避免大范围改动现有合成逻辑。
- `skip_work_ref=true` 是新字段，用于告诉 OmniVoice 后端：这个参考音已经是目标语言，不要再生成 `work_ref_SPEAKER_00.wav`。
- 旧 manifest 没有 `skip_work_ref` 字段时，后端用 `.get(...)` 读取即可保持向后兼容，不需要迁移。

---

## 10. Step ④：翻译并克隆

### 前置检查

- 已完成转录。
- 已存在并确认 `SPEAKER1.wav/txt`。
- 翻译 API Key 已配置。
- 目标语言有效。

### 后端流程

重要约束：

- 不调用 `logic.run_full()`。
- 不调用 `logic.run_extract_references()`。
- 原因：`run_full()` 会执行 `run_extract_references()`，而 `run_extract_references()` 会用自动 refsel 重新写入 `speakers.ref_audio/ref_text`，覆盖用户确认的 `SPEAKER1.wav/txt`。
- 这里必须直接走 `run_translate()` + `run_synthesize()`。

单文件：

```python
logic.run_translate(video, target_language=target_language)
logic.run_synthesize(
    video,
    models_root=models_root,
    text_field="tgt_text",
    language=target_language,
    loudness_mode=...,
    envelope_alpha=...,
)
```

批量目录：

```
for video in videos:
  apply SPEAKER1 to manifest
  translate
  synthesize
```

输出：

```
<视频名>.si.wav
```

后续回混仍使用现有「混合视频音轨」tab。

---

## 11. 需要改造的后端点

### 11.1 新增 `tool_clonevoice/single_clone.py`

建议把新流程编排放进独立模块，避免 `gui.py` 继续膨胀。

核心函数：

```python
def scan_videos(input_path: str, batch: bool) -> list[str]:
    ...

def run_single_transcribe(...):
    ...

def collect_single_candidates(...):
    ...

def generate_candidate_target_sample(...):
    ...

def score_candidate_similarity(...):
    ...

def save_speaker1_basis(...):
    ...

def apply_speaker1_to_manifest(...):
    ...

def translate_and_synthesize_single(...):
    ...
```

### 11.2 扩展 `refsel.py`

现状：

- `extract_references()` 自动选一个最佳候选，写入 manifest。

需要新增：

- 返回候选列表。
- 允许裁剪 Top N。
- 不立即写入 final `speakers.ref_audio`。

避免重复逻辑：

- 复用 `_turn_candidates`
- 复用 `_candidate_spans`
- 复用 `_score`
- 复用 `_cut_ref`

### 11.3 扩展 `omnivoice_backend.py`

需要新增公开能力：

- 获取固定目标语言样句。
- 生成目标语言试听样音：文本只有一条固定样句，但内部默认多 take 并选优，最终只保存一条样音。
- 计算 ECAPA 相似度。
- 在 `_build_speaker_prompts()` 中识别 `skip_work_ref=true`。

关键改造：

```python
if (
    text_field == "tgt_text"
    and info.get("skip_work_ref")
    and _same_language(info.get("ref_language"), language)
):
    prompts[spk] = model.create_voice_clone_prompt(str(ref_audio), ref_text, preprocess_prompt=True)
    log(f"[synth] {spk}: using target-language SPEAKER1 basis directly")
    continue
```

注意：

- 目标语言比较需要标准化，例如 `Chinese`、`chinese`、`zh`。
- 如果用户误配目标语言，应提示，不应静默二次 work_ref。
- `skip_work_ref` 只应在 `text_field="tgt_text"` 且参考音语言与目标语言一致时生效；如果后续用 `src_text` 做源语言测试合成，应自然回退到旧路径。
- 现有 `_build_speaker_prompts()` 目前没有 `text_field` 参数，实施时需要从 `synthesize()` 传入，或在调用前解析出是否为目标语言合成。
- 旧 manifest 没有 `skip_work_ref`，默认仍走原有 work_ref 两跳流程，保持向后兼容。

### 11.4 扩展 i18n

至少新增中文键：

- `tab_single_clone`
- `single_clone_note`
- `step_transcribe`
- `step_select_basis`
- `step_confirm_basis`
- `step_translate_clone`
- `btn_prev_step`
- `btn_next_step`
- `btn_run_step`
- `btn_play_source`
- `btn_generate_sample`
- `btn_play_sample`
- `btn_rank_ecapa`
- `btn_use_as_speaker1`
- `btn_import_wav`
- `btn_import_txt`
- `btn_voice_design`
- `lbl_speaker1_wav`
- `lbl_speaker1_txt`
- 错误提示若干

英文和日文可先给直译，避免语言切换缺 key。

---

## 12. UI 状态机建议

内部状态：

```python
single_state = {
  "step": 0,
  "input_mode": "single|batch",
  "videos": [],
  "target_language": "Chinese",
  "manifests_ready": False,
  "candidates": [],
  "selected_candidate_id": "",
  "speaker1_wav": "",
  "speaker1_txt": "",
  "speaker1_confirmed": False
}
```

按钮可用性：

| 步骤 | 上一步 | 下一步 | 执行当前步骤 |
|---|---|---|---|
| ① 转录 | 禁用 | 转录完成后启用 | 开始转录 |
| ② 选择基准 | 启用 | 选中或导入基准后启用 | 抽候选 / 生成样句 / ECAPA 排序 |
| ③ 确认 SPEAKER1 | 启用 | 保存成功后启用 | 保存/应用 SPEAKER1 |
| ④ 翻译并克隆 | 启用 | 禁用 | 开始翻译并克隆 |

线程策略：

- 每个重任务进入后台线程。
- 和现有「一键克隆」一样使用 `stop_event`。
- OmniVoice / ECAPA / ASR 模型释放逻辑要沿用现有主线程释放模式，避免后台线程析构崩溃。

---

## 13. 验证计划

### 单元/轻量测试

新增测试建议：

| 测试 | 目标 |
|---|---|
| `test_single_clone_generic_ref_text` | 目标语言能映射到 `_GENERIC_REF_TEXTS` |
| `test_single_clone_manifest_basis` | `SPEAKER1.wav/txt` 写入 manifest 字段正确 |
| `test_single_clone_skip_work_ref` | `skip_work_ref=true` 时不生成 `work_ref_*` |
| `test_single_clone_skip_work_ref_tgt_only` | `skip_work_ref` 仅在 `tgt_text` 目标语言合成时生效，`src_text` 路径不误用 |
| `test_single_clone_language_normalize` | `Chinese/chinese/zh` 等语言名可归一比较 |
| `test_single_clone_no_run_full_overwrite` | 单人流程不调用 `run_full/run_extract_references` 覆盖 SPEAKER1 |
| `test_single_clone_single_file_visible_name` | 单文件模式用户可见副本带视频名前缀，避免同目录覆盖 |
| `test_single_clone_candidate_sort` | 候选列表可按质量分/ECAPA 分排序 |
| `test_single_clone_batch_apply_basis` | 批量目录中所有 manifest 都应用同一 SPEAKER1 |
| `test_i18n_json_parse` | 中英日 JSON 均可解析 |

### 手工验证

1. 单视频：
   - 转录。
   - 抽取候选。
   - 生成目标语言固定样句。
   - 播放原音和样句。
   - 选择 SPEAKER1。
   - 翻译并输出 `.si.wav`。

2. 批量目录：
   - 对多个视频转录。
   - 从目录级候选池选 SPEAKER1。
   - 所有视频共用同一 SPEAKER1 生成 `.si.wav`。

3. 用户导入：
   - 导入外部 WAV/TXT。
   - 不调用候选抽取。
   - 直接翻译合成。

4. 音色设计：
   - 输入 instruct。
   - 生成 `SPEAKER1.wav/txt`。
   - 用该基准合成最终音频。

---

## 14. 风险与待外部专家讨论的问题

### 14.1 ECAPA 跨语言评分定位

ECAPA 跨语言评分不是全新机制：现有 `_build_speaker_prompts()` 已经在生产路径中比较“原始源语言参考音”与“目标语言 work_ref take”的相似度，并用它选择最佳 take。

本 tab 的风险不在于“是否完全未知”，而在于：

- ECAPA 分数只能辅助排序，不能替代人耳试听。
- 不同候选的音质、情绪、背景声会影响分数。
- 对候选列表做全量 ECAPA 排序前，必须先生成对应目标语言样音，成本较高。

建议：

- 首版只把 ECAPA 作为排序辅助。
- 不自动选择最高分。
- UI 保留人工试听决策。
- 如果后续需要自动推荐，可用 `refsel` 质量分 + ECAPA 分数的组合分，但仍要求用户确认。

### 14.2 目标语言固定样句长度

现有 `_GENERIC_REF_TEXTS` 中中文/英文有固定 duration：

- `chinese`: 13.0 秒
- `english`: 13.5 秒

其他语言通过 `_estimate_ref_duration()` 自动估算。

实施要求：

- Phase 2 必须为 `_GENERIC_REF_TEXTS` 中所有语言补齐 `_GENERIC_REF_DURATION`。
- 目标时长建议统一控制在约 8-14 秒，具体按语言文本长度微调。
- 不应依赖 `_estimate_ref_duration()` 作为用户试听样音的长期默认策略，因为语速异常会直接误导用户判断相似度。

待讨论：

- 是否需要改短部分固定样句，或仅补 duration。
- 是否需要把“评分用短样句”和“最终 SPEAKER1 固定样句”分离。首版不建议新增第二套文本，优先控制候选数量和 take 数。

### 14.3 目录模式候选池规模

问题：

- 大目录全部转录和生成样句会很慢。

建议首版：

- 转录阶段仍处理全部视频。
- 默认只对用户选中的候选生成目标语言样音。
- “一键 ECAPA 排序”作为显式重任务：执行前提示需要为多个候选生成样音，允许用户选择 Top N 范围，例如 Top 5。
- 排序任务必须两阶段执行：OmniVoice 生成并释放 -> ECAPA 评分并释放，避免显存冲突。

### 14.4 音色设计二次克隆漂移

问题：

- 使用 `instruct` 生成 `SPEAKER1.wav`，再用 `SPEAKER1.wav/txt` 克隆整片，可能有二次漂移。

替代方案：

- 最终合成阶段直接传 `instruct`，不生成 `SPEAKER1.wav`。

暂定方案：

- 仍生成 `SPEAKER1.wav/txt`，统一所有入口。
- 同时保存 `SPEAKER1.meta.json`，未来可支持 instruct 直通。

### 14.5 试听方式

首版使用系统默认播放器，优点是低风险、少依赖。

待讨论：

- 是否需要内置播放/暂停控件。
- 如果需要，选择 `sounddevice`、`pygame`、`winsound` 还是 Tk 外部播放器。

### 14.6 文件命名

用户指定类似：

```
SPEAKER1.wav
SPEAKER1.txt
```

现有 manifest speaker id 是：

```
SPEAKER_00
```

暂定：

- 每个 `<视频名>.clone/` 工作目录内固定使用 `SPEAKER1.wav/txt`，保持 manifest 简单稳定。
- 单文件模式的用户可见副本使用 `<视频名>.SPEAKER1.wav/txt`，避免同目录多视频互相覆盖。
- 批量模式的根目录共享副本使用 `SPEAKER1.wav/txt`，表示整目录共用同一个基准音色。
- manifest 内继续用 `SPEAKER_00`。
- 不改现有多说话人命名体系，降低回归风险。

---

## 15. 分阶段实施建议

### Phase 1：可用骨架

- 新增 tab。
- 新增四步 UI。
- Step ① 单文件转录可用。
- Step ② 能列出 Top N 原音候选并播放。
- 不进入最终合成，避免在尚未生成目标语言样音前误把源音保存为 SPEAKER1。

### Phase 2：目标语言固定样句试听

- 复用 `_GENERIC_REF_TEXTS`。
- 补齐 `_GENERIC_REF_DURATION`。
- 从候选原音生成目标语言固定样句：文本一条，内部默认多 take + ECAPA 选优，最终展示一条。
- 播放样句。
- 保存样句为 `SPEAKER1.wav/txt`。
- Step ④ 使用 `SPEAKER1.wav/txt` 翻译并合成，完整跑通单文件模式。

### Phase 3：ECAPA 排序

- 生成目标样句后计算相似度。
- 表格按 ECAPA 分排序。
- 分数写入 `candidates.json`。

### Phase 4：批量目录

- 目录级候选池。
- 目录级 `SPEAKER1.wav/txt`。
- 所有视频应用同一个基准音。
- 批量翻译合成。

### Phase 5：用户导入与音色设计

- 导入 WAV/TXT。
- 音色设计 instruct 控件。
- 生成并保存 `SPEAKER1.meta.json`。

---

## 16. 首版推荐范围

为避免一次性改动过大，建议首版交付：

1. 新 tab + 四步向导骨架。
2. 单文件模式完整跑通。
3. 复用现有固定目标语言样句：文本只有一条，内部多 take 选优，最终保存一条试听样音。
4. 单文件用户可见副本保存为 `<视频名>.SPEAKER1.wav/txt`，工作目录内保存为 `SPEAKER1.wav/txt`。
5. 最终直接 `run_translate()` + `run_synthesize()` 生成 `.si.wav`，不调用 `run_full/run_extract_references`。
6. ECAPA 排序可以先做候选样音内的 take 选优；全候选“一键排序”若性能风险较大，可作为 Phase 3。

不建议首版同时做：

- 内置音频播放器。
- 复杂可配置采样矩阵；默认多 take 仅用于沿用既有质量保障。
- instruct 最终合成直通。
- 目录级超大候选池自动全量样句生成。

---

## 17. 当前结论

本功能不需要推翻现有 `tool_clonevoice`，而是把现有自动 pipeline 中隐藏的“参考音筛选”和“目标语言工作参考音”显式暴露出来。

本 tab 不是新增克隆能力，而是把 `_build_speaker_prompts` 里自动的“源参考音选择 + work_ref 生成 + ECAPA 选优”前移为用户可试听、可干预的显式步骤；最终 `SPEAKER1.wav` 即一段冻结的目标语言 work_ref，`skip_work_ref=true` 让合成阶段直接复用它，跳过运行时重复生成。

关键设计点：

- 单人场景强制 `diarize_backend="single"`。
- 目标语言样句复用 `_GENERIC_REF_TEXTS`，每种语言一条，不新增三句。
- 从视频候选生成时，最终 `SPEAKER1.wav/txt` 必须是目标语言样音及其文本，不是源音片段。
- manifest 标记 `skip_work_ref=true`，避免目标语言基准音再次被转换。
- 最终阶段必须直接调用 `run_translate()` + `run_synthesize()`，不能调用 `run_full()`。
- 人耳试听优先，ECAPA 排序辅助。
