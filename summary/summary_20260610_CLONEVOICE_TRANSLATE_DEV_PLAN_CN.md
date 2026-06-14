# 语音克隆翻译（tool_clonevoice）开发方案

- 日期：2026-06-10
- 分支：`omnivoice`
- 目标：新增 `tool_clonevoice` 工具，初期实现「语音克隆翻译」——选一个视频，转录+分离说话人，提取每位说话人参考音色，AI 翻译后用 OmniVoice 克隆该说话人音色配音，拼接输出与视频对齐的配音 WAV，并可回混进视频。

---

## 1. 总体流水线

```
选视频
  → ① WhisperX 转录 + 词级对齐 + 说话人分离      → segments[]
  → ② 参考样本提取（每说话人打分选段 + 逐句情绪段） → speakers[] / 每句 emotion_ref
  → ③ AI 翻译（默认 dubbing prompt）              → 每句 tgt_text
  → ④ OmniVoice 克隆合成（硬 duration 对齐）        → 逐句短音频
  → ⑤ 单时间线拼接                                 → <video>.si.wav
  → ⑥（可选）回混进视频                            → <video>_SI.mp4
```

所有阶段读写同一份 **manifest JSON**，支持「保留中间文件」后从任意阶段续跑（调试）。

---

## 2. 已锁定的设计决策

| # | 决策点 | 结论 |
|---|---|---|
| 1 | 参考 / 情绪策略 | **混合**：默认逐句用该句源音频当 ref 带出情绪；源句太短(<3s)/太吵则回退到该说话人的固定高质量 ref |
| 2 | 时长对齐 | **硬对齐到源句时长**，用 OmniVoice 原生 `duration` 参数（不做事后变速） |
| 3 | 交付范围 | 主产物 `<video>.si.wav`；**回混做成独立 Tab**，复用 tool_si 的批量回混 |
| 4 | 说话人分离 | 默认 **ECAPA-WavLM 嵌入 + sklearn 聚类**（离线、免 token）；兜底单说话人直通；可选 pyannote 3.1（已打包，免 token） |
| 5 | UI 形态 | 单窗口、2 个 Tab；复杂翻译配置走 Dialog；含「保留中间文件」「跳过已存在」 |
| 6 | 中间产物 | 一份 manifest JSON 串联全流程，可分步续跑 |

---

## 3. 复用地图（不要重复造轮子）

| 能力 | 现成位置 | 复用方式 |
|---|---|---|
| 时间线拼接 `_mix_timeline_segment`、`fit_audio_to_duration`、`write_wav_mono`/`read_wav_mono`、CUDA 释放、批处理框架 | `tool_si/logic.py` | 抽取/直接调用 |
| 结果回混进视频（ffmpeg 多轨/duck/延迟）、`.si.wav` 命名、`collect_paired_si_mix_tasks`、`batch_mix_si_audio_tracks` | `tool_si/logic.py` | 回混 Tab 几乎照搬 |
| `LLMClient`、`load_trans_config`/`save_trans_config`、keyring、dubbing prompt | `tool_subtitle/logic.py` + `config/translate_prompt_dubbing.txt` | 直接复用 |
| 窗口/Notebook/Tab/日志/线程/停止 模式 | `tool_subtitle/gui.py` | 照搬骨架 |
| 启动器注册（`launch_*` + 按钮） | `main.py`（参考 `launch_si_voice` `main.py:576`） | 加 `launch_clonevoice` |
| HF 模型下载（snapshot_download 到 models/） | `tool_si/logic.py` `download_model` | 改 repo id |

> 重要：WhisperX 在本项目此前未使用（现有字幕走 faster-whisper）。WhisperX 三段解耦——**转录 + wav2vec2 词级对齐都不需要 token**，只有 diarization 需要 pyannote。我们核心依赖前两段做精确 ref 切片。

---

## 4. OmniVoice 接入要点（已读源码确认）

`.venv/.../omnivoice/models/omnivoice.py`：

- `OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda", dtype=torch.float16)`；`sampling_rate` 通常 24000。
- `create_voice_clone_prompt(ref_audio, ref_text, preprocess_prompt=True)` → **可复用 `VoiceClonePrompt`**。
  → 架构核心：**每位说话人的固定 ref 只算一次 prompt，批量复用**。
- `generate(text=[...], voice_clone_prompt=[...], duration=[...], language=...)` → `list[np.ndarray]`。
  - 支持批量 + 逐句 `duration`（硬对齐视频槽位）。
  - 同一批可混不同 ref（变长 padding），但**按说话人分批**最省算力（同 ref 复用）。
- 含非语言标签 `[laughter]`/`[sigh]` 等，后续可用于情绪增强。

**混合策略实现**：对每句先做质量门控（时长≥3s、能量/SNR、无叠音）；合格→用该句源音频切片构造逐句 prompt（带情绪）；不合格→回退该说话人固定 prompt。逐句 prompt 无法跨句复用，故合成耗时高于纯固定策略，属预期。

---

## 5. 说话人分离（可插拔三层）

抽象接口 `diarize(audio16k) -> List[(start, end, speaker)]`，嵌入后端抽象 `EmbeddingBackend.embed(wav16k) -> vec`，聚类（sklearn `AgglomerativeClustering`，已装）在其之上、与后端无关。

| 层 | 后端 | 依赖 / 模型 | 说明 |
|---|---|---|---|
| 默认 | **ECAPA-WavLM 嵌入 + 聚类** | `s3prl`（已装）+ `models/OmniVoice_ECAPA/`（用户下载 `k2-fsa/TTS_eval_models`） | 离线、免 token；质量优于 resemblyzer、近 pyannote |
| 兜底 | **单说话人直通** | 无 | ECAPA 不可用/失败时，整片当一个说话人，全链路仍跑通 |
| 可选 | **pyannote 3.1** | `pyannote.audio>=3.1,<4`（已锁）+ `models/speaker-diarization-3.1/`（已打包，免 token） | 多说话人质量最好 |

- UI「说话人分离」下拉：`自动 / 单说话人 / 本地聚类(ECAPA) / pyannote` + 「说话人数：自动 / 指定 N」（指定 N 显著提升聚类准确率）。
- **pyannote 3.1 加载坑**：`models/speaker-diarization-3.1/config.yaml` 用**相对路径**引用 `segmentation/`、`embedding/`；pyannote 3.x 不保证按 config 目录解析，**加载器须在运行时按实际安装目录改写成绝对路径**再 `Pipeline.from_pretrained()`；**不可写死绝对路径**（ZIP 各机解压位置不同）。

---

## 6. 中间产物 manifest（保留中间文件 + 分步续跑）

```json
{
  "video": "D:/x/a.mp4",
  "language": "Japanese",
  "target_language": "Chinese",
  "diarize_backend": "ecapa",
  "ref_strategy": "hybrid",
  "speakers": {
    "SPEAKER_00": { "ref_audio": "a.clone/spk00.wav", "ref_text": "...", "score": 0.91 }
  },
  "segments": [
    { "id": 1, "start": 1.20, "end": 3.40, "dur": 2.20,
      "speaker": "SPEAKER_00", "src_text": "...", "tgt_text": "...",
      "emotion_ref": "a.clone/seg_0001.wav" }
  ]
}
```

中间文件目录建议 `<video>.clone/`：`manifest.json` + 参考 wav + 逐句情绪 wav（+ 调试可选每说话人 stem）。最终配音写 `<video>.si.wav`（与 tool_si 回混直接兼容）。

---

## 7. UI 布局

**Tab ①「语音克隆翻译」**（单窗口线性流程，贴合现有 ~840px）：

```
输入视频 [____________________] [浏览]
① 转录+说话人分离  模型▾ 源语言▾  分离▾(自动/单人/ECAPA/pyannote) 人数▾  [高级…]
② 参考样本  策略 ○固定 ○逐句情绪 ◉混合
   [Treeview] Speaker│时长│文本│[试听][重选]
③ AI 翻译  目标语言▾  prompt: dubbing(默认)  [翻译高级配置…(Dialog)]
④ 合成输出  num_step guidance ☑硬时长对齐  输出[__]
☑ 保留中间文件   ☑ 跳过已存在
[一键执行] [停止]   分步:[①转录][②参考][③翻译][④合成]
[ 日志 Text ]
```

**Tab ②「音频回混」**：复用 tool_si 的 SI 回混 Tab——目录批量扫描「视频 + 同名 `.si.wav`」→ `batch_mix_si_audio_tracks`，参数（混入声道/原音量/配音音量/延迟/独立轨/duck）沿用。

复杂翻译配置（API URL/Key/模型/Max tokens/prompt 选择等）收进 `[翻译高级配置…]` Dialog，复用 tool_subtitle 字段与 keyring。

---

## 8. 目录结构

```
tool_clonevoice/
  __init__.py
  gui.py                 # ClonevoiceToolsApp(root, on_return) —— Notebook，2 Tab
  logic.py               # 编排：阶段调度 + manifest 读写 + 单时间线拼接
  whisperx_backend.py    # 转录 + 词级对齐（无 token）
  diarize.py             # EmbeddingBackend(ECAPA/单人/pyannote) + 聚类
  refsel.py              # 参考打分选段 + 逐句情绪段切片
  omnivoice_backend.py   # 模型加载 + 混合策略 + duration 硬对齐合成
```

启动器：`main.py` 加 `launch_clonevoice`，字幕组加按钮。i18n：`i18n.translate('clonevoice', key)`。

---

## 9. 依赖与模型（环境已就绪）

**pyproject（`uv lock` 已通过，无 override）**：
```
whisperx, omnivoice, s3prl>=0.4.0, scikit-learn>=1.4, pyannote.audio>=3.1,<4
```
- 关键：pyannote 钉 **3.x**（非 4.x）——4.x→pyannote-metrics 4.x→numpy>=2.2.2，与 GPU 栈 numpy<2.1 硬冲突且会拽降 whisperx；3.x 与现环境干净共存。

**已打包模型（`models/`，均 gitignore，仅随发布 ZIP）**：
- `models/OmniVoice`（TTS 主模型，已在）
- `models/OmniVoice_ECAPA/`（分离嵌入，含 `get_ECAPA.txt`；用户需下 `k2-fsa/TTS_eval_models` 的 `wavlm_large_finetune.pth` + `wavlm_large/wavlm_large.pt`）
- `models/speaker-diarization-3.1/`（pyannote 可选路径，32M，已组装，含 `get_speaker-diarization-3.1.txt`）

---

## 10. 分阶段实施计划

| 阶段 | 内容 | 验收 |
|---|---|---|
| **Spike** | s3prl 在源码 + **PyInstaller 冻结环境**都能加载 ECAPA-WavLM 出嵌入；pyannote 3.1 离线加载本地 bundle（含相对路径→绝对改写） | 两后端各对一段测试音频出向量/RTTM |
| **P0 骨架** | `tool_clonevoice/{__init__,gui,logic}.py` + 启动器入口 + 2 个空 Tab + 日志/线程/停止/返回（照搬 tool_subtitle） | 菜单进得去、能返回 |
| **P1 转录+分离** | `whisperx_backend.py`（转录+词对齐）+ `diarize.py`（先单说话人直通，再 ECAPA 聚类）→ 落 manifest + 中间文件 | 出 segments + speaker 标签 |
| **P2 参考提取** | `refsel.py` 打分选段 + 逐句情绪段；Treeview 试听/重选 | 每说话人有合格 ref，可试听 |
| **P3 翻译** | 接 `LLMClient` + dubbing prompt + 翻译高级 Dialog → 填 tgt_text | manifest 含译文，可续跑 |
| **P4 合成** | `omnivoice_backend.py`（加载/混合策略/duration 硬对齐）+ 单时间线拼接 → `<video>.si.wav` | 配音 WAV 与视频对齐 |
| **P5 回混 + pyannote** | 回混 Tab（复用 tool_si）；接通 pyannote 3.1 可选路径 | 出 `_SI.mp4`；pyannote 可选可用 |

---

## 11. 风险与注意

- **s3prl 打包**：冻结环境 torch.hub 本地加载 hubconf 可能出问题——Spike 先验证，不通过则保持单说话人兜底，功能不阻塞。
- **显存**：WhisperX(large-v3)+分离 与 OmniVoice(LLM)+tokenizer 不同时常驻；**分阶段加载/释放**（沿用 tool_si 的 `gen_holder`+`empty_cache`）。测试机 16GB 串行可行。
- **成人内容**：BGM/喘息/叠音影响分离与 ref 纯净度——ref 选段优先对白清晰、无叠音段。
- **硬 duration 对齐**：译文过长会语速过快——dubbing prompt 已要求等长，必要时给「轻微超时容忍 + 兜底拉伸」开关。
- **pyannote 相对路径**：见 §5，运行时改写为绝对路径。

---

## 12. 当前进度

- [x] 方案多轮敲定（本文档）
- [x] 依赖加入 pyproject 并 `uv lock` 通过（s3prl / scikit-learn / pyannote.audio 3.x）
- [x] 打包模型就位：`models/speaker-diarization-3.1/`（已组装）、`models/OmniVoice_ECAPA/get_ECAPA.txt`
- [ ] Spike：s3prl 冻结环境 + pyannote 离线加载
- [ ] P0~P5 实施
