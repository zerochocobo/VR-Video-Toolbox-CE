# tool_clonevoice「多人语音克隆」Tab 开发方案（交付研发版）

日期：2026-07-03（2026-07-03 更新：锁定首轮范围三项决策）
状态：方案设计，首轮范围已确认，可进入实现
目标：在 `tool_clonevoice` 新增「多人语音克隆」tab。当一个视频里有多个说话人时，允许用户**逐个说话人**挑选/导入/设计基准语音，再一次性翻译并克隆输出 `.SI.WAV`。

本方案面向接手的其他研发人员，尽量给到可直接落地的函数签名、文件落点、UI 结构与状态机。

> **首轮（Phase 1）已确认范围（一句话）：单文件 + 强制翻译(开工前预检) + 视频候选选择 + 跳过说话人。导入 WAV / 音色设计 → Phase 2（音色设计如工期允许可提前）。** 三项决策详见 2.1。

---

## 1. 背景

现有两个相关 tab（`tool_clonevoice/gui.py`）：

- **单人语音克隆**（`_setup_single_clone_tab`，[gui.py:298](../tool_clonevoice/gui.py#L298)）：4 步向导，强制 `diarize_backend="single"`，全片视为一个说话人 `SPEAKER_00`，最终只产出一个 `SPEAKER1` 基准音。
- **一键克隆**（`_setup_clone_tab`，[gui.py:112](../tool_clonevoice/gui.py#L112)）：自动多说话人分离 + 自动选参考音 + 翻译 + 合成，全自动，不暴露参考音筛选。

多人克隆要做的是「一键克隆的多说话人能力」+「单人克隆的可视化人工挑选/导入/设计基准音」的结合：**保留多说话人分离，但让用户对每个说话人显式确认基准语音**。

关键复用事实（已核对代码）：

- 多说话人分离本就支持：`logic.run_transcribe_diarize(..., diarize_backend="auto|pyannote|ecapa", num_speakers=N)` 会在 `manifest["speakers"]` 里产出 `SPEAKER_00 / SPEAKER_01 / ...` 并给 `segments` 打上 speaker（[logic.py:155](../tool_clonevoice/logic.py#L155)、`diar.assign_speakers`）。
- 候选参考音抽取**已按 speaker 参数化**：`refsel.collect_reference_candidates(..., speaker=spk, top_n=..., output_dir_name=...)`（[refsel.py:359](../tool_clonevoice/refsel.py#L359)）。
- **最终合成本来就是多说话人**：`logic.run_synthesize` → `omnivoice_backend._build_speaker_prompts` 会遍历 `manifest["speakers"]` 逐个建 prompt，并已支持 `skip_work_ref`（目标语言基准音直通，[omnivoice_backend.py:838](../tool_clonevoice/omnivoice_backend.py#L838)）。也就是说，只要 manifest 里每个 speaker 都填好了 `ref_audio/ref_text/skip_work_ref`，合成阶段无需任何改动。

**结论：多人克隆 90% 的后端能力已经存在。主要工作量在 UI（逐说话人挑选）和把「单说话人 basis 写入」泛化为「多说话人逐个写入」。**

---

## 2. 范围与非目标

### 2.1 首轮（Phase 1）已确认范围

整体是 3 步向导：① 转录（多说话人分离）② 逐说话人选择基准音 ③ 翻译并克隆。STEP① 与 STEP③ 复用单人 tab 逻辑，STEP② 为核心新 UI（一个说话人一行）。

**三项经确认的首轮决策：**

1. **只做单文件**：批量目录 + 全局分离（第 5.5 节）留到 Phase 4，本轮不碰。
2. **强制翻译 API，但采用「开工前预检」**：点「开始转录」时**先校验翻译配置**，缺 key/endpoint 立即弹窗提示去配置、**不启动转录**（避免单人 tab 那种「转录跑完才报错」的白等）。预检通过再进转录。理由：STEP② 的核心价值（翻译句二次克隆试听 + 翻译句 vs 原音相似度）在没 key 时会塌回「固定样句 vs 原音」的降级体验，更迷惑用户，故强制。详见 5.1。
   - 可选顺手项：同一预检回填单人 tab，修掉它「转录后才报错」的老毛病（不阻塞本期）。
3. **STEP② 首轮只做「视频候选选择 + 跳过说话人」**：
   - **「跳过该说话人」是 Phase 1 的硬约束，不可省**。多说话人天然有次要说话人候选很差或为空，而 STEP② 门控要求「所有 speaker 都有基准」；没有逃生口会被一个空候选 speaker 卡死整个流程。跳过的 speaker 不写基准、不合成，其段落保留原声（合成端遇到无 `ref_audio` 的 speaker 本就 `skipped`，天然兼容）。
   - 导入 WAV / 音色设计 → **Phase 2**。其中**音色设计成本很低**（小对话框 + 复用 `generate_voice_design_basis_with_model` 一个函数），若想让「主要说话人但候选差」也有出路，可视工期提前到 Phase 1；导入 WAV 留 Phase 2。

> 基准音三种来源的完整设计（视频候选二次克隆试听 + ECAPA / 导入 WAV+文本 / OmniVoice 音色设计）与单人 tab 一致，均在下文详述；Phase 1 先落「视频候选 + 跳过」，其余来源按上面的阶段推进。

### 2.2 批量目录模式 —— 用「全局分离」支持

单人 tab 的批量里所有视频共用同一个人声，一个 `SPEAKER1` 打通所有视频。多说话人的难点是：diarization 的 speaker id 只在**单个视频内**有意义——视频 A 的 `SPEAKER_00` 和视频 B 的 `SPEAKER_00` 不保证是同一个人。

**解决方案：全局分离（global / prescan diarization）。** 把整批视频的 16k 音频**拼接成一条**，只做**一次**分离，让分离模型对整批音频统一聚类打标签，得到跨视频一致的 `SPEAKER_XX`，再按偏移量把 turns 拆回每个视频。这样把「跨视频身份对齐」交给分离模型本身，无需自己写声纹聚类。详见第 5.5 节。

**分阶段：**
- **Phase 1 先做单文件**（跑通全链路，风险最低）。
- **Phase 4 加批量**，只需新增一个 `prescan_global_diarize` 模块 + 给 `run_transcribe_diarize` 加一个 `precomputed_turns` 钩子；候选与合成端复用单文件全部逻辑。

批量下每个全局 speaker 就等价于单人 batch：从**所有视频**汇集候选、选一个基准、应用到该 speaker 的所有视频段落。

### 2.3 非目标（首版不做）

- 说话人合并/拆分的可视化编辑（over/under-diarization 的人工纠正）。首版靠 `num_speakers` 控制数量；「合并两个说话人」作为批量场景的后期兜底（见第 5.5 节风险）。
- 内置音频播放器（沿用 `os.startfile` / 现有 inline 播放）。

---

## 3. UI 总体方案

### 3.1 Tab 插入位置

在 [gui.py:100-110](../tool_clonevoice/gui.py#L100) 的 notebook 装配处，插到「单人语音克隆」之后、「一键克隆」之前：

```
单人语音克隆 | 多人语音克隆 | 一键克隆 | 混合视频音轨
```

```python
self.tab_multi_clone = ttk.Frame(self.notebook, padding=10)
self.notebook.add(self.tab_multi_clone, text=get_text("tab_multi_clone"))
self._setup_multi_clone_tab(self.tab_multi_clone)
```

### 3.2 三步向导

复用单人 tab 的步骤骨架（`step_bar` + `nav` 上一步/下一步/停止 + `content` 里多个 step frame + 底部统一 log），见 `_setup_single_clone_tab`（[gui.py:298](../tool_clonevoice/gui.py#L298)）与 `_show_single_clone_step`（[gui.py:810](../tool_clonevoice/gui.py#L810)）。

```
① 转录（多说话人）
② 逐说话人选择基准音
③ 翻译并克隆
```

首行备注（`multi_clone_note`）：
> 当视频中有多个说话人时，为每个说话人分别挑选/导入/设计基准语音，再统一翻译克隆。

底部统一按钮沿用：`[上一步] [下一步] [停止]`，加上每步各自的执行按钮（放在 step frame 内）。

**STEP① 控件**（相对单人 tab 的差异）：单人 tab 强制 single、无分离控件；多人 tab 需在 STEP① 增加：

- 输入模式：单文件 / 批量目录（批量对应 5.5 全局分离）。
- 分离后端：复用一键克隆的 `_diar_map`（auto / ecapa / pyannote，[gui.py:205](../tool_clonevoice/gui.py#L205)）。
- 说话人数 `num_speakers`：复用 `_num_map`（auto / 1–5，[gui.py:210](../tool_clonevoice/gui.py#L210)）。批量模式下文案改为「整批总人数」（`lbl_global_num_speakers`）。
- 其余（源语言 / 模型 / 降噪 / 目标语言 / 翻译配置 / 开始转录）与单人 STEP① 一致。

### 3.3 STEP② 布局（核心新 UI）

顶部一行操作条：

```
[识别说话人/刷新]   每个说话人候选数上限 [12]
```

下面是**说话人列表区**（可滚动）。一个说话人一行：

```
┌────────────────────────────────────────────────────────────────────────────┐
│ SPEAKER_00  ⏱ 片中 42.1s / 18段   [选择基准语音] [导入WAV] [设计音色]  ✅ 视频候选  “你好，很高…” ▶ │
│ SPEAKER_01  ⏱ 片中 15.6s / 7段    [选择基准语音] [导入WAV] [设计音色]  ⛔ 未选择            │
│ SPEAKER_02  ⏱ 片中  3.2s / 2段    [选择基准语音] [导入WAV] [设计音色]  ✅ 音色设计  “女, 青年”  ▶ │
└────────────────────────────────────────────────────────────────────────────┘
```

每行控件（建议做成一个可复用的 row 构造函数 `_build_multi_speaker_row(parent, speaker_id, stats)`）：

- 说话人 id 标签（`SPEAKER_00`）。
- 该说话人在片中的统计（总时长 / 段数），来自 manifest segments，帮助用户判断谁是主角、谁是杂音。
- `[选择基准语音]`：打开**候选选择对话框**（见第 6 节），作用域限定该 speaker。
- `[导入WAV]`：弹小对话框选 WAV + 填写文本。
- `[设计音色]`：弹小对话框填 instruct（性别/年龄/音高/风格 + 自由文本）。
- 状态标签：`未选择 / 已选(来源类型) + ref_text 前若干字`，右侧一个 `▶` 试听已选基准音。
- **`跳过该说话人` 复选（Phase 1 必做）**：勾选后该 speaker 不写基准、不合成，其段落保留原声。用于次要/空候选说话人，避免门控卡死（见 2.1 决策 3）。

STEP② 的「下一步」仅当**每个说话人都「已确认基准音」或「已勾选跳过」**时可用。

> Phase 1 每行入口只有 `[选择基准语音]`（视频候选）+ `跳过` 复选；`[导入WAV]`、`[设计音色]` 是 Phase 2（音色设计可视工期提前）。UI 建议先把三个按钮位都摆好，Phase 1 里未做的先 `disabled` 并标注，减少后续布局改动。

---

## 4. 复用 / 重构地图

| 能力 | 现有位置 | 多人 tab 复用方式 |
|---|---|---|
| 步骤向导骨架 | `_setup_single_clone_tab` / `_show_single_clone_step` | 复制精简为 3 步 |
| 忙碌态/按钮禁用 | `_set_single_clone_busy`（[gui.py:827](../tool_clonevoice/gui.py#L827)） | 复用同款模式 |
| 后台线程 + 主线程释放模型 | `_single_clone_run_async`（[gui.py:904](../tool_clonevoice/gui.py#L904)） | **原样复用**（holder + `release_holder_on_main_thread`） |
| OmniVoice 加载封装 | `_load_single_clone_omnivoice_model`（[gui.py:～1050]） | 复用 |
| 转录（多说话人） | `logic.run_transcribe_diarize` | backend 不再强制 single；批量加 `precomputed_turns` 钩子（5.5.3） |
| 分离（可复用于全局） | `diar.diarize` / `diar.assign_speakers` / `wx.extract_audio_16k` | 批量全局分离 prescan 直接调用（5.5.2） |
| 翻译幂等 | `single_clone.ensure_translated_for_videos`（[single_clone.py:397](../tool_clonevoice/single_clone.py#L397)） | 原样复用 |
| 候选抽取（按 speaker） | `refsel.collect_reference_candidates` | 传 `speaker=` + `output_dir_name=` |
| 候选目标样句/二次克隆试听/ECAPA | `single_clone.build_candidate_target_sample_job` / `finish_candidate_target_sample_jobs` / `generate_candidate_translated_previews_with_model` / `score_candidate_similarities`（[single_clone.py:155](../tool_clonevoice/single_clone.py#L155) 起） | **原样复用**（这些函数是 speaker 无关的，只吃 candidate dict） |
| 音色设计 | `single_clone.generate_voice_design_basis_with_model`（[single_clone.py:328](../tool_clonevoice/single_clone.py#L328)） | 复用（输出路径改为按 speaker） |
| basis 写入 manifest | `single_clone.save_speaker1_basis` / `_copy_basis_to_clone_dir`（[single_clone.py:220](../tool_clonevoice/single_clone.py#L220)） | **需泛化**（见 5.2，不可原样用） |
| 最终翻译+合成 | `single_clone.translate_and_synthesize`（[single_clone.py:303](../tool_clonevoice/single_clone.py#L303)） | **原样复用**（合成本就遍历所有 speaker） |
| 候选表格 UI + 逐行操作 | 单人 tab 的 `single_clone_candidate_tree` 一整套（[gui.py:538](../tool_clonevoice/gui.py#L538) 起） | 抽成可复用组件（见 6.3） |

---

## 5. 后端改造

建议**新建 `tool_clonevoice/multi_clone.py`**，编排多人流程；候选级别的重活直接调用 `single_clone` 里已有的 speaker 无关函数，不重复实现。

### 5.1 转录 + 说话人枚举

```python
# multi_clone.py
def run_multi_transcribe(
    video_path, *, model_key, language, target_language, models_root,
    diarize_backend="auto", num_speakers=None, denoise="none",
    precomputed_turns=None,   # 批量全局分离拆回的 turns；单文件为 None
    log=print, stop_event=None, model_holder=None,
) -> dict:
    """多说话人转录：不强制 single。返回 manifest。
    单文件时 precomputed_turns=None → 内部逐视频分离；
    批量时传入全局分离拆回的 turns → 跳过内部分离(见 5.5.3)。"""
    return logic.run_transcribe_diarize(
        video_path, model_key=model_key, language=language,
        diarize_backend=diarize_backend, num_speakers=num_speakers,
        target_language=target_language, models_root=models_root,
        denoise=denoise, precomputed_turns=precomputed_turns,
        log=log, stop_event=stop_event, model_holder=model_holder,
    )

def list_speakers(video_path) -> list[dict]:
    """从 manifest 读取说话人及统计：[{speaker, total_dur, seg_count}]，按总时长降序。"""
```

STEP① 转录后，与单人一致地跑 `ensure_translated_for_videos([video], target_language=...)`，让候选能拿到 `tgt_text`（二次克隆试听 + 相似度都依赖它）。见单人 tab 的 [gui.py:1131](../tool_clonevoice/gui.py#L1131)。

> **翻译 API：强制 + 开工前预检（已确认，见 2.1 决策 2）。** 不采用「非致命降级」——因为 STEP② 的翻译试听/相似度在没 key 时会塌回「固定样句 vs 原音」，反而更迷惑用户。实现方式：
>
> - 在「开始转录」的 handler **最前面**先校验翻译配置（复用 `_open_single_translate_config` / `_refresh_single_translate_config_status` 读取的同一份 config：有 endpoint/model + keyring 里有 key）。缺失则 `messagebox` 提示「请先配置翻译 API」并聚焦到翻译配置按钮，**直接 return，不启动任何转录**。
> - 预检通过后再进转录 + `ensure_translated_for_videos`。这样是硬要求但 fail-fast，避免单人 tab「转录跑完才报错」的白等。
> - 参考单人 tab 的翻译状态刷新 `_refresh_single_translate_config_status`（[gui.py](../tool_clonevoice/gui.py#L669) 附近），把「未配置」红字提示常驻 STEP①。

### 5.2 逐说话人候选抽取

```python
def collect_speaker_candidates(video_path, speaker, *, top_n=12, log=print) -> list[dict]:
    video = Path(video_path)
    manifest = logic.load_manifest(video)
    cdir = logic.clone_dir(video)
    audio16k = cdir / logic.AUDIO16K_NAME
    return refsel.collect_reference_candidates(
        str(video), manifest, str(audio16k), cdir,
        speaker=speaker,
        top_n=top_n,
        output_dir_name=f"candidates_{speaker}",   # ← 关键：每个 speaker 独立目录，避免 candidates.json / cand_xxx 冲突
        log=log,
    )
```

- `candidate["speaker"]` 由 refsel 写入；`source_audio` 落在 `<video>.clone/candidates_SPEAKER_00/` 下，天然隔离。
- 生成目标样句 / 二次克隆试听 / ECAPA 打分：**直接复用** `single_clone.build_candidate_target_sample_job` → `finish_candidate_target_sample_jobs` → `generate_candidate_translated_previews_with_model` → `score_candidate_similarities`，无需改动（它们只认 candidate dict，不关心哪个 speaker）。

### 5.3 basis 写入（泛化，最关键的后端改动）

现有 `_copy_basis_to_clone_dir`（[single_clone.py:220](../tool_clonevoice/single_clone.py#L220)）有两个**不能用于多人**的硬编码：

1. 固定写 `SPEAKER_ID = "SPEAKER_00"` 一个 speaker；
2. `for seg in segments: seg["speaker"] = SPEAKER_00` —— 把所有段落塌缩成一个说话人。

多人必须新增一个「只更新指定 speaker、绝不塌缩 segments」的版本：

```python
def save_speaker_basis(
    video_path, speaker, *, basis_wav, basis_text, target_language,
    source_kind, meta=None, log=print,
) -> None:
    video = Path(video_path)
    cdir = logic.clone_dir(video); cdir.mkdir(parents=True, exist_ok=True)

    ref_wav = f"{speaker}.basis.wav"
    ref_txt = f"{speaker}.basis.txt"
    ref_meta = f"{speaker}.basis.meta.json"
    shutil.copyfile(basis_wav, cdir / ref_wav)
    (cdir / ref_txt).write_text(basis_text, encoding="utf-8")
    (cdir / ref_meta).write_text(json.dumps({**(meta or {}),
        "source": source_kind, "target_language": target_language,
        "basis_wav": ref_wav, "basis_txt": ref_txt}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    manifest = logic.load_manifest(video)
    manifest["target_language"] = target_language
    manifest.setdefault("speakers", {})[speaker] = {
        "ref_audio": ref_wav,
        "ref_text": basis_text,
        "ref_language": target_language,
        "ref_kind": "target_language_basis",
        "skip_work_ref": True,
        "score": 1.0,
        "source": source_kind,
    }
    # ⚠️ 不要改 segments 的 speaker 归属！保留 diarization 结果。
    logic.save_manifest(video, manifest)

def all_speakers_have_basis(video_path, *, ignore=()) -> bool:
    """除被跳过的 speaker 外，manifest 里每个 speaker 都有 ref_audio。用于 STEP② 门控。"""
```

要点：
- 每个 speaker 的基准音文件命名用 speaker id 前缀（`SPEAKER_00.basis.wav`），互不覆盖。
- 只更新 `manifest["speakers"][speaker]` 一项，其它 speaker 保持不变。
- **不触碰 segments**，保证合成时每段找到自己的 speaker prompt。
- 用户导入的 WAV 无需手动转采样率/声道：OmniVoice `create_voice_clone_prompt` 内部会下混单声道并重采样到模型采样率（已在单人评审中验证 `reference/OmniVoice/omnivoice/utils/audio.py`）。仅建议导入时用 `soundfile.info()` 做一次可解码性校验，提早报错。

### 5.4 最终翻译 + 合成

**直接复用** `single_clone.translate_and_synthesize([video], target_language=..., models_root=..., loudness_mode=..., envelope_alpha=..., skip_existing=...)`（[single_clone.py:303](../tool_clonevoice/single_clone.py#L303)）。它内部 `run_translate` + `run_synthesize(text_field="tgt_text")`，而 `_build_speaker_prompts` 会：

- 对每个填了 `skip_work_ref=True` 且 `ref_language==target` 的 speaker → 直接用基准音做 prompt（[omnivoice_backend.py:840](../tool_clonevoice/omnivoice_backend.py#L840)）；
- 对没设 basis 的 speaker（若允许跳过）→ 记 `missing ref_audio, skipped`，该 speaker 的段落不出声（保留原声，需在混音 tab 处理）。

STEP③ 的 UI（响度模式、envelope 强度、跳过已存在 `.SI.WAV`、开始按钮）与单人 STEP④ 一模一样，直接照搬 [gui.py:607-644](../tool_clonevoice/gui.py#L607)。

### 5.5 批量：全局分离 prescan（Phase 4）

单文件跑通后，批量只需在 STEP① 之前插一个**全局分离**阶段，产出跨视频一致的说话人标签；其余（候选、basis 写入、合成）复用单文件逻辑。

#### 5.5.1 核心思想

把整批视频的 16k 音频顺序拼成一条，视频间插静音间隔；对这条大 wav 只做**一次** `diar.diarize`，得到全局一致的 turns，再按每个视频的偏移量拆回。转录仍逐视频做（转录不需要全局视角，内存可控），只有分离走全局。

#### 5.5.2 新增模块 `multi_clone.py::prescan_global_diarize`

```python
GLOBAL_SEAM_SILENCE_S = 1.0   # 视频之间插入的静音间隔，避免 turn 横跨拼接缝

def prescan_global_diarize(
    videos: list[str], *, models_root: str, num_speakers: int | None,
    diarize_backend: str = "pyannote", device: str = "cpu",
    log=print, stop_event=None,
) -> dict[str, list[tuple[float, float, str]]]:
    """整批全局分离，返回 {video -> 本地 turns(全局一致标签)}。

    步骤：
      1. 逐视频抽 16k 音频(复用 wx.extract_audio_16k，落在各自 clone_dir，
         后续 run_transcribe_diarize 可复用，避免重复抽取)。
      2. 顺序拼接为一条临时 wav，视频 i 占 [off_i, off_i+dur_i)，
         之间插 GLOBAL_SEAM_SILENCE_S 静音；记录 offsets。
      3. diar.diarize(concat_wav, backend=..., num_speakers=<全局人数>, ...)
         -> 全局 turns (gstart, gend, SPEAKER_XX)。
      4. 拆回：对落在 [off_i, off_i+dur_i) 内的全局 turn，
         local=(max(gstart,off_i)-off_i, min(gend,off_i+dur_i)-off_i, spk)，
         丢弃时长 < 0.2s 的碎片；按 seam 硬切断跨缝 turn。
      5. 删除临时拼接 wav。
    """
```

拆分参考实现（关键循环）：

```python
def _split_turns_to_video(global_turns, off, dur, *, min_dur=0.2):
    out = []
    lo, hi = off, off + dur
    for gs, ge, spk in global_turns:
        s, e = max(gs, lo), min(ge, hi)
        if e - s >= min_dur:
            out.append((round(s - off, 3), round(e - off, 3), spk))
    return out
```

#### 5.5.3 给 `run_transcribe_diarize` 加 `precomputed_turns` 钩子（logic.py 小改）

现状 [logic.py:212](../tool_clonevoice/logic.py#L212) 内部自己 `diar.diarize`。加一个可选参数即可让它接受外部算好的 turns：

```python
def run_transcribe_diarize(..., precomputed_turns: Optional[list] = None):
    ...
    resolved_backend = diar.resolve_backend(diarize_backend, models_root)
    if precomputed_turns is not None:
        turns = precomputed_turns            # 用全局拆回的 turns
        resolved_backend = "global"          # 标记来源，便于日志/排查
    else:
        turns = diar.diarize(str(audio_wav), backend=diarize_backend, ...)
    ...
```

> 这是唯一需要动 `logic.py` 的地方，且对现有单文件/一键克隆完全向后兼容（默认 `None` 走原路径）。

#### 5.5.4 批量流程编排

```
STEP①（批量）：
  videos = scan_dir(input_dir)
  turns_by_video = prescan_global_diarize(videos, num_speakers=全局人数, ...)
  for video in videos:
      run_multi_transcribe(video, ..., precomputed_turns=turns_by_video[video])
  ensure_translated_for_videos(videos, ...)   # 逐视频翻译(非致命)

STEP②（批量）：
  speakers = 全局说话人并集(遍历所有 manifest 的 speakers 取并集)
  每个全局 speaker 一行：
    - 候选来自「所有含该 speaker 的视频」，合并排序取 Top N
      （复用单人 batch 的做法：collect_speaker_candidates 逐视频 + 归并）
    - 选定基准后，对每个含该 speaker 的视频调 save_speaker_basis(video, speaker, ...)

STEP③（批量）：
  translate_and_synthesize(videos, ...)   # 每个视频各自多说话人合成
```

#### 5.5.5 批量特有的坑（务必落实）

1. **全局人数**：UI 上批量模式的 `num_speakers` 语义是「整批一共几个人」，不是每个视频。文案要区分（`lbl_global_num_speakers`）。auto 可用但建议显式，数据更多时全局聚类比逐视频更稳。
2. **内存/耗时**：拼接 wav 会很长（16k 单声道 1 小时 ≈ 115MB + pyannote 内部开销）。中等批量 OK；给一个总时长上限告警（如 > 2h 提示可能慢/OOM），超限可提示分批。
3. **拼接缝**：必须插 `GLOBAL_SEAM_SILENCE_S` 静音并按 seam 硬切，否则 turn 跨视频。
4. **录音条件差异**：不同视频麦克风/房间差异大时，同一人可能被拆成两个 cluster。这是跨视频固有风险，拼接法已是单模型下最抗漂移的做法；后期兜底提供「把全局 SPEAKER_A 与 SPEAKER_B 视为同一人合并」的手动开关。
5. **说话人只在部分视频出现**：全局 speaker 的并集里，某 speaker 可能只在部分视频有段落——没段落的视频不为它写 basis、不合成即可，天然兼容。`save_speaker_basis` 只对「该视频 manifest 里确有此 speaker」的视频调用。
6. **候选归集**：`collect_speaker_candidates` 需带 `output_dir_name=f"candidates_{speaker}"`，且候选 dict 里 `video` 字段标明来自哪个视频（UI 表格已有该列），方便用户判断。

---

## 6. 候选选择对话框（STEP② 的「选择基准语音」）

单人 tab 把候选表格与全部逐行操作（播放原音 / 播放翻译试听 / 播放样句 / 采纳）都绑死在 `self.single_clone_*` 上。多人 tab 每个 speaker 都要这套，**强烈建议先把它抽成可复用组件**，避免复制粘贴四份状态。

### 6.1 推荐：抽出 `CandidateBasisPanel` 组件

新建一个不依赖具体 tab 的类（可放 `tool_clonevoice/gui_candidate_panel.py`）：

```python
class CandidateBasisPanel:
    """一个说话人的候选选择面板：候选表 + 生成/试听/ECAPA/采纳。

    依赖注入，避免耦合具体 tab：
      - run_async(worker, done): 复用宿主的后台线程执行器
      - load_model(holder): 复用宿主的 OmniVoice 加载
      - log(msg): 日志回调
      - models_root, target_language 提供者
    产出：on_basis_chosen(speaker, wav_path, text, source_kind, meta)
    """
```

它内部完全复用 `single_clone` 的候选函数（第 5.2 节列的四个）。这样单人 tab 未来也能切换到同一组件，减少长期维护成本。

> 若工期紧，允许**先在多人 tab 内复制**单人的候选表逻辑（约 200 行），但需在代码注释里标注「与 single_clone 候选逻辑重复，后续合并」，并登记技术债。

### 6.2 对话框形态

`[选择基准语音]` 点击后弹一个 `Toplevel` 模态窗（标题带 speaker id），内含 `CandidateBasisPanel`：

```
选择 SPEAKER_00 的基准语音
每个说话人候选数上限 [12]   [抽取并生成样句]
┌ 播放原音 | 播放翻译试听 | 播放样句 | 排名 | 时间 | 时长 | 质量分 | 相似度 | 原文 ┐
│  ▶        ▶            ▶        1     ...                                      │
└──────────────────────────────────────────────────────────────────────────────┘
[采纳选中为基准]                                            [取消]
```

- 「抽取并生成样句」= 单人的 `_single_clone_collect_generate_samples` 流程，但 `collect_speaker_candidates(video, speaker)` 作用域到该 speaker：抽候选 → OmniVoice 单次加载生成所有目标样句 takes → 释放 → ECAPA 选最佳 take → OmniVoice 再单次加载生成二次克隆翻译试听 → 释放 → ECAPA 用「翻译试听 vs 原音」打分排序。全部复用 `single_clone` 现有实现。
- 「采纳选中为基准」→ 调 `save_speaker_basis(video, speaker, basis_wav=候选的 target_sample_audio, basis_text=候选的 target_sample_text, source_kind="candidate_target_sample", meta={...})`，关闭对话框，回填该行状态。
  - 注意：基准音存的是**固定样句的目标语言样音**（`target_sample_audio`），不是翻译试听、也不是原音片段。翻译试听只用于人耳/ECAPA 验证。与单人 tab 的 `_single_clone_basis_from_candidate`（[gui.py:1225](../tool_clonevoice/gui.py#L1225)）一致。

### 6.3 「导入 WAV」「设计音色」小对话框（Phase 2；音色设计可视工期提前）

- 导入：选 WAV + 一个多行文本框（填该 WAV 的准确目标语言文本）+ `soundfile.info()` 校验 → `save_speaker_basis(source_kind="user_import")`。
- 设计音色：instruct 结构化控件（性别/年龄/音高/风格 + 自由文本，参考单人 `_single_clone_voice_design`）→ `generate_voice_design_basis_with_model` 产出样音 → `save_speaker_basis(source_kind="voice_design", meta={"instruct": ...})`。

---

## 7. 状态与门控

### 7.1 内部状态

```python
self.multi_clone_state = {
    "video": "",
    "target_language": "Chinese",
    "manifest_ready": False,
    "speakers": [],          # [{id, total_dur, seg_count}]
    "basis": {},             # speaker_id -> {wav, text, source_kind, meta} (已确认)
    "skipped": set(),        # 勾选跳过的 speaker
    "step": 0,
}
```

每行状态直接读 `manifest["speakers"][spk].get("ref_audio")` 也可，但用内存镜像刷新 UI 更快。

### 7.2 按钮可用性

| 步骤 | 上一步 | 下一步 | 本步执行 |
|---|---|---|---|
| ① 转录 | 禁用 | 转录完成后启用 | 先预检翻译配置(缺则拦截) → 开始转录（多说话人） |
| ② 选择基准 | 启用 | **每个 speaker「已确认基准」或「已勾选跳过」**后启用 | 每行入口 + 候选对话框；`跳过` 复选 |
| ③ 翻译并克隆 | 启用 | 禁用 | 开始翻译并克隆 |

### 7.3 线程与模型生命周期

- **完全复用** `_single_clone_run_async` 的模式：重活进后台线程，OmniVoice 交 `model_holder` 由主线程释放，避免后台线程析构重模型崩溃。
- VRAM 纪律沿用现状：OmniVoice 与 ECAPA 不同时驻留（先生成 takes 释放，再加载 ECAPA）。候选对话框每次「抽取并生成」是「per speaker」的一轮，模型加载次数 = 说话人数 × 2（OmniVoice+ECAPA），可接受；不要退化成 per-candidate 加载。
- 停止：沿用 `stop_event`，每个 worker 循环内检查。

---

## 8. i18n

新增键（中文必填，英/日先给直译，避免切换语言缺 key 崩溃）：

- Tab / 备注：`tab_multi_clone`、`multi_clone_note`
- 步骤名：`multi_step_1_transcribe`、`multi_step_2_select_basis`、`multi_step_3_translate_clone`
- STEP①：复用单人已有的源语言/模型/降噪/目标语言键 + 输入模式键 `opt_single_file`/`opt_batch_dir`（已有）；分离相关 `lbl_diar_backend`、`opt_diar_auto/ecapa/pyannote`、`lbl_num_speakers`（一键克隆已有，可复用）。批量新增 `lbl_global_num_speakers`（「整批总人数」）、`msg_prescan_global_diarize`（全局分离进度日志）、`warn_batch_total_duration`（总时长过长告警）。
- STEP②：`btn_detect_speakers`、`col_speaker`、`lbl_speaker_stats`、`btn_select_basis`、`btn_import_wav`(已有)、`btn_voice_design`(已有)、`status_basis_none`、`status_basis_set`、`chk_skip_speaker`、`dlg_select_basis_title`（含 `{speaker}` 占位）、`btn_adopt_basis`、`err_not_all_speakers_ready`
- STEP③：复用单人 STEP④ 的键（`lbl_loudness_mode` 等）。

i18n 文件 `i18n/{zh,en,ja}.json` 用 `utf-8-sig` 解析；提交前跑一次解析校验（项目惯例）。

---

## 9. 文件落点约定

单文件 `<video>` 处理后：

```
<video>.clone/
  manifest.json                 # speakers 逐个填 ref_audio=<SPEAKER>.basis.wav + skip_work_ref
  audio16k.wav
  source.srt / translated.srt
  candidates_SPEAKER_00/        # 该 speaker 的候选原音 + 目标样句 + 翻译试听 + candidates.json
  candidates_SPEAKER_01/
  ...
  SPEAKER_00.basis.wav / .txt / .meta.json
  SPEAKER_01.basis.wav / .txt / .meta.json
  ...
<video>.SI.WAV                  # 最终多说话人合成
```

不在视频同级目录放用户可见副本（多人场景没有「一个共享基准」的语义）。如需导出可作为 Phase 2 的「导出基准音」按钮。

批量模式：每个视频各自的 `<video>.clone/` 结构同上；一个全局 speaker 的基准音会被 `save_speaker_basis` 写进**每个含该 speaker 的视频** clone 目录（`SPEAKER_00.basis.wav` 等），互不干扰。全局分离的临时拼接 wav 用完即删，不落地。

---

## 10. 测试计划

### 单元/轻量（`tests/test_clonevoice_multi_clone.py`，全部可 mock，不跑真实模型）

| 测试 | 目标 |
|---|---|
| `test_list_speakers_sorted_by_duration` | manifest 多 speaker 时统计与排序正确 |
| `test_collect_speaker_candidates_isolated_dirs` | 不同 speaker 候选落在各自 `candidates_<spk>/`，`candidates.json` 不冲突 |
| `test_save_speaker_basis_updates_only_target_speaker` | 只改指定 speaker 的 `speakers[spk]`，其它 speaker 与 **segments 归属不变** |
| `test_save_speaker_basis_sets_skip_work_ref` | 写入 `skip_work_ref=True` / `ref_language` / `ref_kind` |
| `test_all_speakers_have_basis_gating` | 缺一个 speaker 未选→False；忽略集生效 |
| `test_multi_transcribe_passes_backend_and_num_speakers` | 不强制 single，backend/num_speakers 透传 |
| `test_multi_translate_synthesize_reuses_single_helper` | STEP③ 走 `translate_and_synthesize`，不误用 `run_full` |
| `test_precomputed_turns_skips_internal_diarize` | `run_transcribe_diarize(precomputed_turns=...)` 时不调用 `diar.diarize`（mock 断言），且 turns 被用于 assign |
| `test_split_turns_to_video_offsets_and_clip` | 全局 turns 拆回：偏移相减、边界裁剪、丢弃 <0.2s 碎片、按 seam 切断 |
| `test_prescan_global_labels_consistent`（可 mock diar） | 拼接偏移与拆回映射正确；跨缝 turn 被切断到各自视频 |
| `test_global_speaker_union_across_videos` | STEP② 全局 speaker 取并集；某 speaker 只在部分视频出现时归集/落盘正确 |

### 手工验证

1. 一段有 2-3 人的视频 → 转录，确认 STEP② 列出对应说话人及时长。
2. 分别用「视频候选 / 导入 WAV / 音色设计」为每个说话人设基准，试听翻译预览。
3. STEP③ 合成，确认 `.SI.WAV` 里不同段落是对应说话人的音色。
4. 边界：某 speaker 只有极短/无文本段落 → 候选为空时对话框提示，改用导入/设计。
5. 边界：勾选跳过某 speaker → 该 speaker 段落在 `.SI.WAV` 中静音/无合成，其余正常。

---

## 11. 风险与待决问题

1. **批量模式**：Phase 1 单文件，Phase 4 用全局分离支持（第 5.5 节）。全局分离的内存/耗时、拼接缝、跨条件漂移见 5.5.5。
2. **过/欠分离**：diarization 可能多切或少切说话人。首版靠 `num_speakers` 手动指定 + 「重新识别」按钮缓解；说话人合并编辑列为后期（批量场景尤其需要，见 5.5.5 第 4 点）。
3. **短/纯杂音说话人**：候选可能为空或全是无文本片段（`refsel._score` 会压低无文本候选）。对话框需处理空候选：提示并引导导入/设计，或允许「跳过」。
4. **翻译前置耦合**：STEP① 依赖翻译 API 配置（候选 `tgt_text` 需要）。建议翻译失败非致命，降级到固定样句相似度（见 5.1）。
5. **候选逻辑复用 vs 复制（6.1）**：优先抽 `CandidateBasisPanel` 组件；若复制，务必登记技术债，避免单人/多人两套候选逻辑长期分叉。
6. **模型加载次数**：per speaker 各一轮 OmniVoice+ECAPA。说话人多时较慢但可接受；不要退化成 per-candidate 加载（单人 tab 已踩过这个坑并修复，参考其批处理写法）。
7. **跳过说话人的下游**：被跳过 speaker 的原声如何保留，取决于「混合视频音轨」tab 的 ducking 逻辑，需一并验证（原声不被完全压掉）。

---

## 12. 分阶段实施

**Phase 1（可用骨架，单文件）— 已确认交付集**
- 新 tab + 3 步向导（单文件）。
- STEP①：多说话人转录（分离后端 + num_speakers 控件）；**开工前预检翻译配置**（缺则拦截，不启动转录）；转录后 `ensure_translated_for_videos`。
- STEP②：列出说话人（含时长/段数统计）；每行 `[选择基准语音]` 对话框（复用/抽出候选组件）跑通**视频候选来源**；**`跳过该说话人` 复选**；门控 = 每个 speaker 已选基准或已跳过。
- STEP③：合成 `.SI.WAV`（直接调 `translate_and_synthesize`）。
- 后端：`multi_clone.py`（run_multi_transcribe / list_speakers / collect_speaker_candidates / save_speaker_basis / all_speakers_have_basis(ignore=skipped)）+ 复用 `single_clone` 候选与合成函数。
- 测试：本文件第 10 节「单元/轻量」中不含 `precomputed_turns` / 全局分离的那几条（那几条属 Phase 4）。

**Phase 2**
- 每行「导入 WAV」「设计音色」两个来源（音色设计成本低，可视工期并入 Phase 1）。
- 空候选/短说话人体验打磨；`CandidateBasisPanel` 组件化（第 6.1 节）。

**Phase 3（可选）**
- 抽 `CandidateBasisPanel` 并让单人 tab 也切过来，消除重复。
- 导出/复用基准音。

**Phase 4（批量：全局分离）**
- `logic.run_transcribe_diarize` 加 `precomputed_turns` 钩子（5.5.3，唯一改动 `logic.py`，向后兼容）。
- 新增 `multi_clone.prescan_global_diarize`（5.5.2）+ `_split_turns_to_video`（5.5.2）+ 单元测试。
- STEP① 增加「批量目录」输入模式与「整批总人数」文案；批量流程编排（5.5.4）。
- STEP② 全局 speaker 取并集，候选跨视频归集，`save_speaker_basis` 写入每个含该 speaker 的视频。
- 总时长上限告警；跨缝静音；`candidates_<speaker>/` 隔离。
- 后期兜底：手动「合并两个全局说话人」。

---

## 13. 给研发的落地顺序建议（Phase 1）

1. 先写 `tool_clonevoice/multi_clone.py` + 单元测试（纯逻辑，不碰 UI，可先绿）：`run_multi_transcribe` / `list_speakers` / `collect_speaker_candidates` / `save_speaker_basis` / `all_speakers_have_basis(ignore=skipped)`。
2. 再抽/复制候选面板组件，单独用一个 speaker 跑通「抽取→生成→试听→ECAPA→采纳」。
3. 拼 3 步向导 UI 与门控：STEP① 加分离后端/num_speakers 控件 + **翻译配置开工前预检**；STEP② 每行 `[选择基准语音]` + **`跳过该说话人`** 复选，门控 = 每 speaker 已选或已跳过；STEP③ 直接调 `translate_and_synthesize`。
4. i18n 三语种补齐，`utf-8-sig` 解析校验，`py_compile` + `pytest tests/test_clonevoice_multi_clone.py`。

> Phase 1 不含 `precomputed_turns`/全局分离（Phase 4）与 导入WAV/音色设计（Phase 2）。第 10 节测试表里带 `precomputed_turns`/全局分离的 4 条属 Phase 4，Phase 1 不必写。

核心心法：**合成端一行不用改**（多说话人本就是原生路径），把精力全部放在「逐说话人把 basis 正确写进 manifest」和「候选选择 UI 复用」上。
