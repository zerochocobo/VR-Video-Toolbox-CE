# tool_clonevoice 校对翻译并导出克隆语音文件 —— 开发计划

日期：2026-07-07
范围：`tool_clonevoice`（单人语音克隆、多人语音克隆两个 Tab 的最后一个 STEP）

## 1. 需求

1. 单人克隆 STEP④、多人克隆 STEP③ 改名为「校对翻译并导出克隆语音文件」。
2. 在该 STEP 界面中增加"每个视频的最终翻译"编辑入口：视频文件名 + 「校对翻译」按钮，单人/多人统一 UI，点击后在新对话框中修改。
3. 校对对话框分两种情况：
   - 只有 `.clone/source.srt` + `.clone/translated.srt`：三列（时间、原文、AI翻译），仅 AI翻译列可改；点「保存」才落盘，落盘前把原版备份为 `translated_org.srt`。
   - 视频同目录存在同名 `视频文件名.srt`（用户外部字幕）：顶部提示"检测到参考字幕 xxx.srt，可直接应用"，显示五列（参考时间、参考翻译、时间、原文、AI翻译），需处理 whisper 时间轴与参考时间轴互有缺失的匹配问题。

## 2. 现状与关键事实（代码调研结论）

这些事实决定了实现方式，先记下来避免踩坑：

### 2.1 数据源的真相：manifest.json 才是主数据，srt 只是导出副本

- 转录后 [logic.py:271-274](../tool_clonevoice/logic.py) 写 `manifest.json`（`segments[].src_text/tgt_text/start/end/speaker`），并从 manifest 导出 `source.srt`。
- 翻译后 [logic.py:381-391](../tool_clonevoice/logic.py) 把译文写回 `segments[].tgt_text`，`save_manifest` 后再用 `write_srt` 导出 `translated.srt`。
- **合成阶段 `run_synthesize`（logic.py:397）读的是 manifest 的 `tgt_text`，根本不读 translated.srt**。

→ 所以「校对」的本质是**编辑 manifest 的 `tgt_text`**，保存时同步重写 `translated.srt`（并按需求备份 `translated_org.srt`）。只改 srt 文件不会影响克隆结果。

### 2.2 翻译发生的时机：STEP① 转录后立刻强制 API 翻译

- 单人 STEP①（`_single_clone_run_transcribe`，gui.py:1591→1628）和多人 STEP①（gui.py:2553）在转录循环结束后都会立刻调 `sc.ensure_translated_for_videos` 强制 API 翻译，按钮名就是"开始转录并翻译"。**用户走到最后一个 STEP 时，manifest 的 `tgt_text` 已全部填好**（实测 `masex.tv@bibivr00174_4_8k.clone` 目录：225/225 段有译文，translated.srt 在转录完成后即生成）。
- 导出入口 `translate_and_synthesize`（single_clone.py:716）内部也调 `ensure_translated`（single_clone.py:557），但那只是安全网：**manifest 已有完整 `tgt_text` 且 `target_language` 一致时直接跳过**。

→ 两个推论：
1. 校对入口的主路径就是"直接打开对话框"，无需先跑翻译。"未翻译"只会出现在异常残局（STEP① 翻译中途被停止/报错、或用户手动删了 manifest），做成兜底分支即可：提示后调 `logic.run_translate` 补翻，不算主流程。
2. `ensure_translated` 的跳过逻辑天然保证：**校对后的译文在导出时不会被 API 重翻覆盖**。这是本功能能成立的关键，无需改动。

### 2.3 其它相关事实

- srt 内容带说话人前缀：`write_srt(..., speaker_prefix=True)` 输出 `[SPEAKER_00] 文本`。校对框直接编辑 manifest 就绕开了前缀解析问题；多人模式在表格中单独加"说话人"列展示即可。
- 单人模式所有段的 speaker 固定为 `SPEAKER_00`（single_clone.py:27），说话人列可隐藏。
- `.si.wav` 已存在且勾选「跳过已存在」时，导出会整段跳过（single_clone.py:739）。**如果用户先导出、后校对、再导出，必须删除 .si.wav 或取消勾选**，需要在保存成功后提示。
- 视频清单：单人 `self.single_clone_videos` / `_single_clone_scan_videos_silent()`（gui.py:1372），多人 `_multi_clone_current_videos()`。两个 Tab 的最后 STEP 都能拿到视频列表。
- 现成 SRT 解析：`tool_subtitle.logic._parse_srt_blocks`（logic.py:2370）可参考，但它返回 ASS 时间字符串，本功能需要 float 秒，自己写一个约 30 行的解析器更干净（utf-8-sig + errors="replace" 容错要保留）。
- 弹窗风格参考：`_open_multi_import_basis_dialog`（gui.py:2817），`Toplevel + transient + grab_set + Escape 关闭`。
- 段落试听：已有 `_single_clone_play_wav`（gui.py:1512，winsound），`.clone/audio16k.wav` 常驻，可按段裁剪临时 wav 试听（P3 加分项）。
- 重新转录会重建 manifest（校对成果被清掉），这是流程上游操作，符合预期，但状态列要能反映"未翻译"让用户看得出来。

## 3. 总体设计

### 3.1 新增文件（避免 gui.py 继续膨胀，现已 4100+ 行）

| 文件 | 职责 |
|---|---|
| `tool_clonevoice/proofread.py` | 纯逻辑，无 tkinter：SRT 解析（秒）、参考对齐算法、读 manifest 生成行数据、保存（manifest + 备份 + 重写 translated.srt）、单视频状态查询 |
| `tool_clonevoice/gui_proofread.py` | UI：STEP 内嵌的视频列表面板 builder + 校对对话框类，单人/多人共用 |

gui.py 只做接线：两个 STEP 各挂一次面板、传入取视频列表的回调和异步执行器（复用 `_single_clone_run_async` / `_multi_clone_run_async`）。

### 3.2 STEP 页内的入口面板（单人 step4 / 多人 step3 共用）

在现有「合成选项行」和「开始生成」按钮之间插入一个 `LabelFrame("校对翻译")`：

```
┌ 校对翻译 ────────────────────────────────────────────┐
│ Treeview：                                            │
│  视频文件名          | 翻译状态      | 参考字幕        │
│  movie_A.mp4        | 已翻译 132 段 | 有 (movie_A.srt) │
│  movie_B.mp4        | 未翻译        | 无              │
│                                        [校对翻译] 按钮 │
└──────────────────────────────────────────────────────┘
```

- 翻译状态取自 manifest：`未转录` / `未翻译` / `已翻译 N 段` / `已校对(改 N 句)`（校对保存时在 manifest 里记 `proofread` 元数据）。
- 「校对翻译」按钮（或双击行）：
  - 已翻译（主路径，STEP① 已强制翻译）→ 直接弹校对对话框。
  - 未转录 → 报错提示先执行 STEP①。
  - 未翻译（兜底：STEP① 翻译中途停止/失败的残局）→ 弹确认"该视频译文不完整，现在调用 API 补翻？"，确认后走异步执行器跑 `logic.run_translate`，完成后自动弹校对对话框。
- 进入该 STEP（`_show_single_clone_step(3)` / `_show_multi_clone_step(2)`）时刷新列表；busy 时按钮进 `*_action_buttons` 列表统一禁用（沿用现有机制）。

### 3.3 校对对话框（核心 UI）

**控件选型结论**：`ttk.Treeview`（表格主体） + **底部固定编辑区**（选中行联动），而不是"双击单元格就地编辑"。理由：

- Treeview 是虚拟渲染，几百上千段不卡；本项目全部表格已用它，风格统一。
- 就地 Entry 覆盖编辑对中文/日文 IME 不友好（候选窗定位、失焦提交时机都易出 bug），且单元格宽度放不下长句。
- 底部用 `tk.Text` 编辑区：多行长句可见、IME 正常、可以同时把"原文 + 参考翻译"完整展示在旁边供对照——校对场景本来就要来回看三份文本，单元格里根本放不下。

布局（`Toplevel`，约 1100×700，可缩放）：

```
┌ 校对翻译 — movie_A.mp4 ────────────────────────────────────────┐
│ [提示行] 检测到参考字幕 movie_A.srt，可直接应用          （仅五列模式）│
│ ┌─Treeview（五列模式）────────────────────────────────────────┐ │
│ │ # | 参考时间 | 参考翻译 | 时间 | (说话人) | 原文 | AI翻译     │ │
│ │ …行，已修改的行高亮 tag（背景淡黄）；纯参考行灰色斜体…          │ │
│ └────────────────────────────────────────────────────────────┘ │
│ ┌─编辑区（选中行联动）────────────────────────────────────────┐ │
│ │ 原文(只读)：…………                                            │ │
│ │ 参考(只读)：…………                        [应用参考到本行]      │ │
│ │ AI翻译(可编辑 Text)：…………                [还原本行]           │ │
│ └────────────────────────────────────────────────────────────┘ │
│ [全部应用参考]                # 已修改 12 句    [保存] [取消]     │
└────────────────────────────────────────────────────────────────┘
```

交互细节：

- 选中行 → 编辑区载入；编辑区失焦 / 按 `Ctrl+Enter` / 切换选中行时把 Text 内容写回内存行数据并刷新表格该行 + 高亮。
- `↑/↓` 在表格中移动即可逐句校对；编辑区内 `Ctrl+↓` 提交并跳下一句（高频操作）。
- 三列模式：隐藏参考两列与「应用参考」按钮（同一个类，`ref_cues=None` 分支），单人模式再隐藏说话人列。
- 「还原本行」：恢复打开对话框时的初始 AI 译文。
- 关闭/取消时若有未保存修改 → `askyesno` 确认。
- 「保存」语义见 3.4；保存成功后若检测到该视频 `.si.wav` 已存在，提示"已存在旧的 .SI.WAV，重新导出前请删除或取消勾选「跳过已存在」"。

### 3.4 保存语义（proofread.py）

```
save(video, rows):
  1. manifest = load_manifest(video)
  2. 按 segment id 回写 rows 里被修改的 tgt_text（strip；允许清空=该句不配音，与现状 write_srt/合成跳过空文本行为一致）
  3. manifest["proofread"] = {"edited_ids": [...], "time": iso8601}   # 供状态列显示
  4. save_manifest(video, manifest)
  5. translated.srt 存在 且 translated_org.srt 不存在 → copyfile 备份   # 只备份"最初的 AI 版本"，多次校对不覆盖首版备份
  6. write_srt(clone/translated.srt, segments, "tgt_text", speaker_prefix=True)
```

备份策略说明：需求是"写入之前把原本的版本保存为 translated_org.srt"。若每次保存都覆盖备份，第二次校对时首版 AI 译文就丢了，故采用**首次校对时备份一次、之后不再覆盖**；`translated_org.srt` 始终等于纯 AI 原版。

## 4. 参考字幕对齐算法（五列模式核心）

输入：manifest 段 `S = [(start, end, src, tgt)...]`（whisper 时间轴）；参考 cue `C = [(start, end, text)...]`（外部 srt，多行文本合并为一行，空格连接）。

**两遍分配法**（避免同一句参考被重复应用到多个段）：

1. **cue → 段 分配**：每个 cue 找与其重叠时长最大的段；若最大重叠 `< max(0.2s, 30% × min(cue时长, 段时长))` 则视为无归属。一个段可以收多个 cue（whisper 合并了参考拆开的句子）。
2. **段的参考文本** = 分配给它的所有 cue 按开始时间排序后用空格 join；参考时间列显示 `首cue起点 → 末cue终点`。
3. **whisper 有、参考没有**：该行参考两列留空——正常现象（whisper 幻听、参考漏字幕），用户自己校对 AI 列。
4. **参考有、whisper 没有**（无归属 cue）：按时间顺序插入**纯参考行**（灰色斜体，时间/原文/AI 列为空，不可编辑、不参与保存）。作用是让用户看见"参考里有这句但 whisper 没识别到"，可手动把文本并入相邻行。
   - *不做*"插入为新配音段"：新段缺少时长对齐和 emotion_ref，会牵动合成端，性价比低（记入 P3 观察项）。

「全部应用参考」= 对所有"参考文本非空"的段执行 `tgt_text ← 参考文本`（计为修改、高亮）；「应用参考到本行」同理单行。应用后仍可继续手改。

对齐纯函数签名（可单测）：

```python
def align_reference(segments: list[dict], cues: list[dict]) -> list[Row]
# Row = {kind: "seg"|"ref_only", seg_id, start, end, speaker,
#        src_text, tgt_text, ref_text, ref_start, ref_end}
```

## 5. 具体改动清单

### 5.1 i18n（zh.json / en.json / ja.json 三份同步）

- 改：`step_4_translate_clone` → `④ 校对翻译并导出克隆语音文件`；`multi_step_3_export` → `③ 校对翻译并导出克隆语音文件`（en/ja 对应翻译）。
- 新增（`clonevoice` 节）：`lbl_proofread_panel`、`col_pf_video`、`col_pf_trans_status`、`col_pf_ref_srt`、`btn_proofread`、`pf_status_no_manifest`、`pf_status_untranslated`、`pf_status_translated`、`pf_status_proofread`、`dlg_proofread_title`、`msg_ref_srt_hint`、`col_pf_ref_time`、`col_pf_ref_text`、`col_pf_time`、`col_pf_speaker`、`col_pf_src`、`col_pf_tgt`、`btn_apply_ref_row`、`btn_apply_ref_all`、`btn_revert_row`、`btn_pf_save`、`msg_pf_modified_count`、`confirm_pf_unsaved`、`confirm_pf_translate_first`、`msg_pf_saved`、`msg_pf_si_exists_warn`、`err_pf_need_transcribe`。
- `tests/test_i18n.py` 已有三语言键一致性检查，新增键必须三份齐全。

### 5.2 代码

| 文件 | 改动 |
|---|---|
| `tool_clonevoice/proofread.py`（新） | `parse_srt_seconds(path)`、`find_reference_srt(video)`（`video.parent / (stem + ".srt")`，注意排除 `.si.srt` 等派生名——精确匹配 stem 即可）、`align_reference(...)`、`load_rows(video)`、`save_rows(video, rows)`、`video_status(video)` |
| `tool_clonevoice/gui_proofread.py`（新） | `build_proofread_panel(parent, app, get_videos, run_async, log_widget, show_speaker)` 返回 frame + refresh 函数；`ProofreadDialog(root, video, on_saved)` |
| `tool_clonevoice/gui.py` | step4/step3 布局插入面板（step4 现有行号 631-682、step3 现有 1008-1059 区段，「开始生成」按钮下移一行）；`_show_*_clone_step` 进入末步时调 refresh；「校对翻译」按钮加入 `*_clone_action_buttons`；"未翻译先翻译"的 worker 接线 |
| `tests/test_clonevoice_proofread.py`（新） | 见 5.3 |

### 5.3 测试（纯逻辑，不测 tkinter）

1. `parse_srt_seconds`：常规、utf-8-sig BOM、多行文本块、坏块跳过。
2. `align_reference`：
   - 一对一重叠；
   - 参考一句跨 whisper 两段（cue 归属重叠最大的段）；
   - whisper 一段对应参考两句（join）；
   - 无归属 cue → ref_only 行且按时间排序插入；
   - 重叠低于阈值不匹配。
3. `save_rows`：tgt_text 回写 manifest；首次保存生成 `translated_org.srt` 且内容为旧版；二次保存不覆盖备份；translated.srt 重写后带说话人前缀；清空文本的行不出现在 srt。
4. `video_status`：无 manifest / 未翻译 / 已翻译 / 已校对 四态。

## 6. 分阶段实施

| 阶段 | 内容 | 验收 |
|---|---|---|
| **P1 基础校对** | STEP 改名；入口面板（视频列表+状态）；三列对话框（含说话人列逻辑）；保存+备份；未翻译残局的补翻兜底；.si.wav 已存在提示 | 单人/多人各跑通：STEP①转录翻译→末步校对改几句→导出，克隆结果使用校对后文本；translated_org.srt 正确 |
| **P2 参考字幕** | `find_reference_srt`；对齐算法+单测；五列模式；单行/全部应用参考；纯参考行展示 | 用外部 srt 实测：时间轴错位、多对一、一对多、参考多余句均正确显示与应用 |
| **P3 体验加分（可选）** | 逐句试听（audio16k.wav 按段裁剪+winsound）；表格内搜索/过滤（只看已修改/只看有参考差异）；`Ctrl+↓` 提交跳下句已在 P1，其余快捷键打磨 | 按需 |

预估：P1 约 1 天，P2 约 1 天（对齐算法半天+联调半天），P3 视需求。

## 7. 风险与边界

- **重新转录清空校对**：STEP① 重跑会重建 manifest，校对丢失属预期，但状态列会退回"未翻译"，用户可感知；不做额外保护。
- **target_language 变更**：校对后若用户改了目标语言再导出，`manifest_has_target_translation` 判定不一致会触发 API 重翻，覆盖校对。P1 在对话框标题栏展示 manifest 的 target_language，暂不做强校验（与现有导出行为一致）。
- **外部 srt 是双语/带样式标签**：P2 仅做 `<i>` 等简单 html 标签剥离 + 多行合并，不做双语拆分（用户可在应用后手改）。
- **参考 srt 编码**：utf-8-sig + errors="replace" 兜底；GBK 等本地编码的 srt 解析出乱码时用户可见（表格里直接暴露），P3 再考虑 chardet。
- **性能**：千段级 Treeview 无压力；对齐 O(n·m) 最坏 1e6 级比较，可先按 start 排序双指针降到 O(n+m)，P2 直接按双指针实现。
