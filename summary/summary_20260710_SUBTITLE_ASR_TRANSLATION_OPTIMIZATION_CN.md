# 字幕转录与 AI 翻译优化总结（2026-07-09 ~ 07-10）

目标：解决用户报告的两个问题——(1) 转录漏段落（对照他人工具字幕试听确认真实漏句）；(2) AI 翻译像逐句独立翻译、缺乏上下文一致性。覆盖 tool_subtitle 与 tool_clonevoice 两条链路。

测试样本：`dsvr-064` 两部 8K 视频，以另一工具生成的 `.ref.srt` 为对照（注意：参考字幕并非真值，自带 30 秒切块边界幻觉与非语言标记）。

## 最终战果

- 参考字幕覆盖率：**77% → 84%**（两部片 534 行参考，缺失 118 → 约 85）
- VAD 漏检（含开头整句丢失、33-54 分钟耳语段成片漏）：**44+ → ≈0**
- 后处理误删真实台词（高潮段重复喊话被整行删除）：**31 → ≈0**
- 翻译侧：上下文/一致性指令、顺序 ID、缺失重试带上下文、AI 原文校对（修错字 + 删语气词）全部上线

---

## 一、翻译链路（07-09 上线）

1. **Prompt 强化**（`config/translate_prompt.txt` / `translate_prompt_dubbing.txt`）：新增 Context & Consistency 段——字幕按播放顺序、是连续对白、必须参考前后行；人名/称呼/术语/语气全片一致；原文是 ASR 输出，按上下文翻译"本意"而非错字字面。
2. **随机 ID → 顺序 ID**（`sequential_ids`）：保留 tag→原始序号映射，但 chunk 内按 1..n 顺序编号，恢复"连续对白"信号。
3. **缺失重试带上下文**（`_with_context`）：漏译行重发时附带前 2 行 + 后 1 行；顺带修复了重试结果覆盖已有翻译的老 bug（`results.update` 累积）。
4. **配音开关解耦**：clonevoice 管线硬编码 `dubbing_optimized=True`；普通字幕翻译不再被共享配置里的配音开关污染（这是"逐句直译感"的一大来源）。
5. **AI 原文校对**（`correct_entries` + `config/asr_correct_prompt.txt`）：翻译前可选 LLM pass，按上下文修同音词/助词/人名；配置项 `source_correction`，字幕工具两页签 + clonevoice 三页签均有独立复选框。clonevoice 校正后回写 manifest `src_text` 并重写 source.srt。
6. **校对删语气词**（07-10）：空标签协议——纯呻吟/笑声/叹息行（あ、ああー、はぁ、ふふ 等各语言通用发声）模型输出 `<id></id>` 标记删除；代码侧安全闸 `_deletable_interjection`（归一化 ≤10 字符且除语气汉字白名单外无实义汉字）防模型偷懒清空真句。clonevoice 删除段同步清空 src/tgt → source.srt、translated.srt、配音合成三处干净。

## 二、转录链路（07-09 ~ 07-10，数据驱动三轮迭代）

### 调试基础设施
字幕生成页新增"生成中间调试文件"复选框 → 每视频输出 `<stem>_debug/` 目录：
- `.raw.srt`：后处理前全部解码输出（附 lp/ns 置信度）
- `.vad.srt`：每个 ASR chunk 的时间区间（可挂播放器直接试听审计 VAD 覆盖）
- `.removed.srt`：后处理删除行 + 原因标签

三个文件把漏句三分归因：VAD 漏检 / 解码器无输出 / 后处理误删。

### 第一轮：后处理修复
- **近重复去重收窄**：只删"文本相似 + 时间区间实际重叠 >0.15s"的段对（重叠双解码来自补窗短 chunk 与降级切块）；时间不重叠的相似台词是真实重复对白，一律保留。删除旧的 16 秒窗口文本匹配、大 large-v2 激进去重、`is_recent_duplicate`。
- **重复喊话压缩替代删除**（`compress_repetition_text`）：单元 ×4+ 压缩为 ×3（"無理×30"→"無理無理無理"），支持标点分隔形式（"ごめん、ごめん、ごめん、ごめん"）；压不动的才删。高潮段台词由此保留。
- **解码器内部静默门限全放开**：三模型 scene 配置 `log_prob_threshold`/`no_speech_threshold` 全部 None（large-v2 原 0.34 最激进），低置信候选统一交给我们自己的后处理裁决（>0.90 & <-1.35 闸、幻觉表仍在）。

### 第二轮：VAD 敏感度三档
`VAD_SENSITIVITY_PRESETS`（实例级参数，缓存模型可直接切档）：

| 档位 | threshold | neg | pad | RMS 闸 |
|---|---|---|---|---|
| standard | 0.50 | 0.35 | 30ms | -50dB |
| **high（默认）** | 0.35 | 0.22 | 200ms | -58dB |
| max | 0.20 | 0.12 | 320ms | 关闭 |

实测 high 档：vad_miss 53→4，开头漏句和耳语段全部回收，新增行大多是呻吟行和参考工具也漏掉的真话，垃圾行仅 2-3 条/部。UI：字幕生成/一键听译/克隆三页签均有选项、默认高；字幕生成页隐藏了只剩单选项的语音分割模型行（下载按钮保留）。

### 第三轮：词级时间戳重锚定（保留但结论如实记录）
`reanchor_segment_times`：段级时间改用首词 start/末词 end（无词/退化跨度回退）。**实测未解决"文本滑移"类假漏**：能量曲线证实（22:59 案例）漂移发生在段级 token 时间层，faster-whisper 的词级 DTW 受段边界约束、跟着一起错。改动保留（边界精化、对配音时隙有小收益），滑移类若后续要啃，方向是"chunk 内能量谷二次切分"。

## 三、剩余缺口构成（已决定冻结转录侧）

- 非语言发声（嘻嘻/唉/完）：Whisper 架构性做不了，放弃；
- 文本滑移（每部 5-10 处）：修复成本高，挂起；
- 高潮段解码为幻觉套话（幻觉表正确拦截，但底下真话解不出）：同上；
- 参考字幕自身幻觉/时间误差若干。

实际有效差距估计 <10%。后续杠杆在配音链路端到端验证与翻译质量。

## 关键文件

- `tool_subtitle/logic.py`：翻译/校对/去重/压缩/敏感度/重锚定/调试文件全部核心逻辑
- `tool_subtitle/gui.py`、`tool_clonevoice/gui.py`：选项 UI 与接线
- `tool_clonevoice/segment_engine.py`：克隆转录引擎（继承 + 同步全部改动）
- `tool_clonevoice/logic.py`：run_translate 校对接入、vad_sensitivity 贯穿
- `config/translate_prompt*.txt`、`config/asr_correct_prompt.txt`：prompt 模板（老用户 config 目录不自动覆盖，发版需带上）
