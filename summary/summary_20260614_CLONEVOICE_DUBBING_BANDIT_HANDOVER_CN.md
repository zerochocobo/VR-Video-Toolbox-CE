# tool_clonevoice 配音模式（bandit-v2 人声分离）开发交接 HANDOVER

日期：2026-06-14
分支：omnivoice

## 1. 做了什么

把 tool_clonevoice「混合音轨」tab 里此前占位「配音模式(开发中)」实现为可用功能：

**配音 = 剥离原声对白 + 保留 music+sfx 背景 + 叠加克隆/翻译后的 `.si.wav` → `<name>_DUB.mp4`**

与「同声翻译模式」（原声完整保留 + 叠加 SI 轨）不同：配音模式**丢弃原始音频**，只保留分离出的背景床，把对白替换成克隆声。

> 适用范围：**通用功能**——任意语言、任意类型视频，不针对成人/日语（测试素材恰好是日语 VR，勿当成功能目标）。

## 2. 技术选型与验证（已 spike 确认）

- 模型：**bandit-v2**（github.com/kwatcharasupat/bandit-v2，Apache-2.0）。checkpoint = `checkpoint-multi.ckpt`（多语种）。
  - ⚠️ README 有道德条款：商业获益者「被强烈建议」自愿捐助音乐类非营利组织（非法律强制，Apache-2.0 仍可商用）。
- spike 决定性结论：
  - 三 stem = **speech / music / sfx**（配音：丢 speech，留 music+sfx）。
  - **原生 mono 48kHz**，正好对齐项目，无需重采样模型输入。
  - 裸 `Bandit(stems=[speech,music,sfx], **mus64_kwargs)` + 灌 `model.*` state_dict → **missing=0 / unexpected=0 完美匹配**，**不需要 hydra / pytorch-lightning system 那套**。
  - GPU 稳态 RTF ≈ 0.13（60s 音频含预热 17s）；**27 分钟视频约 7-8 分钟**。
  - speech+music+sfx 重建误差仅 2.2%（complex mask 基本和为单位，分离无能量丢失）。

## 3. 改动清单（全部在 tool_clonevoice，未动 tool_si）

| 文件 | 说明 |
|---|---|
| `tool_clonevoice/bandit/` | **新增**。vendor 的 bandit-v2 推理代码 |
| `tool_clonevoice/bandit/{bandit,bandsplit,maskestim,tfmodel,utils}.py` | 从仓库 `src/models/bandit/` 拷贝。patch：`bandit.py` 的 `from ..base` → `from .base` |
| `tool_clonevoice/bandit/inference_handler.py` | 从仓库 `src/system/` 拷贝。patch：删 `_fold` 里的调试 `print(stem_output.shape)` |
| `tool_clonevoice/bandit/base.py` | **自写**。`BaseEndToEndModule(nn.Module)`——上游是 `pl.LightningModule`，推理用不到，去掉 lightning 依赖（PyInstaller 更轻） |
| `tool_clonevoice/separate.py` | **新增**。`BanditSeparator` 常驻分离器 + ckpt 预处理 + `is_available()` |
| `tool_clonevoice/dubbing.py` | **新增**。`dub_video` / `batch_dub_videos` + `default_dub_output_path` |
| `tool_clonevoice/gui.py` | **改**。`_dubbing_available`、`_run_single_mix` dub 分支、`_run_dub_task`、`_on_single_mix_mode_change` |

vendor 只依赖 torch / torchaudio / numpy / librosa / tqdm（全已装）。`film.py` 无人引用，未拷。

## 4. 关键设计

- **ckpt 预处理（首次使用）**：`checkpoint-multi.ckpt`（446MB，含 optimizer_states）→ 抽 `model.*` 存 `checkpoint-multi.slim.pt`（**142MB**）缓存于同目录。之后只加载 slim。逻辑在 `separate._ensure_slim_weights`。
- **分离器常驻显存**：配音不跑 OmniVoice，按用户决定整批常驻。`batch_dub_videos` 一次 `BanditSeparator` 用到底（`owns_separator`，结束 `close()`）。单文件模式建一个用完即 `close()`。
- **配音固定参数**：左右声道（both）/ SI 音量 100% / SI 延迟 0s。克隆轨已按源时间线对齐，故无延迟/声道概念。UI 选配音时隐藏全部 SI 选项，并把（隐藏的）声道/SI音量/SI延迟设为 both/100%/0s，切回同传时恢复用户原值（`_si_mode_saved` 快照）。
- **中间文件**：`<video>.dub_mix48k.wav`（抽的原声）和 `<video>.dub_audio.wav`（混好的）用完即删；保留 `<video>.dub_bg.wav`（分离出的背景床，贵，`skip_existing_background=True` 时复用、可调试）。
- **混音**：torch 里 `bg*bgv + voice(重采样48k)*vv`，峰值限幅 0.97 防削波；ffmpeg mux 视频直 copy + dub 音频为唯一主轨（aac 192k 48k stereo）。
- **默认输出**：`<stem>_DUB.mp4`。

## 5. 验证状态

- ✅ py_compile / import / `is_available` 全过。
- ✅ spike：权重完美匹配、RTF、重建误差。
- ✅ 端到端 dub 冒烟：60s clip（test_2person 抽段）→ `_dub_smoke/clip_DUB.mp4`（hevc copy + aac，60s 完整，中间文件清理正确，slim 缓存 142MB 生成）。
- ⏳ **质量待用户耳朵**：`_dub_smoke/clip_DUB.mp4` 待听。注意 test_2person 这素材 pyannote 本就分不开两人、克隆音色有问题（见 clonevoice-tool-plan 记录），但**分离/混音/封装管线本身 OK**；建议拿能正常克隆的通用素材在 GUI 实跑验证整体效果。

## 6. 待办 / 已知局限

1. **分离不可中途 Stop**：`separate_background` 单次调用（27min 约 7-8 分钟）不可打断；Stop 只在分离完成后/ffmpeg 阶段生效。若要中断粒度需子类化 handler 在 batch 循环里查 stop_event。
2. **bg/voice 音量未开放 UI**：配音硬编码 100/100。如需调（如背景压低）再加控件。
3. **发布打包**：`models/bandit-v2/` 纳入 ZIP——保留 `checkpoint-multi.slim.pt`（142MB）即可，`checkpoint-multi.ckpt`（446MB）首次预处理后可删（或不打包 full、只打 slim；但首次预处理需要 full，建议打包时直接放 slim 跳过预处理）。PyInstaller 需确保 `tool_clonevoice/bandit/*` 与 librosa 数据被收集。
4. **清理**：spike 临时物 `_bandit_v2_src`（克隆源）、`_bandit_spike_out` 已删（vendor 后不需要）。`_dub_smoke/clip_DUB.mp4` 留给用户听，听完可删。

## 7. 复现/调试入口

```python
from tool_clonevoice import dubbing as dub
sep = dub.BanditSeparator('models', log=print)
dub.dub_video('视频.mp4', '视频.si.wav', '视频_DUB.mp4', separator=sep, log_callback=print)
sep.close()
```

GUI：混合音轨 tab → 模式选「配音模式」→ 选视频（需同名 `.si.wav` 已生成）→ 开始。

---

# 补充（2026-06-14 后续迭代）

初版 HANDOVER 之后又做了 4 组改动，均已 commit（分支 omnivoice）。

## A. 合成响度三模式（commit 246c135）
克隆合成的输出响度从「均匀」扩展为可选三模式（克隆 tab「输出语音」下拉）：
- **平铺直叙**(flat)：原 `_normalize_peak` 均匀响度，不跟随原片。
- **整句匹配**(sentence)：每句合成音 RMS 匹配对应原句 RMS（`_match_sentence_loudness`，增益 clamp[0.35,2.8]+峰值0.88）。句与句之间跟随原片大小。
- **语调起伏**(envelope，默认)：整句匹配 + 句内短窗能量包络（`_follow_energy_envelope`）。位置归一化粗轮廓迁移（非DTW/词对齐，跨语言不可能词级对齐），±6dB限幅、percentile(40)floor防静音打洞、α混合（明显0.6默认/普通0.3，仅此模式显示「起伏强度」下拉）。
- 任意时刻实际音量 ≈ 该句原始绝对音量 × 句内相对轮廓。**参考是原始混音 audio16k.wav（含背景/串音），非纯人声**。
- 链路：gui→logic.run_full/run_synthesize→omnivoice_backend.synthesize，参数 `loudness_mode`+`envelope_alpha`。测试 tests/test_clonevoice_loudness.py（6 passed）。

## B. 配音分离显存控制（commit bb9fab2 → 4cbc55d）
- **问题**：`separate_background` 整条 wav 一次性送 GPU，chunked handler 把全部 unfold 输入+三 stem 输出全程驻显存 → 长视频 OOM。
- **分块**（bb9fab2）：`_separate_blocked` 重叠时间分块（30s块/4s重叠）+ 梯形交叉淡化（相邻块斜坡互补和=1，无缝），每块用完 empty_cache。峰值不随时长。
- **自动 batch**（4cbc55d）：按实测线性 VRAM 模型（≈0.6+0.85×batch GB）自动选 batch，预算=min(10G上限, 可用显存−2G预留)，clamp[1,16]。16G卡→batch11→峰值~9.6G。可用 `inference_batch_size`/`vram_cap_gb`/`vram_reserve_gb` 覆盖。

## C. 配音分离块间可停止（commit 49a6497）
`separate_background(...,stop_event=)`→`_separate_blocked` 每块开头查 stop_event，置位即抛 `Stopped by user.`；dub_video 已传入。Stop 最长延迟一个 30s 块。**解掉了初版「分离不可中途 Stop」的局限**。

## D. 模型进度条重定向到 GUI + 配音总进度（commit d3bdef0）
- **问题**：PyInstaller 无控制台打包后，模型 tqdm（OmniVoice「Loading weights」、faster-whisper 转录条、bandit 进度、torch 警告）写 stderr → 用户看不到。
- **`tool_clonevoice/log_redirect.py`**（新）：`redirect_stdio(emit)` 上下文管理器临时替换 sys.stdout/stderr 为 `LogWriter`，把完整行转发给回调；`\r` 进度行折叠为单行「可替换更新」（is_progress 标记）。
- **gui.py**：`_make_log_emitter(widget)`→进度行删上一行再插（单行滚动）、普通行追加；`self.log` 标记 `_last_was_progress=False`。克隆 task、配音 `_run_dub_task` 都用 `with redirect_stdio(...)` 包裹。
- **配音总进度**：原来每块各显一个 `Rank 0:` 条、无总进度。改为 handler 内层条可禁用（`inference_handler` tqdm 加 `disable=getattr(self,'disable_tqdm',False)`，separator 置 `handler.disable_tqdm=True`），`_separate_blocked` 用一个 `tqdm(desc="Separating",unit="blk")` 包分块循环作总进度 → log 窗单行「Separating x%|..| i/N blk」。实测 sep_prog=4/rank=0。

## 当前总体待办
1. **质量需真机听**：配音整体效果 + 响度三模式 A/B（test_2person 素材 diarization 分不开两人，不适合判质量，用能正常克隆的通用素材）。
2. bg/voice 音量、envelope 的 α/max_db/floor_pct 暂未全开放 UI。
3. envelope 可改进：用 manifest 词时间戳屏蔽源静音（现 percentile floor 近似）。
4. 发布打包：models/bandit-v2/（slim 142MB）入 ZIP；PyInstaller 收集 tool_clonevoice/bandit/* + librosa 数据。
5. 并发边界：log_redirect 全局替换 sys.stdout/stderr，克隆与配音两 tab 同时跑会相互影响（实际单任务使用，低风险）。
