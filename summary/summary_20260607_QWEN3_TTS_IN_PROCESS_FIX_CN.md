# Qwen3-TTS 在 transformers 5.x 下退化为训练语料的真凶定位与修复

## 背景

`tool_si`（同声翻译/字幕转音频）使用 vendored 的 Qwen3-TTS-12Hz-0.6B-CustomVoice 模型。
项目主环境已升级到 `transformers==5.9.0` + `huggingface_hub==1.17.0`，但 in-process
路径合成出的音频不像目标字幕语言，也不像可识别的中文/日文 — ASR 解出来常常是
"请订阅 / 感谢观看 / 感谢您的观看" 之类 YouTube 训练语料里的高频尾句。

上一版的临时方案（`summary_20260607_QWEN3_TTS_SUBTITLE_AUDIO_FIX_CN.md`）是：
扫描 `runtime_cache/uv_cache/archive-v0` 里残留的 `transformers-4.57.3` 和
`huggingface_hub-0.36.2` 的 dist-info，拼一个 PYTHONPATH 起子进程 worker，
用 vendored 的 `qwen_tts_legacy` 在 4.57.3 里跑 TTS，再通过 JSON-line 把音频回传主进程。

这个方案能解决症状，但：
- 依赖 uv 缓存目录的内部 schema（`archive-v0`），uv 自己升级时改过名字，
  缓存清理后 dist-info 就消失；
- 子进程 + base64 over JSON 把音频跨 GIL/进程倒一遍，性能损耗明显；
- 永远绕开了真正的 bug，下个升级的 vendor 子项目大概率会再撞一遍同样的雷。

本次目标是**不回退 huggingface_hub**（项目其他模块依赖最新版），直接在 vendor 里
patch transformers 5.x 兼容点，让 in-process 路径正确生成。

## 根因（两个独立 bug 叠加）

### Bug 1: 非持久 buffer 在 meta-device 装载后没有自动重算

transformers 5.x 在 `from_pretrained(device_map=...)` 下用 accelerate
做 meta-device 初始化。RoPE 的 `inv_freq` 是用
`register_buffer(..., persistent=False)` 在 `__init__` 里计算并注册的 — 4.x
loader 装载完后会重算这种 buffer，5.x 不会。结果：

- `Qwen3TTSTalkerRotaryEmbedding.inv_freq` shape `(64,)`，**全部为 0**；
- `original_inv_freq` 留在 `meta` 设备上；
- RoPE 频率为 0 → 所有 token 的 cos=1、sin=0 → 模型完全失去位置信息；
- text token 无法被相对定位 → 文本条件失效 → 退化成纯 LM 先验 → 训练分布最高频尾句。

涉及的三个 RoPE 模块：

| 模块 | 文件 |
|---|---|
| `Qwen3TTSTalkerRotaryEmbedding` | `core/models/modeling_qwen3_tts.py` |
| `Qwen3TTSRotaryEmbedding` | `core/models/modeling_qwen3_tts.py` |
| `Qwen3TTSTokenizerV2DecoderRotatoryEmbedding` | `core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py` |

### Bug 2: `talker.text_projection.linear_fc{1,2}.bias` 被 HF 5.x loader 静默丢弃

checkpoint 里这两个 bias 的实际值是非零的（rms ≈ 0.0055 / 0.032），但加载完之后
模型里是全零。`load_state_dict(strict=False)` 不报 missing / unexpected。
也就是说 HF 5.x 内部某条加载路径**吞掉了这两个 key，但没有任何 warning**。

`text_projection` 是把 2048 维 text embedding 投影到 1024 维 talker space 的关键模块。
bias 丢失之后，prefill 阶段确实还能把文本 embedding 当成"差不多正确"的输入扔进去
（weight 是对的），但下游的 attention/conditioning 信号严重失真。

实际表现：

- 前 1-2 步 token 仍依赖文本变化（不同 prompt 产生不同首 token）；
- 但很快熵塌缩到几个固定 token 循环；
- 在合理的 `max_new_tokens` 内**永远不输出 EOS**（codec_eos_token_id = 2150 的
  logit 持续低于 top tokens ~3-4 个单位）；
- 实测 "你好。" 在 `max_new_tokens=1000` 下生成了完整 999 个 token、无 EOS，
  音频 ~83 秒，ASR 无法识别。

仅修 Bug 1 不能解决问题（实测：codec 首 token 随文本变化，但仍永不收敛到 EOS、
音频是非语音噪声）。两个 bug 必须同时修。

## 修复

### 1. RoPE inv_freq lazy 重算（修 Bug 1）

在三个 RoPE 模块的 `forward` 入口加 `_ensure_inv_freq()` —
检测到 `inv_freq` 是 meta / 全零 / 空就用 `self.rope_init_fn(self.config, ref_tensor.device)`
重算并同步更新 `original_inv_freq`。

```python
def _ensure_inv_freq(self, ref_tensor):
    buf = self.inv_freq
    if buf is None or buf.device.type == "meta" or not torch.is_floating_point(buf) \
            or buf.numel() == 0 or not bool(torch.any(buf != 0).item()):
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, ref_tensor.device)
        inv_freq = inv_freq.to(device=ref_tensor.device)
        self.inv_freq = inv_freq
        self.original_inv_freq = inv_freq
```

特性：
- lazy（不增加构造期开销）；
- 幂等（已经正确就什么都不做）；
- 在 transformers 4.x / 5.x 都安全（4.x 的 loader 已经正确重算，零检测不命中）。

### 2. safetensors 兜底再加载（修 Bug 2）

`Qwen3TTSModel.from_pretrained` 走完 HF 的 `AutoModel.from_pretrained` 之后，
再打开本地的 `*.safetensors`，对**每个 checkpoint key 强制 `copy_` 到对应参数**。
HF loader 漏的 key 会在这一步被补齐。

```python
@staticmethod
def _merge_safetensors_into_model(model, pretrained_model_name_or_path) -> None:
    target_state = model.state_dict()
    target_keys = set(target_state.keys())
    for shard in sorted(glob.glob(os.path.join(path_str, "*.safetensors"))):
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                if key not in target_keys:
                    continue
                src = f.get_tensor(key)
                dst = target_state[key]
                if dst.device.type == "meta" or tuple(src.shape) != tuple(dst.shape):
                    continue
                with torch.no_grad():
                    dst.copy_(src.to(device=dst.device, dtype=dst.dtype))
```

这条路径同时也兜住了未来 HF 再静默吃掉其他 key 的可能性，是个稳健的防御。

### 3. 默认路径切回 in-process，移除 legacy worker 整套机制

修好之后 in-process 路径就是正确的；不再需要：

- `tool_si/qwen_tts_worker.py`（子进程入口）
- `tool_si/_vendor/qwen_tts_legacy/`（4.x 时代的 vendor 副本）
- `logic.py` 里：
  - `_find_uv_archive_with_dist` / `_legacy_worker_pythonpath` /
    `_legacy_worker_enabled`（uv 缓存目录 schema 扫描）
  - `QwenTTSWorkerModel`（JSON-line IPC + base64 音频传输）
  - `TTS_LEGACY_WORKER_ENV` / `LEGACY_TRANSFORMERS_DIST` / `LEGACY_HF_HUB_DIST`
  - `subtitle_to_audio` / `batch_subtitle_to_audio` 里给 worker 收尾的 `model.close()`
- `import base64` / `import json`（worker 用的）

全部已删除。

### 4. 跨版本回归测试

新增 `tests/test_qwen3_tts_regression.py`（3 个测试，需 `VRTB_TEST_RUN_TTS_MODEL=1` 启用）：

| 测试 | 防御目标 |
|---|---|
| `test_text_projection_biases_are_loaded` | bias 再次被 HF loader 吃掉 → 立刻红 |
| `test_rotary_inv_freq_recomputes_on_forward` | RoPE inv_freq 又变成全零 → 立刻红 |
| `test_chinese_synthesis_matches_input` | 端到端：合成 "最近肩膀疼。" 后跑 ASR，必须匹配输入 |

CI 跑这一条就能在以后升级 transformers / 改 vendor 时立刻发现破坏。

## 验证

| 项目 | 修复前（裸 5.x） | 修复后 |
|---|---|---|
| `inv_freq` (talker) | shape (64,) 全 0 | 1.24e-6 .. 1.0 正常 |
| `text_projection.linear_fc1.bias` rms | 0.0 | 0.00545 ✓ ckpt |
| `text_projection.linear_fc2.bias` rms | 0.0 | 0.0320 ✓ ckpt |
| "最近肩膀疼。" → max_new_tokens=200 | 9.84 秒 / ASR 无法识别 | 2.48 秒 / ASR 识别 "最近肩膀疼" |
| 现有测试 | — | 16 / 16 通过 |
| 新增 regression 测试 | — | 3 / 3 通过 |

## 改动一览

### vendor 修复（核心）

- `tool_si/_vendor/qwen_tts/core/models/modeling_qwen3_tts.py`
  `Qwen3TTSTalkerRotaryEmbedding` / `Qwen3TTSRotaryEmbedding` 两个类
  加 `_ensure_inv_freq`，forward 入口调用。
- `tool_si/_vendor/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py`
  `Qwen3TTSTokenizerV2DecoderRotatoryEmbedding` 同上。
- `tool_si/_vendor/qwen_tts/inference/qwen3_tts_model.py`
  `from_pretrained` 走完 HF 后调用 `_merge_safetensors_into_model`。

### 上层清理

- `tool_si/logic.py`：删除 worker 分支、`_repo_root` / archive-v0 扫描函数、
  `QwenTTSWorkerModel` 类、相关常量、`tts_model.close()` 调用、
  不再用的 `base64`/`json` import；`own_model` 跟踪变量也一并简化。
- `tool_si/qwen_tts_worker.py`：删除。
- `tool_si/_vendor/qwen_tts_legacy/`：整个目录删除。

### 测试

- `tests/test_qwen3_tts_regression.py`：新增（3 个测试）。

## 后续建议

1. **runtime_cache/uv_cache/archive-v0 可以清理掉**了。如果是 CI 缓存，那条专门
   pin 4.57.3 / 0.36.2 的步骤也可以从 CI 配置里去掉。
2. **CI 上启用回归测试**：`VRTB_TEST_RUN_TTS_MODEL=1 pytest tests/test_qwen3_tts_regression.py`。
   0.6B 模型 + faster-whisper-small，GPU 上 ~20 秒一轮，成本可控。
3. **给 HuggingFace transformers 提 issue**：Bug 2（biases 静默丢弃且不报 missing）
   不是 Qwen3-TTS 独有的，影响面可能更广。值得做一个最小复现报上去。
4. **依赖锁定**：把 `transformers>=5.9` 写进 `pyproject.toml` 的 tts extras 里，
   避免有人误装 4.x 触发我们没测过的代码路径。
