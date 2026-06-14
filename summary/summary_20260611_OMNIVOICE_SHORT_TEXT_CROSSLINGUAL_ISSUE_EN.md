# OmniVoice: cross-lingual voice cloning produces garbage for SHORT target text

**Date:** 2026-06-11
**Reporter:** VR-Video-Toolbox `tool_clonevoice` (Japanese video → Chinese dub, voice-cloned per speaker)
**Repro script:** `summary/omnivoice_short_text_repro.py` (+ `omnivoice_short_text_repro_result.json`)

---

## TL;DR

With a **single fixed reference** (a ~3.7 s Japanese clip + its transcript), `OmniVoice.generate()`:

- **clones LONG Chinese target text correctly**, but
- **produces garbage / wrong content for SHORT Chinese target text** (a few characters),
  regardless of the `language` argument (`"zh"`, `"Chinese"`, `None`) and regardless of
  generation parameters (`guidance_scale`, `position_temperature`, `class_temperature`, `num_step`).

This blocks dubbing, where many lines are short (e.g. "厉害啊" / "嗯") and **must** be spoken in
the **same cloned voice** as the speaker's long lines. We cannot switch to a different/auto voice
for short lines (a sudden voice/gender change is unacceptable).

**Question for experts:** Is there a supported way to clone a *short* line cross-lingually in a
specific voice? Is short target text a known limitation? Any required preprocessing, minimum
length, padding, or parameter setting?

---

## Environment

| Component | Version |
|---|---|
| omnivoice | 0.1.5 |
| model | `k2-fsa/OmniVoice` (local snapshot; `audio_tokenizer/` bundled) |
| torch | 2.8.0+cu128 (CUDA 12.8) |
| transformers | 5.9.0 |
| accelerate | 1.13.0 |
| numpy | 2.0.2 |
| GPU | NVIDIA RTX 5060 Ti (16 GB, sm_120), fp16 |
| read-back ASR | faster-whisper 1.2.1 + Whisper large-v3 (ctranslate2 4.8.0) |

`model = OmniVoice.from_pretrained("models/OmniVoice", device_map="cuda", dtype=torch.float16)`

## Setup

- **Reference (fixed for ALL targets):** a 3.70 s Japanese speech clip + its transcript
  `ref_text = "仕事できない人に対してどう教えていいか分かんないんだよね"` (within the recommended 3–10 s).
- **Targets:** Chinese sentences, long → short.
- Output is transcribed back with Whisper to judge it objectively (not by ear). OmniVoice output
  is quiet, so each clip is peak-normalized to 0.6 before ASR.

### Exact call (identical for every target)

```python
audio = model.generate(
    text=text,                 # the Chinese target line
    ref_audio=REF_AUDIO,       # fixed 3.7s Japanese wav
    ref_text=REF_TEXT,         # its Japanese transcript
    language="Chinese",        # also tried "zh" and None — no difference
    num_step=32,
    guidance_scale=2.0,
)[0]
```

(We also tried the reusable-prompt path `model.create_voice_clone_prompt(...)` + `voice_clone_prompt=` —
identical behavior.)

## Results (Whisper round-trip of the generated audio)

| target (chars) | generated | Whisper read-back | verdict |
|---|---|---|---|
| `你好，今天天气真不错，我们一起出去散步聊聊天吧。` (24) | 3.72 s | `你好今天天气真不错我们一起出去散步聊聊天吧` | **correct** |
| `后辈一下子多了好多。` (9) | 2.00 s | `好歹一生是多了好多` | partial / degraded |
| `加班多吧。` (5) | 1.60 s | `整理&字幕志愿者 杨茜茜` | **garbage** |
| `厉害啊。` (4) | 1.44 s | `很厲害` | partial (drops 啊) |

Short target with different `language` argument — all wrong:

| call | Whisper read-back |
|---|---|
| `language="zh"` | `嗯好玩多了` |
| `language="Chinese"` | `嗯就這麼多吧` |
| `language=None` | `嗯好玩的` |

## Parameter sweep (short target `加班多吧`, same fixed ref)

We swept `guidance_scale ∈ {2,3,4}`, `position_temperature ∈ {0,1,5}`, `class_temperature ∈ {0,0.5}`,
`num_step ∈ {32,64}` (10 combinations). **None** produced the correct short line; read-backs were e.g.
`嗯可惡多了`, `沒可不可以等嗎`, `蛤蟒多嗎`, `太棒了`, `一二三`, `本歌曲来自…`. The 3-char `厉害啊` occasionally
came out as `厲害` (dropping 啊), but `加班多吧` never came out correctly.

OmniVoice generation is also **stochastic** with the default `position_temperature=5.0` (gumbel sampling):
the same short input yields different (still wrong) outputs across runs.

## Reference length (3.7 s vs 10 s) — does NOT help

We rebuilt the reference as a 10 s clip (the upper end of the recommended 3–10 s) of the same
speaker, with its (longer) Japanese transcript, and re-ran the same targets:

| target | 3.7 s ref | 10 s ref |
|---|---|---|
| long | correct | correct |
| `后辈一下子多了好多` | `好歹一生是多了好多` (partial) | `小子天使` (**worse**) |
| `加班多吧` | garbage | `一 二 三` (garbage) |
| `厉害啊` | `很厲害` (partial) | `` (empty, **worse**) |

A longer reference (hence longer `ref_text`) made short/medium targets **worse**, not better —
consistent with issue #50 (long references degrade output) and the 3–10 s recommendation. So the
failure is driven by **short target length**, not reference length.

## Cross-checks (rule out our setup)

- **Auto voice (no reference)** with the *same short text* works: `model.generate(text="加班多吧。", language="zh")`
  → Whisper reads back `加班多嘛` (correct). So the **text and model are fine**; it is the **reference + short
  target** combination that breaks.
- **Long target + same reference** works (table above). So the **reference is fine**; it is the **short target**
  that breaks.
- Therefore the failure is specifically **(reference present) AND (target text short)**, cross-lingual
  (Japanese ref → Chinese target).

## Workaround we found (and why it is unsatisfactory)

Appending a carrier sentence — `text = "加班多吧。" + "我们改天再慢慢聊聊这件事情吧"` — makes the **short line
generate correctly** in the cloned voice (Whisper reads back the short line at the start). But then we must
**trim the carrier off**, and detecting the target/carrier boundary in the synthesized audio is unreliable
(Whisper mis-transcribes short synthetic clips; traditional/simplified mismatch; silence hallucinations).
The carrier also perturbs the target's duration. So this is not a robust solution.

## Questions for experts

1. Is **short target text** (a few characters) a known limitation of cross-lingual voice cloning in OmniVoice?
2. Is there a **supported way** to clone a short line in a specific voice — a minimum text length,
   required punctuation, a recommended `num_step`/`guidance_scale`/`t_shift`/temperature, or a different
   API path?
3. Does the issue relate to the **duration estimate** for short text (`RuleDurationEstimator`, `low_threshold`
   boost) producing too few target tokens? Would forcing more tokens via `duration`/`speed` be the intended
   fix? (We tried `duration=2.5`/`4.0` — still garbage.)
4. Is there guidance on cross-lingual cloning (reference language ≠ target language) for short utterances?

Repro: `python summary/omnivoice_short_text_repro.py` (prints the table and writes the JSON).
