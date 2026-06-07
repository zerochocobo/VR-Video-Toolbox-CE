# Qwen3-TTS Subtitle Audio Speed and Language Fix Summary

## Background

The subtitle-to-audio path in the simultaneous interpretation tool exposed two user-facing issues:

1. **Very slow generation**: `Hoppers...chs.srt` contained 1417 subtitle entries with an output duration of about 6198.448 seconds. The user stopped the task after the first 10 entries; elapsed time was about 2 minutes 41 seconds.
2. **Unrecognizable output language**: `videos/35.srt` contained 11 entries and generated `videos/35.si.wav` in about 55 seconds, but the audio did not sound like usable Chinese or Japanese subtitle speech.

Important clues from the logs:

- `FlashAttention2 is not installed; using eager attention.`
- Small batch generation improved speed, but output quality was still wrong.
- `videos/35.srt` contains Chinese and Japanese lines in each subtitle block, while the old parser did not choose a line based on the target language.

## Root Causes

### 1. Slow TTS inference path

- Without FlashAttention2, the old path fell back directly to eager attention, which is inefficient on CUDA.
- Subtitle entries were generated one by one, causing poor GPU utilization across 1417 small requests.
- `max_new_tokens` was too loose, so short subtitles could still spend unnecessary decoding steps.
- Duration fitting could launch an ffmpeg subprocess per subtitle entry, which is expensive for many short lines.

### 2. Wrong text line selection in bilingual subtitles

- `parse_srt()` previously selected the first non-empty subtitle line.
- In bilingual files such as `videos/35.srt`, Chinese and Japanese lines coexist; the parser needed target-language-aware selection.

### 3. Bad output quality from the current in-process Qwen3-TTS runtime

- The main environment uses `transformers 5.9.0`.
- The local 5.9-compatible vendor path generated unrelated filler such as subscription/thanks text during testing, and ASR could not recognize it as the target subtitle content.
- The official `qwen-tts 0.1.1` runtime with `transformers 4.57.3` and `huggingface_hub 0.36.2` correctly generated the same test line `最近肩膀疼`, and ASR recognized it.

## Fixes

### 1. Faster subtitle TTS

Implemented in `tool_si/logic.py`:

- Prefer PyTorch SDPA attention on CUDA when FlashAttention2 is unavailable; fall back to eager only if SDPA loading fails.
- Generate subtitle entries in small batches instead of one by one. Default `VRTB_TTS_BATCH_SIZE=4`.
- Group subtitles by similar token budgets, reducing distortion from mixing very short and long lines in one batch.
- Added `VRTB_TTS_BATCH_TOKEN_SPREAD`, default `1.5`, capped at `10.0`.
- Tightened `max_new_tokens` using the 12Hz codec duration budget plus a text-length floor.
- Use in-memory `librosa.effects.time_stretch` by default for duration fitting, avoiding per-entry ffmpeg subprocess startup.
- Kept `VRTB_TTS_TIME_FIT_MODE=ffmpeg` for explicitly returning to the old ffmpeg path.
- Strip SRT/ASS override tags such as `{\an8}` before sending text to TTS.

### 2. Target-language-aware subtitle parsing

`parse_srt(path, language=...)` now selects subtitle lines by target language:

- Chinese prefers Chinese/Han text.
- Japanese prefers lines containing kana.
- Korean prefers Hangul lines.
- English prefers Latin text.
- If no line matches, it falls back to the first non-empty line.

### 3. Correct output content with an official legacy worker

Added an official legacy worker path:

- Added `tool_si/_vendor/qwen_tts_legacy/`, based on official `qwen-tts 0.1.1`, while keeping the existing 12Hz lazy-import patch and avoiding unnecessary 25Hz/torchaudio/sox imports.
- Added `tool_si/qwen_tts_worker.py`, which loads the legacy official runtime in a subprocess. The main process sends TTS requests over JSON lines.
- The main process scans `runtime_cache/uv_cache/archive-v0` for `transformers-4.57.3.dist-info` and `huggingface_hub-0.36.2.dist-info`; when available, it uses the official worker by default.
- The worker loads the model only once and still supports batched subtitle requests.
- `VRTB_TTS_LEGACY_WORKER=0` disables the worker and falls back to the in-process runtime.
- `subtitle_to_audio()` and `batch_subtitle_to_audio()` close the worker on completion or error.

## Verification

- Real-model short sentence batch smoke test succeeded through the SDPA path.
- First 10 entries of `videos/Hoppers.2026.BluRay.1080p.10Bit.x265.AAC(7.1).chs.srt`: batched generation after model load took about 34.64 seconds. The original user log reached stop at about 2 minutes 41 seconds for the first 10 entries, including load/stop overhead, so the improvement is substantial.
- Full `videos/35.srt` with 11 entries: `subtitle_to_audio()` output was recognized by ASR as subtitle content, no longer unrelated filler or unrecognizable speech.
- Japanese mode: `parse_srt('videos/35.srt', language='Japanese')` selected `最近肩が痛くて`; generation with `Ono_Anna` was recognized by ASR as the target Japanese line.
- Compile check passed:
  - `.venv\Scripts\python.exe -m py_compile tool_si\logic.py tool_si\qwen_tts_worker.py tests\test_tool_si.py tool_si\_vendor\qwen_tts\core\models\modeling_qwen3_tts.py`
- Tests passed:
  - `.venv\Scripts\python.exe -m pytest tests\test_tool_si.py tests\test_i18n.py -q`
  - Result: `16 passed`

## Operational Notes

- Keep `VRTB_TTS_LEGACY_WORKER` enabled by default to prioritize correct speech content.
- If VRAM is sufficient, try `VRTB_TTS_BATCH_SIZE=6` or `8` and benchmark the same subtitle file.
- If subtitle lengths vary heavily, `VRTB_TTS_BATCH_TOKEN_SPREAD` can be raised moderately, but too high a value may mix very short and long entries and hurt duration fitting.
- If FlashAttention2 is installed later, re-run the same benchmark and compare it against SDPA.

## Remaining Risk

- The first 10 real entries and the full 11-entry small sample were verified. The complete 1417-entry task still needs a full benchmark on the user machine.
- Higher batch sizes may increase VRAM usage, so the default remains conservative.
- The legacy worker depends on official dependency caches under `runtime_cache/uv_cache/archive-v0`. If that cache is removed, the required versions must be restored or `VRTB_TTS_LEGACY_WORKER=0` can be used temporarily.
