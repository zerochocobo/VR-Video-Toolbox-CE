# ASR benchmark

Turns a transcription into numbers, so an ASR change can be judged against a
fixed corpus instead of by listening to one title.

## The corpus

Every `<video>.clone` work directory keeps `audio16k.wav` (exactly what the ASR
saw) and `source.srt` (what it produced). A variant run therefore never touches
the 8K master — the largest input is a ~100 MB wav.

Roots live in `corpus.json`; they are machine-local paths, edit them freely.

```bash
python -m scripts.asr_bench list
```

## Commands

Global options (`--filter`, `--json`, `--examples`, `--root`) go **before** the
subcommand. `--filter` takes a comma-separated list of name substrings.

```bash
# reference-free statistics of the stored baseline
python -m scripts.asr_bench profile

# same, for a variant produced by build/decode
python -m scripts.asr_bench profile --variant postfix

# replay production post-processing over the baseline (seconds, no GPU)
python -m scripts.asr_bench build postfix

# re-run the ASR with different settings (GPU, ~14x realtime)
python -m scripts.asr_bench --filter hnvr00174_2 decode gap03

# A/B two runs over the same audio
python -m scripts.asr_bench compare gap03 --base-variant control

# accuracy against a reference SRT
python -m scripts.asr_bench score --gt-suffix .jp.srt
```

## What the numbers mean

| Metric | Reads as |
|---|---|
| `speech_seconds` / `coverage` | Union of line spans, overlaps counted once. Summing durations double-counts overlapping chunk decodes. |
| `stock_phrase` | Known Whisper filler ("ご視聴ありがとうございました"). Pure invention; in the dub it is spoken aloud. |
| `nonverbal` / `lone_token` | Moans, breaths, lone artifact tokens. Diagnostic only — production abstains below three characters, so this can be non-zero in a variant on purpose. |
| `fragment` | ≤3 characters after normalisation. The pressure gauge: it moved 3–13% → 31–46% between the `high` and `max` VAD presets. |
| `glued_fragment` | A fragment whose neighbour is within 0.35 s — one word torn in half, to be rejoined rather than deleted. |
| `sparse` | ≥3 s at <1 char/s. A syllable the aligner smeared across a pause. |
| `long_line` | Over 8 s. For dubbing a defect on its own: one cloned take stretched over two utterances. |
| `overlapping` | Lines whose spans overlap by more than 0.15 s. |

There is no human ground truth for this corpus, so `score` is only meaningful
once one exists. The old `tool_subtitle` `.jp.srt` files are a **reference, not
ground truth** — their durations were remapped for readability and their
fragments merged, which inflates speech seconds.

## Adding a variant

Post-processing changes go in `variants.py` (`BUILDERS`); they call the real
production methods, never a copy. Decode changes go in `decode.py`
(`variant_specs`), which patches `tool_subtitle.logic` module constants for the
duration of the run.

Keep every variant **one step** from the baseline. Two changes at once cannot be
attributed — and always run `control` alongside, because CUDA decoding is not
deterministic and `control` measures the noise floor the other numbers sit on.

Derived constants do not follow their source: `AUDITOK_MAX_DURATION` and
`DUPLICATE_LOOKBACK_SECONDS` are computed from `CHUNK_SECONDS` at import, so a
framing variant must list them explicitly.

## Baselines

`results/` holds the recorded runs. `baseline_20260902.*` is the state of the
stored `source.srt` files before any of this work — 1630 lines over 5.5 h.
