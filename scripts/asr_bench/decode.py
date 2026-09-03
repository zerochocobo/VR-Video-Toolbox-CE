"""Re-run the ASR over a benchmark item with different settings.

Unlike the post-processing variants in :mod:`variants`, these actually decode,
so they need the GPU and take about as long as a normal transcription. They
exist so a framing or VAD change can be measured the same way a filter change
is: one variable at a time, against the same audio, into ``bench.<tag>.srt``.

Settings are applied by patching module-level constants for the duration of the
run. That is how the production code reads them -- there is no settings object to
pass -- and patching them here keeps the variant definitions honest about which
globals a change really touches.

A constant name may be prefixed with the module that owns it: ``engine.`` for
``tool_clonevoice_v2.segment_engine``, anything unprefixed for
``tool_subtitle.logic``. The decode window lives in the engine because the dub
and the subtitle tool want different values from it.

Derived constants must be listed explicitly: ``AUDITOK_MAX_DURATION`` and
``DUPLICATE_LOOKBACK_SECONDS`` are computed from ``tool_subtitle.logic``'s
``CHUNK_SECONDS`` at import time and will not follow it.
"""
from __future__ import annotations

import contextlib
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.asr_bench import corpus  # noqa: E402
from scripts.asr_bench.variants import _write_srt  # noqa: E402

MODELS_ROOT = str(_REPO_ROOT / "models")


@contextlib.contextmanager
def patched(constants: dict):
    """Temporarily replace module constants named ``[engine.]NAME``."""
    from tool_clonevoice_v2 import segment_engine
    from tool_subtitle import logic as subl

    modules = {"engine": segment_engine, "": subl}
    previous = []
    try:
        for key, value in constants.items():
            prefix, _, name = key.rpartition(".")
            module = modules.get(prefix)
            if module is None:
                raise KeyError(f"unknown constant prefix in {key!r}")
            if not hasattr(module, name):
                raise AttributeError(f"{module.__name__} has no constant {name!r}")
            previous.append((module, name, getattr(module, name)))
            setattr(module, name, value)
        yield
    finally:
        for module, name, value in reversed(previous):
            setattr(module, name, value)


def _preset(**overrides) -> dict:
    """A VAD sensitivity preset built from ``high``, with fields replaced.

    Written as a factory rather than a literal so a future change to the shipped
    ``high`` preset carries into every variant derived from it.
    """
    from tool_subtitle import logic as subl

    preset = dict(subl.VAD_SENSITIVITY_PRESETS["high"])
    preset.update(overrides)
    return preset


def _with_preset(name: str, **overrides) -> dict:
    """Constants that install a one-off sensitivity preset under ``name``."""
    from tool_subtitle import logic as subl

    presets = dict(subl.VAD_SENSITIVITY_PRESETS)
    presets[name] = _preset(**overrides)
    return {"VAD_SENSITIVITY_PRESETS": presets}


# Each entry is (constants, kwargs-for-transcribe). Keep every variant one step
# away from the baseline: two changes at once cannot be attributed.
def variant_specs() -> dict[str, tuple[dict, dict]]:
    return {
        # The pipeline as it stood before 2026-09-02: the long window and none
        # of the repair stages. This is the only honest "before" for an
        # end-to-end number -- "control" below changes the window back but keeps
        # the repairs, so it measures the window alone, not the whole change.
        "legacy": ({"engine.CHUNK_SECONDS": 28.0,
                    "engine.WHISPERSEG_MERGE_GAP_SECONDS": 2.0,
                    "engine.ENABLE_TORN_MERGE": False,
                    "engine.ENABLE_NONVERBAL_DROP": False,
                    "engine.ENABLE_START_TIGHTEN": False}, {}),
        # The old window, new repairs: isolates what the window is worth.
        "control": ({"engine.CHUNK_SECONDS": 28.0,
                     "engine.WHISPERSEG_MERGE_GAP_SECONDS": 2.0}, {}),
        # The new window, old post-processing: isolates what the repairs are worth.
        "norepair": ({"engine.ENABLE_TORN_MERGE": False,
                      "engine.ENABLE_NONVERBAL_DROP": False,
                      "engine.ENABLE_START_TIGHTEN": False}, {}),

        # --- VAD gate ------------------------------------------------------
        # WhisperJAV's JAV-tuned threshold is 0.25, between our high (0.35) and
        # max (0.20). Our own measurements said high is too conservative and max
        # too permissive, so the gap is exactly where the answer should be.
        "vad025": (_with_preset("bench", threshold=0.25, neg_threshold=0.16),
                   {"vad_sensitivity": "bench"}),
        "vadmax": ({}, {"vad_sensitivity": "max"}),
        # Measured worse than both: see the sweep table in
        # summary/summary_20260902_CLONEVOICE_V2_ASR_WHISPERJAV_STUDY_CN.md.

        # --- grouping ------------------------------------------------------
        # We merge speech regions separated by up to 2.0s into one chunk;
        # WhisperJAV groups only across 0.3s. Everything else held fixed.
        "gap03": ({"engine.CHUNK_SECONDS": 28.0,
                   "engine.WHISPERSEG_MERGE_GAP_SECONDS": 0.3}, {}),

        # --- framing (the structural experiment) ---------------------------
        # Short windows without changing the model: if the 28s window is what
        # drives the fragments and stock phrases, this is where it shows.
        "frame8": ({"engine.CHUNK_SECONDS": 8.0,
                    "engine.WHISPERSEG_MERGE_GAP_SECONDS": 2.0}, {}),
        # The shipped default since 2026-09-02; kept as a named variant so a
        # later change can still be measured against it.
        "frame8gap03": ({"engine.CHUNK_SECONDS": 8.0,
                         "engine.WHISPERSEG_MERGE_GAP_SECONDS": 0.3}, {}),
        # WhisperJAV's own JAV grouping is 3.0s / 0.3s -- one utterance per
        # decode. Their recognizer emits text only and takes its timing from the
        # frame; ours still segments inside the window, so this is as close as
        # the current architecture gets to ChronosJAV. If shorter keeps winning,
        # the decoupled pipeline is worth building; if it turns over here, 8s is
        # the floor for a model that wants context.
        "frame5gap03": ({"engine.CHUNK_SECONDS": 5.0,
                         "engine.WHISPERSEG_MERGE_GAP_SECONDS": 0.3}, {}),
        "frame3gap03": ({"engine.CHUNK_SECONDS": 3.0,
                         "engine.WHISPERSEG_MERGE_GAP_SECONDS": 0.3}, {}),
        # Below their 3.0s, to find where a window stops holding an utterance.
        "frame2gap03": ({"engine.CHUNK_SECONDS": 2.0,
                         "engine.WHISPERSEG_MERGE_GAP_SECONDS": 0.3}, {}),

        # --- enhance for VAD only ------------------------------------------
        # The detector listens to a denoised copy, the decoder to the original.
        # Denoising the ASR feed is what our DENOISE_FILTERS presets do today,
        # and it cuts both ways -- quiet speech clears the gate, consonants get
        # dulled. This takes the first half without the second.
        "vadclean": ({}, {"vad_denoise": "balanced"}),
        "vadcleanstrong": ({}, {"vad_denoise": "strong"}),

        # Current production. Named so a later change has something to sit next
        # to; identical to omitting every override.
        "shipped": ({}, {}),
        # 28s window with the over-long-region cap in place: the old framing,
        # the new correctness fix, and the repairs.
        "win28cap": ({"engine.CHUNK_SECONDS": 28.0,
                      "engine.WHISPERSEG_MERGE_GAP_SECONDS": 2.0}, {}),

        # --- model ---------------------------------------------------------
        "largev3": ({}, {"model_key": "large-v3"}),
        # Framing from the VAD, words from a domain-matched recogniser. The
        # decoupled arrangement, with the window already short enough for it.
        # Text from anime-whisper over a 15s window, word times from the
        # forced aligner, lines cut by the same splitting the kotoba path uses.
        "anime": ({}, {"model_key": "anime-whisper"}),
        # The same without the aligner, for attribution.
        "anime3": ({"engine.ANIME_CHUNK_SECONDS": 3.0,
                    "engine.ANIME_MERGE_GAP_SECONDS": 0.3},
                   {"model_key": "anime-whisper"}),
        # anime-whisper stayed silent on ~29% of 3s windows. It is a Whisper
        # model and can take 30s, so the question is whether it goes quiet
        # because the window holds too little to be worth a sentence.
        "anime8": ({"engine.ANIME_CHUNK_SECONDS": 8.0,
                    "engine.ANIME_MERGE_GAP_SECONDS": 1.0},
                   {"model_key": "anime-whisper"}),
    }


def _denoised_copy(item: corpus.Item, preset: str, log) -> str:
    """A denoised 16 kHz copy of the item's audio, cached beside it."""
    import subprocess

    from tool_clonevoice_v2 import whisperx_backend as wx

    target = item.clone_dir / f"bench.vad-{preset}.wav"
    if target.is_file():
        return str(target)
    filter_str = wx.DENOISE_FILTERS.get(preset)
    if not filter_str:
        raise SystemExit(f"unknown denoise preset {preset!r}")
    log(f"[bench] building {target.name} ({preset})")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(item.audio), "-af", filter_str,
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target)],
        check=True,
    )
    return str(target)


def run(tag: str, item: corpus.Item, language: str = "ja",
        model_key: str = "kotoba", log=print) -> Path:
    from tool_clonevoice_v2 import segment_engine

    specs = variant_specs()
    if tag not in specs:
        raise SystemExit(f"unknown decode variant '{tag}'; "
                         f"have: {', '.join(sorted(specs))}")
    constants, kwargs = specs[tag]
    kwargs = {"model_key": model_key, "vad_sensitivity": "high", **kwargs}
    denoise = kwargs.pop("vad_denoise", None)
    if denoise:
        kwargs["vad_audio_path"] = _denoised_copy(item, denoise, log)

    started = time.time()
    with patched(constants):
        result = segment_engine.transcribe(
            str(item.audio),
            models_root=MODELS_ROOT,
            language=language,
            log=log,
            **kwargs,
        )
    elapsed = time.time() - started
    path = _write_srt(item.variant(tag), result["segments"])
    log(f"[bench] {item.name} / {tag}: {len(result['segments'])} lines "
        f"in {elapsed / 60:.1f} min -> {path.name}")
    return path
