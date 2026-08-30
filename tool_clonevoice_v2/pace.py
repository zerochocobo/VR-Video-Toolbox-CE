"""Line-by-line comparison of a dub against the source it replaces.

The complaint this answers: the Chinese line finishes and the original Japanese
is still going underneath it, muffled. Mixing ducks the original across the
whole source span, so whatever the cloned line does not cover is left exposed
-- and measured across four real titles the cloned line covers 29-42% of its
slot at the median. This module puts the per-line numbers behind that where
they can be read and listened to, rather than inferred from the mix.

Nothing here synthesizes or modifies anything; it reads the manifest and the
finished dub track.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from tool_clonevoice_v2 import logic


# Below this a sample is the dub track's own silence rather than a rendition.
# The track is written as clips placed on silence, so the floor only has to
# clear dither and the encoder's noise, not a room tone.
CLIP_SILENCE_FLOOR = 1e-4

# How far past its own slot a line is allowed to be measured when nothing
# follows it. Lines may overrun into the gap behind them by design.
TRAILING_WINDOW_SECONDS = 30.0

# A line covering less than this much of its slot leaves a stretch of ducked
# original with nothing over it -- audible as "the Japanese is still talking".
EXPOSED_COVERAGE = 0.90


def measure_clone_seconds(
    si_path: str | Path,
    segments: list[dict],
    *,
    floor: float = CLIP_SILENCE_FLOOR,
) -> list[Optional[float]]:
    """How long each cloned line actually speaks, read off the dub track.

    The dub holds one clip per line at that line's start time and silence
    everywhere else, so the run of sound from a line's start is its rendition.
    Reading it back needs no intermediate clips, which only exist when the user
    asked to keep them.

    A line is measured up to the next line's start, so a rendition that
    overruns that far is reported as reaching it and no further.
    """
    import soundfile as sf

    path = Path(si_path)
    if not path.is_file() or not segments:
        return [None] * len(segments)

    starts = [float(s.get("start") or 0.0) for s in segments]
    out: list[Optional[float]] = []
    with sf.SoundFile(str(path)) as handle:
        sr = int(handle.samplerate)
        frames = int(handle.frames)
        for index, segment in enumerate(segments):
            start = starts[index]
            if index + 1 < len(segments):
                limit = max(start, starts[index + 1])
            else:
                limit = float(segment.get("end") or start) + TRAILING_WINDOW_SECONDS
            begin = max(0, int(round(start * sr)))
            stop = min(frames, int(round(limit * sr)))
            if stop <= begin:
                out.append(0.0)
                continue
            handle.seek(begin)
            block = handle.read(stop - begin, dtype="float32", always_2d=False)
            if block.ndim > 1:
                block = block.mean(axis=1)
            loud = np.flatnonzero(np.abs(block) > floor)
            out.append(0.0 if loud.size == 0 else float(loud[-1] + 1) / sr)
    return out


def read_window(
    path: str | Path, start: float, end: float
) -> tuple[np.ndarray, int, float]:
    """Mono samples for ``[start, end)``, plus the rate and the real start.

    Clamped to the file, so the returned start says where the window actually
    begins -- a line near time zero cannot have the padding it asked for, and
    drawing it as though it did would put the waveform under the wrong marks.
    """
    import soundfile as sf

    with sf.SoundFile(str(path)) as handle:
        sr = int(handle.samplerate)
        frames = int(handle.frames)
        begin = max(0, min(frames, int(round(float(start) * sr))))
        stop = max(begin, min(frames, int(round(float(end) * sr))))
        if stop <= begin:
            return np.zeros(0, dtype=np.float32), sr, begin / sr
        handle.seek(begin)
        block = handle.read(stop - begin, dtype="float32", always_2d=False)
    if block.ndim > 1:
        block = block.mean(axis=1)
    return np.asarray(block, dtype=np.float32), sr, begin / sr


def peak_envelope(samples: np.ndarray, buckets: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-bucket minimum and maximum, for drawing a waveform.

    Peaks rather than an average: a consonant lasting a few samples is what
    tells you where a line really starts, and averaging buries it.
    """
    mono = np.asarray(samples, dtype=np.float32)
    if mono.ndim > 1:
        mono = mono.mean(axis=1)
    if mono.size == 0 or buckets <= 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    buckets = min(int(buckets), mono.size)
    edges = np.linspace(0, mono.size, buckets + 1, dtype=np.int64)
    low = np.minimum.reduceat(mono, edges[:-1]).astype(np.float32, copy=False)
    high = np.maximum.reduceat(mono, edges[:-1]).astype(np.float32, copy=False)
    return low, high


def display_gain(
    low: np.ndarray, high: np.ndarray, *,
    target: float = 0.7, minimum: float = 1.0, maximum: float = 24.0,
) -> float:
    """Display-only gain so quiet speech is still visible.

    A dub sits well below the source's peaks, and drawn at true scale its line
    endings are a flat line -- which is the thing being looked for here.
    """
    peaks = np.maximum(
        np.abs(np.asarray(low, dtype=np.float32)),
        np.abs(np.asarray(high, dtype=np.float32)),
    )
    if peaks.size == 0:
        return minimum
    reference = float(np.percentile(peaks, 99.0))
    if not np.isfinite(reference) or reference <= 1e-6:
        return minimum
    return float(np.clip(target / reference, minimum, maximum))


def _chars(text: str) -> int:
    return len("".join((text or "").split()))


def analyze_video(video: str | Path) -> dict[str, Any]:
    """Per-line source/dub comparison plus the summary over the whole title.

    Works without a dub track: the text side is still worth reading on its own,
    and the durations simply come back empty.
    """
    video_path = Path(video)
    manifest = logic.load_manifest(video_path)
    if manifest is None:
        raise FileNotFoundError(
            f"Manifest not found; transcribe this video first: {logic.manifest_path(video_path)}"
        )

    from tool_clonevoice_v2 import proofread

    segments = [
        s for s in manifest.get("segments", [])
        if s.get("start") is not None and s.get("end") is not None
    ]
    segments.sort(key=lambda s: float(s.get("start") or 0.0))
    cloned = proofread.cloned_track_path(video_path)
    spoken = measure_clone_seconds(cloned, segments) if cloned else [None] * len(segments)

    rows: list[dict[str, Any]] = []
    for segment, clone_seconds in zip(segments, spoken):
        start = float(segment.get("start") or 0.0)
        end = float(segment.get("end") or 0.0)
        slot = max(0.0, end - start)
        src_text = (segment.get("src_text") or "").strip()
        tgt_text = (segment.get("tgt_text") or "").strip()
        coverage = None
        exposed = None
        if clone_seconds is not None and slot > 0:
            coverage = clone_seconds / slot
            exposed = max(0.0, slot - clone_seconds)
        rows.append({
            "id": segment.get("id"),
            "start": start,
            "end": end,
            "slot": slot,
            "clone": clone_seconds,
            "coverage": coverage,
            "exposed": exposed,
            "speaker": segment.get("speaker") or "",
            "src_text": src_text,
            "tgt_text": tgt_text,
            "src_chars": _chars(src_text),
            "tgt_chars": _chars(tgt_text),
        })

    return {
        "video": str(video_path),
        "cloned_track": str(cloned) if cloned else "",
        "rows": rows,
        "summary": summarize(rows),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The numbers that say whether the dub keeps pace with the source."""
    spoken = [r for r in rows if r.get("coverage") is not None and r.get("slot", 0) > 0]
    translated = [r for r in rows if r.get("tgt_text")]
    summary: dict[str, Any] = {
        "lines": len(rows),
        "translated": len(translated),
        "measured": len(spoken),
        "median_coverage": None,
        "under_exposed": 0,
        "under_half": 0,
        "exposed_total": 0.0,
        "median_exposed": None,
        "median_char_ratio": None,
    }
    pairs = [
        (r["tgt_chars"], r["src_chars"]) for r in translated if r.get("src_chars")
    ]
    if pairs:
        summary["median_char_ratio"] = float(np.median(
            [tgt / src for tgt, src in pairs]
        ))
    if not spoken:
        return summary
    coverage = np.array([r["coverage"] for r in spoken], dtype=float)
    exposed = np.array([r["exposed"] for r in spoken], dtype=float)
    summary["median_coverage"] = float(np.median(coverage))
    summary["under_exposed"] = int((coverage < EXPOSED_COVERAGE).sum())
    summary["under_half"] = int((coverage < 0.5).sum())
    summary["exposed_total"] = float(exposed.sum())
    summary["median_exposed"] = float(np.median(exposed))
    return summary
