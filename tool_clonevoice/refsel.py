"""Per-speaker reference sample selection for tool_clonevoice (P2).

For each diarized speaker, pick the best contiguous span (~4-9s) of that
speaker's speech to use as the OmniVoice voice-clone reference, cut it from the
ORIGINAL video at the model's native sample rate, and record ref_audio/ref_text
in the manifest.

Scoring favours: duration near the ideal, loud/clear audio, low internal
silence, sane text density, and minimal cross-talk overlap with other speakers.
"""
from __future__ import annotations

import math
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

LogCallback = Callable[[str], None]

REF_SR = 24000          # OmniVoice native sample rate
TARGET_MIN = 4.0        # preferred minimum reference duration (s)
TARGET_MAX = 9.0        # preferred maximum reference duration (s)
IDEAL = 6.0             # ideal reference duration (s)
ABS_MIN = 2.0           # absolute minimum usable span (s)
MAX_GAP = 0.8           # max silence between consecutive same-speaker segments to merge
TURN_PAD = 0.08         # trim diarization turn edges to avoid speaker-boundary bleed


def _build_startupinfo():
    if sys.platform != "win32":
        return None
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 0
    return si


def _read_wav_mono(path: str):
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a * a))) if a.size else 0.0


def _span_audio(audio: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    i0 = max(0, int(start * sr))
    i1 = min(audio.size, int(end * sr))
    return audio[i0:i1] if i1 > i0 else np.zeros(1, dtype=np.float32)


def _candidate_spans(segs: List[dict]) -> List[dict]:
    """Build merged contiguous candidate spans from one speaker's segments."""
    cands: List[dict] = []
    n = len(segs)
    for i in range(n):
        j = i
        while (
            j + 1 < n
            and (segs[j + 1]["start"] - segs[j]["end"]) <= MAX_GAP
            and (segs[j + 1]["end"] - segs[i]["start"]) <= TARGET_MAX
        ):
            j += 1
        span_start = float(segs[i]["start"])
        span_end = float(segs[j]["end"])
        dur = span_end - span_start
        if dur < ABS_MIN:
            continue
        text = " ".join(s["src_text"] for s in segs[i : j + 1] if s.get("src_text"))
        internal_gap = sum(
            max(0.0, segs[k + 1]["start"] - segs[k]["end"]) for k in range(i, j)
        )
        cands.append(
            {"start": span_start, "end": span_end, "dur": dur, "text": text, "gap": internal_gap}
        )
    return cands


def _turn_text(turn: dict, segments: List[dict], speaker: str) -> str:
    start, end = float(turn["start"]), float(turn["end"])
    texts = []
    for s in _source_refs(turn, segments, speaker):
        txt = (s.get("src_text") or "").strip()
        if txt:
            texts.append(txt)
    return " ".join(texts)


def _source_refs(turn: dict, segments: List[dict], speaker: str) -> List[dict]:
    start, end = float(turn["start"]), float(turn["end"])
    refs = []
    for s in sorted(segments, key=lambda x: x["start"]):
        if s.get("speaker") != speaker:
            continue
        ss, se = float(s["start"]), float(s["end"])
        overlap = max(0.0, min(end, se) - max(start, ss))
        if overlap <= 0:
            continue
        if overlap / max(0.1, se - ss) >= 0.5:
            refs.append({
                "srt_index": int(s.get("srt_index") or s.get("id") or 0),
                "id": int(s.get("id") or 0),
                "start": round(ss, 3),
                "end": round(se, 3),
                "speaker": s.get("speaker", ""),
                "text": (s.get("src_text") or "").strip(),
                "overlap": round(overlap, 3),
                "overlap_ratio": round(overlap / max(0.1, se - ss), 3),
            })
    return refs


def _overlap_total(start: float, end: float, intervals: List[tuple[float, float]]) -> float:
    return sum(max(0.0, min(end, e) - max(start, s)) for s, e in intervals)


def _subtract_intervals(start: float, end: float, blockers: List[tuple[float, float]]) -> List[tuple[float, float]]:
    spans = [(start, end)]
    for bs, be in sorted(blockers):
        next_spans: List[tuple[float, float]] = []
        for s, e in spans:
            if be <= s or bs >= e:
                next_spans.append((s, e))
                continue
            if bs > s:
                next_spans.append((s, min(bs, e)))
            if be < e:
                next_spans.append((max(be, s), e))
        spans = next_spans
        if not spans:
            break
    return [(s, e) for s, e in spans if e - s >= ABS_MIN]


def _window_span(start: float, end: float) -> List[tuple[float, float]]:
    dur = end - start
    if dur <= TARGET_MAX:
        return [(start, end)]
    windows = []
    t = start
    hop = 1.5
    while t + ABS_MIN <= end:
        w_end = min(end, t + TARGET_MAX)
        if w_end - t >= ABS_MIN:
            windows.append((t, w_end))
        if w_end >= end:
            break
        t += hop
    return windows


def _turn_candidates(speaker: str, manifest: dict, segments: List[dict]) -> List[dict]:
    """Prefer continuous single-speaker diarization spans for reference clips.

    ASR segments may cross speaker boundaries. Pyannote turns are usually cleaner
    for voice reference audio, so build candidates from target-speaker turns
    after subtracting other-speaker turns (expanded by TURN_PAD). ASR segments
    are only used to recover transcript text for the selected audio span.
    """
    all_turns = [
        t for t in manifest.get("diarization_turns", [])
        if float(t.get("end", 0)) > float(t.get("start", 0))
    ]
    other_intervals = [
        (max(0.0, float(t["start"]) - TURN_PAD), float(t["end"]) + TURN_PAD)
        for t in all_turns
        if t.get("speaker") != speaker
    ]

    exclusive_spans: List[tuple[float, float]] = []
    for t in all_turns:
        if t.get("speaker") != speaker:
            continue
        raw_start, raw_end = float(t["start"]), float(t["end"])
        start = raw_start + TURN_PAD
        end = raw_end - TURN_PAD
        if end <= start:
            start, end = raw_start, raw_end
        exclusive_spans.extend(_subtract_intervals(start, end, other_intervals))

    exclusive_spans = sorted(exclusive_spans)
    merged: List[tuple[float, float]] = []
    for start, end in exclusive_spans:
        if merged and start - merged[-1][1] <= MAX_GAP and _overlap_total(merged[-1][1], start, other_intervals) <= 0:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    cands: List[dict] = []
    for span_start, span_end in merged:
        for start, end in _window_span(span_start, span_end):
            dur = end - start
            if dur < ABS_MIN:
                continue
            text = _turn_text({"start": start, "end": end}, segments, speaker)
            cands.append({
                "start": start,
                "end": end,
                "dur": dur,
                "text": text,
                "gap": 0.0,
                "source": "turn",
                "other_turn_overlap": round(_overlap_total(start, end, other_intervals), 3),
                "source_srt_refs": _source_refs({"start": start, "end": end}, segments, speaker),
            })

    if not cands:
        # Overlap-heavy material (e.g. adult VR with constant cross-talk): the
        # other speaker's turns blanket the timeline, so subtracting them wipes
        # out every "exclusive" span and we get no candidate at all — then the
        # caller falls back to a 0.9s scrap. Instead, build candidates from the
        # speaker's RAW turns and let _score penalize cross-talk via
        # other_turn_overlap / overlap, keeping a long usable clip over a clean
        # but tiny one.
        for t in all_turns:
            if t.get("speaker") != speaker:
                continue
            for start, end in _window_span(float(t["start"]), float(t["end"])):
                if end - start < ABS_MIN:
                    continue
                text = _turn_text({"start": start, "end": end}, segments, speaker)
                cands.append({
                    "start": start,
                    "end": end,
                    "dur": end - start,
                    "text": text,
                    "gap": 0.0,
                    "source": "turn_raw",
                    "other_turn_overlap": round(_overlap_total(start, end, other_intervals), 3),
                    "source_srt_refs": _source_refs({"start": start, "end": end}, segments, speaker),
                })
    return cands


def _score(cand: dict, audio: np.ndarray, sr: int, other_segs: List[dict]) -> float:
    dur = cand["dur"]
    dur_s = math.exp(-((dur - IDEAL) ** 2) / (2 * 2.5 ** 2))
    if dur < 3.0:
        dur_s *= 0.5

    clip = _span_audio(audio, sr, cand["start"], cand["end"])
    rms_s = min(1.0, _rms(clip) / 0.08)

    gap_s = max(0.0, 1.0 - cand["gap"] / max(0.5, dur))

    nchars = len(cand["text"].replace(" ", ""))
    cps = nchars / max(0.1, dur)
    if nchars == 0:
        dens_s = 0.0  # no transcript -> likely non-speech (moans/breath): avoid
    else:
        dens_s = 1.0 if 2.0 <= cps <= 12.0 else 0.5

    overlap = 0.0
    for o in other_segs:
        overlap += max(0.0, min(cand["end"], o["end"]) - max(cand["start"], o["start"]))
    ov_s = max(0.0, 1.0 - overlap / max(0.1, dur))

    source_bonus = 0.12 if cand.get("source") == "turn" else 0.0
    turn_purity = max(0.0, 1.0 - float(cand.get("other_turn_overlap", 0.0)) / max(0.1, dur))

    # Text density is weighted heavily: a reference must carry real speech whose
    # transcript matches the audio, otherwise cloning degrades badly.
    return 0.17 * dur_s + 0.18 * rms_s + 0.08 * gap_s + 0.25 * dens_s + 0.15 * ov_s + 0.12 * turn_purity + source_bonus


def _cut_ref(video: str, start: float, end: float, out_wav: str, log: LogCallback) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg not found on PATH.")
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.1, end - start)
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{dur:.3f}",
        "-vn", "-ac", "1", "-ar", str(REF_SR), "-c:a", "pcm_s16le", str(out_wav),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, errors="replace", startupinfo=_build_startupinfo()
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg ref cut failed: {(proc.stderr or '').strip()}")


def _write_reference_report(path: Path, manifest: dict) -> None:
    lines = ["# Voice Clone Reference Sources", ""]
    for spk, info in sorted(manifest.get("speakers", {}).items()):
        lines.append(f"## {spk}")
        lines.append("")
        lines.append(f"- ref_audio: `{info.get('ref_audio', '')}`")
        lines.append(f"- ref_time: {float(info.get('start', 0.0)):.3f}s - {float(info.get('end', 0.0)):.3f}s")
        lines.append(f"- source: {info.get('source', '')}")
        lines.append(f"- score: {info.get('score', '')}")
        refs = info.get("source_srt_refs") or []
        if refs:
            lines.append("- source.srt:")
            for r in refs:
                lines.append(
                    f"  - #{int(r.get('srt_index', 0))} "
                    f"{float(r.get('start', 0.0)):.3f}-{float(r.get('end', 0.0)):.3f}s "
                    f"[{r.get('speaker', '')}] {r.get('text', '')}"
                )
        else:
            lines.append("- source.srt: none matched inside ref span")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def extract_references(
    video: str,
    manifest: dict,
    audio16k_path: str,
    clone_dir: Path,
    log: LogCallback = print,
) -> dict:
    """Select + cut one reference clip per speaker, updating ``manifest`` in place."""
    audio, sr = _read_wav_mono(audio16k_path)
    segments = manifest.get("segments", [])
    speakers = list(manifest.get("speakers", {}).keys())
    if not speakers:
        speakers = sorted({s["speaker"] for s in segments})

    for spk in speakers:
        spk_segs = sorted((s for s in segments if s["speaker"] == spk), key=lambda s: s["start"])
        other_segs = [s for s in segments if s["speaker"] != spk]

        turn_cands = _turn_candidates(spk, manifest, segments)
        # Also offer ASR-segment spans that actually carry transcript text. In
        # overlap-heavy material (adult VR) the cleanest single-speaker turns are
        # often non-speech (moans/breath) with NO transcript; a ref whose ref_text
        # doesn't match its audio clones terribly. Let scored text density (below)
        # decide between a clean-but-wordless turn and a real-speech span.
        text_segs = [s for s in spk_segs if (s.get("src_text") or "").strip()]
        seg_cands = _candidate_spans(text_segs) if text_segs else []
        for c in seg_cands:
            c["source"] = "segment"
            c["other_turn_overlap"] = 0.0
            c["source_srt_refs"] = _source_refs(
                {"start": c["start"], "end": c["end"]}, segments, spk)
        cands = turn_cands + seg_cands
        if cands:
            log(f"[ref] {spk}: {len(turn_cands)} turn + {len(seg_cands)} text-segment candidates")
        elif not spk_segs:
            log(f"[ref] {spk}: no assigned segments or diarization-turn candidates, skipped.")
            continue
        if not cands:
            # Fall back to the single longest segment even if short.
            longest = max(spk_segs, key=lambda s: s["end"] - s["start"])
            cands = [{
                "start": float(longest["start"]), "end": float(longest["end"]),
                "dur": float(longest["end"] - longest["start"]),
                "text": longest.get("src_text", ""), "gap": 0.0, "source": "segment",
                "source_srt_refs": _source_refs(
                    {"start": float(longest["start"]), "end": float(longest["end"])},
                    segments,
                    spk,
                ),
            }]

        best = max(cands, key=lambda c: _score(c, audio, sr, other_segs))
        score = _score(best, audio, sr, other_segs)

        ref_name = f"ref_{spk}.wav"
        ref_path = clone_dir / ref_name
        _cut_ref(video, best["start"], best["end"], str(ref_path), log)

        manifest.setdefault("speakers", {})[spk] = {
            "ref_audio": ref_name,
            "ref_text": best["text"].strip(),
            "score": round(score, 3),
            "source": best.get("source", "segment"),
            "source_srt_refs": best.get("source_srt_refs", []),
            "start": round(best["start"], 3),
            "end": round(best["end"], 3),
            "dur": round(best["dur"], 3),
        }
        refs = manifest["speakers"][spk]["source_srt_refs"]
        ref_idx = ",".join(f"#{r['srt_index']}" for r in refs[:6]) if refs else "none"
        log(
            f"[ref] {spk}: {best['start']:.2f}-{best['end']:.2f}s "
            f"({best['dur']:.1f}s, score={score:.2f}, srt={ref_idx}) -> {ref_name} | {best['text'][:40]}"
        )

    _write_reference_report(clone_dir / "references.md", manifest)
    log(f"[ref] reference source report -> {clone_dir / 'references.md'}")
    return manifest
