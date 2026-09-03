"""Text normalisation, character error rate and timing overlap.

Character-level rather than word-level because the source languages here
(Japanese, Chinese) have no orthographic word boundaries.
"""
from __future__ import annotations

import re
import unicodedata

# Speaker labels in broadcast-style reference subtitles -- （佐藤）, (男性),
# （ナレーション）. ASR never produces them, so they would otherwise count as
# pure deletions against a reference.
_SPEAKER_LABEL_RE = re.compile(r"[（(][^）)]{0,12}[）)]")

# Punctuation whose presence differs between transcription systems without
# changing what was said.
_DROP_CHARS = set("。、！？「」『』（）()…・〜～.,!?\"'  　-")


def normalize_text(text: str) -> str:
    """Fold a subtitle line to the form used for every text comparison."""
    text = unicodedata.normalize("NFKC", text or "")
    text = _SPEAKER_LABEL_RE.sub("", text)
    text = "".join(text.split())
    return "".join(ch for ch in text if ch not in _DROP_CHARS)


def levenshtein(a: str, b: str) -> int:
    """Edit distance, iterative with a single row (lines are short)."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        cur = [i + 1]
        for j, cb in enumerate(b):
            cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(hypothesis: str, reference: str) -> float:
    """Character error rate of ``hypothesis`` against ``reference``.

    Returns 0.0 when both are empty and 1.0 when only the hypothesis has
    content, so an empty reference never divides by zero.
    """
    hyp = normalize_text(hypothesis)
    ref = normalize_text(reference)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(hyp, ref) / len(ref)


def similarity(a: str, b: str) -> float:
    """Normalised character overlap in [0, 1]; the inverse of :func:`cer`."""
    a_n, b_n = normalize_text(a), normalize_text(b)
    if not a_n and not b_n:
        return 1.0
    longest = max(len(a_n), len(b_n))
    if longest == 0:
        return 1.0
    return 1.0 - levenshtein(a_n, b_n) / longest


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Intersection over union of two time ranges."""
    inter = overlap_seconds(a_start, a_end, b_start, b_end)
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def union_seconds(spans: list[tuple[float, float]]) -> float:
    """Total wall-clock time covered by ``spans``, counting overlaps once.

    Summing durations would double-count the overlapping decodes that chunked
    transcription produces, and the 2026-08 comparison against the old
    tool_subtitle output was misleading for exactly that reason.
    """
    if not spans:
        return 0.0
    ordered = sorted(spans)
    total = 0.0
    cur_start, cur_end = ordered[0]
    for start, end in ordered[1:]:
        if start > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    total += cur_end - cur_start
    return total


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile; ``q`` in [0, 1]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1 - frac) + ordered[high] * frac
