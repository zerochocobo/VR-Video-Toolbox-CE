"""Align two transcriptions of the same audio.

WhisperJAV matches greedily in ground-truth order, which lets an early cue take
a partner that a later one needed more. Here every candidate pair is scored
first and the best pair wins globally, so the result does not depend on which
side is called "reference".

The score mixes text similarity with time overlap. Both are needed: on this
material the same sentence is often decoded twice with slightly different
wording, and different sentences are often near-identical text ("イク" repeated
across a scene), so neither signal alone separates them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from scripts.asr_bench import metrics
from scripts.asr_bench.profile import Cue

# A pair must overlap in time at all, and be at least vaguely the same words,
# before it can be considered the same line at all.
MIN_OVERLAP_SECONDS = 0.10
MIN_SIMILARITY = 0.20
TEXT_WEIGHT = 0.7


@dataclass
class Pair:
    a: Cue
    b: Cue
    similarity: float
    iou: float

    @property
    def cer(self) -> float:
        return metrics.cer(self.b.text, self.a.text)

    @property
    def start_shift(self) -> float:
        return self.b.start - self.a.start

    @property
    def end_shift(self) -> float:
        return self.b.end - self.a.end


@dataclass
class Alignment:
    pairs: list[Pair] = field(default_factory=list)
    only_a: list[Cue] = field(default_factory=list)
    only_b: list[Cue] = field(default_factory=list)

    @property
    def matched(self) -> int:
        return len(self.pairs)


def align(a_cues: list[Cue], b_cues: list[Cue],
          min_overlap: float = MIN_OVERLAP_SECONDS,
          min_similarity: float = MIN_SIMILARITY) -> Alignment:
    """Pair cues from ``a`` and ``b``, best-scoring pair first."""
    candidates: list[tuple[float, int, int, float, float]] = []
    for i, a in enumerate(a_cues):
        for j, b in enumerate(b_cues):
            if b.start >= a.end or b.end <= a.start:
                continue
            if metrics.overlap_seconds(a.start, a.end, b.start, b.end) < min_overlap:
                continue
            sim = metrics.similarity(a.text, b.text)
            if sim < min_similarity:
                continue
            overlap_iou = metrics.iou(a.start, a.end, b.start, b.end)
            score = TEXT_WEIGHT * sim + (1.0 - TEXT_WEIGHT) * overlap_iou
            candidates.append((score, i, j, sim, overlap_iou))

    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
    used_a: set[int] = set()
    used_b: set[int] = set()
    result = Alignment()
    for _score, i, j, sim, overlap_iou in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        result.pairs.append(Pair(a_cues[i], b_cues[j], sim, overlap_iou))

    result.pairs.sort(key=lambda p: p.a.start)
    result.only_a = [c for i, c in enumerate(a_cues) if i not in used_a]
    result.only_b = [c for j, c in enumerate(b_cues) if j not in used_b]
    return result


@dataclass
class Comparison:
    """A/B differences between two runs over the same audio."""

    name: str = ""
    a_lines: int = 0
    b_lines: int = 0
    matched: int = 0
    lost: int = 0                # in A, gone from B
    gained: int = 0              # new in B
    text_changed: int = 0        # matched but the words differ
    mean_cer: float = 0.0        # over matched pairs, B against A
    median_start_shift: float = 0.0
    median_end_shift: float = 0.0
    mean_iou: float = 0.0
    lost_seconds: float = 0.0
    gained_seconds: float = 0.0
    examples: dict[str, list[str]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "a_lines": self.a_lines,
            "b_lines": self.b_lines,
            "matched": self.matched,
            "lost": self.lost,
            "gained": self.gained,
            "text_changed": self.text_changed,
            "mean_cer": round(self.mean_cer, 4),
            "median_start_shift": round(self.median_start_shift, 3),
            "median_end_shift": round(self.median_end_shift, 3),
            "mean_iou": round(self.mean_iou, 4),
            "lost_seconds": round(self.lost_seconds, 1),
            "gained_seconds": round(self.gained_seconds, 1),
        }


def compare(a_cues: list[Cue], b_cues: list[Cue], name: str = "",
            example_limit: int = 6) -> Comparison:
    alignment = align(a_cues, b_cues)
    result = Comparison(
        name=name,
        a_lines=len(a_cues),
        b_lines=len(b_cues),
        matched=alignment.matched,
        lost=len(alignment.only_a),
        gained=len(alignment.only_b),
    )
    if alignment.pairs:
        cers = [p.cer for p in alignment.pairs]
        result.text_changed = sum(1 for c in cers if c > 0.0)
        result.mean_cer = sum(cers) / len(cers)
        result.median_start_shift = metrics.percentile([p.start_shift for p in alignment.pairs], 0.5)
        result.median_end_shift = metrics.percentile([p.end_shift for p in alignment.pairs], 0.5)
        result.mean_iou = sum(p.iou for p in alignment.pairs) / len(alignment.pairs)

    result.lost_seconds = metrics.union_seconds(
        [(c.start, c.end) for c in alignment.only_a])
    result.gained_seconds = metrics.union_seconds(
        [(c.start, c.end) for c in alignment.only_b])

    result.examples = {
        "lost": [f"{c.start:7.2f}-{c.end:7.2f}  {c.text[:40]}"
                 for c in alignment.only_a[:example_limit]],
        "gained": [f"{c.start:7.2f}-{c.end:7.2f}  {c.text[:40]}"
                   for c in alignment.only_b[:example_limit]],
        "rewritten": [
            f"{p.a.start:7.2f}  {p.a.text[:28]}  ->  {p.b.text[:28]}"
            for p in sorted(alignment.pairs, key=lambda p: -p.cer)[:example_limit]
            if p.cer > 0.0
        ],
    }
    return result


@dataclass
class Score:
    """Accuracy against a human-verified reference."""

    name: str = ""
    gt_lines: int = 0
    hyp_lines: int = 0
    matched: int = 0
    missed: int = 0
    hallucinated: int = 0
    cer: float = 0.0             # aggregate, over concatenated matched text
    mean_iou: float = 0.0

    def as_dict(self) -> dict:
        recall = self.matched / self.gt_lines if self.gt_lines else 0.0
        precision = self.matched / self.hyp_lines if self.hyp_lines else 0.0
        return {
            "name": self.name,
            "gt_lines": self.gt_lines,
            "hyp_lines": self.hyp_lines,
            "matched": self.matched,
            "missed": self.missed,
            "hallucinated": self.hallucinated,
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "cer": round(self.cer, 4),
            "mean_iou": round(self.mean_iou, 4),
        }


def score(gt_cues: list[Cue], hyp_cues: list[Cue], name: str = "") -> Score:
    """Recall / precision / CER / IoU of ``hyp`` against a real reference."""
    alignment = align(gt_cues, hyp_cues)
    result = Score(
        name=name,
        gt_lines=len(gt_cues),
        hyp_lines=len(hyp_cues),
        matched=alignment.matched,
        missed=len(alignment.only_a),
        hallucinated=len(alignment.only_b),
    )
    if alignment.pairs:
        # Aggregate CER over the concatenation, so a long line weighs more than
        # a two-character one instead of every line counting the same.
        reference = "".join(p.a.text for p in alignment.pairs)
        hypothesis = "".join(p.b.text for p in alignment.pairs)
        result.cer = metrics.cer(hypothesis, reference)
        result.mean_iou = sum(p.iou for p in alignment.pairs) / len(alignment.pairs)
    return result
