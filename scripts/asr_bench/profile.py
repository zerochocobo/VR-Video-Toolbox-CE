"""Reference-free profile of one transcription.

Every number here is one a human counted by hand during the 2026-08 HNVR-174
investigation. Together they separate a run that recovered quiet dialogue from
one that merely let more breathing reach the decoder:

``stock_phrase``
    Lines matching Whisper's known filler ("ご視聴ありがとうございました"
    and friends). Pure invention -- in the dub these are spoken aloud.
``nonverbal`` / ``lone_token``
    Moans, breaths and truncated single tokens, per
    :mod:`tool_subtitle.nonverbal`.
``fragment``
    Three characters or fewer after normalisation. Not necessarily wrong, but
    the ratio jumped from 3-13% to 31-46% between the "high" and "max" VAD
    presets, which is what makes it a useful pressure gauge.
``glued_fragment``
    A fragment whose neighbour starts within a third of a second: the two are
    almost always one word the chunked decode tore in half (``ごめ`` + ``ん``,
    ``ご`` + ``ちそう``). Counting these apart matters because they must be
    rejoined, not deleted -- a moan filter applied to them removes real speech.
``evidence_lines`` / ``evidence_chars``
    Lines carrying an actual sign of language -- kanji, a particle, a verb
    ending, a known word -- and how many characters those lines hold. This is
    the closest thing to a quality score available without ground truth: a
    setting that recovers more real speech raises it, while one that merely
    tears the same speech into more pieces does not, because the torn halves
    lose their grammar. Compare it alongside the fragment count, never alone.
``kanji_chars``
    Kanji among the recognised characters. A *weak* correctness hint, not a
    quality score: Japanese writes content words in kanji, and a recogniser
    that is guessing tends to fall back to kana ("キリにしてあげるね" where the
    line is "綺麗にしてあげるね"). It moves when a model gets words right that
    another got wrong -- which ``evidence_chars`` cannot see, because both
    spellings are the same number of characters. Never read it alone.
``sparse``
    At least three seconds long and under one character per second. A word the
    DTW aligner smeared across a pause (``わ`` stamped over 9.3 seconds); in
    the dub one syllable gets stretched over the whole span.
``repetition``
    Looped text the decoder emitted while stuck.
``long_line``
    Over 8 seconds. For dubbing this is a defect on its own: a line spanning
    two utterances gets one cloned take stretched across both.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.asr_bench import metrics  # noqa: E402
from tool_subtitle import nonverbal  # noqa: E402

# Imported lazily inside the functions that need them: tool_subtitle.logic pulls
# in numpy and utils.app_config, which is more than a pure text profile needs.

FRAGMENT_MAX_CHARS = 3
LONG_LINE_SECONDS = 8.0
# A gap this small between a fragment and its neighbour means the decode was
# torn, not that two things were said.
GLUE_GAP_SECONDS = 0.35
SPARSE_MIN_SECONDS = 3.0
SPARSE_MAX_CPS = 1.0


@dataclass
class Cue:
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Profile:
    name: str = ""
    lines: int = 0
    speech_seconds: float = 0.0          # union, overlaps counted once
    summed_seconds: float = 0.0          # naive sum, for spotting overlap
    audio_seconds: float | None = None

    stock_phrase: int = 0
    nonverbal: int = 0
    lone_token: int = 0
    fragment: int = 0
    repetition: int = 0
    long_line: int = 0
    overlapping: int = 0
    glued_fragment: int = 0
    sparse: int = 0
    evidence_lines: int = 0
    evidence_chars: int = 0
    kanji_chars: int = 0

    duration_p50: float = 0.0
    duration_p90: float = 0.0
    chars_p50: float = 0.0
    cps_p50: float = 0.0                 # characters per second
    cps_p10: float = 0.0                 # slow lines: breathing tracked as words

    examples: dict[str, list[str]] = field(default_factory=dict)

    @property
    def coverage(self) -> float | None:
        if not self.audio_seconds:
            return None
        return self.speech_seconds / self.audio_seconds

    def rate(self, attribute: str) -> float:
        return getattr(self, attribute) / self.lines if self.lines else 0.0

    def as_dict(self) -> dict:
        data = {
            "name": self.name,
            "lines": self.lines,
            "speech_seconds": round(self.speech_seconds, 1),
            "summed_seconds": round(self.summed_seconds, 1),
            "audio_seconds": round(self.audio_seconds, 1) if self.audio_seconds else None,
            "coverage": round(self.coverage, 4) if self.coverage is not None else None,
            "duration_p50": round(self.duration_p50, 2),
            "duration_p90": round(self.duration_p90, 2),
            "chars_p50": round(self.chars_p50, 1),
            "cps_p50": round(self.cps_p50, 2),
            "cps_p10": round(self.cps_p10, 2),
            "evidence_lines": self.evidence_lines,
            "evidence_chars": self.evidence_chars,
            "kanji_chars": self.kanji_chars,
        }
        for key in ("stock_phrase", "nonverbal", "lone_token", "fragment",
                    "glued_fragment", "sparse", "repetition", "long_line",
                    "overlapping"):
            data[key] = getattr(self, key)
            data[f"{key}_rate"] = round(self.rate(key), 4)
        return data


def read_srt(path: str | Path) -> list[Cue]:
    """Parse an SRT into cues, reusing the parser the clone tool ships."""
    from tool_clonevoice_v2.proofread import parse_srt_seconds

    return [Cue(c["start"], c["end"], c["text"]) for c in parse_srt_seconds(path)]


def _stock_phrase_norms() -> set[str]:
    from tool_subtitle import logic as subl

    return set(subl.HARD_HALLUCINATION_NORMS) | set(subl.SHORT_HALLUCINATION_NORMS)


def is_stock_phrase(norm: str, norms: set[str]) -> bool:
    """Whether a normalised line is (or is most of) a known Whisper filler.

    Substring matching in both directions, because the decoder truncates these
    as often as it emits them whole -- "視聴ありがとうございました" is the same
    invention as the full phrase, and "されました。" is its orphaned tail.
    """
    if not norm:
        return False
    if norm in norms:
        return True
    for phrase in norms:
        if len(norm) >= 6 and norm in phrase:
            return True
        if len(phrase) >= 6 and phrase in norm:
            return True
    return False


def audio_duration_seconds(path: str | Path) -> float | None:
    """Duration of a PCM wav without decoding it."""
    import wave

    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
        return frames / rate if rate else None
    except Exception:
        return None


def profile_cues(cues: list[Cue], name: str = "",
                 audio_seconds: float | None = None,
                 example_limit: int = 4) -> Profile:
    from tool_subtitle.logic import SubtitleGenerator as Gen

    norms = _stock_phrase_norms()
    result = Profile(name=name, lines=len(cues), audio_seconds=audio_seconds)
    durations: list[float] = []
    char_counts: list[float] = []
    rates: list[float] = []
    examples: dict[str, list[str]] = {}

    def remember(bucket: str, cue: Cue) -> None:
        seen = examples.setdefault(bucket, [])
        if len(seen) < example_limit:
            seen.append(f"{cue.start:7.2f}-{cue.end:7.2f}  {cue.text[:40]}")

    previous_end = None
    ordered = sorted(cues, key=lambda c: (c.start, c.end))
    for index, cue in enumerate(ordered):
        norm = metrics.normalize_text(cue.text)
        durations.append(cue.duration)
        char_counts.append(len(norm))
        if cue.duration > 0.05:
            rates.append(len(norm) / cue.duration)

        if previous_end is not None and cue.start < previous_end - 0.15:
            result.overlapping += 1
        previous_end = max(previous_end or 0.0, cue.end)

        if norm and nonverbal.has_language_evidence(cue.text):
            result.evidence_lines += 1
            result.evidence_chars += len(norm)
        result.kanji_chars += sum(1 for ch in norm if "一" <= ch <= "鿿")

        if is_stock_phrase(norm, norms):
            result.stock_phrase += 1
            remember("stock_phrase", cue)
        if nonverbal.is_lone_nonverbal_token(cue.text):
            result.lone_token += 1
            remember("lone_token", cue)
        elif nonverbal.is_nonverbal(cue.text):
            result.nonverbal += 1
            remember("nonverbal", cue)
        if len(norm) <= FRAGMENT_MAX_CHARS:
            result.fragment += 1
            remember("fragment", cue)
            before = ordered[index - 1] if index else None
            after = ordered[index + 1] if index + 1 < len(ordered) else None
            glued = (
                (before is not None and cue.start - before.end <= GLUE_GAP_SECONDS)
                or (after is not None and after.start - cue.end <= GLUE_GAP_SECONDS)
            )
            if glued:
                result.glued_fragment += 1
                neighbour = after if (after is not None
                                      and after.start - cue.end <= GLUE_GAP_SECONDS) else before
                remember("glued_fragment", Cue(
                    cue.start, cue.end,
                    f"{cue.text}  +  {neighbour.text[:24] if neighbour else ''}"))
        if cue.duration >= SPARSE_MIN_SECONDS and len(norm) / cue.duration < SPARSE_MAX_CPS:
            result.sparse += 1
            remember("sparse", cue)
        if Gen.is_repetition_noise(cue.text):
            result.repetition += 1
            remember("repetition", cue)
        if cue.duration > LONG_LINE_SECONDS:
            result.long_line += 1
            remember("long_line", cue)

    result.summed_seconds = sum(durations)
    result.speech_seconds = metrics.union_seconds([(c.start, c.end) for c in cues])
    result.duration_p50 = metrics.percentile(durations, 0.5)
    result.duration_p90 = metrics.percentile(durations, 0.9)
    result.chars_p50 = metrics.percentile(char_counts, 0.5)
    result.cps_p50 = metrics.percentile(rates, 0.5)
    result.cps_p10 = metrics.percentile(rates, 0.1)
    result.examples = examples
    return result


def profile_srt(path: str | Path, audio_path: str | Path | None = None,
                name: str = "") -> Profile:
    cues = read_srt(path)
    seconds = audio_duration_seconds(audio_path) if audio_path else None
    return profile_cues(cues, name=name or Path(path).name, audio_seconds=seconds)
