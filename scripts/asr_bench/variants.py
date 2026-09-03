"""Produce a ``bench.<tag>.srt`` for one benchmark item.

Two kinds of variant live here.

**Post-processing variants** replay production post-processing over a
transcription that already exists, so a change to the filters can be measured in
seconds instead of re-decoding 5.5 hours of audio on the GPU. They call the real
methods on the real class -- nothing is reimplemented -- and are therefore only
valid for changes that happen *after* decoding.

**Decode variants** re-run the ASR with different settings. Those need a GPU and
take as long as a normal transcription; they are what the framing and VAD
experiments will use.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.asr_bench import corpus, profile  # noqa: E402


def _write_srt(path: Path, segments: list[dict]) -> Path:
    from tool_clonevoice_v2.logic import _format_srt_ts

    lines = []
    for index, segment in enumerate(segments, start=1):
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        lines.append(
            f"{index}\n"
            f"{_format_srt_ts(segment['start'])} --> {_format_srt_ts(segment['end'])}\n"
            f"{text}\n"
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _bare_transcriber(log):
    """A CloneTranscriber with no model loaded.

    ``__init__`` would load a multi-gigabyte CTranslate2 model, which the
    post-processing methods never touch. Bypassing it keeps the harness honest:
    the code under test is the shipped code, not a copy of it.
    """
    from tool_clonevoice_v2.segment_engine import CloneTranscriber

    transcriber = object.__new__(CloneTranscriber)
    transcriber.log_callback = log
    return transcriber


def postfix(item: corpus.Item, log=print) -> list[dict]:
    """Re-run the torn-fragment repair and the non-verbal drop over the baseline.

    The baseline SRT is what ``_filter_keep_timing`` had already kept, so this
    measures exactly the two stages appended to its tail. Word lists are gone by
    the time a line reaches an SRT; the merge only uses them to decide what to
    carry forward, so their absence does not change which lines merge.
    """
    transcriber = _bare_transcriber(log)
    segments = [
        {"start": cue.start, "end": cue.end, "text": cue.text, "words": []}
        for cue in profile.read_srt(item.baseline)
    ]
    segments = transcriber._merge_torn_fragments(segments)
    segments = transcriber._drop_nonverbal(segments)
    return segments


def merge_only(item: corpus.Item, log=print) -> list[dict]:
    """The fragment repair alone, to separate its effect from the filter's."""
    transcriber = _bare_transcriber(log)
    segments = [
        {"start": cue.start, "end": cue.end, "text": cue.text, "words": []}
        for cue in profile.read_srt(item.baseline)
    ]
    return transcriber._merge_torn_fragments(segments)


def nonverbal_only(item: corpus.Item, log=print) -> list[dict]:
    """The non-verbal filter alone -- deliberately in the unsafe order.

    Running it without the fragment repair first is what a naive port of
    WhisperJAV's filter would do. Comparing this against ``postfix`` shows how
    many real lines that order costs.
    """
    transcriber = _bare_transcriber(log)
    segments = [
        {"start": cue.start, "end": cue.end, "text": cue.text, "words": []}
        for cue in profile.read_srt(item.baseline)
    ]
    return transcriber._drop_nonverbal(segments)


BUILDERS = {
    "postfix": postfix,
    "merge_only": merge_only,
    "nonverbal_only": nonverbal_only,
}


def build(tag: str, item: corpus.Item, log=print) -> Path:
    builder = BUILDERS.get(tag)
    if builder is None:
        raise SystemExit(f"unknown variant '{tag}'; have: {', '.join(sorted(BUILDERS))}")
    segments = builder(item, log=log)
    return _write_srt(item.variant(tag), segments)
