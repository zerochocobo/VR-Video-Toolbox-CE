"""ASR benchmark harness for tool_clonevoice_v2 / tool_subtitle.

Every ASR tuning decision so far (VAD sensitivity, kotoba vs large-v3, the
hallucination rules) was made by listening to one title and counting by hand.
That is expensive and not reproducible. This package turns a transcription into
numbers so a change can be judged against a fixed corpus.

Two modes, because we have no human ground truth:

``profile``
    Reference-free statistics of a single SRT: how much speech it claims, how
    many lines are stock phrases, how many are pure moan/breath kana, how many
    are three-character fragments, the duration and speaking-rate spread. These
    are exactly the quantities the 2026-08 HNVR-174 investigation counted by
    hand, and they are what separates a "high" run from a "max" run.

``compare``
    A/B two transcriptions of the *same audio*: lines only in A (lost), lines
    only in B (new), and where they agree, how far the text and the timings
    drifted. This needs no ground truth, which is what makes it usable today --
    every ``.clone`` work directory kept both ``audio16k.wav`` and the
    ``source.srt`` it produced.

``score``
    Ground-truth mode (CER + timing IoU), for when a human-verified reference
    exists. The old ``tool_subtitle`` ``.jp.srt`` files are NOT ground truth:
    their durations were remapped for readability and their fragments merged,
    so they inflate speech seconds. Use them for coverage/density comparison
    only.
"""

__all__ = ["corpus", "matcher", "metrics", "profile", "report"]
