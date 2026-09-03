"""High-recall transcription engine for tool_clonevoice_v2.

This reuses tool_subtitle's proven ASR front-end — WhisperSeg (ONNX) VAD scene
splitting + auditok fallback, wide-intake faster-whisper options, kotoba
alignment-head repair, and the hallucination/repetition detectors — but adapted
for dubbing instead of subtitles:

  * language is configurable (tool_subtitle hard-codes Japanese, and its
    ``clean_text`` strips inner spaces — overridden here);
  * per-word timestamps are captured and offset-corrected (needed for the
    duration-aligned voice clone), not discarded;
  * decoding reuses ``collect_segment_entries`` + ``extend_entry_tails``, so
    the dub gets word anchoring, pause/silence-valley splitting, tail repair
    and per-line acoustic stats identical to the subtitle pipeline;
  * filtering preserves the *true acoustic* start/end of each line. We must NOT
    apply tool_subtitle's readability-oriented duration remap
    (``subtitle_duration_for_text``) or its fragment merging, which rewrite
    times for readability and would break lip-sync / fit-to-duration in the
    dub; the acoustic hallucination checks (``acoustic_removal_reason``) DO
    apply — an invented line would be synthesised out loud.

Why this fixes the "漏句" (dropped lines) seen with the old whisperx path: the
old path called ``faster_whisper.transcribe(vad_filter=True)`` once over the
whole file, using Silero's default 0.5 speech threshold and the default
``condition_on_previous_text=True``. Quiet / breathy speech fell below the VAD
gate and never reached the decoder. The tool_subtitle front-end gates on the
ASMR-tuned WhisperSeg model and decodes chunk-by-chunk with
``condition_on_previous_text=False`` and low no-speech thresholds, catching the
quiet lines.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from tool_subtitle import logic as subl
from tool_subtitle import nonverbal
from tool_clonevoice_v2 import whisperx_backend as wx

LogCallback = Callable[[str], None]

# Torn-fragment repair. Measured on the ten-part benchmark corpus (5.5 h,
# scripts/asr_bench): 22.7% of all lines are three characters or fewer, and 77
# of them sit within a third of a second of their neighbour. Those are not
# short utterances, they are one word the chunked decode cut in half --
# "ごめ" + "ん", "ご" + "ちそう", "なんかそ" + "わ" + "そわそわしてきちゃった".
# tool_subtitle's readability merge is still the wrong tool here (it rewrites
# durations and merges whole sentences), but leaving these apart is worse than
# either: the dub clones each half as its own line, and "わ" gets a 9.3-second
# take of its own.
TORN_GAP_SECONDS = 0.35
TORN_FRAGMENT_CHARS = 3
# A torn word does not reliably leave its halves next to each other: "ごめ" was
# stamped 230.09-230.15 (0.06s for two characters) and its "ん" only came back
# at 231.57, a gap of 1.42s that the normal threshold rejects. The wider limit
# applies only when both sides are fragments AND at least one of them would
# otherwise be deleted as non-verbal, because then rejoining them is the only
# outcome that is not a loss. Two fragments that both look like real speech
# ("はい" then "うん") are never merged across it.
TORN_ORPHAN_GAP_SECONDS = 1.6
TORN_MAX_CHARS = 30
TORN_MAX_DURATION_SECONDS = 12.0
# Overlapping chunks re-decode the boundary, so the halves share a syllable or
# two ("なんかそわ" + "そわそわして..."). Trim the repeat rather than speaking it.
TORN_MAX_OVERLAP_CHARS = 8

# Non-verbal removal policy. The classifier was tuned against WhisperJAV's own
# frame-level output, which does not tear words; ours does, so a one- or
# two-character line is as likely to be half of "ごめん" as it is to be a moan.
# There is not enough text there to tell, and the classifier answers "sound"
# by default because every kana in a torn Japanese word tends to belong to the
# sound alphabet. Below this length only the lone-token list and the stretched
# rule below may remove a line.
NONVERBAL_MIN_CHARS = 3
# A line covering seconds of audio at under a character per second is broken
# timing -- "す" stamped across 11.0s, "メ" across 16.5s. Length matters as much
# as rate, though: "みーちゃん" over 9.8s and "うまい" over 4.6s are the same
# shape and are real dialogue whose timestamps are simply wrong, and a short
# take sitting in a long slot is something the dub already handles. So the rule
# only fires on a line too short to carry meaning, or on one that is pure sound
# kana at any length.
STRETCHED_MIN_SECONDS = 3.0
STRETCHED_MAX_CHARS_PER_SECOND = 1.0
STRETCHED_MAX_CHARS = 2

# Repairing a smeared start rather than deleting the line. WhisperJAV's
# CpsStartRetimer handles the same defect by holding the end fixed and moving
# the start to ``end - len(text) / reading_speed``. The move is right; the
# estimate is not, for two reasons. Its 7 chars/second is a *reading* speed, so
# it retimes "ごめん" from 3.99s down to 0.43s -- but a drawn-out spoken ごめん
# really does run over a second, and placing the dub take at the very end of
# the slot would put it seconds late. And it has to guess from text length only
# because it runs on a finished SRT.
#
# We are still holding the audio at this point, so the onset can be measured
# instead of estimated. This is the mirror of ``extend_entry_tails``: that walks
# forward through voiced frames to repair an early end, this walks forward
# through silent ones to repair an early start. The end never moves, so no line
# can start after it ends and no new overlap is created.
START_TIGHTEN_MIN_SECONDS = 2.0
START_TIGHTEN_MAX_CHARS_PER_SECOND = 2.0
START_TIGHTEN_SILENCE_DB = -45.0
START_TIGHTEN_MIN_LEAD_SECONDS = 0.5
START_TIGHTEN_ONSET_PAD_SECONDS = 0.15
# Leave the line at least this long, so a take always has somewhere to sit.
START_TIGHTEN_MIN_DURATION_SECONDS = 0.6

# Decode window. Owned here rather than inherited from tool_subtitle so the dub
# can differ from the subtitle tool, but currently set to the same values --
# because the experiment that argued for changing them did not survive scrutiny.
#
# Shortening the window to 3s raised "characters of recognised language" by 25%
# on the benchmark corpus, and that number was wrong to act on. Counting only
# kanji tells the real story:
#
#                        lines   kanji   chars   kanji%   chars/line
#   28s (before)          2514    2711   15592    17.4%      6.2
#   28s + repairs + cap   2225    2726   15826    17.2%      7.1
#    3s + repairs + cap   3318    2732   19551    14.0%      5.9
#
# Kanji is flat across every window. Japanese writes content words in kanji, so
# 25% more characters carrying 0.2% more kanji is not 25% more speech -- it is
# the same speech cut into 50% more lines and spelled worse. Reading the output
# confirms it: one clean line at 28s,
#   "もしかして私に変なことするつもりだったんじゃないでしょうね"
# comes back at 3s as four,
#   "もしかして" / "私に変なことするつもりだった最高" / "この音変態" / "ケガラワシ"
# -- more characters, a wrong word, and 汚らわしい degraded to katakana. Whisper
# picks kanji from context, and a three-second window with
# condition_on_previous_text=False leaves it none.
#
# WhisperJAV pairs its 3s framing with a recogniser fine-tuned for short dialogue
# (anime-whisper). The framing and the model go together; taking one without the
# other is what went wrong here. See AnimeWhisperTranscriber.
CHUNK_SECONDS = 28.0
WHISPERSEG_MERGE_GAP_SECONDS = 2.0

# Off switches for the three repair stages added on 2026-09-02. They exist so
# the benchmark can reproduce the pre-2026-09-02 pipeline and attribute the
# end-to-end change to each stage; nothing in the app turns them off.
ENABLE_TORN_MERGE = True
ENABLE_NONVERBAL_DROP = True
ENABLE_START_TIGHTEN = True

SAMPLE_RATE_HZ = 16000

# anime-whisper's own window, separate from the kotoba path's. It is text-only,
# so it decides for itself whether a window is worth a sentence -- and on
# three-second windows it declined for 29% of them, writing only "…". Given more
# to work with it speaks, and recognises more content words than kotoba does
# (three-title subset, 82 min):
#
#              lines   kanji   chars/line
#    3s          529     986      9.0
#    8s          321    1088     17.8
#   15s          220    1185     27.6   <- here
#
# 27.6 characters per line is unusable for a dub on its own: one line is one
# cloned take, and fifteen seconds holds several utterances. That is what the
# forced aligner is for -- it puts word times back onto the text so the window
# can be cut into one line per utterance by the same splitting the kotoba path
# already uses.
ANIME_CHUNK_SECONDS = 15.0
ANIME_MERGE_GAP_SECONDS = 2.0


class CloneTranscriber(subl.SubtitleGenerator):
    """SubtitleGenerator specialised for dubbing.

    Inherits model loading (incl. kotoba ``alignment_heads`` repair), WhisperSeg
    VAD / auditok scene splitting, and the static hallucination detectors.
    """

    def __init__(self, model_path: str, model_preset: str, log_callback: LogCallback,
                 language: Optional[str], use_gpu: bool = True):
        super().__init__(model_path, model_preset, log_callback, use_gpu=use_gpu)
        # faster-whisper wants None (not "") for auto-detect.
        self.language = language or None

    @staticmethod
    def clean_text(text: str) -> str:
        # tool_subtitle's clean_text strips ALL spaces (Japanese-only); the dub
        # source language is configurable, so inner spaces must survive.
        return (text or "").strip()

    # --- scene splitting with graceful degradation ---------------------------

    def _split_chunks(self, audio_file: str, vad_audio_file: Optional[str] = None):
        """Scene-split, degrading WhisperSeg -> auditok -> fixed if a model/dep
        is missing (clonevoice may run without the WhisperSeg ONNX present).

        ``vad_audio_file`` points the speech detector at a cleaned copy while
        the decoder keeps the original; only the WhisperSeg path honours it, and
        the fallbacks are already the degraded case.
        """
        chunk_seconds = CHUNK_SECONDS
        try:
            return self.split_audio_for_profile(
                audio_file, chunk_seconds, "stable",
                vad_audio_path=vad_audio_file,
                merge_gap_seconds=WHISPERSEG_MERGE_GAP_SECONDS)
        except Exception as exc:  # WhisperSeg ONNX or onnxruntime unavailable
            self.log_callback(
                f"[seg] WhisperSeg split unavailable ({exc}); falling back to auditok"
            )
        try:
            return self.split_audio_auditok(audio_file, chunk_seconds)
        except Exception as exc:
            self.log_callback(f"[seg] auditok split unavailable ({exc}); using fixed chunks")
            return self.split_audio(audio_file, chunk_seconds)

    # --- chunked decode keeping word timestamps ------------------------------

    def transcribe_words(self, audio_file: str,
                         vad_audio_file: Optional[str] = None) -> dict:
        """Chunked decode through the shared collect pipeline.

        ``collect_segment_entries`` gives word anchoring, pause/silence-valley
        splitting and the peak/coverage stats; ``extend_entry_tails`` repairs
        the systematically-early DTW word ends. Both matter even more for
        dubbing than for subtitles: a merged multi-utterance line or a
        truncated tail breaks the duration-aligned voice clone.
        """
        scene_mode = self.is_scene_split_enabled()
        asr_options = self.base_asr_options(scene_mode=scene_mode)
        # The duration-aligned dub needs per-word times; force them on regardless
        # of the inherited preset.
        asr_options["word_timestamps"] = True
        decode_kwargs = {k: v for k, v in asr_options.items() if k != "vad_parameters"}

        # Stale probs from a previous file must not feed this file's speech
        # coverage checks; the whisperseg split path repopulates this.
        self.last_speech_probs = None
        chunks = self._split_chunks(audio_file, vad_audio_file)
        total = len(chunks)
        self.log_callback(
            f"[seg] decoding {total} chunk(s) "
            f"(scene_split={subl.SCENE_SPLIT_METHOD}, model={self.model_preset}, "
            f"lang={self.language or 'auto'})"
        )

        raw: list[dict] = []
        detected_lang: Optional[str] = None
        reanchored_far = 0
        tightened = 0
        # One line per chunk was 200 lines on a one-hour title, 58% of the whole
        # transcription log, and the GUI keeps only so many. Report roughly
        # twenty times instead, plus the first and last.
        progress_every = max(1, total // 20)
        for index, chunk in enumerate(chunks, start=1):
            offset = chunk["offset_sec"]
            if index == 1 or index == total or index % progress_every == 0:
                self.log_callback(
                    f"[seg] chunk {index}/{total} at {offset:.2f}s "
                    f"({100.0 * index / max(total, 1):.0f}%)"
                )
            segments, info = self.model.transcribe(
                chunk["array"],
                language=self.language,
                task="transcribe",
                vad_filter=scene_mode and subl.SCENE_INTERNAL_VAD,
                vad_parameters=(
                    asr_options["vad_parameters"]
                    if scene_mode and subl.SCENE_INTERNAL_VAD
                    else None
                ),
                **decode_kwargs,
            )
            if detected_lang is None:
                detected_lang = getattr(info, "language", None)

            chunk_entries: list[dict] = []
            for segment in segments:
                entries, moved_far = self.collect_segment_entries(
                    segment, offset, chunk_audio=chunk["array"]
                )
                reanchored_far += moved_far
                chunk_entries.extend(entries)
            self.extend_entry_tails(chunk_entries, chunk["array"], offset)
            tightened += self.tighten_entry_starts(
                chunk_entries, chunk["array"], offset)
            raw.extend(chunk_entries)

        if reanchored_far:
            self.log_callback(
                f"[seg] word-anchored timestamps moved {reanchored_far} segment(s) by more than 0.5s"
            )
        if tightened:
            self.log_callback(
                f"[seg] pulled {tightened} smeared start(s) forward to the audible onset"
            )
        segments = self._filter_keep_timing(raw)
        return {"segments": segments, "language": detected_lang or (self.language or "")}

    # --- filtering that preserves true timing & words ------------------------

    def _filter_keep_timing(self, raw_segments: list[dict]) -> list[dict]:
        """Drop hallucinations / repetition / near-duplicates from overlapping
        chunks, keeping each surviving line's real start/end and word list.

        Deliberately omits tool_subtitle's ``subtitle_duration_for_text`` remap
        and the start-clamping that rewrites end times — the dub relies on the
        acoustic boundaries reported by the decoder.
        """
        Gen = subl.SubtitleGenerator
        ordered = sorted(raw_segments, key=lambda s: (s["start"], s["end"]))
        total_end = max((s["end"] for s in ordered), default=0.0)

        kept: list[dict] = []
        removed_noise = removed_hall = removed_dup = removed_acoustic = compressed = 0
        window = subl.DUPLICATE_LOOKBACK_SECONDS
        # When a stock phrase is decoded across two entries, dropping the first
        # leaves the remainder behind as its own line -- on HNVR-174 the credit
        # "この動画の字幕は視聴者の方によって作成されました" lost its head here
        # and "されました。" became SRT #1. The tail cannot be judged on its own
        # (it is ordinary Japanese), only as the completion of what was just
        # removed, so remember that and check the next line against it.
        pending_tail: tuple[str, float] | None = None
        acoustic_shown: dict[str, int] = {}

        for item in ordered:
            text = item["text"]
            norm = Gen.normalize_for_duplicate(text)

            if not text or not subl.HAS_LINGUISTIC_CONTENT_RE.search(text):
                removed_noise += 1
                continue
            if Gen.is_repetition_noise(text):
                short = Gen.compress_repetition_text(text)
                if short != text and not Gen.is_repetition_noise(short):
                    # Real repeated dialogue (climax lines): keep a compressed
                    # rendition. The word list no longer matches the text, so
                    # drop it — timing keeps the segment's acoustic start/end.
                    item["text"] = short
                    item["words"] = []
                    text = short
                    norm = Gen.normalize_for_duplicate(text)
                    compressed += 1
                else:
                    removed_noise += 1
                    continue
            if pending_tail is not None:
                remainder, deadline = pending_tail
                pending_tail = None
                if norm and remainder.startswith(norm) and item["start"] <= deadline:
                    removed_hall += 1
                    self.log_callback(
                        f"[seg] dropped [stock-phrase tail] "
                        f"{item['start']:.2f}-{item['end']:.2f} {text[:30]}"
                    )
                    continue
            if Gen.is_known_hallucination(
                text, item["start"], item["end"], total_end,
                avg_logprob=item.get("avg_logprob"),
                no_speech_prob=item.get("no_speech_prob"),
            ):
                removed_hall += 1
                # If what we dropped was only the front of a stock phrase, the
                # rest of it is probably the next line.
                for phrase in subl.HARD_HALLUCINATION_NORMS:
                    if len(phrase) > len(norm) and phrase.startswith(norm):
                        pending_tail = (phrase[len(norm):], item["end"] + 3.0)
                        break
                continue
            # Acoustic hallucination checks (peak/coverage stats from
            # collect_segment_entries). Vital for dubbing: an invented line
            # would be synthesised out loud over breathing or silence.
            acoustic_reason = self.acoustic_removal_reason(item)
            if acoustic_reason:
                removed_acoustic += 1
                # 102 of these on a one-hour title. A few examples are worth
                # seeing; the rest are counted into the summary line below.
                acoustic_shown[acoustic_reason] = acoustic_shown.get(acoustic_reason, 0) + 1
                if acoustic_shown[acoustic_reason] <= 3:
                    self.log_callback(
                        f"[seg] dropped [{acoustic_reason}] "
                        f"{item['start']:.2f}-{item['end']:.2f} {text[:30]}"
                    )
                continue

            # Near-duplicate against recent kept lines (overlapping chunks repeat
            # the boundary speech). Keep whichever is longer / more confident.
            # Only pairs whose time ranges actually overlap can be duplicated
            # decodes of the same audio; similar text at disjoint times is real
            # repeated dialogue and must be kept.
            dup_index = None
            for j in range(len(kept) - 1, -1, -1):
                prev = kept[j]
                if item["start"] - prev["start"] > window:
                    break
                time_overlap = min(item["end"], prev["end"]) - max(item["start"], prev["start"])
                if time_overlap <= 0.15:
                    continue
                prev_norm = Gen.normalize_for_duplicate(prev["text"])
                if Gen.is_near_duplicate(norm, prev_norm):
                    dup_index = j
                    break
            if dup_index is not None:
                prev = kept[dup_index]
                prev_norm = Gen.normalize_for_duplicate(prev["text"])
                # A recogniser that reports no confidence at all stores the key
                # as None rather than leaving it out, so the default in get()
                # never fires -- fall back on the value, not on the key.
                item_logprob = item.get("avg_logprob")
                prev_logprob = prev.get("avg_logprob")
                better = (
                    len(norm) > len(prev_norm) + 2
                    or (
                        len(norm) >= len(prev_norm)
                        and item_logprob is not None and prev_logprob is not None
                        and item_logprob > prev_logprob + 0.15
                    )
                )
                if better:
                    kept[dup_index] = item
                removed_dup += 1
                continue

            kept.append(item)

        if ENABLE_TORN_MERGE:
            kept = self._merge_torn_fragments(kept)
        if ENABLE_NONVERBAL_DROP:
            kept = self._drop_nonverbal(kept)

        self.log_callback(
            f"[seg] kept {len(kept)} lines "
            f"(removed {removed_dup} dup, {removed_noise} noise, {removed_hall} hallucination, "
            f"{removed_acoustic} acoustic; compressed {compressed} repetition lines)"
            + (f" [acoustic by reason: "
               + ", ".join(f"{k} x{v}" for k, v in sorted(acoustic_shown.items())) + "]"
               if acoustic_shown else "")
        )
        return kept

    # --- smeared-start repair ------------------------------------------------

    def tighten_entry_starts(self, entries: list, audio, offset: float,
                             sampling_rate: int = 16000) -> int:
        """Pull a line's start forward to where its audio actually begins.

        Only lines whose duration is absurd for their text are examined -- a
        normal line's leading silence is its own onset and must be left alone.
        The end is never touched: the decoder is reliable about where speech
        stopped, and holding it fixed means a start can only ever move later,
        which cannot create an overlap with the line before.

        Returns how many entries moved.
        """
        import numpy as np

        if audio is None or not entries or not ENABLE_START_TIGHTEN:
            return 0
        Gen = subl.SubtitleGenerator
        frame = max(1, int(sampling_rate * 0.02))
        moved = 0
        for entry in entries:
            duration = entry["end"] - entry["start"]
            length = len(Gen.normalize_for_duplicate(entry["text"]))
            if duration < START_TIGHTEN_MIN_SECONDS or not length:
                continue
            if length / duration >= START_TIGHTEN_MAX_CHARS_PER_SECOND:
                continue

            lo = max(0, int((entry["start"] - offset) * sampling_rate))
            hi = min(len(audio), int((entry["end"] - offset) * sampling_rate))
            span = np.asarray(audio[lo:hi], dtype=np.float64)
            frames = len(span) // frame
            if frames < 2:
                continue
            rms = np.sqrt((span[:frames * frame].reshape(frames, frame) ** 2).mean(axis=1))
            voiced = np.nonzero(20 * np.log10(rms + 1e-9) >= START_TIGHTEN_SILENCE_DB)[0]
            if len(voiced) == 0:
                # Nothing above the floor anywhere in the span: this is not a
                # smeared start, it is a line over silence, and the acoustic
                # checks own that case.
                continue
            lead = voiced[0] * frame / sampling_rate
            if lead < START_TIGHTEN_MIN_LEAD_SECONDS:
                continue
            new_start = entry["start"] + lead - START_TIGHTEN_ONSET_PAD_SECONDS
            new_start = min(new_start, entry["end"] - START_TIGHTEN_MIN_DURATION_SECONDS)
            if new_start <= entry["start"]:
                continue
            entry["start"] = new_start
            # Word times that predate the new start describe silence.
            entry["words"] = [w for w in (entry.get("words") or [])
                              if w.get("start") is None or w["start"] >= new_start]
            moved += 1
        return moved

    # --- torn-fragment repair ------------------------------------------------

    @staticmethod
    def _join_fragment_text(head: str, tail: str) -> tuple[str, bool]:
        """Concatenate two halves of a torn word, dropping any shared syllables.

        Returns the joined text and whether anything was trimmed. Chunks overlap
        by ``CHUNK_OVERLAP_SECONDS``, so the boundary audio is decoded twice and
        the two halves can share their seam: naive concatenation turned
        ``なんかそわ`` + ``そわそわしてきちゃった`` into ``なんかそわそわそわ…``.
        """
        head = (head or "").rstrip()
        tail = (tail or "").lstrip()
        if not head:
            return tail, False
        if not tail:
            return head, False
        limit = min(len(head), len(tail), TORN_MAX_OVERLAP_CHARS)
        for size in range(limit, 0, -1):
            if head[-size:] == tail[:size]:
                return head + tail[size:], True
        # Latin scripts need the space back; CJK never wants one.
        separator = " " if (head[-1].isascii() and tail[0].isascii()) else ""
        return head + separator + tail, False

    def _merge_torn_fragments(self, lines: list[dict]) -> list[dict]:
        """Rejoin a word the decoder split across two entries.

        Narrower than tool_subtitle's ``merge_adjacent_fragments`` on purpose:
        one side must be a fragment. Two adjacent *complete* short sentences are
        two lines the dub should speak separately, and merging them would be the
        readability rewrite this pipeline deliberately avoids. Timing stays
        acoustic -- the merged line spans from the first start to the last end,
        no duration is invented.
        """
        Gen = subl.SubtitleGenerator
        merged: list[dict] = []
        joined = 0
        for item in lines:
            if merged:
                previous = merged[-1]
                previous_norm = Gen.normalize_for_duplicate(previous["text"])
                item_norm = Gen.normalize_for_duplicate(item["text"])
                text, trimmed = self._join_fragment_text(previous["text"], item["text"])
                both_fragments = max(len(previous_norm), len(item_norm)) <= TORN_FRAGMENT_CHARS
                doomed = bool(
                    self._nonverbal_reason(
                        previous["text"], previous["end"] - previous["start"], Gen)
                    or self._nonverbal_reason(
                        item["text"], item["end"] - item["start"], Gen)
                    # Below NONVERBAL_MIN_CHARS the filter abstains, but a
                    # sound-only one-character line is still exactly the half of
                    # a torn word this merge exists to rejoin.
                    or nonverbal.is_sound_only_line(previous["text"])
                    or nonverbal.is_sound_only_line(item["text"]))
                gap_limit = (TORN_ORPHAN_GAP_SECONDS if both_fragments and doomed
                             else TORN_GAP_SECONDS)
                if (
                    item["start"] - previous["end"] <= gap_limit
                    and min(len(previous_norm), len(item_norm)) <= TORN_FRAGMENT_CHARS
                    and not previous["text"].rstrip().endswith(subl.MERGE_SENTENCE_FINAL_CHARS)
                    and len(Gen.normalize_for_duplicate(text)) <= TORN_MAX_CHARS
                    and item["end"] - previous["start"] <= TORN_MAX_DURATION_SECONDS
                ):
                    previous["text"] = text
                    previous["end"] = max(previous["end"], item["end"])
                    # A trimmed seam means the words no longer spell the text;
                    # an empty list makes the dub fall back to the line's own
                    # acoustic span rather than to stale word times.
                    if trimmed:
                        previous["words"] = []
                    else:
                        previous["words"] = (previous.get("words") or []) + (item.get("words") or [])
                    joined += 1
                    continue
            merged.append(item)
        if joined:
            self.log_callback(f"[seg] rejoined {joined} torn word fragment(s)")
        return merged

    @staticmethod
    def _nonverbal_reason(text: str, duration: float, Gen) -> str | None:
        """Why this line should not be dubbed, or ``None`` to keep it."""
        norm = Gen.normalize_for_duplicate(text)
        if not norm:
            return None
        if nonverbal.is_lone_nonverbal_token(text):
            return "lone-token"
        if (duration >= STRETCHED_MIN_SECONDS
                and len(norm) / duration < STRETCHED_MAX_CHARS_PER_SECOND
                and (len(norm) <= STRETCHED_MAX_CHARS
                     or nonverbal.is_sound_only_line(text))):
            return "stretched"
        if len(norm) >= NONVERBAL_MIN_CHARS and nonverbal.is_nonverbal(text):
            return "nonverbal"
        return None

    def _drop_nonverbal(self, lines: list[dict]) -> list[dict]:
        """Remove lines that are pure moaning, breathing or a lone artifact token.

        Runs *after* the fragment repair, and that order is the whole point:
        before it, ``ごめ`` and ``ん`` both read as non-verbal (every kana in
        them belongs to the sound alphabet) and a real apology would be deleted.
        Rejoined into ``ごめん`` the pair carries dialogue evidence and stays.

        Only the dub needs this. A subtitle line reading ``あぁん`` is noise the
        eye skips; here it is handed to IndexTTS and spoken over the picture.
        """
        Gen = subl.SubtitleGenerator
        kept: list[dict] = []
        removed: dict[str, int] = {}
        shown = 0
        for item in lines:
            reason = self._nonverbal_reason(
                item["text"], max(0.0, item["end"] - item["start"]), Gen)
            if reason:
                removed[reason] = removed.get(reason, 0) + 1
                if shown < 4:
                    shown += 1
                    self.log_callback(
                        f"[seg] dropped [{reason}] "
                        f"{item['start']:.2f}-{item['end']:.2f} {item['text'][:30]}"
                    )
                continue
            kept.append(item)
        if removed:
            self.log_callback(
                "[seg] dropped non-verbal lines: "
                + ", ".join(f"{k} x{v}" for k, v in sorted(removed.items()))
            )
        return kept


# "max" removes the RMS gate entirely, which on hnvr-174 let 2502s of 3467s
# reach the decoder against 2007s at "high" -- 44 gated regions against none --
# and recovered real dialogue that had been lost. It also feeds the decoder
# every breath, and whisper answers a breath with a stock phrase. Counted over
# the transcribed titles: the ones run at "max" carry 12-15% stock-phrase lines
# and 31-46% fragments of three characters or fewer, against 0-5% and 3-13% at
# "high". A missing line is silence; a hallucinated one is spoken aloud in the
# dub, so "high" is the better default and "max" stays available for a title
# whose quiet speech is genuinely being lost.
JAPANESE_ONLY_MODELS = ("kotoba", "anime-whisper")
# What to use when the chosen model cannot be used. For a Japanese source the
# answer is another Japanese model, not the multilingual one: large-v3 invents
# stock phrases on this material rather than transcribing it -- measured on
# hnvr-174 part1, where it rendered the opening monologue as
# "ご視聴ありがとうございました" and produced 7 hallucinations in a 200s bedroom
# scene that kotoba transcribed with none. Tried in order, first present wins,
# best first -- so a missing kotoba lands on anime-whisper rather than on the
# model we measured to be the worst of the three.
JAPANESE_FALLBACKS = ("anime-whisper", "kotoba", "large-v3")
FALLBACK_MODEL_KEY = "large-v3"


def _model_present(model_key: str, models_root: str) -> bool:
    return wx.check_model_files(model_key, models_root)


def resolve_model_for_language(model_key: str, language: Optional[str],
                               models_root: str, log: LogCallback) -> str:
    """Pick a model that is both right for the language and actually on disk.

    anime-whisper reads this material best, but it is a 2.9 GB optional
    download and not what the dropdowns start on -- so an install that has not
    fetched it must land somewhere sensible rather than fail, whichever way the
    user reached it. The order matters: a Japanese source falls back
    to kotoba before large-v3, because large-v3 is worse on this material than
    either Japanese model, not merely different.
    """
    if model_key not in JAPANESE_ONLY_MODELS:
        return model_key
    if language and not str(language).lower().startswith("ja"):
        log(
            f"[seg] {model_key} is Japanese-only; source language is "
            f"{language!r}, using {FALLBACK_MODEL_KEY} instead"
        )
        return FALLBACK_MODEL_KEY
    if _model_present(model_key, models_root):
        return model_key
    for candidate in JAPANESE_FALLBACKS:
        if candidate != model_key and _model_present(candidate, models_root):
            log(
                f"[seg] {model_key} model files are missing; "
                f"using {candidate} instead"
            )
            return candidate
    log(f"[seg] {model_key} model files are missing and no fallback is "
        f"installed; trying {FALLBACK_MODEL_KEY}")
    return FALLBACK_MODEL_KEY


def transcribe(
    audio16k_path: str,
    *,
    model_key: str,
    models_root: str,
    language: Optional[str],
    vad_sensitivity: str = "high",
    log: LogCallback = print,
    model_holder: Optional[list] = None,
    vad_audio_path: Optional[str] = None,
) -> dict:
    """Transcribe a 16 kHz wav, returning ``{'segments': [...], 'language': str}``.

    ``vad_audio_path`` is an optional cleaned copy of the same audio, used only
    to find where the speech is; the decoder always hears ``audio16k_path``.

    Each segment is ``{start, end, text, words:[{word,start,end}], avg_logprob,
    no_speech_prob}`` with absolute (offset-corrected) timestamps, matching the
    shape ``run_transcribe_diarize`` already consumes from the old whisperx path.
    """
    model_key = resolve_model_for_language(model_key, language, models_root, log)
    if model_key == "anime-whisper":
        from tool_clonevoice_v2 import anime_whisper_backend as aw

        gen = AnimeWhisperTranscriber(
            str(aw.model_dir(models_root)), log, language,
            use_gpu=wx.resolve_device()[0] == "cuda",
        )
        gen.set_vad_sensitivity(vad_sensitivity)
        return gen.transcribe_words(audio16k_path, vad_audio_path)

    model_path = str(wx.model_dir(model_key, models_root))
    # Honour clonevoice's CTranslate2 cuDNN probe: trying CUDA when the matching
    # cuDNN DLLs are absent hard-crashes the process (0xc0000409), which the base
    # class's try/except cannot catch. Force CPU in that case.
    asr_device, _ = wx.resolve_asr_device()
    gen = CloneTranscriber(
        model_path, model_key, log, language, use_gpu=(asr_device == "cuda")
    )
    gen.set_vad_sensitivity(vad_sensitivity)
    if model_holder is not None:
        model_holder.append(gen.model)
    return gen.transcribe_words(audio16k_path, vad_audio_path)


class AnimeWhisperTranscriber(CloneTranscriber):
    """Framing from the VAD, words from anime-whisper.

    The decoupled arrangement WhisperJAV calls ChronosJAV: the speech segmenter
    decides where the lines are, and a text-only recogniser fills them in. It
    only became practical here once the window came down to 3 seconds -- at 28
    seconds a window held several utterances and a single text answer for it
    would have been unusable.

    What it gives up is the decoder's own segmentation and its per-word times.
    Neither is a loss the dub feels: the line boundaries it replaces them with
    are measured from the audio rather than inferred by DTW, which is what
    ``extend_entry_tails`` and ``split_group_on_silence`` exist to approximate.
    What it gives up that does matter is ``avg_logprob`` / ``no_speech_prob``,
    so two of the acoustic hallucination checks lose their confidence term and
    fall back to their energy and coverage terms alone.

    **Experimental -- not the default.** Measured on the three-title subset
    (82 min) against the shipped kotoba path:

        kotoba 28s        664 lines   1049 kanji    8.0 chars/line
        anime-whisper     318 lines   1185 kanji   18.8 chars/line

    It recognises 13% more content words, and reads better where it is whole
    ("同時ないように" -> "動じないように"). Two things stop it being the
    default. It writes nothing at all for a large share of windows, so it puts
    those words into half as many lines. And dividing a window's text across
    the VAD's speech regions is done by proportion, which cuts words at the
    boundaries ("平常心" came back as "常心") -- the dub would speak the
    fragment. Doing that division properly needs real forced alignment, and the
    wav2vec2 aligner we ship collapses on this material: on one window it
    stamped the first character across 8.46s and gave the remaining eighteen
    20ms each -- so that route was dropped rather than shipped, and the
    aligner model with it.
    """

    # Windows per forward pass. Whisper pads every input to 30s whatever its
    # real length, so batching is the whole of the speed.
    BATCH_WINDOWS = 8

    def __init__(self, model_path: str, log_callback: LogCallback,
                 language: Optional[str], use_gpu: bool = True):
        # Deliberately not calling super().__init__: it loads a CTranslate2
        # model this path never decodes with. Everything the inherited splitting
        # and filtering touches is set up here instead.
        from tool_clonevoice_v2 import anime_whisper_backend as aw

        self.model_preset = "anime-whisper"
        self.log_callback = log_callback
        self.models_root = str(Path(model_path).parent)
        self.language = language or "ja"
        self.model = None
        self.device = "cuda" if use_gpu else "cpu"
        self.whisperseg_session = None
        self.whisperseg_feature_extractor = None
        self.last_raw_segments = []
        self.last_chunks = []
        self.last_speech_probs = None
        self.set_vad_sensitivity("standard")
        self.generator = aw.AnimeWhisperGenerator(model_path, log_callback, use_gpu)

    @staticmethod
    def voiced_extent(audio, floor_db: float = START_TIGHTEN_SILENCE_DB,
                      sampling_rate: int = SAMPLE_RATE_HZ):
        """First and last audible moment in ``audio``, in seconds, or None.

        A window is padded out to ``WHISPERSEG_MIN_CHUNK_SECONDS`` when the
        speech in it is shorter than that, so the window's own bounds are not
        the line's bounds. The dub needs the real ones -- they become the slot
        the cloned take has to fit.
        """
        import numpy as np

        frame = max(1, int(sampling_rate * 0.02))
        span = np.asarray(audio, dtype=np.float64)
        frames = len(span) // frame
        if frames < 1:
            return None
        rms = np.sqrt((span[:frames * frame].reshape(frames, frame) ** 2).mean(axis=1))
        voiced = np.nonzero(20 * np.log10(rms + 1e-9) >= floor_db)[0]
        if len(voiced) == 0:
            return None
        return (voiced[0] * frame / sampling_rate,
                (voiced[-1] + 1) * frame / sampling_rate)

    def _split_chunks(self, audio_file: str, vad_audio_file: Optional[str] = None):
        """Longer windows than the kotoba path -- see ANIME_CHUNK_SECONDS."""
        try:
            return self.split_audio_for_profile(
                audio_file, ANIME_CHUNK_SECONDS, "stable",
                vad_audio_path=vad_audio_file,
                merge_gap_seconds=ANIME_MERGE_GAP_SECONDS)
        except Exception as exc:
            self.log_callback(
                f"[seg] WhisperSeg split unavailable ({exc}); falling back to auditok")
            return self.split_audio_auditok(audio_file, ANIME_CHUNK_SECONDS)

    # Sentence boundaries anime-whisper actually punctuates. kotoba emits almost
    # no punctuation, which is why this splitter belongs to this path only.
    SENTENCE_END = "。！？!?"

    @staticmethod
    def split_sentences(text: str) -> list:
        """Break a window's text after sentence-final punctuation."""
        pieces: list = []
        current = ""
        for ch in text:
            current += ch
            if ch in AnimeWhisperTranscriber.SENTENCE_END:
                pieces.append(current)
                current = ""
        if current.strip():
            pieces.append(current)
        return [p for p in (piece.strip() for piece in pieces) if p]

    def place_on_regions(self, text: str, chunk: dict) -> list:
        """Lay a window's text onto the speech regions inside that window.

        The recogniser gives text with no times; the VAD already said where the
        speech is. Matching the two is better founded than asking a forced
        aligner to guess, and on this material it has to be: wav2vec2 alignment
        of a whole window collapsed, stamping one character across 8.46s and
        giving the remaining eighteen 20ms each.

        Exactly one line per region, always. A region is one run of speech
        between two pauses, which is the same thing the dub wants a take to be,
        so line boundaries and take boundaries end up identical and no line ever
        spans a silence. Where the sentence count matches the region count each
        sentence keeps its own region; otherwise the text is divided between
        regions in proportion to how long each region lasts, and the division
        is nudged to the nearest sentence or clause break so a cut lands at
        punctuation rather than mid-word.

        Returns ``(start, end, text)`` triples in window-relative seconds.
        """
        regions = self._subdivide_regions(
            [r for r in (chunk.get("regions") or []) if r[1] > r[0]], chunk["array"])
        sentences = self.split_sentences(text)
        if not sentences:
            return []
        if not regions:
            extent = self.voiced_extent(chunk["array"])
            return [(extent[0], extent[1], text)] if extent else []
        if len(sentences) == len(regions):
            return [(start, end, sentence)
                    for (start, end), sentence in zip(regions, sentences)]

        joined = "".join(sentences)
        if len(regions) == 1:
            return [(regions[0][0], regions[0][1], joined)]

        # Divide by speech time, not by region count: a region twice as long
        # should carry roughly twice the text.
        spans = [end - start for start, end in regions]
        total_span = sum(spans)
        placed: list = []
        cursor = 0
        elapsed = 0.0
        pending_start = regions[0][0]
        for index, (start, end) in enumerate(regions):
            elapsed += spans[index]
            if index == len(regions) - 1:
                cut = len(joined)
            else:
                cut = self._nearest_break(
                    joined, int(round(elapsed / total_span * len(joined))), cursor)
                if cut is None:
                    # No punctuation anywhere near where the time says to
                    # divide, so any division here would fall inside a word --
                    # "平常心" came back as "平" plus "常心を保つ訓練". Carry this
                    # region into the next one instead and divide there. A line
                    # slightly too long is a take that runs on; half a word is a
                    # take that says nothing.
                    continue
            piece = joined[cursor:cut].strip()
            cursor = cut
            if piece:
                placed.append((pending_start, end, piece))
            pending_start = end
        return placed

    def _subdivide_regions(self, regions: list, audio) -> list:
        """Cut each speech region at any long silence inside it.

        The VAD only ends a region after 100ms of quiet, so one region can still
        hold two sentences with a breath between them. Measured on the subset,
        23.6% of the lines this splitter produced contained more than a second
        of silence -- and a line with a pause in the middle desynchronises the
        dub, because the cloned take is continuous: it speaks over the pause and
        then falls silent while the speaker is still talking. Splitting on the
        same energy runs ``split_group_on_silence`` uses puts the boundary where
        the speaker actually stopped.
        """
        if audio is None or not regions:
            return regions
        finer: list = []
        for start, end in regions:
            runs = self.find_silence_runs(audio, start, end)
            cursor = start
            for run_start, run_end in runs:
                if run_start - cursor > 0.15:
                    finer.append((cursor, run_start))
                cursor = run_end
            if end - cursor > 0.15:
                finer.append((cursor, end))
            elif not runs:
                finer.append((start, end))
        return finer or regions

    # Where a division may fall when it cannot fall on a sentence end.
    CLAUSE_BREAK = "、,・…"

    def _nearest_break(self, text: str, target: int, floor: int,
                       window: int = 8):
        """Closest punctuation to ``target``, or None if there is none nearby.

        Returning None rather than falling back to ``target`` is the whole
        point: a division that is not at punctuation is a division inside a
        word, and the dub would speak the fragment.
        """
        target = max(floor + 1, min(target, len(text)))
        breaks = self.SENTENCE_END + self.CLAUSE_BREAK
        for distance in range(window + 1):
            for candidate in (target - distance, target + distance):
                if floor < candidate <= len(text) and text[candidate - 1] in breaks:
                    return candidate
        return None

    def transcribe_words(self, audio_file: str,
                         vad_audio_file: Optional[str] = None) -> dict:
        self.last_speech_probs = None
        chunks = self._split_chunks(audio_file, vad_audio_file)
        total = len(chunks)
        self.log_callback(
            f"[seg] decoding {total} window(s) with anime-whisper "
            f"(window={ANIME_CHUNK_SECONDS:.1f}s)"
        )

        # Phase 1: text. The recogniser holds three gigabytes, so it is released
        # before the aligner is loaded rather than the two sharing the card.
        texts: list[str] = []
        progress_every = max(1, (total // self.BATCH_WINDOWS) // 10 or 1)
        try:
            for start in range(0, total, self.BATCH_WINDOWS):
                batch = chunks[start:start + self.BATCH_WINDOWS]
                texts.extend(
                    self.clean_text(text)
                    for text in self.generator.generate_batch(
                        [chunk["array"] for chunk in batch], batch_size=len(batch))
                )
                index = min(start + self.BATCH_WINDOWS, total)
                if index >= total or (start // self.BATCH_WINDOWS) % progress_every == 0:
                    self.log_callback(
                        f"[seg] window {index}/{total} ({100.0 * index / max(total, 1):.0f}%)")
        finally:
            self.generator.unload()

        spoken = sum(1 for text in texts if text)
        self.log_callback(
            f"[seg] anime-whisper wrote text for {spoken}/{total} window(s) "
            f"({100.0 * spoken / max(total, 1):.0f}%)")

        # Phase 2: put the sentences onto the speech regions the VAD found
        # inside each window, so one line is one utterance again.
        raw: list[dict] = []
        for text, chunk in zip(texts, chunks):
            if not text:
                continue
            offset = chunk["offset_sec"]
            entries = []
            for start, end, sentence in self.place_on_regions(text, chunk):
                entries.append({
                    "start": offset + start,
                    "end": offset + end,
                    "text": sentence,
                    "words": [],
                    # The bounds are the VAD's own speech region, measured from
                    # the audio -- a stronger anchor than word alignment, which
                    # is what "anchored" gates in the acoustic checks.
                    "anchored": True,
                    "peak_db": self.span_peak_db(chunk["array"], start, end),
                    "speech_coverage": self.span_speech_coverage(
                        offset + start, offset + end),
                    "avg_logprob": None,
                    "no_speech_prob": None,
                })
            self.extend_entry_tails(entries, chunk["array"], offset)
            self.tighten_entry_starts(entries, chunk["array"], offset)
            raw.extend(entries)

        segments = self._filter_keep_timing(raw)
        return {"segments": segments, "language": self.language}
