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

from typing import Callable, Optional

from tool_subtitle import logic as subl
from tool_clonevoice_v2 import whisperx_backend as wx

LogCallback = Callable[[str], None]


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

    def _split_chunks(self, audio_file: str):
        """Scene-split, degrading WhisperSeg -> auditok -> fixed if a model/dep
        is missing (clonevoice may run without the WhisperSeg ONNX present)."""
        chunk_seconds = subl.CHUNK_SECONDS
        try:
            return self.split_audio_for_profile(audio_file, chunk_seconds, "stable")
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

    def transcribe_words(self, audio_file: str) -> dict:
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
        chunks = self._split_chunks(audio_file)
        total = len(chunks)
        self.log_callback(
            f"[seg] decoding {total} chunk(s) "
            f"(scene_split={subl.SCENE_SPLIT_METHOD}, model={self.model_preset}, "
            f"lang={self.language or 'auto'})"
        )

        raw: list[dict] = []
        detected_lang: Optional[str] = None
        reanchored_far = 0
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
            raw.extend(chunk_entries)

        if reanchored_far:
            self.log_callback(
                f"[seg] word-anchored timestamps moved {reanchored_far} segment(s) by more than 0.5s"
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
                better = (
                    len(norm) > len(prev_norm) + 2
                    or (
                        len(norm) >= len(prev_norm)
                        and item.get("avg_logprob", -99.0) > prev.get("avg_logprob", -99.0) + 0.15
                    )
                )
                if better:
                    kept[dup_index] = item
                removed_dup += 1
                continue

            kept.append(item)

        self.log_callback(
            f"[seg] kept {len(kept)} lines "
            f"(removed {removed_dup} dup, {removed_noise} noise, {removed_hall} hallucination, "
            f"{removed_acoustic} acoustic; compressed {compressed} repetition lines)"
            + (f" [acoustic by reason: "
               + ", ".join(f"{k} x{v}" for k, v in sorted(acoustic_shown.items())) + "]"
               if acoustic_shown else "")
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
JAPANESE_ONLY_MODELS = ("kotoba",)
FALLBACK_MODEL_KEY = "large-v3"


def resolve_model_for_language(model_key: str, language: Optional[str],
                               models_root: str, log: LogCallback) -> str:
    """Keep a Japanese-only ASR model off a non-Japanese source.

    kotoba is the default here because large-v3 invents stock phrases on this
    material rather than transcribing it -- measured on hnvr-174 part1, where
    large-v3 rendered the opening monologue as "ご視聴ありがとうございました"
    and produced 7 hallucinations in a 200s bedroom scene that kotoba
    transcribed with none. It is a Japanese-only fine-tune, so any other source
    language has to fall back.
    """
    if model_key not in JAPANESE_ONLY_MODELS:
        return model_key
    if language and not str(language).lower().startswith("ja"):
        log(
            f"[seg] {model_key} is Japanese-only; source language is "
            f"{language!r}, using {FALLBACK_MODEL_KEY} instead"
        )
        return FALLBACK_MODEL_KEY
    if not wx.check_model_files(model_key, models_root):
        log(
            f"[seg] {model_key} model files are missing; "
            f"using {FALLBACK_MODEL_KEY} instead"
        )
        return FALLBACK_MODEL_KEY
    return model_key


def transcribe(
    audio16k_path: str,
    *,
    model_key: str,
    models_root: str,
    language: Optional[str],
    vad_sensitivity: str = "high",
    log: LogCallback = print,
    model_holder: Optional[list] = None,
) -> dict:
    """Transcribe a 16 kHz wav, returning ``{'segments': [...], 'language': str}``.

    Each segment is ``{start, end, text, words:[{word,start,end}], avg_logprob,
    no_speech_prob}`` with absolute (offset-corrected) timestamps, matching the
    shape ``run_transcribe_diarize`` already consumes from the old whisperx path.
    """
    model_key = resolve_model_for_language(model_key, language, models_root, log)
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
    return gen.transcribe_words(audio16k_path)
