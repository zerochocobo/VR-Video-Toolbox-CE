"""High-recall transcription engine for tool_clonevoice.

This reuses tool_subtitle's proven ASR front-end — WhisperSeg (ONNX) VAD scene
splitting + auditok fallback, wide-intake faster-whisper options, kotoba
alignment-head repair, and the hallucination/repetition detectors — but adapted
for dubbing instead of subtitles:

  * language is configurable (tool_subtitle hard-codes Japanese);
  * per-word timestamps are captured and offset-corrected (needed for the
    duration-aligned voice clone), not discarded;
  * filtering preserves the *true acoustic* start/end of each line. We must NOT
    apply tool_subtitle's readability-oriented duration remap
    (``subtitle_duration_for_text``), which rewrites end times and would break
    lip-sync / fit-to-duration in the dub.

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
from tool_clonevoice import whisperx_backend as wx

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
        scene_mode = self.is_scene_split_enabled()
        asr_options = self.base_asr_options(scene_mode=scene_mode)
        # The duration-aligned dub needs per-word times; force them on regardless
        # of the inherited preset.
        asr_options["word_timestamps"] = True
        decode_kwargs = {k: v for k, v in asr_options.items() if k != "vad_parameters"}

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
        for index, chunk in enumerate(chunks, start=1):
            offset = chunk["offset_sec"]
            self.log_callback(
                f"[seg] chunk {index}/{total} at {offset:.2f}s "
                f"({chunk['duration_sec']:.2f}s)"
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

            for segment in segments:
                text = (segment.text or "").strip()
                if not text:
                    continue
                words = [
                    {"word": w.word, "start": float(w.start) + offset, "end": float(w.end) + offset}
                    for w in (segment.words or [])
                    if w.start is not None and w.end is not None
                ]
                seg_start = float(segment.start) + offset
                seg_end = float(segment.end) + offset
                # Re-anchor to word-level (DTW) times: on difficult audio the
                # decoder stamps a line at the chunk start seconds before it was
                # spoken, while word times stay on the audio. The dub's
                # fit-to-duration depends on these boundaries being acoustic.
                if words and words[-1]["end"] - words[0]["start"] >= 0.05:
                    if abs(words[0]["start"] - seg_start) > 0.5:
                        reanchored_far += 1
                    seg_start = words[0]["start"]
                    seg_end = words[-1]["end"]
                raw.append({
                    "start": seg_start,
                    "end": seg_end,
                    "text": text,
                    "words": words,
                    "avg_logprob": getattr(segment, "avg_logprob", None),
                    "no_speech_prob": getattr(segment, "no_speech_prob", None),
                })

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
        removed_noise = removed_hall = removed_dup = compressed = 0
        window = subl.DUPLICATE_LOOKBACK_SECONDS

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
            if Gen.is_known_hallucination(
                text, item["start"], item["end"], total_end,
                avg_logprob=item.get("avg_logprob"),
                no_speech_prob=item.get("no_speech_prob"),
            ):
                removed_hall += 1
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
            f"(removed {removed_dup} dup, {removed_noise} noise, {removed_hall} hallucination; "
            f"compressed {compressed} repetition lines)"
        )
        return kept


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
