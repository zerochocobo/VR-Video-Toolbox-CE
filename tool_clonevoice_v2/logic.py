"""IndexTTS-2.5 sentence-level clone and translation orchestration.

The active single and batch flows share a speaker-free manifest:
    1. Transcribe into timestamped sentences.
    2. Optionally proofread the source and translate each sentence.
    3. Clone every translated sentence from its corresponding source WAV.
    4. Assemble the aligned ``<video>.si.wav`` timeline.

The guided single-video UI can pause before export for manual proofreading.
The batch entry point runs the same stages automatically without that pause.
"""
from __future__ import annotations

import gc
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Callable, Optional

from tool_clonevoice_v2 import diarize as diar
from tool_clonevoice_v2 import refsel
from tool_clonevoice_v2 import segment_engine as seg_engine
from tool_clonevoice_v2 import whisperx_backend as wx

LogCallback = Callable[[str], None]

MANIFEST_NAME = "manifest.json"
AUDIO16K_NAME = "audio16k.wav"
SOURCE_SRT_NAME = "source.srt"
TRANSLATED_SRT_NAME = "translated.srt"


def load_segments(path: str | Path) -> list[dict]:
    """Load standalone sentence JSON used by the v2 smoke/API entry point."""
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    segments = raw.get("segments", []) if isinstance(raw, dict) else raw
    if not isinstance(segments, list):
        raise ValueError("Segments JSON must be an array or an object containing 'segments'.")
    result = []
    for index, item in enumerate(segments, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Segment {index} is not an object.")
        text = str(item.get("text") or item.get("tgt_text") or "").strip()
        if not text:
            continue
        start, end = float(item["start"]), float(item["end"])
        if end <= start:
            raise ValueError(f"Segment {index} has invalid time range: {start}..{end}")
        result.append({"start": start, "end": end, "text": text})
    return result


# --- intermediate directory / manifest IO ---

def clone_dir(video_path: str | Path) -> Path:
    video = Path(video_path)
    return video.parent / (video.stem + ".clone")


def manifest_path(video_path: str | Path) -> Path:
    return clone_dir(video_path) / MANIFEST_NAME


def load_manifest(video_path: str | Path) -> Optional[dict]:
    path = manifest_path(video_path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_manifest(video_path: str | Path, manifest: dict) -> Path:
    cdir = clone_dir(video_path)
    cdir.mkdir(parents=True, exist_ok=True)
    path = cdir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def log_memory(label: str, log: LogCallback = print) -> None:
    """One line with this process's memory and the GPU's, at a stage boundary.

    Stage boundaries only: the level pass alone used to hold 666 MB for a
    one-hour video and nothing said so, which is the kind of thing you only
    find by looking. Silently does nothing if psutil is unavailable.
    """
    parts = []
    try:
        import psutil

        info = psutil.Process().memory_info()
        parts.append(f"RSS {info.rss / 1e9:.2f} GB")
        available = psutil.virtual_memory().available / 1e9
        parts.append(f"free {available:.1f} GB")
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            parts.append(
                f"VRAM {torch.cuda.memory_allocated() / 1e9:.2f}/"
                f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB peak"
            )
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    if parts:
        log(f"[mem] {label}: " + ", ".join(parts))


def cleanup_clone_dir(video_path: str | Path, *, log: LogCallback = print) -> bool:
    """Delete a video's ``.clone`` working directory once it is fully synthesized.

    Called when the batch UI's "keep intermediate files (debug)" box is left
    unchecked. Until now that box was threaded through every stage but never
    read, so nothing was ever removed.

    This takes the whole directory, not just the bulky per-sentence WAVs: the
    manifest and both SRTs go with it. A later run therefore re-transcribes and
    re-translates the video from scratch (the translation stage calls a paid
    API), and ``skip_existing`` has no checkpoint left to resume from. That is
    the intended meaning of the option.
    """
    cdir = clone_dir(video_path)
    # Guard against a malformed video path turning this into a wide delete.
    if cdir.suffix.lower() != ".clone":
        log(f"[cleanup] refusing to remove a non-.clone path: {cdir}")
        return False
    if not cdir.is_dir():
        return False
    try:
        shutil.rmtree(cdir)
    except OSError as exc:
        log(f"[cleanup] could not remove {cdir}: {exc}")
        return False
    log(f"[cleanup] removed intermediate directory: {cdir}")
    return True


def _load_resume_manifest(video_path: str | Path, log: LogCallback) -> Optional[dict]:
    """Load a checkpoint manifest, treating damaged JSON as incomplete work."""
    try:
        return load_manifest(video_path)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        log(f"[resume] manifest is unreadable; rebuilding this stage: {manifest_path(video_path)} ({exc})")
        return None


def _source_checkpoint_ready(manifest: Optional[dict]) -> bool:
    """Return whether a manifest contains reusable sentence transcription."""
    if not isinstance(manifest, dict):
        return False
    segments = manifest.get("segments")
    if not isinstance(segments, list) or not segments:
        return False
    return all(
        isinstance(item, dict)
        and "id" in item
        and "start" in item
        and "end" in item
        and "src_text" in item
        for item in segments
    )


def _translation_checkpoint_ready(manifest: Optional[dict], target_language: Optional[str]) -> bool:
    """Return whether every source sentence is translated or explicitly cleared."""
    if not _source_checkpoint_ready(manifest):
        return False
    assert manifest is not None
    requested = (target_language or "").strip().casefold()
    completed_target = str(manifest.get("target_language") or "").strip().casefold()
    if requested and requested != completed_target:
        return False
    proofread = manifest.get("proofread") if isinstance(manifest.get("proofread"), dict) else {}
    cleared_ids = {
        int(value)
        for value in (proofread.get("cleared_ids") or [])
        if str(value).lstrip("-").isdigit()
    }
    for item in manifest.get("segments", []):
        if not str(item.get("src_text") or "").strip():
            continue
        if str(item.get("tgt_text") or "").strip():
            continue
        if int(item.get("id", -1)) in cleared_ids:
            continue
        return False
    return True


def _translation_target_matches(manifest: dict, target_language: Optional[str]) -> bool:
    requested = (target_language or "").strip().casefold()
    completed = str(manifest.get("target_language") or "").strip().casefold()
    return not requested or not completed or requested == completed


def _translated_srt_has_cues(video_path: str | Path) -> bool:
    translated_srt = clone_dir(video_path) / TRANSLATED_SRT_NAME
    if not translated_srt.is_file():
        return False
    from tool_clonevoice_v2.proofread import parse_srt_seconds

    return bool(parse_srt_seconds(translated_srt))


def _restore_translations_from_srt(
    video_path: str | Path,
    manifest: dict,
    target_language: Optional[str],
    log: LogCallback,
) -> bool:
    """Restore ``tgt_text`` from an existing translated SRT checkpoint.

    The translated SRT is the user-visible result of a completed translation
    stage. Older/interrupted runs may have written it while leaving manifest
    target fields incomplete. Align its cues by time, persist them back into
    the manifest, and mark SRT-omitted source lines as intentionally cleared.
    """
    translated_srt = clone_dir(video_path) / TRANSLATED_SRT_NAME
    if not translated_srt.is_file() or not _translation_target_matches(manifest, target_language):
        return False

    from tool_clonevoice_v2.proofread import align_reference, parse_srt_seconds

    cues = parse_srt_seconds(translated_srt)
    if not cues:
        log(f"[resume] translated subtitles contain no usable cues: {translated_srt}")
        return False
    segments = manifest.get("segments", [])
    by_id = {str(item.get("id")): item for item in segments}
    restored_ids: set[int] = set()
    for row in align_reference(segments, cues):
        if row.get("kind") != "seg" or not str(row.get("ref_text") or "").strip():
            continue
        item = by_id.get(str(row.get("seg_id")))
        if item is None:
            continue
        item["tgt_text"] = str(row["ref_text"]).strip()
        try:
            restored_ids.add(int(item.get("id")))
        except Exception:
            pass

    if not restored_ids:
        log(f"[resume] translated subtitles could not be aligned to manifest: {translated_srt}")
        return False

    proofread = manifest.get("proofread") if isinstance(manifest.get("proofread"), dict) else {}
    proofread = dict(proofread)
    cleared_ids = {
        int(value)
        for value in (proofread.get("cleared_ids") or [])
        if str(value).lstrip("-").isdigit()
    }
    for item in segments:
        if not str(item.get("src_text") or "").strip() or str(item.get("tgt_text") or "").strip():
            continue
        try:
            cleared_ids.add(int(item.get("id")))
        except Exception:
            continue
    proofread["cleared_ids"] = sorted(cleared_ids)
    manifest["proofread"] = proofread
    if target_language:
        manifest["target_language"] = target_language
    save_manifest(video_path, manifest)
    log(
        f"[resume] restored {len(restored_ids)} translated line(s) from SRT; "
        f"skipping source proofreading and translation API: {translated_srt}"
    )
    return _translation_checkpoint_ready(manifest, target_language)


def translation_checkpoint_ready(video_path: str | Path, target_language: Optional[str]) -> bool:
    """Public read-only checkpoint probe used by the batch UI preflight."""
    try:
        manifest = load_manifest(video_path)
        if _translation_checkpoint_ready(manifest, target_language):
            return True
        return bool(
            isinstance(manifest, dict)
            and _source_checkpoint_ready(manifest)
            and _translation_target_matches(manifest, target_language)
            and _translated_srt_has_cues(video_path)
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return False


def transcription_checkpoint_ready(video_path: str | Path) -> bool:
    """Return whether ASR can be skipped for this video's existing manifest."""
    try:
        return _source_checkpoint_ready(load_manifest(video_path))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return False


def _release_cuda_cache() -> None:
    try:
        import torch
    except Exception:
        return
    if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def release_model_holder(model_holder: Optional[list]) -> None:
    """Release native/torch models collected by a caller-controlled holder."""
    if model_holder is not None:
        model_holder.clear()
    gc.collect()
    _release_cuda_cache()


def _format_srt_ts(t: float) -> str:
    t = max(0.0, float(t))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    if ms >= 1000:
        ms -= 1000
        s += 1
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path: str | Path, segments: list, text_key: str, speaker_prefix: bool = False) -> Path:
    """Write a debug SRT from manifest segments using ``text_key`` for the body."""
    blocks = []
    idx = 0
    for s in segments:
        txt = (s.get(text_key) or "").strip()
        if not txt:
            continue
        idx += 1
        if speaker_prefix and s.get("speaker"):
            txt = f"[{s['speaker']}] {txt}"
        blocks.append(
            f"{idx}\n{_format_srt_ts(s['start'])} --> {_format_srt_ts(s['end'])}\n{txt}\n"
        )
    path = Path(path)
    path.write_text("\n".join(blocks), encoding="utf-8")
    return path


# --- Stage 1: transcribe + align + diarize ---

# faster-whisper occasionally emits a stray word timestamp far from the rest of
# a segment (a 0.3s word at t=2522 while the real speech is at t=2623), which
# inflates the whisper segment to 100s+. Placed at the stray word's time, such a
# segment leaves its whole slot silent until the next one. Split a segment
# wherever consecutive words are farther apart than this, so each burst of
# speech lands at its true time.
MAX_WORD_GAP = 2.0


# Segment bounds may legitimately sit past the word extent: extend_entry_tails
# pushes the end up to ~1.12s beyond the last word's early DTW stamp, and the
# silence-valley clamp moves edges by its pad. Anything further out is the old
# stray-timestamp bug and falls back to the word extent.
WORD_EXTENT_SLACK = 1.3


def _split_on_word_gaps(start: float, end: float, text: str, words: list, max_gap: float = MAX_WORD_GAP) -> list:
    """Split one segment into bursts separated by word gaps > ``max_gap``.

    ``words`` are ``{"w", "start", "end"}`` dicts. With no usable words the
    segment is returned unchanged; a single burst keeps the original (richer)
    text and its energy-verified bounds (unless they stray wildly from the
    word extent); multiple bursts rebuild each sub-text from the per-word
    tokens.
    """
    if not words:
        return [{"start": start, "end": end, "text": text, "words": []}]
    groups = [[words[0]]]
    for w in words[1:]:
        if float(w["start"]) - float(groups[-1][-1]["end"]) > max_gap:
            groups.append([w])
        else:
            groups[-1].append(w)
    if len(groups) == 1:
        # Do NOT clamp sane bounds to the word extent: the tail-extended end
        # carries real trailing voice that DTW word stamps miss (SI_TEST_2
        # lines 1/2/11 lost 0.3-0.4s of speech to the old clamp, audibly
        # truncating the dub).
        w_start, w_end = float(words[0]["start"]), float(words[-1]["end"])
        # A bound may sit outside the word extent, never inside it: a slot that
        # starts after its own first word cannot hold the line. Only the
        # outward direction is a judgement call, so only it consults the slack.
        keep_start = start if abs(start - w_start) <= WORD_EXTENT_SLACK else w_start
        keep_end = end if abs(end - w_end) <= WORD_EXTENT_SLACK else w_end
        keep_start = min(keep_start, w_start)
        keep_end = max(keep_end, w_end)
        return [{"start": keep_start, "end": keep_end, "text": text, "words": words}]
    subs = []
    for g in groups:
        t = "".join(x.get("w", "") for x in g).strip()
        subs.append({"start": float(g[0]["start"]), "end": float(g[-1]["end"]),
                     "text": t, "words": g})
    return subs


# The shared tool_subtitle front-end deliberately over-segments: it breaks an
# entry at every >0.7s word pause and every >0.8s silence valley so subtitles
# track speech exactly. tool_subtitle glues the pieces back together in
# ``merge_adjacent_fragments``; the dub path skips that call (it also rewrites
# timings for readability, which we must not inherit), so one sentence reached
# the synthesizer as several 0.2-1.0s slots. That costs the dub twice: every
# manifest segment is its own TTS inference *and* its speaker reference is cut
# from that same slot, so prosody restarts mid-sentence and the reference is too
# short to clone from. Merge the fragments back, bounded so a merged line still
# fits its slot at a natural tempo.
DUB_MERGE_MAX_GAP_SECONDS = 0.35
# Hard cap on a merged line: longer slots drift away from the picture and give
# ``duration_factor_for_slot`` too much rope.
DUB_MERGE_MAX_DURATION_SECONDS = 8.0
# Silence swallowed by a merge widens the slot without adding any text, and the
# tempo fit pays for that by slowing the whole line down. Cap how much silence
# one merged line may absorb.
DUB_MERGE_MAX_TOTAL_GAP_SECONDS = 0.6

# A sub-0.4s slot, or a one/two-character line inside a sub-1.0s slot, is a
# hanging fragment (失礼いたします | 本 | 日はご来店…): the decoder's DTW stamped
# the head of the next utterance onto the tail of the previous burst. Synthesised
# alone it is unusable, so it is folded into a neighbouring line even across a
# gap the normal merge would not bridge. Its own timestamp is the untrustworthy
# part, so an absorb across such a gap keeps the neighbour's slot and moves only
# the text.
DUB_ORPHAN_MAX_SLOT_SECONDS = 1.0
DUB_ORPHAN_MAX_DURATION_SECONDS = 0.4
DUB_ORPHAN_MAX_CHARS = 2
DUB_ORPHAN_MAX_GAP_SECONDS = 1.2

# Scratch key carrying a line's spent gap budget between the two passes; dropped
# before the manifest is written.
GAP_SPENT_KEY = "_gap_spent"

_PUNCTUATION_RE = re.compile(r"[\s、。，,.!?！？…・「」『』\"']")


def _visible_len(text: str) -> int:
    return len(_PUNCTUATION_RE.sub("", text or ""))


def _slot_seconds(seg: dict) -> float:
    return float(seg["end"]) - float(seg["start"])


def _same_speaker(a: dict, b: dict) -> bool:
    return (a.get("speaker") or "") == (b.get("speaker") or "")


def _is_orphan_fragment(seg: dict) -> bool:
    """A fragment too small to survive as its own TTS inference."""
    slot = _slot_seconds(seg)
    if slot >= DUB_ORPHAN_MAX_SLOT_SECONDS:
        return False
    return (
        slot < DUB_ORPHAN_MAX_DURATION_SECONDS
        or _visible_len(seg.get("src_text")) <= DUB_ORPHAN_MAX_CHARS
    )


def _gap_spent(seg: dict) -> float:
    """Silence this line has already swallowed by growing over its neighbours."""
    return float(seg.get(GAP_SPENT_KEY) or 0.0)


def _gap_between(target: dict, other: dict, *, other_first: bool) -> float:
    return (
        float(target["start"]) - float(other["end"]) if other_first
        else float(other["start"]) - float(target["end"])
    )


# Above this rate a slot cannot hold its line at any plausible tempo, whatever
# the timestamps claim: Japanese and Chinese TTS sit near 5.5-6.5 characters per
# second, so this is already more than twice natural speed. It is deliberately
# not a "reasonable pace" threshold — it only has to separate a merge that kept
# a sane slot from one that left a whole sentence on a sliver.
MAX_SPEAKABLE_CHARS_PER_SECOND = 14.0


def _slot_can_hold_line(seg: dict) -> bool:
    """Whether the slot could carry its own text at a plausible speaking rate."""
    visible = _visible_len(seg.get("src_text"))
    if visible <= 0:
        return True
    slot = _slot_seconds(seg)
    return slot > 0 and visible / slot <= MAX_SPEAKABLE_CHARS_PER_SECOND


def _cover_own_words(seg: dict) -> bool:
    """Grow a slot that cannot hold its line back over the words it carries.

    A slot may legitimately sit *outside* its word extent (``extend_entry_tails``
    pushes the end past the last word's early DTW stamp) and it may legitimately
    sit *inside* it too: when a hanging fragment is folded into a neighbour, the
    fragment's own timestamp is the untrustworthy part, so the merged line keeps
    the neighbour's slot and drops the fragment's timing on purpose.

    What is never legitimate is a slot too small to speak the text it ended up
    with. ``split_group_on_silence`` can clamp a piece edge to a silence run
    while keeping all of that piece's words, and an ``extend=False`` absorb can
    move a whole sentence into a sliver of a neighbour: on sivr-314 part1 that
    left 34 of 81 lines unable to hold their own words, the worst a full sentence
    on 0.12 s. ``fit_audio_to_duration`` then time-compresses the utterance into
    the sliver and the dub is inaudible. Only that case is repaired here, so the
    deliberate fragment-timing behaviour above survives untouched.

    Returns whether the slot was grown.
    """
    words = seg.get("words") or []
    if not words or _slot_can_hold_line(seg):
        return False
    start, end = float(seg["start"]), float(seg["end"])
    covered_start = min(start, float(words[0]["start"]))
    covered_end = max(end, float(words[-1]["end"]))
    if covered_start >= start - 1e-6 and covered_end <= end + 1e-6:
        return False
    seg["start"], seg["end"] = covered_start, covered_end
    return True


def _absorb_fragment(target: dict, other: dict, *, other_first: bool, extend: bool) -> None:
    """Fold ``other``'s text and words into ``target``, keeping timeline order.

    ``extend`` grows the slot over ``other``; without it ``target`` keeps its own
    bounds and takes the text only — and swallows no silence, which is why only
    the extending case charges the gap budget.
    """
    if extend:
        target[GAP_SPENT_KEY] = round(
            _gap_spent(target) + _gap_spent(other)
            + max(0.0, _gap_between(target, other, other_first=other_first)),
            3,
        )
    target_text = target.get("src_text") or ""
    other_text = other.get("src_text") or ""
    target_words = target.get("words") or []
    other_words = other.get("words") or []
    if other_first:
        target["src_text"] = other_text + target_text
        target["words"] = other_words + target_words
        if extend:
            target["start"] = min(float(target["start"]), float(other["start"]))
    else:
        target["src_text"] = target_text + other_text
        target["words"] = target_words + other_words
        if extend:
            target["end"] = max(float(target["end"]), float(other["end"]))
    _cover_own_words(target)
    target["dur"] = round(_slot_seconds(target), 3)
    overlap = float(target.get("speaker_overlap") or 0.0) + float(other.get("speaker_overlap") or 0.0)
    if overlap:
        # Both sides carry the same speaker, so the overlaps simply add up.
        target["speaker_overlap"] = round(overlap, 3)
        target["speaker_overlap_ratio"] = round(min(1.0, overlap / max(1e-3, target["dur"])), 3)


def _absorb_orphan_fragments(segments: list) -> tuple[list, int]:
    """Fold every hanging fragment into a neighbouring line."""
    kept: list = []
    absorbed = 0
    for index, seg in enumerate(segments):
        target = None
        other_first = False
        extend = False
        if _is_orphan_fragment(seg):
            previous = kept[-1] if kept else None
            following = segments[index + 1] if index + 1 < len(segments) else None
            # Forward first: a hanging fragment is nearly always the *head* of the
            # next utterance (本 | 日はご来店…) that the decoder stamped onto the
            # tail of the previous burst, so gluing it backwards would split a
            # word across two inferences. Only when the next line is out of reach
            # is the fragment read as a tail and folded back.
            for neighbour, ahead in ((following, True), (previous, False)):
                if neighbour is None or not _same_speaker(neighbour, seg):
                    continue
                if _slot_seconds(neighbour) > DUB_MERGE_MAX_DURATION_SECONDS:
                    continue
                gap = (
                    float(neighbour["start"]) - float(seg["end"]) if ahead
                    else float(seg["start"]) - float(neighbour["end"])
                )
                if gap > DUB_ORPHAN_MAX_GAP_SECONDS:
                    continue
                span = (
                    float(neighbour["end"]) - float(seg["start"]) if ahead
                    else float(seg["end"]) - float(neighbour["start"])
                )
                target, other_first = neighbour, ahead
                # Growing over the fragment is preferred, but a line that has
                # already swallowed its share of silence just takes the text.
                extend = (
                    gap <= DUB_MERGE_MAX_GAP_SECONDS
                    and span <= DUB_MERGE_MAX_DURATION_SECONDS
                    and _gap_spent(neighbour) + _gap_spent(seg) + max(0.0, gap)
                    <= DUB_MERGE_MAX_TOTAL_GAP_SECONDS
                )
                break
        if target is None:
            kept.append(seg)
            continue
        _absorb_fragment(target, seg, other_first=other_first, extend=extend)
        absorbed += 1
    return kept, absorbed


def _join_adjacent_lines(segments: list) -> tuple[list, int]:
    """Merge neighbours separated by a near-zero gap, within the slot budget."""
    merged: list = []
    joined = 0
    for seg in segments:
        if merged:
            previous = merged[-1]
            gap = float(seg["start"]) - float(previous["end"])
            span = float(seg["end"]) - float(previous["start"])
            spent = _gap_spent(previous) + _gap_spent(seg) + max(0.0, gap)
            if (
                gap <= DUB_MERGE_MAX_GAP_SECONDS
                and span <= DUB_MERGE_MAX_DURATION_SECONDS
                and spent <= DUB_MERGE_MAX_TOTAL_GAP_SECONDS
                and _same_speaker(previous, seg)
            ):
                _absorb_fragment(previous, seg, other_first=False, extend=True)
                joined += 1
                continue
        merged.append(seg)
    return merged, joined


def _merge_dub_fragments(segments: list, log: LogCallback = print) -> list:
    """Glue decoder fragments back into speakable lines, renumbering the result.

    Hanging fragments are re-attached first — while their neighbours are still
    separate lines, so the fragment can still reach the utterance it belongs to —
    then the remaining near-contiguous lines are joined. Speakers are never
    mixed, and no merge may push a line past the slot budget.
    """
    if len(segments) < 2:
        return segments

    result, absorbed = _absorb_orphan_fragments(segments)
    result, joined = _join_adjacent_lines(result)

    # Safety net over both passes: whatever route a line took, it must end up
    # able to hold the words attached to it.
    covered = sum(1 for seg in result if _cover_own_words(seg))
    if covered:
        log(f"[seg] dub merge: widened {covered} line(s) that could not hold their own words")

    for index, seg in enumerate(result, start=1):
        seg.pop(GAP_SPENT_KEY, None)
        seg["id"] = index
        seg["srt_index"] = index
        seg["start"] = round(float(seg["start"]), 3)
        seg["end"] = round(float(seg["end"]), 3)
        seg["dur"] = round(_slot_seconds(seg), 3)
    if joined or absorbed:
        log(
            f"[seg] dub merge: joined {joined} adjacent fragment(s), "
            f"absorbed {absorbed} hanging fragment(s); "
            f"{len(segments)} -> {len(result)} lines"
        )
    return result


def run_transcribe_diarize(
    video_path: str | Path,
    *,
    model_key: str = "anime-whisper",
    language: Optional[str] = None,
    diarize_backend: str = "auto",
    num_speakers: Optional[int] = None,
    target_language: str = "",
    ref_strategy: str = "hybrid",
    models_root: str,
    denoise: str = "none",
    vad_sensitivity: str = "high",
    precomputed_turns: Optional[list] = None,
    log: LogCallback = print,
    stop_event: Optional[Event] = None,
    model_holder: Optional[list] = None,
    skip_diarization: bool = False,
    skip_existing: bool = False,
) -> dict:
    """Transcribe + word-align + diarize a video, writing the manifest.

    ``model_holder`` collects native (CTranslate2/torch) models so the caller
    can release them on the main thread, avoiding a background-thread C++
    destructor crash.
    """
    video = Path(video_path)
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")

    # Reported here rather than in run_full: batch runs drive the stages
    # directly and would otherwise show no memory readings at all.
    log_memory("before transcription", log)

    def _check_stop():
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("Stopped by user.")

    cdir = clone_dir(video)
    cdir.mkdir(parents=True, exist_ok=True)
    audio_wav = cdir / AUDIO16K_NAME

    _check_stop()
    if skip_existing and audio_wav.is_file():
        log(f"[resume] audio exists, reusing: {audio_wav}")
    else:
        wx.extract_audio_16k(
            str(video), str(audio_wav), log=log, stop_event=stop_event, denoise=denoise
        )

    if skip_existing:
        existing = _load_resume_manifest(video, log)
        if _source_checkpoint_ready(existing):
            source_srt = cdir / SOURCE_SRT_NAME
            if source_srt.is_file():
                log(f"[resume] source subtitles exist, reusing: {source_srt}")
            else:
                write_srt(source_srt, existing.get("segments", []), "src_text", speaker_prefix=False)
                log(f"[resume] rebuilt missing source subtitles from manifest: {source_srt}")
            log(f"[resume] transcription manifest exists, skipping ASR: {manifest_path(video)}")
            return existing

    asr_device, asr_compute = wx.resolve_asr_device()
    torch_device, _ = wx.resolve_device()
    log(f"[device] transcription={asr_device}/{asr_compute}, align+diarize={torch_device}")
    if asr_device == "cpu" and torch_device == "cuda":
        log("[device] CTranslate2 cuDNN 8 DLLs not found; transcription runs on CPU to avoid a hard crash.")

    from tool_subtitle import logic as subl
    subl.warn_noisy_high_sensitivity(denoise, vad_sensitivity, log)

    with wx.torch_load_compat():
        _check_stop()
        # High-recall WhisperSeg-VAD + chunked decode (tool_subtitle engine),
        # producing offset-corrected per-word timestamps. Replaces the old
        # single-pass whisperx call that dropped quiet speech.
        result = seg_engine.transcribe(
            str(audio_wav), model_key=model_key, models_root=models_root,
            language=language, vad_sensitivity=vad_sensitivity,
            log=log, model_holder=model_holder,
        )
        detected_lang = result.get("language") or (language or "")
        raw_segments = result.get("segments", [])

        _check_stop()
        resolved_backend = "none" if skip_diarization else diar.resolve_backend(diarize_backend, models_root)
        if skip_diarization:
            turns = []
            log("[diarize] skipped (single voice clone uses each sentence WAV as its reference)")
        elif precomputed_turns is not None:
            turns = [
                (float(ts), float(te), str(spk))
                for ts, te, spk in precomputed_turns
                if float(te) > float(ts)
            ]
            log(f"[diarize] using {len(turns)} precomputed global turn(s)")
        else:
            turns = diar.diarize_primary_speakers(
                str(audio_wav), backend=diarize_backend, num_speakers=num_speakers,
                models_root=models_root, device=torch_device, log=log,
            )

    segments = []
    sidx = 0
    for s in raw_segments:
        if s.get("start") is None or s.get("end") is None:
            continue
        words = [
            {"w": w.get("word", ""), "start": float(w["start"]), "end": float(w["end"])}
            for w in s.get("words", [])
            if w.get("start") is not None and w.get("end") is not None
        ]
        # Repair faster-whisper's stray-timestamp segments (see _split_on_word_gaps).
        for sub in _split_on_word_gaps(float(s["start"]), float(s["end"]),
                                       (s.get("text") or "").strip(), words):
            sidx += 1
            segments.append({
                "id": sidx,
                "srt_index": sidx,
                "start": round(sub["start"], 3),
                "end": round(sub["end"], 3),
                "dur": round(sub["end"] - sub["start"], 3),
                "src_text": sub["text"],
                "tgt_text": "",
                "emotion_ref": "",
                "words": sub["words"],
            })

    if not skip_diarization:
        # Split before labelling: assign_speakers only tags a line with whoever
        # dominates it, so a line holding two speakers would keep one voice.
        segments = diar.split_segments_by_turns(segments, turns, log=log)
        diar.assign_speakers(segments, turns)
    # Speakers first: a merge must never span a speaker change.
    segments = _merge_dub_fragments(segments, log)
    # A stock hallucination split across entries survives the per-line guards:
    # on HNVR-174 the YouTube credit arrived as two lines and became SRT #1/#2,
    # 15 seconds of invented dialogue at the head of the file. Run this after
    # merging, so the pieces are adjacent in their final form.
    before = len(segments)
    segments = subl.drop_split_stock_phrases(segments, text_key="src_text", log_callback=log)
    if len(segments) != before:
        for index, item in enumerate(segments, 1):
            item["id"] = index
            item["srt_index"] = index
    speaker_ids = sorted({seg["speaker"] for seg in segments if seg.get("speaker")})

    manifest = {
        "video": str(video),
        "language": detected_lang,
        "target_language": target_language,
        "diarize_backend": resolved_backend,
        "ref_strategy": ref_strategy,
        "diarization_turns": [] if skip_diarization else [
            {"start": round(float(ts), 3), "end": round(float(te), 3), "speaker": spk}
            for ts, te, spk in turns
        ],
        "segments": segments,
    }
    if not skip_diarization:
        manifest["speakers"] = {spk: {"ref_audio": "", "ref_text": "", "score": 0.0} for spk in speaker_ids}
    save_manifest(video, manifest)
    write_srt(cdir / SOURCE_SRT_NAME, segments, "src_text", speaker_prefix=False)
    log(f"[manifest] {len(segments)} segments" + (f", {len(speaker_ids)} speaker(s)" if not skip_diarization else "") + f" -> {manifest_path(video)}")
    log(f"[srt] source subtitles -> {cdir / SOURCE_SRT_NAME}")

    _release_cuda_cache()
    log_memory("after transcription", log)
    return manifest


# --- Stage 2: per-speaker reference extraction ---

def run_extract_references(
    video_path: str | Path,
    *,
    models_root: str,
    log: LogCallback = print,
    stop_event: Optional[Event] = None,
) -> dict:
    """Select + cut one reference clip per speaker, updating the manifest.

    Requires the manifest produced by :func:`run_transcribe_diarize` and the
    intermediate ``audio16k.wav`` to be present.
    """
    video = Path(video_path)
    manifest = load_manifest(video)
    if not manifest:
        raise FileNotFoundError(f"Manifest not found; run transcription first: {manifest_path(video)}")

    cdir = clone_dir(video)
    audio_wav = cdir / AUDIO16K_NAME
    if not audio_wav.is_file():
        raise FileNotFoundError(f"Intermediate audio missing: {audio_wav}")

    if stop_event is not None and stop_event.is_set():
        raise RuntimeError("Stopped by user.")

    refsel.extract_references(str(video), manifest, str(audio_wav), cdir, log=log)
    save_manifest(video, manifest)
    log(f"[manifest] references updated -> {manifest_path(video)}")
    return manifest


# --- Stage 3: AI translation ---

# Its own prompt, not tool_subtitle's: this one is given each line's measured
# seconds and the character budget that fits them. Measured on sivr-314 part1,
# the character-ratio guidance alone left 23 of 80 lines unable to fit -- the
# ratio assumes the source is spoken at a normal rate, and lines there run to
# 11.2 kana/s against the 6.5 the pipeline assumes.
CLONE_TRANSLATE_PROMPT = "translate_prompt_clonevoice.txt"
# Characters (or words) of the target language that fit in one second, spoken
# unhurried. Mirrors backend.natural_reading_seconds, which decides how long
# the synthesized line is actually allowed to take.
TARGET_CHARS_PER_SECOND = {"zh": 5.5, "ja": 6.5, "en": 13.0, "es": 13.5, "ar": 11.0}
# A line may exceed its budget by this much before it is sent back: the engine
# speaks up to ~20% faster, and past that it runs into the following silence.
TRANSLATION_BUDGET_TOLERANCE = 1.25


def _target_chars_per_second(target_language: str) -> float:
    from . import backend

    return TARGET_CHARS_PER_SECOND.get(backend.normalize_language(target_language), 12.0)


def line_budget(seconds: float, cps: float) -> int:
    """How many characters of the target language fit in ``seconds``."""
    return max(1, int(round(float(seconds) * cps)))


def translation_budgets(segments: list, target_language: str) -> dict:
    """Per-line ``dur``/``max`` attributes for the prompt, keyed by segment id."""
    cps = _target_chars_per_second(target_language)
    budgets = {}
    for item in segments:
        start, end = item.get("start"), item.get("end")
        if start is None or end is None:
            continue
        seconds = max(0.0, float(end) - float(start))
        if seconds <= 0.0:
            continue
        budgets[int(item["id"])] = f'dur="{seconds:.2f}" max="{line_budget(seconds, cps)}"'
    return budgets


def overlong_translations(segments: list, target_language: str,
                          tolerance: float = TRANSLATION_BUDGET_TOLERANCE) -> list:
    """Ids whose translation cannot be spoken in the time the line has.

    Only a real overrun counts. A line shorter than its budget is fine -- the
    dub simply finishes early -- and is never sent back for another pass.
    """
    cps = _target_chars_per_second(target_language)
    over = []
    for item in segments:
        text = (item.get("tgt_text") or "").strip()
        start, end = item.get("start"), item.get("end")
        if not text or start is None or end is None:
            continue
        seconds = max(0.0, float(end) - float(start))
        if seconds <= 0.0:
            continue
        if len("".join(text.split())) > line_budget(seconds, cps) * tolerance:
            over.append(int(item["id"]))
    return over


def run_translate(
    video_path: str | Path,
    *,
    target_language: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.5,
    max_retries: int = 3,
    source_correction: Optional[bool] = None,
    log: LogCallback = print,
    stop_event: Optional[Event] = None,
    skip_existing: bool = False,
) -> dict:
    """Translate each segment's ``src_text`` into ``tgt_text`` for dubbing.

    Reuses tool_subtitle's LLM client, chunking, and the dubbing-optimized prompt
    (config/translate_prompt_dubbing.txt). The API endpoint/model come from the
    shared translation config; the key from the argument or the saved keyring.

    ``source_correction`` (default: the shared config's flag) runs an LLM
    proofread pass over the ASR source text first, fixing recognition errors in
    context; corrected text is written back to the manifest's ``src_text`` so
    the proofread panel and source.srt show the fixed lines.
    """
    video = Path(video_path)
    manifest = load_manifest(video)
    if not manifest:
        raise FileNotFoundError(f"Manifest not found; run earlier stages first: {manifest_path(video)}")

    resumed_target = target_language or manifest.get("target_language")
    if skip_existing and _translation_checkpoint_ready(manifest, resumed_target):
        translated_srt = clone_dir(video) / TRANSLATED_SRT_NAME
        if translated_srt.is_file():
            log(f"[resume] translated subtitles exist, reusing: {translated_srt}")
        else:
            write_srt(translated_srt, manifest.get("segments", []), "tgt_text", speaker_prefix=False)
            log(f"[resume] rebuilt missing translated subtitles from manifest: {translated_srt}")
        log(f"[resume] translation is complete in manifest, skipping translation API: {manifest_path(video)}")
        return manifest

    translated_srt = clone_dir(video) / TRANSLATED_SRT_NAME
    if skip_existing and translated_srt.is_file():
        if _restore_translations_from_srt(video, manifest, resumed_target, log):
            return manifest
        # Even if the SRT is unusable for the selected target, its existence
        # means source proofreading already ran in a prior translation pass.
        # Never spend correction tokens a second time merely because manifest
        # persistence was incomplete.
        source_correction = False
        log(f"[resume] translated subtitles already exist; skipping AI source proofreading: {translated_srt}")

    from tool_subtitle import logic as sl

    cfg = sl.load_trans_config()
    target_language = target_language or manifest.get("target_language") or cfg.get("target_language") or "Chinese"

    # A forced rerun or a changed target language must not leave old target
    # text on lines that the new API call fails to translate. The checkpoint
    # return above is the only path that intentionally preserves prior output.
    segments = manifest.get("segments", [])
    for item in segments:
        item["tgt_text"] = ""
    if isinstance(manifest.get("proofread"), dict):
        manifest["proofread"] = dict(manifest["proofread"])
        manifest["proofread"]["cleared_ids"] = []

    if not api_key:
        try:
            import keyring

            api_key = (
                keyring.get_password("VR_Video_Toolbox", "deepseek_api_key")
                or keyring.get_password("VR_Mosaic_Removal", "deepseek_api_key")
            )
        except Exception:
            api_key = None
    if not api_key:
        raise RuntimeError("No translation API key provided or saved.")

    client = sl.make_llm_client(cfg, api_key, temperature=temperature)

    entries = {
        int(s["id"]): {"text": s["src_text"]}
        for s in segments
        if (s.get("src_text") or "").strip()
    }
    if not entries:
        raise ValueError("No source text to translate.")

    tokens_per_chunk = int(cfg.get("tokens_per_chunk", 500000))
    adult_content = bool(cfg.get("adult_content", True))

    if source_correction is None:
        source_correction = bool(cfg.get("source_correction", True))
    if source_correction:
        lang_names = {"ja": "Japanese", "zh": "Chinese", "en": "English"}
        src_lang = manifest.get("language") or "ja"
        log(f"[translate] AI source proofread ({src_lang}) ...")
        changed, deleted = sl.correct_entries(
            client,
            entries,
            lang_names.get(src_lang, src_lang),
            tokens_per_chunk,
            adult_content,
            max_retries,
            log_callback=log,
            stop_event=stop_event,
        )
        if changed or deleted:
            # Write the corrected source back so the proofread panel and
            # source.srt show what the dub is actually translated from.
            # Interjection-only lines (moans/laughs) marked deleted get an
            # empty src/tgt so source.srt, translated.srt and the synthesized
            # dub all skip them.
            for s in segments:
                sid = int(s["id"])
                if sid in deleted:
                    s["src_text"] = ""
                    s["tgt_text"] = ""
                elif sid in entries:
                    new_src = entries[sid]["text"].strip()
                    if new_src:
                        s["src_text"] = new_src
            write_srt(clone_dir(video) / "source.srt", segments, "src_text", speaker_prefix=False)

    log(f"[translate] {len(entries)} segments -> {target_language} ({cfg.get('model_name')})")
    budgets = translation_budgets(segments, target_language)
    sl.translate_entries(
        client,
        entries,
        target_language,
        tokens_per_chunk,
        keep_original=False,
        adult_content=adult_content,
        # The clone pipeline always dubs, so always use a dubbing prompt — never
        # inherit tool_subtitle's checkbox from the shared config. The clone
        # prompt supersedes it: same rules plus each line's real time budget.
        dubbing_optimized=True,
        prompt_name=CLONE_TRANSLATE_PROMPT,
        budgets=budgets,
        max_retries=max_retries,
        log_callback=log,
        stop_event=stop_event,
    )

    translated = 0
    for s in segments:
        sid = int(s["id"])
        # Only take lines the LLM actually translated: for dropped/missing
        # translations `entries[sid]["text"]` still holds the SOURCE text,
        # which must never end up in tgt_text (the dub would speak Japanese).
        if sid in entries and entries[sid].get("translated") and entries[sid]["text"].strip():
            s["tgt_text"] = entries[sid]["text"].strip()
            translated += 1

    _retighten_overlong(client, segments, entries, target_language, budgets,
                        max_retries, adult_content, log, stop_event)

    # Every line that still has source text and no translation, however it got
    # that way. Asking `entries` which lines came back "translated" missed the
    # ones marked translated whose text was empty or was rejected by the
    # kana-leak backstop: they got no tgt_text and no cleared mark either, so
    # the "already translated" check stayed false forever and every export sent
    # the whole video back to the API. On 3dsvr-1911 part 1, 7 hallucinated
    # lines held 187 entries hostage that way, run after run, and re-running
    # could never fix them -- the LLM has nothing to translate.
    untranslated = sorted(
        int(s["id"]) for s in segments
        if str(s.get("src_text") or "").strip() and not str(s.get("tgt_text") or "").strip()
    )
    # Not when the pass produced nothing at all: that is an API failure, and
    # marking the whole video cleared would call it finished with no dub.
    if untranslated and translated:
        pf = manifest.get("proofread") if isinstance(manifest.get("proofread"), dict) else {}
        pf = dict(pf)
        existing = {int(x) for x in (pf.get("cleared_ids") or []) if str(x).lstrip("-").isdigit()}
        pf["cleared_ids"] = sorted(existing | set(untranslated))
        manifest["proofread"] = pf
        log(f"[translate] {len(untranslated)} line(s) left untranslated (garbled) and marked cleared: {untranslated[:10]}")

    manifest["target_language"] = target_language
    save_manifest(video, manifest)
    write_srt(clone_dir(video) / TRANSLATED_SRT_NAME, segments, "tgt_text", speaker_prefix=False)
    log(f"[translate] {translated}/{len(entries)} translated -> {manifest_path(video)}")
    sl.log_llm_usage(client, log, header="API token usage")
    log(f"[srt] translated subtitles -> {clone_dir(video) / TRANSLATED_SRT_NAME}")
    return manifest


def _retighten_overlong(client, segments: list, entries: dict, target_language: str,
                        budgets: dict, max_retries: int, adult_content: bool,
                        log: LogCallback, stop_event) -> int:
    """Send lines that still overrun their budget back for a second pass.

    The prompt asks for a fitting line; nothing checked that it got one. A line
    kept only if the retry is BOTH shorter and still a translation -- an empty
    or source-language answer leaves the first attempt in place, because a line
    that overruns is better than a line the dub cannot speak.
    """
    from tool_subtitle import logic as sl

    if stop_event is not None and stop_event.is_set():
        return 0
    over = overlong_translations(segments, target_language)
    if not over:
        log("[translate] every line fits its time budget")
        return 0
    by_id = {int(item["id"]): item for item in segments}
    log(f"[translate] {len(over)} line(s) overrun their time budget; asking again: {over[:10]}")
    pool = {sid: (by_id[sid].get("src_text") or "") for sid in over if sid in by_id}
    chunk = sl._with_context(over, pool)
    template = sl._load_prompt_template(adult_content, True, CLONE_TRANSLATE_PROMPT)
    retried = sl._llm_chunk_pass(
        client, chunk, template, {"target_language": target_language},
        max(1, min(2, max_retries)), log, stop_event,
        budgets={sid: budgets[sid] for sid in chunk if sid in budgets},
    )
    cps = _target_chars_per_second(target_language)
    tightened = 0
    for sid in over:
        item = by_id.get(sid)
        candidate = (retried.get(sid) or "").strip()
        if item is None or not candidate:
            continue
        if sl._KANA_RE.search(candidate) and not sl._target_expects_kana(target_language):
            continue
        before = len("".join((item.get("tgt_text") or "").split()))
        after = len("".join(candidate.split()))
        if after >= before:
            continue
        item["tgt_text"] = candidate
        if sid in entries:
            entries[sid]["text"] = candidate
        tightened += 1
        budget = line_budget(float(item["end"]) - float(item["start"]), cps)
        log(f"[translate] {sid}: {before} -> {after} chars (budget {budget})")
    still = len(overlong_translations(segments, target_language))
    log(f"[translate] tightened {tightened} line(s); {still} still overrun "
        "(kept: meaning matters more than the overrun, and the synthesizer "
        "runs those into the silence behind them)")
    return tightened


def synthesis_signature(manifest: dict, *, text_field: str = "tgt_text",
                        language: Optional[str] = None, tempo_fit: str = "moderate",
                        level_match: bool = True, timbre_anchor: bool = True,
                        speaker_refs: dict | None = None) -> str:
    """Fingerprint of everything that decides what the dub sounds like.

    Lets "skip the export when the .si.wav is already there" mean what it says
    it means. Without it the check was purely "does the file exist", so
    proofreading a video and exporting again with the box ticked skipped the
    whole video and silently threw the edits away -- the log said `skipped` and
    nothing else.
    """
    import hashlib

    parts = [
        f"v2|{text_field}|{language or ''}|{tempo_fit}|"
        f"{int(bool(level_match))}|{int(bool(timbre_anchor))}"
    ]
    # The chosen basis decides the voice of every line that speaker has, so
    # swapping it has to re-export as surely as editing the text does. Content,
    # not path: re-picking the same clip from a different candidate file is not
    # a change, and picking a different clip under the same name is.
    for speaker, path in sorted((speaker_refs or {}).items()):
        try:
            digest = hashlib.sha1(Path(path).read_bytes()).hexdigest()[:16]
        except OSError:
            digest = "missing"
        parts.append(f"basis|{speaker}|{digest}")
    for item in manifest.get("segments", []) or []:
        text = str(item.get(text_field) or "").strip()
        if not text:
            continue
        parts.append(
            f"{item.get('id')}|{item.get('speaker') or ''}|"
            f"{float(item.get('start') or 0.0):.3f}|{float(item.get('end') or 0.0):.3f}|{text}"
        )
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()


def synthesis_is_current(video_path: str | Path, **kwargs) -> bool:
    """Whether <video>.si.wav already reflects the manifest as it stands now."""
    from tool_si import logic as si

    if not Path(si.default_si_audio_path(str(video_path))).is_file():
        return False
    manifest = _load_resume_manifest(video_path, lambda _m: None)
    if not isinstance(manifest, dict):
        return False
    recorded = manifest.get("synthesis")
    if not isinstance(recorded, dict):
        # Exported before this marker existed: re-export once so the output is
        # known to match, rather than trusting a file whose provenance is
        # unknown.
        return False
    # Derived here rather than taken from the caller: the check and the export
    # must fingerprint the same basis, and every caller of this would otherwise
    # have to remember to pass it.
    kwargs.setdefault(
        "speaker_refs", chosen_speaker_refs(video_path, manifest, log=lambda _m: None)
    )
    return str(recorded.get("signature") or "") == synthesis_signature(manifest, **kwargs)


def chosen_speaker_refs(video: str | Path, manifest: dict,
                        *, log: LogCallback = print) -> dict[str, str]:
    """Basis WAVs a person picked for this video, by speaker.

    The refined-clone tab writes its choice here and synthesis never read it,
    so auditioning and choosing a speaker's voice changed nothing about the
    dub. Names are stored relative to the clone directory; one that no longer
    exists is reported and dropped rather than silently ignored, because the
    fallback is a different voice.
    """
    speakers = manifest.get("speakers")
    if not isinstance(speakers, dict):
        return {}
    cdir = clone_dir(video)
    out: dict[str, str] = {}
    for speaker, info in sorted(speakers.items()):
        if not isinstance(info, dict):
            continue
        name = str(info.get("ref_audio") or "").strip()
        if not name:
            continue
        path = Path(name)
        if not path.is_absolute():
            path = cdir / name
        if path.is_file():
            out[str(speaker)] = str(path)
        else:
            log(f"[synth] chosen basis for {speaker} is missing, falling back: {path}")
    return out


# --- Stage 4: IndexTTS-2.5 voice-clone synthesis -> <video>.si.wav ---

def run_synthesize(
    video_path: str | Path,
    *,
    models_root: str,
    model=None,
    text_field: str = "tgt_text",
    language: Optional[str] = None,
    tempo_fit: str = "moderate",
    level_match: bool = True,
    timbre_anchor: bool = True,
    log: LogCallback = print,
    stop_event: Optional[Event] = None,
    model_holder: Optional[list] = None,
) -> str:
    """Synthesize the cloned dub and write ``<video>.si.wav``.

    ``language`` defaults to the manifest's target language when synthesizing
    ``tgt_text`` and the source language when synthesizing ``src_text``.
    """
    video = Path(video_path)
    log_memory("before synthesis", log)
    manifest = load_manifest(video)
    if not manifest:
        raise FileNotFoundError(f"Manifest not found; run earlier stages first: {manifest_path(video)}")
    if language is None:
        language = manifest.get("target_language") if text_field == "tgt_text" else manifest.get("language")
    from . import backend
    from tool_si import logic as si
    if level_match:
        # Levels come from the source mixture, measured by the synthesizer off
        # audio16k.wav. A bandit-separated speech stem used to be preferred
        # where one existed; it produced a usable reading on only 140 of 304
        # lines (hnvr-174) and agreed with the mixture to a median 0.0-0.3 dB
        # where it did work, for a 13 GB VRAM peak and a full-length 48 kHz
        # scratch extraction on every run, so it was removed outright.
        log("[level] targets come from the source mixture")
    if model is None:
        model = backend.load_model(models_root, log=log)
        if model_holder is not None:
            model_holder.append(model)
    segments = []
    for item in manifest.get("segments", []):
        text = (item.get(text_field) or "").strip()
        if text:
            item = dict(item)
            item["text"] = text
            segments.append(item)
    speaker_refs = chosen_speaker_refs(video, manifest, log=log)
    out = backend.synthesize_manifest(
        model, clone_dir(video) / AUDIO16K_NAME, segments,
        si.default_si_audio_path(str(video)),
        language=language or "zh", fit_duration=(tempo_fit != "off"), tempo_fit=tempo_fit,
        log=log, stop_event=stop_event,
        intermediate_dir=clone_dir(video) / backend.MANIFEST_INTERMEDIATE_DIR_NAME,
        level_match=level_match, timbre_anchor=timbre_anchor,
        speaker_refs=speaker_refs,
    )
    duck_spans = [
        {"start": float(s["start"]), "end": float(s["end"])}
        for s in segments if (s.get("text") or "").strip()
    ]
    duck_path = si.default_si_duck_key_path(out)
    duck_duration = max((float(s["end"]) for s in segments), default=1.0)
    si.write_duck_key_wav(duck_path, duck_spans, duck_duration, backend.SAMPLE_RATE)
    log(f"[synth] wrote duck key {duck_path}")
    manifest["synthesis"] = {
        "signature": synthesis_signature(
            manifest, text_field=text_field, language=language,
            tempo_fit=tempo_fit, level_match=level_match, timbre_anchor=timbre_anchor,
            speaker_refs=speaker_refs,
        ),
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_manifest(video, manifest)
    _release_cuda_cache()
    log_memory("after synthesis", log)
    return out


# --- Automatic batch pipeline: transcribe -> translate -> synthesize ---

def run_full(
    video_path: str | Path,
    *,
    model_key: str = "anime-whisper",
    language: Optional[str] = None,
    target_language: str = "Chinese",
    models_root: str,
    keep_intermediate: bool = True,
    skip_existing: bool = True,
    denoise: str = "none",
    tempo_fit: str = "moderate",
    level_match: bool = True,
    timbre_anchor: bool = True,
    diarize_backend: str = "none",
    num_speakers: Optional[int] = None,
    source_correction: Optional[bool] = None,
    vad_sensitivity: str = "high",
    log: LogCallback = print,
    stop_event: Optional[Event] = None,
    model_holder: Optional[list] = None,
) -> str:
    """Run the whole voice-clone-translation pipeline, returning the .si.wav path."""
    from tool_si import logic as si

    video = Path(video_path)
    out_path = si.default_si_audio_path(str(video))
    if skip_existing and Path(out_path).exists():
        log(f"[full] output exists, skipping: {out_path}")
        return str(out_path)

    log("=== [1/3] Transcription (sentence references, no speaker detection) ===")
    run_transcribe_diarize(
        video, model_key=model_key, language=language, diarize_backend=diarize_backend,
        num_speakers=num_speakers, target_language=target_language, models_root=models_root,
        denoise=denoise, vad_sensitivity=vad_sensitivity,
        skip_diarization=(diarize_backend == "none"),
        skip_existing=skip_existing,
        log=log, stop_event=stop_event, model_holder=model_holder,
    )
    # ASR models can occupy several GB. Release them before
    # IndexTTS loads, otherwise batch runs may carry both stages in VRAM.
    release_model_holder(model_holder)

    log("=== [2/3] AI translation ===")
    run_translate(
        video, target_language=target_language, source_correction=source_correction,
        skip_existing=skip_existing, log=log, stop_event=stop_event,
    )

    log("=== [3/3] IndexTTS-2.5 sentence-level voice-clone synthesis ===")
    out = run_synthesize(
        video, models_root=models_root, text_field="tgt_text", language=target_language,
        tempo_fit=tempo_fit, level_match=level_match, timbre_anchor=timbre_anchor,
        log=log, stop_event=stop_event,
        model_holder=model_holder,
    )
    if not keep_intermediate:
        cleanup_clone_dir(video, log=log)
    log_memory("done", log)
    log(f"=== Done -> {out} ===")
    return out


def run_batch(
    video_paths,
    *,
    model_key: str = "anime-whisper",
    language: Optional[str] = None,
    target_language: str = "Chinese",
    models_root: str,
    keep_intermediate: bool = True,
    skip_existing: bool = True,
    denoise: str = "none",
    tempo_fit: str = "moderate",
    level_match: bool = True,
    timbre_anchor: bool = True,
    diarize_backend: str = "none",
    num_speakers: Optional[int] = None,
    source_correction: Optional[bool] = None,
    vad_sensitivity: str = "high",
    log: LogCallback = print,
    stop_event: Optional[Event] = None,
    model_holder: Optional[list] = None,
) -> list[str]:
    """Run each directory through transcription, translation, and synthesis.

    A directory is fully synthesized before the next directory starts. ASR and
    IndexTTS are intentionally not kept in VRAM together; IndexTTS is loaded
    at most once per directory. When later directories can skip ASR from their
    checkpoints, the already loaded IndexTTS model is reused across them.
    """
    from . import backend
    from tool_si import logic as si

    videos = [Path(path) for path in video_paths]
    outputs_by_video: dict[Path, str] = {}
    pending_by_directory: dict[Path, list[Path]] = {}
    for video in videos:
        output = Path(si.default_si_audio_path(str(video)))
        if skip_existing and synthesis_is_current(
            video, language=target_language, tempo_fit=tempo_fit,
            level_match=level_match, timbre_anchor=timbre_anchor,
        ):
            log(f"[batch] output is up to date, skipping: {output}")
            outputs_by_video[video] = str(output)
        else:
            pending_by_directory.setdefault(video.parent, []).append(video)

    directory_groups = list(pending_by_directory.items())
    shared_model = None
    for directory_index, (directory, pending) in enumerate(directory_groups, 1):
        can_keep_indextts = skip_existing and all(
            transcription_checkpoint_ready(video) for video in pending
        )
        if shared_model is not None and not can_keep_indextts:
            log("[batch] releasing IndexTTS before ASR for the next directory")
            shared_model = None
            release_model_holder(model_holder)
        log(
            f"=== [batch directory {directory_index}/{len(directory_groups)}] "
            f"{directory} ({len(pending)} video(s)) ==="
        )
        for video_index, video in enumerate(pending, 1):
            if stop_event is not None and stop_event.is_set():
                raise RuntimeError("Stopped by user.")
            log(
                f"=== [directory {directory_index} transcribe+translate "
                f"{video_index}/{len(pending)}] {video} ==="
            )
            run_transcribe_diarize(
                video,
                model_key=model_key,
                language=language,
                diarize_backend=diarize_backend,
                num_speakers=num_speakers,
                target_language=target_language,
                models_root=models_root,
                denoise=denoise,
                vad_sensitivity=vad_sensitivity,
                skip_diarization=(diarize_backend == "none"),
                skip_existing=skip_existing,
                log=log,
                stop_event=stop_event,
                model_holder=model_holder,
            )
            if shared_model is None:
                release_model_holder(model_holder)
            run_translate(
                video,
                target_language=target_language,
                source_correction=source_correction,
                skip_existing=skip_existing,
                log=log,
                stop_event=stop_event,
            )

        if shared_model is None:
            log(
                f"=== [directory {directory_index} synthesize] loading IndexTTS-2.5 "
                f"once for {len(pending)} video(s) ==="
            )
            shared_model = backend.load_model(models_root, log=log)
            if model_holder is not None:
                model_holder.append(shared_model)
        else:
            log(
                f"=== [directory {directory_index} synthesize] reusing loaded "
                f"IndexTTS-2.5 for {len(pending)} video(s) ==="
            )
        try:
            for video_index, video in enumerate(pending, 1):
                if stop_event is not None and stop_event.is_set():
                    raise RuntimeError("Stopped by user.")
                log(
                    f"=== [directory {directory_index} synthesize "
                    f"{video_index}/{len(pending)}] {video} ==="
                )
                outputs_by_video[video] = run_synthesize(
                    video,
                    models_root=models_root,
                    model=shared_model,
                    text_field="tgt_text",
                    language=target_language,
                    tempo_fit=tempo_fit,
                    level_match=level_match,
                    timbre_anchor=timbre_anchor,
                    log=log,
                    stop_event=stop_event,
                )
                # Only videos this run actually synthesized are cleaned; a video
                # skipped because its .SI.WAV already existed keeps whatever
                # checkpoint it has, since this run did no work on it.
                if not keep_intermediate:
                    cleanup_clone_dir(video, log=log)
        except Exception:
            shared_model = None
            release_model_holder(model_holder)
            raise

        log(f"=== [batch directory complete] {directory} ===")

    return [outputs_by_video[video] for video in videos]
