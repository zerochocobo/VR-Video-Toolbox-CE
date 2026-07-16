from __future__ import annotations

import html
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tool_clonevoice import logic

SRT_TIME_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})"
)
HTML_TAG_RE = re.compile(r"<[^>]+>")


def _seconds(value: str) -> float:
    hh, mm, rest = value.replace(",", ".").split(":")
    ss, ms = rest.split(".", 1)
    ms = (ms + "000")[:3]
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def _clean_srt_text(lines: list[str]) -> str:
    text = " ".join(line.strip() for line in lines if line.strip())
    text = HTML_TAG_RE.sub("", text)
    return " ".join(html.unescape(text).split())


def parse_srt_seconds(path: str | Path) -> list[dict[str, Any]]:
    """Parse SRT cues into seconds, keeping malformed blocks out of the result."""
    srt_path = Path(path)
    raw = srt_path.read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"\r?\n\s*\r?\n", raw.strip())
    cues: list[dict[str, Any]] = []
    for block in blocks:
        lines = [line.rstrip("\r") for line in block.splitlines()]
        time_index = -1
        match = None
        for index, line in enumerate(lines):
            match = SRT_TIME_RE.search(line)
            if match:
                time_index = index
                break
        if not match or time_index < 0:
            continue
        text = _clean_srt_text(lines[time_index + 1 :])
        if not text:
            continue
        try:
            start = _seconds(match.group("start"))
            end = _seconds(match.group("end"))
        except Exception:
            continue
        if end <= start:
            continue
        cues.append({"start": start, "end": end, "text": text})
    return cues


def find_reference_srt(video: str | Path) -> Path | None:
    video_path = Path(video)
    ref = video_path.parent / f"{video_path.stem}.srt"
    return ref if ref.is_file() else None


def _fmt_time(start: float | None, end: float | None) -> str:
    if start is None or end is None:
        return ""
    return f"{logic._format_srt_ts(float(start))} -> {logic._format_srt_ts(float(end))}"


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def align_reference(segments: list[dict], cues: list[dict]) -> list[dict[str, Any]]:
    """Attach each reference cue to the segment with the strongest time overlap."""
    seg_rows: list[dict[str, Any]] = []
    assigned: dict[int, list[int]] = {idx: [] for idx, _seg in enumerate(segments)}
    unassigned: list[int] = []

    for cue_index, cue in enumerate(cues):
        cue_start = float(cue.get("start", 0.0))
        cue_end = float(cue.get("end", 0.0))
        cue_dur = max(0.0, cue_end - cue_start)
        best_index: int | None = None
        best_overlap = 0.0
        for seg_index, seg in enumerate(segments):
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", 0.0))
            ov = _overlap(cue_start, cue_end, seg_start, seg_end)
            if ov > best_overlap:
                best_overlap = ov
                best_index = seg_index
        if best_index is None:
            unassigned.append(cue_index)
            continue
        seg = segments[best_index]
        seg_dur = max(0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)))
        threshold = max(0.2, 0.3 * min(cue_dur, seg_dur))
        if best_overlap < threshold:
            unassigned.append(cue_index)
        else:
            assigned[best_index].append(cue_index)

    for seg_index, seg in enumerate(segments):
        ref_indexes = sorted(assigned.get(seg_index, []), key=lambda idx: float(cues[idx].get("start", 0.0)))
        ref_text = " ".join((cues[idx].get("text") or "").strip() for idx in ref_indexes).strip()
        ref_start = float(cues[ref_indexes[0]]["start"]) if ref_indexes else None
        ref_end = float(cues[ref_indexes[-1]]["end"]) if ref_indexes else None
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        tgt_text = (seg.get("tgt_text") or "").strip()
        seg_rows.append(
            {
                "kind": "seg",
                "seg_id": seg.get("id"),
                "start": start,
                "end": end,
                "time": _fmt_time(start, end),
                "speaker": seg.get("speaker") or "",
                "src_text": (seg.get("src_text") or "").strip(),
                "tgt_text": tgt_text,
                "original_tgt_text": tgt_text,
                "ref_text": ref_text,
                "ref_start": ref_start,
                "ref_end": ref_end,
                "ref_time": _fmt_time(ref_start, ref_end),
            }
        )

    ref_only_rows = []
    for cue_index in unassigned:
        cue = cues[cue_index]
        start = float(cue.get("start", 0.0))
        end = float(cue.get("end", 0.0))
        ref_only_rows.append(
            {
                "kind": "ref_only",
                "seg_id": None,
                "start": start,
                "end": end,
                "time": "",
                "speaker": "",
                "src_text": "",
                "tgt_text": "",
                "original_tgt_text": "",
                "ref_text": (cue.get("text") or "").strip(),
                "ref_start": start,
                "ref_end": end,
                "ref_time": _fmt_time(start, end),
            }
        )

    return sorted(seg_rows + ref_only_rows, key=lambda row: (float(row.get("start") or 0.0), 1 if row["kind"] == "ref_only" else 0))


def load_rows(video: str | Path) -> dict[str, Any]:
    manifest = logic.load_manifest(video)
    if manifest is None:
        raise FileNotFoundError(f"Manifest not found: {logic.manifest_path(video)}")
    segments = list(manifest.get("segments") or [])
    ref_srt = find_reference_srt(video)
    cues = parse_srt_seconds(ref_srt) if ref_srt else []
    return {
        "video": str(Path(video)),
        "manifest": manifest,
        "reference_srt": str(ref_srt) if ref_srt else "",
        "rows": align_reference(segments, cues),
    }


def _int_id_set(values) -> set[int]:
    ids: set[int] = set()
    for item in values or []:
        try:
            ids.add(int(item))
        except Exception:
            continue
    return ids


def cleared_segment_ids(manifest: dict) -> set[int]:
    """Ids the user deliberately left untranslated during proofreading."""
    proofread = manifest.get("proofread") if isinstance(manifest.get("proofread"), dict) else {}
    return _int_id_set(proofread.get("cleared_ids"))


def effective_cleared_segment_ids(manifest: dict) -> set[int]:
    """Return only explicitly persisted cleared translation segment ids."""
    return cleared_segment_ids(manifest)


def save_rows(video: str | Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = logic.load_manifest(video)
    if manifest is None:
        raise FileNotFoundError(f"Manifest not found: {logic.manifest_path(video)}")
    segments = manifest.get("segments") or []
    by_id = {str(seg.get("id")): seg for seg in segments}
    prev_cleared = cleared_segment_ids(manifest)

    edited_ids: list[int] = []
    cleared_ids: set[int] = set()
    for row in rows:
        if row.get("kind") != "seg":
            continue
        seg_id = row.get("seg_id")
        seg = by_id.get(str(seg_id))
        if seg is None:
            continue
        new_text = (row.get("tgt_text") or "").strip()
        old_text = (row.get("original_tgt_text") or "").strip()
        seg["tgt_text"] = new_text
        try:
            seg_id_int = int(seg_id)
        except Exception:
            seg_id_int = None
        if seg_id_int is not None:
            # A deliberately emptied line (now, or carried over from an earlier
            # proofread) must not look "untranslated" to ensure_translated,
            # otherwise export would re-translate the whole video and wipe edits.
            if not new_text and (seg.get("src_text") or "").strip() and (old_text or seg_id_int in prev_cleared):
                cleared_ids.add(seg_id_int)
            if new_text != old_text:
                edited_ids.append(seg_id_int)

    proofread = manifest.get("proofread") if isinstance(manifest.get("proofread"), dict) else {}
    all_edited = sorted(_int_id_set(proofread.get("edited_ids")).union(edited_ids))
    manifest["proofread"] = {
        "edited_ids": all_edited,
        "cleared_ids": sorted(cleared_ids),
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    cdir = logic.clone_dir(video)
    translated = cdir / "translated.srt"
    original = cdir / "translated_org.srt"
    logic.save_manifest(video, manifest)
    if translated.is_file() and not original.exists():
        shutil.copyfile(translated, original)
    logic.write_srt(translated, segments, "tgt_text", speaker_prefix=True)
    return {
        "edited_count": len(all_edited),
        "changed_count": len(edited_ids),
        "translated_srt": str(translated),
        "backup_srt": str(original) if original.exists() else "",
    }


def cut_segment_preview(video: str | Path, start: float, end: float, *, pad: float = 0.2) -> Path:
    """Cut [start-pad, end+pad] from the intermediate audio16k.wav for audition.

    Rewrites one shared preview file per video; the caller must stop any
    playback that still holds the previous clip before calling this.
    """
    import soundfile as sf

    cdir = logic.clone_dir(video)
    src = cdir / logic.AUDIO16K_NAME
    if not src.is_file():
        raise FileNotFoundError(f"Intermediate audio missing: {src}")
    with sf.SoundFile(str(src)) as f:
        sr = int(f.samplerate)
        begin = max(0, int((float(start) - pad) * sr))
        stop = min(int(f.frames), int((float(end) + pad) * sr))
        if stop <= begin:
            raise ValueError(f"Empty segment range: {start}..{end}")
        f.seek(begin)
        data = f.read(stop - begin)
    out = cdir / "pf_preview.wav"
    sf.write(str(out), data, sr)
    return out


def video_status(video: str | Path) -> dict[str, Any]:
    manifest = logic.load_manifest(video)
    ref_srt = find_reference_srt(video)
    if manifest is None:
        return {
            "status": "no_manifest",
            "total": 0,
            "translated": 0,
            "edited": 0,
            "reference_srt": str(ref_srt) if ref_srt else "",
        }
    # Only segments that still have source text need a translation: lines the
    # AI proofread removed (interjections/hallucinations) have src_text == ""
    # and must not make a fully translated video look "untranslated".
    segments = [
        s for s in (manifest.get("segments") or [])
        if (s.get("src_text") or "").strip()
    ]
    total = len(segments)
    cleared = effective_cleared_segment_ids(manifest)
    translated_srt = logic.clone_dir(video) / "translated.srt"
    legacy_completed = (
        "proofread" not in manifest
        and translated_srt.is_file()
        and any((seg.get("tgt_text") or "").strip() for seg in segments)
    )

    def _seg_done(seg: dict) -> bool:
        if (seg.get("tgt_text") or "").strip():
            return True
        if legacy_completed:
            return True
        try:
            return int(seg.get("id")) in cleared
        except Exception:
            return False

    translated = sum(1 for seg in segments if _seg_done(seg))
    proofread = manifest.get("proofread") if isinstance(manifest.get("proofread"), dict) else {}
    edited = len(proofread.get("edited_ids") or [])
    if total <= 0 or translated < total:
        status = "untranslated"
    elif edited:
        status = "proofread"
    else:
        status = "translated"
    return {
        "status": status,
        "total": total,
        "translated": translated,
        "edited": edited,
        "reference_srt": str(ref_srt) if ref_srt else "",
    }
