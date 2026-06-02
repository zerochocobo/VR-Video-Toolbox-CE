"""External subtitle loader and language prioritize logic.

Locates external .srt/.ass/.vtt files alongside video and ranks Chinese subtitles first.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUBTITLE_MIME_BY_SUFFIX = {
    ".srt": "application/x-subrip",
    ".ass": "application/x-ass",
    ".ssa": "application/x-ssa",
    ".vtt": "text/vtt",
}


@dataclass(frozen=True)
class SubtitleTrack:
    path: Path
    lang: str
    kind: str
    mime: str

    @property
    def suffix(self) -> str:
        return self.path.suffix.lower()


def subtitle_mime(path: Path) -> str:
    return SUBTITLE_MIME_BY_SUFFIX.get(path.suffix.lower(), "text/plain")


def is_subtitle_path(path: Path) -> bool:
    return path.suffix.lower() in SUBTITLE_MIME_BY_SUFFIX


def _infer_lang(video_stem: str, subtitle_stem: str) -> str:
    prefix = f"{video_stem}."
    if subtitle_stem.casefold().startswith(prefix.casefold()):
        return subtitle_stem[len(prefix):].replace("_", "-")
    return ""


def _lang_rank(lang: str) -> tuple[int, str]:
    key = lang.casefold()
    if not key:
        return (0, key)
    if key in {"zh", "zh-cn", "zh-hans", "cn", "chi", "zho", "chs"} or key.startswith("zh-"):
        return (1, key)
    if key in {"en", "eng", "en-us", "en-gb"} or key.startswith("en-"):
        return (2, key)
    return (3, key)


def find_external_subtitles(video_path: Path, enabled: bool = True, media_library=None) -> list[SubtitleTrack]:
    """Find external subtitles for the given video path."""
    if not enabled:
        return []
    parent = video_path.parent
    stem = video_path.stem
    tracks: list[SubtitleTrack] = []
    seen: set[str] = set()
    for suffix in SUBTITLE_MIME_BY_SUFFIX:
        candidates = [parent / f"{stem}{suffix}"]
        try:
            candidates.extend(sorted(parent.glob(f"{stem}.*{suffix}")))
        except Exception:
            pass
        for path in candidates:
            try:
                resolved = path.resolve()
            except Exception:
                continue
            key = str(resolved).casefold()
            if key in seen or not resolved.is_file():
                continue
            if media_library is not None and not media_library.contains(resolved):
                continue
            seen.add(key)
            kind = resolved.suffix.lower().lstrip(".")
            tracks.append(
                SubtitleTrack(
                    path=resolved,
                    lang=_infer_lang(stem, resolved.stem),
                    kind=kind,
                    mime=subtitle_mime(resolved),
                )
            )
    tracks.sort(key=lambda item: (_lang_rank(item.lang), item.path.name.casefold()))
    return tracks
