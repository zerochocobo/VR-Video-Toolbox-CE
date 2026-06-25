from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MediaRoot:
    label: str
    path: Path


def safe_resolve_path(path: Path) -> Path:
    """Resolve paths without rejecting virtual drives that cannot report volume info."""
    expanded = Path(path).expanduser()
    try:
        return expanded.resolve()
    except (OSError, RuntimeError, ValueError):
        return Path(os.path.abspath(os.fspath(expanded)))


def parse_video_dirs(raw: object, default: Path) -> list[Path]:
    text = str(raw or "").strip()
    parts = [part.strip() for part in text.split("|") if part.strip()]
    if not parts:
        parts = [str(default)]
    roots: list[Path] = []
    seen: set[str] = set()
    for part in parts:
        path = safe_resolve_path(Path(part))
        key = str(path).casefold()
        if key in seen:
            continue
        seen.add(key)
        roots.append(path)
    return roots or [safe_resolve_path(default)]


def build_media_roots(paths: list[Path]) -> list[MediaRoot]:
    used: dict[str, int] = {}
    roots: list[MediaRoot] = []
    for path in paths:
        base = path.name or path.drive.rstrip(":\\") or "Videos"
        index = used.get(base.casefold(), 0) + 1
        used[base.casefold()] = index
        label = base if index == 1 else f"{base}{index}"
        roots.append(MediaRoot(label=label, path=safe_resolve_path(path)))
    return roots


class MediaLibrary:
    def __init__(self, roots: list[MediaRoot]) -> None:
        if not roots:
            raise ValueError("media library requires at least one root")
        self.roots = roots

    @property
    def multi_root(self) -> bool:
        return len(self.roots) > 1

    @property
    def first_root(self) -> MediaRoot:
        return self.roots[0]

    def path_to_key(self, path: Path) -> str:
        resolved = safe_resolve_path(path)
        matches: list[tuple[int, MediaRoot, Path]] = []
        for root in self.roots:
            try:
                rel = resolved.relative_to(root.path)
            except ValueError:
                continue
            matches.append((len(root.path.parts), root, rel))
        if matches:
            _depth, root, rel = max(matches, key=lambda item: item[0])
            rel_text = rel.as_posix()
            if self.multi_root:
                return root.label if not rel_text or rel_text == "." else f"{root.label}/{rel_text}"
            return "" if not rel_text or rel_text == "." else rel_text
        raise ValueError(f"path is outside media roots: {path}")

    def key_to_path(self, key: str) -> Path | None:
        rel = str(key or "").replace("\\", "/").strip("/")
        if Path(rel).is_absolute():
            return None
        if not self.multi_root:
            path = safe_resolve_path(self.first_root.path / rel)
            return path if self._contains_root(self.first_root, path) else None
        label, _, rest = rel.partition("/")
        if not label:
            return None
        if Path(rest).is_absolute():
            return None
        root = self.root_by_label(label)
        if root is None:
            return None
        path = safe_resolve_path(root.path / rest)
        return path if self._contains_root(root, path) else None

    def root_by_label(self, label: str) -> MediaRoot | None:
        wanted = str(label or "").casefold()
        for root in self.roots:
            if root.label.casefold() == wanted:
                return root
        return None

    def contains(self, path: Path) -> bool:
        resolved = safe_resolve_path(path)
        for root in self.roots:
            if self._contains_root(root, resolved):
                return True
        return False

    @staticmethod
    def _contains_root(root: MediaRoot, path: Path) -> bool:
        resolved = safe_resolve_path(path)
        root_path = safe_resolve_path(root.path)
        return resolved == root_path or root_path in resolved.parents
