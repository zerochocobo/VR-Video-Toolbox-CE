"""Discovery of benchmark items.

A benchmark item is one ``<video>.clone`` work directory that kept both the
16 kHz audio the ASR actually saw and the ``source.srt`` it produced. Running a
variant therefore never touches the 8K master -- the largest input is a ~100 MB
wav rather than a 10 GB mp4.

Roots are read from ``corpus.json`` next to this file so the paths stay out of
the code; ``--root`` overrides them for a one-off run.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name("corpus.json")

AUDIO_NAME = "audio16k.wav"
BASELINE_NAME = "source.srt"


@dataclass
class Item:
    clone_dir: Path

    @property
    def name(self) -> str:
        return self.clone_dir.name.removesuffix(".clone")

    @property
    def audio(self) -> Path:
        return self.clone_dir / AUDIO_NAME

    @property
    def baseline(self) -> Path:
        return self.clone_dir / BASELINE_NAME

    @property
    def is_usable(self) -> bool:
        return self.audio.is_file() and self.baseline.is_file()

    def variant(self, tag: str) -> Path:
        """Where a variant run writes its SRT, alongside the baseline."""
        return self.clone_dir / f"bench.{tag}.srt"


def load_roots() -> list[Path]:
    if not CONFIG_PATH.is_file():
        return []
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [Path(entry) for entry in data.get("roots", [])]


def discover(roots: list[Path] | None = None, pattern: str = "") -> list[Item]:
    """All usable items under ``roots``, sorted by name.

    ``pattern`` keeps only items whose directory name contains one of its
    comma-separated parts, so a subset can be swept without editing the config.
    """
    wanted = [part.strip().lower() for part in pattern.split(",") if part.strip()]
    # Keyed by item name, not by path: the same part can sit in two directories
    # (a title gets renamed, the old folder stays), and running both would count
    # one recording twice. The newer baseline wins.
    best: dict[str, Item] = {}
    for root in roots if roots is not None else load_roots():
        if not root.exists():
            continue
        for clone_dir in root.rglob("*.clone"):
            if not clone_dir.is_dir():
                continue
            item = Item(clone_dir)
            if not item.is_usable:
                continue
            if wanted and not any(part in item.name.lower() for part in wanted):
                continue
            previous = best.get(item.name)
            if previous is None or item.baseline.stat().st_mtime > previous.baseline.stat().st_mtime:
                best[item.name] = item
    return sorted(best.values(), key=lambda i: i.name)
