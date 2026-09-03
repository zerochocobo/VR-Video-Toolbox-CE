"""Console tables and JSON dumps for benchmark results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".") or "0"
    return str(value)


def _display_width(text: str) -> int:
    """Terminal columns ``text`` occupies, counting CJK as two."""
    width = 0
    for ch in text:
        width += 2 if ord(ch) > 0x1100 and _is_wide(ch) else 1
    return width


def _is_wide(ch: str) -> bool:
    code = ord(ch)
    return (
        0x1100 <= code <= 0x115F
        or 0x2E80 <= code <= 0xA4CF
        or 0xAC00 <= code <= 0xD7A3
        or 0xF900 <= code <= 0xFAFF
        or 0xFE30 <= code <= 0xFE6F
        or 0xFF00 <= code <= 0xFF60
        or 0xFFE0 <= code <= 0xFFE6
    )


def table(rows: Sequence[dict], columns: Sequence[str], title: str = "") -> str:
    """Render ``rows`` as a fixed-width table, CJK-aware."""
    if not rows:
        return f"{title}\n  (no rows)\n" if title else "  (no rows)\n"

    cells = [[_cell(row.get(col)) for col in columns] for row in rows]
    widths = [
        max(_display_width(str(col)), *(_display_width(row[i]) for row in cells))
        for i, col in enumerate(columns)
    ]

    def line(values: Sequence[str]) -> str:
        parts = []
        for value, width in zip(values, widths):
            pad = width - _display_width(value)
            parts.append(value + " " * max(0, pad))
        return "  ".join(parts).rstrip()

    out = []
    if title:
        out.append(title)
    out.append(line([str(c) for c in columns]))
    out.append("  ".join("-" * w for w in widths))
    out.extend(line(row) for row in cells)
    return "\n".join(out) + "\n"


def examples_block(examples: dict[str, list[str]], indent: str = "    ") -> str:
    out = []
    for bucket, lines in examples.items():
        if not lines:
            continue
        out.append(f"{indent}{bucket}:")
        out.extend(f"{indent}  {line}" for line in lines)
    return "\n".join(out) + ("\n" if out else "")


def totals(rows: Sequence[dict], columns: Sequence[str],
           label_column: str, label: str = "TOTAL") -> dict:
    """Column-wise sum of the integer columns; rates recomputed from them.

    Averaging per-part rates would weight a three-minute part the same as a
    fifty-minute one, so rates are always recomputed from the summed counts.
    """
    result: dict[str, Any] = {label_column: label}
    for column in columns:
        if column == label_column:
            continue
        values = [row.get(column) for row in rows if isinstance(row.get(column), (int, float))]
        if not values:
            result[column] = None
        elif column.endswith("_rate") or column in ("coverage", "mean_cer", "mean_iou"):
            result[column] = None  # filled in by the caller where meaningful
        else:
            result[column] = round(sum(values), 2) if any(
                isinstance(v, float) for v in values) else sum(values)
    return result


def write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target
