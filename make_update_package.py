from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
APP_NAME = "VR_Video_Toolbox"
DEFAULT_SOURCE = ROOT / "dist" / APP_NAME
DEFAULT_BASELINE = ROOT / ".base" / APP_NAME
DEFAULT_OUTPUT = ROOT / "dist" / "update"


class UpdatePackageError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise UpdatePackageError(message)


def info(message: str) -> None:
    print(message, flush=True)


def on_rm_error(func, path, exc_info) -> None:
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        raise


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, onerror=on_rm_error)
    else:
        try:
            path.chmod(stat.S_IWRITE)
        except OSError:
            pass
        path.unlink()


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            dst.chmod(stat.S_IWRITE)
        except OSError:
            pass
    shutil.copy2(src, dst)


def format_bytes(value: int) -> str:
    units = ("B", "KB", "MB", "GB")
    size = float(value)
    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{value} B"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_tree(root: Path) -> dict[str, tuple[int, Path]]:
    files: dict[str, tuple[int, Path]] = {}
    for item in root.rglob("*"):
        if "__pycache__" in item.relative_to(root).parts:
            continue
        if not item.is_file():
            continue
        rel = item.relative_to(root).as_posix()
        files[rel] = (item.stat().st_size, item)
    return files


def is_changed(src: Path, base: Path, src_size: int, base_size: int) -> bool:
    if src_size != base_size:
        return True
    return file_sha256(src) != file_sha256(base)


def validate_paths(source: Path, baseline: Path, output: Path) -> tuple[Path, Path, Path]:
    source = source.resolve()
    baseline = baseline.resolve()
    output = output.resolve()

    if not source.exists():
        fail(f"Source package directory not found: {source}")
    if not source.is_dir():
        fail(f"Source package path is not a directory: {source}")
    if not baseline.exists():
        fail(f"Baseline package directory not found: {baseline}")
    if not baseline.is_dir():
        fail(f"Baseline package path is not a directory: {baseline}")
    if output == source or output == baseline:
        fail("Output directory must be different from source and baseline.")
    if output in source.parents or output in baseline.parents:
        fail("Output directory cannot be a parent of source or baseline.")

    return source, baseline, output


def make_update_package(source: Path, baseline: Path, output: Path, *, clean: bool, dry_run: bool) -> Path:
    source, baseline, output = validate_paths(source, baseline, output)

    if clean and not dry_run:
        remove_path(output)
    if not dry_run:
        output.mkdir(parents=True, exist_ok=True)

    source_files = snapshot_tree(source)
    baseline_files = snapshot_tree(baseline)

    added: list[str] = []
    changed: list[str] = []
    removed = sorted(set(baseline_files) - set(source_files))

    for rel in sorted(source_files):
        src_size, src_path = source_files[rel]
        base_entry = baseline_files.get(rel)
        if base_entry is None:
            added.append(rel)
            if not dry_run:
                copy_file(src_path, output / rel)
                info(f"copied added: {rel}")
            continue
        base_size, base_path = base_entry
        if is_changed(src_path, base_path, src_size, base_size):
            changed.append(rel)
            if not dry_run:
                copy_file(src_path, output / rel)
                info(f"copied changed: {rel}")

    copied = added + changed
    copied_size = sum(source_files[rel][0] for rel in copied)
    source_size = sum(size for size, _ in source_files.values())
    baseline_size = sum(size for size, _ in baseline_files.values())

    lines = [
        f"created_at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"baseline:   {baseline}",
        f"source:     {source}",
        f"output:     {output}",
        f"dry_run:    {dry_run}",
        "",
        f"baseline_files: {len(baseline_files)}",
        f"source_files:   {len(source_files)}",
        f"added:          {len(added)}",
        f"changed:        {len(changed)}",
        f"removed:        {len(removed)}",
        f"copied_files:   {len(copied)}",
        f"baseline_size:  {format_bytes(baseline_size)}",
        f"source_size:    {format_bytes(source_size)}",
        f"update_size:    {format_bytes(copied_size)}",
        "",
        "Install note:",
        "  Extract this update package over the matching full installation directory.",
        "  Removed files are listed below but are not deleted automatically by this package.",
        "",
    ]

    def append_section(title: str, rows: list[str]) -> None:
        lines.append(title)
        if not rows:
            lines.append("  (none)")
        else:
            for rel in rows:
                suffix = ""
                if rel in source_files:
                    suffix = f" ({format_bytes(source_files[rel][0])})"
                lines.append(f"  {rel}{suffix}")
        lines.append("")

    append_section("ADDED", added)
    append_section("CHANGED", changed)
    append_section("REMOVED_NOT_INCLUDED", removed)

    manifest_path = output / "update_manifest.txt"
    if dry_run:
        manifest_path = ROOT / "debug_output" / f"update_manifest_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(lines), encoding="utf-8")

    info("")
    info("Update package:")
    for line in lines[:15]:
        info(line)
    info(f"manifest: {manifest_path}")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an incremental update package by comparing a full dist package with a baseline package."
    )
    parser.add_argument(
        "baseline",
        nargs="?",
        default=str(DEFAULT_BASELINE),
        help=f"Baseline full package directory. Default: {DEFAULT_BASELINE}",
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help=f"New full package directory. Default: {DEFAULT_SOURCE}",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Directory to write changed files into. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument("--no-clean", action="store_true", help="Do not delete the output directory before copying.")
    parser.add_argument("--dry-run", action="store_true", help="Only compare and write a manifest; do not copy files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        make_update_package(
            Path(args.source),
            Path(args.baseline),
            Path(args.output),
            clean=not args.no_clean,
            dry_run=args.dry_run,
        )
    except UpdatePackageError as e:
        info("")
        info(f"Update package failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
