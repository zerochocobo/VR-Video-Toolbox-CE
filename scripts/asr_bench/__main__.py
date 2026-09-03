"""Command line for the ASR benchmark.

    python -m scripts.asr_bench list
    python -m scripts.asr_bench profile                       # baseline corpus
    python -m scripts.asr_bench profile --variant framed      # a variant run
    python -m scripts.asr_bench compare framed                # variant vs baseline
    python -m scripts.asr_bench score --gt-suffix .jp.srt     # against a reference

``--filter`` narrows every command to items whose name contains the string.
``--json PATH`` writes the machine-readable result next to the console table.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.asr_bench import corpus, decode, matcher, profile, report, variants  # noqa: E402

PROFILE_COLUMNS = [
    "name", "lines", "speech_seconds", "coverage",
    "stock_phrase", "nonverbal", "lone_token", "fragment", "glued_fragment",
    "sparse", "repetition", "long_line", "evidence_lines", "evidence_chars",
]
RATE_COLUMNS = [
    "name", "lines", "stock_phrase_rate", "nonverbal_rate", "lone_token_rate",
    "fragment_rate", "glued_fragment_rate", "sparse_rate", "long_line_rate",
]
COMPARE_COLUMNS = [
    "name", "a_lines", "b_lines", "matched", "lost", "gained",
    "text_changed", "mean_cer", "mean_iou",
    "median_start_shift", "median_end_shift", "lost_seconds", "gained_seconds",
]
SCORE_COLUMNS = [
    "name", "gt_lines", "hyp_lines", "matched", "missed", "hallucinated",
    "recall", "precision", "cer", "mean_iou",
]


def _items(args) -> list[corpus.Item]:
    roots = [Path(r) for r in args.root] if args.root else None
    items = corpus.discover(roots, pattern=args.filter or "")
    if not items:
        print("No benchmark items found. Check scripts/asr_bench/corpus.json.",
              file=sys.stderr)
    return items


def _srt_for(item: corpus.Item, variant: str | None) -> Path:
    return item.variant(variant) if variant else item.baseline


def cmd_list(args) -> int:
    items = _items(args)
    rows = []
    for item in items:
        seconds = profile.audio_duration_seconds(item.audio)
        rows.append({
            "name": item.name,
            "minutes": round(seconds / 60.0, 1) if seconds else None,
            "baseline_bytes": item.baseline.stat().st_size,
            "variants": ", ".join(
                sorted(p.name.removeprefix("bench.").removesuffix(".srt")
                       for p in item.clone_dir.glob("bench.*.srt"))) or "-",
        })
    print(report.table(rows, ["name", "minutes", "baseline_bytes", "variants"],
                       title=f"{len(items)} benchmark item(s)"))
    total = sum(r["minutes"] or 0 for r in rows)
    print(f"  total audio: {total / 60.0:.1f} h")
    return 0


def cmd_profile(args) -> int:
    items = _items(args)
    rows, rate_rows, payload = [], [], []
    for item in items:
        srt = _srt_for(item, args.variant)
        if not srt.is_file():
            print(f"  skip {item.name}: {srt.name} missing", file=sys.stderr)
            continue
        result = profile.profile_srt(srt, item.audio, name=item.name)
        data = result.as_dict()
        rows.append(data)
        rate_rows.append(data)
        payload.append({**data, "examples": result.examples})
        if args.examples:
            print(f"\n{item.name}")
            print(report.examples_block(result.examples), end="")

    label = args.variant or "baseline"
    print()
    print(report.table(rows, PROFILE_COLUMNS, title=f"profile: {label}"))
    print(report.table(rate_rows, RATE_COLUMNS, title="rates (share of lines)"))
    _print_profile_totals(rows)
    if args.json:
        print(f"  wrote {report.write_json(args.json, payload)}")
    return 0


def _print_profile_totals(rows: list[dict]) -> None:
    if not rows:
        return
    lines = sum(r["lines"] for r in rows)
    if not lines:
        return
    parts = []
    for key in ("stock_phrase", "nonverbal", "lone_token", "fragment",
                "glued_fragment", "sparse", "repetition", "long_line"):
        count = sum(r[key] for r in rows)
        parts.append(f"{key} {count} ({100.0 * count / lines:.1f}%)")
    speech = sum(r["speech_seconds"] for r in rows)
    audio = sum(r["audio_seconds"] or 0 for r in rows)
    evidence_lines = sum(r["evidence_lines"] for r in rows)
    evidence_chars = sum(r["evidence_chars"] for r in rows)
    print(f"  corpus: {lines} lines, speech {speech / 60:.1f} min"
          + (f" / audio {audio / 60:.1f} min = {100.0 * speech / audio:.1f}%"
             if audio else ""))
    print("  " + ", ".join(parts))
    kanji = sum(r["kanji_chars"] for r in rows)
    print(f"  language evidence: {evidence_lines} lines ({100.0 * evidence_lines / lines:.1f}%), "
          f"{evidence_chars} chars, {kanji} kanji")


def cmd_build(args) -> int:
    items = _items(args)
    for item in items:
        quiet = (lambda *_a, **_k: None) if not args.verbose else print
        path = variants.build(args.variant, item, log=quiet)
        print(f"  {item.name}: wrote {path.name}")
    return 0


def cmd_decode(args) -> int:
    items = _items(args)
    for index, item in enumerate(items, start=1):
        print(f"[{index}/{len(items)}] {item.name} / {args.variant}")
        decode.run(args.variant, item, language=args.language,
                   model_key=args.model, log=print if args.verbose else _step_log)
    return 0


def _step_log(message: str) -> None:
    """Only the lines that say how the run is going."""
    if message.startswith(("[seg] chunk", "[seg] kept", "[seg] rejoined",
                           "[seg] dropped non-verbal", "[seg] pulled", "[bench]",
                           "WhisperSeg split")):
        print("   ", message)


def cmd_compare(args) -> int:
    items = _items(args)
    rows, payload = [], []
    for item in items:
        a_path = _srt_for(item, args.base_variant)
        b_path = _srt_for(item, args.variant)
        if not a_path.is_file() or not b_path.is_file():
            missing = a_path if not a_path.is_file() else b_path
            print(f"  skip {item.name}: {missing.name} missing", file=sys.stderr)
            continue
        result = matcher.compare(
            profile.read_srt(a_path), profile.read_srt(b_path), name=item.name)
        rows.append(result.as_dict())
        payload.append({**result.as_dict(), "examples": result.examples})
        if args.examples:
            print(f"\n{item.name}")
            print(report.examples_block(result.examples), end="")

    base_label = args.base_variant or "baseline"
    print()
    print(report.table(rows, COMPARE_COLUMNS,
                       title=f"compare: {base_label} (A) -> {args.variant} (B)"))
    if rows:
        lost = sum(r["lost"] for r in rows)
        gained = sum(r["gained"] for r in rows)
        matched = sum(r["matched"] for r in rows)
        a_total = sum(r["a_lines"] for r in rows)
        print(f"  corpus: {matched} matched, {lost} lost "
              f"({100.0 * lost / a_total:.1f}% of A), {gained} gained")
    if args.json:
        print(f"  wrote {report.write_json(args.json, payload)}")
    return 0


def cmd_score(args) -> int:
    items = _items(args)
    rows = []
    for item in items:
        hyp_path = _srt_for(item, args.variant)
        gt_path = _ground_truth(item, args.gt_suffix)
        if gt_path is None or not hyp_path.is_file():
            print(f"  skip {item.name}: no ground truth or hypothesis", file=sys.stderr)
            continue
        result = matcher.score(
            profile.read_srt(gt_path), profile.read_srt(hyp_path), name=item.name)
        rows.append(result.as_dict())
    print()
    print(report.table(rows, SCORE_COLUMNS,
                       title=f"score: {args.variant or 'baseline'} vs {args.gt_suffix}"))
    print("  NOTE: tool_subtitle .jp.srt is a reference, not ground truth -- its")
    print("        durations were remapped for readability and fragments merged.")
    if args.json:
        print(f"  wrote {report.write_json(args.json, rows)}")
    return 0


def _ground_truth(item: corpus.Item, suffix: str) -> Path | None:
    """A reference SRT sitting next to the video, e.g. ``<video>.jp.srt``."""
    stem = item.clone_dir.name.removesuffix(".clone")
    candidate = item.clone_dir.parent / f"{stem}{suffix}"
    return candidate if candidate.is_file() else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.asr_bench",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", action="append",
                        help="override corpus.json roots (repeatable)")
    parser.add_argument("--filter", help="only items whose name contains this")
    parser.add_argument("--json", help="also write results as JSON to this path")
    parser.add_argument("--examples", action="store_true",
                        help="print sample lines for each bucket")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="show the corpus").set_defaults(func=cmd_list)

    p_profile = sub.add_parser("profile", help="reference-free statistics")
    p_profile.add_argument("--variant", help="bench.<tag>.srt instead of source.srt")
    p_profile.set_defaults(func=cmd_profile)

    p_build = sub.add_parser("build", help="produce bench.<tag>.srt for every item")
    p_build.add_argument("variant", choices=sorted(variants.BUILDERS))
    p_build.add_argument("--verbose", action="store_true", help="show the pipeline log")
    p_build.set_defaults(func=cmd_build)

    p_decode = sub.add_parser("decode", help="re-run the ASR with variant settings")
    p_decode.add_argument("variant", choices=sorted(decode.variant_specs()))
    p_decode.add_argument("--language", default="ja")
    p_decode.add_argument("--model", default="kotoba")
    p_decode.add_argument("--verbose", action="store_true")
    p_decode.set_defaults(func=cmd_decode)

    p_compare = sub.add_parser("compare", help="A/B two runs over the same audio")
    p_compare.add_argument("variant", help="tag of the B side (bench.<tag>.srt)")
    p_compare.add_argument("--base-variant", help="tag of the A side; default source.srt")
    p_compare.set_defaults(func=cmd_compare)

    p_score = sub.add_parser("score", help="accuracy against a reference SRT")
    p_score.add_argument("--variant", help="bench.<tag>.srt instead of source.srt")
    p_score.add_argument("--gt-suffix", default=".jp.srt",
                         help="reference file suffix next to the video")
    p_score.set_defaults(func=cmd_score)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
