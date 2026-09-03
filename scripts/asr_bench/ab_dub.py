"""Render the same passage twice, once from each transcript, for listening.

The benchmark can count characters and measure silence; it cannot say whether a
cloned take running over a one-second pause is unpleasant to hear. That is the
only question left between the kotoba path and the anime-whisper one, so this
builds the artefact that answers it.

Both sides get identical treatment -- same audio, same voice prompts, same
synthesiser, same fit -- so the only difference is where the lines start and end
and what they say. The source language is synthesised rather than a translation:
the question is about timing, and running the text through an LLM first would
add a second variable and a second failure mode.

    python -m scripts.asr_bench.ab_dub --filter 13dsvr01911_1 --start 700 --length 120
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.asr_bench import corpus, profile  # noqa: E402

MODELS_ROOT = str(_REPO_ROOT / "models")


def find_source_media(item: corpus.Item) -> Path | None:
    """The video the clone directory was made from, for full-rate audio.

    ``audio16k.wav`` is what the ASR heard, but 16 kHz makes a poor voice prompt
    and the point here is to listen. The video sits next to the .clone folder
    under the same stem.
    """
    stem = item.clone_dir.name.removesuffix(".clone")
    for suffix in (".mp4", ".mkv", ".m4v", ".avi", ".wmv"):
        candidate = item.clone_dir.parent / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def slice_audio(media: Path, start: float, length: float, target: Path) -> Path:
    """Mono 22.05 kHz slice, seeking before the input so an 8K file stays cheap."""
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(media),
         "-vn", "-ac", "1", "-ar", "22050", "-c:a", "pcm_s16le", str(target)],
        check=True,
    )
    return target


def lines_in_window(srt: Path, start: float, length: float) -> list[dict]:
    """Cues inside the window, retimed to window-relative seconds."""
    end = start + length
    picked = []
    for cue in profile.read_srt(srt):
        if cue.end <= start or cue.start >= end:
            continue
        picked.append({
            "start": max(0.0, cue.start - start),
            "end": min(length, cue.end - start),
            "text": cue.text,
        })
    return [p for p in picked if p["end"] - p["start"] > 0.15]


def mix(dub: Path, background: Path, target: Path, background_db: float = -8.0) -> Path:
    """Dub over the original at a reduced level, so timing is audible in context."""
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(background), "-i", str(dub),
         "-filter_complex",
         f"[0:a]volume={background_db}dB[bg];[bg][1:a]amix=inputs=2:duration=first:normalize=0[out]",
         "-map", "[out]", "-ac", "1", "-ar", "22050", "-c:a", "pcm_s16le", str(target)],
        check=True,
    )
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.asr_bench.ab_dub",
                                     description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--filter", required=True, help="benchmark item name substring")
    parser.add_argument("--start", type=float, required=True, help="window start, seconds")
    parser.add_argument("--length", type=float, default=120.0)
    parser.add_argument("--variants", default="win28cap,anime",
                        help="comma-separated bench tags to render")
    parser.add_argument("--language", default="ja")
    parser.add_argument("--out", default=str(_REPO_ROOT / "debug_output" / "ab_dub"))
    args = parser.parse_args(argv)

    items = corpus.discover(pattern=args.filter)
    if not items:
        print(f"no benchmark item matches {args.filter!r}", file=sys.stderr)
        return 1
    item = items[0]
    media = find_source_media(item)
    if media is None:
        print(f"no video found next to {item.clone_dir}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{item.name}_{int(args.start)}s"
    original = slice_audio(media, args.start, args.length, out_dir / f"{tag}.original.wav")
    print(f"  original slice -> {original.name}")

    from tool_clonevoice_v2 import backend

    model = backend.load_model(MODELS_ROOT)
    try:
        for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
            srt = item.variant(variant)
            if not srt.is_file():
                print(f"  skip {variant}: {srt.name} missing", file=sys.stderr)
                continue
            segments = lines_in_window(srt, args.start, args.length)
            print(f"  {variant}: {len(segments)} line(s)")
            dub = out_dir / f"{tag}.{variant}.dub.wav"
            backend.synthesize_segments(
                model, str(original), segments, dub,
                language=args.language, fit_duration=True,
                log=lambda message: None,
                intermediate_dir=str(out_dir / f"work_{variant}"),
            )
            mixed = mix(dub, original, out_dir / f"{tag}.{variant}.mixed.wav")
            print(f"    -> {dub.name} / {mixed.name}")
    finally:
        backend.release_model(model)

    print(f"\n  listen in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
