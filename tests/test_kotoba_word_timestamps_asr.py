from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_kotoba_word_timestamps_transcribe_does_not_crash():
    if os.environ.get("RUN_KOTOBA_ASR_TEST") != "1":
        pytest.skip("Set RUN_KOTOBA_ASR_TEST=1 to run the heavy Kotoba ASR smoke test.")

    repo_root = Path(__file__).resolve().parents[1]
    video_path = repo_root / "videos" / "sub_35s.mp4"
    model_dir = repo_root / "models" / "kotoba-whisper-v2.0-faster"

    if not video_path.exists():
        pytest.skip(f"Missing ASR test video: {video_path}")
    if not (model_dir / "config.json").exists() or not (model_dir / "model.bin").exists():
        pytest.skip(f"Missing Kotoba CT2 model files: {model_dir}")

    script = f"""
import json
from pathlib import Path

from faster_whisper import WhisperModel
from tool_subtitle.logic import repair_kotoba_alignment_heads

video_path = Path({json.dumps(str(video_path))})
model_dir = Path({json.dumps(str(model_dir))})

logs = []
repair_kotoba_alignment_heads("kotoba", str(model_dir), logs.append)

device = "cuda"
try:
    model = WhisperModel(str(model_dir), device="cuda", compute_type="auto", num_workers=1)
except Exception as exc:
    print("CUDA_LOAD_FAILED:" + repr(exc))
    device = "cpu"
    model = WhisperModel(str(model_dir), device="cpu", compute_type="int8", num_workers=1)

segments, info = model.transcribe(
    str(video_path),
    language="ja",
    task="transcribe",
    vad_filter=False,
    beam_size=1,
    best_of=1,
    temperature=0.0,
    condition_on_previous_text=False,
    word_timestamps=True,
)

segment_count = 0
word_count = 0
word_field_seen = False
for segment in segments:
    segment_count += 1
    if segment.words is not None:
        word_field_seen = True
        word_count += len(segment.words)
    if segment_count >= 2 and word_field_seen:
        break

print("ASR_RESULT " + json.dumps({{
    "device": device,
    "segments": segment_count,
    "words": word_count,
    "word_field_seen": word_field_seen,
}}, ensure_ascii=False))

if segment_count < 1:
    raise SystemExit("Kotoba ASR produced no segments")
if not word_field_seen:
    raise SystemExit("Kotoba ASR did not expose segment.words with word_timestamps=True")
"""

    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
    )

    assert result.returncode == 0, (
        f"Kotoba word_timestamps smoke test failed with code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    if result.stdout:
        print(result.stdout, end="")
    assert "ASR_RESULT" in result.stdout
