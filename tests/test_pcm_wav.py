from pathlib import Path
import ast
import importlib
import re
import wave

import numpy as np
import pytest
import torch
import torchaudio

from utils.pcm_wav import save_tensor_pcm16_wav


def test_project_pcm_writer_interleaves_stereo_without_codec_backend(tmp_path: Path) -> None:
    output = tmp_path / "stereo.wav"
    samples = torch.tensor([
        [-1.0, 0.0, 1.0],
        [0.5, -0.5, 0.25],
    ], dtype=torch.float32)

    save_tensor_pcm16_wav(output, samples, 48000)

    with wave.open(str(output), "rb") as wav_file:
        assert wav_file.getnchannels() == 2
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 48000
        pcm = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2")
    assert pcm.tolist() == [-32767, 16384, 0, -16384, 32767, 8192]


def test_project_pcm_writer_keeps_int16_samples_unscaled(tmp_path: Path) -> None:
    """Integer input is already in PCM16 units and must be written verbatim."""
    output = tmp_path / "int16.wav"
    samples = torch.tensor([[0, 6547, -6547, 32767, -32768]], dtype=torch.int16)

    save_tensor_pcm16_wav(output, samples, 22050)

    with wave.open(str(output), "rb") as wav_file:
        pcm = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2")
    assert pcm.tolist() == [0, 6547, -6547, 32767, -32768]


def test_project_pcm_output_roundtrips_through_pinned_soundfile_backend(tmp_path: Path) -> None:
    output = tmp_path / "roundtrip.wav"
    expected = torch.tensor([[0.0, 0.25, -0.5, 0.75]], dtype=torch.float32)
    save_tensor_pcm16_wav(output, expected, 22050)

    decoded, sample_rate = torchaudio.load(str(output), backend="soundfile")

    assert sample_rate == 22050
    assert decoded.shape == expected.shape
    assert torch.allclose(decoded, expected, atol=1.0 / 32767.0)


# Every packaged tool that touches PCM audio, so a new caller in any of them
# cannot reintroduce an unpinned torchaudio read/write.
AUDIO_SOURCE_ROOTS = (
    Path("tool_clonevoice"),
    Path("tool_clonevoice_v2"),
    Path("tool_si"),
    Path("tool_pinyin_wav_generator"),
    Path("utils"),
)

# IndexTTS keeps `wav` as float pre-scaled to +-32767, so these files must cast
# before saving. Passing the float tensor straight through makes the writer
# treat it as normalized [-1, 1] and emit a full-scale square wave.
INDEXTTS_INFER_SOURCES = (
    Path("tool_clonevoice_v2/vendor/indextts/infer.py"),
    Path("tool_clonevoice_v2/vendor/indextts/infer_v2.py"),
    Path("tool_clonevoice_v2/vendor/indextts/infer_v2_5.py"),
)


def test_runtime_sources_do_not_save_wav_through_torchaudio() -> None:
    pattern = re.compile(r"\b(?:torchaudio|ta)\.save\s*\(")
    offenders = []
    for root in AUDIO_SOURCE_ROOTS:
        for source in root.rglob("*.py"):
            if pattern.search(source.read_text(encoding="utf-8-sig")):
                offenders.append(str(source))
    assert offenders == []


def test_indextts_save_calls_cast_prescaled_float_to_int16() -> None:
    offenders = []
    checked_calls = 0
    for source in INDEXTTS_INFER_SOURCES:
        tree = ast.parse(source.read_text(encoding="utf-8-sig"), filename=str(source))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id != "save_pcm16_wav" or len(node.args) < 2:
                continue
            checked_calls += 1
            waveform = node.args[1]
            is_int16_cast = (
                isinstance(waveform, ast.Call)
                and isinstance(waveform.func, ast.Attribute)
                and waveform.func.attr == "type"
                and any(ast.unparse(arg) == "torch.int16" for arg in waveform.args)
            )
            if not is_int16_cast:
                offenders.append(f"{source}:{node.lineno}")
    assert checked_calls == 5
    assert offenders == []


def test_runtime_torchaudio_loads_are_pinned_to_soundfile() -> None:
    offenders = []
    for root in AUDIO_SOURCE_ROOTS:
        for source in root.rglob("*.py"):
            tree = ast.parse(source.read_text(encoding="utf-8-sig"), filename=str(source))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                owner = node.func.value
                if node.func.attr != "load" or not isinstance(owner, ast.Name):
                    continue
                if owner.id not in {"ta", "torchaudio"}:
                    continue
                backend = next((kw.value for kw in node.keywords if kw.arg == "backend"), None)
                if not isinstance(backend, ast.Constant) or backend.value != "soundfile":
                    offenders.append(f"{source}:{node.lineno}")
    assert offenders == []
