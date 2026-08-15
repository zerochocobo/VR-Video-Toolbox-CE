from __future__ import annotations

import numpy as np

import soundfile as sf

from tool_clonevoice_v2 import backend


class QuietFakeModel:
    def __init__(self, amplitude: float = 0.02):
        self.amplitude = amplitude

    def infer(self, **kwargs):
        wav = np.full(backend.SAMPLE_RATE // 2, self.amplitude, dtype=np.float32)
        sf.write(kwargs["output_path"], wav, backend.SAMPLE_RATE)
        return kwargs["output_path"]


def test_synthesize_segments_preserves_quiet_indextts_output(tmp_path) -> None:
    source = tmp_path / "source.wav"
    output = tmp_path / "output.wav"
    sf.write(source, np.zeros(backend.SAMPLE_RATE, dtype=np.float32), backend.SAMPLE_RATE)

    backend.synthesize_segments(
        QuietFakeModel(),
        source,
        [{"start": 0.0, "end": 0.5, "text": "whisper"}],
        output,
        language="en",
        fit_duration=False,
    )

    wav, sample_rate = sf.read(output, dtype="float32")
    assert sample_rate == backend.SAMPLE_RATE
    assert np.isclose(np.max(np.abs(wav)), 0.02, atol=5e-5)


def test_synthesize_manifest_preserves_quiet_indextts_output(tmp_path) -> None:
    source = tmp_path / "source.wav"
    output = tmp_path / "output.wav"
    sf.write(source, np.zeros(backend.SAMPLE_RATE * 3, dtype=np.float32), backend.SAMPLE_RATE)

    backend.synthesize_manifest(
        QuietFakeModel(),
        source,
        [{"id": 1, "start": 0.0, "end": 0.5, "text": "whisper"}],
        output,
        language="en",
        fit_duration=False,
    )

    wav, sample_rate = sf.read(output, dtype="float32")
    assert sample_rate == backend.SAMPLE_RATE
    assert np.isclose(np.max(np.abs(wav)), 0.02, atol=5e-5)
