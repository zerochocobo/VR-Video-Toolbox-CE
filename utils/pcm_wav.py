"""Codec-library-independent PCM WAV output used by packaged audio tools."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import torch


def save_tensor_pcm16_wav(path: str | Path, waveform: torch.Tensor, sample_rate: int) -> None:
    """Save a channels-first tensor as a little-endian PCM16 WAV.

    Scale contract, matching ``torchaudio.save``: a floating-point ``waveform``
    is normalized audio in ``[-1, 1]``; an integer ``waveform`` is already in
    PCM16 units.  Callers holding float samples pre-scaled to +-32767 must cast
    with ``.type(torch.int16)`` first, or every audible sample clamps to +-1 and
    the file becomes a full-scale square wave.

    Do not use ``torchaudio.save`` for PCM output in the frozen application.
    Its backend dispatcher can bind torio to PyNvVideoCodec's FFmpeg DLLs,
    which support decoding but do not expose the ``pcm_s16le`` encoder.
    """
    samples = waveform.detach().cpu()
    if samples.ndim == 1:
        samples = samples.unsqueeze(0)
    if samples.ndim != 2:
        raise ValueError(f"Expected [channels, samples] audio, got shape {tuple(samples.shape)}")
    if samples.shape[0] < 1:
        raise ValueError("Cannot save audio with no channels")

    if torch.is_floating_point(samples):
        samples = torch.round(torch.clamp(samples, -1.0, 1.0) * 32767.0)
    samples = torch.clamp(samples, -32768, 32767).to(torch.int16)
    interleaved = np.asarray(samples.transpose(0, 1).contiguous().numpy(), dtype="<i2")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as wav_file:
        wav_file.setnchannels(int(samples.shape[0]))
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(interleaved.tobytes())
