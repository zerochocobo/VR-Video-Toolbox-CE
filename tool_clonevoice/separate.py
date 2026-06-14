"""Bandit-v2 cinematic source separation for the dubbing mode.

Separates a video's audio into speech / music / sfx stems and returns the
``music + sfx`` background bed, i.e. the original audio with the dialogue
removed. The cloned/translated voice is later mixed onto this bed so the
result keeps the original score and sound effects but replaces the speech.

The 446 MB Lightning checkpoint (``checkpoint-multi.ckpt``) carries optimizer
state we never use. On first use we strip it down to a slim ``model.*``
state_dict (``checkpoint-multi.slim.pt``) cached next to it, so subsequent
launches load far less from disk.
"""
from __future__ import annotations

import os
from pathlib import Path
from threading import Event
from typing import Callable

import torch
import torchaudio as ta
from tqdm import tqdm

from tool_clonevoice.bandit.bandit import Bandit
from tool_clonevoice.bandit.inference_handler import (
    StandardTensorChunkedInferenceHandler,
)

LogCallback = Callable[[str], None]

FS = 48000
STEMS = ["speech", "music", "sfx"]
BANDIT_DIRNAME = "bandit-v2"
CKPT_NAME = "checkpoint-multi.ckpt"
SLIM_NAME = "checkpoint-multi.slim.pt"

# Chunked inference (configs/inference/chunked-tensor.yaml). batch=8 keeps
# peak VRAM well under the 16 GB budget at 48 kHz / 8 s chunks.
_CHUNK_SECONDS = 8.0
_HOP_SECONDS = 1.0

# Peak VRAM scales ~linearly with the model's per-forward batch. Measured on
# RTX 5060 Ti (sm_120, fp32): batch 1=1.39, 2=2.20, 3=3.02, 4=3.83 GB ->
# ~0.81 GB/chunk over a ~0.58 GB base. We pick the batch automatically to fill
# the budget below.
_VRAM_BASE_GB = 0.6
_VRAM_PER_CHUNK_GB = 0.85  # slightly above measured slope for safety margin
_VRAM_CAP_GB = 10.0        # never use more than this, even on big cards
_VRAM_RESERVE_GB = 2.0     # leave headroom for ffmpeg / other processes
_MAX_BATCH = 16
_CPU_BATCH = 2

# The chunked handler keeps the whole unfolded mixture and all three stem
# outputs resident on the device, so feeding a full-length track (tens of
# minutes) also blows past VRAM via accumulation. We separate in overlapping
# time blocks and cross-fade them, so peak VRAM stays bounded regardless of
# track length (the per-forward batch above sets the actual ceiling).
_BLOCK_SECONDS = 30.0
_OVERLAP_SECONDS = 4.0


def _auto_batch_size(
    device: str, cap_gb: float, reserve_gb: float, log: LogCallback
) -> int:
    """Choose a per-forward batch that keeps peak VRAM within budget.

    Budget = min(``cap_gb``, free_vram - ``reserve_gb``); batch is back-solved
    from the measured linear VRAM model and clamped to [1, _MAX_BATCH].
    """
    if device != "cuda":
        return _CPU_BATCH
    try:
        free_bytes, _total = torch.cuda.mem_get_info()
        free_gb = free_bytes / (1024 ** 3)
    except Exception:
        return 4
    budget = min(cap_gb, free_gb - reserve_gb)
    batch = int((budget - _VRAM_BASE_GB) / _VRAM_PER_CHUNK_GB)
    batch = max(1, min(_MAX_BATCH, batch))
    log(
        f"VRAM auto: free {free_gb:.1f} GB, reserve {reserve_gb:.0f} GB, "
        f"cap {cap_gb:.0f} GB -> batch {batch} "
        f"(~{_VRAM_BASE_GB + _VRAM_PER_CHUNK_GB * batch:.1f} GB peak)"
    )
    return batch

# bandit-mus64 model kwargs (configs/models/bandit-mus64.yaml).
_MODEL_KW = dict(
    in_channels=1,
    band_type="musical",
    n_bands=64,
    normalize_channel_independently=False,
    treat_channel_as_feature=True,
    n_sqm_modules=8,
    emb_dim=128,
    rnn_dim=256,
    bidirectional=True,
    rnn_type="GRU",
    mlp_dim=512,
    hidden_activation="Tanh",
    hidden_activation_kwargs=None,
    complex_mask=True,
    use_freq_weights=True,
    n_fft=2048,
    win_length=2048,
    hop_length=512,
    window_fn="hann_window",
    wkwargs=None,
    power=None,
    center=True,
    normalized=True,
    pad_mode="reflect",
    onesided=True,
    fs=FS,
)


def bandit_dir(models_root: str | os.PathLike[str]) -> Path:
    return Path(models_root) / BANDIT_DIRNAME


def ckpt_path(models_root: str | os.PathLike[str]) -> Path:
    return bandit_dir(models_root) / CKPT_NAME


def slim_path(models_root: str | os.PathLike[str]) -> Path:
    return bandit_dir(models_root) / SLIM_NAME


def is_available(models_root: str | os.PathLike[str]) -> bool:
    """True when either the slim or the full checkpoint is present."""
    return slim_path(models_root).is_file() or ckpt_path(models_root).is_file()


def _resolve_device() -> str:
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _ensure_slim_weights(models_root: str | os.PathLike[str], log: LogCallback) -> Path:
    """Return the slim weights path, building it from the full ckpt if needed."""
    slim = slim_path(models_root)
    if slim.is_file():
        return slim
    full = ckpt_path(models_root)
    if not full.is_file():
        raise FileNotFoundError(
            f"Bandit checkpoint not found: {full} (or {slim})"
        )
    log(f"First-time setup: extracting slim weights from {full.name} ...")
    ck = torch.load(full, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    msd = {k[len("model.") :]: v for k, v in sd.items() if k.startswith("model.")}
    tmp = slim.with_suffix(".tmp")
    torch.save(msd, tmp)
    tmp.replace(slim)
    size_mb = slim.stat().st_size / (1024 * 1024)
    log(f"Slim weights cached: {slim.name} ({size_mb:.0f} MB)")
    return slim


class BanditSeparator:
    """Resident bandit-v2 separator. Build once, reuse across many videos."""

    def __init__(
        self,
        models_root: str | os.PathLike[str],
        device: str | None = None,
        log: LogCallback = print,
        block_seconds: float = _BLOCK_SECONDS,
        overlap_seconds: float = _OVERLAP_SECONDS,
        inference_batch_size: int | None = None,
        vram_cap_gb: float = _VRAM_CAP_GB,
        vram_reserve_gb: float = _VRAM_RESERVE_GB,
    ) -> None:
        self.log = log
        self.device = device or _resolve_device()
        self.block_seconds = float(block_seconds)
        self.overlap_seconds = float(overlap_seconds)
        slim = _ensure_slim_weights(models_root, log)

        # Decide the batch from currently-free VRAM, before we allocate anything.
        if inference_batch_size is not None:
            batch = max(1, int(inference_batch_size))
        else:
            batch = _auto_batch_size(self.device, vram_cap_gb, vram_reserve_gb, log)

        log(f"Loading bandit-v2 separator on {self.device} ...")
        model = Bandit(stems=STEMS, **_MODEL_KW)
        state = torch.load(slim, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            log(
                f"Warning: bandit weights mismatch "
                f"(missing={len(missing)} unexpected={len(unexpected)})"
            )
        model.to(self.device).eval()
        self.model = model
        self.handler = StandardTensorChunkedInferenceHandler(
            chunk_size_seconds=_CHUNK_SECONDS,
            hop_size_seconds=_HOP_SECONDS,
            inference_batch_size=batch,
            fs=FS,
        ).to(self.device)
        # We show one overall progress bar over time blocks instead of the
        # handler's per-block "Rank 0:" bar.
        self.handler.disable_tqdm = True
        log("Bandit-v2 separator ready.")

    def separate_background(
        self,
        audio_path: str | os.PathLike[str],
        out_path: str | os.PathLike[str],
        stop_event: Event | None = None,
    ) -> str:
        """Separate ``audio_path`` and write the music+sfx bed to ``out_path``.

        The input is loaded as mono 48 kHz and processed in overlapping blocks
        so peak VRAM stays bounded regardless of track length. The dialogue stem
        is discarded. ``stop_event`` is honoured between blocks. Returns
        ``out_path``.
        """
        audio, fs = ta.load(str(audio_path))
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        if fs != FS:
            audio = ta.functional.resample(audio, fs, FS)
        wav = audio.reshape(-1)  # 1-D, CPU

        bg = self._separate_blocked(wav, stop_event)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        ta.save(str(out_path), bg.unsqueeze(0), FS)
        return str(out_path)

    def _run_handler(self, segment: torch.Tensor) -> torch.Tensor:
        """Separate one CPU 1-D segment, return its music+sfx bed (CPU 1-D)."""
        mixture = segment[None, None, :].to(self.device)
        try:
            with torch.inference_mode():
                out = self.handler(mixture, self.model)
            est = out["estimates"]
            bg = (
                est["music"]["audio"][0].reshape(-1).detach().cpu().float()
                + est["sfx"]["audio"][0].reshape(-1).detach().cpu().float()
            )
        finally:
            del mixture
            if self.device == "cuda":
                torch.cuda.empty_cache()
        return bg

    def _separate_blocked(self, wav: torch.Tensor, stop_event: Event | None = None) -> torch.Tensor:
        n = wav.numel()
        block = max(1, int(self.block_seconds * FS))
        ov = max(0, int(self.overlap_seconds * FS))
        step = max(1, block - ov)

        # Precompute block starts so we can drive one overall progress bar across
        # the whole track (the handler's per-block bar is disabled).
        starts: list[int] = []
        s = 0
        while s < n:
            starts.append(s)
            if s + block >= n:
                break
            s += step

        out = torch.zeros(n, dtype=torch.float32)
        wsum = torch.zeros(n, dtype=torch.float32)
        for s in tqdm(starts, desc="Separating", unit="blk"):
            if stop_event is not None and stop_event.is_set():
                raise RuntimeError("Stopped by user.")
            e = min(n, s + block)
            seg_bg = self._run_handler(wav[s:e])
            w = self._block_weights(e - s, lead=ov if s > 0 else 0, trail=ov if e < n else 0)
            out[s:e] += seg_bg * w
            wsum[s:e] += w
        return out / wsum.clamp_min(1e-6)

    @staticmethod
    def _block_weights(n: int, lead: int, trail: int) -> torch.Tensor:
        """Trapezoidal cross-fade weights: ramp up over ``lead`` samples, ramp
        down over ``trail`` samples; consecutive blocks' ramps sum to 1."""
        w = torch.ones(n, dtype=torch.float32)
        lead = min(lead, n)
        trail = min(trail, n)
        if lead > 0:
            w[:lead] = torch.linspace(0.0, 1.0, lead, dtype=torch.float32)
        if trail > 0:
            w[n - trail :] = torch.linspace(1.0, 0.0, trail, dtype=torch.float32)
        return w

    def close(self) -> None:
        try:
            del self.model
            del self.handler
        except Exception:
            pass
        try:
            if self.device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass
