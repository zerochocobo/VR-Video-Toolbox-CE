"""Speaker diarization backends for tool_clonevoice (pluggable).

A turn is ``(start, end, speaker)``. Backends:
  - ``single``  : whole audio = one speaker (zero deps, always available).
  - ``pyannote``: bundled pyannote community-1 pipeline (pyannote.audio 4.x,
    models/speaker-diarization-community-1). Replaced 3.1, whose clustering
    merged similar/overlapping speakers into one.
  - ``ecapa``   : ECAPA-WavLM embeddings (models/OmniVoice_ECAPA) + clustering.
  - ``auto``    : pick the stable bundled backend (pyannote > single).
"""
from __future__ import annotations

import sys
import wave
import shutil
import gc
from pathlib import Path
from typing import Callable, List, Optional, Tuple

LogCallback = Callable[[str], None]
Turn = Tuple[float, float, str]

PYANNOTE_BUNDLE = "speaker-diarization-community-1"
ECAPA_BUNDLE = "OmniVoice_ECAPA"
ECAPA_AUTO_MAX_SPEAKERS = 5


def pyannote_available(models_root: str) -> bool:
    return (Path(models_root) / PYANNOTE_BUNDLE / "config.yaml").exists()


def ecapa_available(models_root: str) -> bool:
    base = Path(models_root) / ECAPA_BUNDLE / "speaker_similarity"
    return (base / "wavlm_large_finetune.pth").exists() and (base / "wavlm_large" / "wavlm_large.pt").exists()


def resolve_backend(backend: str, models_root: str) -> str:
    if backend and backend != "auto":
        return backend
    if pyannote_available(models_root):
        return "pyannote"
    return "single"


def _wav_duration(path: str) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate() or 1)


def _read_wav_mono(path: str) -> tuple["np.ndarray", int]:
    import numpy as np

    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sampwidth = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sampwidth != 2:
        raise ValueError(f"Expected 16-bit PCM WAV for ECAPA diarization: {path}")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    return audio, sr


def _speech_regions(
    audio,
    sr: int,
    *,
    frame_ms: int = 30,
    min_speech_s: float = 0.45,
    min_silence_s: float = 0.35,
    pad_s: float = 0.15,
) -> List[tuple[float, float]]:
    """Simple adaptive energy VAD for ECAPA windowing.

    The ECAPA bundle gives embeddings, not speech activity. This lightweight VAD
    is intentionally conservative; faster-whisper still decides text segments,
    and these regions only provide speaker turns for overlap assignment.
    """
    import numpy as np

    if audio.size == 0:
        return []
    frame = max(1, int(sr * frame_ms / 1000))
    n = int(np.ceil(audio.size / frame))
    if n <= 0:
        return []
    padded = np.pad(audio, (0, n * frame - audio.size))
    frames = padded.reshape(n, frame)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    if float(np.max(rms)) < 1e-5:
        return []

    floor = float(np.percentile(rms, 25))
    high = float(np.percentile(rms, 85))
    threshold = max(0.006, floor * 2.5, high * 0.18)
    voiced = rms >= threshold

    min_speech = max(1, int(min_speech_s * 1000 / frame_ms))
    min_silence = max(1, int(min_silence_s * 1000 / frame_ms))
    pad_frames = max(0, int(pad_s * 1000 / frame_ms))

    regions: List[tuple[int, int]] = []
    start = None
    silence = 0
    for i, is_voice in enumerate(voiced):
        if is_voice:
            if start is None:
                start = i
            silence = 0
        elif start is not None:
            silence += 1
            if silence >= min_silence:
                end = i - silence + 1
                if end - start >= min_speech:
                    regions.append((max(0, start - pad_frames), min(n, end + pad_frames)))
                start = None
                silence = 0
    if start is not None and n - start >= min_speech:
        regions.append((max(0, start - pad_frames), n))

    if not regions:
        return []

    merged: List[tuple[int, int]] = []
    for s, e in regions:
        if merged and s - merged[-1][1] <= min_silence:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s * frame / sr, min(audio.size / sr, e * frame / sr)) for s, e in merged]


def _window_regions(
    regions: List[tuple[float, float]],
    *,
    window_s: float = 2.4,
    hop_s: float = 1.2,
    min_window_s: float = 1.0,
) -> List[tuple[float, float]]:
    windows: List[tuple[float, float]] = []
    for start, end in regions:
        dur = end - start
        if dur <= 0:
            continue
        if dur <= window_s:
            if dur >= min_window_s:
                windows.append((start, end))
            continue
        t = start
        while t + min_window_s <= end:
            w_end = min(end, t + window_s)
            if w_end - t >= min_window_s:
                windows.append((t, w_end))
            if w_end >= end:
                break
            t += hop_s
    return windows


def _load_ecapa_class(models_root: str):
    try:
        from omnivoice.eval.models.ecapa_tdnn_wavlm import ECAPA_TDNN_WAVLM

        return ECAPA_TDNN_WAVLM
    except Exception:
        # Development-tree fallback: the PyPI dependency may not be installed in
        # the shell used for static checks, while the checked-in reference source
        # has the exact same model definition.
        repo_root = Path(models_root).resolve().parent
        ref_root = repo_root / "reference" / "OmniVoice"
        if ref_root.is_dir() and str(ref_root) not in sys.path:
            sys.path.insert(0, str(ref_root))
        from omnivoice.eval.models.ecapa_tdnn_wavlm import ECAPA_TDNN_WAVLM

        return ECAPA_TDNN_WAVLM


def _load_ecapa_model(models_root: str, device: str, log: LogCallback):
    import torch

    base = Path(models_root) / ECAPA_BUNDLE / "speaker_similarity"
    sv_model_path = base / "wavlm_large_finetune.pth"
    ssl_model_path = base / "wavlm_large"
    if not sv_model_path.is_file() or not (ssl_model_path / "wavlm_large.pt").is_file():
        raise FileNotFoundError(
            f"ECAPA model not found under {Path(models_root) / ECAPA_BUNDLE}. "
            "Expected speaker_similarity/wavlm_large_finetune.pth and "
            "speaker_similarity/wavlm_large/wavlm_large.pt."
        )
    # OmniVoice 0.1.5 calls torch.hub.load(dirname(ssl_model_path), ...), so
    # hubconf.py must be one level above wavlm_large.pt. Some bundled model zips
    # place hubconf.py inside wavlm_large/ instead; copy only that tiny loader.
    parent_hubconf = base / "hubconf.py"
    child_hubconf = ssl_model_path / "hubconf.py"
    if not parent_hubconf.is_file() and child_hubconf.is_file():
        shutil.copyfile(child_hubconf, parent_hubconf)

    ECAPA_TDNN_WAVLM = _load_ecapa_class(models_root)
    log(f"[diarize] loading ECAPA-WavLM ({device})")
    model = ECAPA_TDNN_WAVLM(
        feat_dim=1024,
        channels=512,
        emb_dim=256,
        sr=16000,
        ssl_model_path=str(ssl_model_path),
    )
    try:
        state = torch.load(str(sv_model_path), map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(str(sv_model_path), map_location="cpu")
    model.load_state_dict(state.get("model", state), strict=False)
    model.to(device)
    model.eval()
    return model


def _extract_ecapa_embeddings(model, audio, sr: int, windows: List[tuple[float, float]], device: str, log: LogCallback):
    import numpy as np
    import torch
    import torch.nn.functional as F

    embs = []
    min_samples = int(sr * 1.0)
    with torch.no_grad():
        for idx, (start, end) in enumerate(windows, 1):
            s = max(0, int(start * sr))
            e = min(audio.size, int(end * sr))
            clip = audio[s:e]
            if clip.size < min_samples:
                clip = np.pad(clip, (0, min_samples - clip.size))
            wav = torch.from_numpy(clip.astype(np.float32)).to(device)
            emb = model([wav])
            emb = F.normalize(emb, dim=-1)
            embs.append(emb.detach().cpu().numpy()[0])
            if idx == 1 or idx == len(windows) or idx % 10 == 0:
                log(f"[diarize] ECAPA embeddings {idx}/{len(windows)}")
    return np.asarray(embs, dtype=np.float32)


def _make_agglomerative_clusterer(k: int):
    from sklearn.cluster import AgglomerativeClustering

    try:
        return AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
    except TypeError:
        return AgglomerativeClustering(n_clusters=k, affinity="cosine", linkage="average")


def _estimate_ecapa_speaker_count(embeddings, log: LogCallback) -> int:
    import numpy as np
    from sklearn.metrics import silhouette_score

    n = int(len(embeddings))
    if n <= 2:
        return max(1, n)

    # If all windows are already very close, do not invent multiple speakers.
    sim = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    tri = 1.0 - sim[np.triu_indices(n, k=1)]
    if tri.size and float(np.percentile(tri, 75)) < 0.24:
        log("[diarize] ECAPA auto-k: compact embeddings -> 1 speaker")
        return 1

    max_k = min(ECAPA_AUTO_MAX_SPEAKERS, n - 1)
    best_k = 2
    best_score = -1e9
    candidates = []
    for k in range(2, max_k + 1):
        labels = _make_agglomerative_clusterer(k).fit_predict(embeddings)
        if len(set(int(x) for x in labels)) < 2:
            continue
        try:
            sil = float(silhouette_score(embeddings, labels, metric="cosine"))
        except Exception:
            continue
        # Penalize extra speakers; ECAPA windows often vary by emotion/noise, so
        # raw silhouette tends to over-split real dialogue.
        score = sil - 0.08 * max(0, k - 2)
        candidates.append(f"k={k}:sil={sil:.3f}/score={score:.3f}")
        if score > best_score:
            best_k, best_score = k, score
    if candidates:
        log("[diarize] ECAPA auto-k " + ", ".join(candidates) + f" -> {best_k}")
    return best_k


def _cluster_embeddings(embeddings, num_speakers: Optional[int], log: LogCallback):
    import numpy as np

    n = int(len(embeddings))
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    if n == 1:
        return np.zeros(1, dtype=np.int32)
    if num_speakers is not None and num_speakers > 0:
        k = max(1, min(int(num_speakers), n))
        log(f"[diarize] ECAPA speaker count forced: {k}")
    else:
        k = _estimate_ecapa_speaker_count(embeddings, log)
    labels = _make_agglomerative_clusterer(k).fit_predict(embeddings)
    unique = sorted(set(int(x) for x in labels))
    remap = {old: i for i, old in enumerate(unique)}
    out = np.asarray([remap[int(x)] for x in labels], dtype=np.int32)
    log(f"[diarize] ECAPA clustered {n} windows -> {len(unique)} speaker(s)")
    return out


def _labels_to_turns(windows: List[tuple[float, float]], labels, merge_gap_s: float = 0.45) -> List[Turn]:
    turns: List[Turn] = []
    for (start, end), label in sorted(zip(windows, labels), key=lambda x: x[0][0]):
        spk = f"SPEAKER_{int(label):02d}"
        if turns and turns[-1][2] == spk and start - turns[-1][1] <= merge_gap_s:
            turns[-1] = (turns[-1][0], max(turns[-1][1], end), spk)
        else:
            turns.append((float(start), float(end), spk))
    return turns


def diarize(
    audio16k_path: str,
    *,
    backend: str = "auto",
    num_speakers: Optional[int] = None,
    models_root: str,
    device: str = "cpu",
    log: LogCallback = print,
) -> List[Turn]:
    resolved = resolve_backend(backend, models_root)
    if resolved != backend:
        log(f"[diarize] backend '{backend}' -> '{resolved}'")
    else:
        log(f"[diarize] backend '{resolved}'")

    if resolved == "single":
        return [(0.0, _wav_duration(audio16k_path), "SPEAKER_00")]
    if resolved == "pyannote":
        return _diarize_pyannote(audio16k_path, num_speakers, models_root, device, log)
    if resolved == "ecapa":
        return _diarize_ecapa(audio16k_path, num_speakers, models_root, device, log)
    raise ValueError(f"Unknown diarization backend: {resolved}")


def _patch_speechbrain_lazy_dunder() -> None:
    """Stop speechbrain's lazy integration modules from breaking inspect.stack().

    speechbrain registers ``LazyModule`` objects (e.g. integrations.k2_fsa,
    integrations.nlp) in ``sys.modules``. When pyannote loads a model,
    pytorch_lightning calls ``inspect.stack()``, whose ``getmodule`` does
    ``hasattr(module, '__file__')`` on every sys.modules entry. For a LazyModule
    that resolves a dunder via ``__getattr__``, this triggers a real import of an
    optional, uninstalled dependency, and the LazyModule raises ImportError (not
    AttributeError), aborting diarization.

    We never use those optional integrations, so patch ``LazyModule.__getattr__``
    to raise AttributeError for dunder names (``__file__`` etc.). ``hasattr`` then
    returns False and the import is never triggered; real attribute access still
    works as before.
    """
    try:
        from speechbrain.utils import importutils as _iu
    except Exception:
        return
    lazy_cls = getattr(_iu, "LazyModule", None)
    if lazy_cls is None or getattr(lazy_cls, "_clonevoice_dunder_patched", False):
        return
    original_getattr = lazy_cls.__getattr__

    def _safe_getattr(self, attr):
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        return original_getattr(self, attr)

    lazy_cls.__getattr__ = _safe_getattr
    lazy_cls._clonevoice_dunder_patched = True


def _diarize_pyannote(
    audio16k_path: str,
    num_speakers: Optional[int],
    models_root: str,
    device: str,
    log: LogCallback,
) -> List[Turn]:
    import numpy as np
    import torch
    from pyannote.audio import Pipeline

    _patch_speechbrain_lazy_dunder()

    config_path = str(Path(models_root) / PYANNOTE_BUNDLE / "config.yaml")
    log(f"[diarize] loading community-1 pipeline ({device})")
    # community-1's config uses "$model/..." placeholders that pyannote 4.x
    # resolves against the config file's own directory, so the bundled segmentation/
    # embedding/plda load with no path rewriting and no auth token.
    pipe = None
    waveform = None
    out = None
    try:
        pipe = Pipeline.from_pretrained(config_path)
        pipe.to(torch.device(device))

        # Feed a waveform tensor instead of a path: pyannote 4.x decodes files via
        # torchcodec, whose native DLL fails to load on Windows. The waveform path
        # skips torchcodec entirely.
        audio, sr = _read_wav_mono(audio16k_path)
        waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)  # (1, T)
        out = pipe({"waveform": waveform, "sample_rate": sr}, num_speakers=num_speakers or None)

        # pyannote 4.x returns a DiarizeOutput; the Annotation is .speaker_diarization.
        ann = getattr(out, "speaker_diarization", out)
        turns: List[Turn] = [
            (float(seg.start), float(seg.end), str(spk))
            for seg, _, spk in ann.itertracks(yield_label=True)
        ]
        log(f"[diarize] {len(turns)} turns, {len({t[2] for t in turns})} speakers")
        return turns
    finally:
        del out, waveform, pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass


def _diarize_ecapa(
    audio16k_path: str,
    num_speakers: Optional[int],
    models_root: str,
    device: str,
    log: LogCallback,
) -> List[Turn]:
    audio, sr = _read_wav_mono(audio16k_path)
    if sr != 16000:
        raise ValueError(f"ECAPA diarization expects 16kHz audio, got {sr}: {audio16k_path}")

    regions = _speech_regions(audio, sr)
    if not regions:
        log("[diarize] ECAPA VAD found no speech; falling back to single speaker.")
        return [(0.0, audio.size / float(sr or 1), "SPEAKER_00")]
    windows = _window_regions(regions)
    if not windows:
        log("[diarize] ECAPA VAD windows are too short; falling back to single speaker.")
        return [(0.0, audio.size / float(sr or 1), "SPEAKER_00")]

    log(f"[diarize] ECAPA VAD: {len(regions)} speech region(s), {len(windows)} embedding window(s)")
    model = None
    try:
        model = _load_ecapa_model(models_root, device, log)
        embeddings = _extract_ecapa_embeddings(model, audio, sr, windows, device, log)
        labels = _cluster_embeddings(embeddings, num_speakers, log)
        turns = _labels_to_turns(windows, labels)
        log(f"[diarize] ECAPA {len(turns)} turns, {len({t[2] for t in turns})} speaker(s)")
        return turns
    finally:
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass


def assign_speakers(segments: List[dict], turns: List[Turn]) -> List[dict]:
    """Tag each transcript segment with the speaker of maximum time overlap."""
    if not turns:
        for seg in segments:
            seg["speaker"] = "SPEAKER_00"
        return segments
    for seg in segments:
        s, e = float(seg["start"]), float(seg["end"])
        best_speaker, best_overlap = turns[0][2], 0.0
        for ts, te, spk in turns:
            overlap = min(e, te) - max(s, ts)
            if overlap > best_overlap:
                best_overlap, best_speaker = overlap, spk
        dur = max(1e-3, e - s)
        seg["speaker"] = best_speaker
        seg["speaker_overlap"] = round(best_overlap, 3)
        seg["speaker_overlap_ratio"] = round(best_overlap / dur, 3)
    return segments
