"""Speaker diarization backends for tool_clonevoice_v2 (pluggable).

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
import warnings
import wave
import shutil
import gc
from pathlib import Path
from typing import Callable, List, Optional, Tuple

# pyannote.audio.core.io warns at import time when torchcodec can't be loaded.
# torchcodec is deliberately excluded from the packaged build (we always feed an
# in-memory waveform, never pyannote's file decoder), so the warning is pure
# noise. Silence it before pyannote is imported anywhere below.
warnings.filterwarnings(
    "ignore",
    message=r"(?s).*torchcodec is not installed correctly.*",
    category=UserWarning,
)
# pyannote disables TF32 on import (a reproducibility/accuracy trade-off it makes
# deliberately) and logs a ReproducibilityWarning each run. We accept that choice,
# so silence the notice. Matched by message because the custom warning category
# lives in pyannote and need not be imported here.
warnings.filterwarnings(
    "ignore",
    message=r"(?s).*TensorFloat-32 \(TF32\) has been disabled.*",
)
# pyannote's statistics pooling computes std() over very short (often 1-frame)
# speaker-embedding windows, which torch warns about ("degrees of freedom is
# <= 0"). The std of a single frame is just 0 here and does not affect turns.
warnings.filterwarnings(
    "ignore",
    message=r"(?s).*std\(\): degrees of freedom is <= 0.*",
    category=UserWarning,
)

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


def speaker_centroids(
    audio16k_path: str,
    turns: List[Turn],
    *,
    models_root: str,
    device: str,
    max_windows_per_speaker: int = 12,
    min_window: float = 1.0,
    log: LogCallback = print,
):
    """One averaged voice embedding per speaker label in ``turns``.

    Used to line up the labels two separately-diarized parts of the same title
    gave the same person. Returns ``{speaker: unit-norm vector}``, empty if the
    ECAPA bundle is not installed.
    """
    import numpy as np

    if not ecapa_available(models_root):
        return {}
    by_speaker: dict[str, List[tuple[float, float]]] = {}
    for start, end, speaker in turns:
        if float(end) - float(start) >= min_window:
            by_speaker.setdefault(str(speaker), []).append((float(start), float(end)))
    if not by_speaker:
        return {}
    # Longest turns first: the cleanest look at the voice, and a bounded number
    # of them keeps this to seconds even on a long part.
    windows: List[tuple[float, float]] = []
    index: List[str] = []
    for speaker, spans in sorted(by_speaker.items()):
        spans.sort(key=lambda span: span[0] - span[1])
        for span in spans[:max_windows_per_speaker]:
            windows.append(span)
            index.append(speaker)
    if not windows:
        return {}
    audio, sr = _read_wav_mono(audio16k_path)
    model = None
    try:
        model = _load_ecapa_model(models_root, device, log)
        embeddings = _extract_ecapa_embeddings(
            model, audio, sr, windows, device, lambda _m: None
        )
    except Exception as exc:
        # The bundle's files can be on disk while the code that loads them is
        # not: its hubconf imports s3prl, which a build may have left out. Voice
        # matching only refines a split the diarizer already made, so lose that
        # refinement rather than the whole transcription run.
        log(f"[diarize] voice embeddings unavailable ({exc}); continuing without them")
        return {}
    finally:
        del model
    centroids: dict[str, "np.ndarray"] = {}
    for speaker in sorted(set(index)):
        rows = np.asarray(
            [embeddings[i] for i, name in enumerate(index) if name == speaker],
            dtype=np.float32,
        )
        mean = rows.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        if norm > 0:
            centroids[speaker] = mean / norm
    log(f"[diarize] voice centroids for {len(centroids)} speaker(s) from {len(windows)} window(s)")
    return centroids


def match_speakers_across_parts(centroids_by_part: list[dict], *, min_similarity: float = 0.55):
    """Give the same person the same label in every part of a title.

    Returns one ``{local_label: global_label}`` per part. Greedy nearest match
    on cosine similarity; anything below ``min_similarity`` is treated as a
    voice that part does not share and gets a label of its own, because a wrong
    merge sends one person's lines out in another person's voice.
    """
    import numpy as np

    globals_: list[tuple[str, "np.ndarray"]] = []
    mappings: list[dict[str, str]] = []
    for centroids in centroids_by_part:
        mapping: dict[str, str] = {}
        taken: set[str] = set()
        for speaker, vector in sorted(centroids.items()):
            best_name, best_score = None, -1.0
            for name, reference in globals_:
                if name in taken:
                    continue
                score = float(np.dot(vector, reference))
                if score > best_score:
                    best_name, best_score = name, score
            if best_name is not None and best_score >= min_similarity:
                mapping[speaker] = best_name
                taken.add(best_name)
            else:
                name = f"SPEAKER_{len(globals_):02d}"
                globals_.append((name, vector))
                mapping[speaker] = name
                taken.add(name)
        mappings.append(mapping)
    return mappings


# Over-splitting on purpose is what makes an odd voice recognisable, and it is
# free: pyannote's work does not depend on the count it is asked for. Measured
# on 3dsvr-1911 part 1, whose 11.60-18.34 is a man among a woman's 634s. Asked
# for 2 clusters his came out at 39.7s and cos 0.357 against hers, still
# carrying her voice; asked for 3 or 4 it came out at 26.8s/28.1s and cos
# 0.111/0.113. Sub-clusters of one person rejoin each other, so the headroom
# costs nothing when the extra voices are not there.
SPEAKER_SPLIT_HEADROOM = 2
MAX_SPLIT_CLUSTERS = 10

# Where "the same person over-split" stops and "someone else" starts. On the
# title above, sub-clusters of the one woman scored 0.689-0.887 against each
# other, within a part and across parts, while the man scored 0.092-0.144
# against everyone once his cluster was clean. 0.45 sits between the two bands
# with room on either side.
MERGE_INTO_PRIMARY_SIMILARITY = 0.45

# Collapsing two of the voices the caller asked for needs better evidence than
# absorbing a leftover cluster into one of them, so it gets its own threshold
# at the bottom of the measured same-person band rather than in the gap. Asked
# for 2 on the title above, the two longest clusters in part 1 were both the
# woman at cos 0.794; being asked for the pair is not a reason to clone her
# twice. The pairs that sit in the ambiguous middle, 0.46-0.56, stay apart --
# there the caller's own count is the better evidence.
SAME_VOICE_SIMILARITY = 0.65


def diarize_primary_speakers(
    audio16k_path: str,
    *,
    backend: str = "auto",
    num_speakers: Optional[int],
    models_root: str,
    device: str = "cpu",
    headroom: int = SPEAKER_SPLIT_HEADROOM,
    merge_similarity: float = MERGE_INTO_PRIMARY_SIMILARITY,
    same_voice_similarity: float = SAME_VOICE_SIMILARITY,
    log: LogCallback = print,
) -> List[Turn]:
    """Diarize for ``num_speakers`` voices to clone, keeping distinct extras.

    ``num_speakers`` says how many voices the caller wants held together, not
    how many the recording holds. Handing that straight to the diarizer as an
    exact cluster count answers a different question, and answers it badly at
    both ends. At 1 every voice in the title lands in one bucket, so a man's
    lines are cloned from the woman's anchor and her bucket, polluted with his
    voice, no longer matches her own other parts. At 2 the two longest clusters
    are taken as two people without ever being compared, so a title with one
    woman and one man came back with the woman cloned as two different voices.

    So ask the diarizer for more clusters than that, and walk them longest
    first. A cluster that clearly is one of the voices already taken on rejoins
    it; otherwise it becomes one of them, until ``num_speakers`` of them are
    held. Everything after that either rejoins a held voice or, if it resembles
    none of them, keeps a label of its own. The count is therefore a bound on
    how many voices are held together rather than a number to hit, and a
    clearly different voice is still separated out to get its own anchor.

    Labels come back numbered by speaking time, so ``SPEAKER_00`` is the voice
    the title is mostly made of.

    Without the ECAPA bundle there is nothing to merge on, so this falls back
    to a plain exact-count diarization.
    """
    import numpy as np

    requested = int(num_speakers) if num_speakers else 0
    if requested <= 0:
        return diarize(audio16k_path, backend=backend, num_speakers=None,
                       models_root=models_root, device=device, log=log)

    if resolve_backend(backend, models_root) != "pyannote" or not ecapa_available(models_root):
        log(f"[diarize] no voice embeddings available; asking for exactly {requested} speaker(s)")
        return diarize(audio16k_path, backend=backend, num_speakers=requested,
                       models_root=models_root, device=device, log=log)

    split_k = min(requested + max(0, int(headroom)), MAX_SPLIT_CLUSTERS)
    log(f"[diarize] splitting into {split_k} cluster(s) to keep {requested} voice(s) together")
    turns = diarize(audio16k_path, backend=backend, num_speakers=split_k,
                    models_root=models_root, device=device, log=log)

    seconds: dict[str, float] = {}
    for start, end, speaker in turns:
        seconds[str(speaker)] = seconds.get(str(speaker), 0.0) + max(0.0, float(end) - float(start))
    if len(seconds) <= requested:
        return turns

    centroids = speaker_centroids(audio16k_path, turns, models_root=models_root,
                                  device=device, log=log)
    if not centroids:
        log("[diarize] no voice centroids; keeping the split as the diarizer left it")
        return turns

    ordered = sorted(seconds, key=lambda name: (-seconds[name], name))
    if not any(name in centroids for name in ordered):
        log("[diarize] no centroid for any cluster; keeping the split as it is")
        return turns

    mapping: dict[str, str] = {}
    primaries: list[str] = []
    merged: list[str] = []
    kept: list[str] = []
    for name in ordered:
        vector = centroids.get(name)
        best_name, best_score = None, -1.0
        if vector is not None:
            for primary in primaries:
                score = float(np.dot(vector, centroids[primary]))
                if score > best_score:
                    best_name, best_score = primary, score
        # While voices are still being taken on, the caller has asked for this
        # many, so only collapse two of them on strong evidence of one person.
        # Once the count is used up the choice is between rejoining and adding
        # a voice nobody asked for, and a weaker match settles that.
        threshold = same_voice_similarity if len(primaries) < requested else merge_similarity
        if best_name is not None and best_score >= threshold:
            mapping[name] = best_name
            merged.append(f"{name}({seconds[name]:.0f}s)->{best_name} cos {best_score:.2f}")
        elif vector is not None and len(primaries) < requested:
            primaries.append(name)
            mapping[name] = name
        else:
            mapping[name] = name
            reason = "no embedding" if vector is None else f"cos {best_score:.2f}"
            kept.append(f"{name}({seconds[name]:.0f}s, {reason})")

    grouped: dict[str, float] = {}
    for name, target in mapping.items():
        grouped[target] = grouped.get(target, 0.0) + seconds.get(name, 0.0)
    rename = {
        target: f"SPEAKER_{index:02d}"
        for index, target in enumerate(sorted(grouped, key=lambda name: (-grouped[name], name)))
    }
    if merged:
        log("[diarize] rejoined as one voice: " + ", ".join(merged))
    if kept:
        log("[diarize] kept apart as its own voice: " + ", ".join(kept))
    log(
        f"[diarize] {len(grouped)} voice(s) out of {len(seconds)} cluster(s): "
        + ", ".join(
            f"{rename[target]}={grouped[target]:.0f}s"
            for target in sorted(grouped, key=lambda name: rename[name])
        )
    )
    return [
        (float(start), float(end), rename[mapping[str(speaker)]])
        for start, end, speaker in turns
    ]


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


# A conversational turn runs about a second; the diarizer also emits much
# shorter fragments, and on a two-person title it collected 126 turns averaging
# 0.50s and 68 averaging 0.31s into two extra "speakers" beside the real pair
# (1.37s and 1.07s means). Those fragment clusters are backchannels, breaths and
# overlap, and letting them vote produced a false split inside a 2.16s line that
# held seven turns, three of them under 0.1s.
SUBSTANTIAL_SPEAKER_MIN_MEAN_TURN = 0.7
# The share of total speech used to matter here too, at 10%, and that is what
# actually discarded the man on 3dsvr-1911: he introduces the new hire in the
# opening minute and says nothing after, 26.3s of 673.5s -- 3.9%. His turns
# average 0.73s, so he speaks in sentences; he is simply brief. Dropping him
# meant no line was ever cut at his boundary and his dialogue reached the
# synthesizer in the woman's voice.
#
# Share was the wrong measure. What distinguishes a pile of backchannels and
# breaths is that every piece is short -- the clusters this rule was written
# for average 0.31-0.65s -- not that they add up to little. Mean turn length
# alone still rejects every one of them across the 39 clusters in the existing
# manifests. These two only guard against a cluster of one or two stray turns
# whose mean happens to look conversational.
SUBSTANTIAL_SPEAKER_MIN_SECONDS = 5.0
SUBSTANTIAL_SPEAKER_MIN_TURNS = 3
# A speaker change is only believed once this much of the line has moved with
# it; below that it is diarizer jitter over a single speaker.
TURN_SPLIT_MIN_RUN_WORDS = 3
TURN_SPLIT_MIN_RUN_SECONDS = 0.8


def substantial_speakers(
    turns: List[Turn],
    *,
    min_mean_turn: float = SUBSTANTIAL_SPEAKER_MIN_MEAN_TURN,
    min_seconds: float = SUBSTANTIAL_SPEAKER_MIN_SECONDS,
    min_turns: int = SUBSTANTIAL_SPEAKER_MIN_TURNS,
) -> set:
    """Labels that behave like a person rather than a pile of fragments.

    Judged on how the speaker talks, not on how much of the video they hold: a
    brief part is still a part.
    """
    totals: dict[str, list] = {}
    for start, end, speaker in turns:
        entry = totals.setdefault(speaker, [0, 0.0])
        entry[0] += 1
        entry[1] += max(0.0, float(end) - float(start))
    if not totals:
        return set()
    keep = {
        speaker for speaker, (count, span) in totals.items()
        if count >= min_turns and span >= min_seconds and span / count >= min_mean_turn
    }
    return keep or set(totals)


def _word_speakers(words: List[dict], turns: List[Turn]) -> List[Optional[str]]:
    """Attribute each word to the speaker it overlaps most, filling the gaps."""
    labels: List[Optional[str]] = []
    for word in words:
        start, end = float(word["start"]), float(word["end"])
        best, best_overlap = None, 0.0
        for turn_start, turn_end, speaker in turns:
            overlap = min(end, turn_end) - max(start, turn_start)
            if overlap > best_overlap:
                best_overlap, best = overlap, speaker
        labels.append(best)
    # A word inside a diarization gap belongs with whoever was last speaking.
    previous = None
    for index, label in enumerate(labels):
        if label:
            previous = label
        else:
            labels[index] = previous
    following = None
    for index in range(len(labels) - 1, -1, -1):
        if labels[index]:
            following = labels[index]
        else:
            labels[index] = following
    return labels


def _smooth_runs(labels: List[str], words: List[dict],
                 min_run_words: int, min_run_seconds: float) -> List[str]:
    """Absorb runs too short to be a real turn into their neighbour."""
    out = list(labels)
    index = 0
    while index < len(out):
        end = index
        while end + 1 < len(out) and out[end + 1] == out[index]:
            end += 1
        span = float(words[end]["end"]) - float(words[index]["start"])
        undersized = (end - index + 1) < min_run_words or span < min_run_seconds
        if undersized and (index > 0 or end + 1 < len(out)):
            filler = out[index - 1] if index > 0 else out[end + 1]
            for position in range(index, end + 1):
                out[position] = filler
            index = 0  # neighbours may now merge into one another
            continue
        index = end + 1
    return out


def split_segments_by_turns(
    segments: List[dict],
    turns: List[Turn],
    *,
    min_run_words: int = TURN_SPLIT_MIN_RUN_WORDS,
    min_run_seconds: float = TURN_SPLIT_MIN_RUN_SECONDS,
    log: LogCallback = print,
) -> List[dict]:
    """Cut lines that carry more than one speaker at the speaker change.

    :func:`assign_speakers` only tags a line with whoever dominates it, so a
    question and its answer reach the synthesizer as one inference and come back
    in one voice — on a two-person title that was 23% of the lines. Words carry
    their own timestamps, so each word is attributed instead and the line cut
    where the attribution changes, rebuilding each side's text from its words.
    """
    if not turns or not segments:
        return segments
    keep = substantial_speakers(turns)
    dropped = {speaker for _s, _e, speaker in turns} - keep
    if dropped:
        log(f"[diarize] ignoring fragment cluster(s) {sorted(dropped)} when splitting")
    usable = [turn for turn in turns if turn[2] in keep]
    if not usable:
        return segments

    result: List[dict] = []
    split_lines = 0
    for segment in segments:
        words = segment.get("words") or []
        if len(words) < max(2, min_run_words):
            result.append(segment)
            continue
        labels = _word_speakers(words, usable)
        if not any(labels):
            result.append(segment)
            continue
        labels = _smooth_runs([str(x) for x in labels], words,
                              min_run_words, min_run_seconds)
        if len(set(labels)) < 2:
            result.append(segment)
            continue
        start = 0
        for index in range(1, len(words) + 1):
            if index == len(words) or labels[index] != labels[start]:
                group = words[start:index]
                piece = dict(segment)
                piece["start"] = round(float(group[0]["start"]), 3)
                piece["end"] = round(float(group[-1]["end"]), 3)
                piece["dur"] = round(piece["end"] - piece["start"], 3)
                piece["src_text"] = "".join(w.get("w", "") for w in group).strip()
                piece["words"] = group
                piece["speaker"] = labels[start]
                # Any translation on the parent described the whole line.
                piece["tgt_text"] = ""
                result.append(piece)
                start = index
        split_lines += 1

    if split_lines:
        log(
            f"[diarize] split {split_lines} line(s) at speaker changes; "
            f"{len(segments)} -> {len(result)} lines"
        )
    for index, segment in enumerate(result, start=1):
        segment["id"] = index
        segment["srt_index"] = index
    return result


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
