"""WhisperX transcription + word-level alignment for tool_clonevoice.

Only the transcription and alignment stages are used here — both are
token-free (faster-whisper backbone + wav2vec2 aligner). Diarization lives in
``diarize.py``.
"""
from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

LogCallback = Callable[[str], None]

SAMPLE_RATE = 16000


@contextlib.contextmanager
def torch_load_compat():
    """Temporarily force ``torch.load(weights_only=False)``.

    PyTorch 2.6+ defaults ``weights_only=True``, which rejects the omegaconf
    globals (ListConfig/DictConfig/...) stored in the pyannote/whisperx VAD and
    diarization checkpoints. Those are our bundled, trusted files, so relax the
    default for the duration of model loading only, then restore it so the rest
    of the app keeps PyTorch's safe default.
    """
    import torch

    original = torch.load

    def _patched(*args, **kwargs):
        # Force False even when callers pass weights_only=None/True explicitly
        # (pyannote -> lightning _load passes weights_only=None for local files,
        # which torch 2.6+ treats as True). setdefault would not override that.
        kwargs["weights_only"] = False
        return original(*args, **kwargs)

    torch.load = _patched
    try:
        yield
    finally:
        torch.load = original

# UI model key -> local faster-whisper directory name under models/.
MODEL_DIR_NAMES = {
    "large-v3": "faster-whisper-large-v3",
    "large-v2": "faster-whisper-large-v2",
    "kotoba": "kotoba-whisper-v2.0-faster",
}

MODEL_REPO_IDS = {
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v2": "Systran/faster-whisper-large-v2",
    "kotoba": "kotoba-tech/kotoba-whisper-v2.0-faster",
}

MODEL_ESTIMATED_BYTES = {
    "large-v3": 3_090_837_754,
    "large-v2": 3_089_121_016,
    "kotoba": 1_516_480_672,
}

# Language code -> shallow local wav2vec2 align model dir name under models/whisperx/.
# Kept flat (no HF cache layout) so non-technical users can drop downloaded files in.
ALIGN_HF_DIR_NAMES = {
    "ja": "wav2vec2-large-xlsr-53-japanese",
    "zh": "wav2vec2-large-xlsr-53-chinese-zh-cn",
}


def _resolve_align_args(language_code: str, align_model_dir: Optional[str]):
    """Return (model_name, model_dir) for whisperx.load_align_model.

    For HF-aligned languages, prefer a shallow local model directory
    (models/whisperx/<name>) loaded directly via ``model_name`` so the on-disk
    layout stays flat. The English (torchaudio) model is a single flat file and
    just uses ``model_dir``.
    """
    if not align_model_dir:
        return None, None
    name = ALIGN_HF_DIR_NAMES.get(language_code)
    if name:
        local = Path(align_model_dir) / name
        if (local / "config.json").exists():
            return str(local), align_model_dir
    return None, align_model_dir


def _build_startupinfo():
    if sys.platform != "win32":
        return None
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0
    return startupinfo


def resolve_device() -> tuple[str, str]:
    """Torch device (for wav2vec2 align + pyannote diarize). Prefers CUDA.

    Torch ships its own cuDNN 9, so GPU here is safe whenever CUDA is available.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def _ctranslate2_cuda_usable() -> bool:
    """Whether faster-whisper (CTranslate2) can safely run on CUDA here.

    CTranslate2 >= 4.5 links cuDNN 9 (cudnn_ops64_9.dll / cudnn_cnn64_9.dll),
    which torch 2.8 bundles in torch/lib. Older 4.4 links cuDNN 8, which torch
    does NOT ship. When the required cuDNN DLLs are missing, CTranslate2 does not
    raise a catchable error — it hard-crashes the process (0xc0000409). So we
    probe for the matching cuDNN DLLs and fall back to CPU when absent.
    """
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() <= 0:
            return False
        major_minor = tuple(int(x) for x in ctranslate2.__version__.split(".")[:2])
    except Exception:
        return False

    if sys.platform != "win32":
        # On Linux the cuDNN .so usually comes from torch or the nvidia wheels and
        # is resolvable via the dynamic loader; trust the CUDA device count.
        return True

    import os

    try:
        # Importing torch puts torch/lib (with its bundled cuDNN) on the DLL path.
        import torch

        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    except Exception:
        return False

    if major_minor >= (4, 5):
        needed = ("cudnn_ops64_9.dll", "cudnn_cnn64_9.dll")
    else:
        needed = ("cudnn_ops_infer64_8.dll", "cudnn_cnn_infer64_8.dll")
    return all(os.path.exists(os.path.join(torch_lib, name)) for name in needed)


def resolve_asr_device() -> tuple[str, str]:
    """CTranslate2 device for transcription. CUDA only if cuDNN 8 DLLs exist."""
    if _ctranslate2_cuda_usable():
        return "cuda", "float16"
    return "cpu", "int8"


def resolve_model_arg(model_key: str, models_root: str) -> str:
    """Resolve a UI model key to a local model dir path, else the bare key.

    faster-whisper's ``WhisperModel`` accepts a local CTranslate2 directory
    directly, so a bundled ``models/faster-whisper-large-v3`` is used without
    any re-download. If the local dir is missing, the bare key is returned so
    whisperx/faster-whisper downloads it on demand.
    """
    dirname = MODEL_DIR_NAMES.get(model_key, model_key)
    local = Path(models_root) / dirname
    if (local / "model.bin").exists():
        return str(local)
    return model_key


def model_dir(model_key: str, models_root: str) -> Path:
    dirname = MODEL_DIR_NAMES.get(model_key, model_key)
    return Path(models_root) / dirname


def check_model_files(model_key: str, models_root: str) -> bool:
    local = model_dir(model_key, models_root)
    return (local / "model.bin").is_file() and (local / "config.json").is_file()


def remote_file_plan(model_key: str, log: LogCallback = print) -> tuple[list[tuple[str, int | None]], int | None]:
    repo_id = MODEL_REPO_IDS.get(model_key)
    if not repo_id:
        log(f"Error: unknown ASR model: {model_key}")
        return [], None
    try:
        from huggingface_hub import HfApi
    except ImportError:
        log("Error: huggingface_hub package is not installed.")
        return [], MODEL_ESTIMATED_BYTES.get(model_key)
    try:
        info = HfApi().model_info(repo_id, files_metadata=True)
    except Exception as exc:
        log(f"Could not query remote file sizes for {repo_id}: {exc}")
        return [], MODEL_ESTIMATED_BYTES.get(model_key)
    files: list[tuple[str, int | None]] = []
    total = 0
    complete = True
    for sibling in info.siblings:
        filename = getattr(sibling, "rfilename", "") or ""
        if not filename or filename.startswith(".cache/"):
            continue
        size = getattr(sibling, "size", None)
        files.append((filename, size))
        if isinstance(size, int):
            total += size
        else:
            complete = False
    return files, total if complete else MODEL_ESTIMATED_BYTES.get(model_key)


def download_model(model_key: str, models_root: str, log: LogCallback = print) -> bool:
    repo_id = MODEL_REPO_IDS.get(model_key)
    if not repo_id:
        log(f"Error: unknown ASR model: {model_key}")
        return False

    local_dir = model_dir(model_key, models_root)
    local_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(Path(models_root) / ".hf_home"))
    try:
        from huggingface_hub import hf_hub_download
        import huggingface_hub.constants
    except ImportError:
        log("Error: huggingface_hub package is not installed.")
        return False

    try:
        files, _total = remote_file_plan(model_key, log)
        filenames = [name for name, _size in files]
        if not filenames:
            from huggingface_hub import list_repo_files
            filenames = list_repo_files(repo_id)
        log(f"HuggingFace Endpoint: {huggingface_hub.constants.ENDPOINT}")
        log(f"Downloading {repo_id} to {local_dir}")
        for filename in filenames:
            if filename.endswith("/"):
                continue
            log(f"Downloading file: {filename}")
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(local_dir),
            )
        log("ASR model download finished.")
        return check_model_files(model_key, models_root)
    except Exception as exc:
        log(f"Download failed: {exc}")
        return False


def extract_audio_16k(
    video_path: str,
    out_wav: str,
    log: LogCallback = print,
    stop_event=None,
) -> str:
    """Decode the video's audio to 16kHz mono PCM WAV via ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg not found on PATH.")
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_s16le", str(out_wav),
    ]
    log(f"[extract] {Path(video_path).name} -> 16kHz mono wav")
    proc = subprocess.run(
        cmd, capture_output=True, text=True, errors="replace",
        startupinfo=_build_startupinfo(),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {(proc.stderr or '').strip()}")
    return str(out_wav)


def transcribe(
    audio,
    model_arg: str,
    device: str,
    compute_type: str,
    language: Optional[str],
    log: LogCallback = print,
    batch_size: int = 8,
    model_holder: Optional[list] = None,
    beam_size: int = 5,
) -> dict:
    """Transcribe with faster-whisper directly, with built-in word timestamps.

    Returns ``{'segments': [{start, end, text, words:[{word,start,end}]}], 'language'}``.

    We use faster-whisper's native sentence-level segmentation + word timestamps
    rather than whisperx's VAD-merged segments: whisperx tends to merge a whole
    speech region into one long segment (collapsing multi-speaker dialogue and
    breaking the wav2vec2 word alignment). faster-whisper keeps natural
    phrase-level segments, each cleanly assignable to a diarized speaker.
    """
    import torch  # noqa: F401  -- ensure torch/lib (cuDNN) is on the DLL path first
    from faster_whisper import WhisperModel

    log(f"[transcribe] loading model ({device}/{compute_type})")
    model = WhisperModel(model_arg, device=device, compute_type=compute_type)
    if model_holder is not None:
        model_holder.append(model)

    segment_iter, info = model.transcribe(
        audio,
        language=language or None,
        task="transcribe",
        beam_size=beam_size,
        word_timestamps=True,
        vad_filter=True,
    )
    segments = []
    for seg in segment_iter:
        words = [
            {"word": w.word, "start": w.start, "end": w.end}
            for w in (seg.words or [])
            if w.start is not None and w.end is not None
        ]
        segments.append(
            {"start": seg.start, "end": seg.end, "text": seg.text, "words": words}
        )
    log(f"[transcribe] {len(segments)} segments, lang={info.language}")
    return {"segments": segments, "language": info.language}


def align(
    segments: list,
    language_code: str,
    audio,
    device: str,
    log: LogCallback = print,
    model_holder: Optional[list] = None,
    align_model_dir: Optional[str] = None,
) -> dict:
    """Word-level forced alignment. Returns {'segments': [... with 'words'], ...}.

    ``align_model_dir`` is passed through to whisperx as ``model_dir`` so the
    wav2vec2 aligner loads from a bundled directory (models/whisperx) instead of
    re-downloading: HF models cache there, the torchaudio English model lands
    there as a flat file. See models/whisperx/get_whisperx.txt.
    """
    import whisperx

    model_name, model_dir = _resolve_align_args(language_code, align_model_dir)
    log(f"[align] loading aligner for '{language_code}'" + (f" from {model_name}" if model_name else ""))
    align_model, metadata = whisperx.load_align_model(
        language_code=language_code, device=device, model_name=model_name, model_dir=model_dir
    )
    if model_holder is not None:
        model_holder.append(align_model)
    result = whisperx.align(
        segments, align_model, metadata, audio, device, return_char_alignments=False
    )
    return result
