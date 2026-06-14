from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

LogCallback = Callable[[str], None]


@dataclass(frozen=True)
class ModelDownloadSpec:
    key: str
    label: str
    repo_id: str
    local_dir_name: str
    required_files: tuple[str, ...]
    allow_patterns: tuple[str, ...] = ("*",)
    estimated_bytes: int | None = None


OMNIVOICE_SPEC = ModelDownloadSpec(
    key="omnivoice",
    label="OmniVoice",
    repo_id="k2-fsa/OmniVoice",
    local_dir_name="OmniVoice",
    required_files=(
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "audio_tokenizer/config.json",
        "audio_tokenizer/model.safetensors",
        "audio_tokenizer/preprocessor_config.json",
    ),
    estimated_bytes=3_267_470_106,
)

ECAPA_SPEC = ModelDownloadSpec(
    key="ecapa",
    label="OmniVoice_ECAPA",
    repo_id="k2-fsa/TTS_eval_models",
    local_dir_name="OmniVoice_ECAPA",
    required_files=(
        "speaker_similarity/wavlm_large_finetune.pth",
        "speaker_similarity/wavlm_large/wavlm_large.pt",
    ),
    allow_patterns=(
        "speaker_similarity/wavlm_large_finetune.pth",
        "speaker_similarity/wavlm_large/wavlm_large.pt",
        "speaker_similarity/hubconf.py",
        "speaker_similarity/wavlm_large/hubconf.py",
    ),
    estimated_bytes=2_563_896_678,
)


def format_bytes(size: int | None) -> str:
    if size is None:
        return "unknown"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{size} B"


def model_dir(models_root: str | os.PathLike[str], spec: ModelDownloadSpec) -> Path:
    return Path(models_root) / spec.local_dir_name


def check_model_files(models_root: str | os.PathLike[str], spec: ModelDownloadSpec) -> bool:
    base = model_dir(models_root, spec)
    return all((base / rel).is_file() for rel in spec.required_files)


def _matches_any(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def remote_file_plan(spec: ModelDownloadSpec, log: LogCallback = print) -> tuple[list[tuple[str, int | None]], int | None]:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        log("Error: huggingface_hub package is not installed.")
        return [], spec.estimated_bytes

    try:
        info = HfApi().model_info(spec.repo_id, files_metadata=True)
    except Exception as exc:
        log(f"Could not query remote file sizes for {spec.repo_id}: {exc}")
        return [], spec.estimated_bytes

    files: list[tuple[str, int | None]] = []
    total = 0
    complete = True
    for sibling in info.siblings:
        filename = getattr(sibling, "rfilename", "") or ""
        if not filename or not _matches_any(filename, spec.allow_patterns):
            continue
        size = getattr(sibling, "size", None)
        files.append((filename, size))
        if isinstance(size, int):
            total += size
        else:
            complete = False
    return files, total if complete else spec.estimated_bytes


def download_model(models_root: str | os.PathLike[str], spec: ModelDownloadSpec, log: LogCallback = print) -> bool:
    local_dir = model_dir(models_root, spec)
    local_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(Path(models_root) / ".hf_home"))
    try:
        from huggingface_hub import hf_hub_download
        import huggingface_hub.constants
    except ImportError:
        log("Error: huggingface_hub package is not installed.")
        return False

    try:
        files, _total = remote_file_plan(spec, log)
        filenames = [name for name, _size in files]
        if not filenames:
            filenames = list(spec.allow_patterns)
        log(f"HuggingFace Endpoint: {huggingface_hub.constants.ENDPOINT}")
        log(f"Downloading {spec.repo_id} to {local_dir}")
        for filename in filenames:
            if any(ch in filename for ch in "*?[]"):
                continue
            log(f"Downloading file: {filename}")
            try:
                hf_hub_download(
                    repo_id=spec.repo_id,
                    filename=filename,
                    local_dir=str(local_dir),
                )
            except Exception as exc:
                if filename.endswith("hubconf.py"):
                    log(f"Optional file skipped: {filename} ({exc})")
                    continue
                raise
        if spec is ECAPA_SPEC:
            parent_hubconf = local_dir / "speaker_similarity" / "hubconf.py"
            child_hubconf = local_dir / "speaker_similarity" / "wavlm_large" / "hubconf.py"
            if not parent_hubconf.is_file() and child_hubconf.is_file():
                parent_hubconf.write_bytes(child_hubconf.read_bytes())
        log(f"{spec.label} download finished.")
        return check_model_files(models_root, spec)
    except Exception as exc:
        log(f"Download failed: {exc}")
        return False
