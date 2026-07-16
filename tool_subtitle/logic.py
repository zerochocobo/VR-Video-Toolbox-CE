import os
import gc
import re
import sys
import time
import subprocess
import json
import shutil
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from utils import app_config

# Set mirror for Chinese locale users BEFORE importing any libraries
# that might cache the huggingface endpoint (like faster_whisper or huggingface_hub)
import locale
try:
    if app_config.get_language() == 'zh':
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
except Exception:
    pass

FFmpeg = None
WhisperModel = None
decode_audio = None
FeatureExtractor = None


def _ensure_ffmpy3():
    global FFmpeg
    if FFmpeg is None:
        from ffmpy3 import FFmpeg as _FFmpeg
        FFmpeg = _FFmpeg
    return FFmpeg


def _ensure_faster_whisper():
    global WhisperModel, decode_audio, FeatureExtractor
    if WhisperModel is None or decode_audio is None or FeatureExtractor is None:
        from faster_whisper import WhisperModel as _WhisperModel
        from faster_whisper.audio import decode_audio as _decode_audio
        from faster_whisper.feature_extractor import FeatureExtractor as _FeatureExtractor

        WhisperModel = _WhisperModel
        decode_audio = _decode_audio
        FeatureExtractor = _FeatureExtractor
    return WhisperModel, decode_audio, FeatureExtractor

# --- Configuration & Constants ---

DENOISE_FILTERS = {
    "none": "",
    "mild": "afftdn=nr=6:nf=-35:tn=1:ad=0.35:gs=6",
    "balanced": "afftdn=nr=10:nf=-40:tn=1:ad=0.5:gs=8",
    "strong": "afftdn=nr=14:nf=-45:tn=1:ad=0.65:gs=10",
}

HF_REPO_IDS = {
    "kotoba": "kotoba-tech/kotoba-whisper-v2.0-faster",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v2": "Systran/faster-whisper-large-v2",
    "whisperSeg": "TransWithAI/Whisper-Vad-EncDec-ASMR-onnx",
}

KOTOBA_DECODER_LAYERS = 2
KOTOBA_DECODER_ATTENTION_HEADS = 20
KOTOBA_ALIGNMENT_HEADS = [[1, head] for head in range(KOTOBA_DECODER_ATTENTION_HEADS)]
KOTOBA_CONFIG_BACKUP_SUFFIX = ".kotoba_alignment_heads.bak"

# Options adapted from original script
KOTOBA_BALANCED_OPTIONS = {
    "vad_parameters": {
        "threshold": 0.005,
        "min_speech_duration_ms": 90,
        "max_speech_duration_s": 28.0,
        "min_silence_duration_ms": 120,
        "speech_pad_ms": 500,
    },
    "beam_size": 3,
    "best_of": 3,
    "patience": 2.2,
    "temperature": [0.0, 0.2, 0.4],
    "compression_ratio_threshold": 2.6,
    "log_prob_threshold": -1.5,
    "no_speech_threshold": 0.34,
    "condition_on_previous_text": False,
    "word_timestamps": True,
    "suppress_tokens": None,
    "suppress_blank": True,
    "without_timestamps": False,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
}

KOTOBA_SCENE_OPTIONS = {
    "vad_parameters": {
        "threshold": 0.01,
        "min_speech_duration_ms": 90,
        "max_speech_duration_s": 28.0,
        "min_silence_duration_ms": 150,
        "speech_pad_ms": 400,
    },
    "beam_size": 3,
    "best_of": 3,
    "patience": 2.2,
    "temperature": [0.0, 0.2, 0.4],
    "compression_ratio_threshold": 2.6,
    "log_prob_threshold": None,
    "no_speech_threshold": None,
    "condition_on_previous_text": False,
    "word_timestamps": True,
    "suppress_tokens": None,
    "suppress_blank": True,
    "without_timestamps": False,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
}

MODEL_SCENE_OVERRIDES = {
    "kotoba": {
        "condition_on_previous_text": False,
        "word_timestamps": True,
    },
    "large-v3": {
        "compression_ratio_threshold": 2.4,
        # Decode-side silence gating OFF (like kotoba scene options): with
        # -1.0/0.70 faster-whisper silently dropped quiet lines inside chunks
        # before they ever reached raw output / our own postprocess filters
        # (debug session 2026-07-09: ~1/3 of missing lines vs reference subs).
        # Our postprocess still drops no_speech>0.90 & logprob<-1.35 lines.
        "log_prob_threshold": None,
        "no_speech_threshold": None,
        "condition_on_previous_text": False,
        "word_timestamps": True,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
    },
    "large-v2": {
        "compression_ratio_threshold": 2.4,
        # Decode-side silence gating OFF for the same reason as large-v3 below;
        # the old 0.34 no-speech gate was the most aggressive of all models.
        # Anti-loop measures (repetition_penalty / no_repeat_ngram_size) stay.
        "log_prob_threshold": None,
        "no_speech_threshold": None,
        "condition_on_previous_text": False,
        "word_timestamps": True,
        "repetition_penalty": 1.15,
        "no_repeat_ngram_size": 3,
    },
}

WIDE_INTAKE_OVERRIDES = {
    "no_speech_threshold": None,
    "log_prob_threshold": None,
    "condition_on_previous_text": False,
    "repetition_penalty": 1.12,
    "no_repeat_ngram_size": 0,
    "max_initial_timestamp": 30.0,
}

OLD_STYLE_VAD_PARAMETERS = {
    "threshold": 0.5,
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 300,
    "speech_pad_ms": 30,
}

HAS_LINGUISTIC_CONTENT_RE = re.compile(
    r"[\u3041-\u3096\u309d-\u309f"
    r"\u30a1-\u30fa\u30fc-\u30ff"
    r"\u4e00-\u9fffA-Za-z0-9]"
)
JAPANESE_TEXT_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
HALLUCINATION_PHRASES = {
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございます",
    "チャンネル登録と高評価をお願いします",
    "チャンネル登録お願いします",
    "最後まで見ていただきありがとうございました",
    "次の動画でお会いしましょう",
    "また次の動画でお会いしましょう",
    "次の動画で会いましょう",
    "次回の動画でお会いしましょう",
    "字幕by",
    "字幕バイ",
}
SHORT_HALLUCINATION_PHRASES = {
    "今日はこの辺で",
    "ありがとうございました",
    "おわり",
    "終わり",
    "バイバイ",
    # Sleep-scene breathing is persistently transcribed as goodnight lines
    # (mdvr-433 #3: six of seven were invented). The short-phrase guard keeps
    # a confidently decoded real goodnight.
    "おやすみなさい",
}
HARD_HALLUCINATION_NORMS = {
    re.sub(r"[、。！？!?….\s]+", "", phrase).lower()
    for phrase in HALLUCINATION_PHRASES
}
SHORT_HALLUCINATION_NORMS = {
    re.sub(r"[、。！？!?….\s]+", "", phrase).lower()
    for phrase in SHORT_HALLUCINATION_PHRASES
}

TRANSCRIBE_MODE = "chunked"
CHUNK_SECONDS = 28.0
CHUNK_OVERLAP_SECONDS = 2.0
SCENE_SPLIT_METHOD = "whisperseg"
AUDITOK_MAX_DURATION = min(CHUNK_SECONDS, 29.0)
AUDITOK_MIN_DURATION = 0.2
AUDITOK_PAD_SECONDS = 0.0
AUDITOK_MIN_CHUNK_SECONDS = 6.0
AUDITOK_MERGE_GAP_SECONDS = 1.6
AUDITOK_SHORT_MERGE_GAP_SECONDS = 3.0
AUDITOK_PASS1_MIN_DURATION = 0.3
AUDITOK_PASS1_MAX_DURATION = 2700.0
AUDITOK_PASS1_MAX_SILENCE = 1.8
AUDITOK_PASS1_ENERGY_THRESHOLD = 32.0
AUDITOK_PASS2_MAX_DURATION = max(AUDITOK_MAX_DURATION - 1.0, AUDITOK_MIN_DURATION)
AUDITOK_PASS2_MIN_DURATION = 0.3
AUDITOK_PASS2_MAX_SILENCE = 0.94
AUDITOK_PASS2_ENERGY_THRESHOLD = 38.0
SCENE_INTERNAL_VAD = False
EnableVAD = True
WHISPERSEG_THRESHOLD = 0.5
WHISPERSEG_NEG_THRESHOLD = 0.35
WHISPERSEG_MIN_SPEECH_MS = 250.0
WHISPERSEG_MIN_SILENCE_MS = 100.0
WHISPERSEG_SPEECH_PAD_MS = 30.0
WHISPERSEG_MERGE_GAP_SECONDS = 2.0
WHISPERSEG_MIN_CHUNK_SECONDS = 2.0
ENABLE_RMS_SPEECH_GATE = True
RMS_SPEECH_MIN_DB = -50.0

# Splitting decoded segments at intra-segment pauses (mdvr-433 debug,
# 2026-07-14): the decoder can merge two utterances separated by >1s of
# silence into a single segment, and DTW word alignment can absorb that
# pause into the first word after it (通 stamped 21.1-22.9s while speech
# resumed at 22.5s), so the pause never shows up as an inter-word gap.
# Energy-trimming the silent lead of long words re-exposes the pause; the
# segment is then split at inter-word gaps so one subtitle line never spans
# a long silence.
WORD_PAUSE_SPLIT_SECONDS = 0.7
WORD_TIGHTEN_MIN_WORD_SECONDS = 1.0
WORD_TIGHTEN_SILENCE_DB = -45.0
WORD_TIGHTEN_MIN_LEAD_SECONDS = 0.4
WORD_TIGHTEN_ONSET_PAD_SECONDS = 0.1

# Acoustic hallucination checks in postprocess (2026-07-14). Two classes:
# a line whose whole span never rises above a floor far below every VAD gate
# is decoder invention over silence; a line whose span barely overlaps
# WhisperSeg speech probability is invention over non-speech sound (moans,
# music) and is removed only when the decoder itself was unsure (AND with
# low confidence), so quiet-but-confident whispers survive. Lines without
# word-anchored times get the silence check only under the same AND guard,
# because a mis-stamped decoder span can cover real silence.
HALLUCINATION_SILENCE_PEAK_DB = -55.0
HALLUCINATION_SILENCE_FLOOR_MARGIN_DB = 5.0
HALLUCINATION_SILENCE_PEAK_DB_NO_GATE = -60.0
HALLUCINATION_MIN_SPEECH_COVERAGE = 0.25
HALLUCINATION_LOW_CONFIDENCE_LOGPROB = -0.70
HALLUCINATION_LOW_CONFIDENCE_NO_SPEECH = 0.50

# DTW word-end stamps land 0.3-0.8s before the audible end of an utterance
# (mdvr-433 entries 13/14: subtitles ended at 44.56/47.56 while speech ran to
# 45.3/48.25). Extend a word-anchored end through contiguous voiced frames,
# capped, never into the next line.
TAIL_EXTEND_MAX_SECONDS = 1.0
TAIL_EXTEND_VOICED_DB = -45.0
TAIL_EXTEND_RELEASE_PAD_SECONDS = 0.12
TAIL_EXTEND_GUARD_SECONDS = 0.05
# Speech has short intra-word energy dips (glottal stops: 60ms at -50dB inside
# はいありがとうございます, mdvr-433 44.78-44.84s) that must not end the
# extension scan; only a silence run longer than this hangover does.
TAIL_EXTEND_MAX_SILENCE_RUN_SECONDS = 0.18

# DTW can distribute a long pause across several word boundaries so that no
# single inter-word gap or over-long word reveals it (mdvr-433 #3 entry 13:
# one 12.5s line covering four utterances with 0.9-1.5s silences between).
# Energy is authoritative: split entries at internal silence runs, assigning
# words to sides by midpoint and clamping the edges to the run boundaries.
ENTRY_SPLIT_MIN_SILENCE_SECONDS = 0.8
ENTRY_SPLIT_SILENCE_DB = -45.0
ENTRY_SPLIT_EDGE_PAD_SECONDS = 0.12

# A line that Whisper itself doubts (high no_speech) whose loudest frame sits
# far below any real dialogue peak is invention over near-silence. Calibrated
# on mdvr-433 1/2/3: every real line peaked >= -42dB, while 次の動画で…/ご視聴…
# hallucinations and moan transcriptions sat at ns>=0.65 with peaks <= -48dB.
HALLUCINATION_QUIET_NO_SPEECH = 0.65
HALLUCINATION_QUIET_PEAK_DB = -48.0
# Graduated second stage: when Whisper is nearly certain there is no speech
# (ns>0.82) even a breathing-level peak (-38dB) cannot vouch for the line
# (mdvr-433 #2 entry 18: 視聴ありがとうございました ns=0.86 pk=-41 survived
# the -48dB floor). Every observed real line with a peak that quiet had ns
# well below this bar.
HALLUCINATION_QUIET_STRONG_NO_SPEECH = 0.82
HALLUCINATION_QUIET_STRONG_PEAK_DB = -38.0

# Breathing tracked as words stretches a stock phrase over many seconds
# (mdvr-433 #3: おやすみなさい spanning 11s at 0.6 chars/sec). A word-anchored
# line lasting several times its reading time while Whisper doubts the speech
# is invention; real slow speech gets split at >=0.8s silences first.
HALLUCINATION_SPARSE_NO_SPEECH = 0.40
HALLUCINATION_SPARSE_FACTOR = 3.0
HALLUCINATION_SPARSE_MIN_SECONDS = 4.0

# The decoder often emits one sentence as several adjacent segments
# (通常コースで | よろしかったですか? with a zero gap), which reads as
# fragmented subtitles. Merge consecutive lines back together when the gap is
# tiny and the combined text stays a readable single line; real pauses
# (> WORD_PAUSE_SPLIT_SECONDS) are never bridged.
MERGE_MAX_GAP_SECONDS = 0.30
MERGE_MAX_CHARS = 24
MERGE_MAX_DURATION_SECONDS = 8.0
MERGE_SENTENCE_FINAL_CHARS = ("?", "？", "!", "！", "。")

# VAD sensitivity presets (how quiet a sound still counts as speech).
# "standard" reproduces the historical constants above. Higher levels lower the
# WhisperSeg gate, widen the speech padding, and relax/disable the RMS energy
# gate so breathy or whispered lines reach the decoder; the decode-side and
# postprocess confidence filters still guard against the extra noise.
VAD_SENSITIVITY_PRESETS = {
    "standard": {
        "threshold": WHISPERSEG_THRESHOLD,
        "neg_threshold": WHISPERSEG_NEG_THRESHOLD,
        "speech_pad_ms": WHISPERSEG_SPEECH_PAD_MS,
        "rms_gate_db": RMS_SPEECH_MIN_DB,
    },
    "high": {
        "threshold": 0.35,
        "neg_threshold": 0.22,
        "speech_pad_ms": 200.0,
        "rms_gate_db": -58.0,
    },
    "max": {
        "threshold": 0.20,
        "neg_threshold": 0.12,
        "speech_pad_ms": 320.0,
        "rms_gate_db": None,
    },
}

DUPLICATE_LOOKBACK_SECONDS = max(6.0, CHUNK_OVERLAP_SECONDS + 2.0)
NEAR_DUPLICATE_RATIO = 0.92

PROFILE_CONFIGS = {
    "stable": {
        "label": "stable",
        "chunk_seconds": CHUNK_SECONDS,
        "options": {},
    },
}

VIDEO_EXTENSIONS = {".mp4", ".mkv"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
SI_SIDECAR_MEDIA_SUFFIXES = (".si.wav", ".si.duck.wav", ".si.mp4")
GENERATED_MP4_SUFFIXES = ("_si.mp4", "_dub.mp4")
GENERATED_WORK_DIR_SUFFIXES = ("_debug", ".clone")

# --- Utility Functions ---

def is_si_sidecar_media_file(path: str | os.PathLike[str]) -> bool:
    return Path(path).name.lower().endswith(SI_SIDECAR_MEDIA_SUFFIXES)


def is_generated_output_mp4(path: str | os.PathLike[str]) -> bool:
    return Path(path).name.lower().endswith(GENERATED_MP4_SUFFIXES)


def is_generated_work_directory(path: str | os.PathLike[str]) -> bool:
    return Path(path).name.lower().endswith(GENERATED_WORK_DIR_SUFFIXES)


def is_generated_work_path(path: str | os.PathLike[str]) -> bool:
    return any(is_generated_work_directory(part) for part in Path(path).parts)


def is_speaker_basis_wav(path: str | os.PathLike[str]) -> bool:
    candidate = Path(path)
    return candidate.suffix.lower() == ".wav" and candidate.stem.lower().startswith("speaker")


def is_supported_source_media_file(path: str | os.PathLike[str]) -> bool:
    candidate = Path(path)
    return (
        candidate.suffix.lower() in (VIDEO_EXTENSIONS | AUDIO_EXTENSIONS)
        and not is_generated_work_path(candidate)
        and not is_speaker_basis_wav(candidate)
        and not is_si_sidecar_media_file(candidate)
        and not is_generated_output_mp4(candidate)
    )


def is_subtitle_video_candidate(path: str | os.PathLike[str]) -> bool:
    candidate = Path(path)
    if (
        is_generated_work_path(candidate)
        or is_si_sidecar_media_file(candidate)
        or is_generated_output_mp4(candidate)
    ):
        return False
    name = candidate.name.lower()
    return candidate.suffix.lower() in VIDEO_EXTENSIONS and not name.endswith("_srt.mkv")


def _walk_user_media_directories(base_dir: str | os.PathLike[str]):
    """Walk user media while pruning generated debug/clone work directories."""
    for root, dirs, files in os.walk(base_dir):
        if is_generated_work_path(root):
            dirs[:] = []
            continue
        dirs[:] = [name for name in dirs if not is_generated_work_directory(name)]
        yield Path(root), files

def check_ffmpeg():
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

def has_subtitle_stream(video_path):
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-select_streams", "s",
            video_path
        ]
        startupinfo = None
        if sys.platform.startswith('win'):
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, startupinfo=startupinfo)
        data = json.loads(result.stdout)
        return len(data.get("streams", [])) > 0
    except Exception as e:
        print(f"Error checking subtitle streams: {e}")
        return False

def run_process(cmd, log_callback, process_callback=None, stop_event=None):
    cmd_str = ' '.join(cmd) if isinstance(cmd, (list, tuple)) else str(cmd)
    if log_callback:
        log_callback(f"Executing: {cmd_str}")
    else:
        print(f"Executing: {cmd_str}")
        
    startupinfo = None
    if sys.platform.startswith('win'):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        errors='replace',
        startupinfo=startupinfo
    )
    if process_callback: process_callback(process)

    try:
        for line in process.stdout:
            if stop_event and stop_event.is_set():
                try: process.kill()
                except Exception: pass
                break
            if log_callback: log_callback(line.strip())
    finally:
        try:
            if process.stdout:
                process.stdout.close()
        except Exception:
            pass
        process.wait()
    if stop_event and stop_event.is_set():
        raise Exception("Process stopped by user.")
    if process.returncode != 0:
        err_msg = f"Command failed with code {process.returncode}"
        try:
            checker_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if checker_path not in sys.path:
                sys.path.append(checker_path)
            from utils import ffmpeg_checker
            ffmpeg_checker.handle_ffmpeg_error(cmd, err_msg, log_callback)
        except Exception as e:
            if log_callback: log_callback(f"Checker error: {e}")
            pass
        raise Exception(err_msg)

def get_model_dir(model_key: str, models_root: str) -> str:
    if model_key == "whisperSeg":
        return os.path.join(models_root, "Whisper-Vad-EncDec-ASMR")
    repo_id = HF_REPO_IDS.get(model_key)
    if not repo_id:
        return os.path.join(models_root, model_key)
    return os.path.join(models_root, repo_id.split("/")[-1])

def check_model_files(model_key: str, models_root: str) -> bool:
    """Check if model directory contains required config/model files."""
    model_dir = get_model_dir(model_key, models_root)
    if not os.path.exists(model_dir):
        return False
        
    if model_key == "whisperSeg":
        return os.path.exists(os.path.join(model_dir, "model.onnx"))
        
    # faster-whisper CTranslate2 model must have config.json and model.bin
    has_config = os.path.exists(os.path.join(model_dir, "config.json"))
    has_model = os.path.exists(os.path.join(model_dir, "model.bin"))
    
    return has_config and has_model

def _read_json_utf8_sig(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)

def _write_json_utf8(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")

def _is_valid_alignment_head(value) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    layer, head = value
    if not isinstance(layer, int) or isinstance(layer, bool):
        return False
    if not isinstance(head, int) or isinstance(head, bool):
        return False
    return (
        0 <= layer < KOTOBA_DECODER_LAYERS
        and 0 <= head < KOTOBA_DECODER_ATTENTION_HEADS
    )

def kotoba_alignment_heads_need_repair(config: dict) -> bool:
    heads = config.get("alignment_heads")
    if not isinstance(heads, list) or not heads:
        return True
    return any(not _is_valid_alignment_head(head) for head in heads)

def repair_kotoba_alignment_heads(model_key: str, model_dir: str, log_callback=None) -> bool:
    """Patch Kotoba's CT2 alignment heads so word_timestamps cannot crash CT2."""
    if model_key != "kotoba":
        return False

    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Kotoba model config not found: {config_path}")

    config = _read_json_utf8_sig(config_path)
    if not kotoba_alignment_heads_need_repair(config):
        return False

    backup_path = config_path.with_name(config_path.name + KOTOBA_CONFIG_BACKUP_SUFFIX)
    if not backup_path.exists():
        shutil.copy2(config_path, backup_path)

    config["alignment_heads"] = [head[:] for head in KOTOBA_ALIGNMENT_HEADS]
    _write_json_utf8(config_path, config)

    if log_callback:
        log_callback(
            "Patched kotoba alignment_heads for word_timestamps "
            f"(backup: {backup_path.name})"
        )
    return True

def download_model(model_key: str, models_root: str, log_callback) -> bool:
    """Download model from HuggingFace hub."""
    repo_id = HF_REPO_IDS.get(model_key)
    if not repo_id:
        log_callback(f"Error: Unknown model key {model_key}")
        return False
        
    model_dir = get_model_dir(model_key, models_root)
    try:
        if os.environ.get("HF_ENDPOINT") == "https://hf-mirror.com":
            log_callback("use hf-mirror.com for faster downloads")
            
        from huggingface_hub import hf_hub_download, list_repo_files
        import huggingface_hub.constants
        log_callback(f"HuggingFace Endpoint: {huggingface_hub.constants.ENDPOINT}")
        import fnmatch

        # Download only necessary files
        if model_key == "whisperSeg":
            allow_patterns = ["model.onnx"]
        else:
            allow_patterns = ["config.json", "model.bin", "vocabulary.*", "tokenizer.json", "preprocessor_config.json"]
        
        endpoint = huggingface_hub.constants.ENDPOINT
        log_callback(f"Fetching file list for {repo_id} from {endpoint}/{repo_id}/tree/main ...")
        try:
            repo_files = list_repo_files(repo_id=repo_id)
        except Exception as e:
            log_callback(f"Failed to fetch file list: {e}")
            return False
            
        # Filter files based on allow_patterns
        files_to_download = []
        for file in repo_files:
            for pattern in allow_patterns:
                if fnmatch.fnmatch(file, pattern):
                    files_to_download.append(file)
                    break
                    
        if not files_to_download:
            log_callback(f"No necessary model files found in {repo_id}.")
            return False

        # Patch sys.stderr and sys.stdout if None (common in PyInstaller --noconsole EXE)
        # to prevent tqdm or huggingface_hub from crashing on 'NoneType' object has no attribute 'write'
        class DummyStream:
            def write(self, data): pass
            def flush(self): pass
            
        if sys.stderr is None:
            sys.stderr = DummyStream()
        if sys.stdout is None:
            sys.stdout = DummyStream()

        for idx, file in enumerate(files_to_download, 1):
            log_callback(f"[{idx}/{len(files_to_download)}] Downloading: {file} ...")
            hf_hub_download(
                repo_id=repo_id,
                filename=file,
                local_dir=model_dir,
                local_dir_use_symlinks=False
            )
            log_callback(f"[{idx}/{len(files_to_download)}] Finished: {file}")
        
        log_callback("All files downloaded successfully.")
        repair_kotoba_alignment_heads(model_key, model_dir, log_callback)
        return True
    except ImportError:
        log_callback("Error: huggingface_hub package is not installed.")
        return False
    except Exception as e:
        log_callback(f"Download failed: {str(e)}")
        return False

# --- Batch Add Subtitles ---

def batch_add_srt(base_dir, search_subdirs=True, replace_original=False, auto_load_srt=True, skip_if_has_sub=False, prefer_ass=False, log_callback=lambda x: None, process_callback=None):
    if not check_ffmpeg():
        log_callback("[!] Error: ffmpeg or ffprobe not found. Please refer to '说明.txt' or 'readme.txt' for installation steps.")
        return False

    if not os.path.exists(base_dir):
        log_callback(f"Error: Directory not found: {base_dir}")
        return False

    # Collect files first to determine total task size and handle stop easily
    tasks = []
    
    if search_subdirs:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if is_subtitle_video_candidate(file):
                    tasks.append((root, file))
    else:
        try:
            files = os.listdir(base_dir)
            for file in files:
                if is_subtitle_video_candidate(file) and os.path.isfile(os.path.join(base_dir, file)):
                    tasks.append((base_dir, file))
        except Exception as e:
            log_callback(f"Error reading directory: {e}")
            return False

    if not tasks:
        log_callback("No valid mp4/mkv files found.")
        return True

    for root, file in tasks:
        file_path = os.path.join(root, file)
        file_name_no_ext = os.path.splitext(file)[0]
        
        # Look for subtitle file
        sub_file = None
        try:
            dir_files = os.listdir(root)
            exts_to_try = [".ass", ".srt"] if prefer_ass else [".srt", ".ass"]
            
            for ext in exts_to_try:
                for f in dir_files:
                    if f.lower() == (file_name_no_ext + ext).lower() and os.path.isfile(os.path.join(root, f)):
                        sub_file = os.path.join(root, f)
                        break
                if sub_file is not None:
                    break
        except Exception as e:
            log_callback(f"Error listing files in {root}: {e}")
            continue

        # Skip files that already have embedded subtitle streams
        if skip_if_has_sub:
            if has_subtitle_stream(file_path):
                log_callback(f"Skipped (already has subtitles): {file}")
                continue

        if sub_file and os.path.exists(sub_file):
            sub_ext = os.path.splitext(sub_file)[1].lower()
            # Include source extension in temp output so foo.mp4 and foo.mkv don't collide
            src_ext = os.path.splitext(file)[1].lstrip(".").lower()
            output_file = os.path.join(root, f"{file_name_no_ext}_{src_ext}_srt.mkv")
            log_callback(f"--- Processing: {file} (subtitle: {os.path.basename(sub_file)}) ---")
            
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
                "-i", file_path,
                "-i", sub_file,
                "-map", "0:v",
                "-map", "0:a?",
                "-map", "1:s",
                "-c", "copy",
                "-reserve_index_space", "1000k",
                "-metadata:s:s:0", "title=subtitle",
            ]

            if auto_load_srt:
                cmd.extend(["-disposition:s:0", "default"])

            cmd.append(output_file)

            try:
                run_process(cmd, log_callback, process_callback)
                log_callback(f"Success: {os.path.basename(output_file)}")

                # Handle original file replacement
                final_mkv_path = os.path.join(root, file_name_no_ext + ".mkv")
                if replace_original:
                    try:
                        # Avoid silently clobbering a different existing file when source ext != .mkv
                        if os.path.abspath(final_mkv_path) != os.path.abspath(file_path) and os.path.exists(final_mkv_path):
                            log_callback(f"Skip rename: target already exists: {os.path.basename(final_mkv_path)}")
                        else:
                            os.remove(file_path)
                            os.rename(output_file, final_mkv_path)
                            log_callback(f"Replaced original: {os.path.basename(final_mkv_path)}")
                    except Exception as e:
                        log_callback(f"Error replacing original video: {e}")
                        
            except Exception as e:
                log_callback(f"Error processing {file}: {e}")
                # Continue with next task on error

    log_callback("Batch Add SRT Task Completed.")
    return True

def batch_remove_srt(base_dir, search_subdirs=True, delete_mkv=False, log_callback=lambda x: None, process_callback=None):
    if not check_ffmpeg():
        log_callback("[!] Error: ffmpeg or ffprobe not found. Please refer to '说明.txt' or 'readme.txt' for installation steps.")
        return False

    if not os.path.exists(base_dir):
        log_callback(f"Error: Directory not found: {base_dir}")
        return False

    tasks = []
    
    if search_subdirs:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.lower().endswith(".mkv"):
                    tasks.append((root, file))
    else:
        try:
            files = os.listdir(base_dir)
            for file in files:
                if file.lower().endswith(".mkv") and os.path.isfile(os.path.join(base_dir, file)):
                    tasks.append((base_dir, file))
        except Exception as e:
            log_callback(f"Error reading directory: {e}")
            return False

    if not tasks:
        log_callback("No valid mkv files found.")
        return True

    for root, file in tasks:
        file_path = os.path.join(root, file)
        file_name_no_ext = os.path.splitext(file)[0]
        
        if not has_subtitle_stream(file_path):
            log_callback(f"Skipped (no subtitles): {file}")
            continue

        output_file = os.path.join(root, f"{file_name_no_ext}.mp4")
        log_callback(f"--- Restoring: {file} to {os.path.basename(output_file)} ---")
        
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
            "-i", file_path,
            "-map", "0:v",
            "-map", "0:a?",
            "-c", "copy",
            output_file
        ]

        try:
            run_process(cmd, log_callback, process_callback)
            log_callback(f"Success: {os.path.basename(output_file)}")

            if delete_mkv:
                try:
                    os.remove(file_path)
                    log_callback(f"Deleted original MKV: {file}")
                except Exception as e:
                    log_callback(f"Error deleting original MKV: {e}")
                    
        except Exception as e:
            log_callback(f"Error processing {file}: {e}")

    log_callback("Batch Remove SRT Task Completed.")
    return True

# --- Subtitle Generation ---

class SubtitleGenerator:
    def __init__(self, model_path: str, model_preset: str, log_callback, use_gpu: bool = True):
        WhisperModel, _, _ = _ensure_faster_whisper()

        self.model_preset = model_preset
        self.log_callback = log_callback
        self.models_root = os.path.dirname(model_path)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"CTranslate2 model directory not found: {model_path}"
            )

        repair_kotoba_alignment_heads(model_preset, model_path, log_callback)

        cpu_count = os.cpu_count() or 4
        num_workers = min(4, max(1, cpu_count // 2))

        if use_gpu:
            try:
                self.model = WhisperModel(
                    model_path,
                    device="cuda",
                    compute_type="auto",
                    num_workers=num_workers,
                )
                self.device = "cuda"
                self.log_callback(f"Loaded ASR model on CUDA (workers: {num_workers})")
            except Exception as exc:
                self.log_callback(f"CUDA model load failed, falling back to CPU int8: {exc}")
                try:
                    language = app_config.get_language()
                    if language == 'zh':
                        self.log_callback("【Nvidia显卡加速提示】若想开启 GPU 加速，请下载最新的 cuBLAS 和 cuDNN 独立包并解压至 exe 同级目录。")
                        self.log_callback("下载地址: https://github.com/Purfview/whisper-standalone-win/releases/tag/libs")
                    elif language == 'ja':
                        self.log_callback("【Nvidia GPU高速化】GPU利用にはcuBLAS/cuDNNをexe同階層へ展開してください。")
                        self.log_callback("DL: https://github.com/Purfview/whisper-standalone-win/releases/tag/libs")
                    else:
                        self.log_callback("【Nvidia GPU Acceleration Tip】To enable GPU acceleration, please download cuBLAS & cuDNN libs and extract to the software root directory.")
                        self.log_callback("Download URL: https://github.com/Purfview/whisper-standalone-win/releases/tag/libs")
                except Exception:
                    self.log_callback("Download cuBLAS & cuDNN for GPU acceleration: https://github.com/Purfview/whisper-standalone-win/releases/tag/libs")
                    
                self.model = WhisperModel(model_path, device="cpu", compute_type="int8", num_workers=num_workers)
                self.device = "cpu"
        else:
            self.model = WhisperModel(model_path, device="cpu", compute_type="int8", num_workers=num_workers)
            self.device = "cpu"
            self.log_callback(f"Loaded ASR model on CPU (int8, workers: {num_workers})")
        
        self.whisperseg_session = None
        self.whisperseg_feature_extractor = None
        self.last_raw_segments = []
        self.last_chunks = []
        self.last_speech_probs = None
        self.set_vad_sensitivity("standard")

    def set_vad_sensitivity(self, preset_name: str):
        preset = VAD_SENSITIVITY_PRESETS.get(preset_name)
        if preset is None:
            self.log_callback(f"Unknown VAD sensitivity {preset_name!r}; using 'standard'")
            preset_name = "standard"
            preset = VAD_SENSITIVITY_PRESETS["standard"]
        self.vad_sensitivity = preset_name
        self.vad_threshold = preset["threshold"]
        self.vad_neg_threshold = preset["neg_threshold"]
        self.vad_speech_pad_ms = preset["speech_pad_ms"]
        self.vad_rms_gate_db = preset["rms_gate_db"]

    @staticmethod
    def scene_to_chunk(
        audio: np.ndarray,
        start_sec: float,
        end_sec: float,
        sampling_rate: int = 16000,
        pad_seconds: float = 0.0,
    ):
        if pad_seconds > 0:
            total_duration = len(audio) / sampling_rate
            start_sec = max(0.0, start_sec - pad_seconds)
            end_sec = min(total_duration, end_sec + pad_seconds)
        start_sample = max(0, int(start_sec * sampling_rate))
        end_sample = min(len(audio), int(end_sec * sampling_rate))
        if end_sample <= start_sample:
            return None
        return {
            "array": audio[start_sample:end_sample].astype(np.float32, copy=False),
            "offset_sec": start_sample / sampling_rate,
            "duration_sec": (end_sample - start_sample) / sampling_rate,
        }

    @staticmethod
    def brute_force_scene_chunks(audio: np.ndarray, start_sec: float, end_sec: float, chunk_seconds: float):
        chunks = []
        cursor = start_sec
        while cursor < end_sec:
            chunk_end = min(end_sec, cursor + chunk_seconds)
            if chunk_end - cursor >= AUDITOK_MIN_DURATION:
                chunk = SubtitleGenerator.scene_to_chunk(audio, cursor, chunk_end)
                if chunk:
                    chunks.append(chunk)
            cursor = chunk_end
        return chunks

    @staticmethod
    def coalesce_auditok_chunks(audio: np.ndarray, chunks: list, max_duration: float):
        if not chunks:
            return []

        merged = []
        sorted_chunks = sorted(chunks, key=lambda chunk: chunk["offset_sec"])
        current_start = sorted_chunks[0]["offset_sec"]
        current_end = current_start + sorted_chunks[0]["duration_sec"]

        for chunk in sorted_chunks[1:]:
            next_start = chunk["offset_sec"]
            next_end = next_start + chunk["duration_sec"]
            current_duration = current_end - current_start
            next_duration = next_end - next_start
            gap = max(0.0, next_start - current_end)
            combined_duration = next_end - current_start
            short_pair = (
                current_duration < AUDITOK_MIN_CHUNK_SECONDS
                or next_duration < AUDITOK_MIN_CHUNK_SECONDS
            )
            should_merge = (
                combined_duration <= max_duration
                and (
                    gap <= AUDITOK_MERGE_GAP_SECONDS
                    or (short_pair and gap <= AUDITOK_SHORT_MERGE_GAP_SECONDS)
                )
            )

            if should_merge:
                current_end = next_end
                continue

            merged_chunk = SubtitleGenerator.scene_to_chunk(audio, current_start, current_end)
            if merged_chunk:
                merged.append(merged_chunk)
            current_start = next_start
            current_end = next_end

        merged_chunk = SubtitleGenerator.scene_to_chunk(audio, current_start, current_end)
        if merged_chunk:
            merged.append(merged_chunk)

        return merged

    def split_audio(self, audio_path: str, chunk_seconds: float):
        """Return fixed-size chunks with absolute offsets."""
        _, decode_audio, _ = _ensure_faster_whisper()
        audio = decode_audio(audio_path, sampling_rate=16000)
        if audio.ndim != 1:
            audio = np.asarray(audio).reshape(-1)

        chunk_samples = int(chunk_seconds * 16000)
        overlap_samples = int(CHUNK_OVERLAP_SECONDS * 16000)
        step_samples = max(16000, chunk_samples - overlap_samples)
        chunks = []
        for start in range(0, len(audio), step_samples):
            end = min(len(audio), start + chunk_samples)
            if end - start < 16000:
                break
            chunks.append({
                "array": audio[start:end].astype(np.float32, copy=False),
                "offset_sec": start / 16000.0,
                "duration_sec": (end - start) / 16000.0,
            })

        self.log_callback(
            f"Chunked audio into {len(chunks)} pieces "
            f"of up to {chunk_seconds:.1f}s "
            f"with {CHUNK_OVERLAP_SECONDS:.1f}s overlap"
        )
        return chunks

    def split_audio_auditok(self, audio_path: str, chunk_seconds: float):
        try:
            import auditok
        except ImportError as exc:
            self.log_callback("auditok import failed; falling back to fixed chunks.")
            return self.split_audio(audio_path, chunk_seconds)

        _, decode_audio, _ = _ensure_faster_whisper()
        audio = decode_audio(audio_path, sampling_rate=16000)
        if audio.ndim != 1:
            audio = np.asarray(audio).reshape(-1)

        sampling_rate = 16000
        total_duration = len(audio) / sampling_rate
        audio_bytes = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        max_duration = min(chunk_seconds, AUDITOK_MAX_DURATION)

        pass1_params = {
            "sampling_rate": sampling_rate,
            "channels": 1,
            "sample_width": 2,
            "min_dur": AUDITOK_PASS1_MIN_DURATION,
            "max_dur": AUDITOK_PASS1_MAX_DURATION,
            "max_silence": min(total_duration * 0.95, AUDITOK_PASS1_MAX_SILENCE),
            "energy_threshold": AUDITOK_PASS1_ENERGY_THRESHOLD,
            "max_trailing_silence": 0,
        }

        try:
            coarse_regions = list(auditok.split(audio_bytes, **pass1_params))
        except Exception as exc:
            self.log_callback(f"auditok pass 1 failed: {exc}; falling back to fixed chunks")
            return self.split_audio(audio_path, chunk_seconds)

        chunks = []
        direct_count = 0
        split_count = 0
        fallback_count = 0

        for region in coarse_regions:
            region_start = max(0.0, float(region.start))
            region_end = min(total_duration, float(region.end))
            region_duration = region_end - region_start
            if region_duration < AUDITOK_MIN_DURATION:
                continue

            if region_duration <= max_duration:
                chunk = self.scene_to_chunk(
                    audio,
                    region_start,
                    region_end,
                    sampling_rate,
                    AUDITOK_PAD_SECONDS,
                )
                if chunk:
                    chunks.append(chunk)
                    direct_count += 1
                continue

            start_sample = int(region_start * sampling_rate)
            end_sample = int(region_end * sampling_rate)
            region_audio = audio[start_sample:end_sample]
            region_bytes = (np.clip(region_audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            pass2_params = {
                "sampling_rate": sampling_rate,
                "channels": 1,
                "sample_width": 2,
                "min_dur": AUDITOK_PASS2_MIN_DURATION,
                "max_dur": min(AUDITOK_PASS2_MAX_DURATION, max_duration),
                "max_silence": min(region_duration * 0.95, AUDITOK_PASS2_MAX_SILENCE),
                "energy_threshold": AUDITOK_PASS2_ENERGY_THRESHOLD,
                "max_trailing_silence": 0,
            }

            try:
                fine_regions = list(auditok.split(region_bytes, **pass2_params))
            except Exception as exc:
                self.log_callback(f"auditok pass 2 failed at {region_start:.2f}s: {exc}; using fixed split for this scene")
                fine_regions = []

            if fine_regions:
                for fine in fine_regions:
                    fine_start = region_start + float(fine.start)
                    fine_end = min(region_end, region_start + float(fine.end))
                    if fine_end - fine_start < AUDITOK_MIN_DURATION:
                        continue
                    chunk = self.scene_to_chunk(
                        audio,
                        fine_start,
                        fine_end,
                        sampling_rate,
                        AUDITOK_PAD_SECONDS,
                    )
                    if chunk:
                        chunks.append(chunk)
                        split_count += 1
            else:
                scene_chunks = self.brute_force_scene_chunks(audio, region_start, region_end, max_duration)
                chunks.extend(scene_chunks)
                fallback_count += len(scene_chunks)

        if not chunks:
            self.log_callback("auditok found no speech scenes; falling back to fixed chunks")
            return self.split_audio(audio_path, chunk_seconds)

        raw_chunk_count = len(chunks)
        chunks = self.coalesce_auditok_chunks(audio, chunks, max_duration)
        kept_duration = sum(chunk["duration_sec"] for chunk in chunks)
        self.log_callback(
            f"auditok scene split into {raw_chunk_count} raw scenes, "
            f"coalesced to {len(chunks)} chunks "
            f"(direct={direct_count}, split={split_count}, fallback={fallback_count}), "
            f"kept {kept_duration:.2f}s / {total_duration:.2f}s"
        )
        return chunks

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        if seconds is None:
            return "00:00:00,000"
        ms = int(round((seconds % 1) * 1000))
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    @staticmethod
    def clean_text(text: str) -> str:
        return text.strip().replace(" ", "")

    @staticmethod
    def normalize_for_duplicate(text: str) -> str:
        return re.sub(r"[、。！？!?….\s]+", "", text).lower()

    @staticmethod
    def is_near_duplicate(current_norm: str, previous_norm: str) -> bool:
        if not current_norm or not previous_norm:
            return False
        if current_norm == previous_norm:
            return True

        min_len = min(len(current_norm), len(previous_norm))
        max_len = max(len(current_norm), len(previous_norm))
        if min_len < 8:
            return False
        if min_len / max_len >= 0.72 and (
            current_norm in previous_norm or previous_norm in current_norm
        ):
            return True

        return SequenceMatcher(None, current_norm, previous_norm).ratio() >= NEAR_DUPLICATE_RATIO

    @staticmethod
    def duplicate_strength(current_norm: str, previous_norm: str) -> float:
        if not current_norm or not previous_norm:
            return 0.0
        if current_norm == previous_norm:
            return 1.0

        min_len = min(len(current_norm), len(previous_norm))
        max_len = max(len(current_norm), len(previous_norm))
        if min_len < 3:
            return 0.0
        if current_norm in previous_norm or previous_norm in current_norm:
            return min_len / max_len
        return SequenceMatcher(None, current_norm, previous_norm).ratio()

    @staticmethod
    def is_repetition_noise(text: str) -> bool:
        compact = SubtitleGenerator.normalize_for_duplicate(text)
        if len(compact) < 12:
            return False

        # Catch "abcabcabc" style loops.
        for size in range(2, min(8, len(compact) // 2) + 1):
            unit = compact[:size]
            if unit and unit * (len(compact) // size) == compact[: size * (len(compact) // size)]:
                if len(compact) / size >= 4:
                    return True

        # Catch high repeated n-gram density inside one subtitle.
        n = 5
        if len(compact) >= n * 3:
            grams = [compact[i:i + n] for i in range(len(compact) - n + 1)]
            duplicate_ratio = 1.0 - len(set(grams)) / len(grams)
            if duplicate_ratio > 0.58:
                return True

        return False

    @staticmethod
    def compress_repetition_text(text: str, repeats_kept: int = 3) -> str:
        """Collapse looped repetition to a few units ("無理無理×20" -> "無理無理無理").

        Real climax lines genuinely repeat one word dozens of times; dropping the
        whole line (the old behaviour) deleted real dialogue. Keep a readable,
        compressed rendition instead and let the timing stay untouched.
        """
        s = text.strip()
        # Any unit of 1..12 chars repeated 4+ times consecutively -> keep 3.
        s = re.sub(
            r"(.{1,12}?)\1{" + str(repeats_kept) + r",}",
            lambda m: m.group(1) * repeats_kept,
            s,
            flags=re.DOTALL,
        )

        # Same, but with a punctuation separator between units and no trailing
        # separator ("ごめん、ごめん、ごめん、ごめん"), which the pattern above
        # cannot fold into whole units.
        sep_pattern = re.compile(
            r"(.{1,12}?)(([、。，,．.!！?？~〜・\s…]+)\1(?:\3\1)*)",
            re.DOTALL,
        )

        def collapse(match):
            unit, tail, sep = match.group(1), match.group(2), match.group(3)
            total = 1 + tail.count(sep + unit)
            if total > repeats_kept:
                return (unit + sep) * (repeats_kept - 1) + unit
            return match.group(0)

        return sep_pattern.sub(collapse, s)

    @staticmethod
    def subtitle_duration_for_text(text: str, current_duration: float,
                                   allow_shrink: bool = True) -> float:
        text_len = len(SubtitleGenerator.normalize_for_duplicate(text))
        if text_len <= 0:
            return current_duration

        target = text_len / 5.2 + 0.8
        max_duration = max(1.4, min(7.5, target))
        min_duration = min(max_duration, max(0.7, min(1.4, text_len / 12.0 + 0.35)))
        capped = min(current_duration, max_duration) if allow_shrink else current_duration
        return max(min_duration, capped)

    def load_whisperseg_session(self):
        if self.whisperseg_session is not None:
            return self.whisperseg_session

        model_path = os.path.join(self.models_root, "Whisper-Vad-EncDec-ASMR", "model.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"WhisperSeg ONNX model not found: {model_path}")

        import onnxruntime

        session_options = onnxruntime.SessionOptions()
        cpu_count = os.cpu_count() or 4
        threads = min(8, max(2, cpu_count // 2))
        session_options.inter_op_num_threads = threads
        session_options.intra_op_num_threads = threads
        available_providers = onnxruntime.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available_providers:
            providers.insert(0, "CUDAExecutionProvider")
        self.whisperseg_session = onnxruntime.InferenceSession(
            model_path,
            providers=providers,
            sess_options=session_options,
        )
        self.log_callback(
            "Loaded WhisperSeg ONNX with providers: "
            f"{self.whisperseg_session.get_providers()}"
        )
        _, _, FeatureExtractor = _ensure_faster_whisper()
        self.whisperseg_feature_extractor = FeatureExtractor(
            feature_size=80,
            sampling_rate=16000,
            chunk_length=30,
        )
        return self.whisperseg_session

    def whisperseg_speech_probs(self, audio: np.ndarray, sampling_rate: int = 16000) -> np.ndarray:
        session = self.load_whisperseg_session()
        extractor = self.whisperseg_feature_extractor
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)

        chunk_samples = 30 * sampling_rate
        frame_samples = int(0.02 * sampling_rate)
        all_probs = []

        total_blocks = len(range(0, len(audio), chunk_samples))
        for idx, start in enumerate(range(0, len(audio), chunk_samples)):
            if total_blocks > 3 and (idx == 0 or (idx + 1) % 5 == 0 or idx == total_blocks - 1):
                self.log_callback(f"WhisperSeg VAD processing block {idx + 1}/{total_blocks} ({(idx + 1) / total_blocks * 100:.1f}%)")
                
            chunk = audio[start:start + chunk_samples]
            valid_frames = max(1, int(np.ceil(len(chunk) / frame_samples)))
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode="constant")
            elif len(chunk) > chunk_samples:
                chunk = chunk[:chunk_samples]

            features = extractor(chunk, padding=0)
            if features.shape[1] < 3000:
                features = np.pad(features, ((0, 0), (0, 3000 - features.shape[1])), mode="constant")
            elif features.shape[1] > 3000:
                features = features[:, :3000]

            input_name = session.get_inputs()[0].name
            output = session.run(None, {input_name: features[None, :, :].astype(np.float32, copy=False)})[0]
            logits = np.asarray(output, dtype=np.float32).reshape(-1)
            probs = 1.0 / (1.0 + np.exp(-logits))
            all_probs.append(probs[:valid_frames])

        if not all_probs:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(all_probs).astype(np.float32, copy=False)

    @staticmethod
    def frame_probs_to_segments(
        probs: np.ndarray,
        audio_samples: int,
        threshold: float,
        neg_threshold: float,
        min_speech_ms: int,
        min_silence_ms: int,
        speech_pad_ms: int,
        sampling_rate: int = 16000,
        frame_duration_ms: int = 20,
    ) -> list:
        segments = []
        if len(probs) == 0:
            return segments

        frame_samples = int(sampling_rate * frame_duration_ms / 1000)
        min_speech_frames = max(1, int(min_speech_ms / frame_duration_ms))
        min_silence_frames = max(1, int(min_silence_ms / frame_duration_ms))
        speech_pad_frames = int(speech_pad_ms / frame_duration_ms)

        triggered = False
        current_start = 0
        silence_start = None

        def add_segment(start_frame: int, end_frame: int):
            start_frame = max(0, start_frame - speech_pad_frames)
            end_frame = min(len(probs), end_frame + speech_pad_frames)
            if end_frame - start_frame >= min_speech_frames:
                start_sample = max(0, start_frame * frame_samples)
                end_sample = min(audio_samples, end_frame * frame_samples)
                if end_sample > start_sample:
                    segments.append((start_sample, end_sample))

        for index, prob in enumerate(probs):
            if not triggered and prob >= threshold:
                triggered = True
                current_start = index
                silence_start = None
                continue

            if not triggered:
                continue

            if prob < neg_threshold:
                if silence_start is None:
                    silence_start = index
                elif index - silence_start >= min_silence_frames:
                    add_segment(current_start, silence_start)
                    triggered = False
                    silence_start = None
            else:
                silence_start = None

        if triggered:
            add_segment(current_start, len(probs))

        return segments

    def whisperseg_segment_auditok_chunk(self, chunk: dict, sampling_rate: int = 16000) -> list:
        audio = chunk["array"]
        offset = chunk["offset_sec"]
        probs = self.whisperseg_speech_probs(audio, sampling_rate=sampling_rate)
        raw_segments = self.frame_probs_to_segments(
            probs,
            len(audio),
            threshold=self.vad_threshold,
            neg_threshold=self.vad_neg_threshold,
            min_speech_ms=WHISPERSEG_MIN_SPEECH_MS,
            min_silence_ms=WHISPERSEG_MIN_SILENCE_MS,
            speech_pad_ms=self.vad_speech_pad_ms,
            sampling_rate=sampling_rate,
        )
        if not raw_segments:
            return [chunk]

        merged = []
        current_start, current_end = raw_segments[0]
        max_samples = int(AUDITOK_MAX_DURATION * sampling_rate)
        merge_gap_samples = int(WHISPERSEG_MERGE_GAP_SECONDS * sampling_rate)

        for start, end in raw_segments[1:]:
            gap = start - current_end
            combined = end - current_start
            if gap <= merge_gap_samples and combined <= max_samples:
                current_end = end
                continue
            merged.append((current_start, current_end))
            current_start, current_end = start, end
        merged.append((current_start, current_end))

        segmented_chunks = []
        min_samples = int(WHISPERSEG_MIN_CHUNK_SECONDS * sampling_rate)
        for start, end in merged:
            # Pad rather than drop short detected speech (see split_audio_whisperseg).
            if end - start < min_samples:
                deficit = min_samples - (end - start)
                start = max(0, start - deficit // 2)
                end = min(len(audio), start + min_samples)
                start = max(0, end - min_samples)
            segmented_chunks.append({
                "array": audio[start:end].astype(np.float32, copy=False),
                "offset_sec": offset + start / sampling_rate,
                "duration_sec": (end - start) / sampling_rate,
            })

        return segmented_chunks or [chunk]

    def split_audio_auditok_whisperseg(self, audio_path: str, chunk_seconds: float):
        auditok_chunks = self.split_audio_auditok(audio_path, chunk_seconds)
        _, decode_audio, _ = _ensure_faster_whisper()
        audio = decode_audio(audio_path, sampling_rate=16000)
        if audio.ndim != 1:
            audio = np.asarray(audio).reshape(-1)

        probs = self.whisperseg_speech_probs(audio, sampling_rate=16000)
        # Kept for postprocess: per-line speech coverage (hallucination check).
        self.last_speech_probs = probs
        global_segments = self.frame_probs_to_segments(
            probs,
            len(audio),
            threshold=self.vad_threshold,
            neg_threshold=self.vad_neg_threshold,
            min_speech_ms=WHISPERSEG_MIN_SPEECH_MS,
            min_silence_ms=WHISPERSEG_MIN_SILENCE_MS,
            speech_pad_ms=self.vad_speech_pad_ms,
            sampling_rate=16000,
        )
        if not global_segments:
            self.log_callback("WhisperSeg found no speech; using auditok chunks")
            return auditok_chunks
        global_segments, removed_by_energy = self.filter_regions_by_rms(audio, global_segments)
        if not global_segments:
            self.log_callback("WhisperSeg speech was below RMS gate; using auditok chunks")
            return auditok_chunks

        segmented_chunks = []
        fallback_chunks = 0
        max_samples = int(AUDITOK_MAX_DURATION * 16000)
        merge_gap_samples = int(WHISPERSEG_MERGE_GAP_SECONDS * 16000)

        for chunk in auditok_chunks:
            chunk_start = int(chunk["offset_sec"] * 16000)
            chunk_end = chunk_start + int(chunk["duration_sec"] * 16000)
            overlaps = []
            for speech_start, speech_end in global_segments:
                start = max(chunk_start, speech_start)
                end = min(chunk_end, speech_end)
                if end > start:
                    overlaps.append((start, end))

            if not overlaps:
                segmented_chunks.append(chunk)
                fallback_chunks += 1
                continue

            merged = []
            current_start, current_end = overlaps[0]
            for start, end in overlaps[1:]:
                gap = start - current_end
                combined = end - current_start
                if gap <= merge_gap_samples and combined <= max_samples:
                    current_end = end
                    continue
                merged.append((current_start, current_end))
                current_start, current_end = start, end
            merged.append((current_start, current_end))

            if len(merged) > 8:
                segmented_chunks.append(chunk)
                fallback_chunks += 1
                continue

            min_samples = int(WHISPERSEG_MIN_CHUNK_SECONDS * 16000)
            added = 0
            for start, end in merged:
                # Pad rather than drop short detected speech (see split_audio_whisperseg).
                if end - start < min_samples:
                    deficit = min_samples - (end - start)
                    start = max(0, start - deficit // 2)
                    end = min(len(audio), start + min_samples)
                    start = max(0, end - min_samples)
                segmented_chunks.append({
                    "array": audio[start:end].astype(np.float32, copy=False),
                    "offset_sec": start / 16000.0,
                    "duration_sec": (end - start) / 16000.0,
                })
                added += 1

            if added == 0:
                segmented_chunks.append(chunk)
                fallback_chunks += 1

        kept_duration = sum(chunk["duration_sec"] for chunk in segmented_chunks)
        auditok_duration = sum(chunk["duration_sec"] for chunk in auditok_chunks)
        self.log_callback(
            f"WhisperSeg refined {len(auditok_chunks)} auditok chunks "
            f"to {len(segmented_chunks)} ASR chunks, "
            f"kept {kept_duration:.2f}s / {auditok_duration:.2f}s "
            f"(speech_regions={len(global_segments)}, fallback_chunks={fallback_chunks}, "
            f"sensitivity={self.vad_sensitivity}, threshold={self.vad_threshold}, "
            f"rms_gate_removed={removed_by_energy})"
        )
        return segmented_chunks

    def split_audio_whisperseg(self, audio_path: str, chunk_seconds: float):
        _, decode_audio, _ = _ensure_faster_whisper()
        audio = decode_audio(audio_path, sampling_rate=16000)
        if audio.ndim != 1:
            audio = np.asarray(audio).reshape(-1)

        probs = self.whisperseg_speech_probs(audio, sampling_rate=16000)
        # Kept for postprocess: per-line speech coverage (hallucination check).
        self.last_speech_probs = probs
        speech_regions = self.frame_probs_to_segments(
            probs,
            len(audio),
            threshold=self.vad_threshold,
            neg_threshold=self.vad_neg_threshold,
            min_speech_ms=WHISPERSEG_MIN_SPEECH_MS,
            min_silence_ms=WHISPERSEG_MIN_SILENCE_MS,
            speech_pad_ms=self.vad_speech_pad_ms,
            sampling_rate=16000,
        )
        speech_regions, removed_by_energy = self.filter_regions_by_rms(audio, speech_regions)

        if not speech_regions:
            self.log_callback("WhisperSeg found no speech; falling back to fixed chunks")
            return self.split_audio(audio_path, chunk_seconds)

        chunks = []
        current_start, current_end = speech_regions[0]
        max_samples = int(min(chunk_seconds, AUDITOK_MAX_DURATION) * 16000)
        merge_gap_samples = int(WHISPERSEG_MERGE_GAP_SECONDS * 16000)
        min_samples = int(WHISPERSEG_MIN_CHUNK_SECONDS * 16000)

        def add_chunk(start: int, end: int):
            # A region WhisperSeg already classified as speech must never be
            # dropped just for being short -- that silently loses real, often
            # isolated/quiet lines (e.g. a 1s reply between long pauses). Pad a
            # too-short region with surrounding audio up to the minimum window so
            # it still reaches the ASR; any overlap is removed later by dedup.
            if end - start < min_samples:
                deficit = min_samples - (end - start)
                start = max(0, start - deficit // 2)
                end = min(len(audio), start + min_samples)
                start = max(0, end - min_samples)
            chunks.append({
                "array": audio[start:end].astype(np.float32, copy=False),
                "offset_sec": start / 16000.0,
                "duration_sec": (end - start) / 16000.0,
            })

        for start, end in speech_regions[1:]:
            gap = start - current_end
            combined = end - current_start
            if gap <= merge_gap_samples and combined <= max_samples:
                current_end = end
                continue
            add_chunk(current_start, current_end)
            current_start, current_end = start, end
        add_chunk(current_start, current_end)

        if not chunks:
            self.log_callback("WhisperSeg speech regions were too short; falling back to fixed chunks")
            return self.split_audio(audio_path, chunk_seconds)

        total_duration = len(audio) / 16000.0
        kept_duration = sum(chunk["duration_sec"] for chunk in chunks)
        self.log_callback(
            f"WhisperSeg split into {len(speech_regions)} speech regions, "
            f"coalesced to {len(chunks)} ASR chunks, "
            f"kept {kept_duration:.2f}s / {total_duration:.2f}s "
            f"(sensitivity={self.vad_sensitivity}, threshold={self.vad_threshold}, "
            f"rms_gate_removed={removed_by_energy})"
        )
        return chunks

    @staticmethod
    def segment_rms_db(audio: np.ndarray) -> float:
        if audio.size == 0:
            return -200.0
        rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))
        return 20.0 * np.log10(rms + 1e-10)

    def filter_regions_by_rms(self, audio: np.ndarray, regions: list, sampling_rate: int = 16000) -> tuple:
        gate_db = self.vad_rms_gate_db
        if not ENABLE_RMS_SPEECH_GATE or gate_db is None or not regions:
            return regions, 0

        filtered = []
        removed = 0
        for start, end in regions:
            region = audio[max(0, start):min(len(audio), end)]
            duration = (end - start) / sampling_rate
            rms_db = SubtitleGenerator.segment_rms_db(region)
            if rms_db < gate_db and duration < 4.0:
                removed += 1
                continue
            filtered.append((start, end))
        return filtered, removed

    def split_audio_for_profile(self, audio_path: str, chunk_seconds: float, profile_name: str):
        if SCENE_SPLIT_METHOD == "whisperseg":
            return self.split_audio_whisperseg(audio_path, chunk_seconds)
        if SCENE_SPLIT_METHOD == "auditok_whisperseg":
            return self.split_audio_auditok_whisperseg(audio_path, chunk_seconds)
        if SCENE_SPLIT_METHOD == "auditok":
            return self.split_audio_auditok(audio_path, chunk_seconds)
        if SCENE_SPLIT_METHOD not in {"fixed", "none"}:
            self.log_callback(f"Unknown SCENE_SPLIT_METHOD={SCENE_SPLIT_METHOD!r}; using fixed chunks")
        return self.split_audio(audio_path, chunk_seconds)

    @staticmethod
    def tighten_word_start(start: float, end: float, audio,
                           sampling_rate: int = 16000) -> float:
        """Move a word's start time past leading silence inside [start, end).

        DTW word alignment can absorb an inter-utterance pause into the word
        that follows it, so the pause never appears as an inter-word gap (see
        WORD_PAUSE_SPLIT_SECONDS). Only long words are examined, and the start
        only ever moves forward, keeping a small onset pad.
        """
        if audio is None or end - start < WORD_TIGHTEN_MIN_WORD_SECONDS:
            return start
        frame = max(1, int(sampling_rate * 0.02))
        lo = max(0, int(start * sampling_rate))
        hi = min(len(audio), int(end * sampling_rate))
        span = np.asarray(audio[lo:hi], dtype=np.float64)
        frames = len(span) // frame
        if frames < 2:
            return start
        rms = np.sqrt((span[: frames * frame].reshape(frames, frame) ** 2).mean(axis=1))
        voiced = np.nonzero(20 * np.log10(rms + 1e-9) >= WORD_TIGHTEN_SILENCE_DB)[0]
        if len(voiced) == 0:
            return start
        lead = voiced[0] * frame / sampling_rate
        if lead < WORD_TIGHTEN_MIN_LEAD_SECONDS:
            return start
        return start + lead - WORD_TIGHTEN_ONSET_PAD_SECONDS

    @staticmethod
    def span_peak_db(audio, start: float, end: float,
                     sampling_rate: int = 16000):
        """Loudest 20ms RMS frame (dBFS) inside [start, end), or None."""
        if audio is None:
            return None
        frame = max(1, int(sampling_rate * 0.02))
        lo = max(0, int(start * sampling_rate))
        hi = min(len(audio), int(end * sampling_rate))
        span = np.asarray(audio[lo:hi], dtype=np.float64)
        frames = len(span) // frame
        if frames < 1:
            return None
        rms = np.sqrt((span[: frames * frame].reshape(frames, frame) ** 2).mean(axis=1))
        return float(20 * np.log10(rms.max() + 1e-9))

    def span_speech_coverage(self, start: float, end: float):
        """Fraction of WhisperSeg 20ms frames >= vad_neg_threshold in the
        absolute time span [start, end), or None when no global probs exist
        (fixed/auditok-only chunking, whole-file mode)."""
        probs = getattr(self, "last_speech_probs", None)
        if probs is None or len(probs) == 0:
            return None
        lo = max(0, int(start / 0.02))
        hi = min(len(probs), int(np.ceil(end / 0.02)))
        if hi <= lo:
            return None
        return float((probs[lo:hi] >= self.vad_neg_threshold).mean())

    @staticmethod
    def extend_entry_tails(entries: list, audio, offset: float,
                           sampling_rate: int = 16000) -> None:
        """Extend word-anchored ends through trailing voiced audio, in place.

        DTW word-end stamps are systematically early while the voice is still
        decaying (see TAIL_EXTEND_MAX_SECONDS). Walk 20ms frames past each
        entry's end while they stay voiced, capped and stopped short of the
        next entry; entries keep their order and never overlap.
        """
        if audio is None or not entries:
            return
        frame = max(1, int(sampling_rate * 0.02))
        chunk_end = offset + len(audio) / sampling_rate
        ordered = sorted(entries, key=lambda e: (e["start"], e["end"]))
        for index, entry in enumerate(ordered):
            if not entry.get("anchored"):
                continue
            next_start = (
                ordered[index + 1]["start"] - TAIL_EXTEND_GUARD_SECONDS
                if index + 1 < len(ordered) else chunk_end
            )
            limit = min(entry["end"] + TAIL_EXTEND_MAX_SECONDS, next_start, chunk_end)
            if limit <= entry["end"]:
                continue
            pos = max(0, int((entry["end"] - offset) * sampling_rate))
            end_pos = min(len(audio), int((limit - offset) * sampling_rate))
            new_end = entry["end"]
            silence_run = 0.0
            while pos + frame <= end_pos:
                rms = float(np.sqrt(np.mean(np.square(
                    np.asarray(audio[pos:pos + frame], dtype=np.float64)))))
                pos += frame
                if 20 * np.log10(rms + 1e-9) < TAIL_EXTEND_VOICED_DB:
                    # Ride through short intra-word dips (glottal stops);
                    # only a real silence run ends the scan.
                    silence_run += frame / sampling_rate
                    if silence_run > TAIL_EXTEND_MAX_SILENCE_RUN_SECONDS:
                        break
                    continue
                silence_run = 0.0
                new_end = offset + pos / sampling_rate
            if new_end > entry["end"]:
                entry["end"] = min(new_end + TAIL_EXTEND_RELEASE_PAD_SECONDS, limit)

    @staticmethod
    def find_silence_runs(audio, start: float, end: float,
                          sampling_rate: int = 16000) -> list:
        """Contiguous sub-ENTRY_SPLIT_SILENCE_DB runs >= the minimum length
        inside [start, end) of chunk-relative audio, as (start, end) seconds."""
        frame = max(1, int(sampling_rate * 0.02))
        lo = max(0, int(start * sampling_rate))
        hi = min(len(audio), int(end * sampling_rate))
        span = np.asarray(audio[lo:hi], dtype=np.float64)
        frames = len(span) // frame
        if frames < 1:
            return []
        rms = np.sqrt((span[: frames * frame].reshape(frames, frame) ** 2).mean(axis=1))
        silent = 20 * np.log10(rms + 1e-9) < ENTRY_SPLIT_SILENCE_DB
        frame_sec = frame / sampling_rate
        runs = []
        run_start = None
        for index, is_silent in enumerate(silent):
            if is_silent:
                if run_start is None:
                    run_start = index
                continue
            if run_start is not None:
                if (index - run_start) * frame_sec >= ENTRY_SPLIT_MIN_SILENCE_SECONDS:
                    runs.append((start + run_start * frame_sec, start + index * frame_sec))
                run_start = None
        if run_start is not None and (frames - run_start) * frame_sec >= ENTRY_SPLIT_MIN_SILENCE_SECONDS:
            runs.append((start + run_start * frame_sec, start + frames * frame_sec))
        return runs

    def split_group_on_silence(self, group: list, chunk_audio,
                               sampling_rate: int = 16000) -> list:
        """Split one word group at long internal silences, energy-first.

        DTW can distribute a pause across several word boundaries so that no
        inter-word gap or over-long word reveals it (see
        ENTRY_SPLIT_MIN_SILENCE_SECONDS). Words go to the side their midpoint
        falls on; piece edges are clamped to the silence run boundaries.
        Returns a list of (words, start, end) pieces.
        """
        start, end = group[0][0], group[-1][1]
        default = [(group, start, end)]
        if chunk_audio is None or len(group) < 2:
            return default
        runs = self.find_silence_runs(chunk_audio, start, end, sampling_rate)
        if not runs:
            return default
        centers = [(run_start + run_end) / 2 for run_start, run_end in runs]
        buckets: list[list] = [[] for _ in range(len(runs) + 1)]
        for word in group:
            midpoint = (word[0] + word[1]) / 2
            buckets[sum(1 for center in centers if center < midpoint)].append(word)
        pieces = []
        for index, bucket in enumerate(buckets):
            if not bucket:
                continue
            piece_start = bucket[0][0]
            piece_end = bucket[-1][1]
            if index > 0:
                piece_start = max(piece_start, runs[index - 1][1] - ENTRY_SPLIT_EDGE_PAD_SECONDS)
            if index < len(runs):
                piece_end = min(piece_end, runs[index][0] + ENTRY_SPLIT_EDGE_PAD_SECONDS)
            if piece_end - piece_start >= 0.1:
                pieces.append((bucket, piece_start, piece_end))
        return pieces or default

    def collect_segment_entries(self, segment, offset: float, chunk_audio=None,
                                sampling_rate: int = 16000) -> tuple[list, int]:
        """Convert one decoded segment into raw subtitle dict(s), absolute times.

        Prefers word-level times (DTW alignment) over the decoder's segment
        times: on difficult audio (quiet lines, climax loops) the decoder often
        stamps a line several seconds early, at the chunk start, while the word
        timestamps stay anchored to the audio (debug session 2026-07-09).
        Word-anchored segments are additionally split at intra-segment pauses
        (inter-word gap > WORD_PAUSE_SPLIT_SECONDS, after tighten_word_start)
        so one subtitle line never spans a long silence. Returns
        ``(entries, moved_far)`` where moved_far counts anchors that moved the
        start by more than 0.5s from the decoder's own stamp.
        """
        clean_text = self.clean_text(segment.text)
        if not clean_text:
            return [], 0
        base = {
            "avg_logprob": getattr(segment, "avg_logprob", None),
            "no_speech_prob": getattr(segment, "no_speech_prob", None),
        }

        def annotate(entries: list) -> list:
            # Acoustic evidence for the postprocess hallucination checks;
            # entry times are absolute, chunk_audio is chunk-relative.
            for entry in entries:
                entry["peak_db"] = self.span_peak_db(
                    chunk_audio, entry["start"] - offset, entry["end"] - offset, sampling_rate
                )
                entry["speech_coverage"] = self.span_speech_coverage(entry["start"], entry["end"])
            return entries

        words = [
            (float(w.start), float(w.end), w.word)
            for w in (getattr(segment, "words", None) or [])
            if w.start is not None and w.end is not None
        ]
        def word_dicts(piece_words: list) -> list:
            # Absolute-time word list; kept on every entry so duration-aligned
            # consumers (tool_clonevoice dubbing) get per-word timing for free.
            return [
                {"word": text, "start": offset + start, "end": offset + end}
                for start, end, text in piece_words
            ]

        if not words or words[-1][1] - words[0][0] < 0.05:
            return annotate([{
                "start": offset + float(segment.start),
                "end": offset + float(segment.end),
                "text": clean_text,
                "anchored": False,
                "words": word_dicts(words),
                **base,
            }]), 0

        moved_far = 1 if abs(words[0][0] - float(segment.start)) > 0.5 else 0
        groups: list[list[tuple[float, float, str]]] = []
        for word_start, word_end, word_text in words:
            word_start = self.tighten_word_start(word_start, word_end, chunk_audio, sampling_rate)
            if groups and word_start - groups[-1][-1][1] <= WORD_PAUSE_SPLIT_SECONDS:
                groups[-1].append((word_start, word_end, word_text))
            else:
                groups.append([(word_start, word_end, word_text)])

        pieces = []
        for group in groups:
            pieces.extend(self.split_group_on_silence(group, chunk_audio, sampling_rate))

        entries = []
        for piece_words, piece_start, piece_end in pieces:
            text = clean_text if len(pieces) == 1 else self.clean_text("".join(w[2] for w in piece_words))
            if not text:
                continue
            entries.append({
                "start": offset + piece_start,
                "end": offset + piece_end,
                "text": text,
                "anchored": True,
                "words": word_dicts(piece_words),
                **base,
            })
        if not entries:
            entries = [{
                "start": offset + words[0][0],
                "end": offset + words[-1][1],
                "text": clean_text,
                "anchored": True,
                "words": word_dicts(words),
                **base,
            }]
        return annotate(entries), moved_far

    @staticmethod
    def is_known_hallucination(
        text: str,
        start: float,
        end: float,
        total_end: float,
        avg_logprob=None,
        no_speech_prob=None,
    ) -> bool:
        norm = SubtitleGenerator.normalize_for_duplicate(text)
        duration = max(0.0, end - start)
        is_low_confidence = (
            (avg_logprob is not None and avg_logprob < -0.85)
            or (no_speech_prob is not None and no_speech_prob > 0.55)
        )
        near_edge = start < 45.0 or (total_end > 0.0 and total_end - end < 120.0)

        for phrase in HARD_HALLUCINATION_NORMS:
            if norm == phrase:
                return True
            if phrase in norm and len(norm) <= len(phrase) + 8:
                return True
            if phrase in norm and (duration <= 8.0 or is_low_confidence or near_edge):
                return True
            # The decoder sometimes drops the leading character(s) of a stock
            # outro (視聴ありがとうございました, mdvr-433 #2 entry 18); a norm
            # that is nearly the whole phrase is the same hallucination.
            if len(norm) >= max(6, len(phrase) - 2) and norm in phrase:
                return True

        if norm in SHORT_HALLUCINATION_NORMS:
            return duration <= 2.6 and (is_low_confidence or near_edge)

        return False

    def silence_floor_db(self) -> float:
        """Silence floor for the no_audio_energy check: stay well below the VAD
        RMS gate for the active sensitivity so a line the VAD would keep can
        never be killed for quietness alone."""
        rms_gate_db = getattr(self, "vad_rms_gate_db", RMS_SPEECH_MIN_DB)
        if rms_gate_db is None:
            return HALLUCINATION_SILENCE_PEAK_DB_NO_GATE
        return min(
            HALLUCINATION_SILENCE_PEAK_DB,
            rms_gate_db - HALLUCINATION_SILENCE_FLOOR_MARGIN_DB,
        )

    def acoustic_removal_reason(self, item: dict):
        """Acoustic hallucination verdict for one line, or None to keep it.

        Shared by tool_subtitle postprocess and tool_clonevoice's dubbing
        filter. Expects the peak_db / speech_coverage stats attached by
        collect_segment_entries; lines without stats are never removed here.
        """
        avg_logprob = item.get("avg_logprob")
        no_speech_prob = item.get("no_speech_prob")
        low_confidence = (
            (avg_logprob is not None and avg_logprob < HALLUCINATION_LOW_CONFIDENCE_LOGPROB)
            or (no_speech_prob is not None and no_speech_prob > HALLUCINATION_LOW_CONFIDENCE_NO_SPEECH)
        )
        peak_db = item.get("peak_db")
        if (
            peak_db is not None
            and peak_db < self.silence_floor_db()
            and (item.get("anchored", False) or low_confidence)
        ):
            return "no_audio_energy"
        if (
            peak_db is not None
            and no_speech_prob is not None
            and (
                (no_speech_prob > HALLUCINATION_QUIET_NO_SPEECH
                 and peak_db < HALLUCINATION_QUIET_PEAK_DB)
                or (no_speech_prob > HALLUCINATION_QUIET_STRONG_NO_SPEECH
                    and peak_db < HALLUCINATION_QUIET_STRONG_PEAK_DB)
            )
        ):
            return "quiet_no_speech"
        norm = SubtitleGenerator.normalize_for_duplicate(item.get("text", ""))
        duration = item["end"] - item["start"]
        if (
            item.get("anchored", False)
            and no_speech_prob is not None
            and no_speech_prob > HALLUCINATION_SPARSE_NO_SPEECH
            and duration > max(
                HALLUCINATION_SPARSE_MIN_SECONDS,
                HALLUCINATION_SPARSE_FACTOR * (len(norm) / 5.2 + 0.8),
            )
        ):
            return "sparse_text"
        coverage = item.get("speech_coverage")
        if (
            coverage is not None
            and coverage < HALLUCINATION_MIN_SPEECH_COVERAGE
            and low_confidence
        ):
            return "low_speech_coverage"
        return None

    def postprocess_segments(self, raw_segments: list, scene_mode: bool = False,
                             removal_log: list | None = None) -> list:
        filtered = []
        removed_duplicates = 0
        removed_noise = 0
        removed_hallucinations = 0
        removed_low_confidence = 0
        removed_acoustic: dict[str, int] = {}
        compressed_repetitions = 0
        total_end = max((item.get("end", 0.0) for item in raw_segments), default=0.0)
        duplicate_window = 10.0 if scene_mode else DUPLICATE_LOOKBACK_SECONDS
        duplicate_threshold = 0.91 if scene_mode else NEAR_DUPLICATE_RATIO

        def log_removal(reason: str, item: dict):
            if removal_log is not None:
                removal_log.append({"reason": reason, **item})

        def find_duplicate_index(item: dict, norm: str):
            # Only a pair that covers the SAME audio can be a duplicated decode
            # (padded short WhisperSeg chunks and auditok/fixed fallback chunks
            # overlap their neighbours). Similar or identical text at disjoint
            # times is real repeated dialogue and must be kept — the old
            # window-based text matching silently deleted such lines.
            for index in range(len(filtered) - 1, -1, -1):
                previous = filtered[index]
                if item["start"] - previous["start"] > duplicate_window:
                    break

                time_overlap = min(item["end"], previous["end"]) - max(item["start"], previous["start"])
                if time_overlap <= 0.15:
                    continue

                previous_norm = SubtitleGenerator.normalize_for_duplicate(previous["text"])
                if norm == previous_norm:
                    return index
                if SubtitleGenerator.duplicate_strength(norm, previous_norm) >= duplicate_threshold:
                    return index

            return None

        for item in sorted(raw_segments, key=lambda seg: (seg["start"], seg["end"])):
            text = item["text"]
            norm = SubtitleGenerator.normalize_for_duplicate(text)
            duration = item["end"] - item["start"]
            avg_logprob = item.get("avg_logprob")
            no_speech_prob = item.get("no_speech_prob")

            if not text or not HAS_LINGUISTIC_CONTENT_RE.search(text):
                removed_noise += 1
                log_removal("no_linguistic_content", item)
                continue
            if SubtitleGenerator.is_repetition_noise(text):
                compressed = SubtitleGenerator.compress_repetition_text(text)
                if compressed != text and not SubtitleGenerator.is_repetition_noise(compressed):
                    # Real repeated dialogue: keep it, compressed to a readable form.
                    item["text"] = compressed
                    text = compressed
                    norm = SubtitleGenerator.normalize_for_duplicate(text)
                    compressed_repetitions += 1
                else:
                    removed_noise += 1
                    log_removal("repetition_noise", item)
                    continue
            if SubtitleGenerator.is_known_hallucination(
                text,
                item["start"],
                item["end"],
                total_end,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
            ):
                removed_hallucinations += 1
                log_removal("hallucination", item)
                continue
            if (
                no_speech_prob is not None
                and avg_logprob is not None
                and no_speech_prob > 0.90
                and avg_logprob < -1.35
            ):
                removed_low_confidence += 1
                log_removal("low_confidence", item)
                continue

            acoustic_reason = self.acoustic_removal_reason(item)
            if acoustic_reason:
                removed_acoustic[acoustic_reason] = removed_acoustic.get(acoustic_reason, 0) + 1
                log_removal(acoustic_reason, item)
                continue

            if duration > 0:
                # The reading-speed cap may only shrink decoder-timed lines; a
                # word-anchored end is acoustic evidence, and cutting it drops
                # real trailing speech (mdvr-433 entry 3 lost its last 2s).
                item["end"] = item["start"] + SubtitleGenerator.subtitle_duration_for_text(
                    text,
                    duration,
                    allow_shrink=not item.get("anchored", False),
                )

            duplicate_index = find_duplicate_index(item, norm)
            if duplicate_index is not None:
                previous = filtered[duplicate_index]
                previous_norm = SubtitleGenerator.normalize_for_duplicate(previous["text"])
                should_replace = (
                    len(norm) > len(previous_norm) + 2
                    or (
                        len(norm) >= len(previous_norm)
                        and item.get("avg_logprob", -99.0) > previous.get("avg_logprob", -99.0) + 0.15
                    )
                )
                if should_replace:
                    log_removal("overlap_duplicate_replaced", previous)
                    filtered[duplicate_index] = item
                else:
                    log_removal("overlap_duplicate", item)
                removed_duplicates += 1
                continue

            if filtered and item["start"] < filtered[-1]["end"]:
                item["start"] = max(item["start"], filtered[-1]["end"] + 0.03)
                if item["end"] <= item["start"]:
                    item["end"] = item["start"] + SubtitleGenerator.subtitle_duration_for_text(text, 2.0)

            filtered.append(item)

        self.log_callback(
            "Postprocess removed "
            f"{removed_duplicates} duplicates, {removed_noise} noisy/repeated lines, "
            f"{removed_hallucinations} known hallucinations, "
            f"{removed_low_confidence} low-confidence silence lines, "
            f"{sum(removed_acoustic.values())} acoustic hallucinations "
            f"({removed_acoustic or 'none'}, silence floor {self.silence_floor_db():.0f}dB); "
            f"compressed {compressed_repetitions} repetition lines"
        )
        return self.merge_adjacent_fragments(filtered)

    def merge_adjacent_fragments(self, lines: list) -> list:
        """Merge decoder-fragmented sentences back into single lines.

        Kotoba often emits one sentence as several adjacent segments with a
        zero gap (通常コースで | よろしかったですか?). Only near-contiguous
        neighbours are merged, and only while the combined text stays a
        readable single line, so real pauses and full sentences (final
        punctuation) keep their own line.
        """
        merged: list = []
        merged_count = 0
        for item in lines:
            if merged:
                previous = merged[-1]
                gap = item["start"] - previous["end"]
                combined_text = previous["text"] + item["text"]
                if (
                    gap <= MERGE_MAX_GAP_SECONDS
                    and not previous["text"].endswith(MERGE_SENTENCE_FINAL_CHARS)
                    and len(SubtitleGenerator.normalize_for_duplicate(combined_text)) <= MERGE_MAX_CHARS
                    and item["end"] - previous["start"] <= MERGE_MAX_DURATION_SECONDS
                ):
                    previous["text"] = combined_text
                    previous["end"] = max(previous["end"], item["end"])
                    merged_count += 1
                    continue
            merged.append(item)
        if merged_count:
            self.log_callback(f"Merged {merged_count} adjacent sentence fragments")
        return merged

    @staticmethod
    def is_scene_split_enabled() -> bool:
        return SCENE_SPLIT_METHOD in {"auditok", "auditok_whisperseg", "whisperseg"}

    def base_asr_options(self, scene_mode: bool = False) -> dict:
        if scene_mode:
            asr_options = dict(KOTOBA_SCENE_OPTIONS)
            asr_options.update(MODEL_SCENE_OVERRIDES.get(self.model_preset, {}))
            return asr_options

        asr_options = dict(KOTOBA_BALANCED_OPTIONS)
        asr_options.update(WIDE_INTAKE_OVERRIDES)
        return asr_options

    def transcribe_profile(self, audio_file: str, profile_name: str,
                           removal_log: list | None = None) -> list:
        if profile_name not in PROFILE_CONFIGS:
            raise ValueError(f"Unknown ASR profile: {profile_name}")

        profile = PROFILE_CONFIGS[profile_name]
        raw_segments = []
        self.last_chunks = []
        # Stale probs from a previous file must not feed this file's speech
        # coverage checks; the whisperseg split paths repopulate this.
        self.last_speech_probs = None
        reanchored_far = 0
        scene_mode = self.is_scene_split_enabled()
        asr_options = self.base_asr_options(scene_mode=scene_mode)
        asr_options.update(profile["options"])
        chunk_seconds = profile["chunk_seconds"]

        self.log_callback(
            f"Running ASR profile '{profile_name}' "
            f"(chunk={chunk_seconds:.1f}s, scene_split={SCENE_SPLIT_METHOD}, "
            f"model={self.model_preset}, "
            f"scene_internal_vad={SCENE_INTERNAL_VAD if scene_mode else False})"
        )

        if TRANSCRIBE_MODE == "external_vad":
            vad_segments = self.apply_vad(audio_file)
            if not vad_segments:
                self.log_callback(f"No speech detected in {audio_file}. Creating empty SRT.")

            for vad_index, vad_segment in enumerate(vad_segments, start=1):
                offset = vad_segment["vad_start_sec"]
                duration = vad_segment["vad_end_sec"] - vad_segment["vad_start_sec"]
                self.log_callback(
                    f"Transcribing VAD segment {vad_index}/{len(vad_segments)} "
                    f"at {offset:.2f}s, duration {duration:.2f}s"
                )

                segments, info = self.model.transcribe(
                    vad_segment["array"],
                    language="ja",
                    task="transcribe",
                    vad_filter=False,
                    **{
                        key: value
                        for key, value in asr_options.items()
                        if key != "vad_parameters"
                    },
                )

                vad_entries = []
                for segment in segments:
                    entries, moved_far = self.collect_segment_entries(
                        segment, offset, chunk_audio=vad_segment["array"]
                    )
                    reanchored_far += moved_far
                    vad_entries.extend(entries)
                self.extend_entry_tails(vad_entries, vad_segment["array"], offset)
                raw_segments.extend(vad_entries)
        elif TRANSCRIBE_MODE == "chunked":
            chunks = self.split_audio_for_profile(audio_file, chunk_seconds, profile_name)
            self.last_chunks = [
                {"offset_sec": c["offset_sec"], "duration_sec": c["duration_sec"]}
                for c in chunks
            ]
            total_chunks = len(chunks)
            for chunk_index, chunk in enumerate(chunks, start=1):
                offset = chunk["offset_sec"]
                self.log_callback(
                    f"Transcribing chunk {chunk_index}/{total_chunks} "
                    f"at {offset:.2f}s, duration {chunk['duration_sec']:.2f}s"
                )

                segments, info = self.model.transcribe(
                    chunk["array"],
                    language="ja",
                    task="transcribe",
                    vad_filter=scene_mode and SCENE_INTERNAL_VAD,
                    vad_parameters=(
                        asr_options["vad_parameters"]
                        if scene_mode and SCENE_INTERNAL_VAD
                        else None
                    ),
                    **{
                        key: value
                        for key, value in asr_options.items()
                        if key != "vad_parameters"
                    },
                )

                chunk_entries = []
                for segment in segments:
                    entries, moved_far = self.collect_segment_entries(
                        segment, offset, chunk_audio=chunk["array"]
                    )
                    reanchored_far += moved_far
                    chunk_entries.extend(entries)
                self.extend_entry_tails(chunk_entries, chunk["array"], offset)
                raw_segments.extend(chunk_entries)
        else:
            segments, info = self.model.transcribe(
                audio_file,
                language="ja",
                task="transcribe",
                vad_filter=EnableVAD or TRANSCRIBE_MODE == "internal_vad",
                vad_parameters=(
                    KOTOBA_BALANCED_OPTIONS["vad_parameters"]
                    if EnableVAD or TRANSCRIBE_MODE == "internal_vad"
                    else None
                ),
                **{
                    key: value
                    for key, value in asr_options.items()
                    if key != "vad_parameters"
                },
            )

            self.log_callback(
                f"Detected language: {info.language} "
                f"(probability={info.language_probability:.2f}); "
                f"duration_after_vad={getattr(info, 'duration_after_vad', None)}"
            )

            for segment in segments:
                entries, moved_far = self.collect_segment_entries(segment, 0.0)
                reanchored_far += moved_far
                raw_segments.extend(entries)

        if reanchored_far:
            self.log_callback(
                f"Word-anchored timestamps moved {reanchored_far} segment(s) by more than 0.5s"
            )
        # Copies, not references: postprocess mutates these dicts in place
        # (duration cap, overlap trimming) and .raw.srt must keep showing the
        # decoder output, not the postprocessed times.
        self.last_raw_segments = [
            dict(item) for item in sorted(raw_segments, key=lambda seg: (seg["start"], seg["end"]))
        ]
        return self.postprocess_segments(raw_segments, scene_mode=scene_mode, removal_log=removal_log)
    @staticmethod
    def write_srt(segments: list, output_file: str):
        lines = []
        for index, segment in enumerate(segments, start=1):
            lines.extend([
                str(index),
                f"{SubtitleGenerator.format_timestamp(segment['start'])} --> {SubtitleGenerator.format_timestamp(segment['end'])}",
                segment["text"],
                ""
            ])

        with open(output_file, "w", encoding="utf-8") as file:
            file.write("\n".join(lines))

    def debug_base_path(self, output_file: str) -> str:
        """foo.jp.srt / foo.srt -> <dir>/foo_debug/foo (folder is created)."""
        path = Path(output_file)
        name = path.name
        if name.lower().endswith(".jp.srt"):
            stem = name[:-7]
        else:
            stem = path.stem
        debug_dir = path.parent / f"{stem}_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        return str(debug_dir / stem)

    def write_debug_files(self, output_file: str, removal_log: list):
        """Debug files for auditing dropped speech (enabled by the GUI checkbox),
        written into a per-video ``<stem>_debug`` folder next to the output SRT.

        .raw.srt     decoder output before postprocess (word-anchored and
                     pause-split, see collect_segment_entries), with
                     confidence values;
        .vad.srt     each ASR chunk as one entry — shows exactly which time
                     ranges the VAD sent to the decoder (a missing interval here
                     means VAD miss; present here but absent in .raw.srt means
                     the decoder produced nothing; present in .raw.srt but not
                     in the final SRT means postprocess removed it);
        .removed.srt lines postprocess removed, prefixed with the reason.
        """
        base = self.debug_base_path(output_file)

        raw_entries = []
        for item in self.last_raw_segments:
            lp = item.get("avg_logprob")
            ns = item.get("no_speech_prob")
            pk = item.get("peak_db")
            cov = item.get("speech_coverage")
            lp_text = f"{lp:.2f}" if lp is not None else "n/a"
            ns_text = f"{ns:.2f}" if ns is not None else "n/a"
            pk_text = f"{pk:.0f}" if pk is not None else "n/a"
            cov_text = f"{cov:.2f}" if cov is not None else "n/a"
            raw_entries.append({
                "start": item["start"],
                "end": item["end"],
                "text": f"{item['text']} {{lp={lp_text} ns={ns_text} pk={pk_text} cov={cov_text}}}",
            })
        self.write_srt(raw_entries, f"{base}.raw.srt")

        chunk_entries = [
            {
                "start": chunk["offset_sec"],
                "end": chunk["offset_sec"] + chunk["duration_sec"],
                "text": f"chunk {index}/{len(self.last_chunks)} dur={chunk['duration_sec']:.2f}s",
            }
            for index, chunk in enumerate(self.last_chunks, start=1)
        ]
        self.write_srt(chunk_entries, f"{base}.vad.srt")

        removed_entries = [
            {
                "start": item["start"],
                "end": item["end"],
                "text": f"[{item['reason']}] {item['text']}",
            }
            for item in sorted(removal_log, key=lambda it: (it["start"], it["end"]))
        ]
        self.write_srt(removed_entries, f"{base}.removed.srt")

        self.log_callback(
            f"Debug files saved to {os.path.dirname(base)}: "
            f".raw.srt ({len(raw_entries)} raw), .vad.srt ({len(chunk_entries)} chunks), "
            f".removed.srt ({len(removed_entries)} removed)"
        )

    def retain_debug_audio(self, audio_file: str, output_file: str) -> str:
        """Move the ASR WAV into the per-video debug directory."""
        destination = f"{self.debug_base_path(output_file)}.wav"
        os.replace(audio_file, destination)
        self.log_callback(f"Debug WAV saved: {destination}")
        return destination

    def transcribe(self, audio_file: str, output_file: str, debug_files: bool = False):
        start_time = time.time()

        removal_log: list | None = [] if debug_files else None
        final_segments = self.transcribe_profile(audio_file, "stable", removal_log=removal_log)

        if not final_segments:
            self.log_callback(f"No speech detected in {audio_file}. Creating empty SRT.")

        self.write_srt(final_segments, output_file)

        if debug_files:
            try:
                self.write_debug_files(output_file, removal_log or [])
            except Exception as e:
                self.log_callback(f"Failed to write debug files: {e}")

        elapsed = time.time() - start_time
        self.log_callback(f"Saved SRT: {output_file} | Time elapsed: {elapsed:.2f}s")

        try:
            if debug_files:
                self.retain_debug_audio(audio_file, output_file)
            else:
                os.remove(audio_file)
        except Exception as e:
            self.log_callback(f"Failed to finalize temp audio file: {e}")

        return output_file

def extract_audio(video_path: str, audio_path: str, denoise_preset: str, log_callback, stop_event):
    if stop_event and stop_event.is_set():
        return False

    log_callback(f"Converting video to audio: {os.path.basename(video_path)}")
    ffmpeg_opts = '-hide_banner -loglevel warning -y -vn -ar 16000 -ac 1 -b:a 128k -f wav'
    if denoise_preset and denoise_preset in DENOISE_FILTERS and DENOISE_FILTERS[denoise_preset]:
        filter_str = DENOISE_FILTERS[denoise_preset]
        log_callback(f"Audio DSP preset: {denoise_preset}")
        ffmpeg_opts += f' -af {filter_str}'
    FFmpeg = _ensure_ffmpy3()
    ff = FFmpeg(executable="ffmpeg", inputs={video_path: None}, outputs={audio_path: ffmpeg_opts})
    try:
        run_process(ff.cmd, log_callback, stop_event=stop_event)
        return True
    except Exception as e:
        log_callback(f"Error extracting audio: {e}")
        return False


def analysis_wav_path(media_path: str | os.PathLike[str]) -> Path:
    """Return the shared WAV cache used by debug and standalone analysis."""
    media = Path(media_path)
    return media.parent / f"{media.stem}_debug" / f"{media.stem}.wav"


def generate_analysis_wav(media_path: str | os.PathLike[str], log_callback,
                          stop_event=None) -> str | None:
    """Extract a 16 kHz mono PCM WAV without loading any ASR model."""
    destination = analysis_wav_path(media_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.stem}.part.wav")
    try:
        if temporary.exists():
            temporary.unlink()
        if not extract_audio(str(media_path), str(temporary), "none", log_callback, stop_event):
            return None
        if stop_event and stop_event.is_set():
            return None
        os.replace(temporary, destination)
        log_callback(f"Analysis WAV saved: {destination}")
        return str(destination)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass
def warn_noisy_high_sensitivity(denoise_preset: str, vad_sensitivity: str, log_callback):
    """Warn when high/max VAD sensitivity runs without denoise.

    The low gate lets background noise through to the decoder, and Whisper
    hallucinates text on noise; the denoise preset raises the SNR of the ASR
    feed (never the dubbing bed) and suppresses most of it.
    """
    if (denoise_preset or "none") != "none" or vad_sensitivity not in ("high", "max"):
        return
    try:
        if app_config.get_language() == 'zh':
            log_callback("[!] 提示：未开启降噪且人声检测敏感度为高/极高时，噪声容易进入识别并产生幻觉字幕，建议选择轻度降噪。")
            return
        if app_config.get_language() == 'ja':
            log_callback("[!] ヒント：ノイズ除去なし＋音声検出感度が高/最高の場合、ノイズによる幻覚字幕が出やすくなります。軽度ノイズ除去を推奨します。")
            return
    except Exception:
        pass
    log_callback(
        "[warn] denoise=none with high/max voice-detection sensitivity lets noise reach "
        "the decoder and often produces hallucinated lines; consider the 'mild' denoise preset."
    )

# Module-level generator cache — we intentionally never destroy the WhisperModel
# during the app's lifetime. CTranslate2's thread pool destructor is not safe to call
# from inside a running Tkinter app (causes ucrtbase.dll crash 0xc0000409).
# The model is only freed when the Python interpreter exits, which CTranslate2 handles safely.
_generator_cache: dict = {}  # key: (model_key, use_gpu) -> SubtitleGenerator


def _get_or_create_subtitle_generator(model_key: str, models_root: str, use_gpu: bool, log_callback):
    model_path = get_model_dir(model_key, models_root)
    if not check_model_files(model_key, models_root):
        log_callback(f"Error: Required model files missing in {model_path}. Please download them first.")
        return None

    cache_key = (model_key, use_gpu)
    if cache_key in _generator_cache:
        log_callback("Using cached ASR model...")
        generator = _generator_cache[cache_key]
        generator.log_callback = log_callback
        return generator

    log_callback("Initializing ASR model...")
    try:
        generator = SubtitleGenerator(model_path, model_key, log_callback, use_gpu)
        _generator_cache[cache_key] = generator
        return generator
    except Exception as e:
        log_callback(f"Failed to initialize model: {e}")
        return None


def batch_generate_srt(base_dir: str, search_subdirs: bool, skip_if_exists: bool,
                       denoise_preset: str, model_key: str, models_root: str,
                       use_gpu: bool, log_callback, stop_event=None, gen_holder=None,
                       debug_files: bool = False, vad_sensitivity: str = "high"):
                       
    if not check_ffmpeg():
        log_callback("[!] Error: ffmpeg or ffprobe not found. Please refer to '说明.txt' or 'readme.txt' for installation steps.")
        return False

    if not os.path.exists(base_dir):
        log_callback(f"Error: Directory not found: {base_dir}")
        return False
        
    # Collect files
    tasks = []
    
    if search_subdirs:
        for root, files in _walk_user_media_directories(base_dir):
            for file in files:
                filepath = root / file
                if is_supported_source_media_file(filepath):
                    tasks.append(filepath)
    else:
        try:
            if not is_generated_work_path(base_dir):
                for file in os.listdir(base_dir):
                    filepath = Path(base_dir) / file
                    if filepath.is_file() and is_supported_source_media_file(filepath):
                        tasks.append(filepath)
        except Exception as e:
            log_callback(f"Error reading directory: {e}")
            return False

    if not tasks:
        log_callback("No valid video or audio files found.")
        return True

    # Initialize model (use cached instance to avoid destroying CTranslate2 thread pool)
    generator = _get_or_create_subtitle_generator(model_key, models_root, use_gpu, log_callback)
    if generator is None:
        return False
    generator.set_vad_sensitivity(vad_sensitivity)
    warn_noisy_high_sensitivity(denoise_preset, vad_sensitivity, log_callback)

    total_tasks = len(tasks)
    for task_idx, filepath in enumerate(tasks, start=1):
        if stop_event and stop_event.is_set():
            log_callback("Process stopped by user.")
            break
            
        if filepath.stem.endswith(".asr"):
            continue

        audio_file = filepath.with_name(f"{filepath.stem}.asr.wav")
        # target SRT format defined by batch script: .jp.srt
        src_srt_file = filepath.with_name(f"{filepath.stem}.jp.srt")
        std_srt_file = filepath.with_name(f"{filepath.stem}.srt")

        if skip_if_exists and (src_srt_file.exists() or std_srt_file.exists()):
            log_callback(f"[{task_idx}/{total_tasks}] Skipping {filepath.name}: SRT already exists")
            continue

        log_callback(f"[{task_idx}/{total_tasks}] Processing: {filepath.name}")
        
        # Ensure we have audio
        if not extract_audio(str(filepath), str(audio_file), denoise_preset, log_callback, stop_event):
            continue
            
        if stop_event and stop_event.is_set():
            break

        log_callback("Transcription started...")
        try:
            generator.transcribe(str(audio_file), str(src_srt_file), debug_files=debug_files)
            log_callback(f"[{task_idx}/{total_tasks}] Transcription complete.")
        except Exception as e:
            log_callback(f"Transcription failed: {e}")

    log_callback("Batch SRT Generation Completed.")
    # The generator is intentionally kept alive in _generator_cache — do NOT delete it.
    return True

# ============================================================
# Subtitle Translation Logic
# ============================================================

DEFAULT_TRANS_CONFIG = {
    "api_base_url": "https://api.deepseek.com/",
    "model_name": "deepseek-v4-flash",
    "tokens_per_chunk": 500000,
    "temperature": 0.5,
    "max_retries": 3,
    "target_language": "Chinese",
    "keep_original": True,
    "adult_content": True,
    "dubbing_optimized": False,
    "source_correction": True
}

DEFAULT_PROMPT = """\
Translate the following subtitles to {target_language}.
The subtitles are numbered in playback order and form one continuous dialogue:
always use the surrounding lines as context, and keep person names, forms of
address, and tone consistent across the whole file. The source text is ASR
output and may contain recognition errors — infer the intended meaning from
context and translate that meaning.
Keep the XML tags <id>...</id> intact and one-to-one. Only translate the text content.

{subtitles}
"""

DEFAULT_CORRECT_PROMPT = """\
The following subtitles are raw speech-recognition (ASR) output in {source_language},
numbered in playback order. Correct recognition errors (homophones, wrong particles,
wrong or inconsistent person names, garbled words) using the surrounding lines as
context. Do NOT translate — output must stay in the same language as the input.
If a line is already correct or unrecoverable, return it unchanged.
If a line consists ONLY of non-lexical vocalizations or fillers (moans, sighs,
laughter, hums — e.g. あ, ああー, ん, うん, 嗯, はぁ, ふふ, あはは), output its
tag EMPTY like <id></id> to mark it for removal; keep meaningful short lines
(はい, え?, だめ). Also output an EMPTY tag for hallucinated stock phrases that
appear abruptly with no contextual support (ご視聴ありがとうございました,
おやすみなさい, またお会いしましょう and similar closing/greeting lines).
Keep the XML tags <id>...</id> intact and one-to-one; never leave an id out.

{subtitles}
"""

def get_config_dir():
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "config")

def load_trans_config():
    config_dir = get_config_dir()
    os.makedirs(config_dir, exist_ok=True)
    config_file = os.path.join(config_dir, "subtitle_trans_config.json")
    prompt_file = os.path.join(config_dir, "translate_prompt.txt")

    config = dict(DEFAULT_TRANS_CONFIG)
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8-sig") as f:
                config.update(json.load(f))
        except Exception as e:
            print(f"Error loading trans config: {e}")

    if not os.path.exists(prompt_file):
        try:
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(DEFAULT_PROMPT)
        except:
            pass

    correct_prompt_file = os.path.join(config_dir, "asr_correct_prompt.txt")
    if not os.path.exists(correct_prompt_file):
        try:
            with open(correct_prompt_file, "w", encoding="utf-8") as f:
                f.write(DEFAULT_CORRECT_PROMPT)
        except:
            pass

    return config

def save_trans_config(new_config):
    config_dir = get_config_dir()
    os.makedirs(config_dir, exist_ok=True)
    config_file = os.path.join(config_dir, "subtitle_trans_config.json")
    try:
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(new_config, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving trans config: {e}")
        return False

class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.5):
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        self.model = model
        self.temperature = temperature
        self.input_tokens = 0
        self.output_tokens = 0

    def complete(self, prompt: str) -> str:
        import requests

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": prompt}],
            "temperature": self.temperature,
        }
        resp = requests.post(self.url, headers=self.headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        self.input_tokens += usage.get("prompt_tokens", 0)
        self.output_tokens += usage.get("completion_tokens", 0)
        return data["choices"][0]["message"]["content"]

def parse_srt(text: str) -> dict:
    entries = {}
    lines = text.strip().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        try:
            sid = int(line)
        except ValueError:
            i += 1
            continue
        if i + 1 >= len(lines):
            break
        timestamp = lines[i + 1]
        i += 2
        body = []
        while i < len(lines) and lines[i].strip():
            body.append(lines[i])
            i += 1
        entries[sid] = {"timestamp": timestamp, "text": "\n".join(body)}
        i += 1
    if not entries:
        raise ValueError("Failed to parse SRT data — no valid entries found.")
    return entries

def render_srt(entries: dict) -> str:
    parts = []
    for sid in sorted(entries):
        e = entries[sid]
        parts.append(f"{sid}\n{e['timestamp']}\n{e['text']}\n")
    return "\n".join(parts)

def split_into_chunks(entries: dict, limit: int) -> list[dict]:
    chunks, current, size = [], {}, 0
    for sid, info in entries.items():
        length = len(info["text"])
        if current and size + length > limit:
            chunks.append(current)
            current, size = {}, 0
        current[sid] = info["text"]
        size += length
    if current:
        chunks.append(current)
    return chunks

def sequential_ids(chunk: dict) -> tuple[str, dict]:
    """Tag entries with sequential ids (1..n, playback order).

    Sequential tags keep the "these lines are consecutive dialogue" signal for
    the LLM; the previous randomized ids deliberately hid the ordering, which
    encouraged line-by-line isolated translation. The tag -> original-id mapping
    still shields the model from the real SRT numbering.
    """
    mapping = {}
    lines = []
    for seq, (orig_id, text) in enumerate(chunk.items(), start=1):
        mapping[seq] = orig_id
        lines.append(f"<{seq}>{text}</{seq}>")
    return "\n".join(lines), mapping

_TAG_PAIR = re.compile(r"<(\d+)>(.*?)</\d+>", re.DOTALL | re.MULTILINE)
_TAG_OPEN_ONLY = re.compile(r"<(\d+)>(.*?)\n", re.DOTALL)

def deobfuscate_ids(response: str, mapping: dict) -> dict:
    results = {}
    matches = _TAG_PAIR.findall(response)
    if not matches:
        matches = _TAG_OPEN_ONLY.findall(response)
    for tag_id_str, text in matches:
        tag_id = int(tag_id_str)
        if tag_id in mapping:
            results[mapping[tag_id]] = text.strip()
    return results

def _strip_adult_background(template: str) -> str:
    # Remove Content Background section up to "--------" separator
    return re.sub(r'Content Background:.*?(?=-{5,})', '', template, flags=re.DOTALL)

def _load_prompt_template(adult_content: bool = True, dubbing_optimized: bool = False) -> str:
    prompt_name = "translate_prompt_dubbing.txt" if dubbing_optimized else "translate_prompt.txt"
    prompt_file = os.path.join(get_config_dir(), prompt_name)
    template = DEFAULT_PROMPT
    if os.path.isfile(prompt_file):
        with open(prompt_file, encoding="utf-8-sig") as f:
            template = f.read()

    if not adult_content:
        template = _strip_adult_background(template)

    return template

def _load_correct_prompt_template(adult_content: bool = True) -> str:
    prompt_file = os.path.join(get_config_dir(), "asr_correct_prompt.txt")
    template = DEFAULT_CORRECT_PROMPT
    if os.path.isfile(prompt_file):
        with open(prompt_file, encoding="utf-8-sig") as f:
            template = f.read()

    if not adult_content:
        template = _strip_adult_background(template)

    return template

def _build_prompt(tagged_text: str, template: str, placeholders: dict) -> str:
    prompt = template.replace("{subtitles}", tagged_text.strip())
    for key, value in placeholders.items():
        prompt = prompt.replace("{" + key + "}", value)
    return prompt

def _with_context(missing_ids: list, pool: dict, before: int = 2, after: int = 1) -> dict:
    """Missing entries plus their neighbours from ``pool``, in playback order.

    Retried lines used to be re-sent alone, i.e. translated with zero context;
    surrounding lines give the model something to anchor on.
    """
    order = list(pool)
    index = {sid: i for i, sid in enumerate(order)}
    keep = set()
    for sid in missing_ids:
        i = index.get(sid)
        if i is None:
            continue
        keep.update(order[max(0, i - before): i + after + 1])
    return {sid: pool[sid] for sid in order if sid in keep}

def _llm_chunk_pass(client: LLMClient, chunk: dict, template: str, placeholders: dict,
                    max_retries: int, log_callback, stop_event) -> dict:
    tagged, mapping = sequential_ids(chunk)
    results = {}

    for attempt in range(1, max_retries + 1):
        if stop_event and stop_event.is_set():
            break
        try:
            prompt = _build_prompt(tagged, template, placeholders)
            response = client.complete(prompt)
            # Accumulate across attempts: a retry response only contains the
            # re-sent lines and must not wipe earlier successful ones.
            results.update(deobfuscate_ids(response, mapping))
        except Exception as exc:
            log_callback(f"  [WARN] Chunk LLM error (attempt {attempt}): {exc}")
            if attempt == max_retries:
                break
            continue

        missing_ids = [oid for oid in chunk if oid not in results]
        if not missing_ids:
            break
        if attempt < max_retries:
            log_callback(f"  [INFO] {len(missing_ids)} entries missing, retrying with context...")
            retry_chunk = _with_context(missing_ids, chunk)
            tagged, mapping = sequential_ids(retry_chunk)
        else:
            log_callback(f"  [WARN] Still missing {len(missing_ids)} entries after {max_retries} attempts.")

    return results

def _run_entries_llm(client: LLMClient, entries: dict, template: str, placeholders: dict,
                     tokens_per_chunk: int, max_retries: int, log_callback, stop_event,
                     label: str = "Translating") -> dict:
    """Run one LLM pass over all entries, returning {sid: response_text}."""
    chunks = split_into_chunks(entries, tokens_per_chunk)
    log_callback(f"[INFO] Total chunks: {len(chunks)}")

    all_results = {}

    for idx, chunk in enumerate(chunks):
        if stop_event and stop_event.is_set():
            log_callback(f"{label} stopped by user.")
            break

        log_callback(f"[INFO] {label} chunk {idx + 1}/{len(chunks)} ({len(chunk)} entries)...")
        result = _llm_chunk_pass(client, chunk, template, placeholders, max_retries, log_callback, stop_event)

        if not result and len(chunk) > 1:
            mid = len(chunk) // 2
            items = list(chunk.items())
            for half_label, half in [("first", dict(items[:mid])), ("second", dict(items[mid:]))]:
                if stop_event and stop_event.is_set():
                    break
                log_callback(f"  [INFO] Retrying {half_label} half ({len(half)} entries)...")
                half_result = _llm_chunk_pass(client, half, template, placeholders, max_retries, log_callback, stop_event)
                all_results.update(half_result)
        else:
            all_results.update(result)

    return all_results

_KANA_RE = re.compile(r"[ぁ-ゟ゠-ヿ]")

def _target_expects_kana(lang: str) -> bool:
    return (lang or "").strip().lower() in ("japanese", "ja", "jp", "日本語", "日语", "日文")

def translate_entries(client: LLMClient, entries: dict, lang: str,
                      tokens_per_chunk: int, keep_original: bool, adult_content: bool,
                      dubbing_optimized: bool, max_retries: int, log_callback, stop_event) -> dict:
    template = _load_prompt_template(adult_content, dubbing_optimized)
    all_translated = _run_entries_llm(
        client, entries, template, {"target_language": lang},
        tokens_per_chunk, max_retries, log_callback, stop_event,
        label="Translating",
    )

    # Kana-leak backstop: on garbled ASR fragments the model tends to echo the
    # Japanese source (or half-translate it) instead of answering empty. Retry
    # those lines once with context; drop the translation if kana remains, so a
    # bilingual SRT shows only the original and the dub skips the line.
    if not _target_expects_kana(lang) and not (stop_event and stop_event.is_set()):
        leaked = sorted(sid for sid, text in all_translated.items() if _KANA_RE.search(text))
        if leaked:
            log_callback(f"[WARN] {len(leaked)} translations still contain source kana; retrying with context...")
            pool = {sid: info["text"] for sid, info in entries.items()}
            retry_chunk = _with_context(leaked, pool)
            retry_result = _llm_chunk_pass(
                client, retry_chunk, template, {"target_language": lang},
                1, log_callback, stop_event,
            )
            dropped = 0
            for sid in leaked:
                new_text = (retry_result.get(sid) or "").strip()
                if new_text and not _KANA_RE.search(new_text):
                    all_translated[sid] = new_text
                else:
                    all_translated.pop(sid, None)
                    dropped += 1
            if dropped:
                log_callback(f"[WARN] Dropped {dropped} untranslatable (garbled) lines after retry.")

    for sid, trans_text in all_translated.items():
        if sid in entries:
            orig = entries[sid]["text"]
            entries[sid]["text"] = f"{trans_text}\n{orig}" if keep_original else trans_text
            # Consumers that need "was this line actually translated?" (the
            # dub must not speak source text left behind by a dropped/missing
            # translation) check this flag instead of the text itself.
            entries[sid]["translated"] = True

    return entries

# Deletion safety gate: the model may answer an empty tag to mean "drop this
# line", but we only honour that when the source line itself looks droppable —
# a pure interjection (short, no real words) or a known ASR stock-phrase
# hallucination. A lazy empty answer can never wipe a real sentence.
_INTERJECTION_MAX_NORM_CHARS = 10
_INTERJECTION_HANZI = "啊哈嗯哦呃唔呀噢喔嘿呜哎唉咦哇"
_CJK_IDEOGRAPH_RE = re.compile(r"[一-鿿]")

_STOCK_HALLUCINATION_EXTRA = {
    "おやすみなさい",
    "またお会いしましょう",
    "また見てね",
    "さようなら",
    "お疲れ様でした",
    "ご視聴してくださって本当にありがとうございます",
    "Thank you so much for watching until the end",
    "Thank you for watching",
}
_STOCK_PHRASE_NORMS = (
    HARD_HALLUCINATION_NORMS
    | SHORT_HALLUCINATION_NORMS
    | {re.sub(r"[、。！？!?….\s]+", "", p).lower() for p in _STOCK_HALLUCINATION_EXTRA}
)

def _deletable_interjection(text: str) -> bool:
    norm = SubtitleGenerator.normalize_for_duplicate(text)
    if not norm or len(norm) > _INTERJECTION_MAX_NORM_CHARS:
        return False
    remainder = re.sub(f"[{_INTERJECTION_HANZI}]", "", text)
    return not _CJK_IDEOGRAPH_RE.search(remainder)

def _deletable_line(text: str) -> bool:
    if _deletable_interjection(text):
        return True
    norm = SubtitleGenerator.normalize_for_duplicate(text)
    for phrase in _STOCK_PHRASE_NORMS:
        if phrase and (norm == phrase or (phrase in norm and len(norm) <= len(phrase) + 6)):
            return True
    return False

def correct_entries(client: LLMClient, entries: dict, source_language: str,
                    tokens_per_chunk: int, adult_content: bool, max_retries: int,
                    log_callback, stop_event) -> tuple[int, set]:
    """LLM proofread pass over ASR source text before translation.

    Fixes recognition errors (homophones, wrong names/particles) using the
    surrounding lines as context, in the source language. Mutates each entry's
    ``text`` in place. An empty model answer marks a line for deletion (pure
    interjections/moans, or out-of-place ASR stock-phrase hallucinations); it
    is honoured only when the source line itself looks droppable (see
    :func:`_deletable_line`), and the entry is then removed from ``entries``.

    Returns ``(changed_count, deleted_ids)``.
    """
    template = _load_correct_prompt_template(adult_content)
    total = len(entries)
    results = _run_entries_llm(
        client, entries, template,
        {"source_language": source_language or "Japanese"},
        tokens_per_chunk, max_retries, log_callback, stop_event,
        label="Proofreading source",
    )

    changed = 0
    deleted: set = set()
    for sid, text in results.items():
        if sid not in entries:
            continue
        old_text = entries[sid]["text"].strip()
        new_text = text.strip()
        if not new_text:
            if _deletable_line(old_text):
                del entries[sid]
                deleted.add(sid)
            continue
        if new_text != old_text:
            entries[sid]["text"] = new_text
            changed += 1
    log_callback(
        f"[INFO] Source proofread changed {changed}/{total} lines, "
        f"removed {len(deleted)} interjection-only lines"
    )
    return changed, deleted


def _translate_srt_path(src_path: Path, out_path: Path, client: LLMClient, config: dict, log_callback, stop_event=None) -> bool:
    target_lang = config["target_language"]
    tokens_per_chunk = int(config["tokens_per_chunk"])
    keep_original = config.get("keep_original", True)
    adult_content = config.get("adult_content", True)
    dubbing_optimized = config.get("dubbing_optimized", False)
    source_correction = config.get("source_correction", False)
    max_retries = int(config.get("max_retries", 3))

    log_callback(f"[INFO] Translating: {src_path.name}")
    try:
        with open(src_path, "r", encoding="utf-8-sig") as f:
            srt_data = f.read()

        entries = parse_srt(srt_data)

        if source_correction:
            log_callback("[INFO] AI source proofread started...")
            correct_entries(
                client,
                entries,
                "Japanese",
                tokens_per_chunk,
                adult_content,
                max_retries,
                log_callback,
                stop_event,
            )
            if stop_event and stop_event.is_set():
                return False

        entries = translate_entries(
            client,
            entries,
            target_lang,
            tokens_per_chunk,
            keep_original,
            adult_content,
            dubbing_optimized,
            max_retries,
            log_callback,
            stop_event,
        )

        if stop_event and stop_event.is_set():
            return False

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(render_srt(entries))
        log_callback(f"[INFO] Written: {out_path.name}")
        return True
    except Exception as e:
        log_callback(f"[ERROR] Failed to translate {src_path.name}: {e}")
        return False


def batch_translate_srt(base_dir: str, search_subdirs: bool, skip_if_exists: bool, api_key: str, config: dict, log_callback, stop_event=None):
    if not os.path.exists(base_dir):
        log_callback(f"Error: Directory not found: {base_dir}")
        return False
        
    client = LLMClient(config["api_base_url"], api_key, config["model_name"], config.get("temperature", 0.5))

    # Collect .jp.srt files
    tasks = []
    if search_subdirs:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith(".jp.srt"):
                    tasks.append(Path(root) / file)
    else:
        try:
            for file in os.listdir(base_dir):
                if file.endswith(".jp.srt"):
                    filepath = Path(base_dir) / file
                    if filepath.is_file():
                        tasks.append(filepath)
        except Exception as e:
            log_callback(f"Error reading directory: {e}")
            return False

    if not tasks:
        log_callback("No .jp.srt files found.")
        return True

    for src_path in tasks:
        if stop_event and stop_event.is_set():
            log_callback("Process stopped by user.")
            break
            
        out_path = src_path.with_name(src_path.name[:-7] + ".srt")
        if skip_if_exists and out_path.exists():
            log_callback(f"[INFO] Skipping {src_path.name}: Output SRT already exists. (取消勾选“忽略已存在翻译”复选框可重新生成)")
            continue
            
        _translate_srt_path(src_path, out_path, client, config, log_callback, stop_event)

    log_callback(f"[INFO] Batch Translation Completed. API usage — input: {client.input_tokens} tokens, output: {client.output_tokens} tokens")
    return True


def _collect_listen_translate_videos(base_dir: str, search_subdirs: bool, log_callback) -> list[Path] | None:
    tasks = []
    if search_subdirs:
        for root, files in _walk_user_media_directories(base_dir):
            for file in files:
                filepath = root / file
                if filepath.is_file() and is_subtitle_video_candidate(filepath):
                    tasks.append(filepath)
    else:
        try:
            if not is_generated_work_path(base_dir):
                for file in os.listdir(base_dir):
                    filepath = Path(base_dir) / file
                    if filepath.is_file() and is_subtitle_video_candidate(filepath):
                        tasks.append(filepath)
        except Exception as e:
            log_callback(f"Error reading directory: {e}")
            return None
    return tasks


def batch_listen_translate_srt(base_dir: str, search_subdirs: bool, skip_if_translated: bool,
                               keep_jp_srt: bool, denoise_preset: str, model_key: str,
                               models_root: str, use_gpu: bool, api_key: str, config: dict,
                               log_callback, stop_event=None, gen_holder=None,
                               vad_sensitivity: str = "high"):
    if not os.path.exists(base_dir):
        log_callback(f"Error: Directory not found: {base_dir}")
        return False

    tasks = _collect_listen_translate_videos(base_dir, search_subdirs, log_callback)
    if tasks is None:
        return False

    if not tasks:
        log_callback("No valid video files found.")
        return True

    pending = []
    for filepath in tasks:
        jp_srt_file = filepath.with_name(f"{filepath.stem}.jp.srt")
        out_srt_file = filepath.with_name(f"{filepath.stem}.srt")
        if skip_if_translated and out_srt_file.exists():
            log_callback(f"[INFO] Skipping {filepath.name}: Output SRT already exists.")
            continue
        pending.append((filepath, jp_srt_file, out_srt_file))

    if not pending:
        log_callback("[INFO] No videos need one-click listening translation.")
        return True

    generator = None
    needs_transcription = any(not jp_srt_file.exists() for _, jp_srt_file, _ in pending)
    if needs_transcription:
        if not check_ffmpeg():
            log_callback("[!] Error: ffmpeg or ffprobe not found. Please refer to '说明.txt' or 'readme.txt' for installation steps.")
            return False
        generator = _get_or_create_subtitle_generator(model_key, models_root, use_gpu, log_callback)
        if generator is None:
            return False
        generator.set_vad_sensitivity(vad_sensitivity)
        warn_noisy_high_sensitivity(denoise_preset, vad_sensitivity, log_callback)

    client = LLMClient(config["api_base_url"], api_key, config["model_name"], config.get("temperature", 0.5))
    total_tasks = len(pending)

    for task_idx, (filepath, jp_srt_file, out_srt_file) in enumerate(pending, start=1):
        if stop_event and stop_event.is_set():
            log_callback("Process stopped by user.")
            break

        log_callback(f"[{task_idx}/{total_tasks}] One-click listening translation: {filepath.name}")

        if not jp_srt_file.exists():
            audio_file = filepath.with_name(f"{filepath.stem}.asr.wav")
            if not extract_audio(str(filepath), str(audio_file), denoise_preset, log_callback, stop_event):
                continue

            if stop_event and stop_event.is_set():
                break

            log_callback("Transcription started...")
            try:
                generator.transcribe(str(audio_file), str(jp_srt_file))
                log_callback(f"[{task_idx}/{total_tasks}] Transcription complete.")
            except Exception as e:
                log_callback(f"Transcription failed: {e}")
                continue
        else:
            log_callback(f"[INFO] Using existing Japanese subtitle: {jp_srt_file.name}")

        if stop_event and stop_event.is_set():
            break

        if not jp_srt_file.exists():
            log_callback(f"[WARN] Japanese subtitle missing after transcription: {jp_srt_file.name}")
            continue

        if skip_if_translated and out_srt_file.exists():
            log_callback(f"[INFO] Skipping translation for {jp_srt_file.name}: Output SRT already exists.")
            continue

        translated = _translate_srt_path(jp_srt_file, out_srt_file, client, config, log_callback, stop_event)
        if translated and not keep_jp_srt:
            try:
                jp_srt_file.unlink()
                log_callback(f"[INFO] Removed Japanese subtitle: {jp_srt_file.name}")
            except Exception as e:
                log_callback(f"[WARN] Could not remove Japanese subtitle {jp_srt_file.name}: {e}")

    log_callback(f"[INFO] One-click Listening Translation Completed. API usage — input: {client.input_tokens} tokens, output: {client.output_tokens} tokens")
    return True

# ===============================
# SRT to ASS Logic
# ===============================
def _get_video_resolution(video_path: str) -> tuple[int, int]:
    """Get video resolution using ffprobe, returns (width, height). Defaults to 1920x1080 if failed."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "v:0",
        video_path,
    ]
    try:
        if sys.platform.startswith('win'):
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, startupinfo=startupinfo)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
        info = json.loads(result.stdout)
        stream = info["streams"][0]
        width = int(stream["width"])
        # In VR 180/360 LeftRight format, width is usually divided by 2
        width = width // 2
        return width, int(stream["height"])
    except Exception as e:
        return 1920, 1080

def _srt_time_to_ass(srt_time: str) -> str:
    """Convert SRT timestamp to ASS format."""
    srt_time = srt_time.strip().replace(",", ".")
    m = re.match(r"(\d+):(\d{2}):(\d{2})\.(\d+)", srt_time)
    if not m:
        raise ValueError(f"Could not parse timestamp: {srt_time!r}")
    h, mi, s, ms_str = m.groups()
    cs = ms_str[:2].ljust(2, "0")
    return f"{int(h)}:{mi}:{s}.{cs}"

def _is_japanese(text: str) -> bool:
    """Check if the text contains Japanese characters (Hiragana/Katakana)."""
    for ch in text:
        if "\u3040" <= ch <= "\u30ff":
            return True
    return False

def _parse_srt_blocks(srt_path: str) -> list[dict]:
    """Parse SRT file and return a list of blocks with detailed timestamps and lines."""
    with open(srt_path, 'r', encoding="utf-8-sig", errors="replace") as f:
        text = f.read()
        
    raw_blocks = re.split(r"\n\s*\n", text.strip())
    blocks = []
    for raw in raw_blocks:
        lines = [l.rstrip() for l in raw.strip().splitlines()]
        if len(lines) < 2:
            continue
        
        time_line_idx = -1
        for i, l in enumerate(lines):
            if "-->" in l:
                time_line_idx = i
                break
        
        if time_line_idx == -1:
            continue

        time_line = lines[time_line_idx]
        m = re.match(r"(.+?)\s*-->\s*(.+)", time_line)
        if not m:
            continue
            
        start_raw, end_raw = m.group(1), m.group(2)
        try:
            start = _srt_time_to_ass(start_raw)
            end   = _srt_time_to_ass(end_raw)
        except ValueError:
            continue
            
        text_lines = [l for l in lines[time_line_idx + 1:] if l.strip()]
        if not text_lines:
            continue
        blocks.append({"start": start, "end": end, "lines": text_lines})
    return blocks

def batch_convert_srt_to_ass(base_dir: str, alignment: int, base_cn_size: int, base_jp_size: int,
                             search_subdirs: bool, skip_exists: bool, only_bilingual: bool,
                             log_callback, stop_event,
                             default_primary_colour: str = "&H005AFF65",
                             default_outline_colour: str = "&H00000000",
                             secondary_primary_colour: str = "&H00FFFFFF",
                             secondary_outline_colour: str = "&H00000000"):
    base_path = Path(base_dir)
    pattern = "**/*.srt" if search_subdirs else "*.srt"
    srt_files = list(base_path.glob(pattern))
    
    # Filter out .jp.srt and transcription debug files
    _excluded_suffixes = (".jp.srt", ".raw.srt", ".vad.srt", ".removed.srt")
    srt_files = [f for f in srt_files if not f.name.lower().endswith(_excluded_suffixes)]
    
    if not srt_files:
        log_callback("[INFO] No valid .srt files found.")
        return

    # Load Template
    template_path = os.path.join(get_config_dir(), "subtitle_ass_templates.txt")
    if not os.path.exists(template_path):
        log_callback(f"[ERROR] Cannot find ASS template: {template_path}")
        return
        
    with open(template_path, 'r', encoding='utf-8') as f:
        template_content = f.read()
        
    ass_header_tmpl = template_content.split("[Events]")[0] + "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    dialogue_tmpl = "Dialogue: 0,{start},{end},{style},,0,0,0,,{text}\n"

    for srt_path in srt_files:
        if stop_event and stop_event.is_set():
            break

        ass_path = srt_path.with_suffix(".ass")
        if skip_exists and ass_path.exists():
            log_callback(f"[INFO] Skip: {ass_path.name} already exists.")
            continue

        blocks = _parse_srt_blocks(str(srt_path))
        if not blocks:
            log_callback(f"[WARNING] No valid blocks in {srt_path.name}")
            continue

        if only_bilingual:
            bilingual_blocks = sum(1 for b in blocks if len(b["lines"]) >= 2)
            if bilingual_blocks < len(blocks) * 0.2:
                log_callback(f"[INFO] Skip: {srt_path.name} is not bilingual.")
                continue

        # Try to find corresponding video (MP4 or MKV)
        video_path = None
        for suffix in [".mp4", ".mkv"]:
            potential_path = srt_path.with_suffix(suffix)
            if potential_path.exists() and not is_si_sidecar_media_file(potential_path):
                video_path = potential_path
                break

        if video_path:
            width, height = _get_video_resolution(str(video_path))
        else:
            width, height = 1920, 1080
            
        # Scale logic
        scale = ((width * height) / (1280 * 720)) ** 0.5
        cn_size = round(base_cn_size * scale)
        jp_size = round(base_jp_size * scale)
        marginv = round(32 * scale)

        try:
            with open(ass_path, "w", encoding="utf-8") as f:
                f.write(ass_header_tmpl.format(
                    width=width, height=height,
                    cn_size=cn_size, jp_size=jp_size, marginv=marginv,
                    alignment=alignment,
                    DefaultPrimaryColour=default_primary_colour,
                    DefaultOutlineColour=default_outline_colour,
                    SecondaryPrimaryColour=secondary_primary_colour,
                    SecondaryOutlineColour=secondary_outline_colour
                ))
                for block in blocks:
                    start, end = block["start"], block["end"]
                    lines = block["lines"]

                    if len(lines) == 1:
                        line = lines[0]
                        style = "Secondary" if _is_japanese(line) else "Default"
                        f.write(dialogue_tmpl.format(start=start, end=end, style=style, text=line))
                    elif len(lines) >= 2:
                        # Top line Chinese, second line Japanese usually
                        f.write(dialogue_tmpl.format(start=start, end=end, style="Default", text=lines[0]))
                        f.write(dialogue_tmpl.format(start=start, end=end, style="Secondary", text=lines[1]))
            log_callback(f"[INFO] ✓ Generated ASS: {ass_path.name}")
        except Exception as e:
            log_callback(f"[ERROR] Failed to convert {srt_path.name}: {e}")
            
# ===============================
# Rank Subtitles Logic
# ===============================
import statistics

RANK_TIMESTAMP_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*"
    r"(?P<end>\d{2}:\d{2}:\d{2},\d{3})"
)
RANK_JP_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
RANK_TEXT_RE = re.compile(r"[^\W\d_]", re.UNICODE)

class SrtEntry:
    def __init__(self, start: float, end: float, text: str):
        self.start = start
        self.end = end
        self.text = text

def _rank_parse_timestamp(value: str) -> float:
    hour, minute, rest = value.split(":")
    second, ms = rest.split(",")
    return int(hour) * 3600 + int(minute) * 60 + int(second) + int(ms) / 1000.0

def _rank_parse_srt(path: Path) -> tuple[list[SrtEntry], int]:
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"\n\s*\n", raw.replace("\r\n", "\n").replace("\r", "\n").strip())
    entries = []
    invalid_blocks = 0

    for block in blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if not lines:
            continue

        time_index = next(
            (idx for idx, line in enumerate(lines) if RANK_TIMESTAMP_RE.search(line)),
            None,
        )
        if time_index is None:
            invalid_blocks += 1
            continue

        match = RANK_TIMESTAMP_RE.search(lines[time_index])
        try:
            start = _rank_parse_timestamp(match.group("start"))
            end = _rank_parse_timestamp(match.group("end"))
        except ValueError:
            invalid_blocks += 1
            continue
        text = "".join(lines[time_index + 1 :]).strip()
        entries.append(SrtEntry(start=start, end=end, text=text))

    return entries, invalid_blocks

def _rank_clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))

def _rank_repeated_line_ratio(texts: list[str]) -> float:
    cleaned = [text for text in texts if text]
    if not cleaned:
        return 1.0
    return 1.0 - len(set(cleaned)) / len(cleaned)

def _rank_repeated_ngram_ratio(text: str, size: int = 8) -> float:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < size * 2:
        return 0.0
    grams = [compact[i : i + size] for i in range(0, len(compact) - size + 1)]
    return 1.0 - len(set(grams)) / len(grams)

def _rank_score_density(chars_per_minute: float) -> float:
    if chars_per_minute <= 0:
        return 0.0
    if chars_per_minute < 25:
        return chars_per_minute / 25.0
    if chars_per_minute <= 220:
        return 1.0
    if chars_per_minute >= 420:
        return 0.15
    return 1.0 - (chars_per_minute - 220) / 200.0 * 0.85

def _rank_score_coverage(coverage_ratio: float) -> float:
    if coverage_ratio <= 0:
        return 0.0
    if coverage_ratio < 0.22:
        return coverage_ratio / 0.22
    if coverage_ratio <= 0.55:
        return 1.0
    if coverage_ratio >= 0.90:
        return 0.25
    return 1.0 - (coverage_ratio - 0.55) / 0.35 * 0.75

def _rank_score_duration(duration: float) -> float:
    if duration <= 0:
        return 0.0
    if duration < 0.4:
        return duration / 0.4
    if duration <= 9.0:
        return 1.0
    if duration >= 20.0:
        return 0.2
    return 1.0 - (duration - 9.0) / 11.0 * 0.8

def _rank_analyze(path: Path) -> dict:
    entries, invalid_blocks = _rank_parse_srt(path)
    texts = [entry.text for entry in entries]
    text = "".join(texts)
    text_chars = len(re.sub(r"\s+", "", text))
    jp_chars = len(RANK_JP_RE.findall(text))
    letter_chars = len(RANK_TEXT_RE.findall(text))

    if entries:
        starts = [entry.start for entry in entries]
        ends = [entry.end for entry in entries]
        span = max(ends) - min(starts)
        subtitle_duration = sum(max(0.0, entry.end - entry.start) for entry in entries)
        durations = [max(0.0, entry.end - entry.start) for entry in entries]
    else:
        starts = []
        ends = []
        span = 0.0
        subtitle_duration = 0.0
        durations = []

    invalid_durations = sum(1 for entry in entries if entry.end <= entry.start)
    overlaps = sum(
        1
        for prev, cur in zip(entries, entries[1:])
        if cur.start < prev.end - 0.05
    )
    large_gaps = sum(
        1
        for prev, cur in zip(entries, entries[1:])
        if cur.start - prev.end > 45.0
    )
    empty_text = sum(1 for value in texts if not value.strip())

    coverage_ratio = subtitle_duration / span if span > 0 else 0.0
    chars_per_minute = text_chars / (span / 60.0) if span > 0 else 0.0
    avg_duration = statistics.mean(durations) if durations else 0.0
    median_duration = statistics.median(durations) if durations else 0.0
    jp_ratio = jp_chars / max(1, letter_chars)
    duplicate_ratio = _rank_repeated_line_ratio(texts)
    repeat_ratio = _rank_repeated_ngram_ratio(text)

    count_score = _rank_clamp(len(entries) / 30.0)
    coverage_score = _rank_score_coverage(coverage_ratio)
    density_score = _rank_score_density(chars_per_minute)
    duration_score = _rank_score_duration(median_duration or avg_duration)
    language_score = _rank_clamp(jp_ratio / 0.85)
    structure_penalty = _rank_clamp(
        (
            invalid_blocks
            + invalid_durations * 2
            + overlaps
            + empty_text
            + large_gaps * 0.4
        )
        / max(1, len(entries))
    )
    repetition_penalty = _rank_clamp(max(duplicate_ratio, repeat_ratio * 1.8))

    score = 100.0 * (
        0.22 * count_score
        + 0.22 * coverage_score
        + 0.22 * density_score
        + 0.14 * duration_score
        + 0.12 * language_score
        + 0.08 * (1.0 - structure_penalty)
    )
    score *= 1.0 - 0.35 * repetition_penalty
    score = round(_rank_clamp(score, 0.0, 100.0), 2)

    return {
        "file": str(path),
        "score": score,
        "entries": len(entries),
        "span_min": round(span / 60.0, 2),
        "coverage": round(coverage_ratio, 3),
        "chars_per_min": round(chars_per_minute, 1),
        "jp_ratio": round(jp_ratio, 3),
        "duplicates": round(duplicate_ratio, 3),
        "large_gaps": large_gaps,
    }

def batch_rank_srt(base_dir: str, log_callback, stop_event, callback):
    """Scan root of base_dir for SRT files and analyze them, returning the sorted list via callback."""
    base_path = Path(base_dir)
    # Only direct children, no subdirectories
    srt_files = list(base_path.glob("*.srt"))
    
    if not srt_files:
        log_callback(f"[INFO] No .srt files found in {base_dir}")
        return

    results = []
    log_callback(f"[INFO] Found {len(srt_files)} SRT files. Starting assessment...")
    for srt_path in srt_files:
        if stop_event and stop_event.is_set():
            break
            
        try:
            res = _rank_analyze(srt_path)
            results.append(res)
        except Exception as e:
            log_callback(f"[ERROR] Failed to analyze {srt_path.name}: {e}")

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Callback to render UI
    if not (stop_event and stop_event.is_set()):
        callback(results)
