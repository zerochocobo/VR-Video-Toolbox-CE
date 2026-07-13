"""
utils/app_config.py
Global application configuration read/write module.
The config file is vr_toolbox_config.json. In release mode it is stored beside
the executable; in development mode it is stored at the script root. It is
created automatically when missing.
"""
import json
import locale
import os
import sys

# --- Config file path resolution ---
# After PyInstaller packaging, sys.executable points to the .exe.
# In development mode, use the directory two levels above this file: the project root.
_frozen = getattr(sys, 'frozen', False)
_config_dir = os.path.dirname(sys.executable) if _frozen else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_config_dir, 'vr_toolbox_config.json')

# --- Defaults ---
_DEFAULTS = {
    'engine': 'native_gpu',  # Default to the built-in GPU engine: in-process Lada + direct GPU NVENC encoding.
    'custom_args_lada': '',
    'custom_args_jasna': '',
    'language': '',
    'dlna_server_name': 'VR Video Server',
    'dlna_port': 8090,
    'dlna_video_dirs': '',
    'dlna_auto_subtitles': True,
    'dlna_si_enabled': True,
    'dlna_si_mix_channel': 'both',
    'dlna_si_original_volume_percent': 100,
    'dlna_si_volume_percent': 100,
    'dlna_si_delay_seconds': 1.0,
    'dlna_si_duck_original': True,
    'dlna_si_duck_preset': 'normal',
    # GPU pipeline from the gpu_engine refactor.
    'transcode_backend': 'auto',  # auto | gpu | ffmpeg
    'mosaic_engine': 'lada',      # lada | jasna | native_gpu placeholder, not implemented.
    'gpu_log_verbose': False,
    'gpu_bitrate_multiplier': 1.2,  # Intermediate target bitrate = source bitrate * this. Keep intermediates clearer than source while limiting temp file size.
    'gpu_bitrate_final_multiplier': 1.0,  # Final OneClick outputs converge to source bitrate by default.
    'gpu_encode_profile': 'balanced_high_quality',  # UI-level profile; expands in utils.encode_config so every OneClick encode stage shares one definition. Known profiles take precedence; the raw keys below are read only for custom/unknown profiles.
    'gpu_encode_preset': 'P4',      # Default profile is balanced high quality: P4 + AQ + fullres multipass.
    'gpu_encode_multipass': 'fullres',  # off | qres | fullres. Two-pass RC improves 8K patch/background quality at fixed bitrate.
    'gpu_encode_aq': True,          # Spatial AQ biases bits toward high-detail restored regions.
    'gpu_encode_aq_strength': 6,     # FFmpeg/Lada spatial AQ strength; matches Lada's historical NVENC preset value.
    'gpu_encode_temporal_aq': False, # Temporal AQ is config-gated pending visual flicker validation on motion-heavy videos.
    'gpu_encode_maxrate_multiplier': 2.0,  # VBR peak cap when no explicit maxrate is supplied; gives extra headroom for detailed intermediate frames.
    'gpu_final_encode_maxrate_multiplier': 1.1,  # Final delivered encode peak cap; keeps keep-bitrate outputs near source bitrate.
    'gpu_final_encode_bframes': 2,  # Final ffmpeg re-encode only. Keep paste/crop frame-accurate paths at bf=0.
    'gpu_final_encode_gop_sec': 2.0,  # Final ffmpeg re-encode GOP length in seconds; 0 disables explicit -g.
    'native_stream_enabled': False,
    'native_detection_model': '',
    'progress_log_interval_s': 5.0,
    'progress_log_min_pct': 5.0,
    'progress_log_vram': True,
    'progress_vram_query_interval_s': 5.0,
    'progress_native_log_interval_s': 5.0,
    'progress_native_log_min_pct': 20.0,
    'output_mp4_faststart': 'auto',  # auto | always | off. Auto disables faststart for very large muxes.
    # OneClick pre-extract: detect/crop mosaic time ranges and regions before sending them to lada/jasna.
    'pre_extract_detection_model': 'lada_vr_mosaic_detection_model_v2_accurate.pt',#'lada_vr_mosaic_detection_model_v2_fast.pt',
    'pre_extract_sample_stride_s': 0.5,
    'pre_extract_yolo_batch': 1,
    'pre_extract_head_tail_pad_s': 2.0,
    'pre_extract_merge_gap_s': 1.5,
    'pre_extract_min_gap_s': 2.0,
    'pre_extract_min_segment_s': 1.5,
    'pre_extract_rect_expand': 1.5,
    'pre_extract_rect_align': 16,
    'pre_extract_rect_min_px': 512,
    'pre_extract_feather_px': 12,
    'pre_extract_yolo_imgsz': 2048,  # Fixed YOLO input size to avoid original-size VRAM spikes.
    'pre_extract_yolo_conf': 0.20,
    'pre_extract_fine_yolo_conf': 0.50,
    'pre_extract_use_mask_boxes': True,
    'pre_extract_cluster_gap_ratio': 0.03,
    'pre_extract_outlier_center_factor': 3.0,
    'pre_extract_spatial_cluster_enabled': True,
    'pre_extract_spatial_cluster_radius_px': 0.0,
    'pre_extract_spatial_cluster_radius_ratio': 0.20,
    'pre_extract_spatial_cluster_radius_factor': 3.0,
    'pre_extract_spatial_cluster_score_ratio': 0.15,
    'pre_extract_spatial_cluster_min_conf': 0.50,
    'pre_extract_spatial_cluster_high_conf': 0.70,
    'pre_extract_spatial_cluster_min_boxes': 2,
    'pre_extract_far_box_min_conf': 0.50,
    'pre_extract_empty_scan_cache': True,
    'pre_extract_pair_min_overlap_s': 0.25,
    'pre_extract_pair_min_spatial_overlap': 0.05,
    'pre_extract_pair_keep_unmatched_conf': 0.60,
    'pre_extract_extract_group_max': 8,
    'pre_extract_pipeline_enabled': False,  # P9 producer-consumer: BROKEN — concurrent NVDEC (extract thread + restore in main) corrupts seek state on subsequent groups, returning content from the prior keyframe (~5s earlier). Default False until per-decoder CUDA-context isolation is added.
    'pre_extract_save_detection_debug': True,
    'pre_extract_keep_segments': False,
    'pre_extract_inject_keyframes': 'auto',
    'pre_extract_inject_gop_sec': 2.0,
    'pre_extract_keyframe_scan_backend': 'auto',  # auto | gpu | cpu. Source keyframe scan uses GPU when safe.
    'paste_passthrough_enabled': True,
    'paste_passthrough_min_frames': 60,
    'paste_passthrough_max_subseg': 32,
    # ThreadedDecoder keeps decoded frames in device memory. At 8K P016 the old
    # default buffer of 32 frames can reserve multiple GiB during final paste.
    'gpu_paste_decoder_buffer_size': 8,
    # OneClick source-scan: scan the source SBS first and process only time ranges containing mosaics.
    'source_scan_enabled': True,
    'source_scan_strategy': 'keyframes',
    'source_scan_scale_max_px': 0,  # legacy; source keyframe scan now uses left-eye original size
    'source_scan_merge_gap_s': 30.0,
    'source_scan_min_segment_s': 30.0,
    'source_scan_head_tail_pad_s': 5.0,
    'source_scan_max_segment_s': 0.0,
    'source_scan_whole_source_min_coverage': 0.98,
    'source_scan_whole_source_max_edge_gap_s': 15.0,
    'source_scan_keep_segments': False,
    'source_scan_fast_hevc_demuxer': False,
    'source_scan_final_merge_mode': 'auto',  # auto | fast | gpu
}

_CODE_DEFAULT_ONLY_PREFIXES = ('pre_extract', 'source_scan')
_CODE_DEFAULT_ONLY_KEYS = {
    'gpu_log_verbose',
    'gpu_bitrate_multiplier',
    'gpu_bitrate_final_multiplier',
    'transcode_backend',
    'mosaic_engine',
    'gpu_encode_preset',
    'gpu_encode_multipass',
    'gpu_encode_aq',
    'gpu_encode_aq_strength',
    'gpu_encode_temporal_aq',
    'gpu_encode_maxrate_multiplier',
    'gpu_final_encode_maxrate_multiplier',
    'gpu_final_encode_bframes',
    'gpu_final_encode_gop_sec',
    'native_stream_enabled',
    'native_detection_model',
    'progress_log_interval_s',
    'progress_log_min_pct',
    'progress_log_vram',
    'progress_vram_query_interval_s',
    'progress_native_log_interval_s',
    'progress_native_log_min_pct',
    'output_mp4_faststart',
    'paste_passthrough_enabled',
    'paste_passthrough_min_frames',
    'paste_passthrough_max_subseg',
    'gpu_paste_decoder_buffer_size',
}

# --- In-memory cache to avoid frequent IO ---
_cache: dict = {}


def _is_code_default_only_key(key: object) -> bool:
    key_text = str(key)
    return key_text in _CODE_DEFAULT_ONLY_KEYS or key_text.startswith(_CODE_DEFAULT_ONLY_PREFIXES)


def _strip_code_default_only(data: dict) -> dict:
    return {key: value for key, value in data.items() if not _is_code_default_only_key(key)}


def _load() -> dict:
    global _cache
    if _cache:
        return _cache
    if os.path.exists(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
                _cache = {**_DEFAULTS, **_strip_code_default_only(data)}
                return _cache
        except Exception:
            pass
    _cache = dict(_DEFAULTS)
    return _cache


def _save(data: dict):
    global _cache
    _cache = data
    try:
        with open(_CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(_strip_code_default_only(data), f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[app_config] Failed to save config: {e}")


def get_engine() -> str:
    """Return the current engine name: 'lada', 'jasna', or 'native_gpu'."""
    return _load().get('engine', _DEFAULTS['engine'])


def set_engine(engine: str):
    """Persist the engine selection immediately to disk."""
    data = _load()
    data['engine'] = engine
    _save(data)


def get_custom_args(engine: str) -> str:
    """Return custom arguments for the given engine."""
    key = f'custom_args_{engine}'
    return _load().get(key, _DEFAULTS.get(key, ''))


def set_custom_args(engine: str, args: str):
    """Persist custom arguments for the given engine."""
    data = _load()
    data[f'custom_args_{engine}'] = args
    _save(data)


def get_system_language() -> str:
    try:
        locale.setlocale(locale.LC_ALL, '')
        sys_lang = locale.getlocale()[0]
        sys_lang_lower = sys_lang.lower() if sys_lang else ''
        if 'zh' in sys_lang_lower or 'chinese' in sys_lang_lower:
            return 'zh'
        if 'ja' in sys_lang_lower or 'japanese' in sys_lang_lower:
            return 'ja'
    except Exception:
        pass
    return 'en'


def normalize_language(language: object) -> str:
    value = str(language or '').strip().lower()
    if value in {'zh', 'zh-cn', 'zh_cn', 'chinese', '简体中文', '中文'}:
        return 'zh'
    if value in {'ja', 'ja-jp', 'ja_jp', 'japanese', '日本語'}:
        return 'ja'
    if value in {'en', 'en-us', 'en_us', 'english'}:
        return 'en'
    return ''


def get_language() -> str:
    stored = normalize_language(_load().get('language', ''))
    return stored or get_system_language()


def set_language(language: str):
    normalized = normalize_language(language)
    if not normalized:
        normalized = get_system_language()
    data = _load()
    data['language'] = normalized
    _save(data)


def get(key: str, default=None):
    if _is_code_default_only_key(key):
        return _DEFAULTS.get(key, default)
    return _load().get(key, default)


def set(key: str, value):
    if _is_code_default_only_key(key):
        return
    data = _load()
    data[key] = value
    _save(data)
