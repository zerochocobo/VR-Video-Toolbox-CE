import subprocess
import os
import shutil
import json
import sys

try:
    from utils.ffmpeg_checker import get_startupinfo, get_video_bitrate
except ImportError:
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from utils.ffmpeg_checker import get_startupinfo, get_video_bitrate

def check_dependencies():
    missing = []
    for tool in ["ffmpeg", "ffprobe"]:
        if not shutil.which(tool):
            missing.append(tool)
    return missing


def get_video_codec(file_path):
    try:
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0", file_path]
        codec = subprocess.check_output(cmd, startupinfo=get_startupinfo(), text=True, encoding='utf-8', errors='replace').strip()
        return codec
    except Exception as e:
        print(f"Error getting video codec: {e}")
        return None

def run_process(cmd, log_callback, process_callback=None):
    if log_callback:
        log_callback(f"Executing: {' '.join(cmd)}")
    else:
        print(f"Executing: {' '.join(cmd)}")
        
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, errors='replace', startupinfo=get_startupinfo())
    if process_callback: process_callback(process)
    
    for line in process.stdout:
        if log_callback: log_callback(line.strip())
    process.wait()
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

def _output_path(input_path, output_dir, mode):
    """Compute the output file path from mode using the naming logic."""
    filename = os.path.splitext(os.path.basename(input_path))[0]
    ext = os.path.splitext(input_path)[1]
    if mode == 'hequirect2fisheye':
        final_name = filename + '_fisheye'
    elif mode == 'fisheye2hequirect':
        final_name = filename.replace('_fisheye', '').replace('fisheye', '')
        final_name = final_name.replace('__', '_').strip('_') + '_hequirect'
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return os.path.join(output_dir, final_name + ext)


# mode name -> gpu_engine projection kind.
_GPU_MODE = {'hequirect2fisheye': 'heq2fisheye', 'fisheye2hequirect': 'fisheye2heq'}


def convert_projection(input_path, output_dir, mode, dual_screen=False, keep_original_bitrate=False, log_callback=None, process_callback=None):
    """
    Convert VR projection.
    mode: 'hequirect2fisheye', 'fisheye2hequirect'
    dual_screen: if True, process SBS video (split, convert, merge back to SBS)
    keep_original_bitrate: if True, use original video bitrate for output

    backend=auto prefers GPU through gpu_engine and automatically falls back to
    ffmpeg on failure. HDR10 or unsupported sources are statically routed to ffmpeg.
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = _output_path(input_path, output_dir, mode)

        from gpu_engine import probe as gpu_probe, fallback as gpu_fallback, files as gpu_files

        meta, decision = gpu_probe.route(input_path)
        gpu_kind = _GPU_MODE.get(mode)

        def _gpu_fn():
            token = gpu_files.CancelToken()
            if process_callback:
                process_callback(token)
            gpu_files.vr_projection(
                input_path, output_file, gpu_kind,
                dual_screen=dual_screen,
                cq=None if keep_original_bitrate else 18,
                bitrate_bps=(meta.bitrate_bps if keep_original_bitrate else None),
                keep_audio=True, log_callback=log_callback, cancel_token=token,
            )
            return True

        def _ffmpeg_fn():
            return _convert_projection_ffmpeg(
                input_path, output_file, mode, dual_screen,
                keep_original_bitrate, log_callback, process_callback,
            )

        return gpu_fallback.run_with_fallback(
            _gpu_fn, _ffmpeg_fn,
            gpu_eligible=(decision.is_gpu and gpu_kind is not None),
            log_callback=log_callback, label=f"v360 {mode}",
        )
    except Exception as e:
        if log_callback:
            log_callback(f"Error converting video: {e}")
        return False


def _convert_projection_ffmpeg(input_path, output_file, mode, dual_screen=False, keep_original_bitrate=False, log_callback=None, process_callback=None):
    """Original ffmpeg implementation used as the fallback path."""
    try:
        # Detect Codec for hardware accel
        if log_callback: log_callback(f"Detecting codec for {input_path}...")
        codec = get_video_codec(input_path)
        decoder_opts = []
        if codec == 'h264':
            decoder_opts = ["-hwaccel", "cuda", "-c:v", "h264_cuvid"]
        elif codec == 'hevc':
            decoder_opts = ["-hwaccel", "cuda", "-c:v", "hevc_cuvid"]
        
        cmd = ["ffmpeg", "-hide_banner", "-y"]
        cmd.extend(decoder_opts)
        cmd.extend(["-i", input_path])
        
        if dual_screen:
            # Dual screen mode: split, convert each half, merge back to SBS
            if mode == 'hequirect2fisheye':
                filter_complex = "[0:v]split=2[l_src][r_src];[l_src]crop=iw/2:ih:0:0,v360=hequirect:fisheye[left];[r_src]crop=iw/2:ih:iw/2:0,v360=hequirect:fisheye[right];[left][right]hstack=inputs=2[v]"
            elif mode == 'fisheye2hequirect':
                filter_complex = "[0:v]split=2[l_src][r_src];[l_src]crop=iw/2:ih:0:0,v360=fisheye:hequirect[left];[r_src]crop=iw/2:ih:iw/2:0,v360=fisheye:hequirect[right];[left][right]hstack=inputs=2[v]"
            
            cmd.extend(["-filter_complex", filter_complex])
            cmd.extend(["-map", "[v]", "-map", "0:a?"])
        else:
            # Single screen mode
            if mode == 'hequirect2fisheye':
                projection_filter = "v360=hequirect:fisheye"
            elif mode == 'fisheye2hequirect':
                projection_filter = "v360=fisheye:hequirect"
            
            cmd.extend(["-vf", projection_filter])
        
        cmd.extend(["-c:a", "copy"])
        
        if keep_original_bitrate:
            # Use original bitrate for output
            original_bitrate = get_video_bitrate(input_path, log_callback)
            if original_bitrate:
                target_kbps = int(original_bitrate / 1000 * 1.2)
                target_bitrate = f"{target_kbps}k"
                max_rate = f"{int(target_kbps * 1.2)}k"
                buf_size = f"{int(target_kbps * 2)}k"
                cmd.extend([
                    "-c:v", "hevc_nvenc", 
                    "-preset", "p7", 
                    "-rc", "vbr",
                    "-b:v", target_bitrate,
                    "-maxrate:v", max_rate,
                    "-bufsize:v", buf_size
                ])
            else:
                # Fallback to CQ mode if bitrate detection fails
                cmd.extend([
                    "-c:v", "hevc_nvenc", 
                    "-preset", "p7", 
                    "-cq", "18"
                ])
        else:
            # Use CQ mode for quality control
            cmd.extend([
                "-c:v", "hevc_nvenc", 
                "-preset", "p7", 
                "-cq", "18"
            ])
        cmd.extend([output_file])

        mode_desc = f"{mode}{', dual_screen' if dual_screen else ''}"
        if log_callback: log_callback(f"Starting conversion ({mode_desc}) for {input_path}...")
        run_process(cmd, log_callback, process_callback)

        return True
    except Exception as e:
        if log_callback: log_callback(f"Error converting video: {e}")
        return False
