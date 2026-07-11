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

def _split_outputs(input_path, mode, output_dir, to_fisheye):
    """Compute split output paths as {crop_mode: dst}."""
    filename = os.path.splitext(os.path.basename(input_path))[0]
    ext = os.path.splitext(input_path)[1]
    fs = "_fisheye" if to_fisheye else ""
    od = output_dir
    if mode == 'left_and_right':
        return {'left': os.path.join(od, f"{filename}_L{fs}{ext}"),
                'right': os.path.join(od, f"{filename}_R{fs}{ext}")}
    if mode == 'top_and_bottom':
        return {'top': os.path.join(od, f"{filename}_T{fs}{ext}"),
                'bottom': os.path.join(od, f"{filename}_B{fs}{ext}")}
    suffix = {'left': '_L', 'right': '_R', 'top': '_T', 'bottom': '_B'}[mode]
    return {mode: os.path.join(od, f"{filename}{suffix}{fs}{ext}")}


def split_video(input_path, mode, output_dir, to_fisheye=False, log_callback=None, process_callback=None):
    """
    Split VR video.
    mode: 'left', 'right', 'left_and_right', 'top', 'bottom', 'top_and_bottom'
    to_fisheye: if True, convert from VR (hequirect) to fisheye after cropping
    backend=auto prefers GPU and falls back to ffmpeg on failure or unsupported input.
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        out_paths = _split_outputs(input_path, mode, output_dir, to_fisheye)

        from gpu_engine import probe as gpu_probe, fallback as gpu_fallback, files as gpu_files
        meta, decision = gpu_probe.route(input_path)

        def _gpu_fn():
            token = gpu_files.CancelToken()
            if process_callback:
                process_callback(token)
            if log_callback: log_callback(f"Splitting ({mode}{', fisheye' if to_fisheye else ''}) on GPU...")
            gpu_files.split_video(input_path, out_paths, to_fisheye=to_fisheye,
                                  cq=18, keep_audio=True, log_callback=log_callback,
                                  cancel_token=token)
            return True

        def _ffmpeg_fn():
            return _split_video_ffmpeg(input_path, mode, output_dir, to_fisheye, log_callback, process_callback)

        return gpu_fallback.run_with_fallback(
            _gpu_fn, _ffmpeg_fn, gpu_eligible=decision.is_gpu,
            log_callback=log_callback, label=f"split {mode}",
        )
    except Exception as e:
        if log_callback: log_callback(f"Error splitting video: {e}")
        return False


def _split_video_ffmpeg(input_path, mode, output_dir, to_fisheye=False, log_callback=None, process_callback=None):
    """Original ffmpeg split implementation used as the fallback path."""
    try:
        filename = os.path.splitext(os.path.basename(input_path))[0]
        ext = os.path.splitext(input_path)[1]

        # Detect Codec for hardware accel
        if log_callback: log_callback(f"Detecting codec for {input_path}...")
        codec = get_video_codec(input_path)
        decoder_opts = []
        if codec == 'h264':
            decoder_opts = ["-hwaccel", "cuda", "-c:v", "h264_cuvid"]
        elif codec == 'hevc':
            decoder_opts = ["-hwaccel", "cuda", "-c:v", "hevc_cuvid"]
        
        # Dual output mode: use filter_complex to output both files in one pass
        if mode == 'left_and_right':
            fisheye_suffix = "_fisheye" if to_fisheye else ""
            output_left = os.path.join(output_dir, f"{filename}_L{fisheye_suffix}{ext}")
            output_right = os.path.join(output_dir, f"{filename}_R{fisheye_suffix}{ext}")
            
            if to_fisheye:
                filter_complex = "[0:v]crop=iw/2:ih:0:0,v360=hequirect:fisheye[left];[0:v]crop=iw/2:ih:iw/2:0,v360=hequirect:fisheye[right]"
            else:
                filter_complex = "[0:v]crop=iw/2:ih:0:0[left];[0:v]crop=iw/2:ih:iw/2:0[right]"
            
            cmd = ["ffmpeg", "-hide_banner", "-y"]
            cmd.extend(decoder_opts)
            cmd.extend(["-i", input_path])
            cmd.extend(["-filter_complex", filter_complex])
            # Left output
            cmd.extend(["-map", "[left]", "-map", "0:a?", "-c:a", "copy"])
            cmd.extend(["-c:v", "hevc_nvenc", "-preset", "p7", "-cq", "18"])
            cmd.extend([output_left])
            # Right output
            cmd.extend(["-map", "[right]", "-map", "0:a?", "-c:a", "copy"])
            cmd.extend(["-c:v", "hevc_nvenc", "-preset", "p7", "-cq", "18"])
            cmd.extend([output_right])
            
            if log_callback: log_callback(f"Starting split (left_and_right{', to_fisheye' if to_fisheye else ''}) for {input_path}...")
            run_process(cmd, log_callback, process_callback)
            
        elif mode == 'top_and_bottom':
            fisheye_suffix = "_fisheye" if to_fisheye else ""
            output_top = os.path.join(output_dir, f"{filename}_T{fisheye_suffix}{ext}")
            output_bottom = os.path.join(output_dir, f"{filename}_B{fisheye_suffix}{ext}")
            
            if to_fisheye:
                filter_complex = "[0:v]crop=iw:ih/2:0:0,v360=hequirect:fisheye[top];[0:v]crop=iw:ih/2:0:ih/2,v360=hequirect:fisheye[bottom]"
            else:
                filter_complex = "[0:v]crop=iw:ih/2:0:0[top];[0:v]crop=iw:ih/2:0:ih/2[bottom]"
            
            cmd = ["ffmpeg", "-hide_banner", "-y"]
            cmd.extend(decoder_opts)
            cmd.extend(["-i", input_path])
            cmd.extend(["-filter_complex", filter_complex])
            # Top output
            cmd.extend(["-map", "[top]", "-map", "0:a?", "-c:a", "copy"])
            cmd.extend(["-c:v", "hevc_nvenc", "-preset", "p7", "-cq", "18"])
            cmd.extend([output_top])
            # Bottom output
            cmd.extend(["-map", "[bottom]", "-map", "0:a?", "-c:a", "copy"])
            cmd.extend(["-c:v", "hevc_nvenc", "-preset", "p7", "-cq", "18"])
            cmd.extend([output_bottom])
            
            if log_callback: log_callback(f"Starting split (top_and_bottom{', to_fisheye' if to_fisheye else ''}) for {input_path}...")
            run_process(cmd, log_callback, process_callback)
            
        else:
            # Single output mode: left, right, top, or bottom
            fisheye_suffix = "_fisheye" if to_fisheye else ""
            v360_filter = ",v360=hequirect:fisheye" if to_fisheye else ""
            
            if mode == 'left':
                crop_filter = f"crop=iw/2:ih:0:0{v360_filter}"
                suffix = '_L'
            elif mode == 'right':
                crop_filter = f"crop=iw/2:ih:iw/2:0{v360_filter}"
                suffix = '_R'
            elif mode == 'top':
                crop_filter = f"crop=iw:ih/2:0:0{v360_filter}"
                suffix = '_T'
            elif mode == 'bottom':
                crop_filter = f"crop=iw:ih/2:0:ih/2{v360_filter}"
                suffix = '_B'
            else:
                raise ValueError(f"Unknown split mode: {mode}")
            
            output_file = os.path.join(output_dir, f"{filename}{suffix}{fisheye_suffix}{ext}")
            
            cmd = ["ffmpeg", "-hide_banner", "-y"]
            cmd.extend(decoder_opts)
            cmd.extend(["-i", input_path])
            cmd.extend(["-vf", crop_filter])
            cmd.extend(["-c:a", "copy"])
            cmd.extend(["-c:v", "hevc_nvenc", "-preset", "p7", "-cq", "18"])
            cmd.extend([output_file])

            if log_callback: log_callback(f"Starting split ({mode}{', to_fisheye' if to_fisheye else ''}) for {input_path}...")
            run_process(cmd, log_callback, process_callback)

        return True
    except Exception as e:
        if log_callback: log_callback(f"Error splitting video: {e}")
        return False


def combine_video(input_path_1, input_path_2, mode, output_path, from_fisheye=False,
                  bitrate_reference_path=None, log_callback=None, process_callback=None):
    """
    Combine two videos.
    mode: 'left_right' (SBS), 'top_bottom' (Over-Under)
    from_fisheye: if True, convert from fisheye to VR (hequirect) before combining
    backend=auto prefers GPU and falls back to ffmpeg on failure or unsupported input.
    """
    try:
        from gpu_engine import probe as gpu_probe, fallback as gpu_fallback, files as gpu_files
        # Both sources must be GPU-decodable before using the GPU path.
        _, dec1 = gpu_probe.route(input_path_1)
        _, dec2 = gpu_probe.route(input_path_2)
        target_bitrate_bps = None
        max_bitrate_bps = None
        if bitrate_reference_path:
            target_bitrate_bps = get_video_bitrate(bitrate_reference_path, log_callback)
            if target_bitrate_bps:
                target_bitrate_bps = int(target_bitrate_bps)
                max_bitrate_bps = int(target_bitrate_bps * 2)
                if log_callback:
                    log_callback(f"Using bitrate reference: {bitrate_reference_path} -> {target_bitrate_bps / 1_000_000:.2f} Mbps")
            elif log_callback:
                log_callback("Could not detect the reference video bitrate; falling back to CQ 18.")

        def _gpu_fn():
            token = gpu_files.CancelToken()
            if process_callback:
                process_callback(token)
            if log_callback: log_callback(f"Combining (GPU, {mode}{', from_fisheye' if from_fisheye else ''})...")
            gpu_files.combine_video(input_path_1, input_path_2, output_path, mode,
                                    from_fisheye=from_fisheye,
                                    cq=None if target_bitrate_bps else 18,
                                    bitrate_bps=target_bitrate_bps,
                                    max_bitrate_bps=max_bitrate_bps,
                                    keep_audio=True,
                                    log_callback=log_callback, cancel_token=token)
            return True

        def _ffmpeg_fn():
            return _combine_video_ffmpeg(
                input_path_1, input_path_2, mode, output_path, from_fisheye,
                log_callback, process_callback, target_bitrate_bps, max_bitrate_bps,
            )

        return gpu_fallback.run_with_fallback(
            _gpu_fn, _ffmpeg_fn, gpu_eligible=(dec1.is_gpu and dec2.is_gpu),
            log_callback=log_callback, label=f"combine {mode}",
        )
    except Exception as e:
        if log_callback: log_callback(f"Error combining videos: {e}")
        return False


def _combine_video_ffmpeg(input_path_1, input_path_2, mode, output_path, from_fisheye=False,
                          log_callback=None, process_callback=None,
                          bitrate_bps=None, max_bitrate_bps=None):
    """Original ffmpeg combine implementation used as the fallback path."""
    try:
        if log_callback: log_callback(f"Combining {input_path_1} and {input_path_2} into {output_path} (Mode: {mode})")

        # Detect Codec (Assume same codec or handle generic)
        # For simplicity, apply hwaccel to both if possible, or just let ffmpeg handle it.
        # Given potential different codecs, let's try to be smart or just safe.
        # Reference used hevc_cuvid for input.
        
        codec1 = get_video_codec(input_path_1)
        codec2 = get_video_codec(input_path_2)

        input_opts_1 = []
        if codec1 == 'h264':
            input_opts_1 = ["-hwaccel", "cuda", "-c:v", "h264_cuvid"]
        elif codec1 == 'hevc':
            input_opts_1 = ["-hwaccel", "cuda", "-c:v", "hevc_cuvid"]

        input_opts_2 = []
        if codec2 == 'h264':
            input_opts_2 = ["-hwaccel", "cuda", "-c:v", "h264_cuvid"]
        elif codec2 == 'hevc':
            input_opts_2 = ["-hwaccel", "cuda", "-c:v", "hevc_cuvid"]
        
        cmd = ["ffmpeg", "-hide_banner", "-y"]
        cmd.extend(input_opts_1)
        cmd.extend(["-i", input_path_1])
        cmd.extend(input_opts_2)
        cmd.extend(["-i", input_path_2])

        if from_fisheye:
            # Convert fisheye to VR before stacking
            if mode == 'left_right':
                filter_complex = "[0:v]v360=fisheye:hequirect[left];[1:v]v360=fisheye:hequirect[right];[left][right]hstack=inputs=2[v]"
            elif mode == 'top_bottom':
                filter_complex = "[0:v]v360=fisheye:hequirect[top];[1:v]v360=fisheye:hequirect[bottom];[top][bottom]vstack=inputs=2[v]"
            else:
                raise ValueError(f"Unknown combine mode: {mode}")
        else:
            if mode == 'left_right':
                filter_complex = "[0:v][1:v]hstack=inputs=2[v]"
            elif mode == 'top_bottom':
                filter_complex = "[0:v][1:v]vstack=inputs=2[v]"
            else:
                raise ValueError(f"Unknown combine mode: {mode}")

        cmd.extend(["-filter_complex", filter_complex])
        cmd.extend(["-map", "[v]", "-map", "0:a?"]) # Take audio from first video
        cmd.extend(["-c:a", "copy"])
        
        cmd.extend(["-c:v", "hevc_nvenc", "-preset", "p7"])
        if bitrate_bps:
            cmd.extend([
                "-rc", "vbr", "-b:v", str(int(bitrate_bps)),
                "-maxrate", str(int(max_bitrate_bps or bitrate_bps * 2)),
                "-bufsize", str(int((max_bitrate_bps or bitrate_bps * 2) * 2)),
            ])
        else:
            cmd.extend(["-cq", "18"])
        # Ensure consistent color (optional but good practice as seen in reference)
        # cmd.extend(["-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"])
        
        cmd.extend([output_path])

        run_process(cmd, log_callback, process_callback)
        return True

    except Exception as e:
        if log_callback: log_callback(f"Error combining videos: {e}")
        return False
