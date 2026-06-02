import subprocess
import os
import json
import shutil
import bisect
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


def get_video_info(file_path):
    try:
        # Get Duration
        cmd_dur = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path]
        print(f"Executing: {' '.join(cmd_dur)}")
        duration_str = subprocess.check_output(cmd_dur, startupinfo=get_startupinfo(), text=True, encoding='utf-8', errors='replace').strip()
        duration = float(duration_str)

        # Get Resolution (Width/Height)
        cmd_res = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", file_path]
        print(f"Executing: {' '.join(cmd_res)}")
        res_str = subprocess.check_output(cmd_res, startupinfo=get_startupinfo(), text=True, encoding='utf-8', errors='replace').strip()
        width, height = map(int, res_str.split(','))

        return {"duration": duration, "width": width, "height": height}
    except Exception as e:
        print(f"Error getting video info: {e}")
        return None

def get_video_codec(file_path):
    try:
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0", file_path]
        print(f"Executing: {' '.join(cmd)}")
        codec = subprocess.check_output(cmd, startupinfo=get_startupinfo(), text=True, encoding='utf-8', errors='replace').strip()
        return codec
    except Exception as e:
        print(f"Error getting video codec: {e}")
        return None

def time_to_sec(t):
    if not t: return 0
    parts = list(map(float, t.split(':')))
    if len(parts) == 1: return parts[0]
    if len(parts) == 2: return parts[0]*60 + parts[1]
    if len(parts) == 3: return parts[0]*3600 + parts[1]*60 + parts[2]
    return 0

def extract_clip(input_file, is_left_eye, start_time, end_time, log_callback=None, process_callback=None):
    try:
        directory = os.path.dirname(input_file)
        filename = os.path.splitext(os.path.basename(input_file))[0]
        ext = os.path.splitext(input_file)[1]
        
        side_suffix = "_L" if is_left_eye else "_R"
        crop_filter = "crop=iw/2:ih:0:0" if is_left_eye else "crop=iw/2:ih:iw/2:0"
        
        ss_part = start_time.replace(":", "") if start_time else "START"
        to_part = end_time.replace(":", "") if end_time else "END"
        
        output_file = os.path.join(directory, f"{filename}{side_suffix}_S{ss_part}_E{to_part}{ext}")
        
        # Detect Codec
        if log_callback: log_callback(f"Detecting codec for {input_file}...")
        codec = get_video_codec(input_file)
        if log_callback: log_callback(f"Detected codec: {codec}")
        
        decoder_opts = []
        if codec == 'h264':
            decoder_opts = ["-hwaccel", "cuda", "-c:v", "h264_cuvid"]
        elif codec == 'hevc':
            decoder_opts = ["-hwaccel", "cuda", "-c:v", "hevc_cuvid"]
        else:
            if log_callback: log_callback(f"Warning: Codec {codec} not explicitly supported for hardware decoding in this script. Using default.")
        
        cmd = ["ffmpeg"]
        if start_time: cmd.extend(["-ss", start_time])
        if end_time: cmd.extend(["-to", end_time])
        
        cmd.extend(["-hide_banner"])
        cmd.extend(decoder_opts)
        
        cmd.extend([
            "-i", input_file,
            "-vf", crop_filter,
            "-c:a", "copy",
            "-c:v", "hevc_nvenc", "-preset", "p7", "-cq", "18",
            "-pix_fmt", "p010le", "-color_range", "tv",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            output_file, "-y"
        ])
        
        if log_callback: log_callback(f"Starting extraction for {input_file}...")
        if log_callback: log_callback(f"Output: {output_file}")
        
        run_process(cmd, log_callback, process_callback)
        return output_file
    except Exception as e:
        if log_callback: log_callback(f"Error: {e}")
        return None

def get_vr_frame(input_file, time, output_image):
    try:
        # Extract full VR frame (Equirectangular)
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", str(time),
            "-i", input_file,
            "-frames:v", "1",
            "-y", output_image
        ]
        print(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, startupinfo=get_startupinfo())
        return True
    except Exception as e:
        print(f"Error getting VR frame: {e}")
        return False

def get_vr_frame_image(input_file, time):
    import av

    target_seconds = max(0.0, float(time))
    container = av.open(input_file)
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if target_seconds > 0 and stream.time_base:
            container.seek(int(target_seconds / stream.time_base), backward=True, stream=stream)

        selected_frame = None
        for frame in container.decode(stream):
            if frame.pts is None:
                selected_frame = frame
                break
            frame_time = float(frame.pts * stream.time_base)
            selected_frame = frame
            if frame_time >= target_seconds:
                break

        if selected_frame is None:
            raise RuntimeError("Unable to decode a video frame at the selected time")
        return selected_frame.to_image().convert("RGB")
    finally:
        container.close()

def get_flat_frame(input_file, time, yaw, pitch, fov, flat_w, flat_h, output_image):
    try:
        # Extract Flat frame using v360 filter (Step 2 method)
        # v360=hequirect:flat:d_fov={fov}:yaw={yaw}:pitch={pitch}
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", str(time),
            "-i", input_file,
            "-vf", f"v360=hequirect:flat:d_fov={fov}:yaw={yaw}:pitch={pitch}:w={flat_w}:h={flat_h}",
            "-frames:v", "1",
            "-y", output_image
        ]
        print(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, startupinfo=get_startupinfo())
        return True
    except Exception as e:
        print(f"Error getting Flat frame: {e}")
        return False

def run_pipeline(input_file, yaw, pitch, fov, width, height, log_callback=None, process_callback=None):
    try:
        directory = os.path.dirname(input_file)
        filename = os.path.splitext(os.path.basename(input_file))[0]

        # Force width/height to be multiple of 10
        width = round(width / 10) * 10
        height = round(height / 10) * 10

        suffix = f"_Y{int(yaw)}_P{int(pitch)}_D{int(fov)}"
        final_output = os.path.join(directory, f"{filename}_flat{suffix}.mp4")

        if os.path.exists(final_output):
            if log_callback: log_callback(f"Output file exists: {final_output}. Skipping.")
            return

        from gpu_engine import probe as gpu_probe, fallback as gpu_fallback, files as gpu_files

        meta, decision = gpu_probe.route(input_file)

        def _gpu_fn():
            token = gpu_files.CancelToken()
            if process_callback:
                process_callback(token)
            if log_callback: log_callback("Step 1: Extracting Flat Video (GPU)...")
            gpu_files.vr_to_flat(
                input_file, final_output, yaw, pitch, fov, width, height,
                cq=18, keep_audio=True, log_callback=log_callback, cancel_token=token,
            )
            if log_callback: log_callback(f"Done! Output: {final_output}")
            return True

        def _ffmpeg_fn():
            return _run_pipeline_ffmpeg(input_file, final_output, yaw, pitch, fov, width, height, log_callback, process_callback)

        return gpu_fallback.run_with_fallback(
            _gpu_fn, _ffmpeg_fn, gpu_eligible=decision.is_gpu,
            log_callback=log_callback, label="vr2flat",
        )
    except Exception as e:
        if log_callback: log_callback(f"Error in pipeline: {e}")


def _run_pipeline_ffmpeg(input_file, final_output, yaw, pitch, fov, width, height, log_callback=None, process_callback=None):
    """Original ffmpeg VR->flat implementation used as the fallback path."""
    try:
        if log_callback: log_callback("Step 1: Extracting Flat Video (ffmpeg)...")
        cmd1 = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-stats"]
        cmd1.extend(["-hwaccel", "cuda", "-c:v", "hevc_cuvid"])
        cmd1.extend([
            "-i", input_file,
            "-vf", f"scale=in_color_matrix=bt709,format=yuv420p10le,v360=hequirect:flat:d_fov={fov}:yaw={yaw}:pitch={pitch}:w={width}:h={height}:rorder=ypr,scale=out_color_matrix=bt709:out_range=limited,format=yuv420p10le",
            "-c:a", "copy", "-c:v", "hevc_nvenc", "-preset", "p7", "-cq", "18",
            "-pix_fmt", "p010le", "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            final_output, "-y"
        ])
        run_process(cmd1, log_callback, process_callback)
        if log_callback: log_callback(f"Done! Output: {final_output}")
        return True
    except Exception as e:
        if log_callback: log_callback(f"Error in pipeline: {e}")
        return False


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
        # If killed, returncode might be non-zero (e.g. 1 or -9). 
        # We can check if it was intended, but for now just raise exception or let it pass if killed?
        # Usually we want to know if it failed.
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
