import subprocess
import os
import shutil
import json
import bisect
import sys

# Import engine layer and helper methods.
try:
    from utils import engine_runner
    from utils.ffmpeg_checker import get_startupinfo, get_video_bitrate
except ImportError:
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from utils import engine_runner
    from utils.ffmpeg_checker import get_startupinfo, get_video_bitrate

def check_dependencies():
    missing = []
    tools = ["ffmpeg", "ffprobe"]
    if not engine_runner.is_native_engine():
        engine_cli = engine_runner.get_engine_executable()
        if engine_cli:
            tools.append(engine_cli)
    for tool in tools:
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
            "-c:v", "hevc_nvenc", "-preset", "p7", 
            "-cq", "18",
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

def extract_clip_both(input_file, start_time, end_time, log_callback=None, process_callback=None):
    """Extract both left and right eye clips in a single ffmpeg call."""
    try:
        directory = os.path.dirname(input_file)
        filename = os.path.splitext(os.path.basename(input_file))[0]
        ext = os.path.splitext(input_file)[1]
        
        ss_part = start_time.replace(":", "") if start_time else "START"
        to_part = end_time.replace(":", "") if end_time else "END"
        
        output_left = os.path.join(directory, f"{filename}_L_S{ss_part}_E{to_part}{ext}")
        output_right = os.path.join(directory, f"{filename}_R_S{ss_part}_E{to_part}{ext}")
        
        # Detect Codec
        if log_callback: log_callback(f"Detecting codec for {input_file}...")
        codec = get_video_codec(input_file)
        if log_callback: log_callback(f"Detected codec: {codec}")
        
        decoder_opts = []
        if codec == 'h264':
            decoder_opts = ["-hwaccel", "cuda", "-c:v", "h264_cuvid"]
        elif codec == 'hevc':
            decoder_opts = ["-hwaccel", "cuda", "-c:v", "hevc_cuvid"]
        
        cmd = ["ffmpeg"]
        if start_time: cmd.extend(["-ss", start_time])
        if end_time: cmd.extend(["-to", end_time])
        cmd.extend(["-hide_banner"])
        cmd.extend(decoder_opts)
        cmd.extend(["-i", input_file])
        
        # Use filter_complex to output both files in one pass
        filter_complex = "[0:v]crop=iw/2:ih:0:0[left];[0:v]crop=iw/2:ih:iw/2:0[right]"
        cmd.extend(["-filter_complex", filter_complex])
        
        # Left output
        cmd.extend(["-map", "[left]", "-map", "0:a?", "-c:a", "copy"])
        cmd.extend(["-c:v", "hevc_nvenc", "-preset", "p7", 
                    "-cq", "18",
                    "-pix_fmt", "p010le", "-color_range", "tv",
                    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"])
        cmd.extend([output_left, "-y"])
        
        # Right output
        cmd.extend(["-map", "[right]", "-map", "0:a?", "-c:a", "copy"])
        cmd.extend(["-c:v", "hevc_nvenc", "-preset", "p7", 
                    "-cq", "18",
                    "-pix_fmt", "p010le", "-color_range", "tv",
                    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"])
        cmd.extend([output_right, "-y"])
        
        if log_callback: log_callback(f"Starting extraction (dual)...")
        if log_callback: log_callback(f"Output: {output_left}, {output_right}")
        
        run_process(cmd, log_callback, process_callback)
        return output_left, output_right
    except Exception as e:
        if log_callback: log_callback(f"Error: {e}")
        return None, None


def get_preview_frame(input_file, time, output_image):
    try:
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
        print(f"Error getting preview frame: {e}")
        return False

def get_preview_frame_image(input_file, time):
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


def time_to_sec(t):
    if not t: return 0
    parts = list(map(float, t.split(':')))
    if len(parts) == 1: return parts[0]
    if len(parts) == 2: return parts[0]*60 + parts[1]
    if len(parts) == 3: return parts[0]*3600 + parts[1]*60 + parts[2]
    return 0

def run_pipeline(input_file, crop_w, crop_h, center_x, center_y, start_time=None, end_time=None, log_callback=None, process_callback=None, keep_intermediate=False, overwrite_input=False):
    try:
        directory = os.path.dirname(input_file)
        filename = os.path.splitext(os.path.basename(input_file))[0]
        
        # Ensure integers
        crop_w = int(float(crop_w))
        crop_h = int(float(crop_h))
        center_x = int(float(center_x))
        center_y = int(float(center_y))
        
        # Force Mod 10 (as per batch script)
        crop_w = (crop_w // 10) * 10
        crop_h = (crop_h // 10) * 10
        if crop_w <= 0: crop_w = 10
        if crop_h <= 0: crop_h = 10
        
        # Calculate Top-Left
        half_w = crop_w // 2
        half_h = crop_h // 2
        tl_x = center_x - half_w
        tl_y = center_y - half_h
        
        # Construct Suffix
        time_suffix = ""
        start_seconds = end_seconds = 0
        if start_time:
            safe_start = start_time.replace(":", "")
            time_suffix += f"_ST{safe_start}"
            start_seconds = time_to_sec(start_time)
        if end_time:
            safe_end = end_time.replace(":", "")
            time_suffix += f"_ET{safe_end}"
            end_seconds = time_to_sec(end_time)
        
        suffix = f"{time_suffix}_CX{center_x}_CY{center_y}_W{crop_w}_H{crop_h}"
        
        crop_extract = os.path.join(directory, f"{filename}_crop{suffix}.mp4")
        crop_restored = os.path.join(directory, f"{filename}_crop{suffix}.restored.mp4")
        final_output = os.path.join(directory, f"{filename}.restored.mp4")
        
        if os.path.exists(final_output):
             if log_callback: log_callback(f"Output file exists: {final_output}. Skipping.")
             return

        # Detect Codec
        if log_callback: log_callback(f"Detecting codec for {input_file}...")
        codec = get_video_codec(input_file)
        decoder_opts = []
        if codec == 'h264':
            decoder_opts = ["-hwaccel", "cuda", "-c:v", "h264_cuvid"]
        elif codec == 'hevc':
            decoder_opts = ["-hwaccel", "cuda", "-c:v", "hevc_cuvid"]
        
        # Step 1: Extract Crop
        if log_callback: log_callback("Step 1: Extracting Crop...")
        
        vf_crop = f"crop={crop_w}:{crop_h}:{tl_x}:{tl_y},scale=in_color_matrix=bt709:out_color_matrix=bt709,format=yuv420p10le"
        
        cmd1 = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-stats"]
        
        # Time options for input
        if start_time:
            cmd1.extend(["-ss", str(start_seconds)])
        if end_time:
            cmd1.extend(["-ss", str(start_seconds)])
            cmd1.extend(["-to", str(end_seconds)])
            
        cmd1.extend(decoder_opts)
        cmd1.extend([
            "-i", input_file,
            "-vf", vf_crop,
            "-c:a", "copy",
            "-c:v", "hevc_nvenc", "-preset", "p7", "-cq", "18",
            "-pix_fmt", "p010le", "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            crop_extract, "-y"
        ])
        run_process(cmd1, log_callback, process_callback)
        
        # Step 2: mosaic removal with Lada / Jasna / built-in engine.
        tool_name = engine_runner.get_mosaic_tool_name()
        if log_callback: log_callback(f"Step 2: Running {tool_name}...")
        if engine_runner.is_native_engine():
            from gpu_engine import native_mosaic
            from gpu_engine.files import CancelToken
            token = CancelToken()
            if process_callback:
                process_callback(token)
            ok = native_mosaic.restore_file(crop_extract, crop_restored, log_callback=log_callback, cancel_token=token)
            if not ok:
                raise Exception("native_gpu restore failed or was cancelled")
        else:
            cmd2 = engine_runner.build_engine_cmd(
                input_file=crop_extract,
                output_file=crop_restored,
                encoder_options=" -cq 18 -preset p7 -pix_fmt p010le -color_range tv -colorspace bt709 -color_primaries bt709 -color_trc bt709",
            )
            run_process(cmd2, log_callback, process_callback)
        
        #get video duration
        video_info = get_video_info(crop_restored)
        video_duration = video_info.get("duration", 0)
        if video_duration < 1:
            if log_callback: log_callback("Error: Invalid video duration.")
            return
        
        # Step 3: Overlay
        if log_callback: log_callback("Step 3: Overlaying...")
        
        background_file = input_file
        output_file = final_output
        
        # Input 1: Restored Crop (with offset if needed)

        clip_intermediate_files = []
        concat_files = []
        real_start_time = 0
        if start_seconds > 0 or end_seconds > 0:
            #get keyframes information for Keyframe Alignment
            
            if log_callback: log_callback("Getting Keyframes before split...")
            cmd = [
                "ffprobe", 
                "-loglevel", "error",
                "-select_streams", "v:0",
                "-show_entries", "packet=pts_time,flags",
                "-of", "json",
                input_file
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, startupinfo=get_startupinfo())
            data = json.loads(result.stdout)
            
            # filter flag contains 'K' (Keyframe)
            keyframes = []
            for packet in data.get('packets', []):
                if 'K' in packet.get('flags', ''):
                    keyframes.append(float(packet['pts_time']))
            keyframes = sorted(keyframes)

            if not keyframes:
                if log_callback: log_callback("Error: No keyframes detected, cannot perform smart cutting.")
                return None

            idx_start = bisect.bisect_right(keyframes, start_seconds) - 1
            if idx_start < 0: idx_start = 0
            real_start_time = keyframes[idx_start]

            # Find the most recent keyframe after the patch ends as the starting point of the Tail
            desired_end_time = start_seconds + video_duration
            idx_end = bisect.bisect_right(keyframes, desired_end_time)
            if idx_end >= len(keyframes):
                real_end_time = keyframes[-1] # If it exceeds, take the last one
            else:
                real_end_time = keyframes[idx_end]

            print(f"Original: {start_seconds}s -> {start_seconds + video_duration}s")
            print(f"Smart Adjusted: {real_start_time}s -> {real_end_time}s (Aligned with Keyframes)")

            # Calculate patch delay
            # Because Body started earlier (real_start_time < start_seconds)
            # So the patch needs to be displayed (start_seconds - real_start_time) seconds later
            # patch_delay = start_seconds - real_start_time

            # New total duration of Body
            # body_duration = real_end_time - real_start_time

            if real_start_time > 0:
                temp_head = os.path.join(directory, f"{filename}_crop{suffix}.head.mp4")
                clip_intermediate_files.append(temp_head)
                concat_files.append(temp_head)
                cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-stats"]
                cmd.extend(["-ss", "0"])
                cmd.extend(["-to", str(real_start_time)])
                cmd.extend(["-i", input_file])
                cmd.extend(["-c", "copy"])
                cmd.extend([temp_head, "-y"])
                run_process(cmd, log_callback, process_callback)

            temp_body = os.path.join(directory, f"{filename}_crop{suffix}.body.mp4")
            clip_intermediate_files.append(temp_body)
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-stats"]
            cmd.extend(["-ss", str(real_start_time)])
            if end_seconds > 0:
                cmd.extend(["-to", str(real_end_time)])
            cmd.extend(["-i", input_file])
            cmd.extend(["-c", "copy"])
            cmd.extend(["-avoid_negative_ts", "make_zero"])
            cmd.extend([temp_body, "-y"])
            run_process(cmd, log_callback, process_callback)
            background_file = temp_body
            output_file = os.path.join(directory, f"{filename}_crop{suffix}.body.restored.mp4")
            clip_intermediate_files.append(output_file)
            concat_files.append(output_file)

            if end_seconds > 0:
                temp_tail = os.path.join(directory, f"{filename}_crop{suffix}.tail.mp4")
                clip_intermediate_files.append(temp_tail)
                concat_files.append(temp_tail)
                cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-stats"]
                cmd.extend(["-ss", str(real_end_time)])
                cmd.extend(["-i", input_file])
                cmd.extend(["-c", "copy"])
                cmd.extend(["-avoid_negative_ts", "make_zero"])
                cmd.extend([temp_tail, "-y"])
                run_process(cmd, log_callback, process_callback)

        # if start_time:
        #     cmd3.extend(["-itsoffset", start_seconds])
        


        # get background_file bitrate
        input_bitrate = 60000000
        try:
            cmd = [
                "ffprobe", "-v", "error", 
                "-select_streams", "v:0", 
                "-show_entries", "stream=bit_rate", 
                "-of", "default=noprint_wrappers=1:nokey=1", 
                background_file
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            bitrate = int(result.stdout.strip())
            input_bitrate = bitrate
        except:
            input_bitrate = None 
        
        if not input_bitrate or input_bitrate == 0:
            if log_callback: log_callback("Warning: Could not detect bitrate, using default 60M.")
            target_bitrate = 60000000
        else:
            target_bitrate = input_bitrate * 1.2

        max_bitrate = int(target_bitrate * 1.2)
        buf_size = int(target_bitrate * 2)

        cmd3 = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-stats"]
        cmd3.extend(decoder_opts)
        
        
        # Input 1: VR Patch
        # Note: When using split body, the timestamp of body starts at 0.
        # The flat_restored also starts at 0 (relative to the clip).
        # However, the flat_restored corresponds to the original video's [start_seconds, start_seconds+duration].
        # The body corresponds to [real_start_time, real_end_time].
        # So the patch needs to be delayed by (start_seconds - real_start_time).
        
        patch_delay = start_seconds - real_start_time if (start_seconds > 0) else 0
        
        # Also, the patch itself is just the restored part.
        # We need to project it back to equirectangular.
        # And overlay it on the background (which is now the body clip).
        
        # filter_complex logic:
        # 1. Project flat_restored to equirectangular (hequirect).
        # 2. Overlay on background.
        
        # If we have a patch_delay, we need to setpts of the patch.
        
        
        # Input 0: Original File
        cmd3.extend(["-i", background_file])
        
        # Input 1: VR Patch (with offset if needed)
        cmd3.extend(["-c:v", "hevc_cuvid", "-i", crop_restored])
        if real_start_time > 0:
            vf_overlay = f"[1:v]setpts=PTS-STARTPTS+{patch_delay}/TB[delayed];[0:v][delayed]overlay=x={tl_x}:y={tl_y}:eof_action=pass:format=auto[v]"
        else:
            vf_overlay = f"[0:v][1:v]overlay=x={tl_x}:y={tl_y}:eof_action=pass:format=auto[v]"
        
        cmd3.extend([
            "-filter_complex", vf_overlay,
            "-map", "[v]", "-map", "0:a?",
            "-c:v", "hevc_nvenc", "-preset", "p7", 
            
            # Use CBR/VBR hybrid mode, not pure CQ
            # Note: After setting -b:v, -cq is usually used with -rc vbr, or directly overridden by bitrate.
            # For the most stable concatenation, it is recommended to strictly limit the bitrate:
            # "-rc", "vbr_hq",       # Use high-quality variable bitrate control
            "-rc", "cbr", 
            "-b:v", str(target_bitrate),
            "-maxrate", str(target_bitrate), # maxrate must equal b:v.
            "-minrate", str(target_bitrate), # minrate must equal b:v; this is the key forced-padding setting.
            "-bufsize", str(target_bitrate * 2),
            "-bufsize", str(buf_size), 
            
            # Ensure keyframe interval is as consistent as possible with the original video (VR videos usually have short GOP, e.g., 120 or 60 for 60fps)
            # If unsure, it can be left unset, but setting it makes concatenation more stable
            # "-g", "120", 

            "-pix_fmt", "p010le", 
            "-color_range", "tv", 
            "-colorspace", "bt709", 
            "-color_primaries", "bt709", 
            "-color_trc", "bt709",

           
            # "-cq", "18",
            # "-pix_fmt", "p010le", "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-c:a", "copy",
            output_file, "-y"
        ])
        run_process(cmd3, log_callback, process_callback)
        concat_list_file = os.path.join(directory,f"{filename}_crop{suffix}.concat_list.txt")
        if len(concat_files) > 0:
            with open(concat_list_file, "w", encoding="utf-8") as f:
                for clip_file in concat_files:
                    f.write(f"file '{clip_file}'\n")

            #merge
            cmd_concat = [
                "ffmpeg", "-hide_banner", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_list_file,
                "-c", "copy", # copy fast
                final_output
            ]
            run_process(cmd_concat, log_callback, process_callback)

        # Cleanup
        if not keep_intermediate:
            if os.path.exists(crop_extract): os.remove(crop_extract)
            if os.path.exists(crop_restored): os.remove(crop_restored)
            if len(clip_intermediate_files) > 0:
                for clip_file in clip_intermediate_files:
                    if os.path.exists(clip_file): os.remove(clip_file)
            if os.path.exists(concat_list_file): os.remove(concat_list_file)

        
        # Overwrite Logic
        if overwrite_input:
            if log_callback: log_callback(f"Overwriting input file: {input_file}")
            try:
                shutil.move(final_output, input_file)
                final_output = input_file
            except Exception as e:
                if log_callback: log_callback(f"Error overwriting input file: {e}")

        if log_callback: log_callback(f"Done! Output: {final_output}")
        
    except Exception as e:
        if log_callback: log_callback(f"Error in pipeline: {e}")

def merge_channels(left_file, right_file, log_callback=None, process_callback=None):
    try:
        directory = os.path.dirname(left_file)
        filename = os.path.splitext(os.path.basename(left_file))[0]
        final_name = filename.replace("_L_", "_sbs_").replace("_l_", "_sbs_")
        if final_name == filename: final_name += "_sbs"
        output_file = os.path.join(directory, final_name + ".mp4")
        
        # Calculate target bitrate based on left file (same as tool_split_combine)
        original_bitrate = get_video_bitrate(left_file, log_callback)
        if not original_bitrate:
            target_bitrate = "12M"
            max_rate = "15M"
            buf_size = "24M"
        else:
            target_kbps = int(original_bitrate / 1000 * 2.2)  # combine needs double bitrate
            target_bitrate = f"{target_kbps}k"
            max_rate = f"{int(target_kbps * 1.2)}k"
            buf_size = f"{int(target_kbps * 2)}k"
        
        if log_callback: log_callback(f"Target Bitrate: {target_bitrate}, Max: {max_rate}")
        
        cmd = [
            "ffmpeg",
            "-hwaccel", "cuda", "-c:v", "hevc_cuvid", "-i", left_file,
            "-hwaccel", "cuda", "-c:v", "hevc_cuvid", "-i", right_file,
            "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]",
            "-map", "[v]", "-map", "0:a?",
            "-c:a", "copy",
            "-c:v", "hevc_nvenc", "-preset", "p7", 
            "-rc", "vbr",
            "-b:v", target_bitrate,
            "-maxrate:v", max_rate,
            "-bufsize:v", buf_size,
            "-shortest", output_file, "-y"
        ]
        run_process(cmd, log_callback, process_callback)
        return output_file
    except Exception as e:
        if log_callback: log_callback(f"Error merging: {e}")
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
