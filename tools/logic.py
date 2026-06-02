import os
import subprocess
import shutil
import re
import json
import bisect
import sys
import tempfile

try:
    from utils.ffmpeg_checker import get_startupinfo, get_video_bitrate
except ImportError:
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from utils.ffmpeg_checker import get_startupinfo, get_video_bitrate

def check_ffmpeg():
    """Check if both ffmpeg and ffprobe are available in the system path."""
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

def has_subtitle_stream(video_path):
    """Check if a video file already has embedded subtitle streams.
    Returns True if at least one subtitle stream is found, False otherwise."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-select_streams", "s",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, startupinfo=get_startupinfo())
        data = json.loads(result.stdout)
        return len(data.get("streams", [])) > 0
    except Exception as e:
        print(f"Error checking subtitle streams: {e}")
        return False

def parse_time_to_seconds(time_str):
    """
    Parse time string (HH:MM:SS, MM:SS, or SS) to seconds.
    Returns float seconds or None if invalid (including negatives or out-of-range components).
    """
    try:
        parts = time_str.strip().split(':')
        if len(parts) == 3: # HH:MM:SS
            h, m, s = map(float, parts)
            if h < 0 or m < 0 or s < 0 or m >= 60 or s >= 60:
                return None
            return h * 3600 + m * 60 + s
        elif len(parts) == 2: # MM:SS
            m, s = map(float, parts)
            if m < 0 or s < 0 or s >= 60:
                return None
            return m * 60 + s
        elif len(parts) == 1: # SS
            v = float(parts[0])
            return v if v >= 0 else None
        else:
            return None
    except ValueError:
        return None

def get_video_duration(file_path):
    """Get video duration in seconds using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path,
        ]
        output = subprocess.check_output(cmd, startupinfo=get_startupinfo(), text=True)
        return float(output.strip())
    except Exception as e:
        print(f"Error getting duration: {e}")
        return None

def get_video_codec(file_path):
    """Get video codec name using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name",
            "-of", "csv=p=0",
            file_path,
        ]
        output = subprocess.check_output(cmd, startupinfo=get_startupinfo(), text=True)
        return output.strip()
    except Exception as e:
        print(f"Error getting codec: {e}")
        return "unknown"



def get_video_resolution(file_path):
    """Get video resolution (width, height) using ffprobe.
    Always returns a 2-tuple; (None, None) on failure to keep tuple-unpacking safe."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            file_path,
        ]
        output = subprocess.check_output(cmd, startupinfo=get_startupinfo(), text=True)
        width, height = map(int, output.strip().split(','))
        return width, height
    except Exception as e:
        print(f"Error getting resolution: {e}")
        return None, None

def get_video_keyframes(file_path):
    """Get list of keyframe timestamps (seconds) using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_packets",
            "-show_entries", "packet=pts_time,flags",
            "-of", "json",
            file_path,
        ]
        output = subprocess.check_output(cmd, startupinfo=get_startupinfo(), text=True)
        data = json.loads(output)

        keyframes = []
        for packet in data.get('packets', []):
            if 'K' in packet.get('flags', ''):
                keyframes.append(float(packet['pts_time']))

        keyframes.sort()
        return keyframes
    except Exception as e:
        print(f"Error getting keyframes: {e}")
        return None

def run_process(cmd, log_callback, process_callback=None):
    if log_callback:
        log_callback(f"Executing: {' '.join(cmd)}")
    else:
        print(f"Executing: {' '.join(cmd)}")
        
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, errors='replace', startupinfo=get_startupinfo())
    if process_callback: process_callback(process)

    try:
        for line in process.stdout:
            if log_callback: log_callback(line.strip())
    finally:
        try:
            if process.stdout:
                process.stdout.close()
        except Exception:
            pass
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

def extract_screenshot(video_path, timestamp, output_path, log_callback=print):
    """
    Extract a screenshot from the video at the given timestamp.
    """
    if not check_ffmpeg():
        log_callback("[!] Error: ffmpeg or ffprobe not found. Please refer to '说明.txt' or 'readme.txt' for installation steps.")
        return False

    log_callback(f"Extracting screenshot from '{video_path}' at {timestamp}...")
    
    # ffmpeg -hide_banner -loglevel error -ss %TIMESTAMP% -i "%INPUT_FILE%" -frames:v 1 -q:v 2 "%OUTPUT_FILE%" -y
    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error',
        '-ss', str(timestamp),
        '-i', video_path,
        '-frames:v', '1',
        '-q:v', '2', # High quality
        output_path,
        '-y'
    ]
    
    try:
        run_process(cmd, log_callback)
        log_callback(f"Success: Screenshot saved to '{output_path}'")
        return True
    except Exception as e:
        log_callback(f"Exception: {e}")
        return False

def extract_frame_image(video_path, timestamp):
    import av

    target_seconds = parse_time_to_seconds(str(timestamp))
    if target_seconds is None:
        raise ValueError(f"Invalid timestamp: {timestamp}")
    container = av.open(video_path)
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

def patch_video(main_video, patch_video, start_time_str, output_path, keep_original_bitrate=False, log_callback=print, process_callback=None):
    """
    Patch the main video with the patch video starting at start_time.
    keep_original_bitrate: if True, use original video bitrate for output
    """
    if not check_ffmpeg():
        log_callback("[!] Error: ffmpeg or ffprobe not found. Please refer to '说明.txt' or 'readme.txt' for installation steps.")
        return False

    start_seconds = parse_time_to_seconds(start_time_str)
    if start_seconds is None:
        log_callback("Error: Invalid start time format.")
        return False

    log_callback(f"Analyzing patch video: {patch_video}")
    patch_duration = get_video_duration(patch_video)
    if patch_duration is None:
        log_callback("Error: Could not determine patch video duration.")
        return False
    
    resume_point = start_seconds + patch_duration
    log_callback(f"Patch Duration: {patch_duration:.2f}s")
    log_callback(f"Resume Point: {resume_point:.2f}s")

    log_callback(f"Analyzing main video: {main_video}")
    codec = get_video_codec(main_video)
    log_callback(f"Main Video Codec: {codec}")

    # Determine decoder options based on codec (mimicking batch script)
    decoder_opt = []
    if codec == 'h264':
        decoder_opt = ['-hwaccel', 'cuda', '-c:v', 'h264_cuvid']
    elif codec == 'hevc':
        decoder_opt = ['-hwaccel', 'cuda', '-c:v', 'hevc_cuvid']
    
    # Construct complex filter
    # [0:v]trim=0:START,setpts=PTS-STARTPTS[v1];
    # [1:v]setpts=PTS-STARTPTS[v2];
    # [0:v]trim=RESUME:,setpts=PTS-STARTPTS[v3];
    # [v1][v2][v3]concat=n=3:v=1:a=0[outv];
    # ... audio ...
    
    filter_complex = (
        f"[0:v]trim=0:{start_seconds},setpts=PTS-STARTPTS[v1];"
        f"[1:v]setpts=PTS-STARTPTS[v2];"
        f"[0:v]trim={resume_point},setpts=PTS-STARTPTS[v3];"
        f"[v1][v2][v3]concat=n=3:v=1:a=0[outv];"
        f"[0:a]atrim=0:{start_seconds},asetpts=PTS-STARTPTS[a1];"
        f"[1:a]asetpts=PTS-STARTPTS[a2];"
        f"[0:a]atrim={resume_point},asetpts=PTS-STARTPTS[a3];"
        f"[a1][a2][a3]concat=n=3:v=0:a=1[outa]"
    )

    cmd = ["ffmpeg","-hide_banner", "-loglevel", "error","-stats"] + decoder_opt + [
        '-i', main_video,
        '-hwaccel', 'cuda'    ]
    
    cmd.extend(['-hwaccel', 'cuda', '-c:v', 'hevc_cuvid', '-i', patch_video])
    
    cmd.extend([
        '-filter_complex', filter_complex,
        '-map', '[outv]',
        '-map', '[outa]'
    ])
    
    if keep_original_bitrate:
        # Use original bitrate for output
        original_bitrate = get_video_bitrate(main_video, log_callback)
        if original_bitrate:
            target_kbps = int(original_bitrate / 1000 * 1.2)
            target_bitrate = f"{target_kbps}k"
            max_rate = f"{int(target_kbps * 1.2)}k"
            buf_size = f"{int(target_kbps * 2)}k"
            cmd.extend([
                '-c:v', 'hevc_nvenc',
                '-preset', 'p7',
                '-rc', 'vbr',
                '-b:v', target_bitrate,
                '-maxrate:v', max_rate,
                '-bufsize:v', buf_size
            ])
        else:
            cmd.extend([
                '-c:v', 'hevc_nvenc',
                '-preset', 'p7',
                '-cq', '18'
            ])
    else:
        cmd.extend([
            '-c:v', 'hevc_nvenc',
            '-preset', 'p7',
            '-cq', '18'
        ])
    
    cmd.extend([
        '-c:a', 'aac',
        '-b:a', '320k',
        '-y', output_path
    ])

    log_callback(f"Executing FFmpeg command...")
    # log_callback(" ".join(cmd)) # Optional: print command for debug

    try:
        run_process(cmd, log_callback, process_callback)
        log_callback(f"Success: Patched video saved to '{output_path}'")
        return True

    except Exception as e:
        log_callback(f"Exception: {e}")
        return False

def merge_files(file_paths, output_path, log_callback=print, process_callback=None):
    """
    Merge multiple video files into one using FFmpeg concat demuxer.
    """
    if not check_ffmpeg():
        log_callback("[!] Error: ffmpeg or ffprobe not found. Please refer to '说明.txt' or 'readme.txt' for installation steps.")
        return False

    if len(file_paths) < 2:
        log_callback("Error: Need at least 2 files to merge.")
        return False

    # Create concat list file (unique name to avoid clobber when multiple merges run concurrently)
    out_dir = os.path.dirname(output_path) or "."
    try:
        fd, concat_list_path = tempfile.mkstemp(prefix="concat_list_", suffix=".txt", dir=out_dir)
        os.close(fd)
    except Exception as e:
        log_callback(f"Error creating concat list temp file: {e}")
        return False
    try:
        with open(concat_list_path, 'w', encoding='utf-8') as f:
            for path in file_paths:
                # ffmpeg concat demuxer treats '\' as escape char; use forward slashes on Windows
                safe_path = path.replace("\\", "/")
                # Escape single quotes for ffmpeg concat file
                safe_path = safe_path.replace("'", "'\\''")
                f.write(f"file '{safe_path}'\n")
    except Exception as e:
        log_callback(f"Error creating concat list: {e}")
        return False

    log_callback(f"Created concat list at: {concat_list_path}")
    
    # ffmpeg -f concat -safe 0 -i concat_list.txt -c copy output.mp4 -y
    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-stats',
        '-f', 'concat',
        '-safe', '0',
        '-i', concat_list_path,
        '-c', 'copy',
        '-y', output_path
    ]

    log_callback(f"Merging {len(file_paths)} files...")
    
    try:
        run_process(cmd, log_callback, process_callback)
        log_callback(f"Success: Merged video saved to '{output_path}'")
        return True
    except Exception as e:
        log_callback(f"Exception: {e}")
        return False
    finally:
        # Cleanup
        if os.path.exists(concat_list_path):
            try:
                os.remove(concat_list_path)
            except:
                pass

def quick_safe_cut(input_file, cut_points, log_callback=print, process_callback=None):
    """
    Split video into segments based on cut_points.
    Segments: Start->P1, P1->P2, ..., Pn->End
    """
    if not check_ffmpeg():
        log_callback("[!] Error: ffmpeg or ffprobe not found. Please refer to '说明.txt' or 'readme.txt' for installation steps.")
        return False

    # Parse points
    points_sec = []
    for p in cut_points:
        s = parse_time_to_seconds(p)
        if s is not None:
            points_sec.append(s)
    
    points_sec.sort()

    # Smart Keyframe Adjustment
    log_callback("Analyzing keyframes for safe cutting...")
    keyframes = get_video_keyframes(input_file)
    
    adjusted_points = []
    if keyframes:
        for p in points_sec:
            # Find nearest keyframe
            idx = bisect.bisect_left(keyframes, p)
            
            # Check left and right neighbors
            candidates = []
            if idx < len(keyframes):
                candidates.append(keyframes[idx])
            if idx > 0:
                candidates.append(keyframes[idx-1])
            
            if not candidates:
                adjusted_points.append(p)
                continue
                
            # Pick closest
            best_k = min(candidates, key=lambda k: abs(k - p))
            
            if abs(best_k - p) > 0.01:
                log_callback(f"Adjusted cut point: {p:.2f}s -> {best_k:.2f}s (Keyframe)")
            else:
                log_callback(f"Cut point {p:.2f}s is already on keyframe.")
                
            adjusted_points.append(best_k)
    else:
        log_callback("Warning: Could not detect keyframes. Using original time points (might cause artifacts).")
        adjusted_points = points_sec
    
    # Remove duplicates and sort again just in case
    adjusted_points = sorted(list(set(adjusted_points)))

    # Add Start (0) and End (None/Duration) logic implicitly by segments
    # Segments:
    # 1. 0 -> P1
    # 2. P1 -> P2
    # ...
    # N. Pn -> End
    
    segments = []
    current_start = 0.0
    
    for p in adjusted_points:
        segments.append((current_start, p))
        current_start = p
    
    # Last segment: Pn -> End
    segments.append((current_start, None))
    
    directory = os.path.dirname(input_file)
    filename = os.path.splitext(os.path.basename(input_file))[0]
    ext = os.path.splitext(input_file)[1]
    
    log_callback(f"Splitting '{input_file}' into {len(segments)} segments...")
    
    for i, (start, end) in enumerate(segments):
        idx = i + 1
        output_file = os.path.join(directory, f"{filename}_{idx}{ext}")
                
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-stats']
        
        if start > 0:
            cmd.extend(['-ss', str(start)])
        
        if end is not None:
            cmd.extend(['-to', str(end)])
            
        cmd.extend(['-i', input_file])
        cmd.extend(['-c', 'copy'])
        cmd.extend(['-avoid_negative_ts', 'make_zero'])
        cmd.extend(['-y', output_file])
        
        log_callback(f"Segment {idx}: {start}s -> {end if end else 'End'}")
        
        try:
            run_process(cmd, log_callback, process_callback)
        except Exception as e:
            log_callback(f"Error processing segment {idx}: {e}")
            return False
            
    return True



def batch_extract_keyframes(video_path, output_dir, start_time=None, end_time=None, 
                            eye_mode='full', log_callback=print, process_callback=None):
    """
    Extract keyframes from video.
    eye_mode: 'full', 'left', 'right'
    """
    if not check_ffmpeg():
        log_callback("[!] Error: ffmpeg or ffprobe not found. Please refer to '说明.txt' or 'readme.txt' for installation steps.")
        return False

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Get Geometry for crop
    width, height = get_video_resolution(video_path)
    if not width:
        log_callback("Failed to get video resolution.")
        return False
        
    # Crop Logic
    filters = []
    if eye_mode == 'left':
        filters.append(f"crop={width//2}:{height}:0:0")
    elif eye_mode == 'right':
        filters.append(f"crop={width//2}:{height}:{width//2}:0")
    
    filter_str = ",".join(filters)
    
    # Time logic
    start_sec = parse_time_to_seconds(start_time) if start_time else 0
    end_sec = parse_time_to_seconds(end_time) if end_time else None
    
    # Calculate duration if end is set
    duration_arg = []
    if end_sec:
        duration = end_sec - start_sec
        if duration > 0:
            duration_arg = ['-t', str(duration)]
            
    # Output pattern
    basename = os.path.splitext(os.path.basename(video_path))[0]
    output_pattern = os.path.join(output_dir, f"{basename}_%06d.jpg")
    
    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-stats',
        '-skip_frame', 'nokey',  # Decode ONLY keyframes
        '-ss', str(start_sec),
        '-i', video_path
    ]
    
    if duration_arg:
        cmd.extend(duration_arg)
        
    if filter_str:
        cmd.extend(['-vf', filter_str])
        
    cmd.extend([
        '-vsync', 'vfr', # Metric for dropping/duping
        '-q:v', '2',     # High quality JPEG
        '-y', output_pattern
    ])
    
    log_callback(f"Extracting keyframes to {output_dir}...")
    
    try:
        run_process(cmd, log_callback, process_callback)
        log_callback(f"Success. Keyframes saved.")
        return True
    except Exception as e:
        log_callback(f"Error: {e}")
        return False

def batch_add_srt(base_dir, search_subdirs=True, replace_original=False, auto_load_srt=True, skip_if_has_sub=False, log_callback=lambda x: None, process_callback=None):
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
                if (file.lower().endswith(".mp4") or file.lower().endswith(".mkv")) and not file.endswith("_srt.mkv"):
                    tasks.append((root, file))
    else:
        try:
            files = os.listdir(base_dir)
            for file in files:
                if (file.lower().endswith(".mp4") or file.lower().endswith(".mkv")) and not file.endswith("_srt.mkv") and os.path.isfile(os.path.join(base_dir, file)):
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
        
        # Look for subtitle file: ass takes priority over srt
        sub_file = None
        try:
            dir_files = os.listdir(root)
            # First pass: try .ass
            for f in dir_files:
                if f.lower() == (file_name_no_ext + ".ass").lower() and os.path.isfile(os.path.join(root, f)):
                    sub_file = os.path.join(root, f)
                    break
            # Second pass: fall back to .srt
            if sub_file is None:
                for f in dir_files:
                    if f.lower() == (file_name_no_ext + ".srt").lower() and os.path.isfile(os.path.join(root, f)):
                        sub_file = os.path.join(root, f)
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
                "-map", "0:a",
                "-map", "1:s",
                "-c", "copy",
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

