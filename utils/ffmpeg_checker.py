import subprocess
import re
import sys
import locale

from utils import app_config, i18n

# --- i18n Setup ---


def get_text(key):
    return i18n.translate('ffmpeg_checker', key)

def get_startupinfo():
    if sys.platform.startswith('win'):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0 # subprocess.SW_HIDE
        return startupinfo
    return None

def check_ffmpeg_version():
    """
    Check if FFmpeg version is too old.
    Returns (is_ok: bool, version: str, prompt_message: str)
    """
    try:
        cmd = ["ffmpeg", "-version"]
        result = subprocess.run(cmd, capture_output=True, text=True, startupinfo=get_startupinfo())
        if result.returncode != 0:
            return False, None, get_text('err_run_version')
        
        output = result.stdout.strip()
        first_line = output.split('\n')[0]
        
        match_std = re.search(r'version\s+(\d+)\.(\d+)', first_line, re.IGNORECASE)
        match_date = re.search(r'version\s+(\d{4})-\d{2}-\d{2}', first_line, re.IGNORECASE)
        match_n = re.search(r'version\s+N-', first_line, re.IGNORECASE)
        
        if match_std:
            major = int(match_std.group(1))
            minor = int(match_std.group(2))
            version_str = f"{major}.{minor}"
            
            # Assume < 6.0 is low
            if major < 6:
                return False, version_str, get_text('warn_version_low').format(version=version_str)
            else:
                return True, version_str, None
        elif match_date:
            year = int(match_date.group(1))
            version_str = f"git-{year}"
            
            if year < 2023: # FFmpeg 6.0 was released in early 2023
                return False, version_str, get_text('warn_version_low').format(version=version_str)
            else:
                return True, version_str, None
        elif match_n:
            # Usually N- builds are recent master builds
            return True, "N-build", None
        else:
            return True, "Unknown", get_text('warn_unknown_ver')
            
    except FileNotFoundError:
        return False, None, get_text('err_not_found')
    except Exception as e:
        return False, None, f"Error checking ffmpeg version: {e}"

def handle_ffmpeg_error(cmd, error_msg, log_callback):
    """
    Analyze ffmpeg error and print a friendly version prompt.
    """
    cmd_str = ' '.join(cmd) if isinstance(cmd, (list, tuple)) else str(cmd)
    
    if "ffmpeg" not in cmd_str.lower() and "ffprobe" not in cmd_str.lower():
        # Not ffmpeg related, just do nothing
        return
        
    is_ok, version, prompt = check_ffmpeg_version()
    
    if log_callback:
        log_callback(f"{get_text('err_prefix')} {error_msg}")
        log_callback("\n" + "="*45)
        if not is_ok and prompt:
            log_callback(f"{get_text('tip_header')} {prompt}")
        elif is_ok and prompt:
            log_callback(f"{get_text('tip_header')} {prompt}")
        else:
            log_callback(f"{get_text('tip_header')} {get_text('tip_generic')}")
        log_callback("="*45 + "\n")

def get_video_bitrate(input_file, log_callback=None):
    try:
        import json
        import traceback
        
        def safe_decode(byte_data):
            encodings = ['utf-8', 'gbk', locale.getpreferredencoding(), 'utf-16', 'utf-8-sig']
            seen = set()
            candidate_encodings = []
            for enc in encodings:
                if enc and enc.lower() not in seen:
                    candidate_encodings.append(enc.lower())
                    seen.add(enc.lower())
            for enc in candidate_encodings:
                try:
                    return byte_data.decode(enc), enc
                except UnicodeDecodeError:
                    continue
            return byte_data.decode('utf-8', errors='replace'), 'utf-8-replace'

        cmd = [
            'ffprobe', '-v', 'error', '-print_format', 'json', 
            '-show_format', '-show_streams', input_file
        ]
        
        if log_callback:
            log_callback(f"[Diagnostics] Running: {' '.join(cmd)}")
            
        result = subprocess.run(cmd, capture_output=True, startupinfo=get_startupinfo())
        
        stdout_str = ""
        used_enc = "none"
        if result.stdout:
            stdout_str, used_enc = safe_decode(result.stdout)
            
        stderr_str = ""
        if result.stderr:
            stderr_str, _ = safe_decode(result.stderr)
            
        if result.returncode != 0:
            msg = f"[Diagnostics] ffprobe failed with code {result.returncode}"
            if stderr_str:
                msg += f"\nStderr: {stderr_str.strip()}"
            if stdout_str:
                msg += f"\nStdout: {stdout_str.strip()}"
            if log_callback:
                log_callback(msg)
            else:
                print(msg)
            return None
            
        if not stdout_str.strip():
            if log_callback:
                log_callback("[Diagnostics] ffprobe output is empty.")
            return None
            
        data = json.loads(stdout_str)
        
        # 1. Prefer extracting bitrate from the video stream and validate that it is reasonable.
        bitrate = None
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                raw_br = stream.get('bit_rate')
                if raw_br:
                    try:
                        val = int(raw_br)
                        if val > 1000:
                            bitrate = val
                            break
                        else:
                            if log_callback:
                                log_callback(f"[Warning] 检测到视频流比特率异常低: {val} bps，将尝试使用容器格式比特率。")
                    except ValueError:
                        pass
        
        # 2. If the video stream has no valid bitrate, fall back to the container format bitrate and validate it.
        if bitrate is None:
            raw_fmt_br = data.get('format', {}).get('bit_rate')
            if raw_fmt_br:
                try:
                    val = int(raw_fmt_br)
                    if val > 1000:
                        bitrate = val
                    else:
                        if log_callback:
                            log_callback(f"[Warning] 检测到容器格式比特率亦异常低: {val} bps。")
                except ValueError:
                    pass
                    
        if bitrate and bitrate > 1000:
            if log_callback:
                log_callback(f"[Diagnostics] Bitrate detected: {bitrate} bps (decoded via {used_enc})")
            return bitrate
        else:
            if log_callback:
                log_callback(f"[Diagnostics] Valid bitrate (>1000 bps) not found in JSON.\nOutput: {stdout_str.strip()}")
            return None
            
    except Exception as e:
        import traceback
        err_stack = traceback.format_exc()
        msg = f"[Diagnostics] Error getting video bitrate for '{input_file}': {e}\n{err_stack}"
        if log_callback:
            log_callback(msg)
        else:
            print(msg)
        return None
