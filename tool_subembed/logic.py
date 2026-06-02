import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile


DEFAULT_PREVIEW_TEXT_LINES = ["Test Subtitle Test Subtitle"]
IPD_METERS = 0.063
_TEMP_ASS_FILES = []


def calculate_parallax_px(video_width, distance_m):
    eye_width = int(video_width // 2)
    distance_m = max(0.1, float(distance_m))
    angle_rad = 2 * math.atan(IPD_METERS / (2 * distance_m))
    return -int(round((angle_rad / math.pi) * eye_width))


def transparency_to_alpha_factor(transparency_percent):
    value = max(0.0, min(70.0, float(transparency_percent)))
    return 1.0 - (value / 100.0)


def get_startupinfo():
    if sys.platform.startswith("win"):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        return startupinfo
    return None


def check_dependencies():
    return [tool for tool in ("ffmpeg", "ffprobe") if not shutil.which(tool)]


def run_process(cmd, log_callback=None, process_callback=None):
    if log_callback:
        log_callback("Executing: " + " ".join(cmd))
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        errors="replace",
        startupinfo=get_startupinfo(),
    )
    if process_callback:
        process_callback(process)
    for line in process.stdout:
        if log_callback:
            log_callback(line.strip())
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
            if log_callback:
                log_callback(f"Checker error: {e}")
        raise RuntimeError(err_msg)


def get_video_info(file_path):
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        file_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, startupinfo=get_startupinfo())
    data = json.loads(result.stdout)
    video_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if not video_stream:
        return None
    bitrate = int(video_stream.get("bit_rate", 0) or data.get("format", {}).get("bit_rate", 0) or 0)
    return {
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "duration": float(data.get("format", {}).get("duration", 0) or 0),
        "codec": video_stream.get("codec_name", ""),
        "bitrate": bitrate,
    }


def time_to_sec(value):
    if value in (None, ""):
        return 0.0
    parts = [float(p) for p in str(value).split(":")]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError("Invalid time format")


def sec_to_ass_time(seconds):
    seconds = max(0, float(seconds))
    total = int(seconds)
    centis = int(round((seconds - total) * 100))
    if centis >= 100:
        total += 1
        centis = 0
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}.{centis:02d}"


def ass_time_to_sec(value):
    main, _, frac = value.strip().partition(".")
    parts = [int(p) for p in main.split(":")]
    if len(parts) != 3:
        raise ValueError(f"Invalid ASS time: {value}")
    centis = int((frac + "00")[:2]) if frac else 0
    return parts[0] * 3600 + parts[1] * 60 + parts[2] + centis / 100.0


def _filter_path(path):
    normalized = os.path.abspath(path).replace("\\", "/")
    return normalized.replace(":", "\\:").replace("'", "\\'")


def _decoder_options(codec):
    if codec == "h264":
        return ["-hwaccel", "cuda", "-c:v", "h264_cuvid"]
    if codec == "hevc":
        return ["-hwaccel", "cuda", "-c:v", "hevc_cuvid"]
    return []


def _read_text_best_effort(path):
    for encoding in ("utf-8-sig", "utf-8", "cp932", "gbk"):
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _write_temp_ass(lines):
    tmp = tempfile.NamedTemporaryFile("w", suffix=".ass", delete=False, encoding="utf-8")
    try:
        tmp.write("\n".join(lines) + "\n")
        return tmp.name
    finally:
        tmp.close()


def get_config_dir():
    if getattr(sys, "frozen", False):
        return os.path.join(os.path.dirname(sys.executable), "config")
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")


def _srt_time_to_ass(srt_time):
    srt_time = srt_time.strip().replace(",", ".")
    m = re.match(r"(\d+):(\d{2}):(\d{2})\.(\d+)", srt_time)
    if not m:
        raise ValueError(f"Could not parse timestamp: {srt_time!r}")
    h, mi, s, ms_str = m.groups()
    cs = ms_str[:2].ljust(2, "0")
    return f"{int(h)}:{mi}:{s}.{cs}"


def _is_japanese(text):
    return any("\u3040" <= ch <= "\u30ff" for ch in text)


def _parse_srt_blocks(srt_path):
    text = _read_text_best_effort(srt_path)
    blocks = []
    for raw in re.split(r"\n\s*\n", text.strip()):
        lines = [line.rstrip() for line in raw.strip().splitlines()]
        time_line_idx = next((i for i, line in enumerate(lines) if "-->" in line), -1)
        if time_line_idx < 0:
            continue
        m = re.match(r"(.+?)\s*-->\s*(.+)", lines[time_line_idx])
        if not m:
            continue
        try:
            start = _srt_time_to_ass(m.group(1))
            end = _srt_time_to_ass(m.group(2))
        except ValueError:
            continue
        text_lines = [line for line in lines[time_line_idx + 1:] if line.strip()]
        if text_lines:
            blocks.append({"start": start, "end": end, "lines": text_lines})
    return blocks


def _load_subtitle_template():
    template_path = os.path.join(get_config_dir(), "subtitle_ass_templates.txt")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Cannot find ASS template: {template_path}")
    template = _read_text_best_effort(template_path)
    return template.split("[Events]")[0] + "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"


def _create_ass_from_srt(srt_path, width=4096, height=4096, alignment=5):
    blocks = _parse_srt_blocks(srt_path)
    if not blocks:
        raise RuntimeError(f"No valid SRT blocks found: {srt_path}")

    scale = ((width * height) / (1280 * 720)) ** 0.5
    cn_size = round(42 * scale)
    jp_size = round(30 * scale)
    marginv = round(32 * scale)

    header = _load_subtitle_template().format(
        width=width,
        height=height,
        cn_size=cn_size,
        jp_size=jp_size,
        marginv=marginv,
        alignment=alignment,
        DefaultPrimaryColour="&H005AFF65",
        DefaultOutlineColour="&H00000000",
        SecondaryPrimaryColour="&H00FFFFFF",
        SecondaryOutlineColour="&H00000000",
    )
    lines = [header.rstrip()]
    for block in blocks:
        start = block["start"]
        end = block["end"]
        text_lines = block["lines"]
        if len(text_lines) == 1:
            style = "Default"# "Secondary" if _is_japanese(text_lines[0]) else "Default"
            lines.append(f"Dialogue: 0,{start},{end},{style},,0,0,0,,{text_lines[0]}")
        else:
            lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text_lines[0]}")
            lines.append(f"Dialogue: 0,{start},{end},Secondary,,0,0,0,,{text_lines[1]}")

    output = _write_temp_ass(lines)
    _TEMP_ASS_FILES.append(output)
    return output


def prepare_subtitle_as_ass(subtitle_path, width=4096, height=4096, alignment=5):
    if subtitle_path.lower().endswith(".srt"):
        return _create_ass_from_srt(subtitle_path, width, height, alignment)
    return subtitle_path


def _verticalize_ass_text(text):
    tokens = []
    pending_tags = ""
    index = 0
    while index < len(text):
        if text[index] == "{":
            end = text.find("}", index + 1)
            if end >= 0:
                pending_tags += text[index : end + 1]
                index = end + 1
                continue
        if text.startswith("\\N", index) or text.startswith("\\n", index):
            index += 2
            continue
        if text.startswith("\\h", index):
            char = " "
            index += 2
        else:
            char = text[index]
            index += 1
        if char in "\r\n":
            continue
        tokens.append(pending_tags + char)
        pending_tags = ""
    if pending_tags:
        tokens.append(pending_tags)
    return "{\\fsp0\\q2}" + "\\N".join(tokens) if tokens else text


def _ass_style_indexes(format_fields):
    defaults = {
        "name": 0,
        "fontsize": 2,
        "italic": 8,
        "spacing": 13,
        "alignment": 18,
        "marginl": 19,
        "marginr": 20,
        "marginv": 21,
    }
    for key in defaults:
        if key in format_fields:
            defaults[key] = format_fields.index(key)
    return defaults


def _safe_int(value, fallback=0):
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return fallback


def _vertical_style_margins(style_info):
    default = style_info.get("default", {})
    default_font_size = max(1, default.get("fontsize", 100))
    default_margin = max(default.get("marginl", 0), round(default_font_size * 0.56))
    gap = max(24, round(default_font_size * 0.4))
    return default_margin, default_margin + default_font_size + gap


def _rewrite_style_alignment(line, indexes, alignment):
    prefix, payload = line.split(":", 1)
    parts = payload.split(",")
    if len(parts) <= indexes["alignment"]:
        return line
    parts[indexes["alignment"]] = str(int(alignment))
    return prefix + ":" + ",".join(parts)


def _stabilize_vertical_style_parts(parts, indexes):
    if len(parts) > indexes["spacing"]:
        parts[indexes["spacing"]] = "0"
    if len(parts) > indexes["italic"]:
        parts[indexes["italic"]] = "0"
    if parts:
        last = parts[-1].strip().lower()
        if last in ("\\q0", "\\q1", "\\q2", "\\q3"):
            parts[-1] = "\\q2"
        else:
            parts.append("\\q2")


def _rewrite_vertical_style_line(line, indexes, side, default_margin, secondary_margin, style_name):
    prefix, payload = line.split(":", 1)
    parts = payload.split(",")
    required_index = max(indexes["alignment"], indexes["marginl"], indexes["marginr"])
    if len(parts) <= required_index:
        return line
    _stabilize_vertical_style_parts(parts, indexes)
    is_secondary = style_name == "secondary"
    if side == "right":
        parts[indexes["alignment"]] = "6"
        parts[indexes["marginl"]] = "10"
        parts[indexes["marginr"]] = str(int(secondary_margin if is_secondary else default_margin))
    elif side == "middle":
        parts[indexes["alignment"]] = "5"
        if is_secondary:
            parts[indexes["marginl"]] = str(int(secondary_margin))
            parts[indexes["marginr"]] = str(int(default_margin))
        else:
            parts[indexes["marginl"]] = str(int(default_margin))
            parts[indexes["marginr"]] = str(int(secondary_margin))
    else:
        parts[indexes["alignment"]] = "4"
        parts[indexes["marginl"]] = str(int(secondary_margin if is_secondary else default_margin))
        parts[indexes["marginr"]] = "10"
    return prefix + ":" + ",".join(parts)


def create_layout_ass(source_ass, direction="horizontal_middle"):
    alignment_by_direction = {
        "horizontal_top": 8,
        "horizontal_middle": 5,
        "horizontal_bottom": 2,
        "vertical_left": 4,
        "vertical_middle": 5,
        "vertical_right": 6,
    }
    direction = direction if direction in alignment_by_direction else "horizontal_middle"
    is_vertical = direction.startswith("vertical_")
    vertical_side = direction.rsplit("_", 1)[-1] if is_vertical else None

    content = _read_text_best_effort(source_ass)
    lines = content.splitlines()
    output_lines = []
    in_styles = False
    in_events = False
    style_format_fields = []
    style_indexes = _ass_style_indexes(style_format_fields)
    style_info = {}
    format_fields = []
    text_index = 9
    dialogue_count = 0

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if lower == "[v4+ styles]":
            in_styles = True
            continue
        if stripped.startswith("[") and lower != "[v4+ styles]":
            in_styles = False
            continue
        if in_styles and lower.startswith("format:"):
            style_format_fields = [p.strip().lower() for p in line.split(":", 1)[1].split(",")]
            style_indexes = _ass_style_indexes(style_format_fields)
            continue
        if in_styles and lower.startswith("style:"):
            parts = line.split(":", 1)[1].split(",")
            if len(parts) <= max(style_indexes["name"], style_indexes["fontsize"], style_indexes["marginl"]):
                continue
            style_name = parts[style_indexes["name"]].strip().lower()
            if style_name in ("default", "secondary"):
                style_info[style_name] = {
                    "fontsize": _safe_int(parts[style_indexes["fontsize"]], 100),
                    "marginl": _safe_int(parts[style_indexes["marginl"]], 0),
                }

    default_margin, secondary_margin = _vertical_style_margins(style_info)
    alignment = alignment_by_direction[direction]

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if lower == "[v4+ styles]":
            in_styles = True
            in_events = False
            output_lines.append(line)
            continue
        if stripped.startswith("[") and lower != "[v4+ styles]" and lower != "[events]":
            in_styles = False
            in_events = False
            output_lines.append(line)
            continue
        if in_styles and lower.startswith("format:"):
            style_format_fields = [p.strip().lower() for p in line.split(":", 1)[1].split(",")]
            style_indexes = _ass_style_indexes(style_format_fields)
            output_lines.append(line)
            continue
        if in_styles and lower.startswith("style:"):
            parts = line.split(":", 1)[1].split(",")
            style_name = ""
            if len(parts) > style_indexes["name"]:
                style_name = parts[style_indexes["name"]].strip().lower()
            if is_vertical:
                output_lines.append(
                    _rewrite_vertical_style_line(
                        line,
                        style_indexes,
                        vertical_side,
                        default_margin,
                        secondary_margin,
                        style_name,
                    )
                )
                continue
            else:
                output_lines.append(_rewrite_style_alignment(line, style_indexes, alignment))
                continue
            output_lines.append(line)
            continue
        if lower == "[events]":
            in_styles = False
            in_events = True
            output_lines.append(line)
            continue
        if stripped.startswith("[") and lower != "[events]":
            in_events = False
            output_lines.append(line)
            continue
        if in_events and lower.startswith("format:"):
            format_fields = [p.strip().lower() for p in line.split(":", 1)[1].split(",")]
            text_index = format_fields.index("text") if "text" in format_fields else 9
            output_lines.append(line)
            continue
        if in_events and lower.startswith("dialogue:"):
            prefix, payload = line.split(":", 1)
            max_splits = max(len(format_fields) - 1, 9) if format_fields else 9
            parts = payload.split(",", max_splits)
            if is_vertical and len(parts) > text_index:
                parts[text_index] = _verticalize_ass_text(parts[text_index])
                output_lines.append(prefix + ":" + ",".join(parts))
                dialogue_count += 1
                continue
            dialogue_count += 1
        output_lines.append(line)

    if dialogue_count == 0:
        output_lines.append("Dialogue: 0,0:00:00.00,0:00:10.00,Default,,0,0,0,,{\\alpha&HFF&}.")

    output = _write_temp_ass(output_lines)
    _TEMP_ASS_FILES.append(output)
    return output


def apply_subtitle_direction(source_ass, subtitle_direction="horizontal"):
    direction = str(subtitle_direction).lower()
    if direction == "horizontal":
        direction = "horizontal_middle"
    if direction == "vertical":
        direction = "vertical_left"
    return create_layout_ass(source_ass, direction)


def _header_with_events_format(source_ass):
    content = _read_text_best_effort(source_ass)
    lines = content.splitlines()
    style_name = "Default"
    events_index = None
    format_line = "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
    kept = []
    in_styles = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.lower() == "[v4+ styles]":
            in_styles = True
        elif stripped.startswith("[") and stripped.lower() != "[v4+ styles]":
            in_styles = False
        if in_styles and stripped.lower().startswith("style:"):
            parts = line.split(":", 1)[1].split(",")
            if parts and parts[0].strip():
                style_name = parts[0].strip()
        if stripped.lower() == "[events]":
            events_index = index
            kept.append(line)
            continue
        if events_index is not None and stripped.lower().startswith("format:"):
            format_line = line
            kept.append(line)
            break
        kept.append(line)

    if events_index is None:
        kept.extend(["", "[Events]", format_line])
    elif not any(line.strip().lower().startswith("format:") for line in kept[events_index + 1 :]):
        kept.append(format_line)
    return kept, style_name, format_line, lines


def _event_field_indexes(format_fields):
    indexes = {
        "start": 1,
        "end": 2,
        "style": 3,
        "text": 9,
    }
    for name in indexes:
        if name in format_fields:
            indexes[name] = format_fields.index(name)
    return indexes


def _parse_dialogue_events(lines):
    in_events = False
    format_fields = []
    indexes = _event_field_indexes(format_fields)
    events = []
    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if lower == "[events]":
            in_events = True
            continue
        if stripped.startswith("[") and lower != "[events]":
            in_events = False
            continue
        if in_events and lower.startswith("format:"):
            format_fields = [p.strip().lower() for p in line.split(":", 1)[1].split(",")]
            indexes = _event_field_indexes(format_fields)
            continue
        if in_events and lower.startswith("dialogue:"):
            _prefix, payload = line.split(":", 1)
            max_splits = max(len(format_fields) - 1, 9) if format_fields else 9
            parts = payload.split(",", max_splits)
            if len(parts) > max(indexes.values()):
                events.append({
                    "prefix": _prefix,
                    "parts": parts,
                    "indexes": indexes,
                    "start": parts[indexes["start"]].strip(),
                    "end": parts[indexes["end"]].strip(),
                    "style": parts[indexes["style"]].strip() or "Default",
                    "text": parts[indexes["text"]].strip(),
                })
    return events


def _preview_dialogue_group(lines):
    events = _parse_dialogue_events(lines)
    if not events:
        return []
    grouped = {}
    for event in events:
        key = (event["start"], event["end"])
        grouped.setdefault(key, []).append(event)
    return max(grouped.values(), key=lambda group: len(group))

def _preview_text_for_event(text_lines, index):
    if isinstance(text_lines, str):
        text_lines = [text_lines]
    text_lines = text_lines or DEFAULT_PREVIEW_TEXT_LINES
    if index < len(text_lines):
        text = text_lines[index]
    else:
        text = text_lines[-1]
    return str(text).replace("\n", " ").replace(",", "，")


def create_preview_ass(source_ass, duration_seconds, text_lines=None):
    kept, style_name, _format_line, lines = _header_with_events_format(source_ass)

    start = sec_to_ass_time(0)
    end = sec_to_ass_time(max(10.0, float(duration_seconds or 0)))
    group = _preview_dialogue_group(lines)
    if not group:
        safe_text = _preview_text_for_event(text_lines, 0)
        kept.append(f"Dialogue: 0,{start},{end},{style_name},,0,0,0,,{safe_text}")
        return _write_temp_ass(kept)

    for index, event in enumerate(group):
        parts = list(event["parts"])
        indexes = event["indexes"]
        parts[indexes["start"]] = start
        parts[indexes["end"]] = end
        parts[indexes["text"]] = _preview_text_for_event(text_lines, index)
        kept.append(event["prefix"] + ":" + ",".join(parts))
    return _write_temp_ass(kept)


def create_shifted_ass_for_segment(source_ass, start_seconds, duration_seconds=None):
    start_seconds = max(0.0, float(start_seconds or 0))
    if start_seconds <= 0:
        return source_ass

    content = _read_text_best_effort(source_ass)
    lines = content.splitlines()
    shifted_lines = []
    in_events = False
    format_fields = []
    start_index = 1
    end_index = 2
    dialogue_count = 0

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if lower == "[events]":
            in_events = True
            shifted_lines.append(line)
            continue
        if stripped.startswith("[") and lower != "[events]":
            in_events = False
            shifted_lines.append(line)
            continue
        if in_events and lower.startswith("format:"):
            format_fields = [p.strip().lower() for p in line.split(":", 1)[1].split(",")]
            if "start" in format_fields:
                start_index = format_fields.index("start")
            if "end" in format_fields:
                end_index = format_fields.index("end")
            shifted_lines.append(line)
            continue
        if in_events and lower.startswith("dialogue:"):
            prefix, payload = line.split(":", 1)
            max_splits = max(len(format_fields) - 1, 9) if format_fields else 9
            parts = payload.split(",", max_splits)
            if len(parts) > max(start_index, end_index):
                try:
                    original_start = ass_time_to_sec(parts[start_index])
                    original_end = ass_time_to_sec(parts[end_index])
                    shifted_start = original_start - start_seconds
                    shifted_end = original_end - start_seconds
                    segment_end = float(duration_seconds) if duration_seconds else None
                    if shifted_end <= 0 or (segment_end is not None and shifted_start >= segment_end):
                        continue
                    parts[start_index] = sec_to_ass_time(max(0, shifted_start))
                    parts[end_index] = sec_to_ass_time(shifted_end)
                    shifted_lines.append(prefix + ":" + ",".join(parts))
                    dialogue_count += 1
                    continue
                except Exception:
                    pass
        shifted_lines.append(line)

    if dialogue_count == 0:
        dummy_end = sec_to_ass_time(float(duration_seconds) if duration_seconds else 10.0)
        shifted_lines.append(f"Dialogue: 0,0:00:00.00,{dummy_end},Default,,0,0,0,,{{\\alpha&HFF&}}.")

    output = _write_temp_ass(shifted_lines)
    _TEMP_ASS_FILES.append(output)
    return output


def cleanup_temp_ass_files():
    while _TEMP_ASS_FILES:
        path = _TEMP_ASS_FILES.pop()
        try:
            os.unlink(path)
        except OSError:
            pass


def get_left_eye_frame_image(input_file, time_seconds):
    import av

    target_seconds = max(0.0, float(time_seconds))
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
            frame_seconds = float(frame.pts * frame.time_base)
            if frame_seconds + 0.05 >= target_seconds:
                selected_frame = frame
                break

        if selected_frame is None:
            raise RuntimeError("Unable to decode a video frame at the selected time")

        image = selected_frame.to_image().convert("RGB")
        return image.crop((0, 0, image.width // 2, image.height))
    finally:
        container.close()


def build_filter_complex(ass_file, width, height, fov, yaw, pitch, transparency_percent, mode, distance_m):
    eye_w = int(width // 2)
    eye_h = int(height)
    patch_size = max(512, int(round(min(eye_w, eye_h) / 2)))
    v360_w = eye_w
    v360_h = eye_h
    ass_path = _filter_path(ass_file)
    alpha_factor = transparency_to_alpha_factor(transparency_percent)
    ffmpeg_yaw = -float(yaw)
    ffmpeg_pitch = -float(pitch)
    base = (
        f"[1:v]ass='{ass_path}':alpha=1,split[rgb_src][alpha_src];"
        f"[alpha_src]alphaextract,lutyuv=y='val*{alpha_factor:.3f}',"
        f"v360=input=flat:output=hequirect:w={v360_w}:h={v360_h}:"
        f"id_fov={float(fov):.3f}:yaw={ffmpeg_yaw:.3f}:pitch={ffmpeg_pitch:.3f}:rorder=rpy[alpha_proj];"
        f"[rgb_src]v360=input=flat:output=hequirect:w={v360_w}:h={v360_h}:"
        f"id_fov={float(fov):.3f}:yaw={ffmpeg_yaw:.3f}:pitch={ffmpeg_pitch:.3f}:rorder=rpy[rgb_proj];"
    )
    if mode == "left":
        overlay = (
            "[rgb_proj][alpha_proj]alphamerge,format=yuva420p[patch];"
            "[0:v]format=yuv420p[main];"
            "[main][patch]overlay=x=0:y=0:eof_action=pass:format=yuv420:alpha=straight[final]"
        )
    elif mode == "right":
        overlay = (
            "[rgb_proj][alpha_proj]alphamerge,format=yuva420p[patch];"
            "[0:v]format=yuv420p[main];"
            f"[main][patch]overlay=x={eye_w}:y=0:eof_action=pass:format=yuv420:alpha=straight[final]"
        )
    else:
        parallax_px = calculate_parallax_px(width, distance_m)
        right_x = eye_w + parallax_px
        overlay = (
            "[rgb_proj][alpha_proj]alphamerge,format=yuva420p,split[patch_l][patch_r];"
            "[0:v]format=yuv420p[main];"
            "[main][patch_l]overlay=x=0:y=0:eof_action=pass:format=yuv420:alpha=straight[left_done];"
            f"[left_done][patch_r]overlay=x={right_x}:y=0:eof_action=pass:format=yuv420:alpha=straight[final]"
        )
    color_src = f"color=c=0x00000000:s={patch_size}x{patch_size}:r=30,format=yuva420p"
    return base + overlay, color_src


def generate_preview(
    input_file,
    ass_file,
    time_seconds,
    fov,
    yaw,
    pitch,
    alpha_factor,
    mode,
    distance_m,
    subtitle_direction,
    output_image,
    preview_text_lines=None,
):
    info = get_video_info(input_file)
    if not info:
        raise RuntimeError("Unable to read video info")
    prepared_ass = prepare_subtitle_as_ass(ass_file, info["width"] // 2, info["height"])
    preview_ass = create_preview_ass(prepared_ass, info.get("duration", 3600), preview_text_lines)
    temp_ass = apply_subtitle_direction(preview_ass, subtitle_direction)
    try:
        filter_complex, color_src = build_filter_complex(
            temp_ass,
            info["width"],
            info["height"],
            fov,
            yaw,
            pitch,
            alpha_factor,
            mode,
            distance_m,
        )
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(time_seconds),
        ]
        cmd.extend(_decoder_options(info.get("codec")))
        cmd.extend(
            [
                "-i",
                input_file,
                "-f",
                "lavfi",
                "-i",
                color_src,
                "-filter_complex",
                filter_complex,
                "-map",
                "[final]",
                "-frames:v",
                "1",
                "-y",
                output_image,
            ]
        )
        subprocess.run(cmd, check=True, startupinfo=get_startupinfo())
        return output_image
    finally:
        try:
            if temp_ass != preview_ass:
                os.unlink(preview_ass)
            else:
                os.unlink(temp_ass)
        except OSError:
            pass
        try:
            os.unlink(temp_ass)
        except OSError:
            pass
        cleanup_temp_ass_files()


def default_output_path(input_file, mode, fov, yaw, pitch):
    directory = os.path.dirname(input_file)
    stem = os.path.splitext(os.path.basename(input_file))[0]
    suffix = f"_subembed_{mode}.mp4"
    return os.path.join(directory, stem + suffix)


def default_2d_output_path(input_file):
    directory = os.path.dirname(input_file)
    stem = os.path.splitext(os.path.basename(input_file))[0]
    return os.path.join(directory, f"{stem}_subembed.mp4")


def build_embed_command(
    input_file,
    ass_file,
    output_file,
    start_time,
    duration_seconds,
    fov,
    yaw,
    pitch,
    alpha_factor,
    mode,
    distance_m,
    subtitle_direction="horizontal",
):
    info = get_video_info(input_file)
    if not info:
        raise RuntimeError("Unable to read video info")
    start_seconds = time_to_sec(start_time)
    prepared_ass = prepare_subtitle_as_ass(ass_file, info["width"] // 2, info["height"])
    shifted_ass = create_shifted_ass_for_segment(prepared_ass, start_seconds, duration_seconds)
    filter_ass_file = apply_subtitle_direction(shifted_ass, subtitle_direction)
    filter_complex, color_src = build_filter_complex(
        filter_ass_file,
        info["width"],
        info["height"],
        fov,
        yaw,
        pitch,
        alpha_factor,
        mode,
        distance_m,
    )
    bitrate = info.get("bitrate", 0)
    if bitrate > 0:
        target_bitrate = str(bitrate)
        max_rate = str(int(bitrate * 1.2))
        buf_size = str(int(bitrate * 2))
    else:
        target_bitrate = "40000000"
        max_rate = "48000000"
        buf_size = "80000000"

    cmd = ["ffmpeg", "-hide_banner"]
    if start_time:
        cmd.extend(["-ss", start_time])
    cmd.extend(_decoder_options(info.get("codec")))
    if duration_seconds:
        cmd.extend(["-t", str(duration_seconds)])
    cmd.extend(
        [
            "-i", input_file,
            "-f", "lavfi",
            "-i", color_src,
            "-filter_complex", filter_complex,
            "-map", "[final]",
            "-map", "0:a?",
            "-c:v", "hevc_nvenc",
            "-preset", "p7",
            "-rc", "vbr",
            "-b:v", target_bitrate,
            "-maxrate", max_rate,
            "-bufsize", buf_size,
            "-c:a", "copy",
            "-y", output_file,
        ]
    )
    return cmd


def run_embed(*args, log_callback=None, process_callback=None):
    try:
        cmd = build_embed_command(*args)
        run_process(cmd, log_callback, process_callback)
    finally:
        cleanup_temp_ass_files()


def build_embed_2d_command(input_file, subtitle_file, output_file, start_time, duration_seconds):
    info = get_video_info(input_file)
    if not info:
        raise RuntimeError("Unable to read video info")
    start_seconds = time_to_sec(start_time)
    prepared_ass = prepare_subtitle_as_ass(subtitle_file, info["width"], info["height"], alignment=2)
    filter_ass_file = create_shifted_ass_for_segment(prepared_ass, start_seconds, duration_seconds)
    ass_path = _filter_path(filter_ass_file)

    bitrate = info.get("bitrate", 0)
    if bitrate > 0:
        target_bitrate = str(bitrate)
        max_rate = str(int(bitrate * 1.2))
        buf_size = str(int(bitrate * 2))
    else:
        target_bitrate = "40000000"
        max_rate = "48000000"
        buf_size = "80000000"

    cmd = ["ffmpeg", "-hide_banner"]
    if start_time:
        cmd.extend(["-ss", start_time])
    cmd.extend(_decoder_options(info.get("codec")))
    if duration_seconds:
        cmd.extend(["-t", str(duration_seconds)])
    cmd.extend(
        [
            "-i", input_file,
            "-vf", f"ass='{ass_path}'",
            "-map", "0:v",
            "-map", "0:a?",
            "-c:v", "hevc_nvenc",
            "-preset", "p7",
            "-rc", "vbr",
            "-b:v", target_bitrate,
            "-maxrate", max_rate,
            "-bufsize", buf_size,
            "-c:a", "copy",
            "-y", output_file,
        ]
    )
    return cmd


def run_embed_2d(*args, log_callback=None, process_callback=None):
    try:
        cmd = build_embed_2d_command(*args)
        run_process(cmd, log_callback, process_callback)
    finally:
        cleanup_temp_ass_files()
