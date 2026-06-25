"""Pre-scan videos with LADA's YOLO mosaic detector and aggregate useful clips."""
from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from gpu_engine import probe


_DETECTOR = None
_DETECTOR_CONFIG = None
_DETECTOR_LOCK = threading.Lock()
_RESIZE_LUT_CACHE = {}


@dataclass
class MosaicSegment:
    seg_id: int
    start_s: float
    end_s: float
    start_s_kf: float
    end_s_kf: float
    x: int
    y: int
    w: int
    h: int
    conf_max: float

    @property
    def duration_s(self) -> float:
        return max(0.0, float(self.end_s) - float(self.start_s))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MosaicSegment":
        return cls(
            seg_id=int(data["seg_id"]),
            start_s=float(data["start_s"]),
            end_s=float(data["end_s"]),
            start_s_kf=float(data.get("start_s_kf", data["start_s"])),
            end_s_kf=float(data.get("end_s_kf", data["end_s"])),
            x=int(data["x"]),
            y=int(data["y"]),
            w=int(data["w"]),
            h=int(data["h"]),
            conf_max=float(data.get("conf_max", 0.0)),
        )


def _cfg(key: str, default):
    try:
        from utils import app_config

        value = app_config.get(key, default)
        return default if value is None else value
    except Exception:
        return default


def _hidden_kwargs() -> dict:
    if sys.platform.startswith("win"):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        return {"startupinfo": startupinfo}
    return {}


def _model_path() -> str:
    name = str(_cfg("pre_extract_detection_model", "lada_vr_mosaic_detection_model_v2_fast.pt") or "").strip()
    if os.path.isabs(name):
        return name
    try:
        from gpu_engine.native_mosaic.models_cfg import models_dir

        return os.path.join(models_dir(), name)
    except Exception:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(root, "models", name)


def _path_identity(path: str | Path) -> dict:
    p = Path(path)
    st = p.stat()
    return {
        "path": str(p.resolve()),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _empty_scan_cache_path(video_path: str | Path, *, mode: str,
                           min_conf: float | None,
                           crop_mode: str = "",
                           to_fisheye: bool = False) -> Path | None:
    if min_conf is None or not bool(_cfg("pre_extract_empty_scan_cache", True)):
        return None
    try:
        source = _path_identity(video_path)
    except OSError:
        return None
    model_path = Path(_model_path())
    try:
        model = _path_identity(model_path)
    except OSError:
        model = {"path": str(model_path), "size": 0, "mtime_ns": 0}
    payload = {
        "format_version": 1,
        "source": source,
        "model": model,
        "mode": str(mode),
        "crop_mode": str(crop_mode or ""),
        "to_fisheye": bool(to_fisheye),
        "min_conf": float(min_conf),
        "sample_stride_s": float(_cfg("pre_extract_sample_stride_s", 0.5) or 0.5),
        "yolo_imgsz": int(_cfg("pre_extract_yolo_imgsz", 2048) or 0),
        "use_mask_boxes": bool(_cfg("pre_extract_use_mask_boxes", True)),
        "detector_box_only": True,
        "detector_box_only_version": 2,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]
    p = Path(video_path)
    return p.with_name(f"{p.stem}.fine_empty_cache") / f"{mode}.{digest}.json"


def _load_empty_scan_cache(path: Path | None, log_callback=None) -> bool:
    if path is None or not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return False
    if not bool(data.get("empty")):
        return False
    if log_callback:
        log_callback(f"[pre-extract] fine empty-scan cache hit: {path}")
    return True


def _write_empty_scan_cache(path: Path | None, log_callback=None) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps({"empty": True}, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        if log_callback:
            log_callback(f"[pre-extract] fine empty-scan cache saved: {path}")
    except Exception as exc:
        if log_callback:
            log_callback(f"[pre-extract] fine empty-scan cache save skipped: {type(exc).__name__}: {exc}")


def _align_down(value: int, align: int) -> int:
    return (max(0, int(value)) // align) * align


def _align_up(value: int, align: int) -> int:
    value = max(0, int(value))
    return ((value + align - 1) // align) * align


def _expanded_rect(boxes: list[tuple[float, float, float, float, float]],
                   frame_w: int, frame_h: int) -> tuple[int, int, int, int, float]:
    boxes = _filter_spatial_outliers(boxes, frame_w, frame_h)
    expand = float(_cfg("pre_extract_rect_expand", 1.5) or 1.5)
    align = max(2, int(_cfg("pre_extract_rect_align", 16) or 16))
    min_px = max(2, int(_cfg("pre_extract_rect_min_px", 512) or 512))

    x1 = max(0.0, min(b[0] for b in boxes))
    y1 = max(0.0, min(b[1] for b in boxes))
    x2 = min(float(frame_w), max(b[2] for b in boxes))
    y2 = min(float(frame_h), max(b[3] for b in boxes))
    conf_max = max(float(b[4]) for b in boxes)

    bw = max(2.0, x2 - x1)
    bh = max(2.0, y2 - y1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    target_w = min(float(frame_w), max(float(min_px), bw * expand))
    target_h = min(float(frame_h), max(float(min_px), bh * expand))

    rx1 = int(round(cx - target_w * 0.5))
    ry1 = int(round(cy - target_h * 0.5))
    rx2 = int(round(cx + target_w * 0.5))
    ry2 = int(round(cy + target_h * 0.5))

    rx1 = max(0, min(rx1, frame_w - 2))
    ry1 = max(0, min(ry1, frame_h - 2))
    rx2 = max(rx1 + 2, min(rx2, frame_w))
    ry2 = max(ry1 + 2, min(ry2, frame_h))

    ax1 = _align_down(rx1, align)
    ay1 = _align_down(ry1, align)
    ax2 = _align_up(rx2, align)
    ay2 = _align_up(ry2, align)
    if ax2 > frame_w:
        ax2 = frame_w if frame_w % align == 0 else _align_down(frame_w, align)
    if ay2 > frame_h:
        ay2 = frame_h if frame_h % align == 0 else _align_down(frame_h, align)
    ax1 = max(0, min(ax1, max(0, ax2 - align)))
    ay1 = max(0, min(ay1, max(0, ay2 - align)))

    # Chroma safety if the source dimensions are not multiples of 16.
    ax1 -= ax1 % 2
    ay1 -= ay1 % 2
    ax2 -= ax2 % 2
    ay2 -= ay2 % 2
    ax2 = max(ax1 + 2, ax2)
    ay2 = max(ay1 + 2, ay2)
    return ax1, ay1, ax2 - ax1, ay2 - ay1, conf_max


def _segments_time_overlap(a: MosaicSegment, b: MosaicSegment) -> bool:
    return max(float(a.start_s), float(b.start_s)) < min(float(a.end_s), float(b.end_s))


def _segments_rect_overlap(a: MosaicSegment, b: MosaicSegment) -> bool:
    ax2 = int(a.x) + int(a.w)
    ay2 = int(a.y) + int(a.h)
    bx2 = int(b.x) + int(b.w)
    by2 = int(b.y) + int(b.h)
    return max(int(a.x), int(b.x)) < min(ax2, bx2) and max(int(a.y), int(b.y)) < min(ay2, by2)


def _merge_two_segments(a: MosaicSegment, b: MosaicSegment, seg_id: int) -> MosaicSegment:
    x1 = min(int(a.x), int(b.x))
    y1 = min(int(a.y), int(b.y))
    x2 = max(int(a.x) + int(a.w), int(b.x) + int(b.w))
    y2 = max(int(a.y) + int(a.h), int(b.y) + int(b.h))
    return MosaicSegment(
        seg_id=seg_id,
        start_s=min(float(a.start_s), float(b.start_s)),
        end_s=max(float(a.end_s), float(b.end_s)),
        start_s_kf=min(float(a.start_s_kf), float(b.start_s_kf)),
        end_s_kf=max(float(a.end_s_kf), float(b.end_s_kf)),
        x=x1,
        y=y1,
        w=max(2, x2 - x1),
        h=max(2, y2 - y1),
        conf_max=max(float(a.conf_max), float(b.conf_max)),
    )


def _merge_overlapping_segments(segments: list[MosaicSegment]) -> list[MosaicSegment]:
    """Merge final rects when both time and space overlap.

    This runs after rect expansion/alignment, because two raw clusters can be
    separate before expansion but overlap in the actual cropped/restored region.
    """
    merged = list(segments)
    changed = True
    while changed:
        changed = False
        out: list[MosaicSegment] = []
        used = [False] * len(merged)
        for i, seg in enumerate(merged):
            if used[i]:
                continue
            current = seg
            used[i] = True
            local_changed = True
            while local_changed:
                local_changed = False
                for j in range(i + 1, len(merged)):
                    if used[j]:
                        continue
                    other = merged[j]
                    if _segments_time_overlap(current, other) and _segments_rect_overlap(current, other):
                        current = _merge_two_segments(current, other, seg_id=len(out))
                        used[j] = True
                        changed = True
                        local_changed = True
            out.append(current)
        merged = [
            _clone_segment_for_merge(seg, idx)
            for idx, seg in enumerate(sorted(out, key=lambda s: (float(s.start_s), float(s.end_s), int(s.y), int(s.x))))
        ]
    return merged


def _clone_segment_for_merge(seg: MosaicSegment, seg_id: int) -> MosaicSegment:
    return MosaicSegment(
        seg_id=seg_id,
        start_s=float(seg.start_s),
        end_s=float(seg.end_s),
        start_s_kf=float(seg.start_s_kf),
        end_s_kf=float(seg.end_s_kf),
        x=int(seg.x),
        y=int(seg.y),
        w=int(seg.w),
        h=int(seg.h),
        conf_max=float(seg.conf_max),
    )


def _spatial_cluster(boxes: list[tuple[float, float, float, float, float]],
                     frame_w: int, frame_h: int) -> list[list[tuple[float, float, float, float, float]]]:
    """Cluster boxes that are spatially close inside one time segment."""
    if not boxes:
        return []
    if len(boxes) <= 1:
        return [boxes]
    gap_ratio = float(_cfg("pre_extract_cluster_gap_ratio", 0.03) or 0.03)
    gap_px = max(20, int(min(frame_w, frame_h) * gap_ratio))
    parent = list(range(len(boxes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(boxes)):
        ax1, ay1, ax2, ay2 = boxes[i][:4]
        for j in range(i + 1, len(boxes)):
            bx1, by1, bx2, by2 = boxes[j][:4]
            if (
                ax2 + gap_px >= bx1
                and bx2 + gap_px >= ax1
                and ay2 + gap_px >= by1
                and by2 + gap_px >= ay1
            ):
                union(i, j)

    clusters: dict[int, list[tuple[float, float, float, float, float]]] = {}
    for idx, box in enumerate(boxes):
        clusters.setdefault(find(idx), []).append(box)
    return list(clusters.values())


def _filter_spatial_clusters(boxes: list[tuple[float, float, float, float, float]],
                             frame_w: int, frame_h: int) -> list[tuple[float, float, float, float, float]]:
    if len(boxes) <= 1 or not bool(_cfg("pre_extract_spatial_cluster_enabled", True)):
        return boxes

    import statistics

    widths = [max(1.0, b[2] - b[0]) for b in boxes]
    heights = [max(1.0, b[3] - b[1]) for b in boxes]
    med_size = max(statistics.median(widths), statistics.median(heights))
    radius_px = float(_cfg("pre_extract_spatial_cluster_radius_px", 0.0) or 0.0)
    if radius_px <= 0.0:
        radius_ratio = float(_cfg("pre_extract_spatial_cluster_radius_ratio", 0.20) or 0.20)
        radius_factor = float(_cfg("pre_extract_spatial_cluster_radius_factor", 3.0) or 3.0)
        radius_px = max(min(frame_w, frame_h) * radius_ratio, med_size * radius_factor)
    radius_sq = radius_px * radius_px

    centers = [((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5) for b in boxes]
    remaining = set(range(len(boxes)))
    clusters: list[list[int]] = []
    while remaining:
        root = remaining.pop()
        cluster = [root]
        stack = [root]
        while stack:
            idx = stack.pop()
            cx, cy = centers[idx]
            linked = []
            for other in remaining:
                ox, oy = centers[other]
                if (cx - ox) ** 2 + (cy - oy) ** 2 <= radius_sq:
                    linked.append(other)
            for other in linked:
                remaining.remove(other)
                cluster.append(other)
                stack.append(other)
        clusters.append(cluster)

    if len(clusters) <= 1:
        return boxes

    scored = []
    for cluster in clusters:
        confs = [max(0.0, float(boxes[i][4])) for i in cluster]
        score = sum(confs)
        max_conf = max(confs) if confs else 0.0
        avg_conf = score / max(1, len(cluster))
        scored.append({
            "indices": cluster,
            "score": score,
            "count": len(cluster),
            "max_conf": max_conf,
            "avg_conf": avg_conf,
        })

    dominant = max(scored, key=lambda c: (c["score"], c["count"], c["max_conf"]))
    dominant_score = max(0.000001, float(dominant["score"]))
    keep_score_ratio = float(_cfg("pre_extract_spatial_cluster_score_ratio", 0.15) or 0.15)
    min_secondary_conf = float(_cfg("pre_extract_spatial_cluster_min_conf", 0.50) or 0.50)
    high_conf = float(_cfg("pre_extract_spatial_cluster_high_conf", 0.70) or 0.70)
    min_secondary_boxes = max(1, int(_cfg("pre_extract_spatial_cluster_min_boxes", 2) or 2))

    keep_indices: set[int] = set(dominant["indices"])
    for cluster in scored:
        if cluster is dominant:
            continue
        score_ratio = float(cluster["score"]) / dominant_score
        if float(cluster["max_conf"]) >= high_conf:
            keep_indices.update(cluster["indices"])
            continue
        if (
            int(cluster["count"]) >= min_secondary_boxes
            and score_ratio >= keep_score_ratio
            and float(cluster["avg_conf"]) >= min_secondary_conf
        ):
            keep_indices.update(cluster["indices"])

    filtered = [box for idx, box in enumerate(boxes) if idx in keep_indices]
    return filtered or boxes


def _filter_spatial_outliers(boxes: list[tuple[float, float, float, float, float]],
                             frame_w: int, frame_h: int) -> list[tuple[float, float, float, float, float]]:
    """Drop obvious detector outliers before taking a segment-level union.

    A single false-positive tall box can otherwise pull a 60s segment rect to
    nearly the full fisheye frame. LADA restores each detection as a scene, while
    pre-extract uses one rect per aggregated time segment, so this needs to be
    more conservative than raw YOLO box union.
    """
    if len(boxes) <= 1:
        return boxes
    hard_filtered = []
    for box in boxes:
        x1, y1, x2, y2, _conf = box
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        if bw <= 1.0 or bh <= 1.0:
            continue
        hard_filtered.append(box)
    if not hard_filtered:
        hard_filtered = boxes
    if len(hard_filtered) <= 3:
        return hard_filtered

    import statistics

    centers_x = [(b[0] + b[2]) * 0.5 for b in hard_filtered]
    centers_y = [(b[1] + b[3]) * 0.5 for b in hard_filtered]
    widths = [max(1.0, b[2] - b[0]) for b in hard_filtered]
    heights = [max(1.0, b[3] - b[1]) for b in hard_filtered]
    med_cx = statistics.median(centers_x)
    med_cy = statistics.median(centers_y)
    med_w = statistics.median(widths)
    med_h = statistics.median(heights)
    center_factor = float(_cfg("pre_extract_outlier_center_factor", 3.0) or 3.0)
    max_center_dist = max(min(frame_w, frame_h) * 0.20, max(med_w, med_h) * center_factor)
    far_box_min_conf = float(_cfg("pre_extract_far_box_min_conf", 0.50) or 0.50)

    filtered = []
    for box in hard_filtered:
        cx = (box[0] + box[2]) * 0.5
        cy = (box[1] + box[3]) * 0.5
        dist = ((cx - med_cx) ** 2 + (cy - med_cy) ** 2) ** 0.5
        if dist > max_center_dist and float(box[4]) < far_box_min_conf:
            continue
        filtered.append(box)
    return filtered or hard_filtered


def _mask_box(result, index: int, raw_box_xyxy: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
    if not bool(_cfg("pre_extract_use_mask_boxes", True)):
        return raw_box_xyxy
    masks_obj = getattr(result, "masks", None)
    if masks_obj is None:
        return raw_box_xyxy
    try:
        from lada.utils.mask_utils import clean_mask
        from lada.utils.ultralytics_utils import convert_yolo_box, convert_yolo_mask_tensor

        yolo_mask = result.masks[index]
        yolo_box = result.boxes[index]
        box_tlbr = convert_yolo_box(yolo_box, result.orig_shape)
        mask = convert_yolo_mask_tensor(yolo_mask, result.orig_shape).cpu().numpy()
        _cleaned_mask, cleaned_box = clean_mask(mask, box_tlbr)
        t, l, b, r = cleaned_box
        if r <= l or b <= t:
            return None
        return float(l), float(t), float(r), float(b)
    except Exception:
        return raw_box_xyxy


def _box_to_list(box) -> list[float] | None:
    if box is None:
        return None
    return [round(float(v), 3) for v in box]


def _box_limit_reason(box: tuple[float, float, float, float], frame_w: int, frame_h: int) -> str | None:
    x1, y1, x2, y2 = box
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw <= 1.0 or bh <= 1.0:
        return "empty_box"
    return None


def _box_passes_frame_limits(box: tuple[float, float, float, float], frame_w: int, frame_h: int) -> bool:
    return _box_limit_reason(box, frame_w, frame_h) is None


def _extract_boxes(result, min_conf: float | None = None) -> list[tuple[float, float, float, float, float]]:
    boxes, _debug = _extract_boxes_with_debug(result, min_conf=min_conf)
    return boxes


def _extract_boxes_with_debug(result, min_conf: float | None = None) -> tuple[list[tuple[float, float, float, float, float]], list[dict]]:
    boxes_obj = getattr(result, "boxes", None)
    if boxes_obj is None or len(boxes_obj) == 0:
        return [], []
    frame_h, frame_w = result.orig_shape[:2]
    xyxy = boxes_obj.xyxy.detach().cpu().numpy()
    conf = boxes_obj.conf.detach().cpu().numpy()
    out = []
    debug = []
    for idx, box in enumerate(xyxy):
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        raw_box = (x1, y1, x2, y2)
        conf_value = float(conf[idx]) if idx < len(conf) else 0.0
        mask_box = _mask_box(result, idx, (x1, y1, x2, y2))
        record = {
            "index": int(idx),
            "conf": round(conf_value, 6),
            "raw_box_xyxy": _box_to_list(raw_box),
            "mask_box_xyxy": _box_to_list(mask_box),
            "used_box_xyxy": None,
            "accepted": False,
            "reject_reason": "",
        }
        if min_conf is not None and conf_value < float(min_conf):
            record["reject_reason"] = f"low_conf<{float(min_conf):.2f}"
            debug.append(record)
            continue
        if mask_box is None:
            record["reject_reason"] = "empty_mask"
            debug.append(record)
            continue
        x1, y1, x2, y2 = mask_box
        if x2 <= x1 or y2 <= y1:
            record["reject_reason"] = "invalid_box"
            debug.append(record)
            continue
        reason = _box_limit_reason((x1, y1, x2, y2), frame_w, frame_h)
        if reason:
            record["reject_reason"] = reason
            debug.append(record)
            continue
        record["used_box_xyxy"] = _box_to_list((x1, y1, x2, y2))
        record["accepted"] = True
        debug.append(record)
        out.append((x1, y1, x2, y2, conf_value))
    return out, debug


def _resolve_detector_imgsz(frame_w: int | None = None, frame_h: int | None = None) -> int:
    configured = int(_cfg("pre_extract_yolo_imgsz", 2048) or 0)
    if configured > 0:
        return configured
    if frame_w and frame_h:
        value = max(int(frame_w), int(frame_h))
        return max(32, value + (-value % 32))
    return 2048


def _build_detector(log_callback=None, *, imgsz: int | None = None):
    model_path = _model_path()
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"pre-extract detection model not found: {model_path}")

    from gpu_engine import native_mosaic

    native_mosaic._prepare()
    import torch
    from lada.models.yolo.yolo11_segmentation_model import Yolo11SegmentationModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = bool(device == "cuda")
    imgsz = int(imgsz or _resolve_detector_imgsz())
    conf = float(_cfg("pre_extract_yolo_conf", 0.20) or 0.20)
    if log_callback:
        log_callback(
            f"[pre-extract] loading detector {os.path.basename(model_path)} "
            f"on {device}, imgsz={imgsz}, conf={conf:.2f}"
        )
    return Yolo11SegmentationModel(model_path, device=device, imgsz=imgsz, fp16=fp16, conf=conf)


def _get_detector(log_callback=None, *, frame_w: int | None = None, frame_h: int | None = None):
    global _DETECTOR, _DETECTOR_CONFIG
    imgsz = _resolve_detector_imgsz(frame_w, frame_h)
    conf = float(_cfg("pre_extract_yolo_conf", 0.20) or 0.20)
    config = (_model_path(), int(imgsz), float(conf))
    with _DETECTOR_LOCK:
        if _DETECTOR is None or _DETECTOR_CONFIG != config:
            _DETECTOR = _build_detector(log_callback, imgsz=imgsz)
            _DETECTOR_CONFIG = config
        return _DETECTOR


def _is_detector_oom(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text or "cublas_status_alloc_failed" in text


def _run_detector_batch(detector, frames: list, log_callback=None, *, boxes_only: bool = False):
    try:
        preprocessed = detector.preprocess(frames)
        if boxes_only:
            box_fn = getattr(detector, "inference_and_postprocess_boxes", None)
            if callable(box_fn):
                return list(box_fn(preprocessed, frames))
        return list(detector.inference_and_postprocess(preprocessed, frames))
    except (RuntimeError, MemoryError) as exc:
        if len(frames) <= 1 or not _is_detector_oom(exc):
            raise
        mid = max(1, len(frames) // 2)
        if log_callback:
            log_callback(
                f"[pre-extract] detector batch OOM at batch={len(frames)}; "
                f"retrying as {mid}+{len(frames) - mid}"
            )
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        return (
            _run_detector_batch(detector, frames[:mid], log_callback=log_callback, boxes_only=boxes_only)
            + _run_detector_batch(detector, frames[mid:], log_callback=log_callback, boxes_only=boxes_only)
        )


def release_detector(log_callback=None) -> None:
    """Release the cached YOLO detector and return its VRAM to CUDA."""
    global _DETECTOR, _DETECTOR_CONFIG
    with _DETECTOR_LOCK:
        if _DETECTOR is None:
            return
        _DETECTOR = None
        _DETECTOR_CONFIG = None
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    if log_callback:
        log_callback("[pre-extract] released detector")


def _scan_hits(video_path: str | Path, log_callback=None, cancel_token=None,
               min_conf: float | None = None) -> tuple[list[dict], probe.VideoMetadata, list[dict]]:
    import cv2
    import numpy as np
    import torch
    from gpu_engine.fallback import OperationCancelled

    meta = probe.probe_video(video_path)
    fps = meta.source_fps or 30.0
    total_frames = meta.nb_frames or int(round((meta.duration or 0.0) * fps))
    if total_frames <= 0:
        raise RuntimeError(f"cannot determine frame count for {video_path}")

    stride_s = max(0.05, float(_cfg("pre_extract_sample_stride_s", 0.5) or 0.5))
    stride_frames = max(1, int(round(stride_s * fps)))
    batch_size = max(1, int(_cfg("pre_extract_yolo_batch", 1) or 1))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video for pre-extract scan: {video_path}")
    detector = _get_detector(log_callback, frame_w=meta.width, frame_h=meta.height)

    hits: list[dict] = []
    debug_records: list[dict] = []
    batch_frames = []
    batch_times = []
    batch_indices = []
    sampled = 0
    t0 = time.perf_counter()
    last_log = 0.0

    def _flush_batch():
        nonlocal batch_frames, batch_times, batch_indices
        if not batch_frames:
            return
        results = _run_detector_batch(detector, batch_frames, log_callback=log_callback, boxes_only=True)
        for frame_idx, ts, result in zip(batch_indices, batch_times, results):
            boxes, detections = _extract_boxes_with_debug(result, min_conf=min_conf)
            if detections:
                debug_records.append({
                    "frame_idx": int(frame_idx),
                    "t": round(float(ts), 6),
                    "frame_size": [int(meta.width), int(meta.height)],
                    "detections": detections,
                    "accepted_boxes_xyxy": [_box_to_list(b[:4]) for b in boxes],
                })
            if boxes:
                hits.append({"t": ts, "boxes": boxes})
        batch_frames = []
        batch_times = []
        batch_indices = []

    try:
        frame_idx = 0
        while True:
            if cancel_token is not None and getattr(cancel_token, "cancelled", False):
                raise OperationCancelled("cancelled by user")
            ok = cap.grab()
            if not ok:
                break
            if frame_idx % stride_frames == 0:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    frame = np.ascontiguousarray(frame)
                    batch_frames.append(torch.from_numpy(frame))
                    batch_times.append(frame_idx / fps)
                    batch_indices.append(frame_idx)
                    sampled += 1
                    if len(batch_frames) >= batch_size:
                        _flush_batch()
            now = time.perf_counter()
            if log_callback and now - last_log >= 5.0:
                last_log = now
                pct = 100.0 * min(frame_idx + 1, total_frames) / total_frames
                elapsed = max(0.001, now - t0)
                log_callback(f"[pre-extract] scanned {sampled} samples ({pct:.1f}%) at {sampled / elapsed:.1f} samples/s")
            frame_idx += 1
        _flush_batch()
    finally:
        cap.release()

    if log_callback:
        log_callback(f"[pre-extract] detector hits: {len(hits)} sampled frames")
    return hits, meta, debug_records


def _keyframe_left_eye_geometry(meta: probe.VideoMetadata, video_path: str | Path) -> tuple[int, int, int, int]:
    src_w = max(2, int(meta.width or 0))
    src_h = max(2, int(meta.height or 0))
    if src_w <= 2 or src_h <= 2:
        raise RuntimeError(f"cannot determine source size for keyframe scan: {video_path}")
    out_w = max(2, src_w // 2)
    out_h = src_h
    out_w -= out_w % 2
    out_h -= out_h % 2
    return src_w, src_h, out_w, out_h


def _keyframe_scan_meta(video_path: str | Path, src_meta: probe.VideoMetadata,
                        out_w: int, out_h: int) -> probe.VideoMetadata:
    return probe.VideoMetadata(
        path=str(video_path),
        codec_name=src_meta.codec_name,
        profile=src_meta.profile,
        pix_fmt=src_meta.pix_fmt,
        width=int(out_w),
        height=int(out_h),
        bit_depth=src_meta.bit_depth,
        duration=src_meta.duration,
        nb_frames=src_meta.nb_frames,
        source_fps=src_meta.source_fps,
        is_cfr=src_meta.is_cfr,
        bitrate_bps=src_meta.bitrate_bps,
        color=src_meta.color,
        audio_codec="",
    )


def _scan_hits_keyframes_lowres(video_path: str | Path, log_callback=None,
                                cancel_token=None,
                                min_conf: float | None = None) -> tuple[list[dict], probe.VideoMetadata, list[dict]]:
    import numpy as np
    import torch
    from gpu_engine.fallback import OperationCancelled
    from utils.keyframe_cutter import list_keyframes

    meta = probe.probe_video(video_path)
    keyframes = list_keyframes(video_path)
    if not keyframes:
        if log_callback:
            log_callback("[source-scan] no keyframe list available; falling back to normal scan")
        return _scan_hits(video_path, log_callback=log_callback, cancel_token=cancel_token, min_conf=min_conf)

    src_w, src_h, out_w, out_h = _keyframe_left_eye_geometry(meta, video_path)
    frame_bytes = out_w * out_h * 3

    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg,
        "-hide_banner", "-loglevel", "error",
        "-skip_frame", "nokey",
        "-i", str(video_path),
        "-an", "-sn",
        "-vf", f"crop={out_w}:{out_h}:0:0",
        "-vsync", "0",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]
    if log_callback:
        log_callback(
            f"[source-scan] fast keyframe scan: {len(keyframes)} keyframes, "
            f"left-eye original-size crop {src_w}x{src_h} -> {out_w}x{out_h}"
        )
        log_callback(f"Executing: {' '.join(cmd)}")

    detector = _get_detector(log_callback, frame_w=out_w, frame_h=out_h)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        **_hidden_kwargs(),
    )
    hits: list[dict] = []
    debug_records: list[dict] = []
    batch_frames = []
    batch_times = []
    batch_indices = []
    batch_size = max(1, int(_cfg("pre_extract_yolo_batch", 1) or 1))
    sampled = 0
    t0 = time.perf_counter()
    last_log = 0.0

    def _flush_batch():
        nonlocal batch_frames, batch_times, batch_indices
        if not batch_frames:
            return
        results = _run_detector_batch(detector, batch_frames, log_callback=log_callback, boxes_only=True)
        for frame_idx, ts, result in zip(batch_indices, batch_times, results):
            boxes, detections = _extract_boxes_with_debug(result, min_conf=min_conf)
            if detections:
                debug_records.append({
                    "frame_idx": int(frame_idx),
                    "t": round(float(ts), 6),
                    "frame_size": [int(out_w), int(out_h)],
                    "detections": detections,
                    "accepted_boxes_xyxy": [_box_to_list(b[:4]) for b in boxes],
                })
            if boxes:
                hits.append({"t": ts, "boxes": boxes})
        batch_frames = []
        batch_times = []
        batch_indices = []

    try:
        assert proc.stdout is not None
        while True:
            if cancel_token is not None and getattr(cancel_token, "cancelled", False):
                raise OperationCancelled("cancelled by user")
            data = proc.stdout.read(frame_bytes)
            if not data:
                break
            if len(data) != frame_bytes:
                break
            ts = keyframes[sampled] if sampled < len(keyframes) else (sampled * 1.0)
            frame = np.frombuffer(data, dtype=np.uint8).reshape((out_h, out_w, 3)).copy()
            batch_frames.append(torch.from_numpy(frame))
            batch_times.append(float(ts))
            batch_indices.append(int(round(float(ts) * (meta.source_fps or 30.0))))
            sampled += 1
            if len(batch_frames) >= batch_size:
                _flush_batch()
            now = time.perf_counter()
            if log_callback and now - last_log >= 5.0:
                last_log = now
                pct = 100.0 * min(sampled, len(keyframes)) / max(1, len(keyframes))
                elapsed = max(0.001, now - t0)
                log_callback(f"[source-scan] scanned {sampled} keyframes ({pct:.1f}%) at {sampled / elapsed:.1f} samples/s")
        _flush_batch()
    finally:
        if proc.stdout:
            proc.stdout.close()
        if cancel_token is not None and getattr(cancel_token, "cancelled", False):
            try:
                proc.kill()
            except Exception:
                pass
        proc.wait()
    if proc.returncode not in (0, None):
        raise RuntimeError(f"source keyframe scan ffmpeg failed with code {proc.returncode}")
    if log_callback:
        log_callback(f"[source-scan] detector hits: {len(hits)} keyframes")
    return hits, _keyframe_scan_meta(video_path, meta, out_w, out_h), debug_records


def _scan_hits_keyframes_gpu(video_path: str | Path, log_callback=None,
                             cancel_token=None,
                             min_conf: float | None = None) -> tuple[list[dict], probe.VideoMetadata, list[dict]]:
    import PyNvVideoCodec as nvc
    from gpu_engine.fallback import OperationCancelled
    from utils.keyframe_cutter import list_keyframes

    src_meta, decision = probe.route(video_path)
    if not decision.is_gpu:
        raise RuntimeError(f"source keyframe GPU scan is not available: {decision.reason}")

    keyframes = list_keyframes(video_path)
    if not keyframes:
        raise RuntimeError("no keyframe list available for GPU keyframe scan")

    src_w, src_h, out_w, out_h = _keyframe_left_eye_geometry(src_meta, video_path)
    bd = 10 if src_meta.bit_depth > 8 else 8
    demuxer = nvc.CreateDemuxer(str(Path(video_path)))
    decoder = nvc.CreateDecoder(
        gpuid=0,
        codec=demuxer.GetNvCodecId(),
        usedevicememory=True,
        outputColorType=nvc.OutputColorType.NATIVE,
    )
    try:
        if log_callback:
            log_callback(
                f"[source-scan] GPU keyframe scan: {len(keyframes)} keyframes, "
                f"left-eye original-size crop {src_w}x{src_h} -> {out_w}x{out_h}, "
                f"route={decision.backend}, decoder=key-packet"
            )

        detector = _get_detector(log_callback, frame_w=out_w, frame_h=out_h)
        hits: list[dict] = []
        debug_records: list[dict] = []
        batch_frames = []
        batch_times = []
        batch_indices = []
        batch_size = max(1, int(_cfg("pre_extract_yolo_batch", 1) or 1))
        sampled = 0
        key_packets = 0
        t0 = time.perf_counter()
        last_log = 0.0
        pending_keys: list[dict] = []

        def _flush_batch():
            nonlocal batch_frames, batch_times, batch_indices
            if not batch_frames:
                return
            results = _run_detector_batch(detector, batch_frames, log_callback=log_callback, boxes_only=True)
            for frame_idx, ts, result in zip(batch_indices, batch_times, results):
                boxes, detections = _extract_boxes_with_debug(result, min_conf=min_conf)
                if detections:
                    debug_records.append({
                        "frame_idx": int(frame_idx),
                        "t": round(float(ts), 6),
                        "frame_size": [int(out_w), int(out_h)],
                        "detections": detections,
                        "accepted_boxes_xyxy": [_box_to_list(b[:4]) for b in boxes],
                    })
                if boxes:
                    hits.append({"t": ts, "boxes": boxes})
            batch_frames = []
            batch_times = []
            batch_indices = []

        def _pop_pending_key(decoded_pts: int | None) -> dict | None:
            if not pending_keys:
                return None
            if decoded_pts is not None:
                for pos, item in enumerate(pending_keys):
                    if item.get("pts") == decoded_pts:
                        if pos > 0:
                            del pending_keys[:pos]
                        return pending_keys.pop(0)
            return pending_keys.pop(0)

        def _handle_decoded_frames(decoded_frames) -> None:
            nonlocal sampled, last_log
            for decoded in decoded_frames or []:
                if cancel_token is not None and getattr(cancel_token, "cancelled", False):
                    raise OperationCancelled("cancelled by user")
                try:
                    decoded_pts = int(decoded.getPTS())
                except Exception:
                    decoded_pts = None
                key_info = _pop_pending_key(decoded_pts)
                if key_info is None:
                    continue
                frame = _decoded_frame_to_gpu_frame(decoded, src_w, src_h, bd)
                y_plane, uv_plane = frame.y_uv_cupy()
                y_plane = y_plane[0:out_h, 0:out_w]
                uv_plane = uv_plane[0:out_h // 2, 0:out_w // 2, :]
                batch_frames.append(_cupy_to_torch_bgr(y_plane, uv_plane, bit_depth=bd))
                batch_times.append(float(key_info["ts"]))
                batch_indices.append(int(key_info["frame_idx"]))
                sampled += 1
                if len(batch_frames) >= batch_size:
                    _flush_batch()
                now = time.perf_counter()
                if log_callback and now - last_log >= 5.0:
                    last_log = now
                    pct = 100.0 * min(key_packets, len(keyframes)) / max(1, len(keyframes))
                    elapsed = max(0.001, now - t0)
                    log_callback(
                        f"[source-scan] scanned {sampled} GPU keyframes ({pct:.1f}%) "
                        f"at {sampled / elapsed:.1f} samples/s"
                    )

        while True:
            now = time.perf_counter()
            if log_callback and now - last_log >= 5.0:
                last_log = now
                pct = 100.0 * min(key_packets, len(keyframes)) / max(1, len(keyframes))
                elapsed = max(0.001, now - t0)
                log_callback(
                    f"[source-scan] scanned {sampled} GPU keyframes ({pct:.1f}%) "
                    f"at {sampled / elapsed:.1f} samples/s"
                )
            if cancel_token is not None and getattr(cancel_token, "cancelled", False):
                raise OperationCancelled("cancelled by user")
            pkt = demuxer.Demux()
            if not getattr(pkt, "bsl", 0):
                _handle_decoded_frames(decoder.Decode(pkt))
                break
            if not bool(getattr(pkt, "key", False)):
                continue
            key_idx = key_packets
            key_packets += 1
            if key_idx < len(keyframes):
                ts = float(keyframes[key_idx])
            else:
                fps = src_meta.source_fps or 30.0
                ts = float(key_idx) / max(1.0, fps)
            pending_keys.append({
                "pts": int(getattr(pkt, "pts", -1)),
                "ts": ts,
                "frame_idx": int(round(ts * (src_meta.source_fps or 30.0))),
            })
            _handle_decoded_frames(decoder.Decode(pkt))
        _flush_batch()
        if log_callback and key_packets != len(keyframes):
            log_callback(
                f"[source-scan] GPU key packet count differs from ffprobe keyframes: "
                f"demux={key_packets}, ffprobe={len(keyframes)}"
            )
        if log_callback and pending_keys:
            log_callback(f"[source-scan] GPU keyframe decoder left {len(pending_keys)} pending keyframes")
        if log_callback:
            log_callback(f"[source-scan] detector hits: {len(hits)} GPU keyframes")
        return hits, _keyframe_scan_meta(video_path, src_meta, out_w, out_h), debug_records
    finally:
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


def _scan_hits_keyframes(video_path: str | Path, log_callback=None,
                         cancel_token=None,
                         min_conf: float | None = None) -> tuple[list[dict], probe.VideoMetadata, list[dict]]:
    from gpu_engine.fallback import OperationCancelled

    backend = str(_cfg("pre_extract_keyframe_scan_backend", "auto") or "auto").strip().lower()
    if backend not in {"auto", "gpu", "cpu"}:
        backend = "auto"
    if backend == "cpu":
        return _scan_hits_keyframes_lowres(
            video_path,
            log_callback=log_callback,
            cancel_token=cancel_token,
            min_conf=min_conf,
        )
    if backend == "gpu":
        return _scan_hits_keyframes_gpu(
            video_path,
            log_callback=log_callback,
            cancel_token=cancel_token,
            min_conf=min_conf,
        )

    try:
        _meta, decision = probe.route(video_path)
    except OperationCancelled:
        raise
    except Exception as exc:
        if log_callback:
            log_callback(f"[source-scan] keyframe scan GPU route check failed; falling back to CPU: {type(exc).__name__}: {exc}")
        return _scan_hits_keyframes_lowres(
            video_path,
            log_callback=log_callback,
            cancel_token=cancel_token,
            min_conf=min_conf,
        )

    if not decision.is_gpu:
        if log_callback:
            log_callback(f"[source-scan] keyframe scan backend: cpu ({decision.reason})")
        return _scan_hits_keyframes_lowres(
            video_path,
            log_callback=log_callback,
            cancel_token=cancel_token,
            min_conf=min_conf,
        )

    try:
        return _scan_hits_keyframes_gpu(
            video_path,
            log_callback=log_callback,
            cancel_token=cancel_token,
            min_conf=min_conf,
        )
    except OperationCancelled:
        raise
    except Exception as exc:
        if log_callback:
            log_callback(f"[source-scan] GPU keyframe scan failed; falling back to CPU: {type(exc).__name__}: {exc}")
        return _scan_hits_keyframes_lowres(
            video_path,
            log_callback=log_callback,
            cancel_token=cancel_token,
            min_conf=min_conf,
        )


def _scene_signature_from_luma(y_plane, bit_depth: int):
    """Downscaled luma histogram of one (cropped/fisheye) frame for shot-cut detection."""
    import cupy as cp

    from utils import scene_detect

    h = int(y_plane.shape[0])
    w = int(y_plane.shape[1])
    sh = max(1, h // 64)
    sw = max(1, w // 64)
    arr = cp.asnumpy(y_plane[::sh, ::sw])
    if int(bit_depth or 8) > 8:
        # P016/P010 store the high-bit-depth sample in the top of a 16-bit
        # container (full 0..65535 range, matching nv12_to_bgr's 255/65535
        # scaling), so the 8-bit downconversion is >> 8 -- not >> (depth-8),
        # which would overflow uint8 and produce a garbage histogram.
        arr = (arr >> 8).astype("uint8")
    else:
        arr = arr.astype("uint8")
    return scene_detect.compute_histogram(arr)


def _decoded_frame_to_gpu_frame(decoded, width: int, height: int, bit_depth: int):
    from gpu_engine.pynv_io import GpuNv12Frame, GpuP016Frame

    if int(bit_depth or 8) > 8:
        return GpuP016Frame.from_decoded_frame(decoded, int(width), int(height))
    return GpuNv12Frame.from_decoded_frame(decoded, int(width), int(height))


def _trim_gpu_memory() -> None:
    """Return cached CuPy-pool and PyTorch-allocator blocks to the driver.

    The per-frame CuPy->Torch dlpack handoff in the GPU scan can strand freed
    blocks in one allocator while the other keeps allocating fresh ones; on long
    8K scans this grows VRAM without bound and thrashes. Periodically trimming
    both pools caps the high-water mark.
    """
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def _cupy_to_torch_bgr(y_plane, uv_plane, bit_depth: int = 8):
    import cupy as cp
    import torch
    from gpu_engine import nv12_kernels

    bgr = cp.ascontiguousarray(nv12_kernels.nv12_to_bgr(y_plane, uv_plane, bit_depth=bit_depth))
    cp.cuda.get_current_stream().synchronize()
    try:
        return torch.utils.dlpack.from_dlpack(bgr)
    except TypeError:
        return torch.utils.dlpack.from_dlpack(bgr.toDlpack())


def _resize_lut(in_w: int, in_h: int, out_w: int, out_h: int):
    import cupy as cp

    key = (int(in_w), int(in_h), int(out_w), int(out_h))
    cached = _RESIZE_LUT_CACHE.get(key)
    if cached is not None:
        return cached
    if int(in_w) == int(out_w) and int(in_h) == int(out_h):
        lut = None
    else:
        xs = cp.arange(int(out_w), dtype=cp.float32)
        ys = cp.arange(int(out_h), dtype=cp.float32)
        xx, yy = cp.meshgrid(xs, ys)
        src_x = (xx + 0.5) * (float(in_w) / float(out_w)) - 0.5
        src_y = (yy + 0.5) * (float(in_h) / float(out_h)) - 0.5
        lut = cp.stack([src_x, src_y], axis=-1).astype(cp.float32)
    _RESIZE_LUT_CACHE[key] = lut
    return lut


def _detector_work_size(frame_w: int, frame_h: int) -> tuple[int, int]:
    imgsz = _resolve_detector_imgsz(frame_w, frame_h)
    if int(frame_w) <= 0 or int(frame_h) <= 0:
        return max(2, int(frame_w)), max(2, int(frame_h))
    scale = min(float(imgsz) / float(frame_w), float(imgsz) / float(frame_h))
    if scale >= 1.0:
        return max(2, int(frame_w)), max(2, int(frame_h))
    out_w = max(2, int(round(float(frame_w) * scale)))
    out_h = max(2, int(round(float(frame_h) * scale)))
    out_w = max(2, out_w - (out_w % 2))
    out_h = max(2, out_h - (out_h % 2))
    return out_w, out_h


def _resize_nv12_planes(y_plane, uv_plane, out_w: int, out_h: int):
    from gpu_engine import nv12_kernels

    in_h, in_w = y_plane.shape
    out_w = max(2, int(out_w))
    out_h = max(2, int(out_h))
    if (in_w, in_h) == (out_w, out_h):
        return y_plane, uv_plane
    y_lut = _resize_lut(in_w, in_h, out_w, out_h)
    uv_lut = _resize_lut(in_w // 2, in_h // 2, out_w // 2, out_h // 2)
    if y_lut is None or uv_lut is None:
        return y_plane, uv_plane
    return (
        nv12_kernels.remap_y(y_plane, y_lut, out_w, out_h),
        nv12_kernels.remap_uv(uv_plane, uv_lut, out_w // 2, out_h // 2),
    )


def _scale_box_xyxy(box: tuple[float, float, float, float], scale_x: float, scale_y: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (
        float(x1) * float(scale_x),
        float(y1) * float(scale_y),
        float(x2) * float(scale_x),
        float(y2) * float(scale_y),
    )


def _scale_debug_record(record: dict, scale_x: float, scale_y: float) -> dict:
    out = dict(record)
    for key in ("raw_box_xyxy", "mask_box_xyxy", "used_box_xyxy"):
        box = out.get(key)
        if box:
            out[key] = _box_to_list(_scale_box_xyxy(tuple(box[:4]), scale_x, scale_y))
    return out


def _scale_boxes_and_debug(boxes, debug, scale_x: float, scale_y: float):
    if scale_x == 1.0 and scale_y == 1.0:
        return boxes, debug
    scaled_boxes = [
        (
            float(x1) * float(scale_x),
            float(y1) * float(scale_y),
            float(x2) * float(scale_x),
            float(y2) * float(scale_y),
            float(conf),
        )
        for x1, y1, x2, y2, conf in boxes
    ]
    scaled_debug = [_scale_debug_record(record, scale_x, scale_y) for record in debug]
    return scaled_boxes, scaled_debug


def _split_sbs_boxes_to_eye(
    boxes: list[tuple[float, float, float, float, float]],
    frame_w: int,
    frame_h: int,
    crop_modes: tuple[str, ...] = ("left", "right"),
) -> dict[str, list[tuple[float, float, float, float, float]]]:
    """Split full-SBS detector boxes into per-eye local coordinates."""
    eye_w = max(1, int(frame_w) // 2)
    eye_h = max(1, int(frame_h))
    full_w = eye_w * 2
    out: dict[str, list[tuple[float, float, float, float, float]]] = {
        mode: [] for mode in crop_modes if mode in {"left", "right"}
    }
    for box in boxes:
        x1, y1, x2, y2, conf = [float(v) for v in box[:5]]
        y1 = max(0.0, min(float(eye_h), y1))
        y2 = max(0.0, min(float(eye_h), y2))
        if y2 - y1 <= 1.0:
            continue
        if "left" in out:
            lx1 = max(0.0, min(float(eye_w), x1))
            lx2 = max(0.0, min(float(eye_w), x2))
            if lx2 - lx1 > 1.0:
                out["left"].append((lx1, y1, lx2, y2, conf))
        if "right" in out:
            rx1 = max(float(eye_w), min(float(full_w), x1)) - float(eye_w)
            rx2 = max(float(eye_w), min(float(full_w), x2)) - float(eye_w)
            if rx2 - rx1 > 1.0:
                out["right"].append((rx1, y1, rx2, y2, conf))
    return out


def _split_sbs_detections_to_eye(
    detections: list[dict],
    frame_w: int,
    frame_h: int,
    crop_modes: tuple[str, ...],
) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {mode: [] for mode in crop_modes if mode in {"left", "right"}}
    for record in detections:
        if not bool(record.get("accepted")):
            continue
        used_box = record.get("used_box_xyxy")
        if not used_box:
            continue
        conf = float(record.get("conf", 0.0))
        split = _split_sbs_boxes_to_eye(
            [(
                float(used_box[0]),
                float(used_box[1]),
                float(used_box[2]),
                float(used_box[3]),
                conf,
            )],
            frame_w,
            frame_h,
            crop_modes,
        )
        for mode, boxes in split.items():
            for box in boxes:
                item = dict(record)
                item["eye"] = mode
                item["sbs_used_box_xyxy"] = record.get("used_box_xyxy")
                item["used_box_xyxy"] = _box_to_list(box[:4])
                out.setdefault(mode, []).append(item)
    return out


def _scan_hits_gpu_transform_pair(video_path: str | Path, *, crop_modes: tuple[str, ...],
                                  to_fisheye: bool = False,
                                  log_callback=None, cancel_token=None,
                                  min_conf: float | None = None) -> tuple[dict[str, list[dict]], probe.VideoMetadata, dict[str, list[dict]], dict[str, list[float]]]:
    import torch
    from gpu_engine.fallback import OperationCancelled
    from gpu_engine.pynv_io import PyNvThreadedSerialDecoder

    if to_fisheye:
        raise ValueError("paired GPU transform scan is non-fisheye only; scan fisheye eyes separately")
    crop_modes = tuple(mode for mode in crop_modes if mode in {"left", "right"})
    if not crop_modes:
        raise ValueError("no SBS eye crop modes requested")

    src_meta = probe.probe_video(video_path)
    bd = 10 if src_meta.bit_depth > 8 else 8
    fps = src_meta.source_fps or 30.0
    dec_buffer = max(2, int(_cfg("pre_extract_decoder_buffer", 8) or 8))
    dec = PyNvThreadedSerialDecoder(Path(video_path), bit_depth=bd, buffer_size=dec_buffer)
    try:
        info = dec.info
        eye_w = max(2, int(info.width) // 2)
        eye_h = max(2, int(info.height))
        scan_w = eye_w * 2
        detector_w, detector_h = _detector_work_size(scan_w, eye_h)
        detector_scale_x = float(scan_w) / float(detector_w)
        detector_scale_y = float(eye_h) / float(detector_h)

        total_frames = len(dec)
        stride_s = max(0.05, float(_cfg("pre_extract_sample_stride_s", 0.5) or 0.5))
        stride_frames = max(1, int(round(stride_s * fps)))
        batch_size = max(1, int(_cfg("pre_extract_yolo_batch", 1) or 1))
        detector = _get_detector(log_callback, frame_w=scan_w, frame_h=eye_h)

        hits_by_mode: dict[str, list[dict]] = {mode: [] for mode in crop_modes}
        debug_by_mode: dict[str, list[dict]] = {mode: [] for mode in crop_modes}
        cuts_by_mode: dict[str, list[float]] = {mode: [] for mode in crop_modes}
        scene_det_by_mode = {}
        scene_det_error: str | None = None
        if bool(_cfg("pre_extract_scene_detect_enabled", True)):
            from utils import scene_detect

            for crop_mode in crop_modes:
                scene_det_by_mode[crop_mode] = scene_detect.SceneCutDetector(
                    min_scene_len_s=float(_cfg("pre_extract_scene_min_len_s", 1.5) or 1.5),
                    floor=float(_cfg("pre_extract_scene_floor", 0.30) or 0.30),
                    k=float(_cfg("pre_extract_scene_k", 3.0) or 3.0),
                )

        out_meta = probe.VideoMetadata(
            path=str(video_path),
            codec_name=src_meta.codec_name,
            profile=src_meta.profile,
            pix_fmt=src_meta.pix_fmt,
            width=int(eye_w),
            height=int(eye_h),
            bit_depth=src_meta.bit_depth,
            duration=src_meta.duration,
            nb_frames=src_meta.nb_frames,
            source_fps=fps,
            is_cfr=src_meta.is_cfr,
            bitrate_bps=src_meta.bitrate_bps,
            color=src_meta.color,
            audio_codec="",
        )

        batch_frames = []
        batch_meta = []
        sampled = 0
        t0 = time.perf_counter()
        last_log = 0.0
        trim_samples = int(_cfg("pre_extract_vram_trim_samples", 32) or 0)
        last_trim = 0

        def _flush_batch():
            nonlocal batch_frames, batch_meta
            if not batch_frames:
                return
            results = _run_detector_batch(detector, batch_frames, log_callback=log_callback, boxes_only=True)
            for meta_item, result in zip(batch_meta, results):
                frame_idx = meta_item["frame_idx"]
                ts = meta_item["ts"]
                boxes, detections = _extract_boxes_with_debug(result, min_conf=min_conf)
                boxes, detections = _scale_boxes_and_debug(
                    boxes,
                    detections,
                    meta_item["scale_x"],
                    meta_item["scale_y"],
                )
                boxes_by_mode = _split_sbs_boxes_to_eye(boxes, scan_w, eye_h, crop_modes)
                debug_split = _split_sbs_detections_to_eye(detections, scan_w, eye_h, crop_modes) if detections else {}
                for crop_mode in crop_modes:
                    eye_boxes = boxes_by_mode.get(crop_mode, [])
                    eye_debug = debug_split.get(crop_mode, [])
                    if eye_debug:
                        debug_by_mode[crop_mode].append({
                            "frame_idx": int(frame_idx),
                            "t": round(float(ts), 6),
                            "frame_size": [int(eye_w), int(eye_h)],
                            "sbs_frame_size": [int(scan_w), int(eye_h)],
                            "detector_frame_size": [int(meta_item["detector_w"]), int(meta_item["detector_h"])],
                            "detections": eye_debug,
                            "accepted_boxes_xyxy": [_box_to_list(b[:4]) for b in eye_boxes],
                        })
                    if eye_boxes:
                        hits_by_mode[crop_mode].append({"t": ts, "boxes": eye_boxes})
            batch_frames = []
            batch_meta = []

        if log_callback:
            modes_text = ",".join(crop_modes)
            log_callback(
                f"[pre-extract] GPU SBS pair scan: eyes={modes_text}, input={scan_w}x{eye_h}, "
                f"detector_input={detector_w}x{detector_h}, "
                f"frames={total_frames}, stride={stride_s:.2f}s, conf_filter={min_conf if min_conf is not None else 'model'}"
            )

        for frame_idx in range(0, total_frames, stride_frames):
            if cancel_token is not None and getattr(cancel_token, "cancelled", False):
                raise OperationCancelled("cancelled by user")
            frame = dec.frame_at(frame_idx)
            y_plane, uv_plane = frame.y_uv_cupy()
            scan_y = y_plane[:eye_h, :scan_w]
            scan_uv = uv_plane[:eye_h // 2, :scan_w // 2, :]
            for crop_mode in crop_modes:
                if crop_mode in scene_det_by_mode:
                    try:
                        x = 0 if crop_mode == "left" else eye_w
                        cropped_y = scan_y[:, x:x + eye_w]
                        sig = _scene_signature_from_luma(cropped_y, bd)
                        if scene_det_by_mode[crop_mode].update(float(frame_idx) / fps, sig):
                            tcut = float(frame_idx) / fps
                            cuts_by_mode[crop_mode].append(tcut)
                            if log_callback:
                                log_callback(f"[pre-extract] scene cut detected at {tcut:.1f}s ({crop_mode})")
                    except Exception as exc:
                        if scene_det_error is None:
                            scene_det_error = f"{type(exc).__name__}: {exc}"
                            if log_callback:
                                log_callback(f"[pre-extract] scene detect disabled after error: {scene_det_error}")
                            scene_det_by_mode = {}
            detector_y, detector_uv = _resize_nv12_planes(scan_y, scan_uv, detector_w, detector_h)
            batch_frames.append(_cupy_to_torch_bgr(detector_y, detector_uv, bit_depth=bd))
            batch_meta.append({
                "frame_idx": frame_idx,
                "ts": float(frame_idx) / fps,
                "scale_x": detector_scale_x,
                "scale_y": detector_scale_y,
                "detector_w": detector_w,
                "detector_h": detector_h,
            })
            sampled += 1
            if len(batch_frames) >= batch_size:
                _flush_batch()
                if trim_samples and sampled - last_trim >= trim_samples:
                    last_trim = sampled
                    _trim_gpu_memory()
            now = time.perf_counter()
            if log_callback and now - last_log >= 5.0:
                last_log = now
                pct = 100.0 * min(frame_idx + 1, total_frames) / max(1, total_frames)
                elapsed = max(0.001, now - t0)
                log_callback(
                    f"[pre-extract] scanned {sampled} GPU samples ({pct:.1f}%) at {sampled / elapsed:.1f} samples/s"
                )
        _flush_batch()
        if log_callback:
            total_hits = sum(len(v) for v in hits_by_mode.values())
            log_callback(f"[pre-extract] detector hits: {total_hits} SBS-split sampled frames")
            if any(cuts_by_mode.values()):
                log_callback(
                    "[pre-extract] scene cuts detected: "
                    + ", ".join(f"{mode}={len(cuts_by_mode[mode])}" for mode in crop_modes)
                )
        return hits_by_mode, out_meta, debug_by_mode, cuts_by_mode
    finally:
        dec.stop()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _scan_hits_gpu_transform(video_path: str | Path, *, crop_mode: str,
                             to_fisheye: bool = False,
                             log_callback=None, cancel_token=None,
                             min_conf: float | None = None) -> tuple[list[dict], probe.VideoMetadata, list[dict]]:
    import torch
    from gpu_engine import nv12_kernels, v360_lut
    from gpu_engine.fallback import OperationCancelled
    from gpu_engine.pynv_io import PyNvThreadedSerialDecoder

    src_meta = probe.probe_video(video_path)
    bd = 10 if src_meta.bit_depth > 8 else 8
    fps = src_meta.source_fps or 30.0
    # Detection only samples sparsely (~stride_s), so a deep decode buffer is
    # pure VRAM cost. At 8K each buffered surface is ~100MB; the default 32 is
    # what makes the fine scan open at ~15GB. 8 keeps the pipeline fed cheaply.
    dec_buffer = max(2, int(_cfg("pre_extract_decoder_buffer", 8) or 8))
    dec = PyNvThreadedSerialDecoder(Path(video_path), bit_depth=bd, buffer_size=dec_buffer)
    try:
        info = dec.info
        if crop_mode == "left":
            x, y0, out_w, out_h = 0, 0, info.width // 2, info.height
        elif crop_mode == "right":
            x, y0, out_w, out_h = info.width // 2, 0, info.width // 2, info.height
        else:
            x, y0, out_w, out_h = 0, 0, info.width, info.height
        lut_y = v360_lut.make_lut("heq2fisheye", out_w, out_h) if to_fisheye else None
        lut_c = v360_lut.make_lut("heq2fisheye", out_w // 2, out_h // 2) if to_fisheye else None

        total_frames = len(dec)
        stride_s = max(0.05, float(_cfg("pre_extract_sample_stride_s", 0.5) or 0.5))
        stride_frames = max(1, int(round(stride_s * fps)))
        batch_size = max(1, int(_cfg("pre_extract_yolo_batch", 1) or 1))
        detector = _get_detector(log_callback, frame_w=out_w, frame_h=out_h)
        detector_w, detector_h = _detector_work_size(out_w, out_h)
        detector_scale_x = float(out_w) / float(detector_w)
        detector_scale_y = float(out_h) / float(detector_h)

        hits: list[dict] = []
        debug_records: list[dict] = []
        batch_frames = []
        batch_times = []
        batch_indices = []
        batch_scales = []
        batch_detector_sizes = []
        sampled = 0
        t0 = time.perf_counter()
        last_log = 0.0
        trim_samples = int(_cfg("pre_extract_vram_trim_samples", 32) or 0)
        last_trim = 0

        # Per-eye shot-cut detection on the same sampled frames (luma histogram).
        scene_cuts: list[float] = []
        scene_det = None
        scene_det_error: str | None = None
        if bool(_cfg("pre_extract_scene_detect_enabled", True)):
            from utils import scene_detect

            scene_det = scene_detect.SceneCutDetector(
                min_scene_len_s=float(_cfg("pre_extract_scene_min_len_s", 1.5) or 1.5),
                floor=float(_cfg("pre_extract_scene_floor", 0.30) or 0.30),
                k=float(_cfg("pre_extract_scene_k", 3.0) or 3.0),
            )

        out_meta = probe.VideoMetadata(
            path=str(video_path),
            codec_name=src_meta.codec_name,
            profile=src_meta.profile,
            pix_fmt=src_meta.pix_fmt,
            width=int(out_w),
            height=int(out_h),
            bit_depth=src_meta.bit_depth,
            duration=src_meta.duration,
            nb_frames=src_meta.nb_frames,
            source_fps=fps,
            is_cfr=src_meta.is_cfr,
            bitrate_bps=src_meta.bitrate_bps,
            color=src_meta.color,
            audio_codec="",
        )

        def _flush_batch():
            nonlocal batch_frames, batch_times, batch_indices, batch_scales, batch_detector_sizes
            if not batch_frames:
                return
            results = _run_detector_batch(detector, batch_frames, log_callback=log_callback, boxes_only=True)
            for frame_idx, ts, result, scale_item, detector_size in zip(
                batch_indices,
                batch_times,
                results,
                batch_scales,
                batch_detector_sizes,
            ):
                boxes, detections = _extract_boxes_with_debug(result, min_conf=min_conf)
                boxes, detections = _scale_boxes_and_debug(
                    boxes,
                    detections,
                    scale_item[0],
                    scale_item[1],
                )
                if detections:
                    debug_records.append({
                        "frame_idx": int(frame_idx),
                        "t": round(float(ts), 6),
                        "frame_size": [int(out_w), int(out_h)],
                        "detector_frame_size": [int(detector_size[0]), int(detector_size[1])],
                        "detections": detections,
                        "accepted_boxes_xyxy": [_box_to_list(b[:4]) for b in boxes],
                    })
                if boxes:
                    hits.append({"t": ts, "boxes": boxes})
            batch_frames = []
            batch_times = []
            batch_indices = []
            batch_scales = []
            batch_detector_sizes = []

        if log_callback:
            suffix = " + fisheye" if to_fisheye else ""
            log_callback(
                f"[pre-extract] GPU scan transform: crop={crop_mode}{suffix}, "
                f"detector_input={detector_w}x{detector_h}, "
                f"frames={total_frames}, stride={stride_s:.2f}s, conf_filter={min_conf if min_conf is not None else 'model'}"
            )

        for frame_idx in range(0, total_frames, stride_frames):
            if cancel_token is not None and getattr(cancel_token, "cancelled", False):
                raise OperationCancelled("cancelled by user")
            frame = dec.frame_at(frame_idx)
            y_plane, uv_plane = frame.y_uv_cupy()
            y_plane = y_plane[y0:y0 + out_h, x:x + out_w]
            uv_plane = uv_plane[y0 // 2:(y0 + out_h) // 2, x // 2:(x + out_w) // 2, :]
            if to_fisheye:
                y_plane = nv12_kernels.remap_y(y_plane, lut_y, out_w, out_h)
                uv_plane = nv12_kernels.remap_uv(uv_plane, lut_c, out_w // 2, out_h // 2)
            if scene_det is not None:
                try:
                    sig = _scene_signature_from_luma(y_plane, bd)
                    if scene_det.update(float(frame_idx) / fps, sig):
                        tcut = float(frame_idx) / fps
                        scene_cuts.append(tcut)
                        if log_callback:
                            log_callback(f"[pre-extract] scene cut detected at {tcut:.1f}s")
                except Exception as exc:
                    if scene_det_error is None:
                        scene_det_error = f"{type(exc).__name__}: {exc}"
                        if log_callback:
                            log_callback(f"[pre-extract] scene detect disabled after error: {scene_det_error}")
                        scene_det = None
            detector_y, detector_uv = _resize_nv12_planes(y_plane, uv_plane, detector_w, detector_h)
            batch_frames.append(_cupy_to_torch_bgr(detector_y, detector_uv, bit_depth=bd))
            batch_times.append(float(frame_idx) / fps)
            batch_indices.append(frame_idx)
            batch_scales.append((detector_scale_x, detector_scale_y))
            batch_detector_sizes.append((detector_w, detector_h))
            sampled += 1
            if len(batch_frames) >= batch_size:
                _flush_batch()
                if trim_samples and sampled - last_trim >= trim_samples:
                    last_trim = sampled
                    _trim_gpu_memory()
            now = time.perf_counter()
            if log_callback and now - last_log >= 5.0:
                last_log = now
                pct = 100.0 * min(frame_idx + 1, total_frames) / max(1, total_frames)
                elapsed = max(0.001, now - t0)
                log_callback(f"[pre-extract] scanned {sampled} GPU samples ({pct:.1f}%) at {sampled / elapsed:.1f} samples/s")
        _flush_batch()
        if log_callback:
            log_callback(f"[pre-extract] detector hits: {len(hits)} transformed sampled frames")
            if scene_det is not None:
                log_callback(f"[pre-extract] scene cuts detected: {len(scene_cuts)}")
        return hits, out_meta, debug_records, scene_cuts
    finally:
        dec.stop()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _merge_hit_groups(groups: list[dict], min_gap_s: float) -> list[dict]:
    if not groups:
        return []
    merged = [groups[0]]
    for group in groups[1:]:
        prev = merged[-1]
        if group["start"] - prev["end"] <= min_gap_s:
            prev["end"] = max(prev["end"], group["end"])
            prev["last_hit"] = max(prev["last_hit"], group["last_hit"])
            prev["boxes"].extend(group["boxes"])
            prev["samples"].extend(group["samples"])
        else:
            merged.append(group)
    return merged


def _union4(boxes) -> tuple[float, float, float, float] | None:
    """Axis-aligned union (x1, y1, x2, y2) of a box list, or None if empty."""
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _segment_area_ratio(union: tuple[float, float, float, float] | None,
                        frame_w: int, frame_h: int) -> float:
    """Area ratio of the *expanded* crop rect that this union would produce.

    Uses ``_expanded_rect`` so the cap is compared against the same rect the
    bypass-crop check sees downstream.
    """
    if union is None:
        return 0.0
    x1, y1, x2, y2 = union
    _x, _y, w, h, _conf = _expanded_rect([(x1, y1, x2, y2, 1.0)], frame_w, frame_h)
    fa = max(1, int(frame_w) * int(frame_h))
    return float(max(0, w) * max(0, h)) / float(fa)


def _split_samples_by_area(samples: list[tuple[float, list]], frame_w: int, frame_h: int,
                           *, max_area_ratio: float, cut_gap_s: float,
                           min_dur_s: float,
                           scene_cuts: list[float] | None = None) -> list[list[tuple[float, list]]]:
    """Split a time-ordered sample run into sub-windows.

    Two complementary triggers:

    * **Scene cut** (whole-frame, from the histogram detector): a hard boundary
      at the cut time regardless of mosaic state -- handles videos made of
      distinct shots. Empty on single-shot footage.
    * **Crop area** (mosaic drift within a shot): a new sub-window starts only
      at a mosaic-free gap (>= ``cut_gap_s``), so cuts never land inside
      continuous mosaic (which would break the temporal restoration). A cut
      happens when the running crop rect would grow past ``max_area_ratio`` and
      the segment so far is at least ``min_dur_s`` long. With no gap to cut at,
      the window is forced to keep growing (rare; bounded in practice).
    """
    if len(samples) <= 1:
        return [samples]
    cuts = sorted(float(c) for c in (scene_cuts or []))
    sc_idx = 0
    area_enabled = max_area_ratio > 0.0

    out: list[list[tuple[float, list]]] = []
    seg_start = 0
    last_gap = None  # index i such that a cuttable gap precedes samples[i]
    union: tuple[float, float, float, float] | None = None

    for i, (t, boxes) in enumerate(samples):
        # Forced scene-cut boundary between samples[i-1] and samples[i].
        if cuts and i > seg_start:
            prev_t = samples[i - 1][0]
            while sc_idx < len(cuts) and cuts[sc_idx] <= prev_t:
                sc_idx += 1
            if sc_idx < len(cuts) and prev_t < cuts[sc_idx] <= t:
                out.append(samples[seg_start:i])
                seg_start = i
                union = _union4(boxes)
                last_gap = None
                continue

        if i > seg_start and last_gap != i and (t - samples[i - 1][0]) > cut_gap_s:
            last_gap = i
        tentative = union
        bu = _union4(boxes)
        if bu is not None:
            tentative = bu if tentative is None else (
                min(tentative[0], bu[0]), min(tentative[1], bu[1]),
                max(tentative[2], bu[2]), max(tentative[3], bu[3]),
            )
        if area_enabled and union is not None \
                and _segment_area_ratio(tentative, frame_w, frame_h) > max_area_ratio \
                and last_gap is not None and last_gap > seg_start \
                and (samples[last_gap - 1][0] - samples[seg_start][0]) >= min_dur_s:
            out.append(samples[seg_start:last_gap])
            seg_start = last_gap
            union = _union4([b for _, bs in samples[seg_start:i + 1] for b in bs])
            last_gap = None
        else:
            union = tentative

    out.append(samples[seg_start:])
    return [s for s in out if s]


def _aggregate_hits(hits: list[dict], meta: probe.VideoMetadata,
                    scene_cuts: list[float] | None = None) -> list[MosaicSegment]:
    if not hits:
        return []
    duration = meta.duration or (meta.nb_frames / meta.source_fps if meta.nb_frames and meta.source_fps else 0.0)
    merge_gap_s = float(_cfg("pre_extract_merge_gap_s", 1.5) or 1.5)
    min_gap_s = float(_cfg("pre_extract_min_gap_s", 2.0) or 2.0)
    pad_s = float(_cfg("pre_extract_head_tail_pad_s", 2.0) or 2.0)
    min_segment_s = float(_cfg("pre_extract_min_segment_s", 1.5) or 1.5)

    max_area_ratio = float(_cfg("pre_extract_segment_max_area_ratio", 0.33) or 0.0)
    cut_gap_s = float(_cfg("pre_extract_segment_cut_gap_s", 0.75) or 0.75)
    seg_min_dur_s = float(_cfg("pre_extract_segment_min_dur_s", 10.0) or 0.0)

    hits = sorted(hits, key=lambda h: float(h["t"]))
    groups: list[dict] = []
    current = None
    for hit in hits:
        t = float(hit["t"])
        if current is None or t - current["last_hit"] > merge_gap_s:
            if current is not None:
                groups.append(current)
            current = {"start": t, "end": t, "last_hit": t,
                       "boxes": list(hit["boxes"]), "samples": [(t, list(hit["boxes"]))]}
        else:
            current["end"] = t
            current["last_hit"] = t
            current["boxes"].extend(hit["boxes"])
            current["samples"].append((t, list(hit["boxes"])))
    if current is not None:
        groups.append(current)

    for group in groups:
        group["start"] = max(0.0, float(group["start"]) - pad_s)
        if duration > 0:
            group["end"] = min(duration, float(group["end"]) + pad_s)
        else:
            group["end"] = float(group["end"]) + pad_s
    groups = _merge_hit_groups(groups, min_gap_s)

    segments: list[MosaicSegment] = []
    for group in groups:
        if group["end"] - group["start"] < min_segment_s:
            continue
        windows = _split_samples_by_area(
            group["samples"], meta.width, meta.height,
            max_area_ratio=max_area_ratio, cut_gap_s=cut_gap_s, min_dur_s=seg_min_dur_s,
            scene_cuts=scene_cuts,
        )
        for w_idx, window in enumerate(windows):
            # Outer ends use the padded group bounds; interior cuts sit at the
            # midpoint of the mosaic-free gap so adjacent windows never overlap.
            if w_idx == 0:
                seg_start = float(group["start"])
            else:
                seg_start = 0.5 * (windows[w_idx - 1][-1][0] + window[0][0])
            if w_idx == len(windows) - 1:
                seg_end = float(group["end"])
            else:
                seg_end = 0.5 * (window[-1][0] + windows[w_idx + 1][0][0])
            if seg_end - seg_start < min_segment_s:
                continue
            window_boxes = [b for _, bs in window for b in bs]
            boxes = _filter_spatial_outliers(window_boxes, meta.width, meta.height)
            for cluster_boxes in _spatial_cluster(boxes, meta.width, meta.height):
                x, y, w, h, conf = _expanded_rect(cluster_boxes, meta.width, meta.height)
                if w <= 0 or h <= 0:
                    continue
                segments.append(MosaicSegment(
                    seg_id=len(segments),
                    start_s=seg_start,
                    end_s=seg_end,
                    start_s_kf=seg_start,
                    end_s_kf=seg_end,
                    x=x, y=y, w=w, h=h,
                    conf_max=conf,
                ))
    return _merge_overlapping_segments(segments)


def _debug_jsonl_path(video_path: str | Path) -> Path:
    p = Path(video_path)
    return p.with_name(f"{p.stem}.detections.jsonl")


def save_detection_debug_jsonl(records: list[dict], path: str | Path, source: str | Path | None = None) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        header = {
            "type": "metadata",
            "source": str(source) if source else "",
            "note": (
                "raw_box_xyxy is direct YOLO output; mask_box_xyxy is the box recomputed from "
                "the segmentation mask when mask postprocess is enabled; accepted_boxes_xyxy are "
                "the boxes used by pre-extract."
            ),
        }
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def scan_segments(video_path: str | Path, log_callback=None, cancel_token=None,
                  scan_strategy: str | None = None,
                  min_conf: float | None = None) -> list[MosaicSegment]:
    strategy = str(scan_strategy or "normal").lower()
    cache_path = _empty_scan_cache_path(video_path, mode=f"scan_{strategy}", min_conf=min_conf)
    if _load_empty_scan_cache(cache_path, log_callback=log_callback):
        return []
    if strategy in {"keyframes", "keyframe", "source_keyframes"}:
        hits, meta, debug_records = _scan_hits_keyframes(video_path, log_callback=log_callback, cancel_token=cancel_token, min_conf=min_conf)
    else:
        hits, meta, debug_records = _scan_hits(video_path, log_callback=log_callback, cancel_token=cancel_token, min_conf=min_conf)
    if bool(_cfg("pre_extract_save_detection_debug", True)):
        debug_path = _debug_jsonl_path(video_path)
        save_detection_debug_jsonl(debug_records, debug_path, source=video_path)
        if log_callback:
            log_callback(f"[pre-extract] saved detection debug: {debug_path}")
    segments = _aggregate_hits(hits, meta)
    if log_callback:
        covered = sum(s.duration_s for s in segments)
        dur = meta.duration or 0.0
        pct = (100.0 * covered / dur) if dur > 0 else 0.0
        log_callback(f"[pre-extract] aggregated {len(segments)} segments, {covered:.1f}s ({pct:.1f}% of video)")
    if not segments:
        _write_empty_scan_cache(cache_path, log_callback=log_callback)
    return segments


def scan_segments_gpu_transform(video_path: str | Path, *, crop_mode: str,
                                to_fisheye: bool = False,
                                log_callback=None, cancel_token=None,
                                min_conf: float | None = None) -> list[MosaicSegment]:
    cache_path = _empty_scan_cache_path(
        video_path,
        mode="gpu_transform",
        min_conf=min_conf,
        crop_mode=crop_mode,
        to_fisheye=to_fisheye,
    )
    if _load_empty_scan_cache(cache_path, log_callback=log_callback):
        return []
    hits, meta, debug_records, scene_cuts = _scan_hits_gpu_transform(
        video_path,
        crop_mode=crop_mode,
        to_fisheye=to_fisheye,
        log_callback=log_callback,
        cancel_token=cancel_token,
        min_conf=min_conf,
    )
    if bool(_cfg("pre_extract_save_detection_debug", True)):
        p = Path(video_path)
        suffix = f"{crop_mode}{'_fisheye' if to_fisheye else ''}"
        debug_path = p.with_name(f"{p.stem}.{suffix}.detections.jsonl")
        save_detection_debug_jsonl(debug_records, debug_path, source=video_path)
        if log_callback:
            log_callback(f"[pre-extract] saved detection debug: {debug_path}")
    segments = _aggregate_hits(hits, meta, scene_cuts=scene_cuts)
    if log_callback:
        covered = sum(s.duration_s for s in segments)
        dur = meta.duration or 0.0
        pct = (100.0 * covered / dur) if dur > 0 else 0.0
        log_callback(f"[pre-extract] aggregated {len(segments)} transformed segments, {covered:.1f}s ({pct:.1f}% of video)")
    if not segments:
        _write_empty_scan_cache(cache_path, log_callback=log_callback)
    return segments


def scan_segments_gpu_transform_pair(video_path: str | Path, *,
                                     to_fisheye: bool = False,
                                     log_callback=None, cancel_token=None,
                                     min_conf: float | None = None) -> tuple[list[MosaicSegment], list[MosaicSegment]]:
    if to_fisheye:
        if log_callback:
            log_callback("[pre-extract] fisheye fine scan uses separate per-eye GPU transforms")
        return (
            scan_segments_gpu_transform(
                video_path,
                crop_mode="left",
                to_fisheye=True,
                log_callback=log_callback,
                cancel_token=cancel_token,
                min_conf=min_conf,
            ),
            scan_segments_gpu_transform(
                video_path,
                crop_mode="right",
                to_fisheye=True,
                log_callback=log_callback,
                cancel_token=cancel_token,
                min_conf=min_conf,
            ),
        )

    modes = ("left", "right")
    cache_paths = {
        mode: _empty_scan_cache_path(
            video_path,
            mode="gpu_transform_pair_sbs",
            min_conf=min_conf,
            crop_mode=mode,
            to_fisheye=to_fisheye,
        )
        for mode in modes
    }
    segments_by_mode: dict[str, list[MosaicSegment] | None] = {}
    scan_modes = []
    for mode in modes:
        if _load_empty_scan_cache(cache_paths[mode], log_callback=log_callback):
            segments_by_mode[mode] = []
        else:
            segments_by_mode[mode] = None
            scan_modes.append(mode)

    if scan_modes:
        hits_by_mode, meta, debug_by_mode, cuts_by_mode = _scan_hits_gpu_transform_pair(
            video_path,
            crop_modes=tuple(scan_modes),
            to_fisheye=to_fisheye,
            log_callback=log_callback,
            cancel_token=cancel_token,
            min_conf=min_conf,
        )
        for mode in scan_modes:
            if bool(_cfg("pre_extract_save_detection_debug", True)):
                p = Path(video_path)
                suffix = f"{mode}{'_fisheye' if to_fisheye else ''}"
                debug_path = p.with_name(f"{p.stem}.{suffix}.detections.jsonl")
                save_detection_debug_jsonl(debug_by_mode.get(mode, []), debug_path, source=video_path)
                if log_callback:
                    log_callback(f"[pre-extract] saved detection debug: {debug_path}")
            segments = _aggregate_hits(
                hits_by_mode.get(mode, []),
                meta,
                scene_cuts=cuts_by_mode.get(mode, []),
            )
            if log_callback:
                covered = sum(s.duration_s for s in segments)
                dur = meta.duration or 0.0
                pct = (100.0 * covered / dur) if dur > 0 else 0.0
                log_callback(
                    f"[pre-extract] aggregated {len(segments)} transformed {mode} segments, "
                    f"{covered:.1f}s ({pct:.1f}% of video)"
                )
            if not segments:
                _write_empty_scan_cache(cache_paths[mode], log_callback=log_callback)
            segments_by_mode[mode] = segments

    return (
        list(segments_by_mode.get("left") or []),
        list(segments_by_mode.get("right") or []),
    )


def _format_hms(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm for human reading."""
    try:
        s = max(0.0, float(seconds))
    except (TypeError, ValueError):
        s = 0.0
    h, rem = divmod(s, 3600.0)
    m, sec = divmod(rem, 60.0)
    return f"{int(h):02d}:{int(m):02d}:{sec:06.3f}"


def _segment_view_dict(seg: MosaicSegment, fps: float | None) -> dict:
    """Serialize a MosaicSegment with extra human/analysis fields appended."""
    data = seg.to_dict()
    start_s = float(seg.start_s)
    end_s = float(seg.end_s)
    start_kf = float(seg.start_s_kf)
    end_kf = float(seg.end_s_kf)
    data["duration_s"] = round(max(0.0, end_s - start_s), 6)
    data["start_hms"] = _format_hms(start_s)
    data["end_hms"] = _format_hms(end_s)
    if fps and fps > 0:
        data["fps"] = float(fps)
        data["start_frame"] = max(0, int(round(start_s * fps)))
        data["end_frame"] = max(0, int(round(end_s * fps)))
        data["start_frame_kf"] = max(0, int(round(start_kf * fps)))
        data["end_frame_kf"] = max(0, int(round(end_kf * fps)))
        data["frame_count"] = max(0, data["end_frame"] - data["start_frame"])
    return data


def save_segments_json(segments: list[MosaicSegment], path: str | Path,
                       source: str | Path | None = None,
                       fps: float | None = None) -> None:
    data = {
        "source": str(source) if source else "",
        "segments": [_segment_view_dict(seg, fps) for seg in segments],
    }
    if fps and fps > 0:
        data["fps"] = float(fps)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_segments_json(path: str | Path) -> list[MosaicSegment]:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return [MosaicSegment.from_dict(item) for item in data.get("segments", [])]
