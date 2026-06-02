"""Pre-scan videos with LADA's YOLO mosaic detector and aggregate useful clips."""
from __future__ import annotations

import json
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
_DETECTOR_LOCK = threading.Lock()


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


def _build_detector(log_callback=None):
    model_path = _model_path()
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"pre-extract detection model not found: {model_path}")

    from gpu_engine import native_mosaic

    native_mosaic._prepare()
    import torch
    from lada.models.yolo.yolo11_segmentation_model import Yolo11SegmentationModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = bool(device == "cuda")
    imgsz = int(_cfg("pre_extract_yolo_imgsz", 2048) or 2048)
    conf = float(_cfg("pre_extract_yolo_conf", 0.50) or 0.50)
    if log_callback:
        log_callback(
            f"[pre-extract] loading detector {os.path.basename(model_path)} "
            f"on {device}, imgsz={imgsz}, conf={conf:.2f}"
        )
    return Yolo11SegmentationModel(model_path, device=device, imgsz=imgsz, fp16=fp16, conf=conf)


def _get_detector(log_callback=None):
    global _DETECTOR
    with _DETECTOR_LOCK:
        if _DETECTOR is None:
            _DETECTOR = _build_detector(log_callback)
        return _DETECTOR


def release_detector(log_callback=None) -> None:
    """Release the cached YOLO detector and return its VRAM to CUDA."""
    global _DETECTOR
    with _DETECTOR_LOCK:
        if _DETECTOR is None:
            return
        _DETECTOR = None
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
    batch_size = max(1, int(_cfg("pre_extract_yolo_batch", 4) or 4))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video for pre-extract scan: {video_path}")
    detector = _get_detector(log_callback)

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
        preprocessed = detector.preprocess(batch_frames)
        results = detector.inference_and_postprocess(preprocessed, batch_frames)
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

    scale_max = max(256, int(_cfg("source_scan_scale_max_px", 2048) or 2048))
    src_w = max(2, int(meta.width or 0))
    src_h = max(2, int(meta.height or 0))
    if src_w <= 2 or src_h <= 2:
        raise RuntimeError(f"cannot determine source size for keyframe scan: {video_path}")
    ratio = min(1.0, float(scale_max) / float(max(src_w, src_h)))
    out_w = max(2, int(round(src_w * ratio)))
    out_h = max(2, int(round(src_h * ratio)))
    out_w -= out_w % 2
    out_h -= out_h % 2
    frame_bytes = out_w * out_h * 3

    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg,
        "-hide_banner", "-loglevel", "error",
        "-skip_frame", "nokey",
        "-i", str(video_path),
        "-an", "-sn",
        "-vf", f"scale={out_w}:{out_h}",
        "-vsync", "0",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]
    if log_callback:
        log_callback(
            f"[source-scan] fast keyframe scan: {len(keyframes)} keyframes, "
            f"{src_w}x{src_h} -> {out_w}x{out_h}"
        )
        log_callback(f"Executing: {' '.join(cmd)}")

    detector = _get_detector(log_callback)
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
    batch_size = max(1, int(_cfg("pre_extract_yolo_batch", 4) or 4))
    sampled = 0
    t0 = time.perf_counter()
    last_log = 0.0

    def _flush_batch():
        nonlocal batch_frames, batch_times, batch_indices
        if not batch_frames:
            return
        preprocessed = detector.preprocess(batch_frames)
        results = detector.inference_and_postprocess(preprocessed, batch_frames)
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
    return hits, meta, debug_records


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
    dec = PyNvThreadedSerialDecoder(Path(video_path), bit_depth=bd)
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
        batch_size = max(1, int(_cfg("pre_extract_yolo_batch", 4) or 4))
        detector = _get_detector(log_callback)

        hits: list[dict] = []
        debug_records: list[dict] = []
        batch_frames = []
        batch_times = []
        batch_indices = []
        sampled = 0
        t0 = time.perf_counter()
        last_log = 0.0

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
            nonlocal batch_frames, batch_times, batch_indices
            if not batch_frames:
                return
            preprocessed = detector.preprocess(batch_frames)
            results = detector.inference_and_postprocess(preprocessed, batch_frames)
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

        if log_callback:
            suffix = " + fisheye" if to_fisheye else ""
            log_callback(
                f"[pre-extract] GPU scan transform: crop={crop_mode}{suffix}, "
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
            batch_frames.append(_cupy_to_torch_bgr(y_plane, uv_plane, bit_depth=bd))
            batch_times.append(float(frame_idx) / fps)
            batch_indices.append(frame_idx)
            sampled += 1
            if len(batch_frames) >= batch_size:
                _flush_batch()
            now = time.perf_counter()
            if log_callback and now - last_log >= 5.0:
                last_log = now
                pct = 100.0 * min(frame_idx + 1, total_frames) / max(1, total_frames)
                elapsed = max(0.001, now - t0)
                log_callback(f"[pre-extract] scanned {sampled} GPU samples ({pct:.1f}%) at {sampled / elapsed:.1f} samples/s")
        _flush_batch()
        if log_callback:
            log_callback(f"[pre-extract] detector hits: {len(hits)} transformed sampled frames")
        return hits, out_meta, debug_records
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
        else:
            merged.append(group)
    return merged


def _aggregate_hits(hits: list[dict], meta: probe.VideoMetadata) -> list[MosaicSegment]:
    if not hits:
        return []
    duration = meta.duration or (meta.nb_frames / meta.source_fps if meta.nb_frames and meta.source_fps else 0.0)
    merge_gap_s = float(_cfg("pre_extract_merge_gap_s", 1.5) or 1.5)
    min_gap_s = float(_cfg("pre_extract_min_gap_s", 2.0) or 2.0)
    pad_s = float(_cfg("pre_extract_head_tail_pad_s", 2.0) or 2.0)
    min_segment_s = float(_cfg("pre_extract_min_segment_s", 1.5) or 1.5)

    hits = sorted(hits, key=lambda h: float(h["t"]))
    groups: list[dict] = []
    current = None
    for hit in hits:
        t = float(hit["t"])
        if current is None or t - current["last_hit"] > merge_gap_s:
            if current is not None:
                groups.append(current)
            current = {"start": t, "end": t, "last_hit": t, "boxes": list(hit["boxes"])}
        else:
            current["end"] = t
            current["last_hit"] = t
            current["boxes"].extend(hit["boxes"])
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
        boxes = _filter_spatial_outliers(group["boxes"], meta.width, meta.height)
        for cluster_boxes in _spatial_cluster(boxes, meta.width, meta.height):
            x, y, w, h, conf = _expanded_rect(cluster_boxes, meta.width, meta.height)
            if w <= 0 or h <= 0:
                continue
            segments.append(MosaicSegment(
                seg_id=len(segments),
                start_s=float(group["start"]),
                end_s=float(group["end"]),
                start_s_kf=float(group["start"]),
                end_s_kf=float(group["end"]),
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
                "the segmentation mask; accepted_boxes_xyxy are the boxes used by pre-extract."
            ),
        }
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def scan_segments(video_path: str | Path, log_callback=None, cancel_token=None,
                  scan_strategy: str | None = None,
                  min_conf: float | None = None) -> list[MosaicSegment]:
    strategy = str(scan_strategy or "normal").lower()
    if strategy in {"keyframes", "keyframe", "source_keyframes"}:
        hits, meta, debug_records = _scan_hits_keyframes_lowres(video_path, log_callback=log_callback, cancel_token=cancel_token, min_conf=min_conf)
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
    return segments


def scan_segments_gpu_transform(video_path: str | Path, *, crop_mode: str,
                                to_fisheye: bool = False,
                                log_callback=None, cancel_token=None,
                                min_conf: float | None = None) -> list[MosaicSegment]:
    hits, meta, debug_records = _scan_hits_gpu_transform(
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
    segments = _aggregate_hits(hits, meta)
    if log_callback:
        covered = sum(s.duration_s for s in segments)
        dur = meta.duration or 0.0
        pct = (100.0 * covered / dur) if dur > 0 else 0.0
        log_callback(f"[pre-extract] aggregated {len(segments)} transformed segments, {covered:.1f}s ({pct:.1f}% of video)")
    return segments


def save_segments_json(segments: list[MosaicSegment], path: str | Path, source: str | Path | None = None) -> None:
    data = {
        "source": str(source) if source else "",
        "segments": [seg.to_dict() for seg in segments],
    }
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_segments_json(path: str | Path) -> list[MosaicSegment]:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return [MosaicSegment.from_dict(item) for item in data.get("segments", [])]
