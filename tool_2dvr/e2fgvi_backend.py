from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class E2FGVIBackendError(RuntimeError):
    pass


def resolve_int_env(name: str, default: int, minimum: int | None = None) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if minimum is not None:
        value = max(minimum, value)
    return value


def default_checkpoint(project_root: Path) -> Path:
    return project_root / "models" / "E2FGVI" / "E2FGVI-HQ-CVPR22.pth"


def resolve_checkpoint(project_root: Path) -> Path:
    raw = os.environ.get("TOOL_2DVR_E2FGVI_CKPT", "").strip()
    path = Path(raw) if raw else default_checkpoint(project_root)
    if not path.is_absolute():
        path = project_root / path
    return path


def _target_size(width: int, height: int, max_width: int) -> tuple[int, int]:
    if max_width <= 0 or width <= max_width:
        return width, height
    scale = max_width / float(width)
    out_w = max(64, int(round(width * scale)))
    out_h = max(64, int(round(height * scale)))
    return out_w, out_h


def _resize_rgb_frames_masked(
    frames: np.ndarray,
    masks: np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:
    # Mask-aware INTER_AREA: each output pixel is the weighted average of only
    # the *valid* (non-hole) source pixels, so the black zeros at hole
    # locations no longer bleed dark fringes into the low-res context that
    # E2FGVI conditions on. Two cv2.resize calls per frame instead of an
    # O(W) Python iterative seed-fill — the previous approach dominated CPU.
    import cv2

    width, height = size
    n, src_h, src_w = frames.shape[:3]
    if (src_w, src_h) == (width, height):
        return frames.astype(np.uint8, copy=False)
    holes = np.asarray(masks, dtype=bool)
    valid = (~holes).astype(np.float32)
    weighted = frames.astype(np.float32) * valid[..., None]
    out = np.empty((n, height, width, 3), dtype=np.uint8)
    for i in range(n):
        ds_f = cv2.resize(weighted[i], (width, height), interpolation=cv2.INTER_AREA)
        ds_v = cv2.resize(valid[i], (width, height), interpolation=cv2.INTER_AREA)
        np.maximum(ds_v, 1.0e-3, out=ds_v)
        out[i] = np.clip(ds_f / ds_v[..., None], 0, 255).astype(np.uint8)
    return out


def _resize_masks(masks: np.ndarray, size: tuple[int, int], dilation: int) -> np.ndarray:
    import cv2

    width, height = size
    out = []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for mask in masks:
        src = mask.astype(np.float32, copy=False)
        src_h, src_w = src.shape[:2]
        if (src_w, src_h) == (width, height):
            m = src
        else:
            interpolation = cv2.INTER_AREA if width < src_w or height < src_h else cv2.INTER_NEAREST
            m = cv2.resize(src, (width, height), interpolation=interpolation)
        # Thin forward-warp holes can be only 1px wide. Area resize preserves
        # fractional coverage while nearest-neighbor downscale can drop them.
        m = (m > 1.0e-6).astype(np.uint8)
        if dilation > 0:
            m = cv2.dilate(m, kernel, iterations=dilation)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
        out.append(m > 0)
    return np.stack(out, axis=0)


def _alpha_from_mask(mask: np.ndarray) -> np.ndarray:
    # Keep the mask core opaque even for 1px forward-warp holes, then apply only
    # a light edge feather so the inpainted fill does not hard-step into source.
    import cv2

    core = mask.astype(np.float32)
    if not np.any(core):
        return core[:, :, None]
    feather = cv2.GaussianBlur(core, (0, 0), sigmaX=0.6, sigmaY=0.6)
    alpha = np.maximum(core, feather)
    return np.clip(alpha, 0.0, 1.0)[:, :, None]


def _cleanup_residual_black_fill(fill_rgb: np.ndarray, mask: np.ndarray, threshold: int) -> np.ndarray:
    residual = np.asarray(mask, dtype=bool) & np.all(fill_rgb <= int(threshold), axis=2)
    if not np.any(residual):
        return fill_rgb
    import cv2

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    residual_u8 = cv2.dilate(residual.astype(np.uint8), kernel, iterations=1) * 255
    cleaned = cv2.inpaint(fill_rgb, residual_u8, 3, cv2.INPAINT_TELEA)
    out = fill_rgb.copy()
    take = residual_u8 > 0
    out[take] = cleaned[take]
    return out


class E2FGVIInpainter:
    def __init__(
        self,
        project_root: Path,
        checkpoint: str | Path | None = None,
        max_width: int | None = None,
        mask_dilation: int | None = None,
        neighbor_stride: int | None = None,
        ref_step: int | None = None,
        log_callback=None,
    ):
        self.project_root = Path(project_root)
        self.checkpoint = Path(checkpoint) if checkpoint else resolve_checkpoint(self.project_root)
        if not self.checkpoint.is_absolute():
            self.checkpoint = self.project_root / self.checkpoint
        self.max_width = max_width if max_width is not None else resolve_int_env("TOOL_2DVR_E2FGVI_MAX_WIDTH", 432, 128)
        self.mask_dilation = mask_dilation if mask_dilation is not None else resolve_int_env("TOOL_2DVR_E2FGVI_MASK_DILATE", 4, 0)
        self.neighbor_stride = neighbor_stride if neighbor_stride is not None else resolve_int_env("TOOL_2DVR_E2FGVI_NEIGHBOR_STRIDE", 5, 1)
        self.ref_step = ref_step if ref_step is not None else resolve_int_env("TOOL_2DVR_E2FGVI_REF_STEP", 10, 1)
        self.black_threshold = resolve_int_env("TOOL_2DVR_E2FGVI_BLACK_THRESH", 40, 0)
        self.log_callback = log_callback
        self._model = None
        self._device = None

    def _log(self, message: str) -> None:
        if self.log_callback:
            self.log_callback(message)

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not self.checkpoint.exists():
            raise E2FGVIBackendError(f"E2FGVI checkpoint not found: {self.checkpoint}")
        try:
            import torch
            from ._vendor.e2fgvi.model.e2fgvi_hq import InpaintGenerator
        except Exception as exc:
            raise E2FGVIBackendError(f"E2FGVI dependencies are unavailable: {exc}") from exc

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            self._log("[2dvr] E2FGVI is running on CPU; this will be very slow.")
        self._log(f"[2dvr] Loading E2FGVI HQ from: {self.checkpoint}")
        model = InpaintGenerator(init_weights=False).to(device)
        data = torch.load(str(self.checkpoint), map_location=device)
        model.load_state_dict(data, strict=True)
        model.eval()
        self._model = model
        self._device = device
        self._log(
            f"[2dvr] E2FGVI device: {device}, max_width={self.max_width}, "
            f"mask_dilate={self.mask_dilation}, neighbor_stride={self.neighbor_stride}, "
            f"black_thresh={self.black_threshold}"
        )
        return model

    def _ref_indices(self, frame_index: int, neighbor_ids: list[int], length: int) -> list[int]:
        return [i for i in range(0, length, self.ref_step) if i not in neighbor_ids]

    def inpaint(self, frames_rgb: np.ndarray, masks: np.ndarray) -> np.ndarray:
        return self._run_single(frames_rgb, masks)

    def inpaint_pair(
        self,
        left_rgb: np.ndarray,
        right_rgb: np.ndarray,
        left_masks: np.ndarray,
        right_masks: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        # E2FGVI's temporal focal transformer caches per-batch masks/windows
        # that don't broadcast cleanly when B>1, so we run left and right as
        # two sequential B=1 passes but share the same model load, autocast
        # session, and GPU compositing path. Net win still comes from
        # mask-aware downscale + FP16 + GPU scatter.
        left_out = self._run_single(left_rgb, left_masks)
        right_out = self._run_single(right_rgb, right_masks)
        return left_out, right_out

    def _run_single(self, frames_rgb: np.ndarray, masks: np.ndarray) -> np.ndarray:
        if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
            raise E2FGVIBackendError(f"Expected RGB frames in BHWC, got {frames_rgb.shape}")
        if masks.shape != frames_rgb.shape[:3]:
            raise E2FGVIBackendError(f"Mask shape {masks.shape} does not match frames {frames_rgb.shape[:3]}")
        if not np.any(masks):
            return frames_rgb.copy()

        import cv2
        import torch

        model = self._load_model()
        assert self._device is not None
        device = self._device
        use_amp = device.type == "cuda"

        original = np.ascontiguousarray(frames_rgb.astype(np.uint8, copy=False))
        length, height, width, _ = original.shape
        work_w, work_h = _target_size(width, height, self.max_width)

        # Mask-aware downscale eliminates the dark-bleed that motivated the
        # earlier Python seed-fill, without the per-frame iterative cost.
        work_frames = _resize_rgb_frames_masked(original, masks, (work_w, work_h))
        work_masks = _resize_masks(masks, (work_w, work_h), self.mask_dilation)

        imgs = torch.from_numpy(work_frames).to(device, non_blocking=True)
        imgs = imgs.permute(0, 3, 1, 2).float().div_(255.0).mul_(2.0).sub_(1.0).unsqueeze(0)
        mask_t = torch.from_numpy(work_masks.astype(np.float32)).to(device, non_blocking=True)
        mask_t = mask_t.unsqueeze(0).unsqueeze(2)  # [1, T, 1, H, W]
        bg = torch.from_numpy(work_frames).to(device).float()  # [T, H, W, 3]

        comp_sum = torch.zeros((length, work_h, work_w, 3), dtype=torch.float32, device=device)
        counts = torch.zeros((length,), dtype=torch.float32, device=device)

        mod_size_h = 60
        mod_size_w = 108
        h_pad = (mod_size_h - work_h % mod_size_h) % mod_size_h
        w_pad = (mod_size_w - work_w % mod_size_w) % mod_size_w

        with torch.inference_mode():
            for f in range(0, length, self.neighbor_stride):
                neighbor_ids = [
                    i for i in range(max(0, f - self.neighbor_stride), min(length, f + self.neighbor_stride + 1))
                ]
                ref_ids = self._ref_indices(f, neighbor_ids, length)
                selected = neighbor_ids + ref_ids
                selected_imgs = imgs[:, selected]
                selected_masks = mask_t[:, selected]
                masked_imgs = selected_imgs * (1.0 - selected_masks)
                if h_pad:
                    masked_imgs = torch.cat([masked_imgs, torch.flip(masked_imgs, [3])], 3)[:, :, :, :work_h + h_pad, :]
                if w_pad:
                    masked_imgs = torch.cat([masked_imgs, torch.flip(masked_imgs, [4])], 4)[:, :, :, :, :work_w + w_pad]
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    pred, _ = model(masked_imgs, len(neighbor_ids))
                pred = pred.float()
                pred = pred[:, :, :work_h, :work_w]
                pred = pred.add(1.0).mul(0.5).clamp(0.0, 1.0)
                # pred: [T_total, 3, H, W]. Keep only neighbor slice.
                pred = pred[:len(neighbor_ids)].permute(0, 2, 3, 1).contiguous() * 255.0  # [T_n, H, W, 3]
                neighbor_idx = torch.tensor(neighbor_ids, device=device, dtype=torch.long)
                bg_sel = bg.index_select(0, neighbor_idx)  # [T_n, H, W, 3]
                mask_sel = mask_t[0, neighbor_ids].squeeze(1).unsqueeze(-1)  # [T_n, H, W, 1]
                comp = mask_sel * pred + (1.0 - mask_sel) * bg_sel
                comp_sum.index_add_(0, neighbor_idx, comp)
                counts.index_add_(0, neighbor_idx, torch.ones(len(neighbor_ids), device=device))

        counts_safe = counts.clamp(min=1.0).view(length, 1, 1, 1)
        smalls = (comp_sum / counts_safe).clamp(0, 255).to(torch.uint8).cpu().numpy()
        del comp_sum, counts, counts_safe, imgs, mask_t, bg

        output = original.copy()
        for i in range(length):
            if not np.any(masks[i]):
                continue
            small = smalls[i]
            if (work_w, work_h) != (width, height):
                fill = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
            else:
                fill = small
            mask = _resize_masks(masks[i:i + 1], (width, height), self.mask_dilation)[0]
            guard_mask = _resize_masks(masks[i:i + 1], (width, height), self.mask_dilation + 3)[0]
            near_black = np.all(output[i] <= self.black_threshold, axis=2)
            mask = mask | (near_black & guard_mask)
            fill = _cleanup_residual_black_fill(fill, mask, self.black_threshold)
            alpha = _alpha_from_mask(mask)
            blended = output[i].astype(np.float32) * (1.0 - alpha) + fill.astype(np.float32) * alpha
            output[i] = np.clip(blended, 0, 255).astype(np.uint8)
        return output
