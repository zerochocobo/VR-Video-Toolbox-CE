"""Fisheye delta write-back for the native streaming pipeline.

The fisheye path used to remap the whole frame twice (hequirect -> fisheye for
detection/restoration, then fisheye -> hequirect for encoding), so every pixel
paid two bilinear resamples even when no mosaic was restored there.  Instead
the decoded hequirect frame stays the output base and only the restoration
delta is inverse-projected onto it:

    output = source_heq + inverse_remap(restored_fisheye - source_fisheye)

Untouched pixels cancel to a zero delta and keep the source projection exactly
(only the NV12 round-trip shared with the non-fisheye path remains); restored
regions receive the same two remaps as before.  The (source_heq,
source_fisheye) reference pair comes from a second NVDEC decode on the encode
side, keeping VRAM flat instead of buffering hundreds of full frames while the
restorer pipeline drains.

The inverse sampling grid is derived from the same fisheye2heq v360 LUT that
the legacy output remap used, so restored content lands on identical
coordinates; the delta is resampled with torch grid_sample (align_corners=True
matches the LUT's ffmpeg ``scale()`` pixel convention).
"""

from __future__ import annotations

_GRID_CACHE: dict = {}


def enabled() -> bool:
    """Config gate: app key ``native_fisheye_delta``, default on."""
    try:
        from utils import app_config
        return app_config.get_bool("native_fisheye_delta", True)
    except Exception:
        return True


def clear_cache() -> None:
    _GRID_CACHE.clear()


def inverse_grid(torch_module, device, eye_w: int, eye_h: int, fov: float = 180.0):
    """Per-eye grid_sample grid mapping hequirect output pixels to fisheye coords."""
    eye_w = int(eye_w)
    eye_h = int(eye_h)
    key = (eye_w, eye_h, round(float(fov), 4), str(device))
    grid = _GRID_CACHE.get(key)
    if grid is not None:
        return grid

    import cupy as cp
    from gpu_engine import v360_lut

    lut = cp.ascontiguousarray(v360_lut.make_lut("fisheye2heq", eye_w, eye_h, float(fov)))
    cp.cuda.get_current_stream().synchronize()
    try:
        lut_t = torch_module.utils.dlpack.from_dlpack(lut)
    except TypeError:
        lut_t = torch_module.utils.dlpack.from_dlpack(lut.toDlpack())
    lut_t = lut_t.to(device=device, dtype=torch_module.float32)
    # LUT holds absolute source pixel coordinates following ffmpeg's
    # scale(): [-1,1] -> [0, size-1], which is grid_sample align_corners=True.
    gx = lut_t[..., 0] * (2.0 / max(1, eye_w - 1)) - 1.0
    gy = lut_t[..., 1] * (2.0 / max(1, eye_h - 1)) - 1.0
    grid = torch_module.stack((gx, gy), dim=-1).unsqueeze(0).contiguous()
    _GRID_CACHE[key] = grid
    return grid


def apply_delta_eye(torch_module, source_heq_eye, source_fish_eye, restored_fish_eye, grid):
    """Write one eye's restoration delta back onto the source projection."""
    t = torch_module
    if t.equal(restored_fish_eye, source_fish_eye):
        return source_heq_eye
    delta = restored_fish_eye.to(t.float32) - source_fish_eye.to(t.float32)
    delta = delta.permute(2, 0, 1).unsqueeze(0).contiguous()
    mapped = t.nn.functional.grid_sample(
        delta, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    out = source_heq_eye.to(t.float32).add_(mapped[0].permute(1, 2, 0))
    return out.round_().clamp_(0, 255).to(t.uint8)


def apply_delta_frame(torch_module, device, source_heq, source_fish, restored_fish,
                      *, sbs: bool, fov: float = 180.0):
    """Delta write-back for a full (H,W,3) BGR frame; SBS frames split per eye."""
    t = torch_module
    if source_heq.shape != source_fish.shape or restored_fish.shape != source_fish.shape:
        raise RuntimeError(
            "fisheye delta frames disagree in shape: "
            f"heq={tuple(source_heq.shape)} fish={tuple(source_fish.shape)} "
            f"restored={tuple(restored_fish.shape)}"
        )
    if t.equal(restored_fish, source_fish):
        return source_heq
    height, width = int(source_heq.shape[0]), int(source_heq.shape[1])
    eye_w = width // 2 if sbs else width
    grid = inverse_grid(t, device, eye_w, height, fov)
    if not sbs:
        return apply_delta_eye(t, source_heq, source_fish, restored_fish, grid)
    eyes = [
        apply_delta_eye(
            t,
            source_heq[:, eye_slice],
            source_fish[:, eye_slice],
            restored_fish[:, eye_slice],
            grid,
        )
        for eye_slice in (slice(0, eye_w), slice(eye_w, eye_w * 2))
    ]
    return t.cat(eyes, dim=1)


def next_reference(reference_iter):
    """Pull the next (source_heq, source_fish) pair; a short reference decode is a bug."""
    try:
        return next(reference_iter)
    except StopIteration:
        raise RuntimeError(
            "fisheye delta reference decode ended before the restored stream"
        ) from None
