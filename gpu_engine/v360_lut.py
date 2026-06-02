"""v360 projection LUT generation with CuPy vectorization, matching ffmpeg vf_v360.c math.

Key ffmpeg helpers (libavfilter/vf_v360.c):
    rescale(i, s) = (2*i + 1) / s - 1          # output pixel index -> [-1,1]
    scale(x, s)   = (0.5*x + 0.5) * (s - 1)     # [-1,1] -> input pixel coordinate
    Default interp = BILINEAR, default rotation is none, and default fisheye fov = 180 (flat_range=1).

Projection functions, matched closely to ffmpeg:
  hequirect_to_xyz: phi=rescale(i,w)*pi/2; theta=rescale(j,h)*pi/2
                    vec=(cosθ·sinφ, sinθ, cosθ·cosφ)
  xyz_to_hequirect: phi=atan2(x,z)/(pi/2); theta=asin(y)/(pi/2)
                    uf=scale(phi,w); vf=scale(theta,h)   # always clamp; do not zero invisible pixels
  fisheye_to_xyz:   uf=(fov/180)·rescale(i,w); vf=(fov/180)·rescale(j,h)
                    phi=atan2(vf,uf); theta=pi/2·(1-hypot(uf,vf))
                    vec=(cosθ·cosφ, cosθ·sinφ, sinθ)
  xyz_to_fisheye:   h=hypot(x,y); phi=atan2(h,z)/pi
                    uf=x/h·phi/(fov/180); vf=y/h·phi/(fov/180)
                    visible = |uf|<0.5 and |vf|<0.5
                    uf=scale(uf·2,w); vf=scale(vf·2,h)   # invisible -> sample (0,0)

LUT convention: return a float32 CuPy array shaped (out_h, out_w, 2), where
[...,0] is source x and [...,1] is source y in input pixel coordinates for the
corresponding plane. Each plane gets its own LUT: full-resolution Y and
half-resolution chroma, matching ffmpeg's per-plane remap behavior.
"""
from __future__ import annotations

import math

_PI = math.pi
_PI_2 = math.pi / 2.0

# LUT cache: key=(kind, w, h, round(fov,4)).
_cache: dict = {}


def _rescale(idx, s):
    import cupy as cp
    return (2.0 * idx + 1.0) / s - 1.0


def _scale(x, s):
    return (0.5 * x + 0.5) * (s - 1.0)


def _meshgrid(w: int, h: int):
    import cupy as cp
    i = cp.arange(w, dtype=cp.float32)
    j = cp.arange(h, dtype=cp.float32)
    I, J = cp.meshgrid(i, j)  # shape (h, w)
    return I, J


def make_heq_to_fisheye_lut(w: int, h: int, fov: float = 180.0):
    """Output=fisheye(w×h), input=hequirect(w×h)."""
    import cupy as cp

    key = ("heq2fisheye", w, h, round(fov, 4))
    cached = _cache.get(key)
    if cached is not None:
        return cached

    I, J = _meshgrid(w, h)
    fr = fov / 180.0
    uf = fr * _rescale(I, w)
    vf = fr * _rescale(J, h)
    phi = cp.arctan2(vf, uf)
    theta = _PI_2 * (1.0 - cp.hypot(uf, vf))
    ct = cp.cos(theta); st = cp.sin(theta)
    cphi = cp.cos(phi); sphi = cp.sin(phi)
    vx = ct * cphi
    vy = ct * sphi
    vz = st
    # xyz_to_hequirect
    phi2 = cp.arctan2(vx, vz) / _PI_2
    theta2 = cp.arcsin(cp.clip(vy, -1.0, 1.0)) / _PI_2
    src_x = _scale(phi2, w)
    src_y = _scale(theta2, h)
    lut = cp.stack([src_x, src_y], axis=-1).astype(cp.float32)
    _cache[key] = lut
    return lut


def make_fisheye_to_heq_lut(w: int, h: int, fov: float = 180.0):
    """Output=hequirect(w×h), input=fisheye(w×h)."""
    import cupy as cp

    key = ("fisheye2heq", w, h, round(fov, 4))
    cached = _cache.get(key)
    if cached is not None:
        return cached

    I, J = _meshgrid(w, h)
    phi = _rescale(I, w) * _PI_2
    theta = _rescale(J, h) * _PI_2
    ct = cp.cos(theta); st = cp.sin(theta)
    cphi = cp.cos(phi); sphi = cp.sin(phi)
    vx = ct * sphi
    vy = st
    vz = ct * cphi
    # xyz_to_fisheye
    hh = cp.hypot(vx, vy)
    lh = cp.where(hh > 0.0, hh, 1.0)
    phi2 = cp.arctan2(hh, vz) / _PI
    fr = fov / 180.0
    uf = vx / lh * phi2 / fr
    vf = vy / lh * phi2 / fr
    visible = (uf > -0.5) & (uf < 0.5) & (vf > -0.5) & (vf < 0.5)
    src_x = _scale(uf * 2.0, w)
    src_y = _scale(vf * 2.0, h)
    src_x = cp.where(visible, src_x, 0.0)
    src_y = cp.where(visible, src_y, 0.0)
    lut = cp.stack([src_x, src_y], axis=-1).astype(cp.float32)
    _cache[key] = lut
    return lut


# --- Quaternion rotation, matching ffmpeg calculate_rotation / rotate. ---

def _qmul(a, b):
    return (
        a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
        a[1]*b[0] + a[0]*b[1] + a[2]*b[3] - a[3]*b[2],
        a[2]*b[0] + a[0]*b[2] + a[3]*b[1] - a[1]*b[3],
        a[3]*b[0] + a[0]*b[3] + a[1]*b[2] - a[2]*b[1],
    )


_RORDER_IDX = {"y": 0, "p": 1, "r": 2}


def _rotation_quaternion(yaw: float, pitch: float, roll: float, rorder: str = "ypr"):
    """Return rotation quaternion q0, matching ffmpeg calculate_rotation."""
    yr = yaw * _PI / 180.0
    pr = pitch * _PI / 180.0
    rr = roll * _PI / 180.0
    m = [
        (math.cos(yr * 0.5), 0.0, math.sin(yr * 0.5), 0.0),          # m[0] yaw
        (math.cos(pr * 0.5), math.sin(pr * 0.5), 0.0, 0.0),          # m[1] pitch
        (math.cos(rr * 0.5), 0.0, 0.0, math.sin(rr * 0.5)),          # m[2] roll
    ]
    order = [_RORDER_IDX[c] for c in rorder]
    q = (1.0, 0.0, 0.0, 0.0)
    for k in order:
        q = _qmul(q, m[k])
    return q


def make_heq_to_flat_lut(out_w: int, out_h: int, in_w: int, in_h: int,
                         yaw: float, pitch: float, d_fov: float,
                         roll: float = 0.0, rorder: str = "ypr"):
    """Output=flat(out_w×out_h), input=hequirect(in_w×in_h), with yaw/pitch/d_fov.

    Matches ffmpeg: flat_to_xyz -> normalize -> rotate(quaternion) -> xyz_to_hequirect.
    d_fov is converted to h_fov/v_fov through fov_from_dfov(FLAT) using the output size.
    """
    import cupy as cp

    key = ("heq2flat", out_w, out_h, in_w, in_h,
           round(yaw, 4), round(pitch, 4), round(d_fov, 4), round(roll, 4), rorder)
    cached = _cache.get(key)
    if cached is not None:
        return cached

    # d_fov(FLAT) -> h_fov / v_fov using the output flat plane size.
    da = math.tan(0.5 * min(d_fov, 359.0) * _PI / 180.0)
    diag = math.hypot(out_w, out_h)
    h_fov = math.atan2(da * out_w, diag) * 360.0 / _PI
    v_fov = math.atan2(da * out_h, diag) * 360.0 / _PI
    if h_fov < 0.0:
        h_fov += 360.0
    if v_fov < 0.0:
        v_fov += 360.0
    fr0 = math.tan(0.5 * h_fov * _PI / 180.0)
    fr1 = math.tan(0.5 * v_fov * _PI / 180.0)

    I, J = _meshgrid(out_w, out_h)
    lx = fr0 * _rescale(I, out_w)
    ly = fr1 * _rescale(J, out_h)
    lz = cp.ones_like(lx)
    # normalize
    n = cp.sqrt(lx * lx + ly * ly + lz * lz)
    vx = lx / n; vy = ly / n; vz = lz / n

    # rotate: v' = q0 ⊗ (0,v) ⊗ conj(q0)
    a0, a1, a2, a3 = _rotation_quaternion(yaw, pitch, roll, rorder)
    c0, c1, c2, c3 = a0, -a1, -a2, -a3
    t0 = -a1 * vx - a2 * vy - a3 * vz
    t1 = a0 * vx + a2 * vz - a3 * vy
    t2 = a0 * vy + a3 * vx - a1 * vz
    t3 = a0 * vz + a1 * vy - a2 * vx
    rx = t1 * c0 + t0 * c1 + t2 * c3 - t3 * c2
    ry = t2 * c0 + t0 * c2 + t3 * c1 - t1 * c3
    rz = t3 * c0 + t0 * c3 + t1 * c2 - t2 * c1

    # xyz_to_hequirect
    phi = cp.arctan2(rx, rz) / _PI_2
    theta = cp.arcsin(cp.clip(ry, -1.0, 1.0)) / _PI_2
    src_x = _scale(phi, in_w)
    src_y = _scale(theta, in_h)
    lut = cp.stack([src_x, src_y], axis=-1).astype(cp.float32)
    _cache[key] = lut
    return lut


def make_lut(kind: str, w: int, h: int, fov: float = 180.0):
    """Dispatch LUT construction by name. kind in {heq2fisheye, fisheye2heq}."""
    if kind == "heq2fisheye":
        return make_heq_to_fisheye_lut(w, h, fov)
    if kind == "fisheye2heq":
        return make_fisheye_to_heq_lut(w, h, fov)
    raise ValueError(f"unknown LUT kind: {kind}")


def clear_cache() -> None:
    _cache.clear()
