"""CuPy RawKernel helpers for bilinear LUT sampling, crop, hstack, and vstack on NV12 / P010 planes.

Sampling matches ffmpeg v360 default BILINEAR behavior, a 2x2 clamp-to-edge bilinear sample:
    ui=floor(x); vi=floor(y); du=x-ui; dv=y-vi
    The four neighbors are clamped to [0,W-1]×[0,H-1] with weights such as (1-du)(1-dv).

Plane layout, identical structure for NV12 8-bit and P010 10-bit:
  Y  : (H, W)           single component
  UV : (H/2, W/2, 2)    interleaved Cb/Cr; chroma uses a half-resolution LUT to match ffmpeg per-plane remap
"""
from __future__ import annotations

_kernels: dict = {}


def _nvrtc_arch(cp) -> str:
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    return f"{int(props.get('major', 0))}{int(props.get('minor', 0))}"


def _arch_fallbacks(primary: str) -> list[str]:
    try:
        current = int(primary)
    except Exception:
        current = 0
    candidates = [primary, "120", "100", "90", "89", "86", "80", "75", "70"]
    out: list[str] = []
    for arch in candidates:
        if arch in out:
            continue
        try:
            if current and int(arch) > current:
                continue
        except Exception:
            pass
        out.append(arch)
    return out


def _compile_raw_kernel(code: str, name: str):
    """Compile with cp.RawKernel, verified stable on Blackwell sm_120 with the CUDA 12.8 stack.

    Historical lesson: this was once changed to NVRTC -> PTX -> RawModule(path=...)
    to avoid RawKernel, but complex kernels with indirect indexing could hang
    after launch on sm_120. Once a kernel faulted, the whole CUDA context became
    contaminated and later launches, including normal cp.RawKernel launches, also
    hung. That created the false impression that cp.RawKernel itself was broken.
    In a fresh process, cp.RawKernel, including sample1 indirect indexing, works
    normally. _nvrtc_arch/_arch_fallbacks are kept only as historical reference
    and are no longer used.
    """
    import cupy as cp
    return cp.RawKernel(code, name)


def _sample1_kernel(ctype: str):
    """Bilinear sampling kernel for a single-component plane (Y)."""
    key = ("sample1", ctype)
    k = _kernels.get(key)
    if k is not None:
        return k
    code = r'''
extern "C" __global__
void sample1(const CTYPE* src, const float* lut,
             CTYPE* dst, int in_w, int in_h, int out_w, int out_h)
{
    int oi = blockDim.x * blockIdx.x + threadIdx.x;
    int oj = blockDim.y * blockIdx.y + threadIdx.y;
    if (oi >= out_w || oj >= out_h) return;
    int idx = oj * out_w + oi;
    float x = lut[idx * 2 + 0];
    float y = lut[idx * 2 + 1];
    int ui = (int)floorf(x);
    int vi = (int)floorf(y);
    float du = x - ui;
    float dv = y - vi;
    int u0 = min(max(ui,     0), in_w - 1);
    int u1 = min(max(ui + 1, 0), in_w - 1);
    int v0 = min(max(vi,     0), in_h - 1);
    int v1 = min(max(vi + 1, 0), in_h - 1);
    float p00 = src[v0 * in_w + u0];
    float p01 = src[v0 * in_w + u1];
    float p10 = src[v1 * in_w + u0];
    float p11 = src[v1 * in_w + u1];
    float top = p00 + (p01 - p00) * du;
    float bot = p10 + (p11 - p10) * du;
    float val = top + (bot - top) * dv;
    dst[idx] = (CTYPE)(val + 0.5f);
}
'''.replace("CTYPE", ctype)
    k = _compile_raw_kernel(code, "sample1")
    _kernels[key] = k
    return k


def _sample2_kernel(ctype: str):
    """Bilinear sampling kernel for a two-component interleaved plane (UV)."""
    key = ("sample2", ctype)
    k = _kernels.get(key)
    if k is not None:
        return k
    code = r'''
extern "C" __global__
void sample2(const CTYPE* src, const float* lut,
             CTYPE* dst, int in_w, int in_h, int out_w, int out_h)
{
    int oi = blockDim.x * blockIdx.x + threadIdx.x;
    int oj = blockDim.y * blockIdx.y + threadIdx.y;
    if (oi >= out_w || oj >= out_h) return;
    int idx = oj * out_w + oi;
    float x = lut[idx * 2 + 0];
    float y = lut[idx * 2 + 1];
    int ui = (int)floorf(x);
    int vi = (int)floorf(y);
    float du = x - ui;
    float dv = y - vi;
    int u0 = min(max(ui,     0), in_w - 1);
    int u1 = min(max(ui + 1, 0), in_w - 1);
    int v0 = min(max(vi,     0), in_h - 1);
    int v1 = min(max(vi + 1, 0), in_h - 1);
    #pragma unroll
    for (int c = 0; c < 2; c++) {
        float p00 = src[(v0 * in_w + u0) * 2 + c];
        float p01 = src[(v0 * in_w + u1) * 2 + c];
        float p10 = src[(v1 * in_w + u0) * 2 + c];
        float p11 = src[(v1 * in_w + u1) * 2 + c];
        float top = p00 + (p01 - p00) * du;
        float bot = p10 + (p11 - p10) * du;
        float val = top + (bot - top) * dv;
        dst[idx * 2 + c] = (CTYPE)(val + 0.5f);
    }
}
'''.replace("CTYPE", ctype)
    k = _compile_raw_kernel(code, "sample2")
    _kernels[key] = k
    return k


def _nv_to_bgr_kernel(ctype: str):
    """NV12/P010 -> BGR uint8 kernel."""
    key = ("nv_to_bgr", ctype)
    k = _kernels.get(key)
    if k is not None:
        return k
    code = r'''
extern "C" __global__
void nv_to_bgr(const CTYPE* y, const CTYPE* uv, unsigned char* bgr,
               int width, int height, float norm)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int yy = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= width || yy >= height) return;

    int pix = yy * width + x;
    int uv_idx = ((yy >> 1) * (width >> 1) + (x >> 1)) * 2;
    float yf = ((float)y[pix]) * norm;
    float cb = ((float)uv[uv_idx + 0]) * norm;
    float cr = ((float)uv[uv_idx + 1]) * norm;

    float c = yf - 16.0f;
    float d = cb - 128.0f;
    float e = cr - 128.0f;
    float rf = 1.16438356f * c + 1.79274107f * e;
    float gf = 1.16438356f * c - 0.21324861f * d - 0.53290933f * e;
    float bf = 1.16438356f * c + 2.11240179f * d;

    rf = fminf(fmaxf(rf, 0.0f), 255.0f);
    gf = fminf(fmaxf(gf, 0.0f), 255.0f);
    bf = fminf(fmaxf(bf, 0.0f), 255.0f);

    int out = pix * 3;
    bgr[out + 0] = (unsigned char)(bf + 0.5f);
    bgr[out + 1] = (unsigned char)(gf + 0.5f);
    bgr[out + 2] = (unsigned char)(rf + 0.5f);
}
'''.replace("CTYPE", ctype)
    k = _compile_raw_kernel(code, "nv_to_bgr")
    _kernels[key] = k
    return k


def _bgr_to_nv12_y_kernel():
    key = ("bgr_to_nv12_y",)
    k = _kernels.get(key)
    if k is not None:
        return k
    code = r'''
extern "C" __global__
void bgr_to_nv12_y(const unsigned char* bgr, unsigned char* y,
                   int width, int height)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int yy = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= width || yy >= height) return;

    int pix = yy * width + x;
    int in = pix * 3;
    float b = (float)bgr[in + 0];
    float g = (float)bgr[in + 1];
    float r = (float)bgr[in + 2];
    float yf = 16.0f + 0.182586f * r + 0.614231f * g + 0.062007f * b;
    yf = fminf(fmaxf(yf, 0.0f), 255.0f);
    y[pix] = (unsigned char)(yf + 0.5f);
}
'''
    k = _compile_raw_kernel(code, "bgr_to_nv12_y")
    _kernels[key] = k
    return k


def _bgr_to_nv12_uv_kernel():
    key = ("bgr_to_nv12_uv",)
    k = _kernels.get(key)
    if k is not None:
        return k
    code = r'''
extern "C" __global__
void bgr_to_nv12_uv(const unsigned char* bgr, unsigned char* uv,
                    int width, int height)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int cw = width >> 1;
    int ch = height >> 1;
    if (x >= cw || y >= ch) return;

    int x0 = x << 1;
    int y0 = y << 1;
    float bsum = 0.0f;
    float gsum = 0.0f;
    float rsum = 0.0f;
    #pragma unroll
    for (int dy = 0; dy < 2; dy++) {
        #pragma unroll
        for (int dx = 0; dx < 2; dx++) {
            int pix = (y0 + dy) * width + (x0 + dx);
            int in = pix * 3;
            bsum += (float)bgr[in + 0];
            gsum += (float)bgr[in + 1];
            rsum += (float)bgr[in + 2];
        }
    }
    float b = bsum * 0.25f;
    float g = gsum * 0.25f;
    float r = rsum * 0.25f;
    float cb = 128.0f - 0.100644f * r - 0.338572f * g + 0.439216f * b;
    float cr = 128.0f + 0.439216f * r - 0.398942f * g - 0.040274f * b;
    cb = fminf(fmaxf(cb, 0.0f), 255.0f);
    cr = fminf(fmaxf(cr, 0.0f), 255.0f);
    int out = (y * cw + x) * 2;
    uv[out + 0] = (unsigned char)(cb + 0.5f);
    uv[out + 1] = (unsigned char)(cr + 0.5f);
}
'''
    k = _compile_raw_kernel(code, "bgr_to_nv12_uv")
    _kernels[key] = k
    return k


def _ctype_for(arr) -> str:
    import cupy as cp
    if arr.dtype == cp.uint8:
        return "unsigned char"
    if arr.dtype == cp.uint16:
        return "unsigned short"
    raise TypeError(f"unsupported plane dtype: {arr.dtype}")


def _launch(out_w: int, out_h: int):
    block = (16, 16, 1)
    grid = ((out_w + block[0] - 1) // block[0],
            (out_h + block[1] - 1) // block[1], 1)
    return grid, block


def remap_y(src_y, lut, out_w: int, out_h: int):
    """Bilinearly remap a Y plane (H,W), returning (out_h,out_w) with the same dtype."""
    import cupy as cp

    src_y = cp.ascontiguousarray(src_y)
    lut = cp.ascontiguousarray(lut.astype(cp.float32))
    in_h, in_w = src_y.shape
    dst = cp.empty((out_h, out_w), dtype=src_y.dtype)
    k = _sample1_kernel(_ctype_for(src_y))
    grid, block = _launch(out_w, out_h)
    k(grid, block, (src_y, lut, dst, in_w, in_h, out_w, out_h))
    return dst


def remap_uv(src_uv, lut, out_w: int, out_h: int):
    """Bilinearly remap an interleaved UV plane (Hc,Wc,2), returning (out_h,out_w,2) with the same dtype.

    The LUT must be in chroma resolution, where out_w,out_h = Wc,Hc.
    """
    import cupy as cp

    src_uv = cp.ascontiguousarray(src_uv)
    lut = cp.ascontiguousarray(lut.astype(cp.float32))
    in_h, in_w = src_uv.shape[0], src_uv.shape[1]
    dst = cp.empty((out_h, out_w, 2), dtype=src_uv.dtype)
    k = _sample2_kernel(_ctype_for(src_uv))
    grid, block = _launch(out_w, out_h)
    k(grid, block, (src_uv, lut, dst, in_w, in_h, out_w, out_h))
    return dst


# --- Slicing and stacking, using zero-copy views or a single copy. ---

def crop_plane(plane, x: int, y: int, w: int, h: int):
    """Slice (y:y+h, x:x+w) from a plane. Coordinates are in Y-plane units; callers halve them for UV."""
    return plane[y:y + h, x:x + w]


def hstack_planes(left, right):
    import cupy as cp
    return cp.concatenate([left, right], axis=1)


def vstack_planes(top, bottom):
    import cupy as cp
    return cp.concatenate([top, bottom], axis=0)


# --- Color conversion. ---

def nv12_to_bgr(y_plane, uv_plane, bit_depth: int = 8):
    """Convert NV12/P010 planes to BGR uint8 (H,W,3).

    P010/P016 input is normalized from its 16-bit container into 8-bit luma/chroma,
    matching the 8-bit LADA input boundary used by the current native_mosaic torch implementation.
    """
    import cupy as cp

    y_plane = cp.ascontiguousarray(y_plane)
    uv_plane = cp.ascontiguousarray(uv_plane)
    h, w = y_plane.shape
    if (h % 2) or (w % 2):
        raise ValueError(f"NV12 requires even dimensions, got {w}x{h}")
    import numpy as np

    uv_flat = uv_plane.reshape(h // 2, w)
    bgr = cp.empty((h, w, 3), dtype=cp.uint8)
    norm = 1.0 if int(bit_depth or 8) <= 8 else (255.0 / 65535.0)
    k = _nv_to_bgr_kernel(_ctype_for(y_plane))
    grid, block = _launch(w, h)
    # Critical: the kernel parameter is float, so pass an np.float32 scalar.
    # A Python float is passed by cp.RawKernel as C double (8 bytes) into a float
    # (4 bytes) slot, shifting the ABI so norm becomes 0.0 and colors are wrong.
    k(grid, block, (y_plane, uv_flat, bgr, np.int32(w), np.int32(h), np.float32(norm)))
    return bgr


def bgr_to_nv12(bgr_frame):
    """Convert BGR uint8 (H,W,3) to NV12 uint8 planes."""
    import cupy as cp
    import numpy as np

    bgr_frame = cp.ascontiguousarray(bgr_frame)
    if bgr_frame.dtype != cp.uint8:
        raise TypeError(f"BGR frame must be uint8, got {bgr_frame.dtype}")
    h, w = bgr_frame.shape[:2]
    if bgr_frame.ndim != 3 or bgr_frame.shape[2] != 3:
        raise ValueError(f"BGR frame must have shape (H,W,3), got {bgr_frame.shape}")
    if (h % 2) or (w % 2):
        raise ValueError(f"NV12 requires even dimensions, got {w}x{h}")
    y = cp.empty((h, w), dtype=cp.uint8)
    uv = cp.empty((h // 2, w // 2, 2), dtype=cp.uint8)
    grid_y, block_y = _launch(w, h)
    _bgr_to_nv12_y_kernel()(grid_y, block_y, (bgr_frame, y, np.int32(w), np.int32(h)))
    grid_uv, block_uv = _launch(w // 2, h // 2)
    _bgr_to_nv12_uv_kernel()(grid_uv, block_uv, (bgr_frame, uv, np.int32(w), np.int32(h)))
    return y, uv
