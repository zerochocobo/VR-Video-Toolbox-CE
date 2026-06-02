# 方案：去马赛克前先「检测+裁切」马赛克片段与区域（OneClick 4 Tab）

- 日期：2026-05-30
- 范围：`one_click/` 的四个 Tab（单文件/单眼/批量/批量单眼）
- 适用引擎：**Phase 1 仅 `lada` / `jasna` CLI**（`native_gpu` 留 Phase 2）
- 底片：**鱼眼与非鱼眼底片都支持**（`opt_fisheye` 勾或不勾都可用）
- 贴回：**Phase 1 直接做自研 GPU 贴回**（不再先走 ffmpeg overlay 的中间形态），技术栈复用现有 `gpu_engine` 的 PyNv 解/编码 + CuPy NV12 算子
- 目标：把"整段送 lada/jasna"改成"只送有马赛克的若干时间段 × 像素 rect 子片段"，端到端整体提速预估 **2–3×**（取决于马赛克密度）
- 关联文档：[[gpu-engine-architecture]]、`summary/summary_20260531_NATIVE_BOTTLENECK_PROFILE_CN.md`

---

## 1. 用户原始需求（含本轮修订）

1. 4 个 Tab 里"去马赛克前先转换成鱼眼"复选框下方，新增一个选项「去马赛克前先提取有马赛克的片段和区域，加快解码效率」。
2. 勾选后：用 `lada_vr_mosaic_detection_model_v2_fast.pt` 扫描当前要送去马赛克的视频，输出一组带时间区间 + 像素 rect 的子片段：
   - 时间区间前后多带几秒、片段不要太短太碎；
   - rect 取当段的并集再 **扩大 2×** 多带周边像素信息，不超出画面边界；
   - **按关键帧切割**（避免覆盖回贴时时间不精确）+ **按 rect 裁切**得到新的小视频。
   - 每段元数据要保存。
3. 复制一份底片为 `*.restored.mp4`（鱼眼或非鱼眼都支持，见 §4 命名）；用 lada / jasna 处理每段小视频；处理完把每段恢复结果按对应时间段和 rect **覆盖回** restored 底片。
4. **本轮修订**：
   - 贴回 Phase 1 直接做**自研 GPU 贴回**（不走 ffmpeg overlay 滤镜中间形态）；
   - 底片**不再要求必须是鱼眼**——鱼眼底片（`xxx_L_fisheye.mp4`）和非鱼眼底片（`xxx_L.mp4` / 整 VR 帧 `xxx_sbs.mp4`）都要支持。
5. Phase 1 不处理 native_gpu 引擎。

---

## 2. 旧 vs 新 流程

### 旧（OneClick 单眼 + 鱼眼，已有）

```
源 → split+VR→Fisheye 一遍   ──►  xxx_L_fisheye.mp4
xxx_L_fisheye.mp4  ──[lada/jasna 全段]──►  xxx_L_fisheye.restored.mp4
xxx_L_fisheye.restored.mp4 ──[Fisheye→VR]──►  xxx_L.restored.mp4
```

旧的非鱼眼路径同理（只是底片名是 `xxx_L.mp4` 等，没有 VR→Fisheye 那一步）。

### 新（带 pre-extract，鱼眼/非鱼眼通用）

```
[准备底片] 鱼眼路径：源 → split+VR→Fisheye → xxx_L_fisheye.mp4
           非鱼眼路径：源 → split           → xxx_L.mp4
           （或 SBS：源 → 复制/裁时间       → xxx_sbs_S..._E....mp4，作为底片）

[扫描] YOLO11-seg(v2_fast) 抽帧扫描底片
       └─► segments.json: [{start,end,rect{x,y,w,h}, seg_id}, ...]

[切] 对每段 (start,end,rect)：
       ├─ 时间用 I 帧对齐 + ffmpeg -c copy 切 → 子段全帧 mp4（无重编码）
       └─ rect 用 GPU NVENC crop 重编一遍 → 子段裁切 mp4（小分辨率）
       存为 BASE.seg{NN}.mp4

[复制底片] BASE.mp4  ──[GPU NVDEC→NVENC bit-exact 或直接 shutil.copy]──►  BASE.restored.mp4
           （目的：作为贴回 in-place 修改的目标文件；优先 shutil.copy 省时间）

[去马] 对每个子段裁切 mp4：
       lada/jasna CLI → BASE.seg{NN}.restored.mp4

[贴回] 自研 GPU 贴回管线（gpu_engine/files.paste_segments_gpu）
       一遍 NVDEC 底片 + N 路 NVDEC 段 → CuPy NV12 平面级 rect 区域羽化 alpha 混合
       → NVENC 编码 → 加回源音轨
       覆盖写 BASE.restored.mp4

[后续] 鱼眼底片：BASE.restored.mp4 ──[Fisheye→VR + merge]──► xxx.restored.mp4
       非鱼眼底片：BASE.restored.mp4 已经是最终输出，或继续 merge L/R
```

时间节省两块：① lada/jasna 只处理"有马赛克的子段 × rect 子分辨率"；② 自研 GPU 贴回一遍 NVDEC+NVENC 就完成，比 ffmpeg overlay 多输入滤镜图省 CPU swscale 与中间产物。

---

## 3. 模块拆分

### 3.1 新增 `utils/mosaic_prescan.py`（检测扫描器）

职责：拿任意 mp4（这里给鱼眼底片），独立加载 `lada_vr_mosaic_detection_model_v2_fast.pt`，抽帧扫描，输出聚合后的 `List[Segment]`。

实现要点：
- **独立加载检测模型**：不依赖 `NativeMosaicEngine`（避免也加载 BasicVSR++ 恢复模型，浪费显存）。直接：
  ```python
  from lada.models.yolo.yolo11_segmentation_model import Yolo11SegmentationModel
  det = Yolo11SegmentationModel(model_path, device="cuda", imgsz=imgsz, fp16=True)
  ```
- **抽帧**：用 `gpu_engine.pynv_io.PyNvSimpleDecoder` 按 `sample_stride_s` 跳帧（如 0.5s/帧，比逐帧推理快 ~60×）。或用 PyAV/cv2 软解（fast 模型推理时间远 > 解码，影响小）。
- **YOLO 推理**：fast 变体在 4K/8K letterbox 后 imgsz=1280 上预期 ~30-60fps。整段 30min × 0.5s stride = 3600 帧 ≈ 60-120s 即可扫完。
- **时间聚合**：
  1. 命中帧（有 box）的时间戳排序。
  2. 相邻命中间隔 ≤ `merge_gap_s`（默认 1.5s）的视为同段。
  3. 段开头/结尾各 pad `head_tail_pad_s`（默认 2.0s）。
  4. pad 后若两段距离 < `min_gap_s`（默认 2.0s）→ 二次合并。
  5. 丢弃 < `min_segment_s`（默认 1.5s）的碎段（pad 后还碎的就跳过，反正肉眼基本看不到几帧马赛克）。
- **空间聚合**：
  1. 段内所有 box 取**外接矩形并集**（不是每帧单独 rect，避免逐帧 rect 变化导致接缝抖动）。
  2. 中心不变，长宽各 × `rect_expand`（默认 2.0）。
  3. clamp 到 `[0,0,W,H]`。
  4. 16 像素对齐（NVENC 友好，且 lada/jasna 解码器友好）。
- **数据结构**：
  ```python
  @dataclass
  class MosaicSegment:
      seg_id: int
      start_s: float        # 检测意义上的起始（pad 后）
      end_s: float
      start_s_kf: float     # I 帧对齐后实际切割的起点
      end_s_kf: float
      x: int; y: int; w: int; h: int   # 16 对齐后的 rect
      conf_max: float       # 段内最大置信度（用于排序/日志）
  ```
- **输出**：`segments.json` 写到与底片同目录，方便复用与排查。

### 3.2 新增 `utils/keyframe_cutter.py`（I 帧对齐切割器）

- `list_keyframes(path) -> list[float]`：`ffprobe -select_streams v -skip_frame nokey -show_frames -show_entries frame=pts_time -of csv=p=0`。
- `align(start_s, end_s, kfs)`：start 向**下**找最近 I 帧（不丢前导帧），end 向**上**找最近 I 帧（不丢尾帧），把这俩存到 segment.start_s_kf / end_s_kf。
- `cut_segment(input, output, start_kf, end_kf, rect)`：
  ```
  ffmpeg -hide_banner -loglevel error
         -ss {start_kf} -to {end_kf} -i input
         -vf "crop={w}:{h}:{x}:{y}"
         -c:v hevc_nvenc -preset p7 -rc vbr -cq 18
         -an output
  ```
  - 用 `-ss <kf>` 配合 I 帧对齐 → 切割精确无前置帧丢失。
  - 用 hevc_nvenc rect crop 重编一遍（**不可避免**，因为要换分辨率；这一遍是 NVENC，相对 lada 时间忽略不计）。
  - 强制 GOP 较小（如 `-g 60`）方便后续贴回的精确性。
  - 不带音频（贴回时音频走底片的）。

> 关于"I 帧密度低"的兜底：如果检测到源鱼眼 GOP 很大（如 >10s），先用 GPU 重编一遍**注入密集 I 帧（`-g {fps*2}`）的"对齐底片"** 作为后续切/贴回的统一基准——这是底片用，**底片只重编 1 次**，子段重编 1 次，每帧总共 2 次 GPU 编码，依然远低于全段 lada 的代价。

### 3.3 复用 `process_lada`（不改）

`one_click/logic.py:387 process_lada(input, output, ...)` 已经支持 CLI 与 native；本期只走 CLI 引擎。对每个 `xxx.seg{NN}.mp4` 调一次即可。

### 3.4 新增 `gpu_engine/files.paste_segments_gpu` + `utils/segment_paster.py`（自研 GPU 贴回器，Phase 1 主路径）

**总体思路**：一遍 NVDEC 底片 + 按需 NVDEC 每段 → CuPy NV12 平面级 rect 区域羽化 alpha 混合 → `_EncodeSink` NVENC 直编 → `mux.mux_hevc_with_audio` 加回底片音轨。技术栈与 `combine_video`/`extract_clip` 同栈（见 [gpu_engine/files.py:589](gpu_engine/files.py:589)/[681](gpu_engine/files.py:681)），增量代码量较小。

**输入约束**（前置确认，让贴回算法简单）：
- 段切割时统一规则：`-ss <kf_aligned> -to <kf_aligned> -i base -c copy` → 段的第 0 帧 = 底片第 `round(start_s_kf * fps)` 帧（PTS 0 对齐）。
- 段 rect 是 16 像素对齐的（NV12 半色度对齐友好，rect 的 (x,y,w,h) 都满足 x%2==0, y%2==0；推荐 16 对齐以保证编码器友好）。
- 段与段在时间轴上**已合并到无重叠**（扫描器聚合阶段保证）。这样每个底片帧最多被 1 段命中。
- 段编码位深、色度子采样与底片一致（NV12 8-bit；10-bit 走 P010）。

**核心数据结构**（`segment_paster.py`）：
```python
@dataclass
class PasteSeg:
    seg_id: int
    path: Path                 # 段 restored 文件
    base_frame_start: int      # 含
    base_frame_end: int        # 含（base_frame_start + len(seg) - 1）
    x: int; y: int; w: int; h: int   # 像素 rect（16 对齐）
    decoder: PyNvThreadedSerialDecoder | None = None   # 懒加载
    alpha_y: cp.ndarray | None = None    # (h, w) float32，0..1，feather 羽化掩码
    alpha_c: cp.ndarray | None = None    # (h/2, w/2) float32（色度半分辨率）
```

**羽化 alpha**（一次预计算，整段复用）：
```python
def make_feather_alpha(w, h, feather_px, dtype=cp.float32):
    # 中间 = 1，4 边各 feather_px 线性 0→1。
    ax = cp.minimum(cp.minimum(cp.arange(w), w-1-cp.arange(w)), feather_px) / feather_px
    ay = cp.minimum(cp.minimum(cp.arange(h), h-1-cp.arange(h)), feather_px) / feather_px
    return (ay[:, None] * ax[None, :]).astype(dtype)
```
色度版本：`feather_px // 2`、宽高减半（NV12 UV 2×2 子采样）。

**贴回主循环**（`gpu_engine/files.py` 新增 `paste_segments_gpu`）：
```python
def paste_segments_gpu(base_src, dst, segments: list[PasteSeg], *,
                      cq=None, bitrate_bps=None, keep_audio=True,
                      log_callback=None, cancel_token=None):
    meta = probe.probe_video(base_src)
    bd   = 10 if meta.bit_depth > 8 else 8
    dec  = PyNvThreadedSerialDecoder(base_src, bit_depth=bd)
    info = dec.info
    fps  = meta.source_fps or info.fps
    n    = len(dec)

    # 按起始帧排索引；活跃集合用 set
    segs_by_start = sorted(segments, key=lambda s: s.base_frame_start)
    seg_iter, next_seg = iter(segs_by_start), None
    next_seg = next(seg_iter, None)
    active: list[PasteSeg] = []

    bitrate_bps = _resolve_bitrate(info.width, info.height, fps, bitrate_bps, meta.bitrate_bps)
    enc = PyNvEncoderSession(info.width, info.height, bit_depth=bd, codec="hevc",
                             **_encoder_kwargs(meta, bitrate_bps))
    raw = Path(tempfile.gettempdir()) / f"{Path(dst).stem}.raw.hevc"

    with open(raw, "wb") as f:
        sink = _EncodeSink(enc, f)
        for i in range(n):
            if cancel_token and cancel_token.cancelled: break

            # 进/出活跃集合
            while next_seg and next_seg.base_frame_start == i:
                next_seg.decoder = PyNvThreadedSerialDecoder(next_seg.path, bit_depth=bd)
                next_seg.alpha_y, next_seg.alpha_c = _build_feathers(next_seg, bd)
                active.append(next_seg)
                next_seg = next(seg_iter, None)
            for s in [s for s in active if s.base_frame_end < i]:
                s.decoder.stop(); active.remove(s)

            base = dec.frame_at(i); cp.cuda.Device().synchronize()
            y, uv = base.y_uv_cupy()

            for s in active:
                sf = s.decoder.frame_at(i - s.base_frame_start)
                sy, suv = sf.y_uv_cupy()
                # rect 区域 alpha blend（Y plane）
                ry = y[s.y:s.y + s.h, s.x:s.x + s.w]
                ry[:] = (s.alpha_y * sy.astype(cp.float32) +
                         (1 - s.alpha_y) * ry.astype(cp.float32)).astype(y.dtype)
                # rect 区域 alpha blend（UV plane）
                cy, cx, ch, cw = s.y // 2, s.x // 2, s.h // 2, s.w // 2
                ruv = uv[cy:cy + ch, cx:cx + cw, :]
                a   = s.alpha_c[..., None]
                ruv[:] = (a * suv.astype(cp.float32) +
                          (1 - a) * ruv.astype(cp.float32)).astype(uv.dtype)

            app = _pack_planes(y, uv, bd)
            sink.feed(app, force_idr=(i == 0))
        sink.flush()

    mux.mux_hevc_with_audio(raw, dst, fps=fps, color=meta.color,
                            audio_source=str(base_src) if keep_audio else None)
```

**性能要点**：
- `_EncodeSink` 已修好 NVENC 输入生命周期与码率控制坑（4K/8K 不出绿块、不爆码率）——直接复用。
- alpha blend 用 CuPy 整数路径或 `(a*src + (1-a)*dst)` 写成单个 RawKernel 可消除 dtype 转换；首版用 cp 矢量化版本即可（融合内核作为 H1 优化）。
- 多段同时活跃（rect 不重叠）需要多个 segment decoder 同时跑。`PyNvThreadedSerialDecoder` 每实例独占一组 NVDEC 资源——RTX 5060 Ti 上 NVDEC engine 1 个，但多 stream 并发不是问题，硬件调度。同时活跃段建议 ≤ 4，超过则段间分批处理（一次 batch 内 ≤ 4，跨 batch 把当前 restored 当作新底片，再贴下一批）。
- 10-bit：底片若 10-bit 则段也是 10-bit（cut 时不变位深），alpha 路径在 float32 上做，写回 uint16，与现有 P010 编码器路径对齐。

**模块分工**：
- `gpu_engine/files.py`：纯 GPU 函数 `paste_segments_gpu(base_src, dst, segments, ...)`，无外部 lada 依赖；
- `utils/segment_paster.py`：组装 `PasteSeg` 列表（PTS→frame 转换、活跃集合排序、批分），调用 GPU 函数；CancelToken 透传；fallback 接口（GPU 不可用时降级，见 §6.6）。

### 3.5 UI 改动

`one_click/main.py` 4 个 Tab 各加一个 Checkbutton（紧邻 `opt_fisheye` 下方）：

```python
# Tab 1 例
self.s_auto_pre_extract = tk.BooleanVar(value=False)
ttk.Checkbutton(tab, text=get_text('opt_pre_extract'),
                variable=self.s_auto_pre_extract).grid(row=4, column=0, columnspan=2,
                                                        sticky='w', padx=5, pady=5)
# 把 chk_keep_inter / chk_keep_bitrate 的 row 各 +1
```

把 `pre_extract` 参数透传到 `logic.run_*_pipeline(...)`。

**互锁逻辑**（建议在 UI 层做、不要在 logic 层悄悄忽略）：
- 当 `app_config.mosaic_engine == "native_gpu"` 时该复选框 **禁用 + 灰显**，hover tooltip 提示"Phase 1 暂不支持内置(GPU)引擎，请切换到 lada/jasna"。
- **与 `opt_fisheye` 互相独立**：鱼眼与非鱼眼底片都支持。logic 层根据 `use_fisheye` 选择底片路径与命名（`*_fisheye.mp4` vs `*.mp4`），扫描/切/贴回模块对底片名透明。

**i18n key**：`opt_pre_extract`
- zh：`去马赛克前先提取有马赛克的片段和区域，加快解码效率`
- en：`Pre-extract mosaic clips & regions before mosaic removal (faster decoding)`
- ja：`モザイク除去前にモザイク区間・領域だけを抽出（高速化）`

### 3.6 logic 层流程

在 `one_click/logic.py` 新增一个分支函数（鱼眼/非鱼眼通用，传入"底片"即可）：

```python
def _run_pre_extract_branch(base_path, restored_path, log_callback, process_callback):
    """对任意底片 base_path（鱼眼或非鱼眼）运行 pre-extract 管线，
    输出到 restored_path。返回 True 表示成功；False 表示无马赛克命中需调用方走全段路径。
    """
    from utils.mosaic_prescan import scan_segments
    from utils.keyframe_cutter import list_keyframes, align_segments, cut_segment
    from utils.segment_paster import paste_segments_gpu_or_fallback
    import shutil

    base_dir = os.path.dirname(base_path)
    stem = os.path.splitext(os.path.basename(base_path))[0]
    segments_json = os.path.join(base_dir, f"{stem}.segments.json")

    log_callback("[pre-extract] Scanning mosaic segments...")
    segments = scan_segments(base_path, log_callback=log_callback)
    if not segments:
        log_callback("[pre-extract] No mosaic detected → fallback to full-segment processing")
        return False

    kfs = list_keyframes(base_path)
    align_segments(segments, kfs)
    save_segments_json(segments, segments_json)

    log_callback(f"[pre-extract] {len(segments)} segments to process")
    seg_restored = []
    for seg in segments:
        seg_in  = os.path.join(base_dir, f"{stem}.seg{seg.seg_id:03d}.mp4")
        seg_out = os.path.join(base_dir, f"{stem}.seg{seg.seg_id:03d}.restored.mp4")
        cut_segment(base_path, seg_in, seg, log_callback, process_callback)
        process_lada(seg_in, seg_out, log_callback, process_callback)
        seg_restored.append(seg_out)

    log_callback(f"[pre-extract] Copying base → {restored_path}")
    shutil.copy2(base_path, restored_path)   # 先拷一份，贴回失败时还有兜底（覆盖再读）

    log_callback("[pre-extract] Pasting segments back (GPU)...")
    paste_segments_gpu_or_fallback(base_path, restored_path, segments, seg_restored,
                                    log_callback, process_callback)
    return True
```

接入 4 个 run_*_pipeline：
- **`run_single_file_pipeline`**：`use_fisheye=True` → 底片是 `xxx_S..._E..._L_fisheye.mp4` 和 `..._R_fisheye.mp4`（左右各自做一次 pre-extract）；`use_fisheye=False` → 底片是 `xxx_S..._E..._L.mp4` 和 `..._R.mp4`。后续 merge/Fisheye→VR 流程不变。
- **`run_single_eye_pipeline`**：单眼，鱼眼底片 `xxx_..._L_fisheye.mp4` 或非鱼眼底片 `xxx_..._L.mp4`。后续 defisheye 仅在鱼眼分支。
- 批量两个同理（对目录里每个文件都做一次单文件流程）。

调用条件：`pre_extract == True and engine != native_gpu`。`use_fisheye` 不参与互锁。

### 3.7 配置项（`utils/app_config`）

```python
"pre_extract_detection_model": "lada_vr_mosaic_detection_model_v2_fast.pt",
"pre_extract_sample_stride_s": 0.5,
"pre_extract_head_tail_pad_s": 2.0,
"pre_extract_merge_gap_s": 1.5,
"pre_extract_min_gap_s": 2.0,
"pre_extract_min_segment_s": 1.5,
"pre_extract_rect_expand": 2.0,
"pre_extract_rect_align": 16,
"pre_extract_rect_min_px": 512,    # 太小的 rect lada 处理质量差，扩到下限
"pre_extract_feather_px": 12,
"pre_extract_yolo_imgsz": 1280,
"pre_extract_yolo_conf": 0.25,
"pre_extract_keep_segments": False,  # debug 用，保留中间段文件
"pre_extract_inject_keyframes": "auto",  # auto/never/always - 大 GOP 自动注入密集 I 帧底片
"pre_extract_inject_gop_sec": 2.0,
```

---

## 4. 文件命名约定（鱼眼/非鱼眼通用）

定义 `BASE` = 当前底片的不带扩展名的文件名（无论鱼眼/非鱼眼/SBS）：

| 鱼眼路径示例 | 非鱼眼路径示例 |
|---|---|
| `xxx_S00..._E..._L_fisheye.mp4`（底片）| `xxx_S00..._E..._L.mp4`（底片） |
| `xxx_S..._L_fisheye.seg000.mp4` | `xxx_S..._L.seg000.mp4` |
| `xxx_S..._L_fisheye.seg000.restored.mp4` | `xxx_S..._L.seg000.restored.mp4` |
| `xxx_S..._L_fisheye.restored.mp4`（贴回输出） | `xxx_S..._L.restored.mp4`（贴回输出） |
| `xxx_S..._L_fisheye.segments.json` | `xxx_S..._L.segments.json` |

- 这个命名规则让 `_run_pre_extract_branch(base, restored)` 完全不需要知道是鱼眼还是非鱼眼，只看 `BASE`/`BASE.restored`。
- 鱼眼分支贴回后接 Fisheye→VR + merge（既有 logic）；非鱼眼分支贴回后直接是单眼/SBS 输出，或继续 merge 左右眼（既有 logic）。
- 清理逻辑：`keep_intermediate=False` 时删 `BASE.seg*.mp4` 与 `BASE.seg*.restored.mp4`；`BASE.segments.json` 默认保留（小，便于复跑/排查）。
- 断点续传：贴回前若 `BASE.seg{NN}.restored.mp4` 已存在则跳过该段的 cut+lada；若 `BASE.restored.mp4` 已存在且与全部段记录的尺寸/PTS 一致，整段跳过。

---

## 5. 性能预估（8K SBS、30min、典型马赛克密度 ~25%）

| 阶段 | 旧 | 新（pre-extract） | 备注 |
|---|---|---|---|
| split + VR→Fisheye | ~3 min | ~3 min | 不变 |
| 扫描检测 | — | ~1–2 min | YOLO fast ×（30min/0.5s 帧）|
| 段切割（NVENC crop） | — | ~2 min | rect 子分辨率小、HEVC nvenc |
| lada/jasna | ~100 min | **~25–35 min** | 7.5 min 时长 + rect 0.5× 面积 |
| 复制底片 | — | ~5–10s | shutil.copy（小于一遍 NVENC 重编）|
| 贴回（**自研 GPU**） | — | **~4–6 min** | 一遍 NVDEC+NVENC + 平面 alpha blend；活跃段 ≤ 4 路 NVDEC |
| Fisheye→VR (+merge) | ~3 min | ~3 min | 不变 |
| **合计** | **~106 min** | **~38–51 min** | **~2.1–2.8×** |

> 贴回从 ffmpeg overlay 的 ~6–10 min 缩到自研 GPU 的 ~4–6 min（省 CPU swscale + 直接复用 `_EncodeSink` 与已调优的码率策略）。
>
> 马赛克密度越低收益越大（10% 密度可达 4–5×）；密度 ≥ 60% 时增加扫描/贴回开销反而略亏，需要 UI 上加运行后日志统计提醒用户。

---

## 6. 风险与边界

1. **漏检** → 残留马赛克。缓解：fast 模型 conf 阈值 0.25 偏低 + stride 0.5s 偏密；段开头/结尾各 pad 2s；用户可一键改用 v2_accurate 模型。
2. **rect 错位** → 贴回出现矩形接缝。缓解：rect 16 像素对齐 + 12px 羽化 alpha blend + lada 在 rect 内的边缘其实是 fade。
3. **I 帧稀疏（大 GOP）** → 切割对齐 pad 过大，时间收益缩水。缓解：检测到 GOP > 5s 时自动先做"密集 I 帧底片"GPU 重编（`pre_extract_inject_keyframes=auto`）。本期 cut 已用 `-c copy + 关键帧对齐`，所以底片必须 I 帧密集才有收益。
4. **rect 过小**（< 256）→ lada 模型质量下降。缓解：`pre_extract_rect_min_px=512` 兜底放大。
5. **多段同时活跃** → 多路 NVDEC 资源竞争。缓解：贴回器活跃段上限 4；超过 4 时分批贴回，每批的输出作为下一批的底片。
6. **音轨**：所有段都不带音频（`-an`），贴回时 `mux_hevc_with_audio(..., audio_source=base_src)` 从底片取一次音轨，全程音频未重编。
7. **检测器与恢复器模型共存**：本期检测器独立加载（不复用 NativeMosaicEngine 单例），占用 ~200MB 显存；考虑 Phase 2 单例化（如果 native_gpu 也支持 pre-extract）。
8. **HDR/10-bit**：
    - 检测器是 8-bit RGB 模型，扫描时把 P010/HDR 降到 8-bit RGB 喂入即可（不影响位置定位）；
    - 切段/贴回路径全程保留底片原位深（8-bit→NV12，10-bit→P010），与 `extract_clip`/`combine_video` 一致；
    - HDR10(smpte2084) / HLG / bt2020 仍按 GPU 引擎现有 fallback 策略：`paste_segments_gpu` 内部 `probe.route` 不通过即抛 `_GpuEncodeSetupError`，降级到 ffmpeg overlay 路径作为兜底（见 §6.10）。
9. **native_gpu Phase 2**：方案大体可复用；但 native 输入端已是 GPU 常驻 + 整段流水，rect crop 的"省解码"价值低；主要收益变成"省 BasicVSR++ 处理量"。可后续单独评估。
10. **GPU 贴回不可用兜底**：`paste_segments_gpu` 失败（HDR 源、NVDEC 资源耗尽、其他设置错误）时 → 降级 ffmpeg overlay 多输入贴回（Phase 1 仍实现一份当 fallback；段数 > 20 分批）。该兜底路径不是主路径，但必须存在，保证 pre-extract 选项稳定可用。
11. **stop/cancel**：每段处理之间检查 `CancelToken` / 进程 kill 标志；贴回主循环每帧检查 `cancel_token.cancelled`（与 `extract_clip`/`combine_video` 一致）。

---

## 7. 实施任务分解（建议开发顺序）

1. **任务 A** — `utils/mosaic_prescan.py`：模型加载、抽帧、聚合规则、JSON 输出，独立可跑。单元自测：拿一段已知马赛克视频，打印 segments.json，目视核对。无 UI 接入。
2. **任务 B** — `utils/keyframe_cutter.py`：ffprobe I 帧列表 + 对齐函数 + ffmpeg `-c copy` 时间切（先不带 rect crop）。验证段时间戳与底片一致。
3. **任务 C** — `utils/keyframe_cutter.py` 补 GPU rect crop（`gpu_engine.files.extract_clip` 已有；新增一个透传式封装支持任意 `(x,y,w,h)` rect 而非 left/right/top/bottom 预设；可能需要给 `extract_clip` 加 `rect=(x,y,w,h)` 参数）。单元自测：拿 A 输出的 segments，切出 BASE.seg{NN}.mp4，人眼检视分辨率/起止时间。
4. **任务 D** — `gpu_engine/files.paste_segments_gpu`：核心自研 GPU 贴回（§3.4 草案）。先实现 1 段贴回的最小可用版（无羽化），编码输出与底片完全一致才算通过；再加多段活跃集合管理；再加羽化 alpha；最后加分批兜底。每个子任务独立 commit。验证：拿 1 段贴回结果 + ffmpeg overlay 等效结果做 PSNR 对比（>40dB 通过）。
5. **任务 E** — `utils/segment_paster.py`：组装 `PasteSeg`、调 `paste_segments_gpu`、HDR/失败兜底降级到 ffmpeg overlay 路径（也在此模块写一份 fallback 版）。
6. **任务 F** — `one_click/logic.py` 加 `_run_pre_extract_branch` 与 4 个 run_*_pipeline 的参数透传/分支；鱼眼/非鱼眼两条路径都接通；命名约定按 §4。
7. **任务 G** — `one_click/main.py` 4 个 Tab 加 Checkbutton + 互锁（仅 native_gpu 灰显，与 `opt_fisheye` 互相独立）+ 引擎切换刷新 + i18n key；把 `pre_extract` 透传到 `logic.run_*`。
8. **任务 H** — `utils/app_config` 加默认配置；`models_cfg`/路径校验也支持 fast 模型存在性。
9. **任务 I** — 端到端实跑：8K SBS 30s/3min/30min 三档 × 鱼眼/非鱼眼两种底片；记录 fps 与最终文件大小，与旧路径对比；写实测结果回 summary。
10. **任务 J** — 大 GOP 自动注入密集 I 帧底片（兜底）+ 断点续传逻辑 + 多 batch 贴回兜底闭环。

---

## 8. 提交建议

每个任务一个 commit；任务 A/B/C/D 单独 commit 便于单元回滚（D 内部 4 个子任务亦可分 commit）；G (UI) 与 F (logic) 可一起 commit；H (config) 与默认值改动单独 commit；实测结果作为 `summary/summary_20260530+_PRE_EXTRACT_BENCH_CN.md` 补充文档。
