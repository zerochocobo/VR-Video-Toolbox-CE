# GPU 流水线重构方案

- 文档日期：2026-05-29
- 适用项目：VR_Video_Toolbox_NE
- 参考项目：reference/PTMediaServer
- 交付对象：技术开发同学
- 状态：已与产品方确认，可进入实施

---

## 1. 背景与目标

当前 VR_Video_Toolbox_NE 的所有视频处理工具都通过 `subprocess` 调 ffmpeg 完成。虽然解码用了 `hevc_cuvid` / `h264_cuvid`、编码用了 `hevc_nvenc`，但中间几何变换（`v360`、`crop`、`hstack`、`vstack`）仍在 CPU 上，导致每次操作都要 NVDEC → 系统内存 → CPU filter → 系统内存 → NVENC 来回拷贝。在 8K SBS VR 素材下，`v360` filter（hequirect ↔ fisheye）是首要瓶颈。

本次重构目标：

1. 把"解码 → 几何变换 → 合成 → 编码"全程改成 GPU 驻留（PyNvVideoCodec + CuPy + 自定义 RawKernel），关键链路获得 3–10× 端到端加速。
2. 保持现有 UI 与业务接口不变，所有改动集中在 `*/logic.py` 与新增的 `gpu_engine/` 模块。
3. 为未来"我们自己重构的 native-gpu mosaic 引擎"留好接口槽位（不在本次实施范围）。
4. 自动失败回退到原 ffmpeg 路径，保证不可解码源仍能完成任务。

明确**不**做：lada-cli / jasna-cli 的本地化、HDR10/HLG 真正的 tone-mapping 处理（只透传/回退）、字幕/DLNA/分发等工具的 GPU 化（这些没有 GPU 计算热点）。

---

## 2. 范围

### In scope（要 GPU 化的 logic.py）

| 模块 | 主要操作 |
|---|---|
| [one_click/logic.py](../one_click/logic.py) | split、heq↔fisheye、hstack、转码（外加 lada/jasna 外部 CLI） |
| [tool_v360_trans/logic.py](../tool_v360_trans/logic.py) | hequirect ↔ fisheye（含 SBS 双目） |
| [tool_vr2flat/logic.py](../tool_vr2flat/logic.py) | VR (hequirect) → flat，带 yaw/pitch/fov |
| [tool_split_combine/logic.py](../tool_split_combine/logic.py) | split（含 fisheye 互转）、combine（含 fisheye 互转） |
| [area_selection_rect_crop/logic.py](../area_selection_rect_crop/logic.py) | crop + 编码 |
| [area_selection_vr2flat/logic.py](../area_selection_vr2flat/logic.py) | 同 tool_vr2flat |

### Out of scope

- lada-cli / jasna-cli 的内部实现：保持外部 CLI 调用，仅在 one_click 中通过文件传递。
- tool_subtitle / tool_subembed / tool_dlna / tool_split_combine 的非视频部分：无 GPU 热点，保留 ffmpeg。
- [utils/ffmpeg_checker.py](../utils/ffmpeg_checker.py)：仍用于版本检测、错误友好化。
- [tool_vr2flat/logic.py](../tool_vr2flat/logic.py) 中的 PyAV 预览取帧（`get_vr_frame_image`）：非计算热点，保留。
- HDR10（smpte2084）/ HLG（arib-std-b67）/ bt2020 色彩的 GPU 处理：自动回退到 ffmpeg，理由见第 6 节。

---

## 3. 现状瓶颈

| 操作 | 当前实现 | GPU 化潜力 |
|---|---|---|
| 解码 | `hevc_cuvid` / `h264_cuvid`（GPU 内，但出口落 CPU） | PyNv 直接产出 GPU NV12 平面，零拷贝 |
| 编码 | `hevc_nvenc`（GPU 内，但入口在 CPU） | PyNv 直接吃 GPU NV12 平面，零拷贝 |
| `crop` | **CPU** | CuPy 切片，零拷贝 |
| `hstack` / `vstack` | **CPU** | CuPy 内存拷贝，<1 ms |
| `v360=hequirect:fisheye` 及反向 | **CPU**，8K 时是首要瓶颈 | LUT + CuPy 双线性 RawKernel，预期 5–10× |
| `v360=hequirect:flat:yaw=:pitch=:fov=` | **CPU** | 同上 |

参考 PTMediaServer 的 [pipeline/pynv_io.py](../reference/PTMediaServer/pipeline/pynv_io.py) 与 [pipeline/matting.py](../reference/PTMediaServer/pipeline/matting.py) 的实现，"全 GPU 驻留 + CuPy RawKernel" 在生产环境已经验证可行。

---

## 4. 整体架构

### 4.1 模块布局

新增一个独立子模块，与 `utils/` 同级：

```
gpu_engine/
├── __init__.py
├── runtime.py        # CUDA/CuPy/PyNv 启动暖机；内存池配置；设备能力探测
├── pynv_io.py        # PyNv 解/编码包装；GpuNv12Frame / GpuP016Frame
├── probe.py          # ffprobe + PyNv 元数据探测；路由决策（pynv vs ffmpeg）
├── v360_lut.py       # 三类 LUT 生成：heq→fisheye、fisheye→heq、heq→flat(yaw,pitch,fov)
├── nv12_kernels.py   # CuPy RawKernel：8-bit/10-bit 双线性采样、crop、hstack、vstack
├── frames.py         # ★ 帧流水线层：Iterator[GpuFrame] 算子组合（decode→split→remap→stack→encode）
├── files.py          # 文件层：把 frames pipeline 端到端跑完 → mp4 文件 + ffmpeg mux 音频
├── mux.py            # ffmpeg "-c copy" 复用裸 HEVC 与原音频
└── fallback.py       # 异常分类与自动回退到 ffmpeg 路径
```

### 4.2 两层接口

这是整个架构的核心决策。

**`frames.py`（算子层）**：所有算子接受 `Iterable[GpuFrame]` 并返回 `Iterable[GpuFrame]`：

```
decode_frames(src_path) -> Iterator[GpuFrame]
split_lr(frames)         -> tuple[Iterator[GpuFrame], Iterator[GpuFrame]]
v360_remap(frames, lut)  -> Iterator[GpuFrame]
hstack(left, right)      -> Iterator[GpuFrame]
encode_frames(frames, dst_path, color_meta, bitrate_opts)
```

**`files.py`（业务层）**：把多个算子串成"文件进 / 文件出"的完整流水线，被各 `*/logic.py` 调用。例如：

```
files.vr_split_to_fisheye(in_path, out_l, out_r, color_meta, bitrate_opts)
files.fisheye_merge_to_sbs(in_l, in_r, out_path, ...)
files.vr_to_flat(in_path, out_path, yaw, pitch, fov, ...)
```

**为什么分两层**：未来 native_gpu mosaic 引擎需要把"decode → fisheye → mosaic restore → fisheye→heq → hstack → encode"做成一条不落盘的 frame pipeline。如果只有文件层，未来要重写；分了算子层后，未来只需在 frame iterator 中间插一个 `native_mosaic.process(frames)` 算子，业务代码不需要动。

### 4.3 算子层数据结构

直接搬 PTMediaServer 的两个 dataclass（[pipeline/pynv_io.py:133-247](../reference/PTMediaServer/pipeline/pynv_io.py)）：

- `GpuNv12Frame`：8-bit NV12，Y 与 UV 两个 CUDA 平面，含 `owned_copy()` 方法用于 ThreadedDecoder 跨生命周期场景。
- `GpuP016Frame`：10-bit P010/P016，同上。
- 类型别名：`GpuFrame = GpuNv12Frame | GpuP016Frame`。

所有算子需同时支持两种 frame，通过 dispatch 选择对应的 RawKernel（uint8 vs uint16 双线性）。

### 4.4 路由决策（`probe.py`）

新增 `decide_backend(meta) -> BackendDecision`，行为对齐 PTMediaServer 的 `utils/video_metadata.py:select_backend()`。三种结果：

- `gpu_nv12`：8-bit 源（yuv420p / yuvj420p / nv12），GPU 路径。
- `gpu_p016`：10-bit SDR 源（pix_fmt 含 10、bt709 或未指定的 primaries/transfer），GPU 路径。
- `ffmpeg_fallback`：以下任一情况自动回退
  - HDR10（transfer = smpte2084）或 HLG（transfer = arib-std-b67）
  - bt2020 色彩（primaries = bt2020）
  - MPEG-4 ASP、msmpeg4v3 等 NVDEC 不能解的源
  - pix_fmt 非 `{yuv420p, yuvj420p, nv12, p010le, yuv420p10le}` 之一
  - 解码/编码运行时抛 CUDA error（在 `fallback.py` 内捕获）

---

## 5. 依赖与环境

### 5.1 本次新增依赖

修改 [pyproject.toml](../pyproject.toml)，新增两个：

```
pynvvideocodec>=2.1.0
cupy-cuda12x
```

### 5.2 系统依赖

- CUDA Toolkit 12.x（PyNvVideoCodec 2.1+ 要求）
- NVIDIA 驱动 ≥ 535（建议 ≥ 550）
- 支持 NVDEC + NVENC 的 GPU（Turing 及以后；推荐 Ampere/Ada 用于 HEVC 10-bit）

### 5.3 不在本次安装（为未来 native_gpu 留笔记）

```
onnxruntime-gpu     # 未来 native_gpu mosaic 模型推理
tensorrt-cu12       # 未来 native_gpu mosaic 模型 TRT 化
```

写入 README 的 ROADMAP，**这次不要装**，避免打包复杂度提前上涨。

---

## 6. 10-bit / HDR 路由策略

### 6.1 决策依据

核查 PTMediaServer 的 offline + live 模式（[utils/video_metadata.py:319-353](../reference/PTMediaServer/utils/video_metadata.py)）：

- **10-bit SDR**（HEVC Main10、pix_fmt = p010le/yuv420p10le、primaries 与 transfer 都是 bt709 或未指定）：走 PyNv，标记 experimental，由 `PT_PASSTHROUGH_PYNV_10BIT` 开关启用。
- **HDR10**（transfer = smpte2084）：明确返回 `ffmpeg_fallback`，理由 `"HDR/Main10/P010 needs a separate color/10-bit policy"`。
- **HLG**（transfer = arib-std-b67）：同 HDR10 处理，回退 ffmpeg。
- **bt2020 primaries**：回退 ffmpeg。

### 6.2 本项目采纳的策略

完全对齐：

| 源类型 | 路径 | 说明 |
|---|---|---|
| 8-bit yuv420p/nv12 bt709 | GPU NV12 | 主流 |
| 8-bit yuv420p bt2020 | ffmpeg 回退 | 色彩边界 |
| 10-bit p010 bt709 | GPU P016 | 用户日常素材，**首期必须支持** |
| 10-bit p010 bt2020 | ffmpeg 回退 | |
| HDR10 (smpte2084) | ffmpeg 回退 | |
| HLG (arib-std-b67) | ffmpeg 回退 | |

### 6.3 ffmpeg 回退路径的色彩元数据透传

回退到 ffmpeg 时，必须保留并显式写入：

```
-color_range tv -colorspace bt2020nc -color_primaries bt2020 -color_trc smpte2084
```

（值按源 ffprobe 结果填）。当前 [tool_vr2flat/logic.py:99-102](../tool_vr2flat/logic.py) 已经有类似的硬编码 bt709 写法，要改成从 `probe.py` 出的 `ColorMetadata` 动态生成。

### 6.4 GPU 路径的色彩元数据透传

PyNv encoder 输出的裸 HEVC bitstream 不含 colr atom，需在 `mux.py` 的 ffmpeg `-c copy` 阶段补：

```
ffmpeg -i raw.hevc -i src.mp4 -map 0:v -map 1:a -c copy \
       -color_range tv -colorspace bt709 -color_primaries bt709 -color_trc bt709 \
       out.mp4
```

### 6.5 RawKernel 实现要点

- `sample_y_bilinear_u8` 与 `sample_y_bilinear_u16`：模板化或两份实现，分别用于 8-bit 与 10-bit Y 平面。
- `sample_uv_bilinear_u8` / `sample_uv_bilinear_u16`：UV 平面是半分辨率交错 2 通道（NV12/P010 共享布局），LUT 坐标除以 2 后采样。
- 10-bit 路径在 CuPy 内部用 uint16，输出仍是 uint16 给 PyNv encoder。
- 注意 P016 与 P010 的差别：PyNv 实际给出的是高 10 bit 放在 uint16 的高位，低 6 bit 是 0；CuPy 计算中保持原样即可，不要做 `>> 6`。参考 PTMediaServer 的处理（搜 `GpuP016Frame.owned_copy`）。

---

## 7. 自动回退机制（`fallback.py`）

### 7.1 触发条件

| 类别 | 触发点 | 例子 |
|---|---|---|
| 静态路由 | `probe.decide_backend()` 返回 `ffmpeg_fallback` | HDR10、MPEG-4 ASP |
| 动态运行时 | GPU 路径执行中抛异常 | CUDA OOM、PyNv decode 失败、NVENC error |
| 启动期 | `runtime.warmup()` 失败 | 无 GPU、驱动太旧、CUDA 初始化失败 |

### 7.2 行为

- **静态路由**：在调用 `files.*` 前就决定，直接走 ffmpeg 路径，日志一行 `[backend=ffmpeg reason=...]`。
- **动态运行时**：单文件级回退。`files.*` 内部 try/except 包住 GPU 路径，捕获后调用同名 ffmpeg 实现重跑此文件。日志高亮 `[gpu→ffmpeg fallback] file=... reason=...`。
- **启动期**：全局降级为 ffmpeg-only 模式，所有工具退化到当前行为，UI 顶部一条 banner 提示。

### 7.3 配置开关

[utils/app_config.py](../utils/app_config.py) 新增字段：

```json
{
  "transcode_backend": "auto",   // auto | gpu | ffmpeg
  "mosaic_engine":     "lada"    // lada | jasna | native_gpu(占位，本期 disable)
}
```

- `auto`：默认；GPU 可用就 GPU，异常自动回退 ffmpeg。
- `gpu`：强制 GPU，失败直接报错（debug 用）。
- `ffmpeg`：强制 ffmpeg，跳过 GPU 路径（兼容/对比用）。

---

## 8. native_gpu 引擎接口槽位（本期不实施）

只做以下三件事，确保未来接入时业务代码不动：

1. [utils/engine_runner.py](../utils/engine_runner.py) 的 `get_engine_executable()` / `build_engine_cmd()` 在 `engine == "native_gpu"` 分支抛 `NotImplementedError`，注释说明"phase 7+"。
2. UI 引擎选择下拉框增加 `native_gpu` 选项，标灰 disabled。
3. `frames.py` 中预留 `mosaic_restore(frames, model_cfg) -> Iterator[GpuFrame]` 算子签名（实现为 `raise NotImplementedError`）。one_click 的 `files.full_pipeline_native_gpu()` 函数骨架可以写出来，但函数体只 raise。

未来动手 native_gpu 时，新增依赖 `onnxruntime-gpu`、`tensorrt-cu12`，并在 `gpu_engine/` 下新增 `native_mosaic/` 子包。

---

## 9. 分阶段实施

### 阶段 0：底座

**交付物**

- [pyproject.toml](../pyproject.toml) 添加 `pynvvideocodec`、`cupy-cuda12x`。
- `gpu_engine/runtime.py`：CUDA 设备探测、CuPy 内存池配置、PyNv 暖机（用一个内置 1 秒小 mp4 跑一次解码+编码）。
- `gpu_engine/pynv_io.py`：把 PTMediaServer 同名文件里的以下类原样搬过来，去掉 PT_ 配置依赖：
  - `CudaPlane`、`GpuNv12Frame`、`GpuP016Frame`、`PyNvVideoInfo`
  - `PyNvSimpleDecoder`、`PyNvThreadedSerialDecoder`
  - `FfmpegNv12SequentialDecoder`（用作不能 PyNv 解码时的兜底）
  - `CudaArrayView`、`GpuNv12AppFrame`、`GpuP016AppFrame`
- `gpu_engine/probe.py`：基于 ffprobe 的元数据解析 + `decide_backend()` 路由（对齐 PTMediaServer 的 [utils/video_metadata.py](../reference/PTMediaServer/utils/video_metadata.py)）。
- `gpu_engine/mux.py`：单个函数 `mux_video_with_audio(raw_hevc_path, src_audio_path, out_mp4, color_meta)`。
- `gpu_engine/fallback.py`：异常分类与回退装饰器。
- [main.py](../main.py) 启动时调用 `gpu_engine.runtime.warmup()`，失败则 fallback 全局 ffmpeg 模式。

**验收**

- 启动期暖机日志：`gpu_engine warmup ok: gpu_id=0 name=... cc=8.6 vram=24.0GB free=22.0GB`。
- `probe.decide_backend()` 对一组测试文件（见第 10 节）输出正确路由。
- 单元测试：8-bit 和 10-bit 各一个短样本，跑通解码 → 直接编码（不变换）的环回，输出 PSNR vs 源 ≥ 50dB（无几何变换的话基本无损）。

### 阶段 1：几何原语 + benchmark

**交付物**

- `gpu_engine/v360_lut.py`：
  - `make_heq_to_fisheye_lut(w, h, fov_deg=180) -> cupy.ndarray[float32]`，shape = (h, w, 2)，每个像素是 (x_src, y_src)。
  - `make_fisheye_to_heq_lut(w, h, fov_deg=180)`：反向。
  - `make_heq_to_flat_lut(w, h, yaw_deg, pitch_deg, fov_deg)`：VR2Flat 用。
  - 注意 yaw/pitch 旋转顺序对齐 ffmpeg `v360=...:rorder=ypr`（[tool_vr2flat/logic.py:205](../tool_vr2flat/logic.py)），否则结果对不上。
  - LUT 缓存：相同 (w, h, params) 复用，避免每帧重算。
- `gpu_engine/nv12_kernels.py`：
  - `sample_nv12_bilinear(src_frame, lut, out_frame)`：内部根据 frame 类型 dispatch u8/u16 双线性。
  - `crop_nv12(src_frame, x, y, w, h) -> GpuFrame`：CuPy 切片包装。
  - `hstack_nv12(left, right) -> GpuFrame` / `vstack_nv12`：单次 cp.concatenate。
- `gpu_engine/frames.py`：上述算子的 iterator 包装。
- `gpu_engine/files.py`：`transcode_only(src, dst)` 端到端实现，用于 benchmark 基线。
- `tests/bench_gpu_vs_ffmpeg.py`：在固定测试集上跑对比，输出耗时表 + PSNR 表。

**验收（go/no-go 关卡）**

在一段 30 秒 8K SBS HEVC 测试片上（含 8-bit 与 10-bit 两个版本）：

| 指标 | 阈值 |
|---|---|
| `tool_v360_trans` 等价流程端到端耗时 | ≥ **3×** 原 ffmpeg 路径 |
| `tool_vr2flat` 等价流程 | ≥ **4×** |
| `split_combine` 不含 fisheye | ≥ **1.5×** |
| 8-bit Y/UV PSNR vs ffmpeg 输出 | ≥ **42 / 38 dB** |
| 10-bit Y/UV PSNR vs ffmpeg 输出 | ≥ **45 / 40 dB** |

PSNR 低于阈值时不强制中断，但由开发者与产品方共同决定是否上线（很可能是 v360 LUT 的边缘像素插值边界差异，肉眼对照确认）。

### 阶段 2：tool_v360_trans

**为什么先做这个**：单一动作（仅 v360），代码最少，风险最低；同时是 GPU 化收益最高的工具，作为生产试点最合适。

**交付物**

- `gpu_engine/files.py` 增加 `vr_projection(src, dst, mode, dual_screen, color_meta, bitrate_opts)`，mode ∈ {`heq2fisheye`, `fisheye2heq`}。
- [tool_v360_trans/logic.py](../tool_v360_trans/logic.py) 的 `convert_projection()` 改为：
  1. `probe.decide_backend(src)`
  2. 路由到 `files.vr_projection()` 或保留的 `_convert_projection_ffmpeg()`（即原实现，改名留作回退）。
  3. 用 `fallback.with_auto_fallback()` 装饰，捕获运行时异常自动重跑 ffmpeg 版。

**验收**

- 8-bit / 10-bit、单目 / SBS 共 4 个变体各跑一遍，输出文件能被现有播放器正常打开。
- 强制 `backend=ffmpeg` 时行为等价当前。
- 强制 `backend=gpu` 在 HDR10 源上应明确报错（不会偷偷回退）。
- `backend=auto` 在 HDR10 源上自动走 ffmpeg，日志有路由信息。

### 阶段 3：vr2flat

**交付物**

- `files.vr_to_flat(src, dst, yaw, pitch, fov, width, height, color_meta, bitrate_opts)`。
- [tool_vr2flat/logic.py](../tool_vr2flat/logic.py) 的 `run_pipeline()` 路由到上述函数。
- [area_selection_vr2flat/logic.py](../area_selection_vr2flat/logic.py) 同样路由。
- PyAV 取预览帧（`get_vr_frame_image`）保留不动。

**验收**

- 对参考 yaw/pitch/fov 组合（例如 yaw=0/pitch=0/fov=110）与 ffmpeg 输出做 PSNR 对比，Y/UV 达标。
- `extract_clip`（crop + 编码）一并迁移走 GPU。

### 阶段 4：split_combine + area_selection_rect_crop

**交付物**

- `files.split_video(src, mode, out_dir, to_fisheye, color_meta, bitrate_opts)`：mode ∈ {`left`, `right`, `top`, `bottom`, `left_and_right`, `top_and_bottom`}，对应 [tool_split_combine/logic.py:57](../tool_split_combine/logic.py:57)。
- `files.combine_video(in_a, in_b, mode, dst, from_fisheye, color_meta, bitrate_opts)`：mode ∈ {`left_right`, `top_bottom`}。
- `files.extract_clip(src, eye_mode, start_time, end_time, dst, color_meta, bitrate_opts)`：area_selection_rect_crop 与 tool_vr2flat 共用。
- start_time / end_time 在 GPU 路径下实现为 PyNv `SimpleDecoder` 的 frame index 起止（需要把 `HH:MM:SS` 转换为 frame index，按 source FPS 算；非 CFR 源需用 `probe_timing_metadata()` 兜底）。

**验收**

- 6 种 split mode × {fisheye, 无 fisheye} 共 12 个变体跑通。
- combine 4 个变体跑通。
- 含 start_time / end_time 的 extract_clip 与 ffmpeg `-ss / -to` 的帧数差异 ≤ 1 帧。

### 阶段 5：one_click

**交付物**

- `files.one_click_sbs_pipeline(src, dst, start_time, end_time, use_fisheye, keep_original_bitrate, mosaic_engine, ...)`：
  - 内部分三段：
    - **Pass A**：decode → split → (optional) heq→fisheye → 写 L/R mp4（lada/jasna 的输入）。
    - **External**：对 L、R 分别调 lada-cli 或 jasna-cli（沿用 `utils/engine_runner.build_engine_cmd()`）。
    - **Pass B**：decode L_restored + R_restored → (optional) fisheye→heq → hstack → encode → mux 原音频。
  - mosaic_engine == `native_gpu` 分支 raise `NotImplementedError`（占位）。
- [one_click/logic.py](../one_click/logic.py) 的 `run_single_file_pipeline`、`run_single_eye_pipeline`、`run_batch_pipeline`、`run_batch_eye_pipeline`、`run_merge_tool` 全部路由到上述 files API。
- Smart Resume：Pass A 的产物文件存在则跳过 Pass A；L/R restored 都存在则跳过 lada/jasna；最终文件存在则全跳过。颗粒度与现状一致。

**验收**

- 单文件、单眼、批处理、合并 4 种入口跑通，输出与现在 ffmpeg 路径肉眼一致、文件大小相近（±15% 容忍）。
- `keep_original_bitrate=True/False` 都正确。
- 中途 kill 后再起，Smart Resume 能从正确步骤继续。
- mosaic_engine == `native_gpu` 时 UI 提示"未实现，请选 lada 或 jasna"。

### 阶段 6：打包 + 收尾

**交付物**

- [build_exe.bat](../build_exe.bat) 改为 PyInstaller **onedir** 模式（参考 PTMediaServer [build_exe.py](../reference/PTMediaServer/build_exe.py) 与 `pt_core.spec`）。
- 不要 UPX（CUDA DLL 会损坏）。删除或弃用 [build_upxexe.bat](../build_upxexe.bat)。
- 把以下 DLL 显式收进 dist 同目录：
  - CUDA Runtime：`cudart64_12.dll`、`nvrtc64_120_0.dll`、`nvJitLink_120_0.dll`、`cublas64_12.dll`、`cublasLt64_12.dll`
  - PyNvVideoCodec 自带 DLL（pip 包内）
  - 不需要 cuDNN（本期没有 ONNX 推理）
- 启动期暖机日志写到 `runtime_cache/gpu_runtime.log`，方便用户报 bug 时附带。
- [README.md](../README.md) / [README_CN.md](../README_CN.md) 更新：新依赖、GPU 要求、回退说明、HDR 限制。

**验收**

- 在干净 Windows 11 机器（仅装 NVIDIA 驱动，无 CUDA Toolkit、无 Python）解压 dist 目录后能直接运行。
- 启动到 UI 就绪 ≤ 8 秒（onedir 优势）。
- exe 体积报告记录（预期 800 MB – 1.2 GB）。

---

## 10. 测试集

放到 `tests/fixtures/`（≥ 4 个文件，每个 ≤ 60 秒，覆盖所有路由分支）：

| 文件 | 编码 | 分辨率 | 颜色 | 期望路由 |
|---|---|---|---|---|
| `sbs_h264_8bit_bt709.mp4` | H.264 yuv420p | 7680×3840 | bt709 | `gpu_nv12` |
| `sbs_hevc_8bit_bt709.mp4` | HEVC Main yuv420p | 7680×3840 | bt709 | `gpu_nv12` |
| `sbs_hevc_10bit_bt709.mp4` | HEVC Main10 p010le | 7680×3840 | bt709 | `gpu_p016` |
| `sbs_hevc_10bit_hdr10.mp4` | HEVC Main10 p010le | 7680×3840 | bt2020 + smpte2084 | `ffmpeg_fallback` |
| `sbs_mpeg4.mp4`（可选） | MPEG-4 ASP | 任意 | bt709 | `ffmpeg_fallback`（PyNv 不能解） |

每个文件提供两份：30 秒 hequirect SBS 与 30 秒 fisheye SBS（除 MPEG-4），共 ≈ 8 个文件。

**测试用例矩阵**：每个工具 × 每个测试文件 × {backend=gpu, backend=ffmpeg, backend=auto} 都要跑过。

---

## 11. 风险与缓解

| 风险 | 缓解 |
|---|---|
| LUT 边缘像素与 ffmpeg v360 双线性边界处理略有不同 | benchmark 阶段对照 PSNR；若低于阈值，引入边缘 clamp/wrap 选项匹配 ffmpeg 行为 |
| PyNvVideoCodec 在某些 H.264 边缘 profile 上失败 | 静态路由识别已知差源；动态运行时回退兜底 |
| CuPy 内存池碎片化导致长批处理 OOM | 每个文件结束后调用 `cp.get_default_memory_pool().free_all_blocks()` |
| start_time/end_time 在 VFR 源上 PyNv seek 不精确 | 用 `probe_timing_metadata()` 判断 VFR，VFR 源走 ffmpeg 回退 |
| onedir 打包后 CUDA DLL 路径解析失败 | 启动期显式 `os.add_dll_directory()`，参考 PTMediaServer [utils/runtime_dll_paths.py](../reference/PTMediaServer/utils/runtime_dll_paths.py) |
| 10-bit 链路 NVENC 配置错误导致输出 8-bit | encode_frames 显式断言：input 是 P016 时 NVENC profile = main10、pix_fmt = p010 |
| lada/jasna 输入文件颜色元数据丢失，导致后续 Pass B 输入色彩错 | Pass A 写文件时显式写 colr atom；Pass B 读时用 ffprobe 重新探测 |
| 用户卡太老（无 NVENC HEVC）| 启动期暖机检测 NVENC HEVC 支持，不支持则全局回退 ffmpeg |

---

## 12. 配置项变更

### [utils/app_config.py](../utils/app_config.py) 新增字段

```python
DEFAULTS = {
    # 原有字段...
    "transcode_backend": "auto",   # auto | gpu | ffmpeg
    "mosaic_engine":     "lada",   # lada | jasna | native_gpu(占位)
    "gpu_warmup_enabled": True,
    "gpu_log_verbose":   False,
}
```

UI 增加：

- 设置面板：transcode_backend 单选（默认 auto）、native_gpu 灰掉的引擎选项。
- 主界面顶部：若启动期 GPU 不可用，显示一行黄色 banner "GPU 加速不可用，已回退 ffmpeg 模式（性能较低）"。

---

## 13. 文件清单（开发交付参考）

### 新增

```
gpu_engine/__init__.py
gpu_engine/runtime.py
gpu_engine/pynv_io.py
gpu_engine/probe.py
gpu_engine/v360_lut.py
gpu_engine/nv12_kernels.py
gpu_engine/frames.py
gpu_engine/files.py
gpu_engine/mux.py
gpu_engine/fallback.py
tests/fixtures/<8 个测试视频>
tests/bench_gpu_vs_ffmpeg.py
tests/test_probe_routing.py
tests/test_v360_lut.py
tests/test_kernels.py
tests/test_files_e2e.py
```

### 修改

```
pyproject.toml                          # 新增 pynvvideocodec + cupy-cuda12x
main.py                                 # 启动期 gpu_engine.runtime.warmup()
utils/app_config.py                     # 新增配置字段
utils/engine_runner.py                  # native_gpu 分支占位
one_click/logic.py                      # 路由到 gpu_engine.files
tool_v360_trans/logic.py                # 同上
tool_vr2flat/logic.py                   # 同上（保留 PyAV 预览）
tool_split_combine/logic.py             # 同上
area_selection_rect_crop/logic.py       # 同上
area_selection_vr2flat/logic.py         # 同上
build_exe.bat                           # 改 onedir
README.md / README_CN.md / README_JP.md # GPU 要求、HDR 限制
```

### 弃用

```
build_upxexe.bat                        # 删除或注释为不要用
```

---

## 14. 待开放问题（开发可在实施中再确认）

1. PyNv 的 `SimpleDecoder.frame_at(index)` 在某些 mp4 上不支持随机访问，需要时改用 `ThreadedDecoder` 顺序读到目标帧。本项目大多是顺序处理，影响小，但 area_selection_* 的 extract_clip 含 start_time，需要测试。
2. 编码器参数（preset / bitrate / cq）的 PyNv 等价映射：当前 ffmpeg 用 `hevc_nvenc -preset p7 -cq 18` 与 `-rc vbr -b:v -maxrate -bufsize`。PyNv 用 `tuning_info=high_quality, preset=P7, rc=cbr/vbr, bitrate=...`。需要在阶段 1 benchmark 时确认参数等价（bitrate vbr 模式下，PyNv 的 maxrate/bufsize 通过 `vbvBufferSize / vbvInitialDelay` 配置）。
3. lada-cli 和 jasna-cli 输入是 mp4 文件，输出也是 mp4 文件。它们自己也走 ffmpeg 编码（lada 有 `--encoder hevc_nvenc`）。我们的 Pass A 输出色彩元数据，lada/jasna 是否原样保留，需要在阶段 5 实测确认；不行的话 Pass B 输入需要额外补色彩元数据。
4. Smart Resume 颗粒度是否需要更细（例如 fisheye→heq 转换文件作为独立 checkpoint）？目前保持与现状一致，待用户反馈再决定。

---

## 15. 时间预估（粗略，仅供参考）

| 阶段 | 工程量 |
|---|---|
| 阶段 0 底座 | 2–3 天 |
| 阶段 1 几何原语 + benchmark | 3–4 天（含 LUT 调对齐） |
| 阶段 2 tool_v360_trans | 1 天 |
| 阶段 3 vr2flat | 1–2 天 |
| 阶段 4 split_combine + crop | 1–2 天 |
| 阶段 5 one_click | 2–3 天 |
| 阶段 6 打包 + 收尾 | 2–3 天 |
| **合计** | **12–18 天** |

不含 native_gpu mosaic 引擎（独立子项目，预计需要额外 4–6 周）。

---

## 16. 参考资料

- [reference/PTMediaServer/pipeline/pynv_io.py](../reference/PTMediaServer/pipeline/pynv_io.py)：PyNv I/O 包装的生产实现
- [reference/PTMediaServer/pipeline/pynv_stream.py](../reference/PTMediaServer/pipeline/pynv_stream.py)：完整解码-处理-编码-mux 流水线参考
- [reference/PTMediaServer/utils/video_metadata.py](../reference/PTMediaServer/utils/video_metadata.py)：源路由决策逻辑
- [reference/PTMediaServer/utils/gpu_runtime_cache.py](../reference/PTMediaServer/utils/gpu_runtime_cache.py)：CUDA / CuPy / ORT 启动暖机
- [reference/PTMediaServer/utils/runtime_dll_paths.py](../reference/PTMediaServer/utils/runtime_dll_paths.py)：onedir 打包后 CUDA DLL 路径处理
- [reference/PTMediaServer/build_exe.py](../reference/PTMediaServer/build_exe.py)：PyInstaller 打包脚本
- [reference/PTMediaServer/offline/convert.py](../reference/PTMediaServer/offline/convert.py)：offline 转换 CLI 入口（路由对齐参考）
- [reference/PTMediaServer/tools/pynv_transcode_probe.py](../reference/PTMediaServer/tools/pynv_transcode_probe.py)：纯 PyNv 转码 probe，可作为 benchmark 起点
- [reference/PTMediaServer/PROJECT.md](../reference/PTMediaServer/PROJECT.md)：PTMediaServer 总体说明
