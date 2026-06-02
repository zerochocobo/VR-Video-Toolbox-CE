# GPU 流水线重构 — 实现总结

- 文档日期：2026-05-29
- 项目：VR_Video_Toolbox_NE
- 配套方案：`summary/summary_20260529_GPU_PIPELINE_REFACTOR_PLAN_CN.md`
- 状态：阶段 0–6 主体完成；两个 area_selection 的 lada+overlay 工作流延后

---

## 1. 已交付

新增 `gpu_engine/` 包，把"解码→几何变换→编码"全程 GPU 驻留：
PyNvVideoCodec(NVDEC) → CuPy/RawKernel → PyNvVideoCodec(NVENC) → ffmpeg 仅复用音频。

| 模块 | 职责 |
|---|---|
| `gpu_engine/_cuda_env.py` | 包导入时配置 CUDA 环境（PTX + CUDA_PATH 指向 ≥13 toolkit + DLL 搜索路径），解决 Blackwell sm_120 与 PyNv 共存 |
| `gpu_engine/pynv_io.py` | 解码/编码包装：GpuNv12Frame(8bit)/GpuP016Frame(10bit)、PyNvSimpleDecoder、PyNvThreadedSerialDecoder、AppFrame、PyNvEncoderSession |
| `gpu_engine/probe.py` | ffprobe 元数据 + decide_backend 路由（gpu_nv12 / gpu_p016 / ffmpeg_fallback） |
| `gpu_engine/v360_lut.py` | 精确复刻 ffmpeg vf_v360.c：heq↔fisheye、heq→flat(yaw/pitch/d_fov, rorder=ypr 四元数旋转) |
| `gpu_engine/nv12_kernels.py` | 8/10-bit 双线性 LUT 采样 RawKernel + crop/hstack/vstack |
| `gpu_engine/files.py` | 文件层流水线：process_video / process_video_multi（多输出）/ vr_projection / vr_to_flat / split_video / combine_video / extract_clip；CancelToken 取消 |
| `gpu_engine/mux.py` | 裸 HEVC + 源音频 → mp4，hevc_metadata bsf 注入色彩 VUI |
| `gpu_engine/fallback.py` | 三态 backend（auto/gpu/ffmpeg）+ 单文件级自动回退 + OperationCancelled |
| `gpu_engine/runtime.py` | 暖机 / 设备与 NVENC 能力探测 / 内存池回收 |

各工具 `logic.py` 改为「GPU 优先 + 失败回退 ffmpeg」，原 ffmpeg 实现保留为 `_xxx_ffmpeg`：

| 工具 | 状态 | GPU 化内容 |
|---|---|---|
| tool_v360_trans | ✅ 完成 | hequirect↔fisheye，单目/SBS 双目 |
| tool_vr2flat | ✅ 完成 | VR→flat（yaw/pitch/d_fov） |
| tool_split_combine | ✅ 完成 | split（6 模式 ±fisheye 双输出）、combine（±fisheye） |
| one_click | ✅ 完成 | 6 个 helper 全 GPU 化；split/fisheye/merge/投影/时间段截取；lada/jasna 保持外部 CLI；全流程 end-to-end 验证 |
| area_selection_vr2flat | ⏸ 延后 | 含 lada + alpha overlay 贴回合成 |
| area_selection_rect_crop | ⏸ 延后 | run_pipeline 同上 |

打包：`VR_Video_Toolbox.spec`（onedir，无 UPX）+ `packaging/hook-cupy.py` + `packaging/runtime_hook_cuda.py` + 重写 `build_exe.bat`（拷贝 CUDA13 DLL/头文件）。
配置：`utils/app_config.py` 新增 `transcode_backend`/`mosaic_engine`/`gpu_log_verbose`；`main.py` 启动后台暖机；`utils/engine_runner.py` 留 native_gpu 占位。

---

## 2. 验证结果

| 项 | 结果 |
|---|---|
| 解码逐位精确（PyNv vs ffmpeg）| Y/UV 100% 匹配（99dB） |
| 几何 vs ffmpeg（裸像素，多帧，ThreadedDecoder 对齐）| heq↔fisheye 8bit Y~62/UV~61dB、10bit Y~75/UV~73dB；heq→flat 74–78dB；crop 99dB（bit-exact）。**全部远超门槛**（8bit 42/38、10bit 45/40） |
| 路由 | 8bit→gpu_nv12、10bit bt709→gpu_p016、HDR10/HLG/bt2020→ffmpeg_fallback，全部正确 |
| 回退 | auto 自动回退、gpu 强制不回退、ffmpeg 强制，全部正确 |
| one_click 全流程 | PassA(GPU split+fisheye)→lada 双眼→PassB(GPU fisheye+merge)，0 回退，输出正确 |
| 端到端速度（8K SBS 10bit）| ~2.5–3× vs 原 ffmpeg 路径 |

测试脚本：`tests/verify_phase0.py`、`tests/verify_lut_geometry.py`（几何回归门槛）、`tests/verify_phase2.py`（路由/回退）、`tests/bench_gpu_vs_ffmpeg.py`（测速）。测试素材在 `tests/fixtures/`（8/10bit、bt709/HDR10、hequirect/fisheye）。

---

## 3. 关键技术结论

1. **瓶颈是解码不是几何**：8K 下解码占 ~89%，CuPy 几何变换仅 ~1%。必须用 ThreadedDecoder 顺序解码（非 SimpleDecoder 随机访问）。端到端提速来自消除 CPU v360，受 NVDEC 硬件解码上限制约（ffmpeg 也付这个底）。
2. **几何精确匹配 ffmpeg**：v360_lut.py 逐字复刻 vf_v360.c（rescale/scale/各投影函数、d_fov→fov、四元数旋转）。默认 fov=180、BILINEAR、无旋转。每平面独立 LUT（Y 全分辨率、色度半分辨率）。
3. **真 10-bit 保真**：P016 解码 → CuPy uint16 → NVENC「P010」编码（注意：编码器格式名是 P010 不是 P016；AppFrame 平面 typestr 必须覆盖为 `|u2`，CuPy 默认 `<u2` 被拒）。不像 PTMediaServer 那样降成 8bit。
4. **色彩 VUI 透传**：`-c:v copy` 不注入裸 HEVC 的 VUI，需用 `hevc_metadata` bsf 写入 colour_primaries/transfer/matrix/range。
5. **环境共存（开发机 RTX 5060 Ti, Blackwell sm_120, CUDA13 驱动）**：CuPy 需 CUDA13 nvrtc+头文件（同 toolkit）+ `CUPY_COMPILE_WITH_PTX=1`；PyNv 需 CUDA12.x cudart。冲突点在 CUDA_PATH——改 CUDA_PATH 会破坏 PyNv，但 add_dll_directory 不会。方案：CUDA_PATH 指 v13、把各 toolkit bin 都 add_dll_directory。依赖 `cupy-cuda13x`（非 12x，12x 无 sm_120 kernel）。

---

## 4. 延后项与后续

- **area_selection_vr2flat / area_selection_rect_crop 的 run_pipeline**：是「选区→提取→lada→alpha 叠加贴回原 VR」的合成工作流。提取/贴回的 v360 可 GPU 化，但 alpha 蒙版生成 + overlay 合成是我当前引擎尚未支持的能力，且贴回对亚像素精度敏感（接缝）。建议作为独立子项目，与 native_gpu 一起规划。
- **native_gpu 自研去马赛克引擎（phase 7+）**：已留接口槽位（engine_runner 抛 NotImplementedError、配置项占位）。届时新增 `gpu_engine/native_mosaic/` 子包，把 `frames.mosaic_restore` 接进 one_click 帧流水线（可做到中间不落盘），并加 onnxruntime-gpu/tensorrt 依赖。
- **打包构建：已验证可用**。onedir 构建成功，frozen exe 实测 `--selftest-gpu` 通过：gpu_available=True、CuPy **RawKernel JIT OK**、nvenc_hevc_10bit=True（在 Blackwell sm_120 上）。踩坑与定论：
  - `graphlib`（stdlib）被 PyInstaller 漏掉 → 加进 spec hiddenimports。
  - CuPy 用 `dirname(dirname(nvrtc))` 推导 CUDA_PATH 并 `add_dll_directory(CUDA_PATH\bin)`：nvrtc 必须**只**放在 `_internal\bin\`（放 `_internal` 根会让 pathfinder 经 site-packages 找到、推导出 exe 目录、拼出不存在的 `exe\bin` 而崩）。
  - 需把 CUDA13 的 nvrtc + cudart64_13 + cublas/cublasLt/cufft/cufftw/curand/cusolver/cusolverMg/cusparse + nvJitLink 全部拷到 `_internal\bin\`，头文件拷到 `_internal\include\`（nvrtc JIT 要 cuda_fp16.h 等）。build_exe.bat 已自动完成。
  - runtime_hook_cuda.py 把 CUDA_PATH 指向含 `bin\nvrtc`/`include` 的 `_internal` 目录（兼容 _MEIPASS 为 exe 目录或其 _internal 子目录两种布局）。
  - models/ 大模型目录不打包，放 exe 同目录。`VR_Video_Toolbox.exe --selftest-gpu` 可在任意机器自检 GPU。
  - 提醒：若 GUI 实际跑某工具时报缺 app 侧模块（faster-whisper/av 等的隐藏导入），按提示补 spec hiddenimports 即可——GPU 链路本身已验证。
