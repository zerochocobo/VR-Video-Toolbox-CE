# 内置去马赛克引擎（native_gpu）实施计划

- 文档日期：2026-05-30
- 适用项目：VR_Video_Toolbox_NE
- 方案：第 5 节研究报告的「方式一（进程内融合 + 保留 torch）」
- 前置研究：`summary/summary_20260530_LADA_ONNX_FUSION_RESEARCH_CN.md`
- 状态：**计划，供随时接手**。torch 已装好并验证，下面是动手清单。

---

## 0. 目标与边界

**目标**：把 lada 的 torch 去马赛克流水线（YOLO11-seg 检测 + BasicVSR++ 恢复）**进程内**集成为第三个引擎 `native_gpu`（界面叫"内置"），与我们的 `gpu_engine`（PyNv 解/编码 + CuPy 几何）协同，做到：
- 不再调 `lada-cli` 子进程；
- 模型只加载一次（跨文件复用）；
- 用我们的 PyNv 编码 + 音频 mux 输出（不再走 lada 的 ffmpeg 编码）；
- one_click 等流程减少中间文件与重复解/编码。

**不做**（本期）：模型转 ONNX/TensorRT、去掉 torch 依赖。这些是后续可选项（研究报告"方式三"）。

**保留**：lada / jasna 两个外部 CLI 引擎选项不变；native_gpu 作为新增第三项。

---

## 1. 环境（已完成）

- 已装：`torch==2.8.0+cu128`、`torchvision==0.23.0+cu128`（cu128 含 Blackwell sm_120 kernel）、`ultralytics==8.4.4`、`mmengine==0.10.7`。
- pyproject 已配 `[[tool.uv.index]] pytorch-cu128 (explicit)` + `[tool.uv.sources]` 钉 torch/torchvision。
- **已验证**：torch.cuda 在 sm_120 可用；torch + cupy-cuda13x + PyNvVideoCodec 同进程共存。
- **关键约束（务必遵守）**：**任何 `import torch` 之前，CuPy 必须先编译过至少一个 kernel**（否则 torch 的 nvrtc-builtins 12.8 与 CuPy nvrtc 13.0 撞车，编译 cuda_fp4.hpp 报错）。详见 memory `cuda-env-blackwell-coexistence`。落实方式见 §4.1。

---

## 2. lada 关键结构（接手必读）

- 入口逻辑：`reference/lada/lada/cli/main.py` → `restorationpipeline/frame_restorer.py`。
- `restorationpipeline/__init__.py: load_models(device, restoration_name, restoration_path, config, detection_path, fp16, detect_face)`：加载检测(YOLO11-seg)+恢复(BasicVSR++)，返回 `(detection_model, restoration_model, pad_mode)`。
- `FrameRestorer(device, video_file, max_clip_length, restoration_name, detection_model, restoration_model, pad_mode)`：
  - 多线程队列流水线：检测 worker → clip 恢复 worker → 帧合成 worker；
  - **文件进、帧出**：内部用自己的 `VideoReader`（ffmpeg）解码 `video_file`；通过 `for (restored_frame_bgr, pts) in frame_restorer` 产出复原帧（numpy/torch，BGR）；
  - `.start()/.stop()`，可 `start(start_ns=)` 定位。
- 检测：`models/yolo/yolo11_segmentation_model.py`（ultralytics）。恢复：`models/basicvsrpp/`（SPyNet 光流 + 二阶可变形对齐，输入 `(1,T,3,256,256)` float[0,1]）。
- 依赖：torch / torchvision / mmengine（mmcv 未用，可变形卷积走 torchvision；mmagic 代码已内联进 `models/basicvsrpp/mmagic/`）。

---

## 3. 模块结构（新增）

```
gpu_engine/
└── native_mosaic/
    ├── __init__.py          # 对外: get_engine(), restore_file(...), available()
    ├── _lada_vendor/        # 从 reference/lada/lada 拷入运行所需子集（见 §3.1）
    │   ├── restorationpipeline/   (frame_restorer, mosaic_detector, basicvsrpp_mosaic_restorer, __init__)
    │   ├── models/ (yolo/, basicvsrpp/ 含 mmagic/)
    │   └── utils/  (lada 运行需要的 utils：image_utils, video_utils, mask_utils, box_utils, scene_utils,
    │               threading_utils, ultralytics_utils, torch_letterbox, audio_utils, os_utils, __init__ 里的类型别名/ModelFiles)
    ├── engine.py            # NativeMosaicEngine: 持有已加载模型(单例), 提供 restore 接口
    └── models_cfg.py        # 模型路径解析(默认指向项目 models/ 下的 VR v2 + generic v1.2)
```

> 为什么 vendor 而不是 `import lada`：lada 不是 pip 包（项目本地源码 + 打了补丁的 ultralytics/mmengine），且其 `lada/__init__.py` 有 Flatpak/翻译等环境假设。拷贝运行所需子集进 `_lada_vendor/`，去掉 GUI/datasetcreation/cli/翻译依赖，最稳。保留其 AGPL 许可头与 LICENSE（合规，见 §7）。

### 3.1 需要 vendor 的最小集合（接手时按 import 报错增量补齐）

确定需要：`restorationpipeline/{frame_restorer,mosaic_detector,basicvsrpp_mosaic_restorer,__init__}.py`、`models/yolo/*`、`models/basicvsrpp/**`（含 `mmagic/**`、`deformconv.py`、`inference.py`）、`utils/**` 中被引用者、顶层 `lada/__init__.py` 里被用到的 `VERSION/ModelFiles/LOG_LEVEL/ImageTensor/Image/Box/...`（多为类型别名，可抽到一个精简 `_lada_vendor/__init__.py`）。

**不要** vendor：`gui/`、`datasetcreation/`、`cli/`、`models/bpjdet/`、`models/deepmosaics/`（除非要支持 deepmosaics 恢复模型——本期只用 basicvsrpp）。

---

## 4. 实现步骤

### 4.1 `native_mosaic/__init__.py` + `engine.py`：模型单例 + import 顺序守卫

```python
# native_mosaic/__init__.py（伪代码）
def _ensure_cupy_first():
    # 关键：torch 之前先让 CuPy 编译一次，锁住 nvrtc 栈
    from gpu_engine import runtime
    runtime.warmup()           # 内部已编译 cupy kernel
def available() -> bool: ...   # torch.cuda.is_available() 且模型文件存在
```

- `NativeMosaicEngine`（engine.py）：
  - 构造时 `_ensure_cupy_first()` → 然后才 `import torch` + lada vendor；
  - `load_models()` 一次，缓存 `(detection_model, restoration_model, pad_mode)`；按 `fp16 = gpu_has_fp16_acceleration()`；
  - 进程内单例（首次用时构建，跨文件复用，避免每文件重载 ~170MB 权重）。
- 模型路径默认（models_cfg.py）：
  - 检测：`models/lada_vr_mosaic_detection_model_v2_accurate.pt`
  - 恢复：`models/lada_mosaic_restoration_model_generic_v1.2.pth`
  - 允许 app_config 覆盖（custom_args 已有机制）。

### 4.2 `engine.restore_file(input_path, output_path, log_callback, cancel_token)`（第一步：drop-in 替换子进程）

最小可行、最大复用：保留 lada 的 `FrameRestorer`（它自己解码 input_path），只把**输出侧**换成我们的 PyNv 编码 + 音频 mux。

```
FrameRestorer(device, input_path, max_clip_length, 'basicvsrpp', det, restore, pad).start()
for (restored_bgr_frame, pts) in frame_restorer:   # BGR uint8 numpy/tensor
    # → 上传/转 NV12 或 P016（CuPy）→ PyNv 编码（沿用 gpu_engine 的 _EncodeSink 生命周期安全）
mux 音频（gpu_engine.mux，从 input_path 取音频）
```

- 颜色：lada 产 BGR uint8；需 BGR→NV12/P010（CuPy kernel，或先 8-bit 跑通）。**注意 10-bit**：lada 内部按 8-bit 处理（YOLO+BasicVSR 都 8-bit），所以 native_gpu 恢复段是 8-bit；若源是 10-bit，恢复段会降到 8-bit——**这点要么接受（lada 本身如此），要么在计划评审时确认**。
- 取消：FrameRestorer 有 `.stop()`；接 CancelToken。
- 进度：复用 `gpu_engine._Progress` 或 lada 帧数回调 → log_callback。
- 收益（此步）：去子进程、模型只载一次、用我们的 NVENC 编码。**中间文件仍存在**（FrameRestorer 读 input_path 文件）。

### 4.3 接线：`utils/engine_runner.py` + one_click `process_lada()`

- `utils/engine_runner.py`：`get_engine()=='native_gpu'` 不再 `NotImplementedError`；新增 `get_mosaic_tool_name()` 返回"内置"。`build_engine_cmd` 对 native_gpu 不构建 CLI（返回 None 或抛专用标记）。
- one_click `process_lada(input, output, ...)`：
  ```
  if app_config.get_engine()=='native_gpu':
      from gpu_engine.native_mosaic import engine
      engine.restore_file(input, output, log_callback, cancel_token)   # 进程内
  else:
      cmd = engine_runner.build_engine_cmd(...)   # lada/jasna CLI（现状）
      run_process(cmd, ...)
  ```
- 其它调用 process_lada 的工具（area_selection_* 等）自动受益。

### 4.4 前端"内置"选项（config_sidebar / 引擎下拉）

- `utils/app_config.py`：`mosaic_engine`/`engine` 取值增加 `native_gpu`（默认仍 lada）。
- 首页引擎选择 UI（lada/jasna 单选/下拉）增加第三项「内置」→ 写 `engine='native_gpu'`。之前 one_click 已加的「画质/速度」预设对 native_gpu 的 NVENC 输出仍生效。
- 依赖检查：native_gpu 时不要求 `lada-cli`/`jasna-cli` 存在，改为检查 torch.cuda + 模型文件存在。

### 4.5（后续，第二步）全 GPU 常驻：去中间文件

- 把 FrameRestorer 的**输入侧**从"读文件"改成"吃我们的 GPU 帧生成器"：one_click 的 split+fisheye 直接产 GPU 帧 → 喂检测/恢复 → 复原帧 → fisheye→VR/merge → PyNv 编码。全程不落盘。
- 需要改 lada `FrameRestorer._frame_restoration_worker`（目前用 VideoReader + seek），或抽象出"帧源"接口。**较深，作为独立子任务**。第一步（4.2）跑通并验证后再做。

---

## 5. 验证

1. **共存自检**：`VR_Video_Toolbox.exe --selftest-gpu` 仍过；新增检查 torch.cuda + 内置引擎可加载模型。
2. **正确性对齐**：同一段 VR 鱼眼片，分别用 (a) lada-cli 子进程 与 (b) native_gpu 进程内，比较复原区域画质/位置一致（PSNR 或肉眼；注意编码器不同，比较解码后对齐帧或抽帧）。
3. **8K 实测**：单眼鱼眼 one_click，记录端到端耗时、显存峰值（torch+cupy+pynv 同驻）、有无绿块。对比改造前（lada 子进程）。
4. **回归**：lada / jasna 两个旧引擎仍正常（未受影响）。
5. **取消/异常**：处理中途停止能干净中止；模型缺失/torch 不可用时 native_gpu 给明确提示（不静默崩）。

---

## 6. 风险与缓解

| 风险 | 缓解 |
|---|---|
| **import 顺序**导致 CuPy 编译崩 | native_mosaic 在 `import torch` 前调 `runtime.warmup()`；main.py 启动暖机早于 torch；§1 约束写进代码注释 |
| torch+cupy+pynv 显存竞争（8K） | 模型 fp16；clip 长度/批量可调（max_clip_length）；必要时处理完释放 torch/cupy 缓存 |
| 10-bit 源经 lada 降 8-bit | 评审确认可接受（lada 本身 8-bit）；或恢复段仅对裁剪小块 8-bit、其余链路保 10-bit（合成时注意） |
| vendor lada 子集 import 断裂 | 按报错增量补 utils；保留 AGPL 头 |
| 打包体积/复杂度（torch ~2-3GB + CUDA12.8） | onedir；torch 自带 CUDA 运行时；与现有 CUDA13/PyNv DLL 共存按 memory 处理；体积评审 |
| 与 ultralytics/mmengine 的补丁差异 | lada 对 ultralytics/mmengine 打过补丁；vendor 时如遇行为差异，参考 lada 仓库补丁说明 |

---

## 7. 许可合规（重要）

lada 是 **AGPL-3.0**。vendor 其源码进本项目意味着**本项目相应部分需遵循 AGPL**（或整体兼容）。接手前请与产品方确认许可策略：保留 lada 文件的 SPDX/版权头、附带 LICENSE、并评估对本项目分发许可的影响。这是**法律前置项**，不是技术项。

---

## 8. 交接 checklist（按序）

- [ ] T10 接通 `engine_runner`/config/前端「内置」选项（可先桩接，engine 未实现时给提示）
- [ ] T11 建 `gpu_engine/native_mosaic/`，vendor lada 子集，模型单例加载，`_ensure_cupy_first` 守卫
- [ ] T11 `engine.restore_file()`：FrameRestorer → PyNv 编码 + mux（§4.2，第一步）
- [ ] T12 one_click `process_lada` 路由 native_gpu；验证（§5）
- [ ] （后续）§4.5 全 GPU 常驻去中间文件
- [ ] 许可合规确认（§7）

---

## 9. 参考

- 研究报告：`summary/summary_20260530_LADA_ONNX_FUSION_RESEARCH_CN.md`
- GPU 流水线总方案：`summary/summary_20260529_GPU_PIPELINE_REFACTOR_PLAN_CN.md`
- memory：`gpu-engine-architecture`、`cuda-env-blackwell-coexistence`（含 torch 共存 import 顺序坑）
- lada：`reference/lada/lada/restorationpipeline/`、`reference/lada/lada/models/`
- 我方引擎：`gpu_engine/`（files.py 的 `_EncodeSink`、mux.py、runtime.py、pynv_io.py）
