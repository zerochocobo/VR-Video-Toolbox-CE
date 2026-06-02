# Lada 模型 ONNX 转换 / 融合可行性研究

- 文档日期：2026-05-30
- 适用项目：VR_Video_Toolbox_NE
- 参考：reference/lada（codeberg.org/ladaapp/lada）
- 目标模型：
  - 检测：`models/lada_vr_mosaic_detection_model_v2_accurate.pt`（90MB）
  - 恢复：`models/lada_mosaic_restoration_model_generic_v1.2.pth`（78MB）
- 状态：**仅研究，未动手**；供讨论

---

## 0. 结论速览（先看这里）

1. **lada 现在就已经全程跑 GPU**（PyTorch CUDA）。它"慢/不够融合"的真正原因不是没上 GPU，而是：
   - 作为**独立子进程**调用（`lada-cli`），每个文件重新加载模型；
   - 内部用 **ffmpeg 解码/编码**（不是我们的 PyNv）；
   - one_click 里和我们的 v360 步骤之间靠**磁盘中间文件**传递。
2. **检测模型（YOLO11-seg）转 ONNX 很容易**（ultralytics 自带导出），可走 ONNX Runtime / TensorRT。
3. **恢复模型（BasicVSR++ GAN）转 ONNX 很难**：它是**时序递归**网络（沿时间轴的 forward/backward 传播 + 二阶可变形对齐），含动态帧数 T 的 Python 循环、`torchvision.ops.deform_conv2d` 和 `grid_sample`。可行但工作量大、且 TensorRT 对这两个算子支持需要 plugin。
4. **建议分三步**，且**第一步不碰 ONNX**：先把 lada 的 torch 流水线"进程内融合"+ 与我们的 PyNv/CuPy GPU 缓冲对接，干掉子进程和中间文件——这是最大、最稳的实际收益。ONNX 化（尤其想彻底去掉 torch 依赖）放到后面，且恢复模型可能长期保留 torch 或走 TensorRT。

---

## 1. 两个模型是什么

### 1.1 检测：`Yolo11SegmentationModel`（YOLO11 实例分割）

- 文件：`reference/lada/lada/models/yolo/yolo11_segmentation_model.py`
- 基于 **ultralytics YOLO**（`task='segment'`），输入 letterbox 到 640，输出框 + 分割掩码。
- 后处理在 torch/Python：`nms.non_max_suppression` + `ops.process_mask` + `scale_boxes`。
- VR v2 accurate 是较大的 YOLO11-seg 权重（90MB）。
- **每帧都跑**（检测马赛克区域）。

### 1.2 恢复：BasicVSR++ GAN（生成器）

- 文件：`reference/lada/lada/models/basicvsrpp/mmagic/basicvsr_plusplus_net.py`、`basicvsrpp_gan.py`
- **时序视频恢复网络**，不是逐帧模型。结构：
  - **SPyNet** 估光流（`flow_warp` = `grid_sample`）；
  - 4 条传播分支（backward_1 / forward_1 / backward_2 / forward_2），沿时间轴**递归**传播特征；
  - **二阶可变形对齐** `SecondOrderDeformableAlignment`，内部用 **`torchvision.ops.deform_conv2d`**（第 327 行）；
  - 输入 `(1, T, 3, 256, 256)` float[0,1]，输出同形；T = 一个 clip 的帧数（最长 `max_clip_length=180`）。
- 重要：lada **没有依赖 mmcv/mmagic 全家桶**——它把需要的 mmagic 代码**内联**进了 `models/basicvsrpp/mmagic/`，且可变形卷积用的是 **torchvision** 而非 mmcv 的 CUDA 算子。依赖只有 `torch / torchvision / mmengine`（mmengine 仅用于 Config/registry/`load_checkpoint`，可绕开）。

---

## 2. lada 现有运行流水线（关键，决定融合方式）

入口 `cli/main.py` → `restorationpipeline/frame_restorer.py`：

```
视频文件
  └─(ffmpeg VideoReader 解码)→ 每帧
       ├─ MosaicDetector(YOLO11-seg)：逐帧检测马赛克 → 掩码/框
       │     └─ 时序跟踪：把跨帧的马赛克区域裁成 256×256，按时间聚成 Clip（最长180帧）
       ├─ Clip 恢复(BasicVSR++)：对整个 Clip 时序恢复 → 复原的 256×256 序列
       └─ 帧合成：用 blend mask 把复原区域贴回原帧
  └─(ffmpeg VideoWriter 编码)→ 输出文件 + 合并音频
```

- 多线程队列流水线（检测 / clip恢复 / 帧合成 三个 worker）。
- 关键数据结构 `Clip`：一段时间内某个马赛克区域的裁剪序列 + 掩码 + 原始位置框。
- **检测和恢复都在 GPU（torch CUDA）**；解码/编码/音频在 ffmpeg 子进程。
- 支持 `device`（cuda）、`fp16`，检测的 `_preprocess_gpu` 已能吃 GPU 张量。

---

## 3. ONNX 转换可行性（逐模型）

### 3.1 检测 YOLO11-seg → ONNX：**容易** ✅

- ultralytics 自带：`YOLO(model).export(format='onnx', dynamic=True, half=True)`，或 `format='engine'` 直接出 TensorRT。
- 算子标准（Conv/SiLU/Concat/Upsample/分割原型），ONNX Runtime / TensorRT 均良好支持。
- NMS + `process_mask`（掩码组装）保留在我们侧用 CuPy/numpy 实现，或用 ultralytics 的 end2end 导出把 NMS 也塞进图。
- 风险：低。这是成熟路径。

### 3.2 恢复 BasicVSR++ → ONNX：**难，但可行** ⚠️

阻碍点：

1. **动态时序递归循环**：`propagate()` 沿 T 做 Python `for` 循环，特征列表动态增长、负索引、`[::-1]` 反转、跨帧状态 `feat_prop`。
   - `torch.onnx` **trace** 会把 T **固定**展开成超大静态图（T 不能再变）；
   - `torch.jit.script` 对"dict 套增长列表 + 负索引 + 反转"很难成功。
   - 解决方向：改写成**固定窗口 T**（比如把 clip 统一 pad 到固定长度，或滑窗），导出固定-T 的 ONNX。代价：图巨大、灵活性差、需要改网络前向逻辑。
2. **`torchvision.ops.deform_conv2d`**：ONNX 可导（torchvision 注册了 symbolic；ONNX opset 19 有 `DeformConv`），**ONNX Runtime 能跑**；但 **TensorRT 原生不支持，需要自定义 plugin**。
3. **`grid_sample`（flow_warp）**：ONNX opset 16+ 支持，ORT 支持；TensorRT 8.5+ 支持。
4. mmengine 的 registry/config 加载需在导出脚本里绕开（直接实例化 `BasicVSRPlusPlusGanNet` 并 `load_checkpoint`）。

结论：**ORT 路线可行**（固定-T + deform_conv + grid_sample 都能跑），**TensorRT 路线需要 deform_conv plugin**，工作量明显更大。

---

## 4. 融合进本项目的三种方式

> 注意两个独立维度：**(A) 是否进程内融合**（干掉子进程/中间文件）与 **(B) 推理后端**（torch / ONNX Runtime / TensorRT）。二者可组合。

### 方式一：进程内融合 + 保留 torch（**推荐起步**，不碰 ONNX）

- 把 lada 的 `FrameRestorer`（torch）**向量化引入**我们的进程，作为 `mosaic_engine='native_gpu'` 的实现（之前预留的接口槽位）。
- 与我们的 `gpu_engine` 对接：PyNv 解码出的 GPU 帧 →（送检测/恢复，torch CUDA）→ 复原 GPU 帧 → PyNv 编码。**无中间文件、无子进程、模型只加载一次**。
- one_click 的"split+fisheye → lada → fisheye→VR+merge"可变成**一条 GPU 常驻管线**（中间不再落盘）。
- 代价：引入 `torch + torchvision + mmengine`（打包 +约 2–3GB；需与 CuPy/PyNv 共享 CUDA 上下文——可行，PyTorch 和 CuPy 能共用 CUDA，注意流同步）。
- 收益：**最大且最稳**——省掉两次 ffmpeg 解码+编码、子进程启动、磁盘 IO、每文件模型重载。
- 风险：中。torch 与 cupy/pynv 的 CUDA 共存 + 显存占用需调；与现有 lada-cli 行为对齐验证。

### 方式二：检测 ONNX + 恢复保留 torch（混合）

- 检测走 ONNX Runtime（去掉一部分 ultralytics/torch 依赖面），恢复仍 torch。
- 问题：**只要恢复还用 torch，就甩不掉 torch 依赖**，打包体积没省下来。性价比一般，除非检测想独立复用。

### 方式三：双模型全 ONNX / TensorRT（**终极目标，工作量最大**）

- 检测 ONNX/TRT（易）+ 恢复 ONNX（固定-T，难）或 TensorRT（需 deform_conv plugin，更难）。
- 收益：**彻底去掉 torch**（包从 GB 级降到 onnxruntime-gpu 量级），且 TensorRT FP16 可能比 torch eager 快 1.5–3×。
- 风险：高。恢复模型导出是硬骨头；需大量精度/数值对齐验证（GAN 输出对算子实现敏感）。

---

## 5. "全程 GPU" 的真实收益拆解

当前 one_click 单眼鱼眼链路（已 GPU 化我方部分）：

```
[我方 GPU] 解码+split+fisheye → 鱼眼文件
[lada 子进程] ffmpeg解码 → 检测(GPU) → 恢复(GPU) → ffmpeg编码 → 复原鱼眼文件
[我方 GPU] 解码+fisheye→VR+merge → 输出
```

瓶颈分析（凭架构推断，需实测）：
- **检测 + 恢复**（神经网络）是绝对大头，且**已在 GPU**。ONNX/TensorRT 能再榨 1.5–3×（主要 TRT）。
- **v360/crop/编解码**在我方已近乎免费 / 受 NVDEC 限制。
- **可立刻省掉的开销**：lada 的 ffmpeg 解码 + ffmpeg 编码（各一遍）、子进程冷启动 + 模型重载、3 个中间文件落盘/读盘。在 8K 大文件上这些不小。

→ **方式一（进程内融合）不依赖 ONNX 就能拿到"全程 GPU 常驻 + 无中间文件"的主要红利**；ONNX 是在此之上的"再加速 + 去 torch 依赖"增量。

---

## 6. 建议的分步方案（供讨论）

| 步骤 | 内容 | 后端 | 难度 | 主要收益 |
|---|---|---|---|---|
| **A** | 进程内融合 lada `FrameRestorer`，与 PyNv 解/编码 + CuPy 几何对接，干掉子进程/中间文件；实现 `mosaic_engine='native_gpu'` 槽位 | torch CUDA | 中 | 全程 GPU 常驻、无中间文件、模型只载一次（**最大实际提速**） |
| **B** | YOLO11-seg 检测 → ONNX，走 ONNX Runtime-GPU（NMS/掩码我方实现） | ORT | 低 | 验证 ONNX 路线、检测可去 ultralytics/部分 torch | 
| **C**（R&D）| BasicVSR++ → ONNX（固定-T）或 TensorRT（deform_conv plugin）；成功则彻底去 torch | ORT/TRT | 高 | 包体大幅缩小 + TRT 再加速 |

- 若**优先实际速度和无中间文件**：做 A 即可，先不碰 ONNX。
- 若**优先去掉 torch、缩小安装包**：必须做到 C（恢复模型是硬门槛），建议先用一个独立 spike 验证 BasicVSR++ 固定-T 导出 + 数值对齐能不能过，再决定投入。
- B 可作为低风险的 ONNX 试水，但只做 B 不做 C 省不掉 torch。

---

## 7. 待你拍板的问题

1. **首要目标是哪个**？（a）实际处理更快 + 无中间文件；（b）彻底去掉 torch、缩小安装包到 onnxruntime 量级；（c）两者都要。
   - 选 (a) → 直接做方式一（torch 进程内融合），不折腾 ONNX。
   - 选 (b)/(c) → 需要啃恢复模型 ONNX/TensorRT，建议先做导出可行性 spike。
2. **能否接受打包里带 torch+torchvision（+约 2–3GB）**？这是 torch 路线 vs 纯 ONNX 路线的分水岭。
3. **检测和恢复要不要拆开推进**？（检测先 ONNX 化风险低，可作为 ONNX 试点）
4. 是否需要我先做一个**最小可行性 spike**：把 `lada_vr_..._v2_accurate.pt` 导出 ONNX 跑通一帧检测（验证 B 路线），以及尝试 BasicVSR++ 固定-T 导出（验证 C 的硬骨头）——这一步才需要"动手"。

---

## 8. 参考文件

- `reference/lada/lada/cli/main.py` — CLI 入口、模型加载与逐文件处理
- `reference/lada/lada/restorationpipeline/frame_restorer.py` — 核心时序流水线（检测→跟踪→恢复→合成）
- `reference/lada/lada/restorationpipeline/__init__.py` — `load_models()` 路由
- `reference/lada/lada/restorationpipeline/basicvsrpp_mosaic_restorer.py` — 恢复 I/O：`(1,T,3,256,256)`
- `reference/lada/lada/models/yolo/yolo11_segmentation_model.py` — 检测封装（ultralytics）
- `reference/lada/lada/models/basicvsrpp/mmagic/basicvsr_plusplus_net.py` — BasicVSR++ 网络（SPyNet/光流/二阶可变形对齐）
- `reference/lada/lada/models/basicvsrpp/inference.py` — 恢复模型构建/加载（mmengine registry）
- `reference/lada/pyproject.toml` — 依赖：ultralytics 8.4.4 / mmengine 0.10.7 / torch 2.8 / torchvision 0.23（无 mmcv）
