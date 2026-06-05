# tool_2dvr 视频时序稳定性改造计划

日期: 2026-06-05
作者: 研究阶段交付
目标: **DA3-Small 单帧推理下，消除/显著减弱视频转换中的 depth 抖动、视差呼吸、羽化边缘闪烁**
前置: 建议在 FPS 优化计划 P0/P1 完成后再做（避免性能未达标时改稳定性反复调参）
范围: `tool_2dvr/logic.py` 为主，必要时小改 `_vendor/da3/depth_anything_3/api.py`

---

## 一、抖动来源拆解

| 来源 | 原因 | 视觉表现 |
| --- | --- | --- |
| **A. DA3 单帧 depth 噪声** | 模型本身无时序输入，低纹理区每帧独立预测 | 静止画面里天空/墙面/肤色"沙沙"抖 |
| **B. 逐帧 lo/hi 归一化** | `_normalize_near` 每帧重算 5/95 百分位 | 整张画面整体呼吸式明暗变化 |
| **C. disparity round 量化** | `_forward_warp_eye` 用 `torch.round(target_x)` 整数化 | 边缘像素级跳变 |
| **D. hole mask 跨帧不一致** | mask 由 round 后的 target_x 决定，每帧不同 | soft_shift 羽化区域"一帧有一帧没" |
| **E. 缺乏运动补偿** | 没有用相邻帧 depth | 快速摄影机移动时一致性更差 |

A 是模型属性；**B 是被严重低估的主因**；C/D 互相耦合；E 是天花板问题。

---

## 二、改造分级（按性价比排序）

### S0 — 必做三件套（消除 90% 肉眼可见闪烁）

**S0-1 跨帧 EMA 归一化（修 B）**
- 文件：`tool_2dvr/logic.py::TorchStereoRenderer`
- 改动：
  - 类增加状态字段 `self.lo_ema: torch.Tensor | None = None`、`self.hi_ema: torch.Tensor | None = None`、`self.norm_alpha: float = 0.15`
  - `_normalize_near` 内：当前帧算出的 `lo_frame, hi_frame` 不直接用，而是
    ```
    if self.lo_ema is None:
        self.lo_ema, self.hi_ema = lo_frame, hi_frame
    else:
        self.lo_ema = α·lo_frame + (1-α)·self.lo_ema
        self.hi_ema = α·hi_frame + (1-α)·self.hi_ema
    near = ((inv - self.lo_ema) / (self.hi_ema - self.lo_ema)).clamp(0,1)
    ```
  - batch 内必须**顺序累计**（不能用 batch quantile 一次性算后覆盖），因此 batch 维上 unroll 一个轻量 Python 循环（每次只是 4 个标量更新，可接受）
  - 提供环境变量 `TOOL_2DVR_NORM_ALPHA` 调节（默认 0.15；0=不更新、1=纯当前帧）
  - 首批 N 帧（如 4 帧）用 batch 整体 quantile 做"暖机"，避免初值偏离

**S0-2 depth tensor EMA（修 A）**
- 文件：`tool_2dvr/logic.py::TorchStereoRenderer`
- 改动：
  - 类增加 `self.depth_ema: torch.Tensor | None = None`、`self.depth_beta: float = 0.5`
  - 在 `render_batch` 收到 depth tensor 后、`_smooth_depth` 之前做：
    ```
    for i in range(B):
        if self.depth_ema is None:
            self.depth_ema = depth_t[i:i+1].clone()
        else:
            self.depth_ema = β·depth_t[i:i+1] + (1-β)·self.depth_ema
        depth_t[i:i+1] = self.depth_ema
    ```
  - 升级版（推荐）：**逐像素自适应 β**（1€ filter 思路）
    ```
    diff = (depth_now - depth_ema).abs()
    beta_map = sigmoid((diff - thresh) * slope)   # 抖动小→小β，运动大→大β
    depth_ema = beta_map·depth_now + (1-beta_map)·depth_ema
    ```
  - 环境变量 `TOOL_2DVR_DEPTH_BETA`（默认 0.5）
  - 注意场景切换：如果 `diff.mean() > scene_cut_thresh`，强制重置 ema = depth_now

**S0-3 disparity sub-pixel splat（修 C+D）**
- 文件：`tool_2dvr/logic.py::TorchStereoRenderer._forward_warp_eye`
- 现状：`torch.round(target_x).long()` 后整数 scatter
- 改为 bilinear splat：
  ```
  tx = pixel_x + near * (max_shift * 0.5 * eye_sign)
  tx0 = tx.floor().long().clamp(0, W-1)
  tx1 = (tx0 + 1).clamp(0, W-1)
  w1 = (tx - tx0.float())
  w0 = 1.0 - w1
  # 两次 scatter_add_，权重图归一
  out = scatter_weighted(frame, [tx0, tx1], [w0, w1], priority=near)
  weight = scatter_add weight
  out = out / weight.clamp(min=eps)
  holes = weight < hole_thresh   # 阈值化代替严格"没人写"
  ```
- 这一改会让 hole mask 变成"软"边界（小数权重），soft_shift 羽化区域自然在时序上稳定
- 注意：z-buffer 仍要保留（近物覆盖远物），用 `priority` 加权或两阶段（先 amax 选近物，再 splat）

**S0 验收**
- 同一段静态镜头 30s，整体亮度（near 均值）跨帧标准差 < 改造前的 30%
- 静态画面 SSIM 跨帧 > 0.98（改造前可能 0.90 左右）
- 快速横摇镜头主观无明显拖影

---

### S1 — 进阶（消除残留闪烁，处理快速运动场景）

**S1-1 batch 内联合归一化（与 S0-1 互补）**
- 文件：同 S0-1
- 改动：S0-1 是跨 batch EMA，本项再加一层 batch-内联合：把整个 batch 的所有 inv_depth 一次 flatten 算 lo/hi，作为"当前 batch 的目标"再去做 EMA 更新。能进一步减少 batch 内首末帧不一致
- 一行实现，与 S0-1 同函数

**S1-2 hole mask 时序膨胀（与 S0-3 互补，可选其一）**
- 如果不做 S0-3 的 sub-pixel splat，可以用更轻量的：
  - 记录 `self.prev_left_holes`、`self.prev_right_holes`
  - 当前帧 mask = `current_mask | prev_mask` 再做一次 1px 腐蚀
- 副作用：羽化区域略大；好处：mask 在时序上"加性平滑"
- 建议仅作为 S0-3 的临时替代

**S1-3 光流引导的 depth warp（核心稳定大招）**
- 文件：`tool_2dvr/logic.py` 新增 `_optical_flow.py` 模块
- 实现：
  - 用 OpenCV `cv2.calcOpticalFlowFarneback`（CPU 轻量，4K 下采到 1280×720 算光流再放大也够）
  - 或 PyTorch RAFT-small（更准但更重，留 P 选）
  - 流程：
    ```
    flow_{t-1→t} = compute_flow(rgb_{t-1}, rgb_t)
    depth_warped = warp(depth_ema_{t-1}, flow_{t-1→t})
    depth_t = β·depth_now + (1-β)·depth_warped
    ```
  - 用 `F.grid_sample` 在 GPU 上做 warp
- 处理"非遮挡区域用 warp、遮挡区域用 raw"：用 forward-backward flow consistency check 出 occlusion mask
- 这是学术界 video depth stability 的标准做法（VDA / ChronoDepth / Stable Video Depth 同类思路）
- 替代/升级 S0-2 的"无运动补偿 EMA"

**S1-4 场景切换检测**
- 在 S0-2 / S1-3 任意一种 EMA 路径上挂检测器
- 简易：连续两帧 `rgb` 直方图 chi-square 距离 / depth_ema 与 depth_now 的 L1 均值
- 检测到切换：清空 `lo_ema/hi_ema/depth_ema/prev_holes`，下一帧重新暖机
- 避免切换后整段 EMA 飘移

**S1 验收**
- 含快速摄影机推拉/横摇 + 场景切换的 30s 测试片：主观无"果冻感"和切换后残影
- 暗场景到亮场景切换 < 5 帧完成 EMA 重置

---

### S2 — 改 DA3 利用相邻帧（天花板方案）

**S2-1 多帧 DA3 输入（利用 vendored backbone 多视图能力）**
- 文件：`tool_2dvr/_vendor/da3/depth_anything_3/api.py` + `model/da3.py`
- 背景：DA3 backbone 本来就支持 `x: (B, N, 3, H, W)` 多视图，现在 `_prepare_model_inputs` 强制 `N=1`
- 改动：
  - 滑窗 3 帧（t-1, t, t+1）作为 N=3 送入 backbone
  - head 输出选中间一帧 depth 作为最终结果
  - cross-attention 自然在 token 维度对齐相邻帧
- 注意：
  - 显存增加约 3 倍（504×N×14 token 数 ×3）；batch_size 需对应缩小
  - 必须验证 head 在 N>1 时的形状行为是否合理（DA3 设计上对多视图友好）
  - 在 inference_depth_only 路径加 `temporal_window: int = 1` 参数
- 收益：从源头解决 flicker，是最"正确"的方案

**S2-2 joint bilateral depth upsample（替代纯 F.interpolate）**
- 在 P0-2 把 depth 504→4K 改为 GPU 上采后，用 **RGB 引导**的 joint bilateral 上采，让边缘吸附到颜色边缘
- 实现：6×6 邻域，权重 = 空间高斯 × 颜色高斯
- 既能在空间上锐化深度边缘，又能时序上更稳（颜色稳定→深度稳定）
- 算力：4K 6×6 一次卷积，GPU 上 ~5ms/帧
- 可单独使用，无依赖

**S2 验收**
- 与 S0+S1 比较：肉眼可分辨的剩余抖动应基本消失
- 跨帧 SSIM > 0.99（静态镜头）

---

## 三、提交顺序与里程碑

| 里程碑 | 包含项 | 验收 | 预计工时 |
| --- | --- | --- | --- |
| T1 | S0-1, S0-2 | 静态镜头亮度呼吸消除 | 1 天 |
| T2 | S0-3 | 边缘 sub-pixel 抖动消除 | 1.5 天 |
| T3 | S1-1, S1-4 | batch 内一致 + 场景切换不残影 | 0.5 天 |
| T4 | S1-3（光流） | 含快速运动场景稳定 | 2.5 天 |
| T5（可选）| S2-1 | 多帧 DA3，从源头降噪 | 2 天 |
| T6（可选）| S2-2 | 边缘锐化 + 上采质量提升 | 1 天 |

最低交付：T1 + T2 + T3，已能覆盖 80% 用户场景。

---

## 四、参数与默认值建议

| 参数 | 默认值 | 范围 | 说明 |
| --- | --- | --- | --- |
| `TOOL_2DVR_NORM_ALPHA` | 0.15 | 0.05–0.5 | lo/hi EMA 系数，越小越稳越慢响应 |
| `TOOL_2DVR_DEPTH_BETA` | 0.5 | 0.2–0.8 | depth EMA 系数 |
| `TOOL_2DVR_ADAPTIVE_BETA` | 1 | 0/1 | 是否启用 1€ filter 自适应 β |
| `TOOL_2DVR_HOLE_THRESH` | 0.3 | 0.1–0.5 | sub-pixel splat 后的 hole 阈值 |
| `TOOL_2DVR_FLOW_BACKEND` | farneback | farneback/raft | S1-3 光流后端 |
| `TOOL_2DVR_SCENE_CUT_THRESH` | 0.4 | 0.2–0.8 | 场景切换检测阈值（depth L1 均值） |
| `TOOL_2DVR_TEMPORAL_WINDOW` | 1 | 1/3/5 | S2-1 DA3 滑窗大小 |

所有项都应通过环境变量可关闭，便于回归对比。

---

## 五、风险与回归测试清单

- **EMA 状态污染**：每次新的 `convert_2d_to_vr` 调用必须重新构造 `TorchStereoRenderer`（或显式 reset）；不能在多次转换间残留
- **裁剪片段 start/end 后**：EMA 暖机帧应在裁剪后起始，不应使用裁剪前帧
- **batch fallback**：CUDA OOM 二分递归路径里，EMA 不能在递归中重置
- **CPU 渲染 fallback**：S0-1/S0-2 必须在 CPU 路径同步实现（或明确文档：CPU 路径不带稳定性增强）
- **debug eye 输出**：保留稳定性增强 OFF 的旁路 debug，便于对比
- **测试集**：
  - 完全静态镜头（30s）：检验"无运动时是否真的不抖"
  - 慢速 pan：检验"运动时不拖影"
  - 快速 cut：检验场景切换重置
  - 暗场到亮场过曝转换：检验 lo/hi EMA 不爆
  - 含字幕烧入的电影片段：检验小目标（字符）是否被 EMA 模糊掉

---

## 六、不做什么（明确范围）

- 不引入需要训练/微调的模型
- 不替换 DA3-Small
- 不改动 stereo 几何（max_disparity_pixels、eye_distance_mm 公式）
- 不改 SBS 输出布局
- 不引入 ONNX/TensorRT（属于 FPS 计划）

---

## 七、与 FPS 计划的协同

| FPS 计划项 | 与稳定性计划的耦合 |
| --- | --- |
| FPS P0-1（裁剪 DA3 路径） | 无冲突，独立模块 |
| FPS P0-2（depth GPU 上采样） | **本计划 S2-2 直接基于此** |
| FPS P1-1（normalize_near 去循环） | **必须与 S0-1 协同实现**：本计划要求 EMA 顺序累计，需要保留 batch 内 Python 循环但只算极轻量标量更新；P1-1 的"batch 一次性 quantile"可以作为 S0-1 的"暖机阶段"使用 |
| FPS P1-3（InputProcessor GPU 化） | 无冲突 |
| FPS P2-1（流水线异步化） | EMA 状态属于渲染器，仍在主线程，无并发问题 |

**建议顺序**：先完成 FPS M1+M2（达到 ~45fps），再开始稳定性 T1+T2+T3（不会显著降 fps，预计稳定性增强后 fps 掉 5–10%）。

---

## 八、关键代码位置速查

| 关注点 | 文件 | 行 |
| --- | --- | --- |
| TorchStereoRenderer 主体 | `tool_2dvr/logic.py` | 959 |
| `_normalize_near` | `tool_2dvr/logic.py` | 995 |
| `_forward_warp_eye`（round 在此） | `tool_2dvr/logic.py` | 1014 |
| `_smooth_depth` | `tool_2dvr/logic.py` | 990 |
| `_fill_eye_holes` (soft_shift 入口) | `tool_2dvr/logic.py` | 1116 |
| 主转换循环 | `tool_2dvr/logic.py` | 1518 |
| DA3 forward（多帧潜在改造点） | `_vendor/da3/depth_anything_3/model/da3.py` | 100 |
| DA3 _prepare_model_inputs（N=1 强制） | `_vendor/da3/depth_anything_3/api.py` | 155 |
