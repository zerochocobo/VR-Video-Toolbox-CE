# tool_2dvr 视频时序稳定性改造计划（修订版 v2）

日期: 2026-06-06（v2 修订）
原版: `summary_20260605_2DVR_VIDEO_STABILITY_PLAN_CN.md`
作者: 研究阶段交付
目标: **DA3-Small 单帧推理下，消除/显著减弱视频转换中的 depth 抖动、视差呼吸、羽化边缘闪烁**
前置: FPS stage-1 已 commit（`45cbdbf` + `5969dd7`）；本计划基于 stage-1 后的代码结构
范围: `tool_2dvr/logic.py` 为主，必要时小改 `_vendor/da3/depth_anything_3/api.py`

---

## 0. 相对原版的修订说明

Stage-1 commit 之后 `logic.py` 增长 ~870 行，原版中的多个代码假设已失效，本版做如下调整：

| 修订点 | 原版 | v2 |
| --- | --- | --- |
| 行号速查表 | 基于 stage-0 行号，全部失效 | 基于 stage-1 commit `5969dd7` 重新对齐 |
| `_normalize_near` 改造前提 | 假设逐帧真实 quantile | 已是「采样 8192 + kthvalue/sort」近似 quantile，EMA 在此基础上叠加 |
| EMA 顺序累计实现 | "batch 维 unroll Python loop" | 必须改为 GPU vectorized scan（torch.cumprod 等价 IIR），禁止 batch 内 for-loop |
| EMA 标量类型 | float 标量 | 必须是 GPU tensor + in-place ops，0 个 `.item()` |
| `depth_ema` 分辨率 | 未指定 | 必须挂在 504 低分辨率，配 shape mismatch reset |
| S0-3 适用范围 | 未区分 | 仅 forward warp 路径；`inverse_warp` 模式跳过 |
| S0-3 z-buffer 写法 | "用 priority 加权或两阶段" | 强制两阶段：先 amax 选近物，再加权 splat |
| S1-4 场景切换阈值 | 绝对值 0.4 | 改为相对 EMA 量程归一化 + RGB 直方图备份 |
| S2-1 batch_size | 仅注释"显存×3" | 必须实现 `temporal_window` 触发的 batch_size 自动缩 |
| S2-2 数据通路 | 未指定 RGB 来源 | 扩展 `input_processor_gpu` 返回 504 raw RGB |
| CPU fallback | 二选一 | 明确不实现稳定性增强，仅 CUDA OOM fallback |
| 总开关 | 多个独立 env | 增加 `TOOL_2DVR_STABILIZE=auto/off/full` + UI 勾选 |
| 与 PyNv stage-2 关系 | 未提 | 可并行；建议先合 PyNv，避免 S2-2 数据通路重做 |
| 工时 | 8.5 天 | 11.5 天 |

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

C/D **只影响 `soft_shift` (forward warp) 路径**；stage-1 新增的 `inverse_warp` 模式不存在 hole，天然无 D。

---

## 二、改造分级（按性价比排序）

### S0 — 必做三件套（消除 90% 肉眼可见闪烁）

**S0-1 跨帧 EMA 归一化（修 B）**
- 文件: `tool_2dvr/logic.py::TorchStereoRenderer`（class 起点 line 1159）
- 修改入口: `_normalize_near`（line 1223）、`_normalize_near_percentiles`（line 1247）、`_near_from_depths`（line 1264）
- 改动:
  - 类增加状态字段（**全部 GPU tensor，禁用 Python float**）:
    ```python
    self.lo_ema: torch.Tensor | None = None   # shape (), on device
    self.hi_ema: torch.Tensor | None = None
    self.norm_alpha: float = float(os.environ.get("TOOL_2DVR_NORM_ALPHA", "0.15"))
    self.warmup_frames_remaining: int = 4
    ```
  - 在 `_normalize_near` 算出 batch 内每帧的 `lo_frame, hi_frame`（shape `(B,)` GPU tensor）之后，**用 vectorized scan 累计 EMA**：
    ```python
    # 等价的 IIR：x_t = α·u_t + (1-α)·x_{t-1}，batch 内串行，但无 Python loop
    # 用 cumprod 构造衰减权重，加权求和即可，0 sync 点
    decay = (1 - α) ** torch.arange(B, device=device)
    contrib = α * lo_frame * decay.flip(0)   # 仅示意，需要补 prev 项
    lo_ema_per_frame = prev_lo_ema * (1-α)**B + cumulative_contrib
    ```
    完整公式见附录 A（vectorized IIR）。
  - **禁止 batch 内 for-loop**（会强制每帧一次 GPU sync，破坏 stage-1 的 `render_batch_async` 异步流水线）。
  - 暖机: `self.warmup_frames_remaining > 0` 时直接用当前 batch 的 `lo_frame.mean()/hi_frame.mean()` 作为 EMA 初值，喂 4 帧后切到正式 EMA。利用现有 `kthvalue` 路径无需额外算 quantile。
  - 采样起点固定: 当前 `_normalize_near` 用 `flat[:, ::stride]`，stride 起点必须固定为 0，保证同一帧多次调用结果一致（已是这样，仅文档明确）。
- 环境变量: `TOOL_2DVR_NORM_ALPHA`（默认 0.15；0=不更新、1=纯当前帧；用于回归测试）
- **inverse_warp 路径也受益**（同样调 `_near_from_depths`）。

**S0-2 depth tensor EMA（修 A）**
- 文件: `tool_2dvr/logic.py::TorchStereoRenderer`
- 修改入口: `_near_from_depths`（line 1264），**在 `_smooth_depth` 之前** 挂 EMA
- 关键: **EMA 必须挂在 504 低分辨率上**（depth 上采到 4K 在 `_near_from_depths` 末尾的 `F.interpolate`，要在 EMA 之后）
- 改动:
  ```python
  self.depth_ema: torch.Tensor | None = None  # shape (1, 1, H_low, W_low)
  self.depth_beta: float = float(os.environ.get("TOOL_2DVR_DEPTH_BETA", "0.5"))

  def _apply_depth_ema(self, depth_t):
      # depth_t shape (B, 1, H_low, W_low)
      if self.depth_ema is None or self.depth_ema.shape[-2:] != depth_t.shape[-2:]:
          self.depth_ema = depth_t[0:1].clone()  # shape mismatch → reset
      # vectorized IIR scan, batch 内串行无 Python loop（见附录 A）
      depth_smoothed = vectorized_ema_scan(depth_t, self.depth_ema, β)
      self.depth_ema = depth_smoothed[-1:].clone()
      return depth_smoothed
  ```
- 升级版（推荐，默认开）: 逐像素自适应 β（1€ filter）
  ```python
  diff = (depth_now - depth_ema).abs()
  thresh = float(os.environ.get("TOOL_2DVR_ADAPTIVE_THRESH", "0.05")) * (hi_ema - lo_ema).clamp_min(1e-3)
  slope  = float(os.environ.get("TOOL_2DVR_ADAPTIVE_SLOPE",  "50.0"))
  beta_map = torch.sigmoid((diff - thresh) * slope)
  # 抖动小 → β小（多用 EMA），运动大 → β大（多用当前）
  depth_ema = beta_map * depth_now + (1 - beta_map) * depth_ema
  ```
- 环境变量:
  - `TOOL_2DVR_DEPTH_BETA`（默认 0.5）
  - `TOOL_2DVR_ADAPTIVE_BETA`（默认 1，0 时退回固定 β）
  - `TOOL_2DVR_ADAPTIVE_THRESH`（默认 0.05）
  - `TOOL_2DVR_ADAPTIVE_SLOPE`（默认 50.0）
- 视频修补模式不再单独抬高 `TOOL_2DVR_DEPTH_BETA`，默认保持 **0.5**。
- 场景切换: S1-4 检测到切换 → `self.depth_ema = None` + `warmup_frames_remaining = 4`。

**S0-3 disparity sub-pixel splat（修 C+D）**
- 文件: `tool_2dvr/logic.py::TorchStereoRenderer._forward_warp_eye`（line 1279）
- **仅适用于 forward warp 路径**（即 `soft_shift / shift_fill / background / inpaint / lama / none` 在 forward warp 路径生效）；`inverse_warp` 模式的 `_render_batch_fast_tensor` 跳过本项。
- 现状: `torch.round(target_x).long()` 后整数 `scatter_reduce_(reduce="amax")`
- 改为**两阶段** bilinear splat:
  ```
  阶段 1: z-buffer 仍用整数 round 选近物深度（保留现有 amax 逻辑），得到 near_buffer
  阶段 2: 对每个源像素做 sub-pixel splat
      tx = pixel_x + near * (max_shift * 0.5 * eye_sign)
      tx0 = tx.floor().long().clamp(0, W-1)
      tx1 = (tx0 + 1).clamp(0, W-1)
      w1 = (tx - tx0.float())
      w0 = 1.0 - w1
      # 只 splat near 与该列 z-buffer 接近的源像素（避免远物色彩污染近物）
      near_mask = (near_at_source >= near_buffer[tx0] - eps)
      out_color.scatter_add_(2, tx0, frame * w0 * near_mask)
      out_color.scatter_add_(2, tx1, frame * w1 * near_mask)
      weight.scatter_add_(2, tx0, w0 * near_mask)
      weight.scatter_add_(2, tx1, w1 * near_mask)
      out_color /= weight.clamp_min(eps)
      holes = weight < float(os.environ.get("TOOL_2DVR_HOLE_THRESH", "0.3"))
  ```
- 软 hole mask: holes 变成连续值（权重小→视为洞），soft_shift 羽化的边界在时序上自然稳定。
- 环境变量: `TOOL_2DVR_HOLE_THRESH`（默认 0.3）

**S0 验收**
- 同一段静态镜头 30s，整体亮度（near 均值）跨帧标准差 < 改造前的 30%
- 静态画面 SSIM 跨帧 > 0.98（改造前可能 0.90 左右）
- 快速横摇镜头主观无明显拖影
- `TOOL_2DVR_STABILIZE=off` 时输出与 stage-1 commit `5969dd7` **bit-exact 一致**

---

### S1 — 进阶（消除残留闪烁，处理快速运动场景）

**S1-1 batch 内联合归一化（与 S0-1 互补）**
- 文件: 同 S0-1，`_normalize_near`
- 改动: S0-1 是跨 batch EMA，本项再加一层 batch-内联合：把整个 batch 的所有 inv_depth 一次 flatten 算 lo/hi（用现有 `kthvalue` 路径），作为「当前 batch 的目标」喂给 EMA，而不是逐帧目标。
- 实现是一行替换: `lo_frame = lo_batch_kth(values_flat_all_batch)`，无新开销。

**S1-2 hole mask 时序膨胀（可选，与 S0-3 互补或替代）**
- 文件: `TorchStereoRenderer._fill_eye_holes`（line 1399）
- 仅 forward warp 路径
- 改动:
  - 类增加状态: `self.prev_left_holes`、`self.prev_right_holes`（GPU bool tensor）
  - 当前帧 mask = `current_mask | prev_mask`，再 `F.max_pool2d(erode=1px)`
- 建议: 完整做了 S0-3 后本项可跳过；仅作为 S0-3 落地前的临时方案。

**S1-3 光流引导的 depth warp（核心稳定大招）**
- 文件: `tool_2dvr/logic.py` 新增 `_optical_flow.py` 模块
- **放置线程: 主线程**（DA3 推理前算光流），保证顺序、避免 reader/writer 跨线程状态污染
- 跨 batch 边界: 在 `TorchStereoRenderer` 增加 `self.prev_rgb_low`、`self.prev_depth_low_warpable` 状态，main 线程每次 batch 起始检查
- 实现:
  - 后端: `cv2.calcOpticalFlowFarneback`（CPU，4K 下采到 1280×720 算后放大）
  - 备选: PyTorch RAFT-small（GPU 但更重，留 `TOOL_2DVR_FLOW_BACKEND=raft` 切换）
  - 流程:
    ```
    flow_{t-1→t} = compute_flow(rgb_low_{t-1}, rgb_low_t)
    depth_warped = grid_sample(depth_ema_{t-1}, flow_normalized)
    occ_mask = forward_backward_consistency(flow)  # 遮挡处置 1
    depth_t  = torch.where(occ_mask, depth_now, β·depth_now + (1-β)·depth_warped)
    ```
- 与 S0-2 关系: 启用 S1-3 时 S0-2 仍生效，但在「带运动补偿的 EMA」基础上做。
- 环境变量:
  - `TOOL_2DVR_FLOW_BACKEND`（默认 `farneback`；`raft`/`off`）
  - `TOOL_2DVR_FLOW_DOWNSCALE`（默认 3，即 4K→1280×720）

**S1-4 场景切换检测**
- 文件: `_near_from_depths` 内挂检测
- **触发条件**（任一）:
  1. depth 相对量程差: `(depth_now - depth_ema).abs().mean() / (hi_ema - lo_ema).clamp_min(1e-3) > 0.5`
  2. RGB 直方图 chi-square: 主线程缓存 `prev_rgb_low` 下采 64×64 → 16-bin 直方图，chi² > `TOOL_2DVR_SCENE_CUT_HIST=0.6`
- 检测到切换: 清空 `lo_ema/hi_ema/depth_ema/prev_holes/prev_rgb_low`，`warmup_frames_remaining = 4`。
- **每 batch 仅一次 GPU sync 点**（取 depth-diff 均值标量），允许；其它路径全保持 GPU 异步。
- 环境变量:
  - `TOOL_2DVR_SCENE_CUT_DEPTH`（默认 0.5，相对量程）
  - `TOOL_2DVR_SCENE_CUT_HIST`（默认 0.6）

**S1 验收**
- 含快速摄影机推拉/横摇 + 场景切换的 30s 测试片: 主观无"果冻感"和切换后残影
- 暗场到亮场切换 < 5 帧完成 EMA 重置

---

### S2 — 改 DA3 利用相邻帧（天花板方案）

**S2-1 多帧 DA3 输入（利用 vendored backbone 多视图能力）**
- 文件: `tool_2dvr/_vendor/da3/depth_anything_3/api.py`（line 155 `_prepare_model_inputs`，line ~150 `inference_depth_only` 已存在）
- 背景: DA3 backbone 本来就支持 `x: (B, N, 3, H, W)` 多视图，stage-1 commit 强制 N=1
- 改动:
  - `inference_depth_only` 增加参数 `temporal_window: int = 1`
  - N=3 时滑窗 (t-1, t, t+1) 送入 backbone，head 输出取中间帧 depth
  - 跨 batch 边界需要保留上一 batch 的最后 N-1 帧（主线程缓存）
  - **batch_size 自动缩**: 在 `convert_2d_to_vr` 内根据 `TOOL_2DVR_TEMPORAL_WINDOW` 调整:
    ```python
    if temporal_window > 1:
        depth_batch_size = max(1, resolve_depth_batch_size() // temporal_window)
    ```
- 风险:
  - 显存增加约 3 倍；RTX 5060 Ti 16GB 在 batch=2、N=3 下应可承受
  - 必须验证 head 在 N>1 时的形状行为（DA3 backbone 设计上对多视图友好，但 stage-1 的 `_depth_tensor_from_output` 形状归一化需要兼容）
- 环境变量: `TOOL_2DVR_TEMPORAL_WINDOW`（默认 1；可选 3、5）

**S2-2 joint bilateral depth upsample（替代纯 F.interpolate）**
- 文件: `_near_from_depths`（line 1264）的 `F.interpolate` 改造；同时改 `input_processor_gpu.gpu_preprocess` 返回值
- **数据通路**: 必须把 504 分辨率的 raw RGB（uint8 或 0-1 float，**未减 mean/未除 std**）传到 renderer
  - 方案: `gpu_preprocess` 增加 `return_low_res_rgb=True`，返回 `(x_normalized, x_raw_uint8)`
  - `DA3DepthEstimator.predict_batch` 透传 `low_res_rgb`；renderer 接受新参数
- 实现: 6×6 邻域 joint bilateral upsample，权重 = 空间高斯 × 颜色高斯
  ```python
  near_4k = joint_bilateral_upsample(
      near_low,           # depth, low res
      rgb_low,            # guide, low res
      rgb_4k,             # guide, target res
      sigma_space=2.0, sigma_color=0.1, kernel=6,
  )
  ```
- 算力: 4K 6×6 一次卷积，GPU 上 ~5ms/帧
- 环境变量: `TOOL_2DVR_JBU=1`（默认 1），失败回 `F.interpolate`

**S2 验收**
- 与 S0+S1 比较: 肉眼可分辨的剩余抖动应基本消失
- 跨帧 SSIM > 0.99（静态镜头）

---

## 三、统一总开关与 UI

新增 `TOOL_2DVR_STABILIZE`，三档:

| 档位 | 含义 | 启用项 |
| --- | --- | --- |
| `off` | 完全关，bit-exact 等于 stage-1 | 无 |
| `auto`（默认）| 推荐档 | S0-1 + S0-2（含 1€） + S0-3（仅 forward warp） + S1-1 + S1-4 |
| `full` | 全功能 | auto + S1-3（光流） + S2-1（多帧 DA3） + S2-2（JBU） |

`tool_2dvr/main.py` UI 在 hole_fill 下方加一个「时序稳定」下拉框（`off / auto / full`），i18n 增加三个文案。

细粒度参数（α、β、阈值等）仍可单独 env 覆盖，但 UI 只暴露三档。

---

## 四、提交顺序与里程碑

| 里程碑 | 包含项 | 验收 | 预计工时 |
| --- | --- | --- | --- |
| T1 | S0-1, S0-2（含 1€） | 静态镜头亮度呼吸消除 | 1.5 天 |
| T2 | S0-3 | 边缘 sub-pixel 抖动消除 | 2 天 |
| T3 | S1-1, S1-4, 总开关 | batch 内一致 + 场景切换不残影 + UI | 1 天 |
| T4 | S1-3（光流） | 含快速运动场景稳定 | 3 天 |
| T5（可选）| S2-1 | 多帧 DA3，从源头降噪 | 2.5 天 |
| T6（可选）| S2-2 | 边缘锐化 + JBU 上采质量提升 | 1.5 天 |

最低交付: **T1 + T2 + T3**（4.5 天），已能覆盖 80% 用户场景。
总工时（含可选）: **11.5 天**。

---

## 五、参数与默认值

| 参数 | 默认值 | 范围 | 说明 |
| --- | --- | --- | --- |
| `TOOL_2DVR_STABILIZE` | `auto` | off/auto/full | 总开关 |
| `TOOL_2DVR_NORM_ALPHA` | 0.15 | 0.05–0.5 | lo/hi EMA 系数 |
| `TOOL_2DVR_DEPTH_BETA` | 0.5 | 0.2–0.8 | depth EMA 系数 |
| `TOOL_2DVR_ADAPTIVE_BETA` | 1 | 0/1 | 1€ filter 自适应 |
| `TOOL_2DVR_ADAPTIVE_THRESH` | 0.05 | 0.01–0.2 | 1€ filter 阈值（相对量程）|
| `TOOL_2DVR_ADAPTIVE_SLOPE` | 50.0 | 10–200 | 1€ filter sigmoid 斜率 |
| `TOOL_2DVR_HOLE_THRESH` | 0.3 | 0.1–0.5 | sub-pixel splat hole 阈值 |
| `TOOL_2DVR_FLOW_BACKEND` | farneback | farneback/raft/off | 光流后端 |
| `TOOL_2DVR_FLOW_DOWNSCALE` | 3 | 2–4 | 光流下采倍数 |
| `TOOL_2DVR_SCENE_CUT_DEPTH` | 0.5 | 0.3–1.0 | 场景切换 depth 阈值（相对量程）|
| `TOOL_2DVR_SCENE_CUT_HIST` | 0.6 | 0.3–1.0 | 场景切换直方图 chi² 阈值 |
| `TOOL_2DVR_TEMPORAL_WINDOW` | 1 | 1/3/5 | DA3 滑窗大小 |
| `TOOL_2DVR_JBU` | 1 | 0/1 | joint bilateral upsample |

---

## 六、风险与回归测试清单

**状态生命周期**:
- 每次 `convert_2d_to_vr` 调用必须**新建** `TorchStereoRenderer`（stage-1 已是这样，确认即可），EMA 不会跨视频残留。
- CUDA OOM 二分递归: 必须保证 `_render_with_depths` 递归路径里**不重置 EMA**（递归调用复用同一 renderer 实例 OK，二分只切 batch 不切状态）。
- 视频内不同 batch 之间: depth_ema shape mismatch → reset。

**CPU 渲染 fallback**:
- **明确不实现稳定性增强**。`logic.py` 顶部注释加一行: "CPU 渲染仅作 CUDA OOM fallback，不带 S0/S1 时序稳定"。
- 生产路径默认 CUDA。

**bit-exact 回归**:
- 必加测试: `TOOL_2DVR_STABILIZE=off` 输出与 stage-1 commit `5969dd7` bit-exact 一致（哈希比对）。
- 必加测试: `α=1.0 β=1.0`（等于不更新 EMA）输出与 `STABILIZE=off` 一致。

**debug eye 输出**:
- 保留 `TOOL_2DVR_DEBUG_EYE=1` 旁路对照路径；debug 输出文件名增加 `_stab_<档位>` 后缀，便于多档位对比。

**测试视频集**:
- 完全静态镜头（30s）: 检验"无运动时是否真的不抖"
- 慢速 pan: 检验"运动时不拖影"
- 快速 cut: 检验场景切换重置
- 暗场到亮场过曝转换: 检验 lo/hi EMA 不爆 + 场景切换直方图阈值
- 含字幕烧入的电影片段: 检验小目标（字符）是否被 EMA 模糊掉
- 长视频（10min+）: 检验 EMA 数值不漂移、不爆 NaN

**性能预算**:
- T1+T2 完成后 fps 下降应 < 10%（相对 stage-1）
- T4 完成后 fps 下降应 < 25%（光流是大头）

---

## 七、与 FPS 计划（stage-1 已合 / stage-2 PyNv）的协同

| FPS 项 | 与稳定性计划的耦合 |
| --- | --- |
| stage-1a（DA3 depth-only） | 已合，本计划基于 |
| stage-1b（GPU 流水线 + NVDEC/NVENC + inverse_warp） | 已合；本计划 S0-3 仅对 forward warp 生效；S0-1/S0-2 对两种模式都生效 |
| stage-2（PyNv GPU-resident decode/encode） | **建议先合 stage-2**，因为它会变更 DA3 输入来源（NV12 plane 而非 numpy RGB），影响 S2-2 的 raw RGB 数据通路。若稳定性先做，S2-2 在 PyNv 切换后需要再改一次 |

**推荐顺序**: 
1. PyNv stage-2 落地（FPS 达到 4K60）
2. 稳定性 T1+T2+T3（最低交付）
3. 视效果决定是否做 T4+T5+T6

---

## 八、关键代码位置速查（基于 stage-1 commit `5969dd7`）

| 关注点 | 文件 | 行 |
| --- | --- | --- |
| `TorchStereoRenderer` class 起点 | `tool_2dvr/logic.py` | 1159 |
| `_normalize_near`（GPU 版，已 stage-1 优化）| `tool_2dvr/logic.py` | 1223 |
| `_normalize_near_percentiles`（kthvalue/sort）| `tool_2dvr/logic.py` | 1247 |
| `_near_from_depths`（EMA 挂载点）| `tool_2dvr/logic.py` | 1264 |
| `_forward_warp_eye`（S0-3 改造点）| `tool_2dvr/logic.py` | 1279 |
| `_fill_eye_holes` (forward warp 路径) | `tool_2dvr/logic.py` | 1399 |
| `_render_batch_fast_tensor`（inverse_warp 路径，不改）| `tool_2dvr/logic.py` | 1423 |
| `_render_batch_tensor`（dispatcher）| `tool_2dvr/logic.py` | 1444 |
| `render_batch_async`（异步流水线，S0-1 标量 GPU 化原因）| `tool_2dvr/logic.py` | 1499 |
| `_normalize_near` numpy 版（CPU fallback）| `tool_2dvr/logic.py` | 569 |
| `convert_2d_to_vr`（主入口，batch_size 调整点）| `tool_2dvr/logic.py` | 1691 |
| `_run_transcode_attempt`（reader/main/writer 线程）| `tool_2dvr/logic.py` | 1994 |
| `_reader_loop`（不动）| `tool_2dvr/logic.py` | 2047 |
| `_writer_loop`（不动）| `tool_2dvr/logic.py` | 2065 |
| DA3 `forward`（多帧改造点）| `_vendor/da3/depth_anything_3/model/da3.py` | 100 |
| DA3 `inference_depth_only`（temporal_window 入口）| `_vendor/da3/depth_anything_3/api.py` | ~150 |
| `gpu_preprocess`（S2-2 raw RGB 返回点）| `_vendor/da3/depth_anything_3/utils/io/input_processor_gpu.py` | 41 |

---

## 九、不做什么（明确范围）

- 不引入需要训练/微调的模型
- 不替换 DA3-Small
- 不改动 stereo 几何（`max_disparity_pixels`、`eye_distance_mm` 公式）
- 不改 SBS 输出布局
- 不引入 ONNX/TensorRT（属于 FPS 计划）
- 不在 CPU 渲染 fallback 上实现稳定性增强（仅文档说明）
- 不动 `inverse_warp` 模式的 `_render_batch_fast_tensor` 几何（S0-3 不适用）

---

## 附录 A. Vectorized IIR EMA scan（无 Python loop）

给定 batch 输入 `u: (B, ...)` 和初值 `x_{-1}`，目标计算:
```
x_t = α·u_t + (1-α)·x_{t-1}, t = 0..B-1
```

闭式展开:
```
x_t = (1-α)^(t+1) · x_{-1} + Σ_{k=0..t} α·(1-α)^(t-k) · u_k
```

GPU 实现:
```python
def vectorized_ema_scan(u, x_prev, alpha):
    # u: (B, ...), x_prev: (1, ...), alpha: float
    B = u.shape[0]
    device = u.device
    t = torch.arange(B, device=device, dtype=u.dtype)
    # decay[t] = (1-α)^(t+1)
    decay = (1 - alpha) ** (t + 1)
    decay_shape = (B,) + (1,) * (u.ndim - 1)
    decay = decay.view(decay_shape)

    # 累积贡献: Σ_{k=0..t} α·(1-α)^(t-k) · u_k
    # 用 cumsum 实现:  let g_k = α · u_k · (1-α)^(-k)
    # 则 contrib_t = (1-α)^t · cumsum(g)[t]
    inv_pow = (1 - alpha) ** (-t)
    inv_pow = inv_pow.view(decay_shape)
    g = alpha * u * inv_pow
    contrib = torch.cumsum(g, dim=0) * (1 - alpha) ** t.view(decay_shape)

    return decay * x_prev + contrib
```

- 0 个 Python loop、0 个 `.item()`、单次 cumsum 即可
- α 很小（0.15）+ B 不大（8）时，`(1-α)^(-k)` 数值稳定
- 若 B 较大（>32），改用 segment-wise 实现避免 `(1-α)^(-k)` 上溢；当前 batch_size ≤ 8 无问题。

返回的最后一行 `result[-1]` 即下次 batch 的 `x_prev`，存回 `self.lo_ema/hi_ema/depth_ema`。
