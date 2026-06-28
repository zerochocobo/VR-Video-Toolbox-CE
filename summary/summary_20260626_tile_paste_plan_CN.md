# Tile-Based Paste 架构方案：技术交接计划

日期：2026-06-26（更新：2026-06-26 开发者可行性验证后修订）  
状态：**Phase 0 卡点，暂不进入实现** — 见第 12 节  
目标：paste 阶段从当前 ~26fps 提升至 >60fps（8K 10bit HEVC）

---

## 1. 为什么现有优化路线触及天花板

已确认的事实（来自 `summary_20260626_paster_performance_optimization_research_CN.md`）：

- paste 阶段 85% 时间花在 NVENC encode 全帧 8K
- alpha paste / decode 合计 <3ms/frame，不是瓶颈
- P4/multipass=fullres 在 RTX 5060 Ti 单 NVENC 引擎上的 8K 吞吐上限约 27fps
- 改 multipass=qres 约 +19%，达到约 30fps
- 无论如何调参，单 NVENC 引擎编 8K 10bit HEVC 的物理上限约 55fps

**结论**：60fps 目标在当前"全帧重编码"架构下不可达，需要架构级改变。

---

## 2. 核心思路：只重编码含 rect 的 tile，其余 stream-copy

### H.265 Tile 机制

H.265 原生支持把一帧分成若干矩形 tile，tile 之间完全独立（可以单独解码）。  
NVENC 支持通过 `numTilesPerRowMinus1` / `numTilesPerColumnMinus1` 输出 tiled HEVC。

**关键特性**：tile 边界必须是 CTU（64px）的整数倍。

### 理论节省

以 `2_1` 样本的 rect 位置为例：
- 左 rect：x=1680-3040，y=1696-3248
- 右 rect：x=5504-6752，y=1664-3264

在 CTU 对齐的 5×3 tile 网格（见第 4 节）下：
- 仅 **2 个 tile** 含 rect → 需要重编码
- **13 个 tile** 无变化 → stream-copy（零编码成本）

| | 当前 | tile 方案 |
|---|---:|---:|
| 重编码像素量 | 33.55M（8K 全帧） | ~4.30M（约 1/7.8）⚠️ |
| encode 时间估算 | ~36ms/frame | ~5ms/frame |
| 理论 paste FPS | ~26fps | **~140fps**（理论） |

⚠️ **像素估算修正**（原文有误）：  
- 左 tile：1408×1600 = 2.25M px  
- 右 tile：1280×1600 = 2.05M px  
- 合计 4.30M / 33.55M ≈ **1/7.8**，非原文的 1/15  
仍然显著，但原文高估了收益。

---

## 3. 流程变化

### 当前流程（pre-extract 分支，"去马赛克前先提取片段和区域"勾选）

```
scan_segments()          ← 细扫：YOLO 检测 rect 位置
align_segments()         ← keyframe 对齐
cut_segment() × N        ← 按 rect 从 base 裁出小片段
process_lada() × N       ← Lada/Jasna/native 去马赛克还原
paste_segments_gpu()     ← paste 还原片段贴回 base，全帧重编码 ← 瓶颈
```

### 新流程（加入 tile 预编码）

```
scan_segments()          ← 不变：细扫得到精确 rect 位置

plan_tile_grid()         ← 【新增】根据 rect 坐标算 CTU 对齐的 tile 网格
                            输出：tile_layout（行列数 + 每列/行的像素范围）

[并行执行]
  A: tile_encode_base()  ← 【新增】仅对 paste 时间区间的 base 做 tile HEVC 预编码
                            输出：base_tiled_seg001.mp4, seg002.mp4 ...
                            （base_tiled.mp4 保留在工作目录，可复用）

  B: cut_segment() × N   ← 不变
     process_lada() × N  ← 不变，通常比 A 耗时，A 可在 B 跑时完成

paste_segments_tile()    ← 【修改】tile-aware paste：
                            对含 rect 的 tile：decode base tile + paste + NVENC encode
                            对无 rect 的 tile：从 base_tiled.mp4 stream-copy
                            输出 tiled HEVC → 拼接成最终文件
```

### 关键优化点：A 和 B 并行

还原（Lada/native）对于高质量模型往往需要数分钟，这段时间 GPU 主要跑推理。  
tile_encode_base 用 NVENC 跑，几乎不占推理资源。  
**如果 B 比 A 慢（通常如此），A 的时间成本可以被完全隐藏。**

---

## 4. tile_grid 设计算法

### 输入

- `rects`：所有 segment 的联合 rect（精确坐标，来自 scan_segments）
- `frame_w`, `frame_h`：视频分辨率（如 8192×4096）
- `ctu`：CTU 大小，通常为 64px

### 算法步骤

```python
def plan_tile_grid(rects, frame_w, frame_h, ctu=64):
    """
    rects: list of (x, y, w, h) - 所有 segment 的 rect，CTU 对齐后向外扩展
    返回：TileLayout(col_boundaries, row_boundaries)
    """
    # 1. 收集所有 rect 的 x 和 y 边界，CTU 对齐（向外取整到 64px 倍数）
    x_cuts = {0, frame_w}
    y_cuts = {0, frame_h}
    for (x, y, w, h) in rects:
        x_cuts.add(align_down(x, ctu))
        x_cuts.add(align_up(x + w, ctu))
        y_cuts.add(align_down(y, ctu))
        y_cuts.add(align_up(y + h, ctu))
    
    # 2. 去重排序，得到 tile 边界列表
    col_boundaries = sorted(x_cuts)   # e.g. [0, 1664, 3072, 5504, 6784, 8192]
    row_boundaries = sorted(y_cuts)   # e.g. [0, 1664, 3264, 4096]
    
    # 3. 对每个 tile，标记是否与任一 rect 相交
    # 相交的 tile → 需要重编码；其余 → stream-copy
    return TileLayout(col_boundaries, row_boundaries)
```

### `2_1` 样本的 tile 网格结果

使用所有 segment 的 rect union：
- 左 rect（CTU 对齐）：x=1664–3072，y=1664–3264
- 右 rect（CTU 对齐）：x=5504–6784，y=1664–3264

| | 0 | 1664 | 3072 | 5504 | 6784 |
|---|---|---|---|---|---|
| **0** | 空 | 空 | 空 | 空 | 空 |
| **1664** | 空 | **★左** | 空 | **★右** | 空 |
| **3264** | 空 | 空 | 空 | 空 | 空 |

★ = 需要重编码（2 tile），空 = stream-copy（13 tile）

### 重要约束

| 约束 | 说明 |
|---|---|
| H.265 最多约 20 列 tile | 通常不会超限（rect 驱动的 tile 数量有限） |
| NVENC tile 支持 | 通过 `numTilesPerRowMinus1` / `numTilesPerColumnMinus1` 设置 |
| tile 数 = (列数-1) × (行数-1) 个分割 | 5列3行 = 15 tiles |
| 每个 tile 宽/高至少 2 个 CTU | 即 ≥128px，通常满足 |

---

## 5. tile_encode_base 效率评估

### 只编码 paste 时间区间

不需要对全片 base 做 tile 预编码，只需要对每个 `segment.start_s – segment.end_s` 区间做：

```python
# 对每个 paste 区间单独 tile 编码
for seg in segments:
    tile_encode(
        base_src,
        output=f"base_tiled_seg{seg.seg_id:03d}.mp4",
        start_frame=seg.base_frame_start,
        end_frame=seg.base_frame_end,
        tile_layout=tile_layout,
    )
```

### 与 Lada 并行时的实际成本

对于典型 20分钟视频、3分钟有马赛克的场景：
- tile_encode_base 只跑 3分钟区间：成本 ≈ 3min × (encode_fps/59.94) × encode_time
- Lada 高质量还原 3分钟素材：通常 >30分钟
- **tile_encode_base 完全被 Lada 并行隐藏，净增时间 ≈ 0**

对于 `2_1` 这种 95% 都有 rect 的极端样本，A 和 B 几乎等长，并行收益有限。但即使如此，总时间不会比现在差（因为原来没有 A 这步）。

### 文件保留策略

- `base_tiled_seg001.mp4` 等可以留在工作目录（用户已确认）
- 下次同视频再处理时可直接复用（通过 sidecar hash 验证）

---

## 6. tile-aware paste 的实现要点

### 对"无 rect tile"的 stream-copy

从 `base_tiled_seg*.mp4` 中提取对应 tile 的 bitstream 片段。  
H.265 tile bitstream 提取：tile 内容在 RBSP 中有明确边界，可通过 annexb 扫描 slice header 里的 `first_slice_segment_in_pic_flag` + `slice_segment_address` 定位每个 tile 的 NAL 范围。

工具选项：
- `openhevc` / `kvazaar` 有 tile 提取示例代码
- 也可基于 bitstream 解析器（如 `hevcesbrowser` 思路）手写轻量提取

### 对"含 rect tile"的重编码

使用独立 NVENC session（仅该 tile 的尺寸）：

```python
tile_enc = PyNvEncoderSession(
    tile_w, tile_h, bit_depth=10,
    preset="P4", multipass="fullres", aq="1", ...
)
# 解码 base 对应区域 → paste rect → encode
```

tile 尺寸仅约 1408×1600（左 rect tile），NVENC 编这个尺寸轻松超过 120fps。

### 输出 bitstream 拼接

每帧：
- 收集所有 tile 的 NAL（部分来自 stream-copy，部分来自 NVENC 新编）
- 按 tile 顺序拼接，确保 slice_segment_address 字段正确
- 写入输出文件

**注意**：拼接时需保证 SPS/PPS 的 tile 结构描述与实际 tile 数量一致。  
首帧（IDR）时，所有 tile 都必须重编码（NVENC 不支持 IDR 帧的 tile stream-copy）。

---

## 7. 边界情况与风险

| 情况 | 处理 |
|---|---|
| rect 跨越多个 tile | 重编码所有相关 tile，仍优于全帧重编码 |
| 不同 segment 的 rect 位置不同 | tile_grid 取所有 rect 的 union → 单一 tile 结构贯穿整个 paste 区间 |
| 多个 segment 的 rect 覆盖不同 tile | 可能导致更多 tile 需要重编码，极端情况退化为接近全帧 |
| base_tiled.mp4 对应时间段不存在 | 回退到全帧重编码（graceful degradation） |
| 播放器兼容性 | H.265 tile 是标准功能，主流播放器（Quest、PCVR）均支持 |
| feather 跨 tile 边界 | tile 边界比 rect 向外扩展了约 64px（CTU 对齐 padding），feather（默认 12px）完全在 tile 内 |

---

## 8. 开发任务分解

### Phase 1：基础验证（~1周）

| # | 任务 | 文件 | 备注 |
|---|---|---|---|
| 1.1 | 实现 `plan_tile_grid(rects, w, h)` | `utils/tile_layout.py`（新建） | 纯几何运算，容易测试 |
| 1.2 | NVENC tile 参数测试：`numTilesPerRowMinus1=N` 能否正常输出 | `scripts/bench_tile_encode.py`（新建） | 验证 PyNvVideoCodec 支持 |
| 1.3 | tile HEVC bitstream 解析：按 tile 提取 NAL 范围 | `utils/tile_bitstream.py`（新建） | 可用 ffprobe -show_packets 辅助验证 |

### Phase 2：tile_encode_base（~1周）

| # | 任务 | 文件 |
|---|---|---|
| 2.1 | `tile_encode_base(src, dst, start_frame, end_frame, tile_layout)` | `gpu_engine/files.py` 或新模块 |
| 2.2 | sidecar hash 验证（复用已有 `restored_sidecar` 思路） | `gpu_engine/restored_sidecar.py` |
| 2.3 | 与 Lada 并行执行的调度（thread/subprocess） | `one_click/logic.py` |

### Phase 3：tile-aware paste（~2周）

| # | 任务 | 文件 |
|---|---|---|
| 3.1 | `paste_segments_tile_gpu()` 主函数 | `gpu_engine/files.py` |
| 3.2 | 每帧：stream-copy tile NAL + 重编码 tile NAL 拼接 | 同上 |
| 3.3 | 替换 `paste_segments_gpu()` 的调用点，加 tile 路径分支 | `utils/segment_paster.py` |
| 3.4 | 性能对比测试（对比当前全帧 paste） | `scripts/bench_tile_paste.py` |

### Phase 4：集成与回退

| # | 任务 |
|---|---|
| 4.1 | 如 tile 方案失败（NVENC 不支持/bitstream 错误），回退到原全帧 paste |
| 4.2 | 更新 `_run_pre_extract_branch()` 集成新流程 |
| 4.3 | 端到端测试：`2_1` 样本，对比输出画质 |

---

## 9. 已确认的现有代码资产

接手开发者可以直接复用的现有模块：

| 资产 | 位置 | 说明 |
|---|---|---|
| NVENC session | `gpu_engine/pynv_io.py::PyNvEncoderSession` | 已支持 P4/10bit/HEVC |
| NVDEC decoder | `gpu_engine/pynv_io.py::PyNvThreadedSerialDecoder` | 已支持 10bit GPU decode |
| segment 数据结构 | `utils/mosaic_prescan.py::MosaicSegment` | x,y,w,h,start_s,end_s,seg_id |
| passthrough plan | `utils/segment_paster.py::_build_passthrough_plan` | 已有 keyframe 对齐逻辑 |
| sidecar hash | `gpu_engine/restored_sidecar.py` | 可用于 base_tiled 文件复用验证 |
| CTU 对齐工具 | `utils/mosaic_prescan.py::_align_down/_align_up` | 直接复用 |
| bench 脚本框架 | `scripts/bench_oneclick_crop_paste.py` | 复用测试框架 |

---

## 10. 不在本方案范围内

- fisheye paste（`paste_fisheye_eye_rects_to_sbs_gpu`）——先验证 non-fisheye，fisheye 单独评估
- 非 pre-extract 路径（全片 Lada/Jasna 不走这个 paste 逻辑）
- 多 GPU 方案
- 编码质量参数变更（P4、AQ、multipass 保持不变）

---

## 12. ⚠️ Phase 0 卡点：开发者可行性验证结果（2026-06-26）

### 验证结论：当前技术栈前提不成立，暂不进入实现

开发者完成了本地验证，输出文件在 `debug_output\tile_feasibility_*`。

---

### 卡点 1：PyNvVideoCodec / ffmpeg hevc_nvenc 均不输出 tiled HEVC

```
# 尝试透传参数
--paste-enc-extra multipass=fullres,numTilesPerRowMinus1=4,numTilesPerColumnMinus1=2
```

ffmpeg `trace_headers` 验证结果：
```
tiles_enabled_flag 0 = 0   ← 参数被接受但被忽略，输出非 tiled
```

`ffmpeg -h encoder=hevc_nvenc` 也无 tile 相关选项。  
`sliceMode=3,sliceModeData=15` 同样被忽略，帧内无 `slice_segment_address` 特征。

**原因**：PyNvVideoCodec Python wrapper 未暴露 HEVC tile 配置。NVENC SDK C 层是否支持待查，但当前所有高层 API 均不可用。

---

### 卡点 2：tile 码流不能从独立小尺寸编码器拼接

原方案设想用 `1408×1600` 独立 NVENC session 编码 rect tile，然后拼回 `8192×4096` tiled frame。  
这在 H.265 规范下**不可行**，原因：

- tile slice 共享整帧的 VPS/SPS/PPS、POC、参考帧
- `slice_segment_address` 必须在 full-frame tile layout 上下文中设置
- inter prediction 默认可跨 tile 边界，除非启用 MCTS（Motion-Constrained Tile Sets）
- deblock/SAO 在 tile 边界有特殊处理规则
- 不同 encode session 的码流上下文完全不兼容

**结论**：要拼接不同来源的 tile，必须有完整的 HEVC bitstream parser + bit writer，复杂度远超 NAL 拼接。

---

### Phase 0 需要首先证明的前提

| # | 前提 | 状态 |
|---|---|---|
| 0.1 | 找到能输出 `tiles_enabled_flag=1` 的 GPU HEVC 编码路径 | ❌ 未找到 |
| 0.2 | 该路径支持固定 tile layout + 禁用跨 tile 运动依赖（MCTS 或等效） | ❌ 未验证 |
| 0.3 | 同一 full-frame tiled 编码上下文下可替换单 tile 内容 | ❌ 未验证 |
| 0.4 | 1 帧 IDR-only 原型：trace + decode + PSNR 全通过 | ❌ 未实现 |

**在 0.1-0.4 全部通过前，不建议开发 `paste_segments_tile()`。**

---

### 可能的 Phase 0 突破路径

**路径 A：NVIDIA Video Codec SDK C 层直接调用**  
绕过 PyNvVideoCodec Python wrapper，用 `NvEncoderCuda` C API 直接配置 `NV_ENC_CONFIG_HEVC.slices` 和 tile 相关字段。  
需要写 C/C++ 扩展或通过 ctypes 调用。工程量较大，但可能是唯一的 NVENC GPU 路径。

**路径 B：x265 软编码 + tile 支持**  
x265 原生支持 `--tiles=4x2` 等参数，输出标准 tiled HEVC。  
缺点：软编 8K 10bit 速度约 1-3fps，远慢于 NVENC，不适合作为生产路径。  
价值：用于验证 tile 拼接逻辑的正确性（Phase 0 原型）。

**路径 C：重新评估方案，放弃 tile 拼接，改为 tile 并行编码**  
不做"部分 tile stream-copy"，而是改成"整帧按 tile 配置编码，利用 NVENC 内部 tile 并行"。  
收益来源变成 NVENC 硬件并行而非 stream-copy 节省。但已知 PyNv 不输出有效 tile，此路径同样卡在 0.1。

---

### 当前实际可用的优化

在 Phase 0 卡点解决前，现有可落地的最大收益方案：

| 方案 | 状态 | 收益 |
|---|---|---|
| `multipass=qres`（需画质验证） | 可立即测试 | ~+19%，约 30fps |
| 维持现状（P4/fullres/AQ） | 当前生产 | 25-26fps |

---

## 11. 接手开发者须知

### 必读背景文档

1. `summary_20260626_paster_performance_optimization_research_CN.md`  
   → paste 现有瓶颈分析、profiler 使用方法、已否决方案
2. `summary_20260626_paster_optimization_extended_CN.md`  
   → 已试方向的测试结果，为什么需要架构变更

### 关键测试视频

- `videos/test_8k_m426_2_demosaic.mp4`：8192×4096、10bit、59.94fps
- `videos/2_1_process_base.log`：真实生产日志，含 rect 坐标和 passthrough 统计

### 第一步建议

```powershell
# 验证 NVENC 是否接受 tile 参数（Phase 1.2）
# 在 pynv_io.py::PyNvEncoderSession.__init__ 里试加：
# kwargs["numTilesPerRowMinus1"] = "4"
# kwargs["numTilesPerColumnMinus1"] = "2"
# 然后跑短 encode，检查输出是否可播放、是否为 tiled HEVC
uv run python scripts\bench_oneclick_crop_paste.py `
  --src videos\test_8k_m426_2_demosaic.mp4 `
  --start 00:00:00 --end 00:00:05 `
  --rect 1680,1696,1360,1552 --crop-mode left `
  --skip-crop --restored <rect_crop.mp4> --no-measure-quality
```

### 最核心的不确定性

**PyNvVideoCodec 是否支持 tile HEVC 输出**（`numTilesPerRowMinus1` 参数）需要第一步验证。  
如果不支持，需要改用 NVIDIA Video Codec SDK 的 C 层，或者改用 ffmpeg NVENC（支持 tile 参数）。
