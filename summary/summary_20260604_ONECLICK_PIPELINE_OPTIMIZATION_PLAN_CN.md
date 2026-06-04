# OneClick 全链路优化计划（基于一次 8K SBS 实测运行）

## 背景与样本

- 源：8192×4096 SBS hequirect，时长 28m30s，HEVC 10bit
- 配置：`mode=sbs, use_fisheye=False, pre_extract_inner=True, keep_intermediate=True`
- 检测器：`lada_vr_mosaic_detection_model_v2_fast.pt`, imgsz=2048
- 实际运行 3h05m 后被用户主动取消，未完成最后一段（4 个粗段中的第 3 段）

### 实测耗时分布

| 阶段 | 耗时 | 备注 |
|---|---|---|
| Stage 1 粗扫（keyframe） | 3'21" | 347 keyframes @ 3.8 samples/s |
| Stage 2 keyframe-copy 切片 | 43" | 4 个 mosaic 段 + 3 个 gap 段，stream-copy |
| Stage 3 段 0（20s clip） | 1'24" | paired fine + extract×2 + restore×2 + paste |
| Stage 3 段 1（30s clip，无检测） | 23" | 仅扫描，命中 0，原片透传 |
| Stage 3 段 2（725s clip） | **2h35m** | 主瓶颈，见下分解 |
| Stage 3 段 3（290s clip） | 取消时刚开始 | — |

### 段 2 内部分解（2h35m）

- L+R paired fine scan：9'
- 10 个 segment 串行 extract + restore：2h14'
  - L.seg002 / R.seg002（3968×1792 大 rect）：47' + 43' ≈ 1.5h
  - L.seg000 / R.seg000（1696×2064）：9.5' + 17'
  - 其余 6 个小 rect（多为 528×528）：每个 1–9'
- paste 阶段（8K SBS @ 37 fps，43442 帧）：21'

观察：**restore 吞吐受 rect 像素数主导**——
- 528×528 ≈ 49 fps
- 1696×2064 ≈ 22 fps
- 3968×1792 ≈ 9.4 fps

所以 paired 分支的小块切分是有效的（开发人员之前提出的"同时间多 rect union 合并"
会大幅退化这块吞吐）。瓶颈在别处。

---

## 优化项一览（按性价比 / 改动量排序）

| Pri | 项目 | 收益 | 改动量 | 风险 |
|---|---|---|---|---|
| P0 | 取消事件不应触发"完整全眼 fallback" | 防止用户取消后系统反而启动 1h+ 全眼重做 | S | 低 |
| P1 | Stage 1 keyframe 粗扫换 GPU NVDEC 解码 | -3 分钟/次（短视频更显著） | M | 低 |
| P2 | GPU 进度行节流：1Hz → 5%/5s 取大者 | 日志文件缩到 1/5，长任务排查更快 | XS | 极低 |
| P3 | 同时间同眼多 rect 合并 decode pass | 重复时间窗口减少 N 倍 decode I/O | M | 中 |
| P4 | `pre_extract_fine_yolo_conf` 默认 0.40 → 0.50 | 配对噪声减少，间接减少 fine 后处理时间 | XS | 低 |
| P5 | extract / restore 流水线并行 | 单段 wallclock -20%~30%（NVDEC 与 inpainter 并行） | L | 中 |
| P6 | paste 阶段 gap 帧 stream-copy 而非重编 | 8K paste 21' → 估 8'~10' | L | 高 |
| P7 | 粗扫 conf 0.20 → 0.25 + 段最短时长抬高 | 减少假阳性 interval | XS | 低 |

---

## P0：取消事件不应触发完整全眼 fallback

### 现状

粗扫与 paired fine 扫描两个入口都用 `try ... except Exception as exc` 把
任意异常（包括 `OperationCancelled`）映射为 `PreExtractResult.SCAN_FAILED`。
上层在收到 `SCAN_FAILED` 后，会自动走"完整全眼 restore"分支。

实测日志最后阶段：

```
[source-scan] paired fine scan failed: OperationCancelled: cancelled by user
[source-scan] paired fine path failed; falling back to full-eye restore
[gpu] cancelled by user
Split dual error: cancelled by user
Error: cancelled by user
```

本次因为 split_dual 阶段立刻又检测到 cancel token 抛出，所以只是"看起来"
退出了。但如果取消时机恰好在 split 完成、刚进入全眼 lada/jasna restore，
**用户取消之后系统会继续跑 30–90 分钟才能停下**。

### 实施方案

1. 引入第三种结果状态 `PreExtractResult.CANCELLED`（与 SCAN_FAILED 并列）。
2. 在所有捕获 `Exception` 的 source-scan / pre-extract 入口里：
   - 先单独捕获 `OperationCancelled`，返回 `CANCELLED`
   - 再 `except Exception` 走原有 `SCAN_FAILED` 路径
3. 上层处理：
   - 收到 `CANCELLED` → 立即 `raise OperationCancelled`，不再尝试任何 fallback
   - 收到 `SCAN_FAILED` → 保持现有 fallback 行为
4. 另在 fallback 入口里做一道"开工前"的 cancel token 检查（双保险）：
   即将启动 full-eye restore 之前再判断一次 `cancel_token.cancelled`，
   命中就直接抛 `OperationCancelled`。
5. 单测覆盖："scan 抛 OperationCancelled → 调用方观察到 CANCELLED → 不调用 fallback"。

### 注意点

- `OperationCancelled` 已是定义在 `gpu_engine.fallback` 里的明确异常，
  改造只是分支判定。
- 现有 fallback 链里的 GPU 操作各自也响应 cancel token，但响应延迟可能
  是数百帧——能在更外层提前阻断最干净。

---

## P1：Stage 1 keyframe 粗扫切到 GPU NVDEC 解码

### 现状

Stage 1 走"keyframes" 策略时，用 `ffmpeg -skip_frame nokey -vf crop=...
-f rawvideo -pix_fmt bgr24 pipe:1` 走 CPU 解码 8K HEVC，从管道读 BGR24
给 YOLO 检测。这条路径在本次 8K 源上跑出 **3.8 samples/s**，347 keyframes
共耗 3'21"。

Stage 3 的 paired fine 扫描走的是另一条 GPU 路径——`PyNvThreadedSerialDecoder`
+ NV12→BGR CUDA kernel + dlpack 到 torch，复用相同的 YOLO detector，
跑出 5–7 samples/s 而且帧分辨率更大。

### 实施方案

复用 Stage 3 GPU 扫描的实现，把 keyframe 模式包装成"按 keyframe 时间戳
seek 的 GPU 解码扫描"：

1. 从 keyframe 列表得到帧 index 集合 `S = {round(t * fps) for t in keyframes}`。
2. 用 GPU 解码器按 frame_at(i) 逐 keyframe 取帧；对 8K HEVC 用 NVDEC 直接拿
   NV12（已支持 10bit）。
3. 在 GPU 上做 crop 取一只眼睛（沿用 Stage 3 现有的 GPU crop slice 写法），
   然后 NV12→BGR 通过 dlpack 进 torch，喂给 detector batch。
4. 与现有 GPU 扫描复用 detector 与批量逻辑；不需要新建模型实例。
5. 进度日志格式保持一致，便于历史对比。
6. 保留旧的 ffmpeg 路径作为 fallback——对非 GPU 路由的素材（probe 判定为
   "不适合 GPU"的源）继续可用。
7. 配置项 `pre_extract_keyframe_scan_backend = "gpu" | "cpu" | "auto"`，
   默认 `auto`（GPU 可用就 GPU）。

### 预期

8K HEVC keyframe-only 扫描应到 30–50 samples/s 之间（GPU decode latency 比
连续 decode 高，但 keyframe 本身 seek 后即解一帧）。**净收益 ~3 分钟/次**，
对短视频（GUI 交互态）尤其明显。

### 风险

- NVDEC 对密集 seek 的吞吐取决于 GOP 结构：纯 keyframe 命中是最理想场景，
  不会出现 B/P 帧跳过开销。
- 实测建议先在 1–2 个真实样本上对比 sample/s。

---

## P2：GPU 进度行节流策略调整

### 现状

进度报告类目前是"每秒一行"。一段 47 分钟的 restore 输出 ≈ 2820 行，整次
运行总日志行数接近 1 万，文件约 900KB。在 GUI / 远程审查日志时翻页非常重。

### 实施方案

把现有节流策略改成"两个阈值取较大间隔"：

- 时间阈值：最少 5 秒（避免高吞吐时段每秒刷屏）
- 进度阈值：每跨越 5% 强制打一次（确保短任务也有足够行数显示进展）
- 完成时强制 final 行（保留现有 `prog.finish()` 调用）

具体改动：

1. 在进度类构造里新增 `min_pct: float = 5.0`（默认 5%），
   把"上次百分比"也作为状态保存。
2. `update()` 内：当前 `(now - self._last) >= min_interval` 或
   `(pct - self._last_pct) >= min_pct` 任一成立就输出。
3. `min_interval` 默认从 1.0 调到 5.0；保留参数化能力，方便短任务路径
   （extract、scan 本身的 ETA 已不长）显式覆盖。

### 注意

- 不要把节流配置藏在 GPU 引擎深处，给 oneclick 上层一个 app_config 入口
  （比如 `progress_log_interval_s` 和 `progress_log_min_pct`），用户日后
  抱怨"日志太少"时无需改代码。
- 取消、错误、mux 等一次性事件不走节流，保持立即写出。

---

## P3：同时间同眼多 rect 合并 decode pass

### 现状

paired fine 阶段对每个 segment **独立**调一次"GPU 解码 + 单一 rect 切出
+ NVENC 出 mp4"。但实际场景中同一时间窗口的多 rect 极其常见，
本次段 2 即出现：

| 段 | 时间窗口 | rect | 像素数 |
|---|---|---|---|
| L.seg001 | 212.214–655.653s | 1792, 1248, 528×528 | 0.28M |
| L.seg002 | 212.214–655.653s | 0, 2304, 3968×1792 | 7.11M |
| R.seg001 | 212.214–655.653s | 1296, 1440, 896×528 | 0.47M |
| R.seg002 | 212.214–655.653s | 0, 2400, 3344×1696 | 5.67M |

同一段 443s 源被 NVDEC 解码 4 次（L 一对 crop=left、R 一对 crop=right）。
单纯重复解码就增加了大量 GPU PCIe / 显存吞吐。

### 设计原则

- **不退回 union 合并**：union 会让小 rect 的 lada 时间从 ~9 分钟膨胀到几十分钟。
- **不影响缓存内容键**：每个 rect 仍按其 (time, rect) 缓存成独立文件，
  paste 阶段无须改动。

### 实施方案

把当前"按 segment 串行"的 extract 循环改写成"按 (side, start_frame, end_frame)
分组、组内多 rect 共解码"模式：

1. **分组**：用 `(side, round(start*fps), round(end*fps))` 作 key 把同一眼
   同一时间窗口的所有 segment 聚成一个 group；不同组继续串行。
2. **逐组解码一次**：每组调用一次解码器，按 stride=1 拿原帧，按 group 内的
   多个 rect 同时切出多块。
3. **多 NVENC session 并发写**：每个 rect 一个独立 NVENC encoder + raw 文件
   sink。在一遍 decode loop 中 `for rect in group: enc[rect].feed(slice)`。
4. 已有 paste 流水线就是"一遍 decode 喂多块"的模式（在主帧上并行修改多个
   矩形区），可以照搬其 active-session 管理与编码器并发释放策略。
5. 命中缓存的 rect 在分组阶段就剔除，不要占用解码 slot。
6. cancel token 在外层 decode loop 检查；任一编码器报错则停止整组并清理 raw。

### 边界

- 同时间但**不同眼**的 segment 不合并（左右眼 crop 不同）。
- 时间**部分重叠**的 segment 暂不合并（实现复杂度高，收益取决于命中模式；
  P3 v2 再做）。
- 同一 group 内 rect 数量上限建议设个配置项（如 4），避免 NVENC session
  开太多造成显存爆。

### 预期收益

- 本次段 2 中两眼共 4 个同时间 rect，理论上 4 次 decode → 2 次（左右各 1 次），
  decode 端节省约 50%。
- decode 在 extract 中占比与 rect 像素数反相关（小 rect 时占比高）。综合估
  本次段 2 节省 **5–10 分钟**。
- 长片中"主马赛克持续整场"的素材收益更显著。

### 验证

- 单测：构造同时间窗口 3 rect 的合成场景，验证三个输出文件帧数、解码次数计数
  （在 decoder mock 上断言 frame_at 的调用次数）。
- 真实样本：在改造前后跑同一素材的同一段，比较 wallclock 与 restore 输出文件
  hash（应一致）。

---

## P4：fine 阶段 conf 默认值上调

### 现状

`pre_extract_fine_yolo_conf` 默认 0.40。一次扫描后日志：

```
paired fine segments: pairs=5, left=5/6, right=5/7,
  skipped_left=1, skipped_right=2, rejected_time=29, rejected_spatial=7
```

36 个候选 pair 仅 5 个通过过滤。实际进入 paste 的 5 对中，conf 都在 0.85+。
0.40–0.50 区间的检测多是低质量噪声，进入 `_scan_hits_gpu_transform` 后续
的聚合、box 扩展、空间聚类、时间合并都有计算开销。

### 实施方案

- `pre_extract_fine_yolo_conf` 默认改 0.50。
- coarse 阶段 `pre_extract_coarse_yolo_conf`（或当前默认 0.20）保持不变，
  因为粗扫的目的是宽召回。
- 配置文件注释明确两者用途差异，避免用户误改成同一值。

### 验证

- 选一组样本（包括马赛克边缘模糊、画面噪点多两类），统计 conf 0.40 vs 0.50
  下 paired segment 数量与最终 paste 视觉效果是否一致。如果出现真正的 0.40 命中
  被漏掉，再回调。

---

## P5：extract / restore 流水线并行

### 现状

当前 paired 分支处理顺序：
- 同步 extract L.segN.mp4
- 同步 restore L.segN.restored.mp4
- 同步 extract L.seg(N+1).mp4
- …

extract 在 NVDEC + GPU crop + NVENC 链路；restore 在 PyTorch inpainter + NVENC。
两者主要瓶颈分别在 NVENC（extract）与 inpainter compute（restore），存在
天然的不同 GPU 子系统分工，**串行运行时存在大段互相等待**。

### 实施方案

1. 用一个固定深度（如 2）的"已就绪输入"队列：
   - extract 工作线程持续生产 `seg_in.mp4` 推入队列
   - restore 工作线程持续消费 → 输出 `seg_out.mp4`
2. extract 与 restore 共用一个 cancel token；任一线程命中 cancel 立即向另一边
   传播。
3. NVDEC、NVENC、cuda inpainter 各自有独立 context，可以同时占用 GPU。需要
   实测 RTX 5060 Ti 的 SM 占用——本机模型已是单实例（`get_engine()` 单例），
   不需要重新加载。
4. inpainter 单例需要确认"是否线程安全"。如果不是，可以走"两个事件循环 +
   独立 CUDA stream"的实现方式：extract 在 stream A，restore 在 stream B，
   主线程驱动一个简单的 producer-consumer。
5. 失败处理：extract 失败时 restore 已开始的那段要么丢弃 raw 要么完成后删除。

### 预期收益

extract 端单段在小 rect 上 ~10s（173 fps），restore 端小 rect ~9 分钟。
extract 阶段在 restore 进行中"白嫖"NVDEC + NVENC，本身不影响 restore；
restore 占用的时间是 wallclock 主要构成。**实际节省主要来自 restore 期间
extract 不再排队**，对"很多小段"的场景帮助大。本次段 2 的 10 个 segment 中
8 个小 rect，估算节省 **10%–20% wallclock**。

### 风险

- inpainter 与 NVDEC 共享 GPU 内存池可能引发显存峰值升高；需要 OOM 保护。
- 单测难度较高，需要先以 mock 验证 producer-consumer 语义。

---

## P6：paste 阶段 gap 帧 stream-copy 而非重编

### 现状

paste 8192×4096 10bit 全程走 NVDEC → blend → NVENC。本次段 2 paste 43442 帧
@ 37 fps 共 21'。但实测大约只有 60% 帧落在某个 active rect 区间内；剩下 40%
"无 rect 命中"的帧仍然被重新解码、重新编码，画面内容毫无变化。

### 实施方案

1. 进入 paste 前，把整段时间轴按 active rect 的并集分成 N 个"重编区间"
   与 M 个"透传区间"。
2. **透传区间**：按 keyframe 对齐切出原片片段（与 Stage 2 keyframe-copy
   完全同一个对齐器），保留原 HEVC 流。
3. **重编区间**：走当前 GPU paste 流水线，输出 raw HEVC，mux 出片段。
4. 末尾用 concat demuxer 把 N+M 个片段串成最终输出。需要保证：
   - 所有片段同一 codec / pix_fmt / color metadata（已有 mux 路径保证）
   - 每个片段起止严格对齐 keyframe
5. 与 Stage 2 那套 `align_segments` 逻辑共用代码。

### 预期收益

本次 paste 21' → 估 8'~10'。

### 风险

- HEVC 流拼接对 SPS/PPS/VPS 一致性极敏感，跨片段如果 NVENC 与原始流参数
  不完全一致，部分播放器会卡顿；需要严格控制重编片段的编码参数。
- keyframe 对齐导致 paste 重编区间会向两边膨胀到最近 keyframe，实际收益略低于估算。
- 复杂度最高，建议在 P0–P5 落地、流程稳定后再启动。

---

## P7：粗扫 conf 与最短段时长微调

### 现状

实测段 1（138.138–168.168s）粗扫被纳入 mosaic interval，进入 paired fine 后
**0 个有效检测**。整个 23 秒走完一遍流程后输出与原片一致（透传）。属于粗扫
假阳性。

粗扫 conf 默认 0.20、`pre_extract_min_segment_s` 默认 1.5。

### 实施方案

- 粗扫 conf 试调到 0.25（实测前后召回率对比）。
- `pre_extract_merge_gap_s` 与 `pre_extract_min_segment_s` 当前默认偏激进，
  可以略保守一点（min 2.0、merge_gap 2.0），减少边缘碎片化造成的小段。
- 这一组都需要 A/B 数据支撑，**不要在没有样本的前提下盲调**。

---

## 已落地基础

以下是当前实现已经做得不错的、修改时应当保留的设计点：

- **检测器与 inpainter 已经是进程级单例**，不会每段重新加载模型。
- **paired 分支缓存改 frame-key 命名**（最新一版）：缓存语义稳定，
  P3 改造同时间多 rect 合并 decode 时仍然可以分别命中独立的 rect 缓存。
- **paste 阶段同帧多 active rect 已正确支持**：rect 互不重叠时各自 in-place
  修改主帧切片；不需要任何"合并 rect"才能工作的假设。

---

## 推荐落地顺序

1. **第 1 批（半天到 1 天）**：P0 + P2 + P4。
   - 风险极低，全部是行为/默认值调整。
   - 收益直接体现：取消行为正确、日志可读、fine 噪声减少。
2. **第 2 批（1–2 天）**：P1 + P3。
   - 主要 wallclock 收益，对长视频显著。
3. **第 3 批（2–3 天）**：P5。
   - 需要专门的并发模型设计、显存峰值监测。
4. **第 4 批（按需）**：P6 + P7。
   - P6 复杂度最高，建议主路径稳定后再做；P7 用 A/B 数据驱动。

每一批都建议保留 keep_intermediate 跑同一组样本对比 wallclock + 输出 hash
（restore 段 hash 必须一致），同时跟踪日志大小变化。
