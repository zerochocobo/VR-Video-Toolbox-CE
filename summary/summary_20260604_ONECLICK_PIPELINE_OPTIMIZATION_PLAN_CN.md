# OneClick 全链路优化计划（基于两次 8K SBS 实测运行）

## 背景与样本

- 源：8192×4096 SBS hequirect，时长 28m30s，HEVC 10bit
- 配置：`mode=sbs, use_fisheye=False, pre_extract_inner=True, keep_intermediate=True`
- 检测器：lada VR mosaic detection v2 fast，imgsz=2048
- 当前 fine pre-extract 阈值已调到 0.50（之前为 0.40 时的同时间同眼多 rect 噪声明显减少）

### 完整运行总耗时分布（4h03m，最新一次跑完）

| 阶段 | 耗时 | 占比 | 备注 |
|---|---|---|---|
| Stage 1 粗扫（keyframe-only） | 3'44" | 1.5% | 347 keyframes, ffmpeg CPU decode |
| Stage 2 keyframe-copy 切片 | 45" | 0.3% | 4 mosaic + 3 gap, 全 stream-copy |
| Stage 3 paired pre-extract（合计） | 1h41' | 41.7% | 4 段，仅其中 2 段有命中 |
| ↳ 无命中段 0、1（合计） | 42" | — | 仅扫描后透传 |
| ↳ 段 2（725s clip） | 1h15'40" | — | 主瓶颈 |
| ↳ 段 3（290s clip） | 25'57" | — | — |
| Stage 4 timeline merge（fast HEVC） | 8'00" | 3.3% | 7 个分片 demux → annex-B → concat |
| 最终 mux（+faststart） | 7'12" | 3.0% | 11.6GB combined.hevc → mp4 重写 |
| 其它（探测、空闲） | ~1h05' | 26.7% | 包括等待 GUI / GPU 上下文切换 / 段间隙 |

### Stage 3 段 2 内部分解（75 分钟）

- L + R paired fine 扫描：9'15"
- 6 个 segment（去重后）串行 extract + restore：~64'
  - R rect 0 (2992×2832, 12330f)：~14'40" restore @ ~14 fps
  - L rect 0 (1696×2160, 12330f)：~9'37" @ ~22 fps
  - R rect 1 (3344×1696, 6840f)：~8' @ ~14 fps
  - L rect 1 (1920×560, 6840f)：~4' @ ~28 fps
  - L rect 2 / R rect 2 (2256×528 / 1824×528, 2850f)：~2' 各
- paste 阶段 + mp4 mux（43442 帧 @ 37 fps）：~21'

> restore 吞吐与 rect 像素数强相关，约 3.5K MPix·fps ≈ 常数；
> 小块切分确实有效。**任何"同时间多 rect 合并 union"的方向都会显著退化此处吞吐，不可取。**

---

## 优化项总览（按性价比 / 改动量排序）

| Pri | 项目 | 收益 | 改动量 | 风险 |
|---|---|---|---|---|
| P0 | 取消事件不应触发完整全眼 fallback | 防止用户取消后系统反向启动 1h+ 全眼重做 | S | 低 |
| P1 | Stage 1 keyframe 粗扫换 GPU NVDEC | -3 分钟/次，对短视频更显著 | M | 低 |
| P2 | 进度行节流：1Hz → 5%/5s 取大者 | 日志缩到 1/5，长任务排查更易 | XS | 极低 |
| P3 | paired 同时间窗合并解码（左右眼跨 crop_mode） | 减少同段重复 NVDEC，~3-5 min/段 | M | 中 |
| P4 | fine 阶段加大 stride + 无命中早停（**不**默认降 imgsz） | YOLO 推理 -20%~40%，无召回损失 | S–M | 低 |
| P5 | Stage 4 timeline merge 用 concat demuxer 直读 | -3~5 min，去掉一轮临时 annex-B 文件 | S | 低 |
| P6 | 最终 mp4 输出提供 "fast local" 模式（关闭 +faststart） | -5~7 min，本地观看场景默认 | S | 低 |
| P7 | restore raw HEVC 跳过中间 mp4 mux，paste 直接消费 | -1~2 min/段 + 临时空间减少 50% | M | 中 |
| P8 | paste 内部"无 rect 帧"按 keyframe 切片 stream-copy | -10~12 min/长段，复杂度最高 | L | 高 |
| P9 | extract / restore 跨段流水线并行 | wallclock -10~20%，主要对碎段 | L | 中 |
| P10 | 粗扫 conf / 段最短时长 A/B 微调 | 减少假阳性 interval，减少空跑段 | XS | 低 |
| P11 | **端到端码率契约：中间 1.5×，最终段收敛到源码率（不动 UI）** | 用户"保持源码率"真正落地、避免最终输出膨胀 | S | 低 |

---

## P0：取消事件不应触发完整全眼 fallback

### 现状

粗扫与 paired fine 两个入口都用 `try ... except Exception as exc`，把任意异常
（包括 `OperationCancelled`）映射为统一的 SCAN_FAILED 状态。上层在收到
SCAN_FAILED 后会自动走"完整全眼 restore" fallback 分支。

上一份日志的最后阶段就是这个路径触发的：

```
[source-scan] paired fine scan failed: OperationCancelled: cancelled by user
[source-scan] paired fine path failed; falling back to full-eye restore
[gpu] cancelled by user
Split dual error: cancelled by user
```

本次因为 split_dual 立刻又触到 cancel token 抛出，所以只是"看起来"立刻退出
了。但如果取消发生在 split 完成、刚进入全眼 lada/jasna restore，**用户取消
之后系统会继续工作 30–90 分钟**，对长视频是严重的体验崩塌。

### 实施方案

1. 引入第三种结果状态 `CANCELLED`，与 `SCAN_FAILED` 并列。
2. 所有捕获 `Exception` 的 source-scan / pre-extract 入口里：
   - 先单独捕获 `OperationCancelled`，返回 `CANCELLED`
   - 再 `except Exception` 走原有 SCAN_FAILED 路径
3. 上层处理：
   - 收到 `CANCELLED` → 立即 `raise OperationCancelled`，不再尝试 fallback
   - 收到 `SCAN_FAILED` → 保持现有 fallback 行为
4. 在 fallback 入口里做一道"开工前"的 cancel token 检查（双保险）：
   即将启动 full-eye restore 之前再判断一次 `cancel_token.cancelled`，命中
   就直接抛 `OperationCancelled`。
5. 单测：scan 抛 `OperationCancelled` → 调用方观察到 `CANCELLED` → 不调用 fallback。

### 风险

`OperationCancelled` 已是定义在 GPU fallback 模块里的明确异常，改造只是分支
判定，没有协议级影响。

---

## P1：Stage 1 keyframe 粗扫切到 GPU NVDEC

### 现状

Stage 1 走"keyframes"策略时，调 ffmpeg `-skip_frame nokey -vf crop=...
-f rawvideo -pix_fmt bgr24 pipe:1` 走 CPU 解码 8K HEVC，再从管道读 BGR24
给 YOLO 检测。两次实测都跑出 **3.8 samples/s**，347 keyframes 共耗 ~3'21"。

Stage 3 的 paired fine 扫描走的是另一条 GPU 路径：threaded NVDEC + GPU crop
slice + NV12→BGR CUDA kernel + dlpack 转 torch，复用同一份 YOLO detector
单例，在更大分辨率上跑出 5–7 samples/s。

### 实施方案

复用 Stage 3 GPU 扫描的实现，把 keyframe 模式包装成"按 keyframe 时间戳
seek 的 GPU 解码扫描"：

1. 从 keyframe 列表得到帧 index 集合 `S = {round(t * fps) for t in keyframes}`。
2. 用 GPU 解码器按 `frame_at(i)` 逐 keyframe 取帧；对 8K HEVC 用 NVDEC 直接拿
   NV12（已支持 10bit）。
3. 在 GPU 上 crop 取一只眼睛（沿用 Stage 3 现有 GPU crop slice），然后
   NV12→BGR 通过 dlpack 进 torch，喂给 detector batch。
4. 与现有 GPU 扫描复用 detector 与批量逻辑；不需要新建模型实例。
5. 进度日志格式保持一致，便于历史对比。
6. 保留旧的 ffmpeg 路径作为 fallback——对非 GPU 路由的素材（probe 判定为
   "不适合 GPU"的源）继续可用。
7. 配置项 `pre_extract_keyframe_scan_backend = "gpu" | "cpu" | "auto"`，默认
   `auto`（GPU 可用就 GPU）。

### 预期

8K HEVC keyframe-only 扫描应到 30–50 samples/s。**净收益 ~3 分钟/次**，对
GUI 交互态尤其重要。

### 风险

NVDEC 对密集 seek 的吞吐取决于 GOP 结构；纯 keyframe 命中是最理想场景，
不会出现 B/P 帧跳过开销。实测建议先在 1–2 个真实样本上对比 samples/s。

---

## P2：GPU 进度行节流策略调整

### 现状

进度报告类目前是"每秒一行"。一段 47 分钟的 restore 输出近 3000 行；一次
完整运行总日志接近 6000 行，文件接近 1MB。

### 实施方案

把现有节流策略改成"两个阈值取较大间隔"：

- 时间阈值：最少 5 秒
- 进度阈值：每跨越 5% 强制打一次
- 完成时强制 final 行（保留现有 `prog.finish()` 调用）

实现细节：

1. 在进度类构造里新增 `min_pct: float = 5.0`，把"上次百分比"也作为状态保存。
2. `update()` 内：`(now - self._last) >= min_interval` **或**
   `(pct - self._last_pct) >= min_pct` 任一成立就输出。
3. `min_interval` 默认从 1.0 调到 5.0；保留参数化能力。
4. 给 oneclick 上层暴露 `progress_log_interval_s` / `progress_log_min_pct`
   两个 app_config 入口，用户日后嫌"日志太少"时无需改代码。
5. 取消、错误、mux setup 等一次性事件不走节流。

### 注意

不要把节流配置藏在 GPU 引擎深处。当前节流只针对"高频 update"的进度行，
对一次性事件无影响。

---

## P3：paired 同时间窗左右眼合并解码

### 现状

paired fine 阶段对左眼、右眼分别独立调"GPU 解码 + crop_mode + 单 rect 切出
+ NVENC 出 mp4"。两次实测都看到：左右眼几乎所有 paired segment 都共享同一
时间窗口（因为配对算法本来就要求时间重叠）。本次段 2 的实际工作：

| 时间窗口（帧） | L rect | R rect | 实际 decode 次数 |
|---|---|---|---|
| 270–12600（206s） | 1696×2160 | 2992×2832 | **2 次**（L 一次 + R 一次） |
| 32310–39150（114s） | 1920×560 | 3344×1696 | **2 次** |
| 40050–42900（48s） | 2256×528 | 1824×528 | **2 次** |

同一段 SBS 源被 NVDEC 解码两次（一次取左眼一次取右眼）。在 8K 源上 decode
本身就有相当带宽成本。

> fine_conf 调到 0.50 后，**同时间同眼多 rect** 的场景已显著减少（本次未出现）。
> P3 的主要收益方向是**左右眼跨 crop_mode 合并**，而非同眼多 rect 合并。

### 设计原则

- **不退回 union 合并**：会把小 rect 的 lada 时间从 ~9 分钟膨胀到几十分钟。
- **不影响缓存内容键**：每个 rect 仍按 (time, rect) 缓存为独立文件，paste 阶段
  无须改动。
- **不影响 mux 输出**：每个 rect 仍是独立 mp4，restore 单独跑。

### 实施方案

把当前"按 segment 串行 + 按 crop_mode 串行"的 extract 循环改成"按时间窗口
分组、组内多 rect 多 crop 共享同一遍 decode"：

1. **分组**：以 `(round(start*fps), round(end*fps))` 作为 key，把同时间窗口的
   所有 paired segment（L + R 各一份甚至多份）聚成一个 group。
2. **逐组解码一次**：每组调用一次 GPU 解码器，按 stride=1 拿原帧。
3. **多 NVENC session 并发出多文件**：组内每个 rect 一个独立 NVENC encoder +
   raw 文件 sink；在 decode loop 内 `for rect in group: enc[rect].feed(slice)`。
   左右眼仅是 crop 的 x 偏移不同，slice 操作是 GPU 上的零拷贝。
4. 已有 paste 流水线就是"一遍 decode 喂多块"的模式（在主帧上并行修改多个矩形区），
   可以照搬其 active-session 管理与编码器并发释放策略。
5. 命中缓存的 rect 在分组阶段就剔除，不要占用解码 slot。
6. cancel token 在外层 decode loop 检查；任一编码器报错则停止整组并清理 raw。

### 边界

- 时间**部分重叠**的 segment 暂不合并（实现复杂度高，本次样本中未出现；
  之后真出现再做 P3 v2）。
- 同一 group 内 rect 数量上限建议设个配置项（如 4），避免 NVENC session 开太多
  造成显存爆。

### 预期收益

本次段 2 中 3 个时间窗口，每个节省 1 次 8K NVDEC。decode 端节省 50% 即可
回收约 3–5 分钟。段 3 同理再省 1–2 分钟。

### 验证

- 单测：构造同时间窗口 L + R 各一 rect 的合成场景，验证两个输出文件帧数与
  rect 切出位置正确，断言 decoder `frame_at` 调用次数为 N 而非 2N。
- 真实样本：在改造前后跑同一素材，比较 wallclock 与每个 rect 输出文件的
  hash（必须 byte-equal）。

---

## P4：fine 阶段降低扫描成本（不牺牲召回率）

### 修订说明

**上一版曾建议把 fine 阶段 imgsz 从 2048 降到 1280 提速。这条路线已撤回。**

YOLO 推理成本确实与 imgsz² 大致成正比，但 imgsz 同时决定了"网络眼里小目标
的有效像素数"。把 2048 降到 1280 后：

- 一个 528×528 像素的小马赛克，在 imgsz=2048 下网络看到的有效尺寸 ≈ 528 px；
  在 imgsz=1280 下被 letterbox 缩小到 ≈ 330 px。
- mosaic 检测器对小目标依赖纹理细节判别，下降 ~40% 有效像素 → **置信度系统性
  降低**，conf=0.50 阈值下漏检率上升。
- 实测样本里 fine 扫描的低置信度 paired pair（0.50 ~ 0.55）已经被人工拉到阈
  值线上；降 imgsz 会把这部分直接打掉。

下面是替代方案，**目标只是降低 fine 扫描时间，召回率不变**。

### 现状

- fine 阶段对每个粗 mosaic clip 全片扫描，stride=0.5s。
- 本次段 2 的 fine 扫描 9'15"：完整 725s clip 双眼各扫一次，约 1450 帧 × 2 次
  detector inference。
- 实测中大量样本帧**完全无检测**（log 显示 `aggregated 0 transformed segments`
  的段 0 / 段 1 整段都是空扫描），但仍按 stride 0.5s 全扫。

### 实施方案

#### 4a. stride 自适应

- 默认 stride 保持 0.5s 不变。
- **当某个段被粗扫标为 mosaic 但 fine 全段都没命中（如本次段 0、段 1）**，
  下一次同样源若再扫到此段，可以用上次 fine 结果作 cache hint，stride 翻倍
  到 1.0s 复验。命中复现就 finalize 为"段内无 fine 命中"，未命中再回到 0.5s
  作进一步保险。
- 这个 cache 用 source video 的 mtime + interval 范围作 key 落到 tmp 目录；
  失效条件：源文件变化或用户手动清理。

#### 4b. 连续无命中早停 + 跨步跳进

- 在 fine 扫描的逐帧循环里维护一个 `consecutive_miss` 计数：
  - 命中（任意 box 通过 conf 阈值）→ 清零。
  - 连续 N 次未命中（N = 30，即 15 秒 stride=0.5s）→ 切换到"扫描模式 B"：
    stride 改为 4s（每 8 个原 stride 抽 1 个）。
  - 一旦扫描模式 B 命中 → 回到附近 4s 窗口内用原 stride 重扫一遍补全。
  - 命中后切回扫描模式 A。
- 等价于：长时间无马赛克的段（黑屏、转场、过渡画面、广告位）按 8 倍 stride
  快速跳过，命中段保持原密度。**召回率几乎不变，因为命中段一旦冒头就回原密度。**

#### 4c. detector batch 增大

- 当前 `pre_extract_yolo_batch` 默认 4。在 8K → imgsz=2048 输入下，单帧 inference
  cost 主导，GPU 利用率其实较低（实测 5–7 samples/s 远低于 GPU 算力上限）。
- 提到 batch=8 或 16 不会显著增加显存（detector model + activations 已加载），
  能更好打满 GPU 流水线，吞吐 +20%~30%。
- 风险：单段视频帧数较少时（< batch 大小）退化为同步串行，需要 flush_batch
  逻辑兜底（已有）。

#### 4d. 用户可选的"激进 fine"档（仅作为高级开关）

- 配置 `pre_extract_fine_detector_imgsz`，默认与粗扫一致（不下调）。
- GUI 暴露一个高级开关："极速 fine 扫描（可能漏检小马赛克）"，启用时把
  imgsz 降到 1280。**用户明确接受 quality/speed 权衡时才生效**。
- 默认关闭。

### 预期

- 4a + 4b 合计：本次段 0、段 1 的空扫从 42" 降到 ~15"；段 2 的 9'15" 估降到
  7'~7'30"；段 3 的 4 分钟扫描估降到 3 分钟。综合 -3~5 分钟。
- 4c：batch 调大，再 -10%~20%。
- 4d 是兜底，给愿意承担风险的用户。

### 风险

- 4a 缓存正确性：cache key 必须包含粗扫 conf、fine conf、stride、detector 模型
  hash，任一变化即失效。
- 4b 跨步跳进可能错过非常短的 mosaic 片段（< 4s 且 conf 在阈值线上）。可以
  通过 N（无命中阈值）调节保守度。
- 4c 显存峰值需实测；ultralytics 默认 batch 推理对 imgsz=2048 的显存占用
  大致与 batch 成正比。

---

## P5：Stage 4 timeline merge 用 concat demuxer 直读

### 现状

Stage 4 当前流程：

1. 对 timeline 中每个 mp4 分片（mosaic_segN.restored.mp4 / gap_segN.mp4）
   ffmpeg `-c copy -bsf:v hevc_mp4toannexb -f hevc partNNNN.hevc`
2. 把 7 个 part 文件 cat 拼成 combined.hevc
3. ffmpeg `-f hevc -i combined.hevc -i source.mp4 -c copy -movflags +faststart out.mp4`

本次实测 8' 全花在第 1+2 步：每个分片走一遍 ffmpeg `-c copy` 转 annex-B 输出，
对大文件（seg002.restored.mp4 ~11GB）单独走一次读写就是 45s+。

### 实施方案

跳过中间 part 文件，**直接用 ffmpeg concat demuxer 一次性串入 mp4 → annex-B → mp4 mux**：

1. 写一个 concat 列表文件：
   ```
   file 'mosaic_seg000.restored.mp4'
   file 'gap_seg000.mp4'
   ...
   ```
2. 单条 ffmpeg 命令：
   ```
   ffmpeg -f concat -safe 0 -i list.txt -i source.mp4 \
     -map 0:v -map 1:a? -c copy -movflags +faststart \
     -bsf:v hevc_mp4toannexb out.mp4
   ```
3. 风险点：concat demuxer 要求所有片段的 codec 参数、SPS/PPS 严格一致；
   现有路径里 mosaic seg 的 restore 用 hevc_nvenc P4 编码，gap seg 是 stream-copy
   的源 HEVC，编码器/参数不同。如果直接 concat 失败：
   - 退化方案 a：维持 part 文件方案，但把读 mp4 / 写 annex-B 改成多线程并发
     （每个分片一个线程，最多 4 个）。
   - 退化方案 b：保留 part 文件，但把 mosaic seg 的 paste 输出直接保留为 raw
     HEVC（不 mux 成 mp4，结合 P7），Stage 4 直接 cat。

### 预期

成功路径下 8' → 2-3'。否则退化方案至少 -2 分钟。

### 验证

输出 mp4 帧数、duration、md5（video stream）与现有路径一致。

---

## P6：最终 mp4 输出提供 "fast local" 模式

### 现状

最终 mux 步骤带 `-movflags +faststart`：把 combined.hevc + audio 写完后，
**整个文件重读一遍把 moov atom 搬到 mdat 之前**。对本次 11.6GB 输出，这步
单独花了 ~7 分钟（全程 IO 受限）。

faststart 对 HTTP 渐进流式播放有意义；对本地播放器、NAS、复制到 VR 头显
本地播放，**毫无收益**。

### 实施方案

1. 新增配置 `output_mp4_faststart = "auto" | "always" | "off"`，默认 `auto`。
2. `auto` 在最终输出文件 > 4GB 时自动关闭 faststart（典型大输出基本不会用
   HTTP 流播）；其余保持开启以兼容渐进流。
3. GUI 增加一个开关："输出 mp4 兼容流式播放（开启会增加 5–10 分钟写盘时间）"。
4. 若用户明确选 always，再带 `+faststart`。
5. 关闭 faststart 的情况下保留其他 metadata 设置（颜色空间 bsf 等）。

### 预期

大输出场景默认省 5–7 分钟。

### 风险

低。播放兼容性受影响的场景仅限"上传到 web、用浏览器边下边播"。

---

## P7：restore raw HEVC 跳过中间 mp4 mux，paste 直接消费

### 现状

每个 paired segment 的 restore 流程：

1. lada/jasna restore → raw HEVC bitstream（已是 GPU NVENC 输出的 annex-B）
2. ffmpeg mux raw HEVC + （可选 audio 源） → segN.restored.mp4
3. paste 阶段重新打开这个 mp4 → demux HEVC → NVDEC decode → blend

第 2 步对每段 1–3GB raw HEVC 做一次完整 IO；第 3 步又把它 demux 一次。
本次 segment 2 的 6 个 restored.mp4 mux 累计约 3–4 分钟。

### 实施方案

让 restore 端**保留 raw HEVC + 一份描述 sidecar**（含 fps / 颜色元数据 /
帧数），paste 端直接消费 raw HEVC（NVDEC 支持直读 annex-B）：

1. restore 输出文件名约定：`<key>.restored.hevc` + `<key>.restored.json`
   （sidecar，仅含帧数 / fps / color metadata）。
2. paste 端 segment 入口判断后缀：`.hevc` 时绕过 mp4 demux，用 sidecar
   里的 fps/color 直接构造 NVDEC 输入。
3. 当用户启用 `pre_extract_keep_segments=True` 调试时，仍可选附带 mp4 mux
   方便查看；默认 False 时直接走 raw HEVC。
4. cleanup 同步按 (raw.hevc, sidecar.json) 配对清理。

### 风险

- 影响 P5 退化方案 b（Stage 4 直接 cat raw HEVC 配 stream-copy mp4 mux）。
  两者协同；先做 P7 再做 P5 可以最大化收益。
- raw HEVC 文件没有时长 metadata，需要 sidecar 保障可恢复性。

### 预期

每段 mux + demux 双向节省，本次合计 -2~3 分钟。中间临时文件磁盘占用减半。

---

## P8：paste 内部"无 rect 帧"按 keyframe 切片 stream-copy

### 现状

paste 8192×4096 10bit 全程走 NVDEC → blend → NVENC。本次段 2 paste 43442
帧 @ 37 fps 共 21'；3 个 active 时间窗口 [270-12600] [32310-39150] [40050-42900]
合计 ~22000 帧，剩余 ~21000 帧（48%）落在"无 rect"区间，**仍被重新解码、
重新编码、画面零变化**。

> 注意：Stage 4 已有"mosaic vs gap"粒度的 stream-copy（不重编 gap_seg），但
> mosaic seg 内部仍是整段 paste 重编。P8 的对象是 mosaic seg 内部的子 gap。

### 实施方案

1. 进入 paste 前，把整段时间轴按 active rect 的并集分成 N 个"重编区间"
   与 M 个"透传区间"。
2. **透传区间**：按 keyframe 对齐切出 base clip 对应片段，stream-copy 保留原 HEVC 流。
3. **重编区间**：走当前 GPU paste 流水线，输出 raw HEVC。
4. 末尾用 concat（可与 P5 共用 demuxer 方案）把 N+M 个片段串成最终段输出。
5. 透传区间向两侧膨胀到最近 keyframe；如果膨胀后两个重编区间靠得很近
   （比如间距 < 60 帧），合并成一个重编区间避免碎片化。
6. 与 Stage 2 那套 `align_segments` 逻辑共用代码。

### 预期

本次段 2 paste 21' → 估 9'~11'，段 3 paste 9' → 估 4'~5'。两段合计 -10~12 分钟。

### 风险

- HEVC 流拼接对 SPS/PPS/VPS 一致性极敏感，重编片段的 NVENC 参数（profile /
  tier / colour metadata）必须与原始流匹配，否则部分播放器卡顿。
- keyframe 对齐导致重编区间会向两侧膨胀，实际收益略低于估算。
- 复杂度最高；建议 P0–P7 落地、流程稳定后再启动。

---

## P9：extract / restore 跨段流水线并行

### 现状

paired 分支当前严格串行：extract A → restore A → extract B → restore B → ...
extract 主要瓶颈在 NVDEC + NVENC（输出小 mp4），restore 主要瓶颈在 PyTorch
inpainter compute + NVENC。两者属于不同的 GPU 子系统分工，串行运行时存在
大段互相等待。

### 实施方案

1. 用一个固定深度（如 2）的"已就绪输入"队列：
   - extract 工作线程持续生产 raw HEVC（结合 P7）
   - restore 工作线程持续消费 → 输出 raw HEVC
2. extract 与 restore 共用一个 cancel token；任一线程命中 cancel 立即向另一边
   传播。
3. 多个 GPU 操作各自独立 CUDA stream，依赖单例 inpainter；inpainter 调用入口
   需要确认线程安全（必要时加细粒度锁，确保模型推理串行但 IO/decode/encode 并行）。
4. 失败处理：extract 失败时 restore 已开始的那段要么丢弃 raw 要么完成后删除。
5. 显存预算监控：超过阈值时强制串行化保险。

### 预期

extract 在小 rect 上 ~10s，restore ~9 分钟；并行后 extract 的 wallclock 被
吸收。本次段 2 有 6 个 segment，预计省 10–15%。对碎段多的素材收益更大。

### 风险

- inpainter 与 NVDEC 共享 GPU 内存池可能引发显存峰值升高。
- 单测难度高，需要先以 mock 验证 producer-consumer 语义。

---

## P10：粗扫 conf / 段最短时长 A/B 微调

### 现状

实测段 0（0–20s）和段 1（138–168s）粗扫被纳入 mosaic interval，进入 paired
fine 后 **0 个有效检测**，整段透传。属于粗扫假阳性，每个白白多花 20–30 秒
（fine 扫描本身的成本）。

### 实施方案

- 粗扫 conf 试调（如 0.20 → 0.25），观察前后召回率。
- `pre_extract_merge_gap_s` 与 `pre_extract_min_segment_s` 可以略保守，减少
  边缘碎片化造成的小段。
- 都需要 A/B 数据支撑。**不要在没有样本的前提下盲调。**

---

## P11：端到端码率契约（中间放宽、最终收敛、不动 UI）

### 设计原则

- **不在 UI 上加新选项**。沿用现有"保持源码率"开关，技术上选最优默认即可。
- **中间阶段允许码率扩大（保留质量裕度）**：per-rect lada restore mp4、其它内部
  转码可按 `gpu_bitrate_multiplier=1.5` 走，给 paste 阶段留出再编一次的画质裕度。
- **最终落到输出 mp4 的那一段编码必须收敛**：mosaic 段在 Stage 3 出口的 paste
  编码，目标码率严格等于源码率。Stage 4 / 最终 mux 都是 stream-copy，不能再
  纠正——所以收敛必须在 paste 编码现场完成。

### 当前码率链路

每个时间段最终在输出 mp4 中的码率，由它在 Stage 3 末端进入 Stage 4 前那一份
mp4 决定。Stage 4 全程 stream-copy，最终 mux 也是 stream-copy，两者都不引入
任何重编。

| timeline 类型 | Stage 3 出口编码 | 进入 Stage 4 后 | 最终输出码率 |
|---|---|---|---|
| gap 段 | 复用 Stage 2 的 stream-copy（源 HEVC 原样） | stream-copy | **= 源码率** ✓ |
| mosaic 段（paired pre-extract） | GPU paste NVENC 整段 8K SBS 重编 | stream-copy | **由 paste 编码参数决定** ← 收敛点 |
| mosaic 段（fallback 全眼 lada） | 全眼 NVENC | stream-copy | 同上，由该编码参数决定 |

### 当前问题

#### 问题 1：默认乘数 2.0 太激进

paired pre-extract 出口处，当用户没有勾选"保持源码率"时：

- paste 调用收到 `bitrate_bps=None`，编码器目标码率解析回退到
  "源码率 × `gpu_bitrate_multiplier`（默认 2.0）"。
- 实测段 2 paste `bitrate=65979kbps`，正好是源 ~33 Mbps 的 2×。
- 结果：gap 段 ~33 Mbps、mosaic 段 ~66 Mbps，最终视频码率严重不均匀，
  mosaic 段文件占双份大小但视觉上没有 2× 的必要。

#### 问题 2：中间产物与最终段共用一个乘数

`gpu_bitrate_multiplier` 同时被用于：

- per-rect lada restore 输出 mp4（**中间产物**，会被 paste 阶段重新解码消费）
- Stage 3 paste 出口（**最终落盘**，进 Stage 4 stream-copy 之后无法再纠正）

两个场景的需求不同：

- 中间产物可以放宽（保 restore 质量、给 paste 提供高质输入）；
- 最终段必须收敛（与 gap 段对齐，避免输出膨胀）。

共用一个乘数，无论改成什么值都会有一边吃亏。

#### 问题 3："保持源码率"开关未审计所有出口

`keep_original_bitrate` 在 paired pre-extract 出口被正确转发到 paste 编码，
但需要审计另外几个 NVENC 出口：

- 全眼 fallback 路径（用户禁用 pre-extract / paired 扫描降级）
- 单眼 fallback 路径
- 左右眼合并的 merge 路径（含 fisheye 变体）

任一节点漏掉，"保持源码率"就变成"部分段保持"。

### 修复方案

#### 11a. 拆开"中间乘数"和"最终乘数"

- `gpu_bitrate_multiplier` 保留语义，但**收窄到只作用于中间产物**：per-rect
  lada restore mp4 mux、extract clip 等不会直接进入最终输出的编码。默认改成
  **1.5**（比当前 2.0 节省 1/4 存储，仍给 paste 阶段留 50% 质量裕度）。
- 新增一个内部默认 `gpu_bitrate_final_multiplier`（仅 app_config 可见，**不上 UI**），
  默认 **1.0**。作用于"会被 Stage 4 stream-copy 进最终输出"的 NVENC 出口：
  - paired pre-extract 的 paste 出口
  - 全眼 / 单眼 fallback 的 restore 出口
  - 左右眼 merge 路径出口
- 两个乘数都受 `keep_original_bitrate=True` 总览：开关启用时，**两者都强制为 1.0**，
  最小磁盘占用。

效果：

- 默认情况（用户没选保持源码率）：mosaic 最终段 = 1.0× 源，gap = 1.0× 源，
  **输出码率均匀**；中间 per-rect mp4 = 1.5× 源（按 rect 面积比例），保留
  restore → paste 之间的质量裕度。
- 用户选了保持源码率：所有阶段 = 1.0× 源，所有中间产物也节省存储；最终画质
  仍由 NVENC P4 + VBR 兜底。

#### 11b. 用基线质量码率作下限保护

代码里已经有按分辨率估算的"基线质量码率"函数（之前用于源码率未知时）。
两个乘数都挂上它作下限：

```
target = max(source_bps * multiplier, baseline_quality_bps(out_w, out_h, fps))
```

防止极端低码率源（< 10 Mbps 的 8K）下，最终段也跟着劣化。

#### 11c. 统一码率决策函数 + 审计所有 NVENC 出口

封装一个内部函数 `resolve_pipeline_bitrate(stage, out_w, out_h, fps,
source_bps, keep_original) -> int`，`stage ∈ {intermediate, final}`。所有
NVENC 调用点都从它取目标码率，不再各自 `if keep_original_bitrate ... else None`。

需要改造的出口（按现有代码搜 `keep_original_bitrate` 引用即可清点）：

- paired pre-extract paste → `stage=final`
- per-rect extract / lada restore mp4 mux → `stage=intermediate`
- 全眼 / 单眼 fallback restore → `stage=final`
- 左右眼 merge 路径（含 fisheye 变体）→ `stage=final`
- split 路径（眼睛片段会被 lada 再处理）→ `stage=intermediate`

改造后审计变成"看哪些调用点还没接入 `resolve_pipeline_bitrate`"，机械可查。

#### 11d. 日志显式标注 stage 与决策路径

现有 `[gpu-encoder] ... bitrate=Xkbps maxbitrate=Ykbps` 日志保留，再加一行：

```
[bitrate] stage=final source=33000kbps keep_original=True -> target=33000kbps
[bitrate] stage=intermediate source=33000kbps -> target=49500kbps (×1.5)
```

让用户在日志里能验证"我选了保持源码率，最终段确实就是源码率"。

#### 11e. 最终输出后的整体码率自检

最终 mp4 mux 完成后，打一行：

```
[bitrate] final mp4: 33215 kbps avg (source 33000 kbps, ratio 1.007×)
```

读最终 mp4 的 duration / size 算平均码率，和源码率对比，让用户立刻可见
"是否真的收敛了"。如果 ratio > 1.20 就额外 warning，便于发现某个出口没接入
契约。

### 不需要做的事

- **不在 UI 增加任何新控件 / 新档位**。现有"保持源码率"开关足够。
- **不在 Stage 4 加重编节点**。Stage 4 stream-copy 是性能基础，破坏它每次多花
  7–15 分钟。
- **不"先临时大码率再压缩"两步走**。paste 编码一次到位即可命中目标码率
  （NVENC VBR + bitrate 参数）。

### 实施顺序

1. **11a + 11b**：~半天。拆乘数 + 默认值 + 下限保护。无 UI 改动，输出体积立即
   归位。
2. **11c**：~半天。统一决策函数 + 把所有 NVENC 出口接入。把零散 if 收敛到
   一处，未来不会再有某条路径漏掉 `keep_original_bitrate` 的回归。
3. **11d + 11e**：~半天。日志标注 + 体积自检。用户体验加分 + 回归防线。

合计约 1.5 天，全部内部逻辑改造，**不动 UI、不动 timeline 结构**。

### 验证

- 跑同一素材，对比改前改后的最终 mp4：
  - duration 完全一致
  - video stream md5 在 gap 段一致（stream-copy 不变）
  - mosaic 段改后平均码率落到源码率附近（±5% 内）
  - 总输出文件大小约缩到改前 60–70%
- 单测：构造伪 source（带 bitrate metadata），两种 `keep_original_bitrate`
  取值 × 两种 stage 类型，断言 `resolve_pipeline_bitrate` 输出符合预期。
- 回归测试：跑全眼 fallback 与单眼 pipeline 各一遍，确认 final mp4 平均码率
  也落在源附近。

## 已落地基础（修改时应保留）

- **检测器与 inpainter 已是进程级单例**，跨段不会重新加载模型。
- **paired 分支缓存改 frame-key 命名**（最新一版）：缓存语义稳定，
  P3 改造同时间多 rect 合并 decode 时仍能分别命中独立的 rect 缓存。
- **paste 阶段同帧多 active rect 已正确支持**：rect 互不重叠时各自 in-place
  修改主帧切片；不需要任何"合并 rect"才能工作的假设。
- **Stage 4 已有 mosaic / gap 粒度的 stream-copy 拼接**：P8 是在此基础上向
  "mosaic seg 内部 gap"细化，不冲突。
- **fine 阶段 conf 已从 0.40 调到 0.50**：同时间同眼多 rect 噪声显著减少。
  P3 的方向相应聚焦到"左右眼跨 crop_mode 合并"。
- **`keep_original_bitrate` 路径在 paired pre-extract 出口正确转发**：P11
  的契约函数改造在此基础上做"全路径审计 + 默认值修正 + 中间/最终乘数分离"，
  不是从零搭建，也不引入新的 UI 选项。

---

## 推荐落地顺序

| 批次 | 项目 | 预估工时 | 选择理由 |
|---|---|---|---|
| 1 | P0 + P2 + P11a/b | 0.5–1 天 | 风险最低，立刻见效（取消正确性 / 日志可读 / 输出体积归位） |
| 2 | P6 + P11c/d | 1–1.5 天 | 大输出 faststart 选项 + 码率契约函数收敛 + 日志标注 |
| 3 | P4 + P11e | 1–1.5 天 | fine 扫描保守加速 + 最终码率自检 |
| 4 | P1 + P3 | 2 天 | 中等改动，主 wallclock 收益 |
| 5 | P5 + P7 | 2 天 | IO 链路优化，互相协同 |
| 6 | P9 | 1.5 天 | 并发模型设计，显存峰值监测 |
| 7 | P8 | 2–3 天 | 复杂度最高，主路径稳定后再启动 |
| 8 | P10 | 按需 | A/B 数据驱动 |

每批次都建议保留 `keep_intermediate=True` 跑同一组样本对比 wallclock + 输出
hash（restore 段输出 hash 必须一致），同时跟踪日志大小变化与最终 mp4 完整性
（duration / frame count / 颜色空间元数据 / 整体码率）。

### 短期最优组合

P0 + P2 + P11a/b + P6 共约 1.5 天：

- **P0** 修取消行为
- **P2** 砍日志
- **P11a** 拆中间/最终乘数：中间 1.5×（裕度）、最终 1.0×（与 gap 段对齐）
- **P11b** 用基线质量码率作下限保护（极低源码率不至于劣化）
- **P6** 关闭大输出 faststart（省 5~7 分钟）

合计**省 10~15 分钟/次 + 输出体积约缩 30~40% + 取消正确 + 日志可读**，
**完全不动 UI**，全部是"内部默认值 / 行为选项"层面的改动，回退极容易。

然后做 P11c/d（码率契约函数收敛 + 日志标注）和 P1 + P3（GPU 扫描 + 跨眼
合并解码）。
