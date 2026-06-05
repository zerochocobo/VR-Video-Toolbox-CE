# OneClick 全链路优化剩余项详细实施方案

承接 [summary_20260604_ONECLICK_PIPELINE_OPTIMIZATION_PLAN_CN.md](summary_20260604_ONECLICK_PIPELINE_OPTIMIZATION_PLAN_CN.md)。

已落地 commit：`4f2dd92`（P0 / P2 / P6 / P11a/c/d 主要出口）和 `476508a`
（P11b 系数 + P11e 自检 + P4c batch×2 + P1 GPU keyframe 粗扫 + P5 concat
demuxer 直读）。

本文按优先级展开剩余项的详细实施方案：

| Pri | 项目 | 工时 | 风险 |
|---|---|---|---|
| P3 | paired 同时间窗左右眼跨 crop_mode 合并解码 | ~1.5 天 | 中 |
| P7 | restore raw HEVC 跳过中间 mp4 mux，paste 直接消费 | ~1 天 | 中 |
| P9 | extract / restore 跨段流水线并行 | ~1.5 天 | 中 |
| P8 | paste 内部"无 rect 帧"按 keyframe 切片 stream-copy | ~2–3 天 | 高 |
| backlog | P4a/P4b 命中缓存与无命中早停 | 按需 | 低–中 |
| backlog | P10 粗扫 conf / 段最短时长 A/B 微调 | A/B 后定 | 低 |

---

## P3：paired 同时间窗左右眼跨 crop_mode 合并解码

### 现状

paired pre-extract 阶段，对每个 paired segment 串行执行
"GPU 解码 base clip → GPU crop（左眼或右眼）→ 切单 rect → NVENC 出 mp4"。
左眼跑完所有 segment 后再跑右眼。

实测段 2 三个时间窗口都是"L 1 rect + R 1 rect"形态，等价于同一段 SBS 源
被 NVDEC 解码 2 次（左右各 1 次）。在 8K 上 decode 端是显著带宽成本，
合并后可立即砍掉一半 decode 工作量。

### 设计原则

- **不合并 rect**：每个 rect 仍按 (time, rect) 落独立缓存文件，paste 阶段无需改动；
  也不破坏 frame-key 命名契约。
- **不影响 lada restore 的并发模型**：restore 仍然按文件串行（这一步是 P9 的范围）。
- **只在 extract 阶段合并**：同一时间窗口的多 rect（左眼任意数 + 右眼任意数）
  共享一遍 decode，并发 NVENC 多个独立 raw / mp4。

### 数据流改造

#### 1. 分组键

对 paired segment 列表（左眼一组、右眼一组）合并成一个"待 extract 任务列表"：

```
ExtractTask = {
    "side": "left" | "right",
    "seg": MosaicSegment,        # 原 seg
    "seg_in_path": str,           # 内容键命名后的 mp4 输出
    "rect": (x, y, w, h),
    "start_frame": int,
    "end_frame": int,
    "bitrate_bps": int | None,
}
```

把所有 ExtractTask 用 `(start_frame, end_frame)` 作为 group key 分组：

```
TimeGroup = {
    "start_frame": int,
    "end_frame": int,
    "tasks": list[ExtractTask],   # 同时间窗内的所有 rect（含左右眼）
}
```

不同 group 之间继续串行，组内同遍 decode、多 rect 并发出文件。

#### 2. 缓存命中剔除

`_paired_segment_paths` 现有 frame-key 命名已经稳定。分组前先逐个 task
检查 seg_in 是否存在；存在且字节非空就标 `cached=True`，剔除出 group
不占解码 slot。group 内全部 cached 就跳过整组。

#### 3. 多 rect 共享 decode 的核心循环

新增内部函数（位于 GPU files 模块）：

```
extract_multi_rect_clip(
    src_path,
    tasks: list[ExtractTask],     # 同 (start_frame, end_frame, side?) 不要求；混合 left/right 允许
    to_fisheye: bool,
    start_sec: float,
    end_sec: float,
    cancel_token,
    log_callback,
)
```

实现要点：

- 单一 `PyNvThreadedSerialDecoder(start_frame=start_idx)` 实例
- 帧循环 `for i in range(start_idx, end_idx)`：
  - 取一帧得到 `(y_full, uv_full)`
  - 若 `to_fisheye=True`：先做 v360 → 完整左眼 / 完整右眼的 fisheye（沿用现有 lut 代码）
  - 对每个 task：
    - 按 `task.side` 选 crop 偏移（左眼 x=0 / 右眼 x=eye_w）
    - 在 GPU 上 slice 出 `(rx, ry, rw, rh)`
    - 通过对应的 NVENC encoder feed 一帧
- 每个 task 维护独立的 `PyNvEncoderSession` + `_EncodeSink` + raw 文件句柄
- 进度日志按组打：`[gpu] extract multi-rect group N: tasks=K, frames=F`

#### 4. 主调度

把现有 `_extract_restore(side, segments, x_offset)` 双调用改成：

1. 构造 ExtractTask 列表（含左右两眼），保留缓存命中过滤。
2. 按 group key 分组、按 `start_frame` 排序。
3. 对每个 group：
   - 若 `len(tasks) == 1`：走原有 `extract_transformed_rect_clip`（保持回归一致）。
   - 否则走新的 `extract_multi_rect_clip`。
4. 所有 extract 完成后，restore 阶段保持现状（按文件串行调 `process_lada`）。
5. paste 阶段不动。

### 边界

- **时间部分重叠不合并**：本期范围只合并 `(start_frame, end_frame)` 完全相同的
  segment。部分重叠的延后到 P3 v2（实现复杂度高、收益取决于命中模式、本次
  实测样本未出现）。
- **同一 group 内 task 数量上限**：默认 8 个（足够覆盖左右眼各 4 rect 极限场景），
  超过走回退到独立 extract。配置 `pre_extract_extract_group_max`。
- **NVENC session 数量**：每个 session 占用编码器实例与显存；同 group 内 N 个
  session 并发。RTX 5060 Ti 上估测 4 session @ 528×528 占用 < 200 MB，安全。
- **bit depth / 颜色一致性**：组内所有 task 共享 base clip 的 bit_depth 与
  color metadata；这一点天然成立。
- **fisheye 模式**：v360 变换的 LUT 是按 eye_w × eye_h 缓存的，左右眼独立 LUT，
  组内任务也独立持有引用即可。

### 失败处理

- decode 端任一异常 → 取消所有未完成 task 的 raw（清理 `.raw.hevc`），抛出。
- 任一 NVENC encoder 出错 → 同 group 内其他 encoder 也强制 flush/close，
  对应 task 的 raw 删除；上层视为"该 group 全失败"，下次重跑会重新命中
  frame-key cache（保留成功的、重做失败的）。
- cancel token 在每帧循环检查；命中即抛 `OperationCancelled`，所有 raw 清理。

### 测试

- 单测 A：mock decoder，构造同时间窗 L+R 各 1 rect，断言 `frame_at` 调用
  次数 == 帧数（不是 2× 帧数），两个输出文件帧数一致、rect 切出位置
  对应 task.rect。
- 单测 B：3 个 task（L×1 + R×2）共享同时间窗，断言 3 个独立 raw 文件
  且 frame_at 调用一次。
- 单测 C：构造 2 个不同时间窗（每个 L+R），断言生成 2 个 group、独立
  decode、合计 4 个文件。
- 单测 D：cancel token 在第 N 帧置位，断言所有 raw 被清理、抛 OperationCancelled。
- 单测 E：组内 1 个 task，验证降级走 `extract_transformed_rect_clip` 原路径
  （hash 完全一致）。
- 回归：跑同素材改前改后，断言每个内容键 mp4 的 video stream md5 完全一致。

### 工时

- 分组数据结构 + 主调度改写：0.5 天
- `extract_multi_rect_clip` 实现（NVENC 多 session 管理）：0.5 天
- 单测 + 真实素材回归：0.5 天

合计 ~1.5 天。

---

## P7：restore raw HEVC 跳过中间 mp4 mux，paste 直接消费

### 现状

每个 paired 段的 restore 全流程：

1. lada/jasna restore → GPU NVENC 输出 raw HEVC 字节流
2. ffmpeg 把 raw HEVC mux 成 mp4（带颜色 metadata + faststart）
3. paste 阶段重新打开 mp4 → 解 demux → NVDEC decode → blend

第 2 步对每段 1–3 GB raw 做一次完整 IO；第 3 步又 demux 一次。本次实测段 2
的 6 个 restored.mp4 mux 累计 3–4 分钟，磁盘也多一份 raw 副本。

### 设计原则

- restore 端**保留 raw HEVC + 一份 sidecar JSON**（含解 mux 必需的元数据）。
- paste / Stage 4 merge 端**直接消费 raw HEVC**，不再走 ffmpeg demux。
- 与 P5 协同：Stage 4 fast HEVC merge 已经把 mp4 → annex-B 作为兜底，
  P7 让 source 端直接是 annex-B，省一次转换。
- 用户在 `pre_extract_keep_segments=True` 调试场景下可选择追加 mp4 mux
  （便于本地播放查看），默认 False 时直接走 raw。

### sidecar 格式

`<key>.restored.hevc` 旁边写 `<key>.restored.json`，纯文本 UTF-8：

```json
{
  "format_version": 1,
  "kind": "restored",
  "codec": "hevc",
  "width": 528,
  "height": 528,
  "bit_depth": 8,
  "fps_num": 60000,
  "fps_den": 1001,
  "frame_count": 26580,
  "color": {
    "primaries": "bt709",
    "transfer": "bt709",
    "matrix": "bt709",
    "range": "tv"
  },
  "encoder": "hevc_nvenc P4 vbr 1234kbps",
  "source": "<原 base clip 路径>",
  "rect": {"x": 1792, "y": 1248, "w": 528, "h": 528},
  "time": {"start_s": 212.214, "end_s": 655.653, "start_frame": 12720, "end_frame": 39300}
}
```

字段都是 NVDEC 重开必须的最小集合 + 调试用的来源信息。**写入是原子的**：
先写到 `<key>.restored.json.tmp`，fsync，再 rename。

### 改造范围

#### 1. native restore 出口

在 `_restore_file_gpu_nvenc` 完成 raw 输出后：

- 若入参 `produce_mp4=True`（默认）：保持现状走 mux。
- 若 `produce_mp4=False`：跳过 mux 步骤，把 raw 文件改名为
  `<output_path 去 .mp4 改成 .hevc>`，同时写 sidecar JSON。

入口 `restore_file(...)` 增加参数 `produce_mp4: bool = True`，向下透传。

#### 2. paste 端消费

paste 现在的代码：

```
PyNvThreadedSerialDecoder(seg.path, bit_depth=seg_bd)
```

`seg.path` 当前指向 mp4。改造为：

- 若后缀 `.hevc`：旁边读 sidecar JSON，构造一个轻量"raw HEVC 解码"包装：
  - 用 PyNv 的 NVDEC 直接消费 annex-B 字节流（PyNvCodec 支持 raw 输入，
    需要确认现有 `PyNvThreadedSerialDecoder` 接口是否能传 raw + framerate；
    不能则新增 `PyNvRawHevcDecoder` 类，按 sidecar 的 fps / color / bit_depth
    构造）。
  - `frame_at(i)` 接口语义与现有一致（仅本地解码循环，不支持随机 seek 也无所谓——
    paste 已经是顺序访问）。
- 若后缀 `.mp4`：保持现路径不变（向后兼容、用户开 keep_intermediate 调试时还能用）。

#### 3. PasteSeg 元数据来源

`build_paste_segments` 当前用 `probe.probe_video(seg.path)` 拿 fps。改造：

- 优先读 sidecar JSON。
- 若 sidecar 不存在（mp4 路径），走原有 probe。
- 帧数从 sidecar.frame_count 拿，不再用 `len(decoder)`（避免对 raw 解码器
  做"先扫一遍数帧"的开销）。

#### 4. Stage 4 fast HEVC merge

`concat_timeline_hevc_fast` 当前对每个分片做 `-c copy -bsf:v hevc_mp4toannexb`
转换。改造：

- timeline entry 指向的若是 `.hevc` 文件：直接 cat 进 combined.hevc（甚至
  可以省掉中间 part 文件，逐 entry append 到 combined.hevc）。
- 指向 mp4：保持现路径（gap_seg 一定是 mp4，因为 stream-copy 自源；mosaic seg
  改造后可能是 .hevc）。
- concat demuxer 直读（P5）已经实现，对 raw .hevc 输入也要测试一遍，
  必要时走"分类拼接": .hevc 部分直接 cat、.mp4 部分先 demuxer 出 annex-B 再 cat。

#### 5. cleanup 路径

按 P11 改造遗留的 `segment_input_paths + restored_paths` cleanup，要把
`.hevc` 和 `.json` 也加进去：

```
for restored in restored_paths:
    base = restored_without_suffix
    for ext in (".restored.mp4", ".restored.hevc", ".restored.json"):
        _remove_file_quiet(base + ext)
```

### 边界

- **PyNv NVDEC raw 输入能力**：需要先确认现有 PyNv 模块是否能直接读 annex-B
  bytestream。如果不行，最稳的退路是 paste 端 popen ffmpeg
  `-f hevc -i raw.hevc -f nut pipe:1` 拿 NUT 容器再喂 NVDEC，仍然省了
  写 mp4 + faststart 的 IO。
- **颜色 metadata**：raw HEVC 不带 mp4 container 的 color tag，全部依赖
  sidecar。paste / merge 出口必须从 sidecar 读取并写到最终 mp4。
- **音频**：paired pre-extract 阶段的 segment 本来就 `keep_audio=False`，
  没有音频丢失问题。Stage 4 final mux 单独从源拼音轨，不受影响。
- **`pre_extract_keep_segments=True` 兼容**：开启时仍写 mp4（多花几分钟），
  方便用户用任意播放器检查段输出。

### 失败处理

- sidecar 写入失败 → 删除已写的 raw 文件、抛错。原子 rename 保证不会留半成品。
- paste 端读到 sidecar 解析失败 / 帧数不匹配 → fallback 用 ffmpeg 临时把
  raw 包装成 NUT 喂 NVDEC，记一行 warning。

### 测试

- 单测 A：mock encoder 写 raw + sidecar，断言 sidecar 字段完整、文件原子可见。
- 单测 B：paste 端给 raw + sidecar 输入，断言能正常构造 PasteSeg、frame_at 调用
  返回正确帧。
- 单测 C：concat_timeline_hevc_fast 给混合 .hevc + .mp4 entries，断言生成的
  combined.hevc 字节数 == 各 entry 的 annex-B 长度之和。
- 回归：跑同素材，对比改前改后的最终 mp4 md5（video stream 应完全一致，
  因为没改编码参数，只改容器封装）。

### 工时

- sidecar 格式 + restore 出口改造：0.3 天
- paste 端 raw 消费 + PyNv 能力确认：0.4 天
- Stage 4 merge .hevc 直拼：0.2 天
- 单测 + 回归：0.1 天

合计 ~1 天。

---

## P9：extract / restore 跨段流水线并行

### 现状

paired 分支严格串行：extract A → restore A → extract B → restore B → ...

- extract 阶段主要瓶颈：NVDEC + GPU crop + NVENC（输出小 rect mp4）
- restore 阶段主要瓶颈：PyTorch inpainter compute（GPU SM） + NVENC

两者主要竞争资源是 NVENC（同 GPU 单实例），但 NVDEC 与 inpainter compute
是独立子系统。串行运行时 restore 期间 NVDEC + GPU crop 空闲，反之 extract
期间 inpainter 空闲。

### 设计原则

- **不增加显存峰值超过 1.3×**：流水线深度只取 2（最多 1 个就绪输入 + 1 个
  正在 restore）。
- **inpainter 仍是单实例串行**：模型推理本身不并发，避免线程安全坑。
- **取消必须立即生效**：任一边命中 cancel token，另一边立即停止。
- **不改 paste 阶段**：paste 仍然等所有 restore 完成后串行 paste。

### 数据结构

```
PipelineJob = {
    "task": ExtractTask,        # P3 引入的同结构，单 task 模式
    "seg_in_path": str,
    "seg_out_path": str,
    "skip_extract": bool,        # 命中缓存时 True
    "skip_restore": bool,
}
```

两个工作线程：

- `extract_worker`：从待处理 job 队列取任务，做 extract，把结果放入"待 restore"
  队列（深度上限 1，达到上限就阻塞）。
- `restore_worker`：从"待 restore"队列取任务，做 restore；完成后通知主线程。

主线程负责：

- 把 job 入队
- 等所有 restore 完成
- 在任一线程报错或 cancel 时拉响两边的 cancel token

### 控制流

```
build_jobs() -> list[PipelineJob]
queue_to_extract = Queue(maxsize=∞)   # 主线程一次性塞满
queue_to_restore = Queue(maxsize=1)    # 流水线深度控制

extract_worker:
    while job = queue_to_extract.get():
        if job is SENTINEL: break
        if cancel_token.cancelled: raise OperationCancelled
        if not job.skip_extract:
            extract(job.task, ...)
        queue_to_restore.put(job)
    queue_to_restore.put(SENTINEL)

restore_worker:
    while job = queue_to_restore.get():
        if job is SENTINEL: break
        if cancel_token.cancelled: raise OperationCancelled
        if not job.skip_restore:
            restore(job.seg_in_path, job.seg_out_path, ...)
        completed.append(job)

main:
    spawn both workers
    wait for both with timeout-based join + exception propagation
    on any exception in either worker:
        cancel_token.cancel()
        drain both queues
        re-raise
```

### 实施步骤

1. 抽取现有 `_extract_restore` 内联的 extract+restore 调用到独立函数
   `_run_extract_one(task)` 和 `_run_restore_one(job)`。
2. 引入 `_pipelined_extract_restore(jobs, cancel_token, process_callback,
   log_callback)`，内部用两个 `threading.Thread` 跑上述循环。
3. 共用 cancel token：把现有 `gpu_files.CancelToken` 提升到外层，extract 和
   restore 各自从这个 token 读 cancelled 标志、`process_callback` 注册一次
   即可控制两边。
4. **inpainter 线程安全审计**：当前 `native_mosaic.get_engine()` 是单例。
   两个 worker 中只有 restore_worker 会调到它，所以 inpainter 调用本来就在
   单线程内串行——无需额外锁。但 `torch.cuda.empty_cache()` 等全局调用需要
   确认不会与 extract_worker 的 NVDEC/NVENC 操作打架。建议给两个 worker
   各自创建独立 CUDA stream，跨 stream 用 event 同步。
5. **NVENC 多 session 协调**：extract 用一个 NVENC session（小 rect 输出），
   restore 用另一个（rect 大小输出）。NVENC 实例数硬件上限通常 ≥ 3，
   2 session 并发安全。
6. **显存监控**：每完成 4 个 job 抽样一次 `torch.cuda.memory_stats()` 峰值，
   超过预设阈值（如 80% VRAM）就警告并临时把流水线 depth 降到 1（即下一个
   extract 等到 restore 完成才开始）。

### 失败处理

- extract 抛错 → cancel token 置位 → restore 立即抛 → 主线程 re-raise。
  已完成的 raw 文件保留（frame-key 命中下次复用）。
- restore 抛错 → 同上方向反转。
- cancel token 置位 → 两个 worker 各自下一帧检查时抛 `OperationCancelled`。
- 显存 OOM → restore worker 抛 → 流水线 depth 临时降 1 后重试一次；仍 OOM
  上报。

### 测试

- 单测 A：mock extract + mock restore，验证两线程能正确推进 N 个 job，
  完成顺序与提交顺序一致。
- 单测 B：mock extract 第 K 帧抛错，验证 restore 立即收到 cancel 退出，
  主线程 re-raise 原异常。
- 单测 C：mock 命中缓存的 job（skip_extract=True），验证直接走 restore，
  decode count == 0。
- 单测 D：cancel token 在 restore 半途置位，验证 extract 也立即停。
- 压力测试：在真实素材跑 8+ 个 job，对比 wallclock 与串行版本。

### 与 P3 / P7 的协同

- **先做 P3**：P3 把 extract 改成"组"为单位；P9 的 PipelineJob 应当对应一个
  group（不是 task），整组共享同一 decode pass。
- **配合 P7**：流水线传输的是 raw HEVC 文件路径，无 mp4 mux 等待。

### 工时

- 数据结构 + 两线程框架：0.5 天
- 显存监控 + CUDA stream 协调：0.4 天
- 单测 + 真实素材压测：0.6 天

合计 ~1.5 天。

---

## P8：paste 内部"无 rect 帧"按 keyframe 切片 stream-copy

### 现状

paste 阶段对整段 mosaic 段做 NVDEC → blend → NVENC。本次实测段 2 共
43442 帧 @ 37 fps 耗时 21'；3 个 active rect 时间窗合计约 22000 帧，**剩下
~21000 帧（48%）落在"无 rect"区间**，仍被重新解码、重新编码、画面零变化。

### 设计原则

- **mosaic 段内部进一步切成"重编子段 + 透传子段"**，透传子段不重编。
- **复用 Stage 4 已有的 fast HEVC merge 链路**：透传段 stream-copy 自源、
  重编段走 paste；末尾用 concat 拼回 mosaic 段的最终 mp4。
- **与 Stage 4 不冲突**：Stage 4 处理 mosaic vs gap 大粒度；P8 处理 mosaic 段
  内部的 sub-gap。

### 数据流

#### 1. 子段切分

paste 入口处对 paste_segments 求并集后取补集得到"无 rect 帧区间"：

```
active_intervals = merge_intervals([(seg.base_frame_start, seg.base_frame_end)
                                    for seg in paste_segments])
inactive_intervals = invert(active_intervals, 0, total_frames)
```

active intervals 之间的间隙就是潜在 sub-gap。

#### 2. keyframe 对齐

base clip 的 keyframe 列表已经由 `list_keyframes` 提供。对每个 sub-gap：

- 起点向后对齐到第一个 ≥ 起点的 keyframe
- 终点向前对齐到最后一个 ≤ 终点的 keyframe
- 对齐后区间长度 < `min_passthrough_frames`（默认 60 帧 ≈ 1s）就放弃透传，
  并入相邻重编区间

剩下的就是合法透传段。其余帧仍走重编。

#### 3. 子段分类与命名

```
SubSegment = {
    "kind": "paste" | "passthrough",
    "start_frame": int,
    "end_frame": int,
    "out_path": str,   # 各自独立的临时 mp4 / .hevc
}
```

命名用 `<base_stem>.paste.<start_frame>-<end_frame>.mp4` 与
`<base_stem>.passthrough.<start_frame>-<end_frame>.mp4`，避免冲突。

#### 4. 透传段输出

走现有 `cut_segment` / `cut_source_by_intervals` 的 stream-copy 模式：

```
ffmpeg -ss <start_s> -i base_clip -t <duration> -map 0:v:0 \
       -c:v copy -an -avoid_negative_ts make_zero <passthrough_subseg.mp4>
```

#### 5. 重编段输出

把现有 `paste_segments_gpu` 抽出"对指定帧区间 N..M 做 paste"的内层函数：

```
paste_segments_gpu_range(
    base_path, dst, segments,
    *, start_frame: int, end_frame: int,
    cq, bitrate_bps, keep_audio=False, ...
)
```

- segments 还是同一份，但循环 `for i in range(start_frame, end_frame)`
- active rect 激活/失活逻辑不变
- 输出 raw HEVC → mux 成对应的 `.paste.<start>-<end>.mp4`

#### 6. 拼接

把 sub-segment 按 start_frame 排序，用现有 `concat_timeline_hevc_fast`
（已有 concat demuxer 直读 + annex-B 兜底）拼接成最终 `<seg>.restored.mp4`。

### 边界

- **HEVC 流参数一致性**：重编 NVENC 的 profile / tier / colour metadata 必须
  与 base clip 完全一致，否则 concat 会引发播放器卡顿。已有 `_encoder_kwargs`
  从 `src_meta` 拿这些参数，确保重编段照搬。tier 字段如不在 `_encoder_kwargs`
  内，加上去。
- **keyframe 对齐导致透传区间向两侧膨胀**：实际节省 < 估算（48% → 实际可能
  30–35%）。仍然显著。
- **`max_passthrough_count`**：单段透传子段数过多（如 > 50）会让 concat 时
  的 entry 列表巨长，影响 ffmpeg 启动开销。配置项控制，超过阈值就退化为
  整段 paste（保守）。
- **配置项**：
  - `paste_passthrough_enabled`（默认 True）
  - `paste_passthrough_min_frames`（默认 60）
  - `paste_passthrough_max_subseg`（默认 32）

### 失败处理

- 任一子段失败 → 清理已生成的所有 sub-segment 临时文件，回退到"全段 paste"
  老路径（不再尝试切分）。这是最稳的兜底，且因为 frame-key cache 已经把
  extract / restore 段缓存了，回退只多一次 paste。
- cancel 在子段循环中检查；任一 sub-segment 抛 OperationCancelled 即向上传播。

### 测试

- 单测 A：构造 3 个 active rect 时间窗 + 4 个 sub-gap，断言切分结果 entry
  数 == 7（3 paste + 4 passthrough），无重叠无漏帧。
- 单测 B：keyframe 列表稀疏到只有 [0, 1000]，sub-gap [100, 900] 对齐后
  长度 800 仍合法，断言生成的透传段时间 == [keyframe(0), keyframe(<=900)]。
- 单测 C：sub-gap 短于 min_passthrough_frames，断言并入相邻 paste 区间。
- 单测 D：max_subseg 超限，断言整段退化为原路径。
- 回归：跑实测段 2，断言最终 mp4 帧数、duration、color metadata 与改前完全
  一致；md5 不要求一致（重编位置略变），但 PSNR > 50 dB 即视为通过。

### 工时

- 子段切分 + keyframe 对齐：0.5 天
- `paste_segments_gpu_range` 抽出 + 测试：0.5 天
- 透传段 cut + concat 集成：0.5 天
- 单测 + 回归 + 兼容性验证：1 天

合计 ~2.5 天。

---

## Backlog

### P4a：fine 全段无命中 stride 翻倍复验缓存

适用场景：用户多次跑同一素材（调参 / 重做），fine 全段无命中的段下次以
2× stride 复验，命中复现就 finalize"段内无 fine 命中"，否则回原 stride。

实现要点：

- cache key = `(source mtime, interval [start,end], coarse_conf, fine_conf,
  stride, detector model file mtime + size)` 的稳定 hash
- cache 文件落到 tmp 目录，内容只有 `{"empty": true}`
- scan 入口先查 cache，命中跳过 fine scan 直接返回空段；未命中正常 scan，
  scan 后若 segments=[] 则写 cache

工时 ~0.3 天。仅对反复跑同素材的场景有效，单次运行无收益。

### P4b：连续无命中早停 + 跨步跳进

实施要点：

- fine scan 帧循环里维护 `consecutive_miss` 计数
- 命中（任意 box 通过 conf 阈值）→ 清零；连续 N=30 次未命中（15s @stride 0.5s）
  → 切到"扫描模式 B"：stride 改 8× 原值（4s）
- 模式 B 命中 → 回到附近"4s 窗口"用原 stride 重扫一遍补全
- 命中后切回扫描模式 A

风险：可能错过极短马赛克片段（< 4s 且 conf 在阈值线上）。建议配置项默认关闭，
让风险偏好用户启用。

工时 ~0.5 天，含 A/B 数据采集。

### P10：粗扫 conf / 段最短时长 A/B 微调

需 A/B 数据驱动：

- 准备 5+ 部不同类型素材（有马赛克 / 无马赛克 / 边缘模糊 / 噪点多）
- 当前默认 `pre_extract_coarse_yolo_conf=0.20` + `pre_extract_min_segment_s=1.5`
- 跑两组对比："0.20 / 1.5" vs "0.25 / 2.0"
- 指标：每段 false positive 数（粗扫被纳入但 fine 后 0 命中的段数）、
  漏检数（粗扫漏掉但人工验证有马赛克的段）、整体 wallclock
- 数据明显倾向新组就升级默认值；否则维持

工时：A/B 跑数据 + 决策 ~半天，落地 5 分钟。

---

## 总体落地建议

按下面顺序落，每批落完跑同一组真实素材回归对比 wallclock + 输出 md5（gap 段
必须完全一致、mosaic 段帧数 / duration / 颜色 metadata 完全一致）。

| 批次 | 项目 | 累计预估收益（基于本次实测） |
|---|---|---|
| 1 | P3 | 主路径 -3~5 min（每个时间窗少一次 8K NVDEC） |
| 2 | P7 | -1~2 min/段 + 临时空间减半 |
| 3 | P9 | 主路径 -10~20% wallclock（碎段越多越大） |
| 4 | P8 | mosaic 段 paste -30~40%（本次段 2 paste 21' → 13' 估） |

P3 + P7 + P9 + P8 全做下来，本次 4h03m 的运行估计能压到 **3h 左右**
（约 25% 减少），且 wallclock 收益对长素材、马赛克密度高的素材线性放大。

backlog 按需做。
