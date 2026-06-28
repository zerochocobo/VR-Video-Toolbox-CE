# Crop/Paste 优化研究过程总结

日期：2026-06-25

## 目的

本报告整理 VR Video Toolbox NE 的 OneClick pre-extract crop/paste 路径、基准测试、已尝试优化、已回退方案和后续可行方向，供外部专家评估进一步优化方案。

测试重点是 8K SBS 视频中的局部马赛克区域处理：

- 源文件：`videos/SI_TEST_2.mp4`
- 源规格：8192x4096，10-bit HEVC，约 59.94 FPS
- 测试区间：`00:01:00` 到 `00:05:00`，短测使用 `00:01:00` 到 `00:01:02`
- 测试 rect：左眼局部坐标 `1024,1024,1024,1024`
- GPU：NVIDIA GeForce RTX 5060 Ti 16GB

## 当前实现路径

### Crop

OneClick paired pre-extract 非鱼眼路径中，crop 由以下函数完成：

- `gpu_engine.files.extract_multi_rect_clip()`
- `gpu_engine.files.extract_transformed_rect_clip()`

当前流程：

1. `PyNvThreadedSerialDecoder` 用 NVDEC 解码 base clip 到 GPU NV12/P016 平面。
2. 用 CuPy 做 eye slice 和 rect slice。
3. 鱼眼模式下会额外做 GPU hequirect -> fisheye remap。
4. 用 `PyNvEncoderSession` 通过 NVENC 编码 rect clip。

结论：在正确模拟 OneClick Stage 3 的 base clip 输入后，crop 本身不是主要瓶颈。

### Paste

OneClick paired pre-extract 非鱼眼路径中，paste 由以下函数完成：

- `utils.segment_paster.paste_segments_gpu_or_fallback()`
- 优先调用 `gpu_engine.files.paste_segments_gpu()`
- 失败后才回退 ffmpeg overlay

当前流程：

1. 解码 base clip 全帧。
2. 解码 restored rect clip。
3. 在 CuPy Y/UV 平面上按 rect 和 feather alpha 贴回。
4. 对每个受影响帧重新 NVENC 编码完整 8192x4096 帧。

鱼眼 paste 是单独路径：

- `gpu_engine.files.paste_fisheye_eye_rects_to_sbs_gpu()`
- 每眼需要单独 hequirect/fisheye 转换，不能把左右眼合并做。

## Baseline 工具

新增脚本：

```powershell
scripts\bench_oneclick_crop_paste.py
```

默认命令：

```powershell
.\.venv\Scripts\python.exe scripts\bench_oneclick_crop_paste.py --start 00:01:00 --end 00:05:00 --rect 1024,1024,1024,1024 --crop-mode left
```

默认模式：

- `--base-mode preclip-copy`
  - 先用 stream copy 生成临时 `base_preclip.mp4`，模拟 OneClick source-scan Stage 2 后交给 Stage 3 的形态。
  - 再在 base clip 的相对时间上测 crop/paste。
- `--bitrate-mode oneclick`
  - crop 使用 OneClick intermediate 码率策略。
  - paste 使用 OneClick final 码率策略。

注意：

- 曾直接从原片 `60s-62s` 非零位置测 crop，出现 `233.2s` 总耗时，但实际帧循环只用了几秒。
- 该耗时来自原片首次非零 start 的 decoder keyframe discovery/setup，不是 crop/paste 计算本身。
- 因此默认 benchmark 使用 `preclip-copy`，避免误判 Stage 3 性能。

## 基准数据

短测命令：

```powershell
.\.venv\Scripts\python.exe scripts\bench_oneclick_crop_paste.py --start 00:01:00 --end 00:01:02 --rect 1024,1024,1024,1024 --crop-mode left
```

初始 baseline，输出：`debug_output/crop_paste_baseline/20260625_194834/metrics.json`

| 阶段 | 耗时 | 吞吐 |
| --- | ---: | ---: |
| preclip copy | 0.072s | N/A |
| crop | 1.234s | 97.2 FPS |
| paste | 4.388s | 27.4 FPS |
| crop+paste 顺序 | 5.622s | 21.3 FPS |

在只保留内存拷贝优化后，输出：`debug_output/crop_paste_baseline/20260625_200512/metrics.json`

| 阶段 | 耗时 | 吞吐 |
| --- | ---: | ---: |
| crop | 1.231s | 97.5 FPS |
| paste | 4.364s | 27.5 FPS |
| crop+paste 顺序 | 5.594s | 21.5 FPS |

结论：

- crop 约 97 FPS，不是主要瓶颈。
- paste 约 27 FPS，明显慢。
- paste 慢的根因不是小 rect alpha 混合，而是每个命中帧都要重新编码完整 8K 10-bit 帧。

## 已尝试优化

### 1. 减少 paste 中一轮整帧 GPU 拷贝

原路径在 `paste_segments_gpu()` 中每帧做了两轮整帧复制：

1. `cp.ascontiguousarray(base_y_src/base_uv_src)` 复制出可写 Y/UV。
2. `_pack_planes()` 再复制成 NVENC packed buffer。

修改后：

- 新增 `_copy_planes_to_packed_views()`。
- 直接把 base Y/UV copy 到 encoder packed buffer。
- 在 packed buffer 的 Y/UV view 上贴 rect。
- NVENC 输入 buffer 生命周期仍由 `_EncodeSink` ring 保持。

结果：

- paste 约 `27.35 FPS -> 27.50 FPS`
- 收益很小，说明整帧 CuPy copy 不是主瓶颈。
- 该修改不改变编码参数，不应引入画质下降，目前保留。

### 2. NVENC preset 实验（已回退）

对同一 8K 120 帧做 encode-only 测试：

| NVENC preset | encode-only 吞吐 |
| --- | ---: |
| P1 | 46.9 FPS |
| P2 | 48.7 FPS |
| P4 | 28.9 FPS |
| P7 | 11.2 FPS |

同时测试 `tuning_info`：

| tuning_info | P4 吞吐 |
| --- | ---: |
| high_quality | 28.3 FPS |
| low_latency | 28.5 FPS |
| ultra_low_latency | 25.4 FPS |

曾尝试给 paste 单独使用 P2：

- paste 从 `4.388s / 27.4 FPS` 提升到 `2.735s / 43.9 FPS`
- 但用户指出 P2 会导致局部画质变差
- 因此该方案已回退
- 当前 paste 仍跟随全局 `gpu_encode_preset`，不强制 P2

## 关键定位结论

1. Crop 性能已经较高，短测约 97 FPS。
2. Paste 性能接近 8K full-frame encode-only 上限。
3. 小 rect 贴回计算本身只带来约 4% 额外开销：
   - encode-only：约 28.6 FPS
   - paste：约 27.4 FPS
4. 在保持全局高质量 NVENC preset 的前提下，paste 很难通过 rect kernel 本身大幅提速。
5. 对任何包含 rect patch 的帧，当前视频编码模型都需要重新编码完整帧，不能只修改压缩码流中的局部矩形。

## 外部专家可重点评估的方向

### A. 减少需要重编码的帧数/时间段

这是最可能带来实际收益的方向。

可研究点：

- 改进 detection/segment 聚合，避免误检导致长时间段 paste。
- 更严格地按 scene cut、mask active window 切分 fine segments。
- 对无 patch 的 gap 使用 stream copy passthrough。
- 尽量使 paste subsegments 更短、更贴近真实马赛克存在时间。

当前已有 passthrough 逻辑，但仍可评估：

- keyframe 对齐导致的时间段膨胀是否过大。
- 是否可以插入 keyframes 或用更细粒度 GOP 策略降低重编码窗口。
- 当前 `paste_passthrough_min_frames` / `paste_passthrough_max_subseg` 是否保守。

### B. Pipeline overlap / 异步化

当前 paste 每帧涉及：

- base decode
- restored rect decode
- CuPy copy/patch
- NVENC encode
- `_EncodeSink.feed()` 前的 device synchronize

可研究点：

- 用 CUDA events 替代 device-wide synchronize。
- base decode、rect decode、patch、encode 是否能用多 stream 或双缓冲重叠。
- PyNvVideoCodec encoder 输入 buffer 生命周期是否能通过 event/ring 更精细管理。
- 现有同步是否为了避免 NVENC 读取未完成 buffer 而过度保守。

风险：

- 之前已有经验表明 NVENC Encode 返回后不代表已读完输入 buffer，过早释放/复用会产生绿色块/闪块。
- PyNv ThreadedDecoder 与多 decoder 并发可能有 seek/state 污染风险。

### C. 保持画质的 NVENC 参数探索

P1/P2 虽快，但画质风险不可接受。

可研究点：

- 在保持 P4/P7 或用户选择 preset 的前提下，是否存在不明显降低局部质量的参数组合。
- 码率、maxrate、AQ、lookahead、rc mode、GOP、B-frame 是否可以提升速度或稳定质量。
- 当前 `bf=0`、`gop=30`、`rc=vbr`、`tuning_info=high_quality` 是否最合适。

注意：

- 已测 `tuning_info` 对 P4 几乎无速度收益。
- 任何降低 preset 的方案都需要画质回归，不宜默认开启。

### D. 避免重新编码整帧的可行性

理论上局部 patch 后的视频帧已经改变。对于 inter-frame HEVC，局部像素改变会影响预测链，通常无法只替换压缩码流局部矩形。

可研究但风险很高：

- Tile/slice 级编码或 region-of-interest 编码是否能让局部区域独立更新。
- 源视频和输出编码器是否能统一 tile layout。
- 播放器兼容性和码流合法性。

初步判断：

- 作为通用 MP4/HEVC 输出方案，局部码流 patch 很可能不可行或复杂度极高。

### E. Crop 侧可研究方向

当前 crop 性能较好，但仍可研究：

- 多 rect 同时间窗已用 `extract_multi_rect_clip()` 共享一次 base decode。
- 可以评估 raw HEVC rect clip + sidecar，减少中间 mp4 mux 开销。
- 可以评估 crop 结果是否必须立即 mux 成 mp4，或是否能直接传给 restoration。
- 对 fisheye 模式，heq->fisheye remap 是额外成本，左右眼不能合并处理。

## 实验补充（2026-06-25 晚）：paste preset + AQ 的画质量化

为验证"P2 局部画质变差"是否成立，给 `bench_oneclick_crop_paste.py` 增加了:

- `--paste-preset` / `--paste-aq` / `--paste-temporal-aq` / `--paste-enc-extra`：只覆盖 paste 阶段 NVENC 参数（通过临时 monkeypatch `files._encoder_kwargs`，不动生产代码）。
- `--measure-quality`：解码 paste 输出与 restored 裁剪片，对 **patch rect 内部**（排除 feather 边）算 Y/U/V PSNR，量化"修复区被编码器损伤多少"。

同一 8192x4096 10bit、120 帧、patch=左眼 1024²、复用同一 restored 参考：

| 变体 | paste FPS | 相对 P4 速度 | rect Y-PSNR |
| --- | ---: | ---: | ---: |
| P4（当前 config 默认） | 27.6 | — | 65.10 dB |
| P2 | 44.5 | +61% | 65.21 dB |
| P2 + aq=1 | 44.3 | +60% | 66.36 dB |
| P1 + aq=1 | 44.6 | +61% | 66.36 dB |
| P7 + aq=1 | 11.0 | -60% | 67.32 dB |

把目标码率强压到 6 Mbps（RC 紧张）复测:

| 变体 | paste FPS | rect Y-PSNR |
| --- | ---: | ---: |
| P4 @ 6M | 27.7 | 59.89 dB |
| P2 @ 6M | 44.9 | 59.79 dB |
| P2+AQ @ 6M | 44.3 | 59.81 dB |

结论:

1. **"P2 修复区画质变差"在 patch-region PSNR 上无法复现**——任何码率下 P2 与 P4 相差 ≤0.1 dB。
2. P2 是 paste 阶段约 **+60% 的免费提速**（27.6 → 44.5 FPS）。
3. 高码率下 spatial AQ 是纯增益(+1.15 dB,0 速度成本);低码率下整帧统一缺码率,AQ 帮不上。
4. P1=P2(NVENC 截断);P7 慢 4 倍只换 +1 dB,不划算。
5. **`P2 + aq=1` 在 paste 阶段全面优于当前 P4 默认。**

### 以 P7 为锚的 preset 曲线（VR 8K 用户实际多选 P7）

很多人处理 VR 8K 会直接选 P7。以 **P7(无 AQ)** 为参考重测同一参考片:

| 变体 | paste FPS | 相对 P7 速度 | rect Y-PSNR | Δ vs P7 |
| --- | ---: | ---: | ---: | ---: |
| P7 无 AQ(今日 VR 默认) | 10.9 | 1.0× | 65.86 dB | — |
| P7 + aq | 10.9 | 1.0× | 67.32 dB | +1.46 |
| P6 + aq | 11.6 | 1.1× | 67.23 dB | +1.37 |
| P5 + aq | 27.5 | 2.5× | 66.28 dB | +0.42 |
| P4 + aq | 27.5 | 2.5× | 66.27 dB | +0.41 |
| P3 + aq | 41.9 | 3.8× | 66.36 dB | +0.50 |
| P2 + aq | 44.3 | 4.1× | 66.36 dB | +0.50 |

关键结论(比 preset 本身更重要):

1. **真正的杠杆是 AQ，不是 preset。** 仅给 P7 打开 aq=1 就是 **patch +1.46 dB、零时间成本**——今日 P7 用户白白丢了这个增益。
2. **AQ 打开后,P2–P5 任意 preset 的 patch 质量都已超过今日 P7(无 AQ)**(+0.4~0.5 dB),且快 2.5–4×。即 `P2+aq`/`P3+aq` 对 VR 8K 是对 P7 的**严格 Pareto 升级**(更好 + 11→44 FPS)。
3. **速度悬崖在 P5→P6**(27→11 FPS),只换 +0.9 dB。P6/P7 仅在必须要那最后 ~0.9 dB 时才值,且那时应跑 **P7+aq** 而非裸 P7。
4. "P2 损伤 patch"依然无法复现——P2+aq 反而超过 P7 无 AQ。

→ 对"用户多选 P7"的回答:开 AQ 后给他们 **P3+aq**(4× 速度)patch 质量已优于今日 P7;或保留 P7 但**至少把 AQ 打开**换免费 +1.46 dB。两条路都指向 **AQ 必开**。

### P1 + 双 AQ：最快且最高质量(本片)

| 变体 @ P1 | paste FPS | 相对 P7 速度 | rect Y-PSNR | Δ vs P7 无AQ |
| --- | ---: | ---: | ---: | ---: |
| P1 无 AQ | 44.5 | 4.1× | 65.21 dB | -0.65 |
| P1 + aq | 45.3 | 4.2× | 66.36 dB | +0.50 |
| **P1 + aq + temporalaq** | **45.6** | **4.2×** | **67.67 dB** | **+1.81** |

- `P1 + aq=1 + temporalaq=1` 在本片上**同时是最快和 patch PSNR 最高**的组合,67.67 dB **超过 P7+aq(67.32)**,且 4.2× 速度。Pareto 碾压包括 P6/P7 在内的所有变体。
- 印证杠杆排序:高码率下 patch 不缺码率,P1 的快速 mode-decision 对 rect 几乎零损失;AQ 决定码率去哪。spatial AQ(+1.15)把码率塞进高细节 patch,temporal AQ(再 +1.31)把码率集中到会传播整个 GOP 的参考帧。
- **特别警告:temporalaq 恰恰是平均 PSNR 最看不出问题的设置**——它最可能引入时域 pumping/闪烁。本片偏静态(固定坐标 rect)是 temporal AQ 的最佳场景;高运动马赛克片上行为可能不同。**三项里 temporalaq 最需要在真实运动片上做视觉 A/B 再上。**

### 实验 1+2+3：背景 PSNR / B 帧 / multipass·lookahead

给 bench 再加"整帧 + 背景(排除 rect)Y-PSNR"(paste 输出 vs base 输入),并扫 bf / multipass / lookahead。

**30 Mbps:**

| 变体 | FPS | 相对 P7 | rect Y | 背景 Y |
| --- | ---: | ---: | ---: | ---: |
| P7 无 AQ(今日默认) | 10.9 | 1.0× | 65.86 | 65.67 |
| P1+aq+tAQ | 46.0 | 4.2× | 67.67 | 66.69 |
| P1+aq + multipass=qres | 38.7 | 3.5× | 71.69 | 70.01 |
| P1+aq + multipass=fullres | 31.4 | 2.9× | 72.20 | 70.75 |
| **P1+aq+tAQ + multipass=fullres** | 31.4 | 2.9× | **74.12** | **73.17** |
| …+ lookahead=8 | 28.9 | — | 74.12 | 73.17(无增益) |
| P1+aq + bf=3 | 41.2 | — | 65.13 ⬇ | 64.20 ⬇ |

**6 Mbps(RC 紧张):**

| 变体 | FPS | rect Y | 背景 Y |
| --- | ---: | ---: | ---: |
| P1+aq | 46.9 | 59.81 | 58.97 |
| P1+aq + lookahead=8 | 40.6 | 61.66 | 61.13 |
| P1+aq + multipass=fullres | 31.7 | 63.73 | 62.63 |
| P1+aq+tAQ + multipass=fullres+LA8 | 28.8 | 64.35 | 64.35 |

结论:

1. **`multipass=fullres`(两遍 RC)是整个研究里最大的单一杠杆**——30M +6 dB、6M +4 dB,patch 和背景**同时**涨,仅约 30% 速度成本,仍比今日 P7 快 2.9×。
2. **temporalaq 可叠加**(+1.9 dB,几乎免费);**multipass 开了之后 lookahead 冗余**(无增益还略慢);**B 帧反而掉质,禁用**。
3. **实验 1 推翻"AQ 抢背景码率"的担忧**:背景 PSNR 与 patch **一起涨**(真正杠杆是全局两遍码率分配,不是零和);背景甚至比今日 P7 还高。
4. **新冠军:`P1 + aq + temporalaq + multipass=fullres`。** 对比今日 P7:paste **10.9 → 31.4 FPS(快 2.9×)**,patch **+8.3 dB**,背景 **+7.5 dB**。每个指标都更快更好,Pareto 碾压。
   - 极速档可用 `multipass=qres`(38.7 FPS,3.5× P7,patch 仍 +5.8 dB)。
   - 注意:fullres 两遍会让 8K NVENC 跑一遍全分辨率分析,显存占用更高(16GB 5060 Ti 8K 实测无压力)。

注意/未决:

- PSNR 已高到 72–74 dB(对修复输入近乎数学无损),此区间差异不可感知;但**相对排序**(multipass≫单遍、tAQ 有用、bf 掉质、LA 冗余)在 30M/6M 两个码率一致且稳健,6M 段(58→64 dB 真实失真)证明 multipass 是真实 RC 改进而非测量假象。
- 以上是单条 2s 片、单个静态 rect 的**平均 PSNR**;PSNR 可能漏掉运动块效应/banding/时域闪烁。
- 之前回退 P2 基于**肉眼**观察,因此切生产默认前应在真实马赛克片上做**视觉 A/B**(或上 VMAF)。
- 本实验只测 patch 区,未测被整帧重编码的背景(那 94% 无论如何都在重编,属 generation loss,与 preset 选择正交)。

## 当前代码状态

保留：

- `scripts/bench_oneclick_crop_paste.py`(已加 `--paste-preset/--paste-aq/--paste-temporal-aq/--paste-enc-extra` 覆盖,及 patch-region + 整帧/背景 PSNR 测量)
- `gpu_engine.files._copy_planes_to_packed_views()`
- `paste_segments_gpu()` 使用 packed buffer view 贴回，减少一轮整帧中间拷贝

已接入生产(`_encoder_kwargs`,crop/paste/fisheye 共用,均可 config 覆盖):

- `gpu_encode_aq`(默认 on)→ `aq=1`
- `gpu_encode_multipass`(默认 `fullres`;可设 `qres` 省时,或 `off` 关闭)→ 两遍 RC
- `gpu_encode_temporal_aq`(默认 **off**,待运动片视觉验证)→ `temporalaq=1`
- `gpu_encode_preset`(未改,仍由用户选);`bf` 仍为 0(实验证明开 B 帧掉质)
- 同 preset 下 multipass 约 +7 dB 但慢约 30%(P4:27.6→21.2 FPS);要同时拿速度需把 preset 调到 P1/P2。
- 三个新开关已加入 `utils.app_config._DEFAULTS`,可通过配置覆盖。
- 编码日志已打印 `multipass=... aq=... temporalaq=...`;无覆盖短测已确认输出
  `multipass=fullres aq=1 temporalaq=0`。

已回退：

- paste 专用 P2 preset
- `paste_encode_preset` 配置项
- `_apply_paste_encoder_preset()`

验证：

```powershell
.\.venv\Scripts\python.exe -m py_compile gpu_engine\files.py utils\app_config.py scripts\bench_oneclick_crop_paste.py
.\.venv\Scripts\python.exe -m pytest tests\test_one_click_pre_extract.py tests\test_source_time_scanner.py -q
```

结果：

- `.\.venv\Scripts\python.exe -m py_compile gpu_engine\files.py utils\app_config.py scripts\bench_oneclick_crop_paste.py tests\test_app_config.py`
- `.\.venv\Scripts\python.exe -m pytest tests\test_app_config.py tests\test_one_click_pre_extract.py tests\test_source_time_scanner.py -q`
- `43 passed`
