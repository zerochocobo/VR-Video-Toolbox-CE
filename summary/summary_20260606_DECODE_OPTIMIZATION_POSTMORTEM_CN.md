# OneClick Decode 优化两次误诊复盘

承接 [summary_20260606_ONECLICK_DECODE_STAGE2_TUNING_PLAN_CN.md](summary_20260606_ONECLICK_DECODE_STAGE2_TUNING_PLAN_CN.md)。

Stage-2 (`09cf774`) 上线后真实 8K 素材 restore 阶段从 ~8 分钟回归到
13 分钟。中间两次诊断方向错误，让开发多跑了 1-2 轮真机测试。最终
hotfix `14b0244` 把 `VRVT_INFERENCE_TUNING` 默认翻到 `0`，恢复到 Stage-1
单独生效时的速度（restore 4K SBS clip 跑到 40 fps 稳定）。

## 真正的回归源

**`cudnn.benchmark=True`**（在 `_torch_tuning.apply_inference_tuning`
里默认开启）。BasicVSR++ 的内部特征图形状随以下三个变量变化：

1. **clip_length**：默认 180，文件末尾余数 clip 长度不固定；
2. **mosaic crop padding**：每个 mosaic box 的 aspect ratio 不同，
   `pad_image` 到 256×256 后内部 feature map 大小仍随 deform_align /
   SPyNet 金字塔层数浮动；
3. **channels_last fallback 状态**：闩位前后两种 stride。

每见到新 shape，cuDNN 在 benchmark 模式下都要跑一次 autotune（每形状
几百 ms），accumulate 到分钟级。**这是教科书里"cudnn.benchmark 适用于
静态形状 CNN"的反例**：BasicVSR++ 视频超分网络只对外暴露固定 batch，
对 cuDNN 而言每层卷积的工作空间都是动态的。

撤掉这一个 flag 之后实测 fps 从 10–30 (不稳定) 直接拉到稳定 40，回到
Stage-1 单独生效时的水平。

## 误诊回顾

### 误诊 #1：HDR / 10-bit 是"一半视频"的瓶颈

**当时判断**：用户说"一半视频走 PyAV"，我直接列了 HDR PQ / HLG /
BT.2020 路由计划，估算"收益巨大"。

**实际事实**：用户给了 `videos/2.mp4` 做样本，ffprobe 显示这是
**10-bit SDR HEVC bt709 Main10**——`probe.decide_backend` 已经判给
`gpu_p016`，Stage-1 主路径就在用 NVDEC P010 解码。所谓"一半视频"的
HDR 命中率从未被验证过。

**错在哪**：基于"内容里很多 8K HDR"的脑补假设直接产计划，没让开发先
扫一遍 `videos/` 实际分布。等用户主动指出 `videos/2.mp4` 是已 GPU 路由
的 10-bit SDR 后才发现假设错了。

**教训**：**任何"优化 N% 的视频"的提案，先用 5 秒 ffprobe 扫描得到实际
命中率，再决定要不要做**。这套扫描脚本应该是 oneclick 仓库的常驻
工具，不是每次都靠脑补。

### 误诊 #2：Stage-1 GPU frame source 在小分辨率上反而慢

**当时判断**：用户报告 13 分钟回归，给了 L (1376×800) / R (1360×816)
restore 阶段的 fps 曲线（10–30 fps，末尾掉到 4–7 fps）。我推断
NVDEC 在小分辨率上 setup 开销大于 PyAV 收益，提议加尺寸阈值
`VRVT_GPU_FRAME_SOURCE_MIN_PIXELS=2_000_000` 让小输入回退 PyAV。

**实际事实**：开发按建议改了之后，1376×800 走 PyAV 的速度直接掉到
**4–5 fps**（比 GPU passthrough 的 10–30 fps 还慢得多）。Stage-1 的
GPU passthrough 在这个分辨率上本来就是赢的，我把方向搞反了。

**错在哪**：
1. 用户的"原本 8 分钟基线"我默认理解成 pre-Stage-1，**没问清是
   Stage-1 之后还是之前**。
2. "NVDEC 在小分辨率上比 CPU 慢" 是从其他项目带过来的直觉，没在本
   仓库验证过。
3. 没让开发先开 `--profile-decode` 拿单项耗时拆解（`decode.frame_at`,
   `decode.nv12_to_bgr`, `model.forward`），就靠 wall-time fps 推断
   瓶颈位置。

**教训**：
- **诊断回归先确认基线时间点**，"什么时候是快的"必须问清楚到 commit
  级别。用户在 hotfix 之前已经留过 `dd71d9a` (Stage-1) 落地的 log，
  对比 base 应当从那条线起步而不是更早。
- **DecodeProfile 不能只是基础设施**，它在 Stage-1 就已经铺好，但两次
  诊断都没用它。每次回归的第一步应当是"开发跑一遍 `--profile-decode`
  把 JSON 贴上来"。
- **不要把"经典知识"当本仓库结论**，cudnn.benchmark / NVDEC 大小阈值
  / channels_last 这些都是要在本机本模型本工作负载上单独证明的。

## 通用结论

1. **新优化项默认 opt-in，不是 opt-out**。Stage-2 一开始就把
   `VRVT_INFERENCE_TUNING / VRVT_CUDA_GRAPH / VRVT_CHANNELS_LAST`
   默认全开，hotfix 一轮才意识到这种"安全开关"在动态形状模型上是
   反向收益。今后**任何新加的优化项先 default 0**，单独 A/B 出 ≥10%
   稳定提升才翻默认。
2. **DecodeProfile 是诊断流程的必经环节**。已经写好的工具应当成为
   reflex：任何回归、任何"为什么慢"，第一条响应应当是开发跑
   `VRVT_PROFILE_DECODE=1` 把 JSON 发出来，不要等模型给方向。
3. **基线必须明确到 commit**。在 PR/discussion 里说"原本 8 分钟"等于
   没说，必须是"`dd71d9a` 8 min / `09cf774` 13 min"。这次教训直接
   对应了两个 hotfix 提交 (`e09c798`, `14b0244`)。
4. **不要在用户已经一次失误后继续猜**。误诊 #2 我应当在写"加尺寸阈值"
   之前先要求 profile JSON，结果反而又错一次。**连续两次误诊的成本
   远大于让开发跑一次 profile**。

## 后续动作（不阻塞当前修复）

- 跑一遍真实素材库的 ffprobe 扫描，统计 `probe.decide_backend` 命中
  分布。如果 HDR / BT.2020 / VFR / 非 HEVC 累计 <10%，HDR 优化整个
  计划归档；≥20% 才考虑动。
- `cudnn.benchmark = True` 单独 A/B：现在默认关，开 `VRVT_INFERENCE_
  TUNING=1` 跑同 segment，确认它就是 13 分钟回归的唯一来源；如果是，
  Stage-2 计划里 cudnn.benchmark 这一项**直接划掉**，不要写"等 shape
  稳定再开"——BasicVSR++ shape 不会稳定。
- CUDA Graph (A1) 的方向重新评估：当前 forward 已经是稳定 40 fps，
  graph 化预期增量收益变小，要先量化才决定是否继续。channels_last 同理。
- TRT 阶段开工前，先在 `summary/` 写一份"DecodeProfile 基线"，记录
  `dd71d9a + 14b0244` 状态下各 section 的 cuda_ms。后续每一次"我觉得
  这里能优化"必须配对这张基线表说话。
