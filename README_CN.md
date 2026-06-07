# VR视频工具箱(CUDA专版)（[English](README.md) | [日本語](README_JP.md)）

面向 VR 视频整理、修复和字幕处理的一组 Windows 工具。

原项目 https://codeberg.org/zelefans/vr_remove_mosaic 只简单利用ffmpeg的cuda硬件加速，大量变换操作还是不得不与CPU交换。

本版本针对 **NVIDIA CUDA** 程序性优化：支持的流程会优先使用 NVIDIA GPU 完成解码、几何变换、AI 处理和编码，不支持的素材或运行环境会自动回退到 FFmpeg 路径。

软件首页：https://github.com/zerochocobo/VR-Video-Toolbox-CE

当前主要功能：

- 马赛克去除
- 字幕生成、翻译、嵌入
- **轻量级局域网 VR 视频 DLNA 服务器**（支持 180° SBS 格式自适应诱导、外部字幕自动关联与多物理目录映射）
- VR 视频拆分、合并、投影转换等辅助工具

历史更新记录见 [CHANGELOG.md](CHANGELOG.md)。

本项目尽量把复杂的视频处理流程做成图形界面和批处理流程，适合不想手写命令的用户使用。

## 适合谁使用

- 想批量处理 VR 视频的用户
- 想给 VR 视频生成字幕、翻译字幕或嵌入字幕的用户
- 想要在 VR 头显（如 Quest/Pico）中用 Skybox 等播放器直接无线播放电脑本地视频并自动关联字幕的用户
- 想尝试用 AI 工具去除视频马赛克的用户
- 想拆分左右眼、合并视频、转换 VR 投影格式的用户

请只处理你有权处理的视频内容，并遵守所在地法律法规。

![VR视频工具箱(CUDA专版)首页](assets/main_cn.png)

## 主要功能

### 1. 马赛克去除

提供多种处理方式，适配不同类型的 VR 视频：

- 一键模式：适合多数常见 VR 视频，操作最简单。
- 选区直接裁剪模式：适合画面中局部矩形区域的处理。
- 选区 VR 转平面模式：适合在 VR 视角中更接近方形、但原始画面边缘发生变形的马赛克。

处理效果主要受 AI 去马赛克引擎（`lada-cli` 或 `jasna-cli`）的识别和修复能力影响。对于复杂变形、遮挡严重或画质很差的视频，结果可能不稳定。

> 程序内置引擎选择器，可在主界面的「AI去马赛克引擎」中切换 **Lada** 和 **Jasna**，选择会自动记忆。
> - Lada：https://codeberg.org/ladaapp/lada
> - Jasna（Lada 的新一代维护分支）：https://github.com/Kruk2/jasna

### NVIDIA CUDA 优化（纯显卡处理流水线）

本软件针对 **NVIDIA CUDA** 做了专门优化。除 lada/jasna 去马赛克本身外，VR 投影转换、左右眼拆分/合并、VR 转平面、一键流程中的几何变换均已改为 **纯显卡处理**：PyNvVideoCodec 解码（NVDEC）→ CuPy/自定义 kernel 几何变换 → PyNvVideoCodec 编码（NVENC），ffmpeg 仅用于复用音频。相比原先的 CPU `v360` 滤镜，8K 素材端到端可达约 2.5–3× 提速，几何结果与 ffmpeg 一致（裸像素 PSNR ~62–79dB）。

- **后端选择**：配置项 `transcode_backend`（`vr_toolbox_config.json`）
  - `auto`（默认）：优先 GPU，遇不支持的源或运行时异常自动逐文件回退 ffmpeg。
  - `gpu`：强制 GPU（调试用，不回退）。
  - `ffmpeg`：强制走原 ffmpeg 路径。
- **10-bit / HDR**：10-bit bt709（HEVC Main10/P010）走 GPU 真 10-bit 保真；HDR10（PQ/smpte2084）、HLG、bt2020 宽色域自动回退 ffmpeg。
- **GPU 要求**：支持 NVDEC+NVENC HEVC 的 NVIDIA 显卡（Turing 及以后，10-bit 建议 Ampere/Ada/Blackwell）；需较新驱动。当前源码环境按 CUDA 12.8 Python wheel 对齐（`cupy-cuda12x` + `nvidia-cuda-* cu12`，通过 PTX JIT 支持 Blackwell sm_120）。
- 无可用 GPU 时整体自动降级为 ffmpeg 模式，功能不受影响（仅速度较低）。

### 2. 字幕生成、翻译、嵌入

字幕工具用于减少手工整理字幕的工作量：

- 从视频中提取语音并生成字幕
- 字幕翻译
- 批量字幕处理
- 将字幕嵌入 VR 视频
- 软字幕或硬字幕相关处理

字幕识别和翻译结果仍建议人工检查，尤其是人名、专有名词和多人对话场景。

### 3. VR 视频辅助工具

项目还包含一些常用小工具：

- VR 左右眼拆分与合并
- VR 转平面预览或输出
- VR 投影格式转换
- 视频截图、局部放大检查等辅助功能
- 批量处理脚本

### 4. VR 视频 DLNA 服务器

一个高内聚、轻量级的局域网 DLNA / UPnP 视频流媒体服务器：

- **VR 视频无线播放**：让同一局域网内的 VR 视频播放器（如 Oculus Quest 中的 Skybox VR 播放器、DeoVR、GizmoVR 等）能够无线连接并直接流畅点播电脑中的视频。
- **180° SBS 格式自适应诱导**：针对 2:1 等距柱状（Equirectangular）的半全景视频，在客户端浏览时自动将虚拟文件名映射重命名为以 `_LR_180_SBS` 结尾，完美诱导 Skybox 等播放器智能自适应渲染，免去手动繁琐调节。
- **外部字幕自动关联**：支持一键开关，自动且优先为播放端关联并排序加载同名目录下的 `.srt`/`.ass`/`.vtt` 外挂中英文字幕文件，支持中文权重优先。
- **流媒体切片 Range 播放**：核心基于 FastAPI + Uvicorn 运行，完美支持 HTTP 字节切片 Range 播放响应（206），保证 VR 终端内进度条任意拖拽播放极其丝滑。
- **多磁盘物理目录虚拟融聚**：支持在配置窗口中添加和删除多个不同的本地磁盘物理目录，DLNA 端将自动融合并整合成统一的虚拟目录层级进行展示。
- **独立服务与防火墙一键通过**：以独立隐藏的后台进程优雅拉起，并在首次启动时自动通过 UAC 申请权限打通 Windows 防火墙 TCP 8090 与 UDP 1900 规则，保证运行无阻。

## 推荐使用方式

新用户建议优先使用图形界面：

```bat
cd GUI\VR_Video_Toolbox
run.bat
```

如果 `run.bat` 无法启动，也可以使用：

```bat
cd GUI\VR_Video_Toolbox
python main.py
```

打开后，在主界面选择需要的工具：

- `One-Click Mode`：一键去马赛克
- `Area Selection Direct Crop Mode`：选区直接裁剪处理
- `Area Selection VR to Flat Mode`：VR 转平面选区处理
- **VR 视频 DLNA 服务器**：一键开启/停止局域网 DLNA 共享，提供独立的配置窗口（管理共享目录、端口及字幕关联）
- `日语批量字幕工具`：字幕生成与翻译相关工具
- `VR Hard Subtitle Embed Tool`：VR 硬字幕嵌入
- 其他按钮：VR 拆分合并、投影转换、小工具箱

## 运行环境

推荐环境：

- Windows 10/11
- NVIDIA 显卡，支持 CUDA。本 CUDA 专版针对 NVIDIA CUDA 优化，建议使用较新的 NVIDIA 驱动。
- Python 3.10 到 3.12（源码运行环境）
- FFmpeg
- AI 去马赛克引擎（二选一）：
  - **Lada CLI**：https://codeberg.org/ladaapp/lada/releases
  - **Jasna CLI**（推荐尝试）：https://github.com/Kruk2/jasna/releases

基础依赖：

- `ffmpeg.exe`
- `ffprobe.exe`
- `lada-cli.exe` 或 `jasna-cli.exe`（二选一）
- 基础 Python 包：`Pillow`、`pyinstaller`、`ffmpy3`、`faster-whisper`、`numpy>=1.26,<2.1`、`auditok`、`huggingface-hub`、`keyring`、`requests`、`av`、`fastapi`、`uvicorn`
- CUDA/视频 Python 包：`pynvvideocodec>=2.1.0`、`cupy-cuda12x>=14.0`、`nvidia-cuda-nvrtc-cu12==12.8.93`、`nvidia-cuda-runtime-cu12==12.8.90`、`nvidia-cuda-cccl-cu12>=12.9.27`
- 内置 AI/GPU 包：`torch==2.8.0` 和 `torchvision==0.23.0`（来自 PyTorch `cu128` wheel 源），以及 `ultralytics==8.4.4`、`mmengine==0.10.7`

Python 依赖安装：

```bat
cd GUI\VR_Video_Toolbox
uv sync
```

如果手动用 `pip` 安装，请以 `pyproject.toml` 中的版本为准，并从 `https://download.pytorch.org/whl/cu128` 安装 PyTorch / torchvision。

FFmpeg 和 AI 引擎（Lada / Jasna）需要能被程序找到。可以放到系统 `PATH` 中，也可以在打包版或运行目录旁边放置相关可执行文件。

## 项目目录

```text
.
├─ GUI/
│  └─ VR_Video_Toolbox/         图形界面主程序
│     ├─ one_click/             一键去马赛克
│     ├─ area_selection_rect_crop/
│     ├─ area_selection_vr2flat/
│     ├─ tool_subtitle/         字幕生成、翻译、批量处理
│     ├─ tool_subembed/         VR 字幕嵌入
│     ├─ tool_dlna/             局域网 DLNA/UPnP 视频服务器
│     ├─ tool_split_combine/    VR 拆分与合并
│     ├─ tool_v360_trans/       VR 投影转换
│     ├─ tool_vr2flat/          VR 转平面
│     └─ tools/                 小工具箱
├─ Scripts/
│  ├─ BatchFile(Windows)/       Windows 批处理脚本
│  └─ Python/                   训练、字幕等 Python 脚本
├─ Models/                      模型目录
└─ prompt/                      工作记录与交接文档
```

## 输出文件

处理后的文件通常会生成在输入视频所在目录或工具指定的输出目录中。常见命名包括：

- `_restored`：马赛克处理后的文件
- `_sbs`：左右眼并排格式
- `_L` / `_R`：左眼或右眼视频
- 字幕工具会根据任务生成 `.srt`、翻译后的字幕文件或嵌入字幕后的视频

实际命名以所选工具界面提示为准。

## 常见问题

### 没有 NVIDIA 显卡可以用吗？

部分视频和字幕流程可能可以运行，但马赛克去除依赖的 AI 处理通常需要 CUDA 环境。没有合适显卡时，速度和可用性都会受到明显影响。

### 为什么去马赛克效果不稳定？

效果取决于原视频清晰度、马赛克形态、VR 投影变形、AI 引擎（Lada / Jasna）能力和参数选择。建议先截取短片段测试，再批量处理完整视频。也可尝试切换引擎（主界面 → AI去马赛克引擎）对比效果。

### 字幕结果能直接发布吗？

不建议直接发布未校对结果。语音识别和机器翻译都可能出错，最好人工检查一遍。

### 我该选哪个去马赛克模式？

不知道选哪个时，先用一键模式测试短片段。如果效果不好，再使用选区模式。主界面中的放大检查工具可以帮助判断马赛克形态。

### 局域网 DLNA 服务器搜不到或搜到打不开怎么办？

1. **防火墙阻挡**：程序会在首次启动时自动通过 UAC 权限开启防火墙端口 TCP 8090 和 UDP 1900。如果拦截，请手动在 Windows 安全中心放行相关端口。
2. **同一局域网**：请绝对确保电脑和您的 VR 眼镜（如 Quest/Pico）连接在同一个路由器的 Wi-Fi 局域网下，且路由器未开启 AP 隔离（Access Point Isolation）功能。
3. **手动添加**：如果 SSDP 广播因为路由器多播限制搜不到，可以在 Skybox 播放器中通过“网络” -> “手动添加服务” -> 输入局域网 IP（主界面上显示的 LAN IP，如 `192.168.x.x:8090`）来进行无线连接。

## 致谢

本项目依赖 FFmpeg、LADA、Jasna、Whisper 相关工具以及社区贡献。感谢所有开源项目作者和反馈问题、提交改进的用户。
