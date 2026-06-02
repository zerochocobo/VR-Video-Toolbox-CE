# VR Video Toolbox (CUDA EDITION) ([中文](README_CN.md) | [日本語](README_JP.md))

A Windows toolkit for VR video cleanup, subtitle work, and common VR video utilities.

The original project, https://codeberg.org/zelefans/vr_remove_mosaic, only made simple use of FFmpeg CUDA hardware acceleration, while many transform operations still had to exchange data with the CPU.

This edition is programmatically optimized for **NVIDIA CUDA**. CUDA-capable workflows use NVIDIA GPU acceleration for decoding, geometry transforms, AI processing, and encoding where supported, with FFmpeg fallback for unsupported sources or runtime environments.

Homepage: https://github.com/zerochocobo/VR-Video-Toolbox-CE

Current main features:

- Mosaic removal
- Subtitle generation, translation, and embedding
- **Lightweight LAN VR Video DLNA Server** (supports 180° SBS format auto-inducing, external subtitle auto-association, and multi-root mapping)
- VR split/combine, projection conversion, and other helper tools

Version history is available in [CHANGELOG.md](CHANGELOG.md).

The goal is to make complex video workflows usable through a GUI and batch scripts, especially for users who do not want to write FFmpeg commands by hand.

## Who This Is For

- Users who batch-process VR videos
- Users who generate, translate, or embed subtitles for VR videos
- Users who want to play local PC videos wirelessly on VR headsets (e.g. Oculus Quest, Pico) with player apps like Skybox and load subtitles automatically
- Users who want to try AI-assisted mosaic removal
- Users who need left/right eye splitting, merging, projection conversion, screenshots, or preview helpers

![VR Video Toolbox (CUDA EDITION) home screen](assets/main_en.png)

## Main Features

### 1. Mosaic Removal

Several workflows are available for different video types:

- One-click mode: the simplest choice for most common VR videos.
- Area selection direct crop mode: useful for local rectangular areas.
- Area selection VR-to-flat mode: useful when the mosaic looks square in VR view but distorted near the edges of the original frame.

The final result depends heavily on the AI mosaic removal engine (`lada-cli` or `jasna-cli`) detection and restoration quality. Complex distortion, heavy compression, or low-quality source video may produce unstable results.

> The program includes a built-in engine selector. Switch between **Lada** and **Jasna** in the main window under "AI Engine". Your choice is saved automatically.
> - Lada: https://codeberg.org/ladaapp/lada
> - Jasna (a newer maintained fork of Lada): https://github.com/Kruk2/jasna

### NVIDIA CUDA Optimization

This CUDA Edition is designed for NVIDIA GPUs. Beyond the external Lada/Jasna engines, projection conversion, left/right eye split and combine, VR-to-flat conversion, and geometry transforms in one-click workflows can use a GPU-first pipeline: PyNvVideoCodec decoding (NVDEC) -> CuPy/custom CUDA kernels -> PyNvVideoCodec encoding (NVENC). FFmpeg is still used for audio muxing and as a fallback path.

- **Backend selection**: `transcode_backend` in `vr_toolbox_config.json`
  - `auto` (default): prefer GPU and automatically fall back to FFmpeg per file when the source or runtime is unsupported.
  - `gpu`: force the CUDA path for debugging.
  - `ffmpeg`: force the original FFmpeg path.
- **10-bit / HDR**: 10-bit bt709 HEVC Main10/P010 can use the GPU path; HDR10 (PQ/smpte2084), HLG, and bt2020 wide-gamut sources fall back to FFmpeg.
- **GPU requirements**: NVIDIA GPU with NVDEC and NVENC HEVC support. Turing or newer is recommended; Ampere, Ada, or Blackwell is recommended for 10-bit work.
- **Fallback behavior**: without a compatible NVIDIA GPU or CUDA runtime, supported features fall back to FFmpeg/CPU paths where possible.

### 2. Subtitle Generation, Translation, and Embedding

Subtitle tools are included to reduce manual subtitle work:

- Generate subtitles from video audio
- Translate subtitles
- Batch subtitle processing
- Embed subtitles into VR videos
- Soft subtitle and hard subtitle related workflows

Speech recognition and translation results should still be reviewed manually, especially for names, domain-specific words, and multi-speaker dialogue.

### 3. VR Video Utilities

The toolkit also includes common VR helpers:

- Split and combine left/right eye video
- Convert VR video to flat preview/output
- Convert VR projection formats
- Take screenshots and inspect local areas
- Run batch processing scripts

### 4. VR Video DLNA Server

A highly cohesive and lightweight LAN DLNA / UPnP video streaming server:

- **Wireless VR Video Playback**: Enables VR players on the same LAN (such as Skybox VR Player in Oculus Quest, DeoVR, GizmoVR, etc.) to connect wirelessly and play videos from your PC smoothly.
- **180° SBS Format Auto-Inducing**: For 2:1 Equirectangular half-panoramic videos, the virtual filename is automatically mapped and renamed to end with `_LR_180_SBS` when browsed by clients. This perfectly induces players like Skybox to automatically render in 180° SBS 3D, avoiding tedious manual setup.
- **External Subtitle Auto-Association**: Auto-associates and prioritizes loading of external `.srt`/`.ass`/`.vtt` subtitles in the same folder, supporting prioritizing Chinese subtitles.
- **Range-supported Chunk Streaming**: Core built on FastAPI + Uvicorn, natively supports HTTP Range requests (206) for effortless, lag-free scrub/progress bar drag interactions on VR players.
- **Multi-Root Virtual Fusion**: Supports adding and removing multiple local drive video directories in the config dialog. The DLNA server will automatically merge them into a single, unified virtual tree directory structure.
- **Silent Service & Firewall Auto-pass**: Runs gracefully in a hidden background process and automatically asks for UAC permission on first run to configure Windows firewall rules for TCP 8090 and UDP 1900 SSDP.

## Recommended Usage

New users should download release file.


From the launcher, choose the tool you need:

- `One-Click Mode`: mosaic removal with minimal setup
- `Area Selection Direct Crop Mode`: local crop-based mosaic processing
- `Area Selection VR to Flat Mode`: VR-to-flat area processing
- **VR Video DLNA Server**: One-click startup/shutdown for LAN DLNA sharing, providing an independent config window for directories, port, and subtitles.
- `JAV Subtitle Tools`: subtitle generation, translation, and batch tools
- `VR Hard Subtitle Embed Tool`: hard subtitle embedding for VR video
- Other buttons: split/combine, projection conversion, flat conversion, and small utilities

## Requirements

Recommended environment:

- Windows 10/11
- NVIDIA GPU with CUDA support. This CUDA Edition is optimized for NVIDIA CUDA and is expected to perform best on a recent NVIDIA driver.
- Python 3.10 to 3.12 for source runs
- FFmpeg
- AI mosaic removal engine (choose one):
  - **Lada CLI**: https://codeberg.org/ladaapp/lada/releases
  - **Jasna CLI**: https://github.com/Kruk2/jasna/releases

Required executables and packages:

- `ffmpeg.exe`
- `ffprobe.exe`
- `lada-cli.exe` or `jasna-cli.exe` (choose one)
- Base Python packages: `Pillow`, `pyinstaller`, `ffmpy3`, `faster-whisper`, `numpy>=1.26,<2.1`, `auditok`, `huggingface-hub`, `keyring`, `requests`, `av`, `fastapi`, `uvicorn`
- CUDA/video Python packages: `pynvvideocodec>=2.1.0`, `cupy-cuda12x>=14.0`, `nvidia-cuda-nvrtc-cu12==12.8.93`, `nvidia-cuda-runtime-cu12==12.8.90`, `nvidia-cuda-cccl-cu12>=12.9.27`
- Native AI/GPU packages: `torch==2.8.0` and `torchvision==0.23.0` from the PyTorch `cu128` wheel index, plus `ultralytics==8.4.4` and `mmengine==0.10.7`

Install Python dependencies:

```bat
cd GUI\VR_Video_Toolbox
uv sync
```

If installing manually with `pip`, keep the CUDA package versions aligned with `pyproject.toml`, and install PyTorch/torchvision from `https://download.pytorch.org/whl/cu128`.

FFmpeg and the AI engine (Lada or Jasna) must be discoverable by the program. You can add them to the system `PATH`, or place the executables next to the packaged app or runtime directory.

## Project Layout

```text
.
├─ GUI/
│  └─ VR_Video_Toolbox/         Main GUI application
│     ├─ one_click/             One-click mosaic removal
│     ├─ area_selection_rect_crop/
│     ├─ area_selection_vr2flat/
│     ├─ tool_subtitle/         Subtitle generation, translation, batch processing
│     ├─ tool_subembed/         VR subtitle embedding
│     ├─ tool_dlna/             LAN DLNA/UPnP video server
│     ├─ tool_split_combine/    VR split/combine tools
│     ├─ tool_v360_trans/       VR projection conversion
│     ├─ tool_vr2flat/          VR-to-flat tools
│     └─ tools/                 Small toolbox
├─ Scripts/
│  ├─ BatchFile(Windows)/       Windows batch scripts
│  └─ Python/                   Training, subtitle, and helper scripts
├─ Models/                      Model directory
└─ prompt/                      Work notes and handover documents
```

## Output Files

Processed files are usually written next to the input video or to the output directory selected in the tool. Common filename markers include:

- `_restored`: mosaic-processed output
- `_sbs`: side-by-side left/right eye format
- `_L` / `_R`: left-eye or right-eye video
- Subtitle tools may generate `.srt`, translated subtitle files, or videos with embedded subtitles

Exact names depend on the selected tool and settings.

## FAQ

### Can I use it without an NVIDIA GPU?

Some video and subtitle tasks may still work, but AI-based mosaic removal usually depends on CUDA. Without a suitable GPU, performance and availability may be limited.

### Why is mosaic removal quality inconsistent?

The result depends on source quality, mosaic shape, VR projection distortion, AI engine (Lada or Jasna) capability, and selected parameters. Test a short clip first before processing a full video. You can also try switching engines (main window → AI Engine) to compare results.

### Are generated subtitles ready to publish?

Usually no. Speech recognition and machine translation can make mistakes, so manual review is recommended.

### Which mosaic removal mode should I choose?

Start with one-click mode on a short clip. If the result is poor, try an area selection workflow. The zoom/inspection tool in the launcher can help identify the mosaic style.

### What should I do if the LAN DLNA server is not found or cannot be opened?

1. **Firewall Blocks**: The program will automatically ask for UAC permission to open firewall ports TCP 8090 and UDP 1900 on first launch. If blocked, please allow them in Windows Security center manually.
2. **Same LAN**: Absolutely make sure that your computer and your VR headset (like Quest/Pico) are connected to the Wi-Fi of the same router, and the router does not have AP Isolation (Access Point Isolation) enabled.
3. **Add Manually**: If SSDP broadcast is not discoverable due to router multicast restrictions, you can connect wirelessly by entering the LAN IP (shown in the main interface, e.g., `192.168.x.x:8090`) in Skybox under "Network" -> "Add manual server".

## Credits

This project builds on FFmpeg, LADA, Jasna, Whisper-related tools, and community contributions. Thanks to the open-source authors and users who report issues and share improvements.
