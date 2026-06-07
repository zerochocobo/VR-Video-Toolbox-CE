════════════════════════════════════════════════════════
     VR Video Toolbox (CUDA EDITION)  —  User Guide
════════════════════════════════════════════════════════

◆ Step 1: Download and Install Required Tools
════════════════════════════════════════════════════════

This program requires two external tools to function. Please download them first:

┌─────────────────────────────────────────────────────┐
│  Tool 1: FFmpeg  (Required for all features)         │
└─────────────────────────────────────────────────────┘

1. Visit: https://www.gyan.dev/ffmpeg/builds/
2. Find "ffmpeg-release-full.7z" (or .zip) and download it.
3. Extract the archive, open the extracted folder, then open the "bin" subfolder.
   It contains: ffmpeg.exe and ffprobe.exe (among others).
4. There are three ways to make these available to the program:
   Option 1: Copy both .exe files into the same folder as VR_Video_Toolbox.exe
   Option 2: Copy both .exe files into C:\Windows or C:\Windows\System32
   Option 3: Add the "bin" folder path to Windows' PATH environment variable

┌─────────────────────────────────────────────────────┐
│  Tool 2: Lada-CLI  (AI mosaic removal, requires GPU) │
└─────────────────────────────────────────────────────┘

1. Visit: https://codeberg.org/ladaapp/lada/releases
2. Download the latest release following the instructions on that page.
3. There are two ways to make lada-cli available to the program:
   Option 1: Copy the entire lada-cli folder (including the _internal directory)
             into the same folder as VR_Video_Toolbox.exe, so that lada-cli.exe
             and VR_Video_Toolbox.exe are in the same directory.
   Option 2: Add the lada-cli folder to Windows' PATH environment variable,
             e.g.: D:\Lada\
4. Note: Lada requires an NVIDIA GPU (faster GPU = faster processing).
         CPU-only processing is extremely slow and not recommended.

┌─────────────────────────────────────────────────────┐
│  Tool 2b: Jasna-CLI  (Recommended Alternative)       │
└─────────────────────────────────────────────────────┘

Jasna is a modern, actively maintained fork of the original Lada engine,
compatible with the same workflow.

1. Visit: https://github.com/Kruk2/jasna/releases
2. Download the latest Windows release.
3. There are two ways to make jasna-cli available to the program:
   Option 1: Extract and copy jasna-cli.exe (and its _internal folder if any)
             into the same folder as VR_Video_Toolbox.exe.
   Option 2: Add the jasna folder to Windows' PATH environment variable.
4. Select the engine in the main window: choose "Jasna" or "Lada" under AI Engine.
   Your choice is remembered automatically.
5. Note: Jasna requires an NVIDIA GPU, same as Lada.


◆ Step 2: Choose a Mosaic Removal Mode
════════════════════════════════════════════════════════

After launching the program, the main screen offers three modes. Use the guide below:

┌─────────────────────────────────────────────────────┐
│  [Recommended] One-Click Mode                        │
└─────────────────────────────────────────────────────┘
  Works for: S1VR, MDVR, VRKM, IPVR, and most major studios.
  Easiest to use — automatically detects and removes the mosaic.

  ▶ How do I know if this mode applies?
    Put on your VR headset and look at the mosaic at the bottom of the video.
    If the mosaic looks like a [fan / arc shape]  → Use "One-Click Mode" directly.
    If the mosaic looks like a [square/grid]      → Enable "Convert to fisheye first"
                                                    in the settings before processing.

┌─────────────────────────────────────────────────────┐
│  Area Selection - Direct Crop Mode                   │
└─────────────────────────────────────────────────────┘
  Works for: Regular (non-VR) videos, or when the mosaic appears as a rectangle
             on a normal flat screen.
  How it works: Manually select the mosaic area → program processes and overlays it back.

┌─────────────────────────────────────────────────────┐
│  Area Selection - VR to Flat Mode                    │
└─────────────────────────────────────────────────────┘
  Works for: FSVSS, SAVR, URVRSP, CRVR, PXVR, and some 3DSVR titles.

  ▶ How do I know if this mode applies?
    In your VR headset, the bottom mosaic looks like a [square/grid],
    but on your PC monitor the mosaic in the raw file looks [trapezoidal / slanted]
    → Use this mode.
  This mode is more complex to configure. Try One-Click Mode first.

Not sure which mode fits your video?
→ Click the "Unsure about mosaic style? Check with Zoom Tool" button on the main screen.


◆ Step 3: Batch Subtitle Generation (Optional)
════════════════════════════════════════════════════════

Click "Japanese Batch Subtitle Tools" on the main screen to access subtitle generation.

This feature works independently of mosaic removal and does NOT require Lada.
It does require:
  1. FFmpeg (same as above)
  2. Speech recognition model files (click "Download Model" on first use;
     files are saved automatically to the "models" folder)
  3. [Optional] NVIDIA GPU acceleration: automatically enabled if a CUDA environment
     is detected — can be tens of times faster than CPU.
     Without a GPU, the program falls back to CPU mode, which is very slow
     but still functional.


◆ Frequently Asked Questions
════════════════════════════════════════════════════════

Q: The program says "ffmpeg not found". What do I do?
A: Make sure ffmpeg.exe is in the same folder as VR_Video_Toolbox.exe,
   or that ffmpeg's bin directory is added to the Windows PATH variable.

Q: The mosaic removal has no effect / poor results?
A: Both Jasna and Lada work best on rectangular mosaics. For some studios, the
   mosaic shape is unusual and the AI cannot reconstruct the image — this is a
   known technical limitation. Try switching engines (main screen → AI Engine)
   or try a different removal mode.

Q: The subtitle tool produces garbled text / not Japanese?
A: The tool transcribes Japanese speech by default. Make sure the video
   contains Japanese audio. Results are saved as .jp.srt in the same folder
   as the video file.

Q: I get an error like "Option 'ad' not found" or "Error initializing filter"?
A: This usually means your ffmpeg version is too old, or you are using a "minimal/lite" 
   build that lacks advanced audio denoise or VR filters. Please download and replace
   it with the latest "Full Build" of ffmpeg (version 6.0 or higher is recommended).
   Download: https://www.gyan.dev/ffmpeg/builds/ (Select ffmpeg-release-full.7z)

Q: Where can I download the latest version of this program?
A: Homepage: https://github.com/zerochocobo/VR-Video-Toolbox-CE

════════════════════════════════════════════════════════
