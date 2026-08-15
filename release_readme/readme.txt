════════════════════════════════════════════════════════
     VR Video Toolbox (CUDA EDITION)  —  User Guide
════════════════════════════════════════════════════════

◆ External Tools Used by Different Features
════════════════════════════════════════════════════════

VR Video Toolbox is a collection of independent tools. Different tools require
different external components:

  - FFmpeg / ffprobe: common video and audio processing dependency.
  - Lada or Jasna: needed only for AI mosaic removal.
  - Speech recognition and IndexTTS-2.5 models: needed for the new sentence-
    level emotional voice-cloning tool. OmniVoice, ECAPA, and pyannote are
    used only by the separate legacy clone-dubbing tool.

┌─────────────────────────────────────────────────────┐
│  FFmpeg  (common video/audio dependency)             │
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
│  Lada-CLI  (AI mosaic removal, requires GPU)         │
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
│  Jasna-CLI  (Recommended Alternative for Mosaic)     │
└─────────────────────────────────────────────────────┘

Jasna is a modern, actively maintained fork of the original Lada engine,
compatible with the same workflow.

1. Visit: https://github.com/Kruk2/jasna/releases
2. Download the latest Windows release.
3. There are two ways to make jasna available to the program:
   Option 1: Extract and copy jasna.exe (and its _internal folder if any)
             into the same folder as VR_Video_Toolbox.exe.
   Option 2: Add the jasna folder to Windows' PATH environment variable.
4. Select the engine in the main window: choose "Jasna" or "Lada" under AI Engine.
   Your choice is remembered automatically.
5. Note: Jasna requires an NVIDIA GPU, same as Lada.


◆ Mosaic Removal Tools
════════════════════════════════════════════════════════

Use these only when you want to remove mosaics. The main screen offers several
mosaic-removal modes:

┌─────────────────────────────────────────────────────┐
│  [Recommended] One-Click Mode                        │
└─────────────────────────────────────────────────────┘
  Works for: S1VR, MDVR, VRKM, IPVR, and most major studios.
  Easiest to use — automatically detects and removes the mosaic.

  ▶ How do I know if this mode applies?
    Put on your VR headset and look at the mosaic at the bottom of the video.
    If the mosaic looks like a [fan / arc shape]  → Use "One-Click Mode" directly.
    If the mosaic looks like a [square/grid]      → Enable "Convert to fisheye before processing".

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


◆ What the Three Fisheye Options Mean
════════════════════════════════════════════════════════

The word "fisheye" appears in three different tools. Use the one that matches
your task:

1. One-Click Mode: "Convert to fisheye before processing"
   Use this for mosaic removal when the mosaic looks square/grid-like in the
   VR headset, especially center-axis or bottom-area mosaics from studios such
   as SAVR/URVRSP. The program converts each eye to fisheye only as an internal
   working view, removes the mosaic, then converts it back. The final result is
   still a normal VR video.

2. Split / Combine Tool: fisheye split or fisheye combine
   Use this for manual workflows. Split with fisheye creates separate left/right
   fisheye eye files. Combine with fisheye expects already-fisheye eye files and
   converts them back before creating an SBS VR video.

3. Projection Conversion Tool: Hequirect <-> Fisheye
   Use this only when you deliberately want to change the projection format of a
   video file. For SBS videos that contain both eyes in one file, enable the
   dual-screen/SBS option so the two halves are converted separately and stacked
   back correctly.

For normal mosaic removal, start with One-Click Mode. Do not run the projection
converter first unless you specifically need a standalone fisheye file.


◆ Batch Subtitle Generation Tool
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


◆ Sentence-Level Emotional Voice Cloning
════════════════════════════════════════════════════════

Click "Sentence-Level Emotional Voice Cloning" on the main screen. This new
tool uses Bilibili's latest open-source IndexTTS-2.5. Each source sentence WAV
is used directly as its timbre and emotion prompt, with no reference transcript
and no speaker diarization or SPEAKER prefix.

Voice Clone (one video)
   1. Click "Transcribe & Translate" and select the source video.
   2. Review and correct the translation.
   3. Click "Proofread Translation and Export Cloned Voice File" to generate
      the cloned audio.
   Every sentence normally uses its matching source WAV. If it is too short,
   the program automatically substitutes a longer source sentence as prompt.

Batch Clone
   Select one input directory. "Search all subdirectories for video files" is
   enabled by default. It completes transcription, translation, and synthesis
   for the current directory before starting the next directory. With "Skip
   existing intermediate files and .SI.WAV" enabled, existing audio16k.wav,
   manifest.json, source.srt, translated.srt,
   and final .si.wav checkpoints are reused instead of processed again.

Output and Mix / Dubbing
   Synthesis creates <video>.si.wav and <video>.si.duck.wav. Mix / Dubbing always
   keeps and lowers the original track, uses the strongest ducking level by
   default, overlays the cloned voice, and outputs _SI.mp4. The DLNA server can
   also live-mix the matching .SI.WAV through [SI]. Per-sentence reference and
   generated WAV files remain in <video>.clone/indextts_v2_manifest/ beside the
   source video; they are not written to the Windows Temp directory.

Place the model under models/IndexTTS-2.5. If files are incomplete, the program
resumes the main/auxiliary model download in this fixed directory (ModelScope
first, then hf-mirror fallback). Translation uses the same API configuration as
subtitle translation. Always review the text and listen to .si.wav or _SI.mp4.

New tool versus legacy tool
   New "Sentence-Level Emotional Voice Cloning": IndexTTS-2.5; one matching
   source WAV per sentence; no reference text; no speaker recognition.
   Legacy "Clone Translation Dubbing": OmniVoice; speaker recognition and
   matching reference text are required; simple voice design is supported.
   These are separate home-screen entries and separate workflows.


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
