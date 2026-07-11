# Changelog

## English

### 2026-07-11

- New: Added an optional source-video bitrate reference to Split/Combine merging. GPU and FFmpeg NVENC paths use the reference bitrate as the target with a 2x peak limit; without a usable reference, the existing CQ 18 behavior remains unchanged.
- Change: Updated the external Jasna mosaic-removal executable from `jasna-cli.exe` to `jasna.exe`, including command construction, tests, README files, and release-package instructions.

### 2026-07-10

- Change: Made high VAD sensitivity the default for subtitle generation, listening translation, and all Clone Translation Dubbing workflows; added sensitivity controls where needed and hid the obsolete single-choice segmentation-model row.
- New: AI source-text correction can delete lines containing only meaningless vocalizations through an empty-tag protocol, protected by a conservative local safety check; Clone Translation Dubbing propagates approved deletions to manifests, subtitles, and synthesis.
- Fix: Added word-timestamp reanchoring so segment boundaries use the first and last valid word timestamps when reliable, with fallback and drift logging.

### 2026-07-07

- Fix: Clone synthesis now retries OmniVoice empty-output postprocessing failures and falls back to a short silence for non-fatal per-line failures, while still surfacing CUDA/OOM and related fatal errors.
- New: Single-speaker and multi-speaker clone workflows can restore saved basis/manifest state after interruption and safely reuse basis audio already at the destination path.
- New: Candidate caches are automatically reloaded and reused; shared-folder workflows generate only missing candidates, samples, previews, and similarity scores across partially completed videos.
- New: Added a shared translation-proofreading workflow for single-speaker and multi-speaker clone modes, including optional same-name SRT references, manifest/SRT write-back, and preservation of the original AI translation.
- Change: Multi-speaker and shared-folder clone workflows now support automatic speaker-count estimation or an explicit range of 1-7 speakers.
- Change: Shortened built-in OmniVoice target-language reference sentences and standardized their target duration to 8 seconds.

### 2026-07-06

- Documentation: Updated the public README files and release-package user guides to explain the three separate fisheye-related entry points: OneClick's internal fisheye working view, Split/Combine fisheye input/output, and explicit Hequirect/Fisheye projection conversion.
- Documentation: Rewrote Clone Translation Dubbing docs for the current guided workflow, including Single-Speaker Clone, Multi-Speaker Clone, target-language basis voices, `SPEAKER1.wav` / `SPEAKER1.txt`, `.si.wav`, `.si.duck.wav`, `_SI.mp4`, `_DUB.mp4`, and DLNA `[SI]` live mixing.
- Documentation: Clarified that Clone Translation Dubbing transcription/translation requires the shared translation API configuration, and documented related OmniVoice, ECAPA, ASR, pyannote, and Bandit-v2 model requirements.
- Documentation: Reorganized the release-package readme files around independent tools instead of sequential "Step 1/2/3" instructions, matching the launcher model where mosaic removal, subtitles, clone dubbing, and utilities are separate tools.
- Documentation: Updated README wording for the migrated 2D-to-3D/VR feature, pointing users to the VR Passthrough Server project instead of the removed local 2D-to-VR implementation.

### 2026-07-05

- New: Added Clone Translation Dubbing tempo-fit pacing. One-click, single-speaker, and multi-speaker clone synthesis can now use bounded OmniVoice `speed` control to better match each translated line to the source segment duration; `moderate` is the default.
- Change: Refined tempo-fit natural-duration estimation by target language, using language-specific non-space character rates for Chinese, Japanese, Korean, Thai, English, German, French, Portuguese, Spanish, Italian, and Russian.
- Change: Updated the dubbing translation prompt to target natural spoken length near the source segment duration instead of always forcing maximum brevity, and fixed the adult-content section delimiter so disabling adult content removes only that section.
- Fix: Made candidate sample generation reproducible by deterministically seeding OmniVoice target-sample and translated-preview generation per candidate, preventing good candidates from disappearing across repeated runs.
- Change: Candidate preselection now evaluates a 2x pool sorted by usable 3-10s duration before ECAPA ranking, and one-click automatic reference selection uses the same 3-10s duration pool with quality-score selection inside the pool.
- New: Added sparse-speaker fallback reference stitching. When a speaker has no continuous 3-10s reference span, short same-speaker clips can be stitched with fades and small gaps into a usable basis reference.
- New: Added one-click clone shared-directory modes: same-folder shared basis, independent per-file batch traversal, and per-subfolder shared-basis batch traversal. Shared modes use global diarization and shared references for the same people across a folder.
- Change: Polished multi-speaker clone UX by making "Keep original" clearer, replacing speaker-list pseudo-buttons with row selection plus real action buttons, moving the candidate stop button, merging basis reuse into WAV import, renaming basis export, and making clonevoice remix volume defaults more conservative.

### 2026-07-04

- Fix: Made multi-speaker clone import lighter by lazily importing `gpu_engine.probe` only inside batch duration estimation, avoiding CUDA/GPU engine initialization during ordinary single-file imports and tests.
- Change: Reduced the multi-speaker basis-candidate dialog height and candidate table height, making it easier to fit on screen.
- Fix: Disabled the candidate generation button and candidate-count input while a multi-speaker candidate task is running, preventing duplicate candidate generation jobs.
- Change: Moved "candidate sample count" from the multi-speaker Step 2 header into the basis-candidate dialog, where it directly controls that dialog's candidate generation.
- Documentation/UI: Added the same 3-10s WAV duration and matching-text requirement note to multi-speaker basis import as the single-speaker `SPEAKER1` flow.

### 2026-07-03

- New: Added the Multi-Speaker Clone tab. It supports diarized transcription, speaker duration/segment summaries, per-speaker basis selection, skipped speakers that keep the original voice, and multi-speaker `.SI.WAV` generation.
- New: Added per-speaker basis import and voice design in the multi-speaker flow, including WAV+TXT validation, OmniVoice design previews, per-speaker `.basis.wav/.basis.txt` storage, and stop controls inside modal candidate/design workflows.
- New: Extracted the shared `CandidateBasisPanel` for source/translation/sample audition tables, then reused it in both single-speaker and multi-speaker candidate selection.
- New: Added basis export and reuse for multi-speaker clone. A selected speaker basis can be exported as reusable WAV/TXT/meta files, and imported basis WAV files can auto-fill same-name TXT sidecars.
- New: Added multi-speaker batch-directory support with global diarization. The tool can concatenate batch audio with offsets, split global turns back into per-video timelines, aggregate global speakers, collect candidates across videos, and apply one basis to every video containing that speaker.
- Change: Clonevoice translation API validation now checks the full config, not only the key, and `skip_synthesis` speakers are excluded from synthesis units and duck-key spans.
- New: DLNA SI live streaming can use a matching `.si.duck.wav` as a third sidechain input and switch to a dubbing-style live mix when enabled, with a launcher config toggle for DLNA SI dubbing mode.

### 2026-07-02

- Change: Renamed and simplified the Clonevoice "Mix / Dubbing" tab for user-facing dubbing workflows. Low-level SI channel/volume/delay controls are hidden behind sensible defaults, while the visible mode choice is now "lower original audio" versus "remove all original vocals".
- New: Added a default-on leakage-prevention option in Clonevoice SI remixing that uses the matching `.si.duck.wav` file to duck original audio across all original subtitle/manifest speech spans.
- Change: `tool_si.logic` now carries optional duck-key paths through single-file and batch SI mix tasks. When ducking is enabled and a `.si.duck.wav` exists, FFmpeg uses it as the sidechain key; if missing, it falls back to the audible SI waveform key.
- Fix: SI delay now applies only to the actual SI audio mix, not to the duck key, so original-audio ducking stays aligned to the original speech timeline.
- UI: Updated Chinese, English, and Japanese wording for Clonevoice mix/dubbing controls, including the original-audio ducking strength label and leak-prevention checkbox.

### 2026-07-01

- Fix: Added prompt-reference fade-out and tail silence for OmniVoice prompt audio, preventing source-reference tail audio from leaking into fixed samples, work references, `SPEAKER1` prompts, and final synthesis.
- Change: Single-speaker candidate translated previews now use the same second-hop clone path as final `.SI.WAV` synthesis: source candidate -> fixed target-language sample -> translated preview from that fixed sample.
- Change: Candidate ECAPA ranking now prefers translated-preview versus source-audio similarity, while retaining fixed-sample similarity as a separate score.
- Change: Single-speaker clone requires a configured translation API before transcription/translation starts, and Step 1 status/buttons now reflect the full translation configuration state.
- Change: Adjusted remix defaults to both channels, 100% SI volume, and 0s SI delay.
- New: Added SI ducking strength presets across SI tools, Clonevoice remixing, DLNA configuration, and DLNA live SI streams.
- New: Clonevoice synthesis now writes `<video>.si.duck.wav` beside `<video>.si.wav`; `tool_si.logic` gained helpers to build and write duck-key timelines from subtitle/manifest spans.
- Change: Simplified SI and Clonevoice single-file remix naming by removing manual SI WAV/output MP4 fields and using same-name sidecar/default output paths automatically.
- Fix: Single-speaker candidate generation now filters candidates without source transcript text, and Step 3 gained clearer `SPEAKER1` text/WAV requirements plus an inline play button.

### 2026-06-30

- New: Implemented the guided Single-Speaker Clone tab before the legacy one-click clone tab. The flow covers transcription/translation, candidate audition, `SPEAKER1` confirmation or import/design, and final `.SI.WAV` generation.
- New: Added `tool_clonevoice/single_clone.py` to orchestrate single-speaker clone workflows without calling `run_full()` or automatic reference extraction, preserving the user-confirmed `SPEAKER1` basis.
- New: Added reusable reference-candidate collection and OmniVoice target-language sample generation/ECAPA scoring helpers, including model-lifecycle handling that avoids keeping OmniVoice and ECAPA loaded at the same time.
- Change: Defined `SPEAKER1.wav` / `SPEAKER1.txt` as target-language basis files, with visible single-file copies named `<video>.SPEAKER1.wav/txt`, shared batch-directory copies named `SPEAKER1.wav/txt`, and per-video clone-dir copies referenced by manifest `skip_work_ref=true`.
- UI: Iterated the Single-Speaker Clone UX with target language and translation API moved to Step 1, candidate source/translation/sample audition columns, an OmniVoice voice-design dialog, ASR model status/download handling, and clearer final export controls.
- Fix: Improved final clonevoice loudness matching by allowing lower gains when generated speech is much louder than the source, raising the upper gain guard, and logging matched output loudness.
- Fix: Added NativeGPU 8K VRAM guards for large frames on 16GB/8GB-class GPUs, lowering clip length, detector batch size, and internal queue budgets, and surfacing FrameRestorer errors instead of reporting only a generic premature stop.

### 2026-06-29

- Change: Renamed the Chinese Clonevoice legacy tab label from "Clone speech and translate" wording to "One-Click Clone" to distinguish it from the new guided clone tabs.
- Planning/Documentation: Added the single-speaker clone tab design plan, defining the new guided workflow as an explicit user-controlled version of the existing automatic source-reference selection, target-language work-reference generation, and ECAPA selection path.
- Planning: Clarified that `SPEAKER1.wav` must be a target-language fixed-sample/work-reference file with matching text, not a raw source-language reference clip.

### 2026-06-28

- New: GPU/native progress logs now show VRAM usage where practical, including GPU pipeline progress, pre-scan/fine-scan progress, NativeGPU FrameRestorer detect/restore/compose stages, and native VideoWriter fallback.
- Optimization: Throttled GPU and NativeGPU progress logging to avoid rapid bursts on short/high-FPS clips, while keeping periodic FPS, ETA, and VRAM visibility.
- Major fix: Removed the paired pre-extract full-eye `BYPASS_CROP` fallback. Large or spatially jumping mosaic regions are now split into crop windows instead of forcing an expensive full-eye restore.
- Major fix: Improved fine-scan window slicing by making position jumps split independently of the area cap, and by measuring area caps per spatial cluster so simultaneous small mosaics do not create oversized combined crops.
- Major fix: Tightened final delivered encode peak bitrate for keep-bitrate workflows. Full-frame paste and final merge/re-encode paths now use the final maxrate policy, including fisheye paste, so output bitrate converges closer to source.
- Change: Cleaned OneClick pre-extract UI: the option is hidden and forced off under highest-quality encoding, the old experimental prefix was removed, and the warning text moved to a localized `?` popup.
- Change: Cleaned `vr_toolbox_config.json` so hidden GPU implementation knobs are no longer persisted. Only the UI-level `gpu_encode_profile` is saved; preset, AQ, multipass, final bitrate, and internal native options now come from code defaults.
- Change: Reduced the Tools safe keyframe cut time-point list height so the log area remains visible.

### 2026-06-27

- New: Added the experimental NativeGPU internal ROI restore + final bake path for built-in GPU mode, with a dedicated OneClick processing mode separate from full-eye processing and region-file pre-extract.
- New: Added an ROI window planner for internal restore, including safe-gap cuts, forced overlap windows with context, per-window cache/process-frame budgets, and fallback reasons for unsafe windows.
- Major optimization: Added trusted ROI restore so internal ROI can reuse the outer fine-scan rectangles instead of running LADA detector/clip a second time inside each ROI.
- Major optimization: Added source-backed internal ROI bake paths for non-fisheye SBS and fisheye workflows, including split-eye fisheye base reuse to avoid remapping full fisheye eyes on every active frame.
- Fix: Internal ROI mode no longer falls back to full-eye restore only because ROI union spread exceeds the generic planner guard; that relaxation is limited to the dedicated internal ROI mode.
- Fix: Raised internal ROI default window length to about 8s and increased the per-window patch-cache budget, reducing repeated context/overlap work without making the cache global for long videos.
- Fix: Made fisheye internal ROI failures explicit instead of silently hiding them behind old fallback paths, and moved fisheye detection debug files into the scan temp directory.
- Change: Localized the patched LADA detector/restorer chain into project-owned NativeGPU modules with AGPL attribution, making future internal ROI optimization less constrained by vendored code layout.

### 2026-06-26

- Major fix: Replaced slow source-scan keyframe listing with PyAV packet-based keyframe extraction, reducing the provided 8K sample from about 99s to about 3-4s while keeping ffprobe as fallback.
- Major optimization: Source-scan now skips Stage 2 copy-cut when one interval covers the whole or near-whole source, using the original source as the Stage 3 input without creating a redundant full-length copy.
- Major fix: Reduced final paste VRAM pressure by releasing cached NativeGPU models before GPU paste and by pre-wrapping raw HEVC restored segments before paste instead of first attempting an unstable raw PyNv seek path.
- Major update: Unified VBR bitrate policy. Intermediate outputs target about 1.2x source bitrate with 2x peak headroom; final outputs still target source bitrate, and PyNv/ffmpeg fallback paths now share the same maxrate policy.
- Major update: Final ffmpeg re-encode fallbacks now preserve 10bit/main10 and color metadata, use final-only B-frames and 2s GOP for better compression, and auto-fall back to `bf=0` on older NVENC hardware that rejects HEVC B-frames.
- Change: OneClick encode profile default is now balanced high quality, and profile options no longer show a "recommended" marker.
- Fix: CuPy/CUDA JIT cache directories now default to the project `runtime_cache` before importing CuPy, including packaged runtime hooks, avoiding hangs or slow first-use behavior from unwritable user cache folders.
- Research: Added and then cleaned paste performance diagnostics. The retained production path stays lean; experimental switches remain for NVENC CUDA Graph and stream synchronization, while `fullres` multipass remains the default.
- Research: Fine-scan PyNv review confirmed current 8K fine-scan time is dominated by sequential decode/prep across the video segment, not the number of detector samples alone.

### 2026-06-25

- Major optimization: Reworked OneClick GPU source keyframe scan away from repeated PyNv random seeking and toward a demuxer/key-packet based path, avoiding the progressive slowdown/crash pattern seen on long HEVC sources.
- Major optimization: Fine scan now uses box-only detector postprocessing and a single full-SBS paired detection pass for non-fisheye SBS, splitting boxes back to left/right eye coordinates instead of decoding and detecting each eye separately.
- Major optimization: Accurate-model fine scan keeps `accurate.pt` and `imgsz=2048` but avoids the full 8K BGR intermediate by resizing NV12/P016 planes on GPU before BGR conversion, then scaling boxes back to source coordinates.
- New: Added semantic OneClick encode profiles for highest quality, balanced high quality, fast high quality, and ultra-fast normal, shared by PyNv, ffmpeg NVENC fallback, LADA CLI, native VideoWriter fallback, paste, concat, and keyframe-cut paths.
- Major fix: Ensured NVENC multipass/AQ profile settings run under VBR rate control where required, added clear Jasna logging that Jasna only receives CQ, and fixed profile i18n namespace issues.
- Optimization: Reduced one full-frame GPU copy in the 8K paste path by pasting directly into encoder-packed Y/UV views; paste research confirmed full-frame 8K NVENC is the dominant cost, not alpha blending.
- Fix: Reworked final raw-HEVC muxing to use a tracked `Popen` wrapper, stream ffmpeg logs, honor cancellation, and let the UI stop button kill the mux child process instead of looking like Python is frozen.
- Change: OneClick pre-extract is now off by default, and DLNA configuration UI gained a virtual-drive/network-directory timeout note plus a shorter, less tall directory list.

### 2026-06-24

- Major fix: DLNA media roots on CloudDrive2 or other virtual drives now use safe path resolution, avoiding `[WinError 1005]` failures while preserving parent-traversal protection for media, subtitles, route checks, probe cache keys, and SI stream paths.

### 2026-06-23

- Major fix: Ported the proven WhisperSeg transcription front-end from Subtitle Tools into Clone Translation Dubbing, including Kotoba alignment-head repair, word timestamp preservation, configurable denoise, and quieter pyannote startup logs.
- Major fix: Fixed dropped short speech regions in the shared WhisperSeg splitter by padding short detected speech windows instead of discarding them; the same fix was mirrored to the sibling `VR_Video_Toolbox` checkout.
- Change: Hid the legacy "try GPU acceleration" row in Subtitle Tools because CUDA is now the default project path while generation and listening-translation tasks still run with GPU enabled.

### 2026-06-22

- Major update: Added native mosaic  CUDA Graph .
- Change: DLNA SI playback is now enabled by default while preserving the existing channel, volume, delay, and ducking defaults.
- Optimization: Reduced Clone Translation Dubbing launch latency by avoiding heavy backend imports during window creation and delaying backend warmup until after the UI paints.
- Fix: Clone Translation Dubbing now blocks tab 1 execution when the translation API key is missing.
- Change: Improved SI and Clonevoice UI clarity: hide the Qwen3-TTS model group when files are ready, add a DLNA live SI note, and suppress the non-actionable FlashAttention2 SDPA notice.
- Change: Clonevoice target languages are now a fixed OmniVoice-compatible list with Thai added; arbitrary "Other" target language and custom sample plumbing were removed.

### 2026-06-21

- Change: Removed the local 2D-to-VR implementation. The launcher button now shows a migration notice and links to the VR Passthrough Server download at https://wapok.com.
- New: Added DLNA SI live MPEG-TS streaming with `[SI]` time-index browsing, quick chapter leaves, and player-specific metadata for default VR players and DeoVR.
- New: Added SI audio settings to the homepage DLNA configuration dialog, including automatic `.SI.WAV` linking, channel selection, original/SI volume, SI delay, and original-audio ducking.

### 2026-06-20

- New: Added an NVDS stabilizer ONNX export tool for fixed-resolution exports from `NVDS_Stabilizer.pth`, with metadata output and ONNX Runtime validation.

### 2026-06-14

- New: Added Clone Translation Dubbing documentation and workflow notes, covering per-speaker voice cloning, SI remix output `_SI.mp4`, and dubbing output `_DUB.mp4`.
- Major update: Improved clonevoice loudness matching by aligning synthesized speech to the source sentence RMS, with bounded gain and a UI toggle.
- Major update: Completed dubbing remix behavior: SI-only controls are hidden in dubbing mode, ducking is disabled, and DUB audio can optionally be added as an independent track.
- Fix: Batch scans now ignore generated `_SI.mp4` and `_DUB.mp4` outputs so they are not reprocessed as source videos.

### 2026-06-05

- New: Added the grouped OneClick paired pre-extract pipeline. Rects sharing the same frame window can now share one GPU decode, and restored rects can use raw HEVC plus sidecar metadata instead of temporary MP4 muxes.
- Major optimization: Added group-level extract/restore pipelining and HEVC passthrough for inactive paste gaps, reducing idle time, temporary storage, and unnecessary re-encoding.
- Major fix: Fixed paired fine matching that could send detected mosaics into passthrough. Non-overlapping windows can reuse the same raw segment, high-confidence unmatched one-eye segments are retained with `pre_extract_pair_keep_unmatched_conf=0.60`, and adjacent same-rect outputs are merged.
- Major fix: Added raw HEVC paste fallback and stronger pipeline error propagation. Failed direct raw paste now retries through a temporary MP4 wrapper before falling back, and producer errors are no longer reported as user cancellation.
- Optimization: Added an empty-result cache for fine scans so repeated no-hit scans can skip detector work.

### 2026-06-04

- Major optimization: Added GPU keyframe coarse scanning for OneClick source-scan, detector batch OOM split/retry, and fast HEVC concat/Annex-B merge mode for source-scan outputs.
- Major optimization: Reworked the OneClick bitrate contract. Intermediate outputs keep quality headroom, final outputs converge to source bitrate, the 8K baseline floor was corrected, and final MP4 bitrate self-check logs oversize warnings.
- Major fix: Improved cancellation and progress behavior. User cancellation now propagates cleanly instead of falling through into expensive fallback paths.
- Major fix: Improved the 2DVR E2FGVI path with black-line cleanup, mask-aware resize, FP16/GPU compositing, and OOM fallback.

### 2026-06-03

- New: Added the 2D to Depth VR tool using local Depth Anything 3 Small, with VR180 SBS output, fisheye/equirectangular projection, start/end trimming, eye-distance control, launcher integration, and i18n.
- Major fix: Repaired Kotoba Whisper CTranslate2 alignment-head configs and safely re-enabled Kotoba word timestamps.
- Major fix: Tuned OneClick source-scan recall by scanning the source left eye at native size, fixing detector cache keys, and aligning detection metadata/debug coordinates.

### 2026-06-01

- New: Added the OneClick source-scan and pre-extract workflow to process only detected mosaic intervals and paste restored rects back onto the base video.
- Major optimization: Added fast HEVC timeline concat for source-scan outputs, with original audio copied once and GPU timeline merge kept as a fallback.
- Major fix: Corrected source-scan final merge playback/seeking issues and final GPU timeline bitrate overshoot.

### 2026-05-29 to 2026-05-30

- New: Added the `gpu_engine` backend with PyNvVideoCodec/CuPy GPU decode, transform, encode, and automatic ffmpeg fallback for major VR geometry workflows.
- New: Added the built-in `native_gpu` mosaic engine, integrating vendored LADA in-process with CUDA dependency checks and UI engine selection.
- Major optimization: GPU-accelerated VR split/merge, fisheye/equirectangular conversion, VR-to-flat projection, and OneClick geometry stages.

## 中文

### 2026-07-11

- 新功能：分割/合并工具的合并页新增可选“原始文件（码率参照）”。GPU 与 FFmpeg NVENC 路径会以参照视频码率为目标、2 倍码率为峰值；未选择有效参照时保持原有 CQ 18 行为。
- 变更：Jasna 外部去马赛克程序由 `jasna-cli.exe` 调整为 `jasna.exe`，并同步更新命令构建、测试、中英日 README 和发布包说明。

### 2026-07-10

- 变更：字幕生成、一键听译及全部克隆翻译配音流程默认使用“高”人声检测敏感度；在需要的页面补充敏感度选项，并隐藏仅有单一选项的旧语音分割模型行。
- 新功能：AI 原文校对支持通过空标签删除纯无意义语气词行，并使用保守的本地安全检查防止误删；克隆翻译配音会把确认删除同步到 manifest、字幕和语音合成。
- 修复：新增词级时间戳重锚定，在时间可靠时使用首词和末词修正段落边界，并保留回退及漂移日志。

### 2026-07-07

- 修复：OmniVoice 单句生成遇到空音频后处理错误时自动重试；非致命单句失败以短静音占位继续任务，同时仍会抛出 CUDA/OOM 等致命错误。
- 新功能：单人和多人克隆可在中断后恢复已保存的 basis/manifest 状态，并安全复用源、目标路径相同的基准音频。
- 新功能：自动载入并复用候选缓存；同目录共用流程面对部分完成的视频时，只补齐缺失候选、样句、翻译试听和相似度。
- 新功能：单人和多人克隆新增共享的翻译校对流程，支持同名 SRT 参考、回写 manifest/字幕，以及保留首版 AI 翻译备份。
- 变更：多人及同目录共用克隆支持自动估算说话人数，也可明确选择 1-7 人。
- 变更：缩短 OmniVoice 内置目标语言基准样句，并统一以 8 秒为目标时长。

### 2026-07-06

- 文档：更新 GitHub 公开 README 和发布包用户说明，明确三个“鱼眼”入口的区别：一键模式内部鱼眼工作视角、拆分/合并工具的鱼眼输入输出、以及 Hequirect/半球 与 Fisheye/鱼眼的显式投影转换。
- 文档：按当前引导式流程重写克隆翻译配音说明，覆盖单人语音克隆、多人语音克隆、目标语言参考音色、`SPEAKER1.wav` / `SPEAKER1.txt`、`.si.wav`、`.si.duck.wav`、`_SI.mp4`、`_DUB.mp4` 和 DLNA `[SI]` 直播混音。
- 文档：补充克隆翻译配音的转录/翻译需要使用字幕翻译共用的翻译 API 配置，并同步说明 OmniVoice、ECAPA、ASR、pyannote、Bandit-v2 等相关模型要求。
- 文档：将发布包 readme 从“第一步/第二步/第三步”式说明调整为按独立小工具组织，更符合主界面中去马赛克、字幕、克隆配音和辅助工具彼此独立的使用方式。
- 文档：更新已迁移的 2D 转 3D/VR 说明，指向 VR Passthrough Server 项目，不再描述已移除的本地 2D 转 VR 实现。

### 2026-07-05

- 新功能：克隆翻译配音新增“语速贴合原片”。一键克隆、单人克隆、多人克隆生成语音时，可用有界的 OmniVoice `speed` 控制让翻译句更贴近原片每句时长，默认使用“适度”。
- 变更：自然朗读时长估算改为按目标语言细分，分别覆盖中文、日文、韩文、泰文、英语、德语、法语、葡萄牙语、西班牙语、意大利语、俄语等语言的非空白字符速率。
- 变更：配音翻译 prompt 从“一味压短”改为匹配原句自然口语时长，并修复 adult content 分隔符，使关闭成人内容时只删除成人背景段。
- 修复：候选样句与翻译试听生成改为按候选确定性播种 OmniVoice，避免同一个候选每次生成不同音频、好候选下次消失。
- 变更：候选预选改为先按 3-10 秒有效时长取 2 倍候选池，再生成样句并按 ECAPA 排序；一键克隆自动基准选择也改用 3-10 秒时长池并在池内按质量分选优。
- 新功能：稀疏说话人参考音回退。某个说话人没有连续 3-10 秒片段时，可把多个短句按时间顺序拼接成带淡入淡出和短静音的参考音。
- 新功能：一键克隆新增同目录共用、批量遍历单文件、批量遍历同目录共用等输入模式；共享模式使用全局说话人分离和跨视频共享参考音，让同一目录同一批人物保持同一组音色。
- 变更：多人克隆体验优化，包括“保留原声”文案、说话人列表真实按钮、候选停止按钮位置、复用基准合并到导入 WAV、导出基准改名，以及混音配音音量默认值更保守。

### 2026-07-04

- 修复：多人语音克隆批量时长估算改为在函数内部惰性导入 `gpu_engine.probe`，避免普通导入、单文件多人克隆和单元测试无谓触发 CUDA/GPU engine 初始化。
- 变更：多人选择基准语音对话框高度和候选表高度缩小，更适合普通屏幕显示。
- 修复：多人候选生成任务运行期间禁用“抽取候选并生成样句”按钮和候选数量输入框，避免重复启动同一任务。
- 变更：多人克隆 Step 2 顶部不再显示候选样本数量，该设置移动到选择基准语音对话框内，更贴近实际作用范围。
- 文档/UI：多人导入基准语音备注补齐 3-10 秒 WAV、语音内容与文本匹配、语言与目标语言一致等约束，与单人 `SPEAKER1` 流程保持一致。

### 2026-07-03

- 新功能：新增“多人语音克隆”tab，支持多人转录/说话人分离、speaker 总时长和段数汇总、逐说话人选择基准、保留原声跳过，以及多人 `.SI.WAV` 生成。
- 新功能：多人流程新增按说话人导入 WAV+TXT 和 OmniVoice 设计音色，支持试听、保存、停止，并固化为 `<speaker>.basis.wav/txt`。
- 新功能：抽出共享候选表组件 `CandidateBasisPanel`，统一单人和多人候选列表中的原音、翻译试听、样句播放和候选选择行为。
- 新功能：多人基准音支持导出和复用，可导出可复用的 WAV/TXT/meta，也可导入带同名 TXT 侧车的 WAV 并自动填文本。
- 新功能：多人克隆支持批量目录全局说话人分离。工具会拼接整批音频、拆回每个视频时间线、跨视频聚合全局 speaker、跨视频收集候选，并把同一个 speaker basis 应用到所有包含该 speaker 的视频。
- 变更：克隆翻译配音的翻译 API 预检从只检查 API Key 升级为检查完整配置；设置 `skip_synthesis` 的说话人会从合成单元和 duck key 时间段中排除。
- 新功能：DLNA `[SI]` 直播流可在发现同名 `.si.duck.wav` 时使用第三路 sidechain key，并切换到配音风格直播混音；主界面 DLNA 配置新增对应开关。

### 2026-07-02

- 变更：克隆语音“混音配音”页面面向普通配音流程重命名和简化，隐藏声道/音量/延迟等低层 SI 控件，模式改为“压低原声”和“移除所有原始人声”。
- 新功能：克隆语音 SI 回混新增默认开启的防漏音选项，使用同名 `.si.duck.wav` 覆盖所有原始字幕/manifest 语音时间段来压低原声。
- 变更：`tool_si.logic` 的单文件和批量混音任务可携带 duck key 路径；开启 duck 且存在 `.si.duck.wav` 时，FFmpeg 使用该文件作为 sidechain key，缺失时回退到旧的 SI 波形 key。
- 修复：SI 延迟只作用于实际混入的 SI 音频，不再移动 duck key，因此原声压低时间仍与原始语音时间线对齐。
- UI：中/英/日同步更新克隆语音混音配音页文案，包括“原声压低强度”和防漏音 checkbox。

### 2026-07-01

- 修复：OmniVoice prompt 参考音新增尾部淡出和静音副本，避免源参考音尾部串入固定样句、work reference、`SPEAKER1` prompt 和最终合成开头。
- 变更：单人克隆候选的翻译试听改为和最终 `.SI.WAV` 一致的二跳链路：源候选生成固定目标语言样句，再由固定样句克隆翻译句试听。
- 变更：候选 ECAPA 排序优先使用“翻译试听 vs 原音”的相似度，同时保留固定样句相似度作为独立分数。
- 变更：单人克隆 Step 1 启动前必须配置翻译 API，界面状态和按钮会反映完整翻译配置状态。
- 变更：混音默认值调整为左右声道、SI 音量 100%、SI 延迟 0 秒。
- 新功能：SI duck 强度预设贯穿同声传译工具、克隆语音混音、DLNA 配置和 DLNA `[SI]` 直播流。
- 新功能：克隆语音生成 `<视频名>.si.wav` 时同步写出 `<视频名>.si.duck.wav`；`tool_si.logic` 新增根据字幕/manifest 时间段生成 duck key timeline 的 helper。
- 变更：同声传译和克隆语音单文件回混移除手填 SI WAV/输出 MP4，统一使用同名 sidecar 和默认输出命名。
- 修复：单人候选生成会过滤没有源转录文本的候选；Step 3 补充更明确的 `SPEAKER1` 文本/WAV 约束和内联播放按钮。

### 2026-06-30

- 新功能：实现单人语音克隆引导式 tab，位于旧一键克隆之前，覆盖转录翻译、候选试听、确认或导入/设计 `SPEAKER1`、最终生成 `.SI.WAV`。
- 新功能：新增 `tool_clonevoice/single_clone.py` 编排单人流程，最终阶段不调用 `run_full()` 或自动参考音提取，避免覆盖用户确认的 `SPEAKER1`。
- 新功能：新增可复用的参考候选收集、OmniVoice 目标语言样句生成、ECAPA 评分等 helper，并调整模型生命周期，避免 OmniVoice 与 ECAPA 同时驻留显存。
- 变更：明确 `SPEAKER1.wav` / `SPEAKER1.txt` 是目标语言基准音色文件；单文件可见副本使用 `<视频名>.SPEAKER1.wav/txt`，批量目录共享副本使用 `SPEAKER1.wav/txt`，每个视频工作目录中仍写入固定 `SPEAKER1.wav/txt` 并在 manifest 中设置 `skip_work_ref=true`。
- UI：单人克隆多轮调整，加入目标语言/API 前置、原音/翻译/样句试听列、OmniVoice 音色设计对话框、ASR 模型状态与下载、最终导出控件等。
- 修复：克隆语音最终响度匹配允许更低增益并提高上限保护，解决生成语音明显大于源语音时仍降不下来的问题，同时日志输出匹配后的响度。
- 修复：NativeGPU 8K 处理新增 16GB/8GB 级别显存保护，自动降低 clip 长度、检测 batch 和队列预算，并暴露 FrameRestorer 真实错误，避免只显示笼统的提前停止。

### 2026-06-29

- 变更：中文界面中克隆语音旧 tab 标签从“克隆语音并翻译”改为“一键克隆”，和后续新增的引导式克隆 tab 区分开。
- 方案/文档：新增单人语音克隆 tab 设计方案，明确它不是新增克隆能力，而是把现有自动源参考选择、目标语言 work_ref 生成和 ECAPA 选优前移为用户可试听、可干预的显式流程。
- 方案：明确 `SPEAKER1.wav` 必须是带匹配文本的目标语言固定样句/work reference，而不是直接保存源语言原音片段。

### 2026-06-28

- 新功能：GPU/native 进度日志尽量显示显存占用，覆盖 GPU pipeline、pre-scan/fine-scan、NativeGPU FrameRestorer detect/restore/compose，以及 native VideoWriter fallback。
- 优化：降低 GPU 和 NativeGPU 进度日志频率，避免短视频或高 FPS 片段一秒内刷出大量日志，同时保留周期性的 FPS、ETA 和显存信息。
- 重大修复：移除 paired pre-extract 的整眼 `BYPASS_CROP` fallback。遇到大范围或跳变马赛克时改为继续切 crop 窗口，而不是直接进入昂贵的整眼恢复。
- 重大修复：改进 fine-scan 窗口切割。马赛克位置跳变现在独立触发切段；面积限制改为按空间 cluster 计算，避免同帧多个小马赛克被合并成一个虚大的 crop。
- 重大修复：收紧 keep-bitrate 流程的最终成品峰值码率策略。全帧 paste、最终 merge/re-encode、fisheye paste 都使用 final maxrate，让最终 MP4 更接近源码率。
- 变更：清理 OneClick pre-extract UI：最高画质下隐藏并强制关闭该选项，移除旧的“实验功能”前缀，说明文字改为本地化 `?` 弹窗。
- 变更：清理 `vr_toolbox_config.json`。隐藏的 GPU 实现参数不再持久化，只保存 UI 层的 `gpu_encode_profile`；preset、AQ、multipass、最终码率和 native 内部选项改由代码默认值提供。
- 变更：缩小工具页“安全帧切割”的时间点列表高度，避免挤占日志区域。

### 2026-06-27

- 新功能：新增实验性的 NativeGPU internal ROI restore + final bake 路径，仅用于内置 GPU 模式，并在 OneClick 中作为独立处理模式，不再混用整眼处理或区域文件预提取选项。
- 新功能：新增 ROI window planner，支持安全空档切割、带上下文的强制 overlap 窗口、每窗口 cache/帧数预算，以及明确的 fallback 原因。
- 重大优化：新增 trusted ROI restore，internal ROI 可复用外层 fine-scan 得到的 rect，避免在 ROI 内再次跑一遍 LADA detector/clip。
- 重大优化：新增非鱼眼 SBS 和鱼眼流程的 source-backed internal ROI bake；鱼眼路径可复用 split-eye fisheye base，避免每个活动帧重复整眼 fisheye remap。
- 修复：internal ROI 模式不再仅因 ROI union spread 超过通用 planner 阈值就 fallback 到整眼恢复；该放宽只作用于专用 internal ROI 模式。
- 修复：internal ROI 默认窗口长度提高到约 8 秒，并提高单窗口 patch cache 预算，减少重复上下文/overlap 工作，同时不为长视频建立全局无限缓存。
- 修复：fisheye internal ROI 失败现在会明确暴露具体原因，不再静默退回旧路径；fisheye detection debug 文件也改为写入 scan temp 目录。
- 变更：将 patched LADA detector/restorer 链路本地化到项目 NativeGPU 模块，并保留 AGPL 来源说明，方便后续继续改造 internal ROI。

### 2026-06-26

- 重大修复：source-scan 关键帧列表改为优先使用 PyAV packet keyframe 提取。用户提供的 8K 样本从约 99 秒降到约 3-4 秒，同时保留 ffprobe 兜底。
- 重大优化：source-scan 只有一个整片/近整片 interval 时跳过 Stage 2 copy-cut，直接把原始视频作为 Stage 3 输入，避免生成几乎完整的临时拷贝。
- 重大修复：降低 final paste 显存压力。GPU paste 前释放 cached NativeGPU 模型，并在 paste 前主动把 raw HEVC restored segments 包装成 MP4，避免先尝试不稳定 raw PyNv seek 再重试。
- 重大更新：统一 VBR 码率策略。中间产物目标约为源码率 1.2 倍、峰值 2 倍；最终成品仍以源码率为目标，PyNv 和 ffmpeg fallback 使用一致的 maxrate 策略。
- 重大更新：最终 ffmpeg 重编码 fallback 现在保留 10bit/main10 和色彩元数据，使用 final-only B 帧与 2 秒 GOP 提升压缩效率，并对不支持 HEVC B 帧的老 NVENC 自动回落到 `bf=0`。
- 变更：OneClick 编码档位默认改为“均衡高画质”，选项文案不再显示“推荐”。
- 修复：CuPy/CUDA JIT 缓存默认指向项目 `runtime_cache`，并覆盖打包 runtime hook，避免用户目录缓存不可写导致首次 JIT 卡住或异常变慢。
- 研究：新增并清理 paste 性能诊断。生产路径保持简洁，仅保留 NVENC CUDA Graph 和 stream sync 实验开关；`fullres` multipass 仍为默认。
- 研究：复查 fine-scan PyNv 路径，确认当前 8K fine-scan 主要受整段顺序 decode/prep 限制，而不是单纯由 detector sample 数决定。

### 2026-06-25

- 重大优化：OneClick GPU source keyframe scan 从反复 PyNv 随机 seek 改向 demuxer/key-packet 路径，规避长 HEVC 源上逐渐变慢甚至崩溃的问题。
- 重大优化：fine scan 新增 box-only detector 后处理，并让非鱼眼 SBS paired scan 一次检测全 SBS，再把 box 拆回左右眼坐标，避免左右眼重复解码和重复检测。
- 重大优化：accurate 模型 fine scan 仍保留 `accurate.pt` 与 `imgsz=2048`，但改为先在 GPU 上缩放 NV12/P016 平面，再转 BGR，并把 box 坐标缩放回原始空间，减少 8K BGR 中间图开销。
- 新功能：新增语义化 OneClick 编码档位：最高画质、均衡高画质、快速高画质、极速普通画质；同一套 profile 被 PyNv、ffmpeg NVENC fallback、LADA CLI、native VideoWriter fallback、paste、concat 和 keyframe-cut 共用。
- 重大修复：确保 NVENC multipass/AQ 档位在需要时使用 VBR rate control；补充 Jasna 日志说明 Jasna 只接收 CQ；修复编码档位 i18n namespace 问题。
- 优化：8K paste 路径减少一次全帧 GPU copy，直接在 encoder-packed Y/UV view 上贴回；性能研究确认主瓶颈是 8K 全帧 NVENC，而不是 alpha blending。
- 修复：raw HEVC 最终 mux 改为 tracked `Popen` 执行，ffmpeg 日志流式写入，支持取消，并允许 UI 停止按钮杀掉 mux 子进程，避免看起来像 Python 卡死。
- 变更：OneClick pre-extract 默认关闭；DLNA 配置界面新增虚拟盘/网盘目录载入超时提示，并缩短目录列表高度。

### 2026-06-24

- 重大修复：DLNA 媒体目录现在对 CloudDrive2 等虚拟盘使用安全路径解析，避免 `[WinError 1005]`；同时仍保留媒体、字幕、路由检查、probe cache 和 SI stream 路径的父目录穿越保护。

### 2026-06-23

- 重大修复：将字幕工具中验证过的 WhisperSeg 转录前端移植到克隆翻译配音，包含 Kotoba alignment-head 修复、词级时间戳保留、可配置降噪，以及减少 pyannote 启动时的干扰日志。
- 重大修复：修复共享 WhisperSeg 分段器漏掉短句的问题。现在对已检测到的短语音窗口做补齐，而不是直接丢弃；同一修复也同步到兄弟项目 `VR_Video_Toolbox`。
- 变更：隐藏字幕工具中的旧版“尝试用 GPU 加速”选项行。项目默认走 CUDA，同时生成字幕和听译任务仍保持 GPU 启用。

### 2026-06-22

- 重大更新：`native_gpu` 支持 CUDA Graph 加速。
- 变更：DLNA SI 播放默认开启，同时保留原有声道、音量、延时和 ducking 默认参数。
- 优化：降低克隆翻译配音入口卡顿，窗口创建阶段不再导入重型后端，并把后端预热延后到 UI 绘制之后。
- 修复：克隆翻译配音 tab1 在未配置翻译 API Key 时会阻止任务启动。
- 变更：优化 SI 与克隆翻译配音界面：模型文件齐备时隐藏 Qwen3-TTS 模型组，标题下方增加 DLNA 直播 SI 提示，并屏蔽非错误的 FlashAttention2 SDPA 提示。
- 变更：克隆翻译配音目标语言改为 OmniVoice 兼容的固定列表，新增泰语，并移除任意“其他”语言和自定义样句通道。

### 2026-06-21

- 变更：移除本地 2D 转 VR 实现。主界面按钮现在显示迁移提示，并链接到 https://wapok.com 下载 VR Passthrough Server。
- 新功能：新增 DLNA SI MPEG-TS 直播流，支持 `[SI]` 时间索引浏览、快速章节入口，以及默认 VR 播放器和 DeoVR 的差异化元数据。
- 新功能：主界面 DLNA 配置新增 SI 音频参数，包括自动关联 `.SI.WAV`、声道选择、原音/SI 音量、SI 延时和原音 ducking。

### 2026-06-20

- 新功能：新增 NVDS stabilizer ONNX 导出工具，可从 `NVDS_Stabilizer.pth` 导出固定分辨率 ONNX，生成元数据并通过 ONNX Runtime 验证。

### 2026-06-14

- 新功能：补充克隆翻译配音文档与流程说明，覆盖按说话人音色克隆、同声传译输出 `_SI.mp4`、配音输出 `_DUB.mp4`。
- 重大更新：改进 clonevoice 响度匹配，合成语音会按原句 RMS 对齐，并加入增益保护和界面开关。
- 重大更新：完善配音回混行为：配音模式隐藏 SI 专用控件、禁用 ducking，并支持将 DUB 音频作为独立音轨加入。
- 修复：批量扫描会忽略已生成的 `_SI.mp4` / `_DUB.mp4`，避免把输出文件再次当作源视频处理。

### 2026-06-05

- 新功能：新增 OneClick paired pre-extract 分组流水线。同一帧窗口内的多个 rect 可共享一次 GPU 解码，恢复小段可使用 raw HEVC + sidecar 元数据，减少临时 MP4 mux。
- 重大优化：新增 group 级 extract/restore 流水线，以及 HEVC inactive gap passthrough，减少等待时间、临时空间占用和不必要的重编码。
- 重大修复：修复 paired fine 匹配把已检测到的马赛克误送入 passthrough 的问题。同一 raw segment 可在不重叠时间窗复用，高置信单眼 unmatched 段通过 `pre_extract_pair_keep_unmatched_conf=0.60` 保留，同侧相邻同 rect 输出会合并。
- 重大修复：增强 raw HEVC paste 兜底和 pipeline 异常传播。直接消费 raw HEVC 失败时会先临时包装成 MP4 重试，producer 普通异常不再被误报成用户取消。
- 优化：新增 fine scan 空结果缓存，重复无命中扫描可跳过 detector。

### 2026-06-04

- 重大优化：OneClick source-scan 新增 GPU keyframe 粗扫、detector batch OOM 自动拆批重试，以及 source-scan 输出的 HEVC Annex-B 快速合并模式。
- 重大优化：重构 OneClick 码率契约。中间产物保留画质裕度，最终输出默认收敛到源码率，修正 8K baseline 下限，并新增最终 MP4 平均码率自检和超限 warning。
- 重大修复：改进取消和进度行为。用户取消会干净向上传播，不再继续进入耗时 fallback。
- 重大修复：改进 2DVR E2FGVI 路径，加入黑线清理、mask-aware resize、FP16/GPU 合成和 OOM 兜底。

### 2026-06-03

- 新功能：新增 2D 转深度 VR 工具，使用本地 Depth Anything 3 Small，支持 VR180 SBS 输出、鱼眼/等距投影、起止时间、眼距控制、主界面入口和多语言文案。
- 重大修复：修复 Kotoba Whisper CTranslate2 alignment-head 配置，安全恢复 Kotoba word timestamps。
- 重大修复：调优 OneClick source-scan 召回率：改为原始尺寸左眼粗扫，修复 detector cache key，并对齐检测 metadata/debug 坐标。

### 2026-06-01

- 新功能：新增 OneClick source-scan / pre-extract 工作流，只处理检测到马赛克的时间区间，并把恢复后的 rect 贴回底片视频。
- 重大优化：新增 source-scan 输出的 HEVC timeline 快速拼接，最终音频只从原始源复制一次，并保留 GPU timeline merge 作为兜底。
- 重大修复：修复 source-scan 最终合并后的播放/seek 问题，以及最终 GPU timeline merge 的码率膨胀问题。

### 2026-05-29 至 2026-05-30

- 新功能：新增 `gpu_engine` 后端，基于 PyNvVideoCodec/CuPy 实现 GPU 解码、几何变换、编码，并为主要 VR 几何流程保留 ffmpeg 自动兜底。
- 新功能：新增内置 `native_gpu` 去马赛克引擎，进程内集成 vendored LADA，支持 CUDA 依赖检查和 UI 引擎选择。
- 重大优化：VR 分割/合并、鱼眼/等距转换、VR 转平面和 OneClick 几何阶段接入 GPU 加速。
