# Standalone Subtitle Analyzer Development Plan

Date: 2026-07-15

## Conclusion

The analyzer can be made independent from subtitle generation. Keep the current debug-session workflow and add a media-file workflow where users select a normal MP4/MKV file, choose from same-stem `.srt` files including `.jp.srt`, and optionally generate the missing analysis WAV on demand.

No ASR model is required. The implementation can reuse FFmpeg, the tolerant SRT parser, NumPy/soundfile waveform processing, and the WinMM playback backend.

## Proposed workflow

- Add a source selector with `Debug Session` and `Media File` modes.
- In media mode, let the user select a supported video.
- Discover same-directory subtitle candidates such as `movie.srt`, `movie.jp.srt`, and `movie.*.srt` while excluding `.raw.srt`, `.vad.srt`, and `.removed.srt` outside debug mode.
- Prefer `movie.srt`, then `movie.jp.srt`, then other matching subtitles.
- Reuse `<video directory>/<stem>_debug/<stem>.wav` when available.
- If WAV is missing, ask whether to generate it.
- If the user declines, show the subtitle list, time ruler, subtitle ranges, and seek position without waveform amplitude or audio playback.
- Show an empty-waveform message that can be clicked later to ask again and start WAV generation.
- After successful generation, refresh waveform and playback in the same window without losing the selected subtitle or current time.

## WAV generation

- Generate 16 kHz mono PCM WAV under `<stem>_debug/<stem>.wav` so normal-media and debug workflows share the cache.
- Run FFmpeg in a background thread and do not initialize Whisper or write recognition subtitles.
- Write to a temporary `.part.wav`, validate it, then atomically replace the final WAV.
- Support cancellation and clean incomplete files when the window closes.
- Detect missing audio streams and damaged/stale WAV files with clear recovery prompts.

## State model

Use explicit states: `EMPTY`, `MEDIA_SELECTED`, `SUBTITLE_ONLY`, `WAV_GENERATING`, `READY`, and `ERROR`.

- `SUBTITLE_ONLY`: subtitle browsing and timeline navigation enabled; playback disabled; waveform click offers generation.
- `WAV_GENERATING`: subtitle browsing remains responsive; duplicate generation disabled.
- `READY`: waveform, playback, seek, cursor, and subtitle following enabled.

## Architecture

- Extend `debug_analyzer.py` with separate `load_debug_session()` and `load_media_file()` paths.
- Represent subtitle dropdown entries as display labels plus absolute paths rather than fixed track strings.
- Make waveform rendering support `audio=None` and render a subtitle-only placeholder timeline.
- Extract source discovery and path rules into an optional `analyzer_source.py` module.
- Add a reusable analysis-WAV extraction function in `logic.py` that is independent of ASR.

## Performance and safety

- Never recursively scan the working directory in media mode; inspect only the selected video's directory and matching stem.
- Run ffprobe, FFmpeg extraction, and waveform preparation off the Tk main thread where appropriate.
- Preserve the local waveform viewport, vectorized peak envelope, visible-only subtitle drawing, and redraw coalescing.
- Use load-generation tokens so late results from a previously selected video are ignored.
- Handle Unicode paths through argument-based subprocess execution rather than shell string composition.

## Delivery phases

1. Add media-source models, subtitle discovery rules, WAV path rules, and state handling.
2. Add source mode and video selection controls.
3. Add subtitle dropdown and subtitle-only timeline mode.
4. Add missing-WAV prompts and waveform-click regeneration.
5. Add asynchronous FFmpeg extraction, cancellation, temporary files, and validation.
6. Refresh waveform/playback in place after generation.
7. Complete Chinese/English/Japanese localization, tests, and packaged-build validation.

## Acceptance criteria

- The analyzer opens and operates independently from subtitle generation.
- Selecting a normal video lists matching `.srt` and `.jp.srt` files.
- Declining WAV generation still permits subtitle browsing and timestamp navigation.
- Clicking the empty waveform can trigger generation later.
- WAV generation keeps the UI responsive and never loads ASR models.
- Successful generation refreshes waveform and playback without reopening the analyzer.
- Existing debug-session raw/VAD/removed analysis remains intact.

