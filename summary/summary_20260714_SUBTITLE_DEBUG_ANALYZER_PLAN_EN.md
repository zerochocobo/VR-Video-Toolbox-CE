# Subtitle Debug Analyzer Development Plan

Date: 2026-07-14

## Feasibility

The feature is feasible. The current subtitle pipeline already produces most required inputs: a 16 kHz mono WAV, final subtitles, raw ASR output, VAD chunks, and removed subtitle diagnostics. The analyzer should be a separate Tkinter `Toplevel` window exposed only when debug output is enabled, so the normal workflow remains unchanged.

## Planned scope

- Show a “Debug Subtitles” button beside the debug checkbox only while debug mode is enabled.
- Open a resizable full analyzer window with a subtitle list in the upper area and a wide horizontal waveform in the lower area.
- Display waveform amplitude, time ruler, subtitle regions, playback cursor, and the active subtitle text.
- Provide play/pause, stop, previous/next subtitle, fit-to-duration, zoom, and horizontal scrolling.
- Clicking a subtitle list entry or subtitle block seeks and centers the matching waveform region.
- During playback, advance the cursor and automatically follow/highlight the active subtitle.

## Debug artifacts

Keep each task in the existing `<video>_debug` directory and add `<video>.wav` beside the existing `.raw.srt`, `.vad.srt`, and `.removed.srt` files. The current transcription method always deletes `.asr.wav`; it must instead retain/move the WAV only in debug mode while preserving existing cleanup in normal mode.

## Implementation design

- Add `tool_subtitle/debug_analyzer.py` to isolate analyzer state and prevent further growth of `gui.py`.
- Parse SRT with `utf-8-sig` first, supporting BOM, multiline entries, CRLF, and malformed optional entries.
- Read audio through existing NumPy and soundfile dependencies.
- Build cached min/max peak envelopes per display bucket instead of plotting raw samples. Render the waveform, ruler, subtitle blocks, and cursor with Tkinter `Canvas`.
- Run waveform preprocessing off the Tk main thread and marshal UI updates through `after()`.
- Do not use `ffplay` or ship an external player. Implement a small Windows playback backend with Python `ctypes` and the system WinMM `waveOut` API.
- Stream PCM through several short reusable buffers instead of loading/copying the whole recording. Seek with `waveOutReset`, clear queued buffers, and resume from the target sample.
- Use `waveOutPause`, `waveOutRestart`, and `waveOutGetPosition` for pause/resume and device-based progress tracking. Keep all `WAVEHDR` structures and PCM buffers alive until playback completes, then unprepare/reset/close them safely.
- Read the actual WAV format and convert unsupported input to 16-bit PCM in memory with existing soundfile/NumPy facilities. No new executable, DLL, or Python playback dependency is required.
- Keep the player behind a GUI-neutral interface so another platform backend can be added later.
- Add Chinese, English, and Japanese localization strings.

## Delivery phases

1. Implement debug WAV retention and tests for file lifecycle/path resolution.
2. Add the analyzer window, task selection, SRT parser, and subtitle list.
3. Add cached waveform rendering, ruler, subtitle lanes, zoom, and scrolling.
4. Add synchronized selection and seek behavior between list and waveform.
5. Add the WinMM playback backend, pause/stop/seek, device-based cursor progress, and active subtitle following.
6. Complete localization, packaging, cleanup, and error handling.
7. Regress with short/long media, silence, overlapping subtitles, Unicode paths, and incomplete debug artifacts.

## Acceptance criteria

- Debug off: no button and no retained WAV. Debug on: button visible and completed tasks retain an analyzable WAV.
- Final/raw/VAD/removed subtitle tracks load with timing aligned to the waveform.
- Speech produces visible peaks and silence is approximately flat without UI stalls on long media.
- Clicking a subtitle synchronizes the viewport, block highlight, cursor, and bottom subtitle text.
- Playback controls and repeated seeks remain visually synchronized, and closing the window leaves no player process behind.
- Existing generation, cancellation, model cleanup, and non-debug temporary-file behavior do not regress.

## Main risks

- The main native risk is WinMM buffer lifetime management: headers and PCM memory must remain valid until playback completes and must be reset/unprepared before release.
- This backend targets the current Windows product. Linux/macOS support would require a separate backend later.
- Long recordings require peak-envelope caching; drawing raw samples on Canvas is not acceptable.
- The first release should load completed tasks only. Live analysis of a WAV still being written should remain out of scope.
- Subtitle editing and drag-to-retime should be deferred to a later editor phase.

## Expected files

- `tool_subtitle/gui.py`
- `tool_subtitle/logic.py`
- `tool_subtitle/debug_analyzer.py` (new)
- `i18n/zh.json`, `i18n/en.json`, `i18n/ja.json`
- `tool_subtitle/audio_player.py` (new WinMM backend)
- `VR_Video_Toolbox.spec` or related build scripts (PyInstaller compatibility verification only; no player binary expected)
- New parser, waveform mapping, and debug lifecycle tests under `tests/`
