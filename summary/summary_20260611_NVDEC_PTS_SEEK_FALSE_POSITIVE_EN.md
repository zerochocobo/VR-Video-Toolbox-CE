# NVDEC PTS Seek False Positive Summary

Date: 2026-06-11

## Symptom

The one-click demosaic workflow failed during GPU extract when creating `PyNvThreadedSerialDecoder(start_frame=1770)`:

```text
PyNvThreadedSerialDecoder NVDEC seek check failed:
start_frame=1770,
expected_pts=2657696,
got_pts=2660666,
delta=2970,
decode_start_frame=1554
```

The failure happened in the NVDEC seek self-check and stopped the pipeline.

After the first fix for the PTS origin offset, a later fine extract segment hit the same self-check:

```text
start_frame=2700,
expected_pts=4054092,
got_pts=4057148,
delta=3056,
normalized_delta=86,
pts_origin_delta=2970,
decode_start_frame=2514
```

After subtracting the container PTS origin, only an `86` tick residual remained.

## Conclusion

This was not caused by an oversized slice, CUDA not being used, or the quality preset. The root cause was a PTS origin mismatch between two PyNv decode paths, plus a sub-frame residual in some B-frame / timestamp-rounding cases:

- `SimpleDecoder` target-frame PTS: `2657696`
- `ThreadedDecoder` target-frame PTS: `2660666`
- Delta: `2970`
- Container first-frame PTS: `2970`

`ThreadedDecoder` preserved the container timeline offset, while `SimpleDecoder` random access returned PTS normalized to zero. After subtracting that offset, some frames can still have a timestamp residual far below one frame. The old check required exact absolute PTS equality, so it rejected a correct frame as a seek mismatch.

## Evidence

The real probe and prior frame comparison showed:

- Container first-frame PTS was `2970`
- `got_pts - expected_pts` was also `2970`
- The target frame content from both decode paths was identical
- Pixel difference was `max=0`, `mean=0.0`
- The second failure point had a normalized residual of `86` ticks; the adjacent-frame PTS step was about `1501` ticks, so the residual was about `5.7%` of one frame
- The second failure point also had identical Simple/Threaded image content, with pixel difference `max=0`, `mean=0.0`

So this was a false positive caused by different PTS baselines, not an actual content-frame mismatch.

## Fix

Updated `gpu_engine/pynv_io.py`:

1. Added first-frame container PTS probing through `ffprobe` using `best_effort_timestamp` / `pts`.
2. During `PyNvThreadedSerialDecoder` initialization, read `SimpleDecoder.frame_at(0).pts` and compute:

```text
pts_origin_delta = container_first_pts - simple_decoder_first_pts
```

3. During initial preroll batch calibration, map `actual_pts - pts_origin_delta` back to the SimpleDecoder frame index first.
4. Estimate the local one-frame PTS step from adjacent target-frame PTS values and allow residuals within `10%` of that frame interval.
5. During first target-frame seek verification, accept:

```text
actual_pts == expected_pts
actual_pts - pts_origin_delta == expected_pts
abs((actual_pts - pts_origin_delta) - expected_pts) <= pts_tolerance
```

6. If normalized PTS still mismatches after that small residual allowance, keep raising `NVDEC seek check failed` and include `normalized_delta`, `pts_origin_delta`, and `pts_tolerance` in the error.

The main path was not switched directly to `SimpleDecoder`: extract reads long contiguous frame ranges, where `ThreadedDecoder` is the fast path. `SimpleDecoder` is better suited for random sampling, and replacing the main path would significantly slow long clips. The fix keeps the fast path and makes the self-check compare on the same PTS timeline with a small sub-frame residual allowance.

## Verification

Unit and caller-side regression tests:

```text
python -m pytest tests/test_pynv_io_preroll.py -q
8 passed

.venv\Scripts\python.exe -m pytest tests/test_pynv_io_preroll.py tests/test_gpu_extract_multi.py tests/test_gpu_fisheye_patch.py tests/test_source_time_scanner.py tests/test_one_click_pre_extract.py tests/test_segment_paster.py -q
50 passed, 1 subtests passed
```

Real PyNv/NVDEC probe:

```text
idx=1770 ok frame_pts=2660666 pts_origin_delta=2970 normalized_delta=0 pts_tolerance=150 decode_start_frame=1554
idx=2700 ok frame_pts=4057148 pts_origin_delta=2970 normalized_delta=86 pts_tolerance=150 decode_start_frame=2514
```

The same reproduction path now passes the first-frame seek check. A short real extract smoke around the failing start point also completed the actual `extract_transformed_rect_clip` encode and mux path.

## Remaining Risk

- The fix depends on `ffprobe` being able to read the first-frame PTS; if probing fails, behavior falls back to the old strict check.
- The change only accepts a stable PTS-origin offset and a sub-frame residual far below a full frame. Real seek mismatches close to or above one frame still fail.
- The system Python environment lacks `PyNvVideoCodec`; real NVDEC probes must use the project `.venv`.
