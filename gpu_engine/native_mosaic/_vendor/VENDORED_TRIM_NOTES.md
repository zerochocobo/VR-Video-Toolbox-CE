# Vendored lada — Trim Notes (AGPL modification record)

This directory vendors lada (AGPL-3.0; license in `LICENSE.lada.md`). VR_Video_Toolbox uses
only the **basicvsrpp restoration + yolo11 detection** path, so unused subtrees were **removed**
from the upstream copy to shrink the code surface and the AGPL maintenance footprint. The result
is functionally equivalent to the upstream basicvsrpp+yolo path.

## 2026-05-31 trim (removed ~108 files / ~9,500 lines)

After running the full de-mosaic pipeline and capturing the set of modules actually loaded,
the following independent subtrees — **none of which appear in that runtime closure** — were
removed:

| Removed | Reason |
|---|---|
| `lada/gui/` | GTK desktop GUI, unused |
| `lada/cli/` | Command-line entry point (we drive `FrameRestorer` directly, not via the CLI) |
| `lada/locale/` | Translation files |
| `lada/models/dover/` | Video quality assessment (only used by dataset creation) |
| `lada/models/bpjdet/` | Body detection (used for dataset creation) |
| `lada/models/centerface/` | Face detection (used for dataset creation) |
| `lada/models/deepmosaics/` | Alternative restoration model (we hardcode basicvsrpp, this branch never runs) |
| `lada/restorationpipeline/deepmosaics_mosaic_restorer.py` | Same as above |
| `lada/datasetcreation/{detectors/, nsfw_scene_detector.py, nsfw_scene_processor.py}` | Training-dataset creation |

## Kept

- `lada/restorationpipeline/`, `lada/models/basicvsrpp/` (incl. `mmagic/`), `lada/models/yolo/`,
  `lada/utils/`: the core engine, used heavily.
- `lada/datasetcreation/restoration_dataset_metadata.py`: imported by
  `models/basicvsrpp/mosaic_video_dataset.py`, so **kept** (the rest of datasetcreation is removed).

## Code changes (removing dead references to deleted modules)

- `restorationpipeline/__init__.py` `load_models()`: removed the `deepmosaics` branch (never taken),
  kept basicvsrpp.
- `restorationpipeline/frame_restorer.py` `_restore_clip_frames()`: removed the `deepmosaics` branch.

## Verification

After the trim, the full pipeline was re-run: engine import + model load succeeded; de-mosaic
output was correct and identical to before the trim (matching input/output RGB means, no green
corruption, mean|A−B| = 1.2); `sys.modules` contained none of the deleted modules.

## How to restore

If a removed feature is needed (e.g. deepmosaics / dataset creation / GUI), re-vendor the
corresponding subtree from the matching upstream lada version.
