# Microalgae Identification and Tracking Source Code

This repository provides the core Python modules for computer-vision-based
microalgal identification, morphological filtering, split recovery, object
tracking, and quality-control visualization described in the associated
article.

The Python modules are the same versions supplied with the accepted manuscript
during peer review.

The repository contains no sample images, experimental datasets, deep-learning
models, or graphical user interface.

## Associated article

This source code accompanies the published article
["Motility-based viability assay for screening acid-tolerant microalgae"](https://doi.org/10.1016/j.biortech.2026.135644)
in *Bioresource Technology*.

**Citation**

Jeong, S.-G., Choi, Y.Y., Kang, S.-M., Lee, E.-H., Choi, H.Y., Yoon, S.,
Park, S.J., Kim, B.-G., 2027. Motility-based viability assay for screening
acid-tolerant microalgae. *Bioresource Technology* **463**, 135644.
[https://doi.org/10.1016/j.biortech.2026.135644](https://doi.org/10.1016/j.biortech.2026.135644)

## Code author and affiliation

Seong-Geun Jeong  
Seoul National University, Republic of Korea

## Contents

### `microalgae_identification.py`

This module performs grayscale conversion, adaptive thresholding, contour
extraction, enclosing-circle and ellipse fitting, morphological filtering,
image-based artifact rejection, faint-object recovery, and split recovery for
eligible adjacent cell-like objects.

The primary function is:

```python
identify_microalgae(image)
```

It returns `detections`, `split_detections`, and `diagnostics`. The base
geometry-filtering stage can be called separately using
`identify_base_cv(image)`.

### `microalgae_tracking.py`

This module assigns persistent object IDs, links detections between consecutive
frames, relinks temporarily missed objects, calculates displacement and
velocity, and exports object-level tracking results to CSV.

The main interfaces are:

```python
CentroidTracker
TrackingParams
identify_and_track_frame
tracking_rows
write_tracking_csv
```

### `microalgae_visualization.py`

This optional module draws detected contours, enclosing circles, fitted
ellipses, trajectories, and persistent object IDs. It also writes equal-sized
overlay frames as an ImageJ-compatible RGB TIFF stack.

The main interfaces are:

```python
OverlayStyle
draw_tracking_overlay
write_imagej_tiff_stack
```

## Requirements

The modules were verified using Python 3.13.9 with the package versions listed
in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Third-party dependencies

This source code calls OpenCV, NumPy, and tifffile as separately
installed dependencies. It does not include or redistribute their source
code. These packages remain subject to their respective copyright notices and
licenses.

## Input images

Images are supplied as NumPy arrays. Supported inputs are two-dimensional
grayscale arrays and OpenCV-style BGR or BGRA arrays. Frames from a time-lapse
sequence must be passed to the tracker in chronological order, beginning with
frame number zero.

No image resizing is performed internally. Numerical detection and tracking
parameters are expressed in the pixel coordinate system of the input images.

## Minimal example

```python
from pathlib import Path

import cv2

from microalgae_tracking import (
    CentroidTracker,
    TrackingParams,
    identify_and_track_frame,
    tracking_rows,
    write_tracking_csv,
)
from microalgae_visualization import (
    draw_tracking_overlay,
    write_imagej_tiff_stack,
)

image_paths = sorted(Path("images").glob("*.tif"))
tracker = CentroidTracker(TrackingParams(time_interval=0.1))
rows = []
overlays = []

for frame_number, image_path in enumerate(image_paths):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    tracked, split_detections, diagnostics = identify_and_track_frame(
        image=image,
        frame_number=frame_number,
        tracker=tracker,
    )
    rows.extend(
        tracking_rows(
            image_name=image_path.name,
            frame_number=frame_number,
            tracked=tracked,
            time_interval=tracker.time_interval,
        )
    )

    overlays.append(
        draw_tracking_overlay(
            image=image,
            tracked=tracked,
            tracks=tracker.all_tracks,
            frame_number=frame_number,
            output_path=Path("overlays") / f"{frame_number:04d}.png",
        )
    )

write_tracking_csv("tracking_results.csv", rows)
write_imagej_tiff_stack(
    "tracking_overlays_ImageJ_stack.tif",
    overlays,
    frame_interval=tracker.time_interval,
)
```

Identification without tracking can be performed as follows:

```python
import cv2
from microalgae_identification import identify_microalgae

image = cv2.imread("frame_000.tif", cv2.IMREAD_UNCHANGED)
detections, split_detections, diagnostics = identify_microalgae(image)
```

## Output units

- Position, diameter, and fitted-ellipse axes: pixels
- Contour area: pixels squared
- Displacement: pixels
- Velocity: pixels per second
- Time: seconds

Velocity is calculated using `TrackingParams.time_interval`.

## Parameter guide

The default identification values below are the *Dunaliella salina* settings
represented in Fig. 4G of the article. They are operational image-analysis
parameters rather than universal biological limits.

### Identification parameters

| Parameter | Default | Purpose and adjustment |
| --- | ---: | --- |
| `adaptive_block_size` | `251` | Odd-sized neighborhood used for the local mean threshold. Choose a window larger than a typical cell. Decrease it for smaller-scale illumination variation; increase it for broader background gradients. It must be an odd integer greater than 1. |
| `adaptive_c` | `5` | Offset subtracted from the local mean. Increasing it makes dark-foreground selection more stringent and usually reduces background noise. Decreasing it can recover weaker boundaries but may increase false positives. |
| `morph_close_size` | `0` | Elliptical closing kernel. A value of `0` disables closing. Use a small kernel such as `3` or `5` only when cell boundaries are fragmented; excessive closing can merge adjacent cells. |
| `morph_open_size` | `0` | Elliptical opening kernel. A value of `0` disables opening. A small kernel can remove isolated foreground specks, but an excessive value can erode real cells. |
| `diameter_min`, `diameter_max` | `7.5`, `45.5` px | Accepted enclosing-circle diameter range. Raise the lower bound to reject small debris; lower it only when verified small cells are being missed. Adjust the upper bound for the largest complete single cells while excluding oversized aggregates. |
| `major_min`, `major_max` | `7.0`, `45.0` px | Accepted fitted-ellipse major-axis range. Adjust it to the expected cell length at the input image scale. |
| `minor_min`, `minor_max` | `5.0`, `37.0` px | Accepted fitted-ellipse minor-axis range. Raise the lower bound to reject thin fragments; expand the range only after inspecting complete cells. |
| `ratio_max` | `5.0` | Maximum major-to-minor axis ratio. Lower values reject elongated artifacts more strongly; higher values are required for genuinely elongated organisms. |
| `area_min`, `area_max` | `280`, `1150` px^2 | Accepted contour-area range. Set these bounds from complete-cell contours and use them together with the circle and ellipse gates. |

`SplitParams` controls recovery of eligible adjacent two-cell contours.
`FaintRecoveryParams` controls supplementary recovery and quality checks for
faint objects. Change these advanced settings only after calibrating the base
threshold and geometry gates with representative images and, preferably,
manually curated reference objects.

### Tracking parameters

| Parameter | Default | Purpose and adjustment |
| --- | ---: | --- |
| `time_interval` | `0.1` s | Time between consecutive frames. Set this to the actual acquisition interval because velocity is calculated from it. |
| `max_distance` | `36.839` px | Maximum distance for direct linking between consecutive frames. Increase it for faster motion or longer frame intervals; decrease it when nearby cells frequently exchange IDs. |
| `max_missed` | `2` frames | Number of consecutive unmatched frames retained before an active track is removed. Increase it for brief detection dropouts, while checking for incorrect continuation. |
| `relink_max_gap` | `8` frames | Largest frame gap considered by gap-aware relinking. Reduce it when long-gap ID assignments are unreliable. |
| `relink_max_distance` | `97.515` px | Absolute spatial limit for gap-aware relinking. Increase only when verified tracks move farther during detection gaps. |
| `relink_min_diameter_ratio` | `0.55` | Minimum ratio between the smaller and larger object diameters during relinking. Increase it to require stronger size similarity; decrease it only when genuine cells change apparent size substantially. |

### Recommended calibration sequence

1. Keep magnification, image dimensions, illumination, focus, and frame
   interval fixed during calibration and analysis.
2. Adjust `adaptive_block_size` and `adaptive_c` until cell boundaries are
   separated from the local background.
3. Use closing or opening only when boundary gaps or isolated noise cannot be
   addressed by threshold adjustment alone.
4. Measure enclosing-circle diameter, fitted-ellipse axes, axis ratio, and
   contour area from representative complete cells, then set the geometry
   gates to cover those cells while excluding debris and aggregates.
5. Inspect false positives and false negatives before changing
   `FaintRecoveryParams` or `SplitParams`.
6. Set `time_interval` from image metadata and tune `max_distance` from
   observed frame-to-frame displacement. Then evaluate ID switches, broken
   tracks, and relinking errors across the full sequence.
7. Validate the final settings across representative pH conditions, cell
   densities, and imaging sessions before applying them to experimental data.

If an image is resized by a scale factor `s`, multiply diameter, ellipse-axis,
kernel, adaptive-window, and tracking-distance values by approximately `s`;
multiply contour-area bounds by `s^2`. Round the adaptive window to a valid odd
integer greater than 1, and preferably use small odd morphological kernels
when enabled. The dimensionless axis-ratio and diameter-ratio limits do not
require scaling.

Custom settings can be created without modifying the supplied defaults:

```python
from dataclasses import replace

from microalgae_identification import DUNALIELLA_IDENTIFICATION_PARAMS
from microalgae_tracking import CentroidTracker, TrackingParams

custom_detection = replace(
    DUNALIELLA_IDENTIFICATION_PARAMS,
    diameter_min=10.0,
    diameter_max=50.0,
)

tracker = CentroidTracker(
    TrackingParams(
        time_interval=0.1,
        max_distance=40.0,
    )
)
```

Pass `custom_detection` to `identify_and_track_frame` through its
`detection_params` argument.

## License

This source code is distributed under the BSD 3-Clause Clear License. See
`LICENSE` for the complete terms.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
THIS LICENSE.

Copyright (c) 2026 Seong-Geun Jeong.
