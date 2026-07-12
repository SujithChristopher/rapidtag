# fasttag

A pure-Rust port of OpenCV's **aruco** marker detection, exposed to Python via
[maturin](https://www.maturin.rs/) / [PyO3](https://pyo3.rs/). No OpenCV runtime
dependency — the detection pipeline and predefined dictionaries are reimplemented
in Rust on top of `image` / `imageproc` / `nalgebra`.

## Status

**v1 — marker detection** (`detectMarkers`, `CORNER_REFINE_NONE`). Verified
against OpenCV 5.0: pixel-perfect corner agreement on clean scenes and identical
detection sets under perspective warp (mean corner error ≈ 0.05 px).

Ported faithfully from `opencv/modules/objdetect/src/aruco/`:

- Adaptive-threshold candidate detection across window-size scales
- Contour → polygon approximation → convex-square filtering
- Near-duplicate candidate grouping (`filterTooCloseCandidates`)
- Perspective removal + Otsu bit extraction (`_extractCellPixelRatio`)
- Border-error rejection and dictionary identification via Hamming distance
  over the 4 rotations (`CellBitMasks`)

Supported dictionaries: all `DICT_{4,5,6,7}X{4,5,6,7}_{50,100,250,1000}`,
`DICT_ARUCO_ORIGINAL`, `DICT_ARUCO_MIP_36h12`.

Not yet ported (future scope): corner sub-pixel refinement, pose estimation
(`solvePnP`), boards, ChArUco, AprilTag dictionaries.

## Build

```bash
# dev install into the current virtualenv
maturin develop --release

# or build a wheel
maturin build --release
```

## Usage

```python
import cv2            # only needed to load/generate images
import fasttag

img = cv2.imread("scene.png")          # HxWx3 BGR, or HxW grayscale uint8
corners, ids = fasttag.detect_markers(img, "DICT_6X6_250")
# corners: list of 4x2 [(x, y), ...] per marker (clockwise)
# ids:     list of marker ids, aligned with corners

# Tunable parameters (same names/defaults as cv2.aruco.DetectorParameters):
p = fasttag.DetectorParameters()
p.adaptive_thresh_constant = 7.0
p.detect_inverted_marker = True
corners, ids = fasttag.detect_markers(img, "DICT_6X6_250", p)

print(fasttag.predefined_dictionaries())   # list supported dictionary names
```

## Regenerating dictionary tables

The byte tables in `src/dictionaries_data.rs` are generated from OpenCV's
`predefined_dictionaries.hpp`:

```bash
python3 gen_dicts.py
```

## Tests

```bash
python tests/crosscheck.py     # cross-validate against cv2.aruco
```
