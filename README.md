# RapidTag

> ⚠️ **Work in progress — not production-ready.** RapidTag is under active development
> and pre-1.0. APIs, behavior, and results may change without notice, and it has not been
> hardened or battle-tested for production use. Use it for research, evaluation, and
> prototyping, and validate it thoroughly against your own data before relying on it for
> anything critical.

**Fast, pure-Rust fiducial marker detection for realtime.** RapidTag is a from-scratch
Rust reimplementation of OpenCV's ArUco / AprilTag marker detector, exposed to Python
via [maturin](https://www.maturin.rs/) / [PyO3](https://pyo3.rs/) — **no OpenCV runtime
dependency**. It matches OpenCV's detections bit-for-bit while running faster on realtime
single-frame workloads.

## Why RapidTag

- ⚡ **Faster than OpenCV** for realtime single-frame detection (~370 vs ~330 FPS on
  1280×800 mono; see below), and much faster for multi-camera / offline batches.
- 🎯 **OpenCV-accurate** — validated at **0.0000 px** corner agreement on synthetic,
  perspective-warped, and real camera data.
- 🧵 **Scales with your cores** — a batch API processes many frames (e.g. a stereo pair,
  or a whole recording) across all cores with the GIL released.
- 📦 **No OpenCV needed at runtime** — the detection pipeline and marker dictionaries are
  reimplemented in pure Rust on top of `image` / `nalgebra`.

## Performance

Measured on real OV9281 dual-camera data (1280×800 monochrome, AprilTag 36h11), 24-core CPU:

| Workload | RapidTag | OpenCV (single-thread) |
|----------|---------:|-----------------------:|
| Single-frame (realtime, one camera) | **~370 FPS** (2.7 ms) | ~330 FPS (3.0 ms) |
| Dual-camera pair (both cameras/tick) | **~590 FPS total** (3.4 ms/pair) | — |
| Offline batch (all cores) | **~700 FPS** | — |

Detection parity vs OpenCV: **0.0000 px** mean corner error; identical marker sets
(RapidTag detects a few extra at the margins).

## Status

**v1 — marker detection** (`detectMarkers`, `CORNER_REFINE_NONE`), faithfully ported from
`opencv/modules/objdetect/src/aruco/`:

- Adaptive-threshold candidate detection across window-size scales (sliding-window box sum)
- Suzuki-Abe contour tracing → polygon approximation → convex-square filtering
- Near-duplicate candidate grouping
- Perspective removal + Otsu bit extraction
- Dictionary identification via Hamming distance over the 4 rotations

Supported dictionaries: all `DICT_{4,5,6,7}X{4,5,6,7}_{50,100,250,1000}`,
`DICT_ARUCO_ORIGINAL`, `DICT_ARUCO_MIP_36h12`, and AprilTag
`DICT_APRILTAG_{16h5,25h9,36h10,36h11}`.

Not yet implemented (future): corner sub-pixel refinement, pose estimation (`solvePnP`),
grid boards, ChArUco.

## Install

```bash
pip install rapidtag
```

Prebuilt wheels are published for:

| OS | Architectures | libc |
|----|---------------|------|
| Linux | x86_64, **aarch64 (arm64)** | glibc (manylinux) + musl (Alpine) |
| macOS | x86_64 (Intel), **arm64 (Apple Silicon)** | — |
| Windows | x64 | — |

Wheels are `abi3` (one wheel works on CPython 3.9+). x86 wheels target the portable
`x86-64-v2` baseline (any CPU since ~2009); arm64 wheels use the standard ARMv8 NEON
baseline. If no wheel matches, `pip` builds from the source distribution (needs a Rust
toolchain).

## Build from source

```bash
# dev install into the current virtualenv
maturin develop --release        # always use --release for performance

# or build a wheel
maturin build --release
```

> Note: `.cargo/config.toml` sets `target-cpu=native` for AVX2/SIMD. This makes the wheel
> non-portable to older CPUs; for distribution use `target-cpu=x86-64-v3` instead.

## Usage

```python
import cv2            # only to load/generate images
import rapidtag

img = cv2.imread("scene.png")          # HxWx3 BGR, or HxW grayscale uint8

# --- realtime: one frame ---
corners, ids = rapidtag.detect_markers(img, "DICT_APRILTAG_36h11")
# corners: list of 4x2 [(x, y), ...] per marker (clockwise)
# ids:     list of marker ids, aligned with corners

# --- multi-camera / batch: process many frames across all cores ---
results = rapidtag.detect_markers_batch([cam0, cam1], "DICT_APRILTAG_36h11")
(c0, i0), (c1, i1) = results

# --- tunable parameters (same names/defaults as cv2.aruco.DetectorParameters) ---
p = rapidtag.DetectorParameters()
p.adaptive_thresh_constant = 7.0
p.detect_inverted_marker = True
corners, ids = rapidtag.detect_markers(img, "DICT_6X6_250", p)

print(rapidtag.predefined_dictionaries())   # list supported dictionary names
```

## Regenerating dictionary tables

The byte tables in `src/dictionaries_data.rs` are generated from OpenCV's headers:

```bash
python3 gen_dicts.py
```

## Tests

```bash
python tests/crosscheck.py     # cross-validate vs cv2.aruco on synthetic scenes
python tests/bench.py          # benchmark + parity on real camera data
```
