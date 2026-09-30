# Marker detection

RapidTag accepts NumPy `uint8` images in either `H x W` grayscale or `H x W x 3`
BGR layout. Marker corners are returned in top-left, top-right, bottom-right,
bottom-left order.

## One image

```python
import cv2
import rapidtag

image = cv2.imread("frame.png", cv2.IMREAD_GRAYSCALE)
corners, ids = rapidtag.detect_markers(
    image,
    "DICT_APRILTAG_36h11",
)

for marker_id, marker_corners in zip(ids, corners):
    print(marker_id, marker_corners)
```

An unknown dictionary name, unsupported image shape, or unsupported channel
count raises `ValueError`. Call `rapidtag.predefined_dictionaries()` to list the
accepted names.

## Detector parameters

`DetectorParameters` starts with OpenCV-compatible defaults for the implemented
fields:

```python
parameters = rapidtag.DetectorParameters()
parameters.adaptive_thresh_constant = 5.0
parameters.detect_inverted_marker = True
parameters.corner_refinement_method = rapidtag.CORNER_REFINE_APRILTAG

corners, ids = rapidtag.detect_markers(
    image, "DICT_6X6_250", parameters
)
```

The writable properties are listed in the [API reference](python-api.md#detectorparameters).

`corner_refinement_method` accepts `CORNER_REFINE_NONE` (default),
`CORNER_REFINE_SUBPIX`, `CORNER_REFINE_CONTOUR`, and `CORNER_REFINE_APRILTAG`.
`SUBPIX` refines decoded marker corners with a local gradient search; its window
is capped by `corner_refinement_win_size` and scaled by
`relative_corner_refinement_win_size`. `CONTOUR` fits lines to threshold contour
pixels. `APRILTAG` runs the AprilTag 2 quad candidate pipeline: tiled local
thresholds, connected components, black/white boundary clusters, and
gradient-weighted four-line fitting. These quads use RapidTag's usual dictionary
decoding. The marker dictionary can be ArUco or AprilTag with any refinement
mode. Implementation details and resize/blur behavior may produce small
differences from OpenCV.

AprilTag-specific controls are `april_tag_quad_decimate` (default `0`, disabled),
`april_tag_quad_sigma` (`0`, no blur; negative sharpens),
`april_tag_min_cluster_pixels` (`5`), `april_tag_max_nmaxima` (`10`),
`april_tag_critical_rad` (10 degrees in radians),
`april_tag_max_line_fit_mse` (`10`), `april_tag_min_white_black_diff` (`5`),
and `april_tag_deglitch` (`False`). They affect the AprilTag candidate path only.

## Multiple cameras or frames

Use one batch call rather than a Python thread per camera:

```python
results = rapidtag.detect_markers_batch(
    [camera_0_frame, camera_1_frame],
    "DICT_APRILTAG_36h11",
)

for corners, ids in results:
    print(ids)
```

The output order matches the input frame order. RapidTag releases the Python GIL
and distributes frame/threshold-scale work through one Rayon pool.

## ChArUco

Construct the board once and reuse it:

```python
board = rapidtag.CharucoBoard(
    7, 5,
    square_length=0.04,
    marker_length=0.02,
    dictionary="DICT_4X4_50",
)

charuco_corners, charuco_ids, marker_corners, marker_ids = (
    rapidtag.detect_charuco_board(image, board)
)
```

Lengths are in any consistent physical unit. The same unit will be used by
translations returned from board pose estimation.

## Generic chessboards

`pattern_size` is `(columns, rows)` of interior corners:

```python
found, corners = rapidtag.find_chessboard_corners(image, (9, 6))
```

Corners are returned in row-major order after sub-pixel refinement.
