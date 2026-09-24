# Python API reference

This page documents supported public names. Names beginning with `_` are test
hooks and may change without notice. Sequence inputs accept normal Python nested
sequences; image inputs must be NumPy `uint8` arrays.

## Detection

### `predefined_dictionaries()`

Returns `list[str]` containing every accepted predefined dictionary name.

### `detect_markers(image, dictionary, parameters=None)`

Detects ArUco or AprilTag markers in an `H x W` grayscale or `H x W x 3` BGR
`uint8` image. Returns `(corners, ids)`, where each marker contains four `(x, y)`
corners in clockwise top-left-first order.

### `detect_markers_batch(images, dictionary, parameters=None)`

Detects markers in several images through one parallel work pool. Returns one
`(corners, ids)` pair per input image, preserving input order.

### `DetectorParameters`

Construct with `DetectorParameters()`. These properties are writable:

| Property | Type | Default |
|---|---:|---:|
| `adaptive_thresh_win_size_min` | `int` | `3` |
| `adaptive_thresh_win_size_max` | `int` | `23` |
| `adaptive_thresh_win_size_step` | `int` | `10` |
| `adaptive_thresh_constant` | `float` | `7.0` |
| `polygonal_approx_accuracy_rate` | `float` | `0.03` |
| `error_correction_rate` | `float` | `0.6` |
| `detect_inverted_marker` | `bool` | `False` |
| `min_side_length_canonical_img` | `int` | `32` |

## ChArUco and chessboards

### `CharucoBoard(squares_x, squares_y, square_length, marker_length, dictionary, ids=None, legacy_pattern=False)`

Precomputes ChArUco geometry. `ids`, when provided, assigns marker IDs in board
layout order. Read-only properties are `chessboard_corners` and `ids`.

### `detect_charuco_board(image, board, parameters=None, min_markers=2, check_markers=True)`

Returns `(charuco_corners, charuco_ids, marker_corners, marker_ids)`. The first
two arrays identify sub-pixel board intersections; the latter two contain the
markers used to locate them. `min_markers` must be 0, 1, or 2.

### `find_chessboard_corners(image, pattern_size)`

Returns `(found, corners)`. `pattern_size` is `(columns, rows)` of interior
corners and both dimensions must exceed two.

### `estimate_pose_charuco_board(charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs=None)`

Returns `(rvec, tvec)` or `None`. IDs index `board.chessboard_corners`. At least
four non-collinear corners are required.

## General pose

### `solve_pnp(object_points, image_points, camera_matrix, dist_coeffs=None, rvec=None, tvec=None)`

Runs iterative PnP and returns `(rvec, tvec)` or `None`. Inputs are corresponding
`N x 3` object points and `N x 2` pixel points. Optional `rvec` and `tvec` must be
provided together.

### `solve_pnp_ransac(object_points, image_points, camera_matrix, dist_coeffs=None, iterations=100, reprojection_error=3.0, confidence=0.99, seed=0)`

Returns `(rvec, tvec, inlier_indices, reprojection_rmse)` or `None`. Planar input
requires at least five correspondences; general 3D input requires six.
`reprojection_error` and RMSE are pixels. `confidence` must be between zero and
one, exclusive.

### `project_points(object_points, rvec, tvec, camera_matrix, dist_coeffs=None)`

Projects `N x 3` object points and returns `N x 2` image points.

### `rodrigues(src)`

Converts a 3x1 or 1x3 rotation vector to a 3x3 rotation matrix, or a 3x3 matrix
to a 3x1 vector.

## Rigid bodies

### `RigidBody(tag_size_m, marker_ids, rotations_marker_to_reference, translations_marker_to_reference)`

Precomputes marker corners in a shared reference frame. Rotations have shape
`M x 3 x 3`; translations have shape `M x 3`. All marker arrays must have the
same length. `tag_size_m` and `marker_ids` are read-only properties.

### `estimate_rigid_body_pose(marker_corners, marker_ids, rigid_body, camera_matrix, dist_coeffs=None, iterations=100, reprojection_error=3.0, confidence=0.99, seed=0)`

Returns `RigidBodyPose` or `None`. Unknown marker IDs are ignored. One known
marker uses ordinary PnP; two or more known markers use RANSAC.

### `RigidBodyPose`

Instances are returned by `estimate_rigid_body_pose` and expose read-only
properties:

| Property | Meaning |
|---|---|
| `rvec` | object-reference to camera rotation vector |
| `tvec` | object-reference to camera translation |
| `inlier_indices` | accepted indices in flattened known-marker corner order |
| `inlier_marker_ids` | marker IDs with at least three accepted corners |
| `used_marker_ids` | detected marker IDs known to the rigid body |
| `reprojection_rmse` | inlier reprojection RMSE in pixels |

## Errors and distortion

Invalid dimensions, mismatched lengths, non-finite values, unknown dictionaries,
and invalid parameters raise `ValueError`. Numerical failure to determine a pose
returns `None`.

Distortion accepts 0, 4, 5, 8, 12, or 14 OpenCV-order coefficients. Nonzero
tilted-sensor coefficients (`taux`, `tauy`) are rejected.
