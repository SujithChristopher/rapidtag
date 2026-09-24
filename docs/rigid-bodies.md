# Multi-marker rigid bodies

A `RigidBody` combines several square markers whose poses are fixed relative to
one reference frame. Construct it once, then feed each frame's detected marker
corners and IDs into `estimate_rigid_body_pose`.

## Transform convention

For each marker, calibration supplies a rotation and translation satisfying:

```text
p_reference = R_marker_to_reference * p_marker + t_marker_to_reference
```

Translations and `tag_size_m` are in metres. Marker-local corners use this
order:

```text
top-left, top-right, bottom-right, bottom-left
```

`rotation_marker_to_reference` must be a finite orthonormal 3x3 rotation matrix
with determinant +1. Marker IDs must be unique.

## Constructing a body from TOML

The Dome calibration uses this shape:

```toml
[meta]
tag_size_m = 0.05

[markers.1]
rotation_marker_to_reference = [
  [1.0, 0.0, 0.0],
  [0.0, 1.0, 0.0],
  [0.0, 0.0, 1.0],
]
translation_marker_to_reference_m = [0.0, 0.0, 0.0]
```

Load it in Python 3.11+ with:

```python
import tomllib
import rapidtag

with open("rigidbody_calibration.toml", "rb") as file:
    config = tomllib.load(file)

marker_ids = sorted(int(value) for value in config["markers"])
markers = config["markers"]
body = rapidtag.RigidBody(
    config["meta"]["tag_size_m"],
    marker_ids,
    [markers[str(i)]["rotation_marker_to_reference"] for i in marker_ids],
    [markers[str(i)]["translation_marker_to_reference_m"] for i in marker_ids],
)
```

## Estimating pose

```python
corners, ids = rapidtag.detect_markers(
    frame, "DICT_APRILTAG_36h11"
)

pose = rapidtag.estimate_rigid_body_pose(
    corners,
    ids,
    body,
    camera_matrix,
    dist_coeffs,
    iterations=100,
    reprojection_error=3.0,
    confidence=0.99,
    seed=7,
)

if pose is not None:
    print("pose:", pose.rvec, pose.tvec)
    print("visible calibrated markers:", pose.used_marker_ids)
    print("consensus markers:", pose.inlier_marker_ids)
    print("corner inliers:", pose.inlier_indices)
    print("RMSE (px):", pose.reprojection_rmse)
```

Unknown detected IDs are ignored. Duplicate known IDs or non-finite known
corners raise `ValueError`. One known marker uses iterative PnP and reports all
four corners as inliers. Two or more known markers use RANSAC; a marker is an
inlier marker when at least three of its four corners are inliers.

The repository benchmark loads the same calibration for both Dome recordings:

```bash
python scripts/bench_ransac_dome.py --samples 1000 --iterations 100
```
