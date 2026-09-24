# Pose estimation

RapidTag follows the OpenCV camera convention. A pose maps an object-space point
into camera coordinates:

```text
p_camera = R(rvec) * p_object + tvec
```

Image coordinates are pixels. Object points and `tvec` share the same physical
unit; use metres throughout if translations should be metres.

## Camera model

`camera_matrix` is a nested 3x3 sequence:

```python
camera_matrix = [
    [800.0, 0.0, 640.0],
    [0.0, 800.0, 400.0],
    [0.0, 0.0, 1.0],
]
```

`dist_coeffs` may contain 0, 4, 5, 8, 12, or 14 OpenCV-order coefficients:
`k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, taux, tauy`.
Thin-prism terms are supported; nonzero tilted-sensor terms are not.

## Iterative PnP

Use `solve_pnp` for clean 3D-2D correspondences:

```python
pose = rapidtag.solve_pnp(
    object_points,
    image_points,
    camera_matrix,
    dist_coeffs,
)
if pose is not None:
    rvec, tvec = pose
```

At least four points are required. General non-planar initialization requires
at least six points. Supplying both `rvec` and `tvec` seeds refinement; supplying
only one is an error.

## RANSAC PnP

Use RANSAC when matches or detected corners may contain outliers:

```python
pose = rapidtag.solve_pnp_ransac(
    object_points,
    image_points,
    camera_matrix,
    dist_coeffs,
    iterations=100,
    reprojection_error=3.0,
    confidence=0.99,
    seed=7,
)

if pose is not None:
    rvec, tvec, inlier_indices, reprojection_rmse = pose
```

Planar geometry requires at least five points and general 3D geometry requires
six. A four-corner single marker has no redundant point to reject; use ordinary
PnP or the rigid-body API instead. The seed makes RapidTag's sampling
deterministic.

## Projection and Rodrigues conversion

```python
pixels = rapidtag.project_points(
    object_points, rvec, tvec, camera_matrix, dist_coeffs
)
rotation_matrix = rapidtag.rodrigues([[rvec[0], rvec[1], rvec[2]]])
```

Passing a 3x3 rotation matrix to `rodrigues` returns a 3x1 rotation vector.

## ChArUco board pose

```python
pose = rapidtag.estimate_pose_charuco_board(
    charuco_corners,
    charuco_ids,
    board,
    camera_matrix,
    dist_coeffs,
)
```

The function returns `None` for fewer than four usable corners or collinear
board points.
