"""Estimate a synthetic two-marker rigid-body pose."""

import rapidtag


tag_size = 0.05
half = tag_size / 2.0
marker_ids = [1, 2]
rotations = [
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
]
translations = [[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0]]
body = rapidtag.RigidBody(tag_size, marker_ids, rotations, translations)

local_corners = [
    [-half, half, 0.0],
    [half, half, 0.0],
    [half, -half, 0.0],
    [-half, -half, 0.0],
]
object_points = [
    [point[0] + translation[0], point[1], point[2]]
    for translation in translations
    for point in local_corners
]

camera_matrix = [
    [800.0, 0.0, 640.0],
    [0.0, 800.0, 400.0],
    [0.0, 0.0, 1.0],
]
projected = rapidtag.project_points(
    object_points,
    [0.15, -0.08, 0.04],
    [0.01, -0.02, 0.65],
    camera_matrix,
)
marker_corners = [projected[:4], projected[4:]]

pose = rapidtag.estimate_rigid_body_pose(
    marker_corners,
    marker_ids,
    body,
    camera_matrix,
    seed=7,
)
if pose is None:
    raise SystemExit("pose solve failed")

print("rvec:", pose.rvec)
print("tvec:", pose.tvec)
print("inlier markers:", pose.inlier_marker_ids)
print("reprojection RMSE:", pose.reprojection_rmse)
