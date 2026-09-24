"""Estimate a synthetic non-planar pose with ordinary and RANSAC PnP."""

import rapidtag


camera_matrix = [
    [800.0, 0.0, 640.0],
    [0.0, 810.0, 400.0],
    [0.0, 0.0, 1.0],
]
dist_coeffs = [-0.12, 0.03, 0.001, -0.0005, 0.005]
object_points = [
    [-0.10, -0.08, 0.00],
    [0.10, -0.08, 0.00],
    [0.10, 0.08, 0.00],
    [-0.10, 0.08, 0.00],
    [-0.07, -0.05, 0.05],
    [0.07, -0.05, 0.05],
    [0.07, 0.05, 0.05],
    [-0.07, 0.05, 0.05],
]
expected_rvec = [0.20, -0.10, 0.06]
expected_tvec = [0.02, -0.01, 0.75]

image_points = rapidtag.project_points(
    object_points,
    expected_rvec,
    expected_tvec,
    camera_matrix,
    dist_coeffs,
)

print("iterative:", rapidtag.solve_pnp(
    object_points, image_points, camera_matrix, dist_coeffs
))
print("ransac:", rapidtag.solve_pnp_ransac(
    object_points,
    image_points,
    camera_matrix,
    dist_coeffs,
    seed=7,
))
