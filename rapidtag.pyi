from collections.abc import Sequence
from typing import Optional, TypeAlias

import numpy as np
from numpy.typing import NDArray

_Point2: TypeAlias = Sequence[float]
_Point3: TypeAlias = Sequence[float]
_Matrix3: TypeAlias = Sequence[Sequence[float]]
_MarkerCorners: TypeAlias = Sequence[_Point2]
_ImageU8: TypeAlias = NDArray[np.uint8]

class DetectorParameters:
    def __init__(self) -> None: ...
    adaptive_thresh_win_size_min: int
    adaptive_thresh_win_size_max: int
    adaptive_thresh_win_size_step: int
    adaptive_thresh_constant: float
    polygonal_approx_accuracy_rate: float
    error_correction_rate: float
    detect_inverted_marker: bool
    min_side_length_canonical_img: int

class CharucoBoard:
    def __init__(
        self,
        squares_x: int,
        squares_y: int,
        square_length: float,
        marker_length: float,
        dictionary: str,
        ids: Optional[Sequence[int]] = None,
        legacy_pattern: bool = False,
    ) -> None: ...
    @property
    def chessboard_corners(self) -> list[list[float]]: ...
    @property
    def ids(self) -> list[int]: ...

class RigidBody:
    def __init__(
        self,
        tag_size_m: float,
        marker_ids: Sequence[int],
        rotations_marker_to_reference: Sequence[_Matrix3],
        translations_marker_to_reference: Sequence[_Point3],
    ) -> None: ...
    @property
    def tag_size_m(self) -> float: ...
    @property
    def marker_ids(self) -> list[int]: ...

class RigidBodyPose:
    @property
    def rvec(self) -> list[float]: ...
    @property
    def tvec(self) -> list[float]: ...
    @property
    def inlier_indices(self) -> list[int]: ...
    @property
    def inlier_marker_ids(self) -> list[int]: ...
    @property
    def used_marker_ids(self) -> list[int]: ...
    @property
    def reprojection_rmse(self) -> float: ...

def predefined_dictionaries() -> list[str]: ...
def detect_markers(
    image: _ImageU8,
    dictionary: str,
    parameters: Optional[DetectorParameters] = None,
) -> tuple[list[list[list[float]]], list[int]]: ...
def detect_markers_batch(
    images: Sequence[_ImageU8],
    dictionary: str,
    parameters: Optional[DetectorParameters] = None,
) -> list[tuple[list[list[list[float]]], list[int]]]: ...
def detect_charuco_board(
    image: _ImageU8,
    board: CharucoBoard,
    parameters: Optional[DetectorParameters] = None,
    min_markers: int = 2,
    check_markers: bool = True,
) -> tuple[list[list[float]], list[int], list[list[list[float]]], list[int]]: ...
def find_chessboard_corners(
    image: _ImageU8,
    pattern_size: tuple[int, int],
) -> tuple[bool, list[list[float]]]: ...
def solve_pnp(
    object_points: Sequence[_Point3],
    image_points: Sequence[_Point2],
    camera_matrix: _Matrix3,
    dist_coeffs: Optional[Sequence[float]] = None,
    rvec: Optional[_Point3] = None,
    tvec: Optional[_Point3] = None,
) -> Optional[tuple[list[float], list[float]]]: ...
def solve_pnp_ransac(
    object_points: Sequence[_Point3],
    image_points: Sequence[_Point2],
    camera_matrix: _Matrix3,
    dist_coeffs: Optional[Sequence[float]] = None,
    iterations: int = 100,
    reprojection_error: float = 3.0,
    confidence: float = 0.99,
    seed: int = 0,
) -> Optional[tuple[list[float], list[float], list[int], float]]: ...
def estimate_rigid_body_pose(
    marker_corners: Sequence[_MarkerCorners],
    marker_ids: Sequence[int],
    rigid_body: RigidBody,
    camera_matrix: _Matrix3,
    dist_coeffs: Optional[Sequence[float]] = None,
    iterations: int = 100,
    reprojection_error: float = 3.0,
    confidence: float = 0.99,
    seed: int = 0,
) -> Optional[RigidBodyPose]: ...
def project_points(
    object_points: Sequence[_Point3],
    rvec: _Point3,
    tvec: _Point3,
    camera_matrix: _Matrix3,
    dist_coeffs: Optional[Sequence[float]] = None,
) -> list[list[float]]: ...
def rodrigues(src: Sequence[Sequence[float]]) -> list[list[float]]: ...
def estimate_pose_charuco_board(
    charuco_corners: Sequence[_Point2],
    charuco_ids: Sequence[int],
    board: CharucoBoard,
    camera_matrix: _Matrix3,
    dist_coeffs: Optional[Sequence[float]] = None,
) -> Optional[tuple[list[float], list[float]]]: ...
