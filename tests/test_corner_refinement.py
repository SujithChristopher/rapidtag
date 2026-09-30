"""Exercise public corner modes on a generated AprilTag."""

import cv2
import numpy as np
import pytest

import rapidtag


def test_corner_refinement_modes_detect_and_move_corners():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, 12, 160)
    image = np.full((260, 300), 255, dtype=np.uint8)
    image[50:210, 70:230] = marker
    image = cv2.GaussianBlur(image, (5, 5), 0.9)

    results = {}
    for name, method in (
        ("none", rapidtag.CORNER_REFINE_NONE),
        ("subpix", rapidtag.CORNER_REFINE_SUBPIX),
        ("contour", rapidtag.CORNER_REFINE_CONTOUR),
        ("apriltag", rapidtag.CORNER_REFINE_APRILTAG),
    ):
        params = rapidtag.DetectorParameters()
        params.corner_refinement_method = method
        corners, ids = rapidtag.detect_markers(image, "DICT_APRILTAG_36h11", params)
        assert ids == [12], (name, ids)
        results[name] = np.asarray(corners[0])
        assert np.isfinite(results[name]).all()

    # The options must affect measured corner positions, while staying near the
    # known image geometry. This tests the Python API and Rust detector together.
    expected = np.array([[70, 50], [229, 50], [229, 209], [70, 209]])
    for name, corners in results.items():
        assert np.max(np.abs(corners - expected)) < 3.0, name
    assert np.max(np.abs(results["subpix"] - results["none"])) > 0.05
    assert np.max(np.abs(results["apriltag"] - results["none"])) > 0.05

    cv_params = cv2.aruco.DetectorParameters()
    cv_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    cv_corners, cv_ids, _ = cv2.aruco.ArucoDetector(dictionary, cv_params).detectMarkers(image)
    assert cv_ids.flatten().tolist() == [12]
    assert np.max(np.abs(results["apriltag"] - cv_corners[0].reshape(4, 2))) < 3.0

    params = rapidtag.DetectorParameters()
    params.corner_refinement_method = rapidtag.CORNER_REFINE_APRILTAG
    batch = rapidtag.detect_markers_batch([image, image], "DICT_APRILTAG_36h11", params)
    assert [ids for _, ids in batch] == [[12], [12]]
    assert np.allclose(batch[0][0][0], batch[1][0][0])


def test_invalid_corner_method_rejected():
    params = rapidtag.DetectorParameters()
    with pytest.raises(ValueError):
        params.corner_refinement_method = 4


def test_apriltag_candidate_path_recovers_small_marker():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, 12, 28)
    image = np.full((160, 160), 255, dtype=np.uint8)
    image[65:93, 66:94] = marker

    # The default contour path rejects this perimeter. The AprilTag quad path
    # generates its own candidate, matching OpenCV's behavior.
    assert rapidtag.detect_markers(image, "DICT_APRILTAG_36h11")[1] == []
    params = rapidtag.DetectorParameters()
    params.corner_refinement_method = rapidtag.CORNER_REFINE_APRILTAG
    assert rapidtag.detect_markers(image, "DICT_APRILTAG_36h11", params)[1] == [12]

    cv_params = cv2.aruco.DetectorParameters()
    cv_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    cv_ids = cv2.aruco.ArucoDetector(dictionary, cv_params).detectMarkers(image)[1]
    assert cv_ids.flatten().tolist() == [12]


@pytest.mark.parametrize("decimate", [0.0, 2.0])
def test_apriltag_quad_path_on_perspective_and_lighting(decimate):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, 12, 160)
    src = np.float32([[0, 0], [159, 0], [159, 159], [0, 159]])
    dst = np.float32([[56, 44], [238, 61], [213, 235], [72, 202]])
    image = cv2.warpPerspective(
        marker, cv2.getPerspectiveTransform(src, dst), (300, 280),
        flags=cv2.INTER_LINEAR, borderValue=255,
    )
    lighting = np.linspace(0.65, 1.0, image.shape[1], dtype=np.float32)[None, :]
    image = np.clip(image.astype(np.float32) * lighting, 0, 255).astype(np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0.7)

    params = rapidtag.DetectorParameters()
    params.corner_refinement_method = rapidtag.CORNER_REFINE_APRILTAG
    params.april_tag_quad_decimate = decimate
    corners, ids = rapidtag.detect_markers(image, "DICT_APRILTAG_36h11", params)
    assert ids == [12]
    assert np.max(np.abs(np.asarray(corners[0]) - dst)) < 5.0

    cv_params = cv2.aruco.DetectorParameters()
    cv_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    cv_params.aprilTagQuadDecimate = decimate
    cv_corners, cv_ids, _ = cv2.aruco.ArucoDetector(dictionary, cv_params).detectMarkers(image)
    assert cv_ids.flatten().tolist() == [12]
    assert np.max(np.abs(np.asarray(corners[0]) - cv_corners[0].reshape(4, 2))) < 5.0
