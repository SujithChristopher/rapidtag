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
    assert np.max(np.abs(results["apriltag"] - cv_corners[0].reshape(4, 2))) < 1.0

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


def test_cropped_apriltag_refines_decimated_corners_and_batch():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, 12, 160)
    source = np.float32([[0, 0], [159, 0], [159, 159], [0, 159]])
    target = np.float32([[56, 44], [238, 61], [213, 235], [72, 202]])
    image = cv2.warpPerspective(
        marker, cv2.getPerspectiveTransform(source, target), (300, 280),
        flags=cv2.INTER_LINEAR, borderValue=255,
    )
    image = cv2.GaussianBlur(image, (3, 3), 0.7)

    full = rapidtag.DetectorParameters()
    full.corner_refinement_method = rapidtag.CORNER_REFINE_APRILTAG
    full_corners, full_ids = rapidtag.detect_markers(image, "DICT_APRILTAG_36h11", full)
    assert full_ids == [12]

    cropped = rapidtag.DetectorParameters()
    assert cropped.april_tag_refine_full_resolution is False
    cropped.corner_refinement_method = rapidtag.CORNER_REFINE_APRILTAG
    cropped.april_tag_quad_decimate = 4.0
    coarse_corners, coarse_ids = rapidtag.detect_markers(image, "DICT_APRILTAG_36h11", cropped)
    assert coarse_ids == [12]
    cropped.april_tag_refine_full_resolution = True
    refined_corners, refined_ids = rapidtag.detect_markers(image, "DICT_APRILTAG_36h11", cropped)
    assert refined_ids == [12]
    assert np.max(np.abs(np.asarray(refined_corners) - np.asarray(full_corners))) < 0.05
    assert np.max(np.abs(np.asarray(coarse_corners) - np.asarray(full_corners))) > 0.05

    batch = rapidtag.detect_markers_batch([image, image], "DICT_APRILTAG_36h11", cropped)
    assert [ids for _, ids in batch] == [[12], [12]]
    assert np.allclose(batch[0][0], refined_corners)
    assert np.allclose(batch[1][0], refined_corners)


def test_crop_refinement_cannot_recover_tag_missed_by_coarse_search():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, 12, 28)
    image = np.full((160, 160), 255, dtype=np.uint8)
    image[65:93, 66:94] = marker
    params = rapidtag.DetectorParameters()
    params.corner_refinement_method = rapidtag.CORNER_REFINE_APRILTAG
    params.april_tag_refine_full_resolution = True
    params.april_tag_quad_decimate = 3.0
    assert rapidtag.detect_markers(image, "DICT_APRILTAG_36h11", params)[1] == [12]
    params.april_tag_quad_decimate = 4.0
    assert rapidtag.detect_markers(image, "DICT_APRILTAG_36h11", params)[1] == []


def test_crop_refinement_matches_multiple_marker_ids():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    image = np.full((280, 500), 255, dtype=np.uint8)
    image[40:200, 30:190] = cv2.aruco.generateImageMarker(dictionary, 12, 160)
    image[60:220, 300:460] = cv2.aruco.generateImageMarker(dictionary, 13, 160)
    full = rapidtag.DetectorParameters()
    full.corner_refinement_method = rapidtag.CORNER_REFINE_APRILTAG
    full_corners, full_ids = rapidtag.detect_markers(image, "DICT_APRILTAG_36h11", full)
    cropped = rapidtag.DetectorParameters()
    cropped.corner_refinement_method = rapidtag.CORNER_REFINE_APRILTAG
    cropped.april_tag_quad_decimate = 4.0
    cropped.april_tag_refine_full_resolution = True
    cropped_corners, cropped_ids = rapidtag.detect_markers(image, "DICT_APRILTAG_36h11", cropped)
    assert set(cropped_ids) == set(full_ids) == {12, 13}
    by_id = dict(zip(full_ids, full_corners))
    for marker_id, corners in zip(cropped_ids, cropped_corners):
        assert np.max(np.abs(np.asarray(corners) - by_id[marker_id])) < 0.05


def test_aruco_search_with_crop_refinement_matches_apriltag_corners():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    image = np.full((280, 500), 255, dtype=np.uint8)
    image[40:200, 30:190] = cv2.aruco.generateImageMarker(dictionary, 12, 160)
    marker = cv2.aruco.generateImageMarker(dictionary, 13, 160)
    source = np.float32([[0, 0], [159, 0], [159, 159], [0, 159]])
    target = np.float32([[290, 50], [470, 70], [450, 250], [300, 220]])
    warped = cv2.warpPerspective(
        marker, cv2.getPerspectiveTransform(source, target), (500, 280),
        flags=cv2.INTER_LINEAR, borderValue=255,
    )
    image[:, 250:] = warped[:, 250:]
    image = cv2.GaussianBlur(image, (3, 3), 0.7)

    full = rapidtag.DetectorParameters()
    full.corner_refinement_method = rapidtag.CORNER_REFINE_APRILTAG
    full_corners, full_ids = rapidtag.detect_markers(image, "DICT_APRILTAG_36h11", full)
    by_id = dict(zip(full_ids, np.asarray(full_corners)))

    params = rapidtag.DetectorParameters()
    params.april_tag_refine_full_resolution = True
    corners, ids = rapidtag.detect_markers(image, "DICT_APRILTAG_36h11", params)
    assert set(ids) == set(full_ids) == {12, 13}
    for marker_id, c in zip(ids, corners):
        assert np.max(np.abs(np.asarray(c) - by_id[marker_id])) < 0.05

    batch = rapidtag.detect_markers_batch([image, image], "DICT_APRILTAG_36h11", params)
    for batch_corners, batch_ids in batch:
        assert batch_ids == ids
        assert np.allclose(batch_corners, corners)


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
    assert np.max(np.abs(np.asarray(corners[0]) - cv_corners[0].reshape(4, 2))) < 1.0
