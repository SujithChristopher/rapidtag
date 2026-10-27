"""Edge, grid, and track refinement on a generated AprilTag with known corners."""

import cv2
import numpy as np
import pytest

import rapidtag

DICT = "DICT_APRILTAG_36h11"
# 160 px marker (8 cells of 20 px) at (70, 50): edges lie between pixel centres
EXPECTED = np.array([[69.5, 49.5], [229.5, 49.5], [229.5, 209.5], [69.5, 209.5]])
INWARD = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], float)


@pytest.fixture(scope="module")
def image():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    img = np.full((260, 300), 255, dtype=np.uint8)
    img[50:210, 70:230] = cv2.aruco.generateImageMarker(dictionary, 12, 160)
    return cv2.GaussianBlur(img, (7, 7), 1.0)


def test_refine_edges_recovers_true_edges(image):
    shrunk = EXPECTED + 1.2 * INWARD  # contour-style corners, inside the true edge
    corners, ok = rapidtag.refine_edges(image, [shrunk])
    assert ok == [True]
    assert corners.shape == (1, 4, 2)
    assert np.abs(corners[0] - EXPECTED).max() < 0.05


def test_refine_grid_recovers_true_corners(image):
    corners, ok = rapidtag.refine_grid(image, [EXPECTED + 0.4 * INWARD], [12])
    assert ok == [True]
    assert np.abs(corners[0] - EXPECTED).max() < 0.05


def test_failed_refinement_keeps_input():
    blank = np.full((260, 300), 128, dtype=np.uint8)
    corners, ok = rapidtag.refine_edges(blank, [EXPECTED])
    assert ok == [False]
    assert np.array_equal(corners[0], EXPECTED)
    corners, ok = rapidtag.refine_grid(blank, [EXPECTED], [12])
    assert ok == [False]
    assert np.array_equal(corners[0], EXPECTED)


def test_opencv_shaped_inputs(image):
    cv_corners = (EXPECTED.reshape(1, 4, 2).astype(np.float32),)
    cv_ids = np.array([[12]], dtype=np.int32)
    corners, ok = rapidtag.refine_grid(image, cv_corners, cv_ids)
    assert ok == [True]
    empty, ok = rapidtag.refine_edges(image, None)
    assert empty.shape == (0, 4, 2) and ok == []


def test_pattern_score(image):
    assert rapidtag.pattern_score(image, EXPECTED, 12) > 0.95
    for rot in (1, 2, 3):
        assert rapidtag.pattern_score(image, np.roll(EXPECTED, -rot, 0), 12) < 0.5
    assert rapidtag.pattern_score(image, EXPECTED, 4) < 0.5
    assert rapidtag.pattern_score(image, EXPECTED + [200, 0], 12) == -1.0


def test_multistage_refinds_missed_tag_within_gap(image):
    ms = rapidtag.MultiStageRefiner(DICT, marker_ids=[12], max_gap=4)
    corners, ids, how = ms.refine(image, [EXPECTED + 1.2 * INWARD], [12], sample=0)
    assert ids == [12] and how == [0]
    assert np.abs(corners[0] - EXPECTED).max() < 0.05

    corners, ids, how = ms.refine(image, None, None, sample=2)  # detector missed it
    assert ids == [12] and how == [1]
    assert np.abs(corners[0] - EXPECTED).max() < 0.05

    _, ids, _ = ms.refine(image, None, None, sample=7)  # last seen 5 samples ago
    assert ids == []

    ms.refine(image, [EXPECTED], [12], sample=8)
    ms.reset()
    _, ids, _ = ms.refine(image, None, None, sample=9)
    assert ids == []


def test_multistage_tracks_in_full_frame_coordinates(image):
    ms = rapidtag.MultiStageRefiner(DICT, marker_ids=[12])
    ms.refine(image, [EXPECTED], [12], sample=0)
    crop = np.ascontiguousarray(image[20:, 30:])  # the crop origin moves; tracks stay put
    corners, ids, how = ms.refine(crop, None, None, sample=1, offset=(30.0, 20.0))
    assert ids == [12] and how == [1]
    assert np.abs(corners[0] + [30, 20] - EXPECTED).max() < 0.05


def test_multistage_ignores_other_ids(image):
    ms = rapidtag.MultiStageRefiner(DICT, marker_ids=[4])
    _, ids, _ = ms.refine(image, [EXPECTED], [12], sample=0)
    assert ids == []
