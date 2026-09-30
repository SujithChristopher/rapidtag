"""Detect markers in an image and print their IDs and corner coordinates."""

from __future__ import annotations

import argparse

import cv2
import rapidtag


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--dictionary", default="DICT_APRILTAG_36h11")
    parser.add_argument("--corner-refinement", choices=("none", "subpix", "contour", "apriltag"), default="none")
    args = parser.parse_args()

    image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SystemExit(f"could not read image: {args.image}")

    params = rapidtag.DetectorParameters()
    params.corner_refinement_method = {
        "none": rapidtag.CORNER_REFINE_NONE,
        "subpix": rapidtag.CORNER_REFINE_SUBPIX,
        "contour": rapidtag.CORNER_REFINE_CONTOUR,
        "apriltag": rapidtag.CORNER_REFINE_APRILTAG,
    }[args.corner_refinement]
    corners, ids = rapidtag.detect_markers(image, args.dictionary, params)
    for marker_id, marker_corners in zip(ids, corners):
        print(marker_id, marker_corners)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
