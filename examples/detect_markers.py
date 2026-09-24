"""Detect markers in an image and print their IDs and corner coordinates."""

from __future__ import annotations

import argparse

import cv2
import rapidtag


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--dictionary", default="DICT_APRILTAG_36h11")
    args = parser.parse_args()

    image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SystemExit(f"could not read image: {args.image}")

    corners, ids = rapidtag.detect_markers(image, args.dictionary)
    for marker_id, marker_corners in zip(ids, corners):
        print(marker_id, marker_corners)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
