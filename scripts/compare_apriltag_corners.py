"""Compare RapidTag and OpenCV AprilTag detections on recorded stereo frames."""

from __future__ import annotations

import argparse
from itertools import islice

import cv2
import msgpack
import msgpack_numpy
import numpy as np

import rapidtag
from bench_corner_refinement import DATA, DICTIONARY


def load_sampled(camera: str, count: int, stride: int) -> list[np.ndarray]:
    path = DATA / f"{camera}_frame.msgpack"
    with path.open("rb") as source:
        unpacker = msgpack.Unpacker(source, object_hook=msgpack_numpy.decode, raw=False)
        return [np.ascontiguousarray(frame) for frame in islice(unpacker, 0, count * stride, stride)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--decimate", type=float, default=0.0)
    parser.add_argument("--opencv-decimate", type=float)
    args = parser.parse_args()
    if args.frames < 1 or args.stride < 1:
        parser.error("--frames and --stride must be positive")

    params = rapidtag.DetectorParameters()
    params.corner_refinement_method = rapidtag.CORNER_REFINE_APRILTAG
    params.april_tag_quad_decimate = args.decimate
    cv_params = cv2.aruco.DetectorParameters()
    cv_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    cv_decimate = args.decimate if args.opencv_decimate is None else args.opencv_decimate
    cv_params.aprilTagQuadDecimate = cv_decimate
    cv_detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), cv_params
    )

    distances: list[float] = []
    disagreements = []
    frame_count = 0
    for camera in ("cam0", "cam1"):
        for frame_number, image in enumerate(load_sampled(camera, args.frames, args.stride)):
            frame_count += 1
            rt_corners, rt_ids = rapidtag.detect_markers(image, DICTIONARY, params)
            cv_corners, cv_ids, _ = cv_detector.detectMarkers(image)
            rt = {int(i): np.asarray(c, dtype=np.float32).reshape(4, 2)
                  for i, c in zip(rt_ids, rt_corners)}
            cv = {int(i): np.asarray(c, dtype=np.float32).reshape(4, 2)
                  for i, c in zip([] if cv_ids is None else cv_ids.flatten(), cv_corners)}
            if rt.keys() != cv.keys():
                disagreements.append((camera, frame_number * args.stride, sorted(rt), sorted(cv)))
            for marker_id in rt.keys() & cv.keys():
                distances.extend(np.linalg.norm(rt[marker_id] - cv[marker_id], axis=1))

    print(f"rapidtag_decimate={args.decimate:g} opencv_decimate={cv_decimate:g} "
          f"frames={frame_count} stride={args.stride} "
          f"matched_corners={len(distances)} detection_disagreements={len(disagreements)}")
    if distances:
        print(f"corner_error_px median={np.median(distances):.6f} "
              f"p95={np.percentile(distances, 95):.6f} "
              f"max={np.max(distances):.6f}")
    for disagreement in disagreements[:10]:
        print("disagreement", disagreement)


if __name__ == "__main__":
    main()
