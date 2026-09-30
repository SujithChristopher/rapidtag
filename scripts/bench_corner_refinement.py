"""Benchmark corner refinement modes on the recorded OV9281 stereo frames.

Run from the repository root after installing the dev dependencies:
    python scripts/bench_corner_refinement.py --frames 24 --repeats 3
Image loading and detector construction are excluded from timings.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter_ns

import cv2
import msgpack
import msgpack_numpy
import numpy as np

import rapidtag


DATA = Path("data/dual_cam_single_aprl_50mm_t0")
DICTIONARY = "DICT_APRILTAG_36h11"


def load(camera: str, count: int) -> list[np.ndarray]:
    with (DATA / f"{camera}_frame.msgpack").open("rb") as source:
        unpacker = msgpack.Unpacker(source, object_hook=msgpack_numpy.decode, raw=False)
        return [np.ascontiguousarray(frame) for _, frame in zip(range(count), unpacker)]


def summarize(label: str, samples: list[float], counts: list[int], pixels: int) -> None:
    times = np.asarray(samples)
    print(
        f"{label:24s} {pixels:>8,d} px  "
        f"median={np.median(times):6.2f} ms  "
        f"p95={np.percentile(times, 95):6.2f} ms  "
        f"mean={np.mean(times):6.2f} ms  "
        f"tags={min(counts)}..{max(counts)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--only", help="run one mode by its exact printed name")
    args = parser.parse_args()
    if args.frames < 1 or args.repeats < 1:
        parser.error("--frames and --repeats must be positive")

    cam0 = load("cam0", args.frames)
    cam1 = load("cam1", args.frames)
    if not cam0 or not cam1:
        raise SystemExit("no frames found")
    pairs = list(zip(cam0, cam1))
    print(f"Frames: {len(pairs)} pairs, shape={cam0[0].shape}, repeats={args.repeats}")

    modes = [
        ("RapidTag NONE", rapidtag.CORNER_REFINE_NONE),
        ("RapidTag SUBPIX", rapidtag.CORNER_REFINE_SUBPIX),
        ("RapidTag CONTOUR", rapidtag.CORNER_REFINE_CONTOUR),
        ("RapidTag APRILTAG", rapidtag.CORNER_REFINE_APRILTAG),
    ]
    detectors = {}
    for name, mode in modes:
        params = rapidtag.DetectorParameters()
        params.corner_refinement_method = mode
        detectors[name] = lambda image, p=params: rapidtag.detect_markers(image, DICTIONARY, p)
    fast_april = rapidtag.DetectorParameters()
    fast_april.corner_refinement_method = rapidtag.CORNER_REFINE_APRILTAG
    fast_april.april_tag_quad_decimate = 2.0
    detectors["RapidTag APRILTAG 2x"] = (
        lambda image: rapidtag.detect_markers(image, DICTIONARY, fast_april)
    )
    cv_params = cv2.aruco.DetectorParameters()
    cv_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    cv_detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), cv_params
    )
    detectors["OpenCV APRILTAG"] = lambda image: cv_detector.detectMarkers(image)[:2]
    cv_fast_params = cv2.aruco.DetectorParameters()
    cv_fast_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    cv_fast_params.aprilTagQuadDecimate = 2.0
    cv_fast_detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), cv_fast_params
    )
    detectors["OpenCV APRILTAG 2x"] = lambda image: cv_fast_detector.detectMarkers(image)[:2]
    if args.only:
        detectors = {
            name: detector for name, detector in detectors.items()
            if args.only.lower() == name.lower()
        }
        if not detectors:
            parser.error("--only matched no detector modes")

    for name, detector in detectors.items():
        for frame in cam0[: min(5, len(cam0))]:
            detector(frame)
        samples, counts = [], []
        for _ in range(args.repeats):
            for frame in cam0:
                start = perf_counter_ns()
                _, ids = detector(frame)
                samples.append((perf_counter_ns() - start) / 1e6)
                counts.append(0 if ids is None else len(ids))
        summarize(name, samples, counts, cam0[0].size)

    for name, mode, decimate in (
        (modes[0][0], modes[0][1], 0.0),
        (modes[-1][0], modes[-1][1], 0.0),
        ("RapidTag APRILTAG 2x", rapidtag.CORNER_REFINE_APRILTAG, 2.0),
    ):
        if args.only and args.only.lower() != name.lower():
            continue
        params = rapidtag.DetectorParameters()
        params.corner_refinement_method = mode
        params.april_tag_quad_decimate = decimate
        for pair in pairs[: min(5, len(pairs))]:
            rapidtag.detect_markers_batch(list(pair), DICTIONARY, params)
        samples, counts = [], []
        for _ in range(args.repeats):
            for pair in pairs:
                start = perf_counter_ns()
                result = rapidtag.detect_markers_batch(list(pair), DICTIONARY, params)
                samples.append((perf_counter_ns() - start) / 1e6)
                counts.append(sum(len(ids) for _, ids in result))
        summarize(f"{name} pair", samples, counts, cam0[0].size + cam1[0].size)


if __name__ == "__main__":
    main()
