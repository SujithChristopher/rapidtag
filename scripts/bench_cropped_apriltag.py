"""Benchmark native crop refinement against a Python reference and OpenCV."""

from __future__ import annotations

import argparse
from time import perf_counter_ns

import cv2
import numpy as np

import rapidtag
from bench_corner_refinement import DICTIONARY
from compare_apriltag_corners import load_sampled


def crop_bounds(corners: np.ndarray, width: int, height: int, margin_rate: float) -> tuple[int, int, int, int]:
    side = max(float(np.ptp(corners[:, 0])), float(np.ptp(corners[:, 1])))
    margin = max(16.0, side * margin_rate)
    # AprilTag's threshold uses 4x4 tiles; preserve the full-frame tile phase.
    x0 = max(0, int(np.floor((float(np.min(corners[:, 0])) - margin) / 4)) * 4)
    y0 = max(0, int(np.floor((float(np.min(corners[:, 1])) - margin) / 4)) * 4)
    x1 = min(width, int(np.ceil((float(np.max(corners[:, 0])) + margin) / 4)) * 4)
    y1 = min(height, int(np.ceil((float(np.max(corners[:, 1])) + margin) / 4)) * 4)
    return x0, y0, x1, y1


def detect_cropped(image: np.ndarray, coarse_params: rapidtag.DetectorParameters,
                   full_params: rapidtag.DetectorParameters, margin_rate: float):
    start = perf_counter_ns()
    coarse_corners, coarse_ids = rapidtag.detect_markers(image, DICTIONARY, coarse_params)
    coarse_ms = (perf_counter_ns() - start) / 1e6
    start = perf_counter_ns()
    output = []
    fallbacks = 0
    crop_pixels = 0
    for initial, marker_id in zip(coarse_corners, coarse_ids):
        initial = np.asarray(initial, dtype=np.float32).reshape(4, 2)
        x0, y0, x1, y1 = crop_bounds(initial, image.shape[1], image.shape[0], margin_rate)
        crop = np.ascontiguousarray(image[y0:y1, x0:x1])
        crop_pixels += crop.size
        refined, ids = rapidtag.detect_markers(crop, DICTIONARY, full_params)
        matches = [np.asarray(c, dtype=np.float32).reshape(4, 2) + (x0, y0)
                   for c, i in zip(refined, ids) if i == marker_id]
        if matches:
            output.append((marker_id, min(matches, key=lambda c: np.linalg.norm(c - initial))))
        else:
            output.append((marker_id, initial))
            fallbacks += 1
    fine_ms = (perf_counter_ns() - start) / 1e6
    return output, coarse_ms, fine_ms, fallbacks, crop_pixels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--decimate", type=float, default=3.0)
    parser.add_argument("--margin", type=float, default=0.5)
    args = parser.parse_args()
    if min(args.frames, args.stride, args.repeats) < 1 or args.decimate <= 1 or args.margin < 0:
        parser.error("invalid frame count, stride, repeat count, decimation, or margin")

    images = [(camera, i * args.stride, image)
              for camera in ("cam0", "cam1")
              for i, image in enumerate(load_sampled(camera, args.frames, args.stride))]
    coarse_params = rapidtag.DetectorParameters()
    coarse_params.corner_refinement_method = rapidtag.CORNER_REFINE_APRILTAG
    coarse_params.april_tag_quad_decimate = args.decimate
    full_params = rapidtag.DetectorParameters()
    full_params.corner_refinement_method = rapidtag.CORNER_REFINE_APRILTAG
    native_params = rapidtag.DetectorParameters()
    native_params.corner_refinement_method = rapidtag.CORNER_REFINE_APRILTAG
    native_params.april_tag_quad_decimate = args.decimate
    native_params.april_tag_refine_full_resolution = True
    cv_params = cv2.aruco.DetectorParameters()
    cv_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    cv_detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), cv_params)

    for _, _, image in images[:5]:
        detect_cropped(image, coarse_params, full_params, args.margin)
    coarse_times, fine_times, totals, distances = [], [], [], []
    fallbacks = 0
    disagreements = []
    reference = {}
    crop_pixels = []
    for repeat in range(args.repeats):
        for camera, index, image in images:
            result, coarse_ms, fine_ms, count, pixels = detect_cropped(
                image, coarse_params, full_params, args.margin)
            coarse_times.append(coarse_ms)
            fine_times.append(fine_ms)
            totals.append(coarse_ms + fine_ms)
            crop_pixels.append(pixels)
            fallbacks += count
            if repeat:
                continue
            cv_corners, cv_ids, _ = cv_detector.detectMarkers(image)
            actual = {int(i): np.asarray(c).reshape(4, 2)
                      for i, c in zip([] if cv_ids is None else cv_ids.flatten(), cv_corners)}
            reference[(camera, index)] = actual
            observed = {i: c for i, c in result}
            if actual.keys() != observed.keys():
                disagreements.append((camera, index, sorted(observed), sorted(actual)))
            for marker_id in actual.keys() & observed.keys():
                distances.extend(np.linalg.norm(observed[marker_id] - actual[marker_id], axis=1))

    for _, _, image in images[:5]:
        rapidtag.detect_markers(image, DICTIONARY, full_params)
    full_times = []
    for _ in range(args.repeats):
        for _, _, image in images:
            start = perf_counter_ns()
            rapidtag.detect_markers(image, DICTIONARY, full_params)
            full_times.append((perf_counter_ns() - start) / 1e6)

    for _, _, image in images[:5]:
        rapidtag.detect_markers(image, DICTIONARY, native_params)
    native_times, native_distances, native_disagreements = [], [], []
    for repeat in range(args.repeats):
        for camera, index, image in images:
            start = perf_counter_ns()
            corners, ids = rapidtag.detect_markers(image, DICTIONARY, native_params)
            native_times.append((perf_counter_ns() - start) / 1e6)
            if repeat:
                continue
            observed = {i: np.asarray(c).reshape(4, 2) for i, c in zip(ids, corners)}
            actual = reference[(camera, index)]
            if actual.keys() != observed.keys():
                native_disagreements.append((camera, index, sorted(observed), sorted(actual)))
            for marker_id in actual.keys() & observed.keys():
                native_distances.extend(np.linalg.norm(observed[marker_id] - actual[marker_id], axis=1))

    def timing(values: list[float]) -> str:
        return f"median={np.median(values):.2f}ms p95={np.percentile(values, 95):.2f}ms"

    print(f"frames={len(images)} shape={images[0][2].shape} repeats={args.repeats} "
          f"decimate={args.decimate:g} margin={args.margin:g}")
    print("coarse", timing(coarse_times))
    print("crop  ", timing(fine_times), f"median_pixels={np.median(crop_pixels):,.0f}")
    print("total ", timing(totals))
    print("full  ", timing(full_times))
    print("native", timing(native_times))
    print(f"fallbacks={fallbacks} detection_disagreements={len(disagreements)}")
    if distances:
        print(f"corner_error_vs_opencv_full_px median={np.median(distances):.6f} "
              f"p95={np.percentile(distances, 95):.6f} max={np.max(distances):.6f}")
    print(f"native_detection_disagreements={len(native_disagreements)}")
    if native_distances:
        print(f"native_corner_error_vs_opencv_full_px median={np.median(native_distances):.6f} "
              f"p95={np.percentile(native_distances, 95):.6f} max={np.max(native_distances):.6f}")
    for disagreement in disagreements[:10]:
        print("disagreement", disagreement)


if __name__ == "__main__":
    main()
