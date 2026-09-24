"""Benchmark robust rigid-body pose on the two real dome recordings.

The saved detection files contain AprilTag corners for both cameras.  Marker
poses from ``dome_rb_def/rigidbody_calibration.toml`` transform each marker's
four local corners into one shared rigid-body/reference frame, giving a real
multi-marker PnP problem without including detector time in the measurement.

Examples:
    python scripts/bench_ransac_dome.py
    python scripts/bench_ransac_dome.py --samples 1000 --iterations 100
"""

from __future__ import annotations

import argparse
import pickle
import statistics
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rapidtag


ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    ROOT / "data/dome/dome_rb_def/jitter_detections.pkl",
    ROOT / "data/dome/dome_random_movement_sep18_26/jitter_detections.pkl",
)
RIGID_BODY = ROOT / "data/dome/dome_rb_def/rigidbody_calibration.toml"


@dataclass
class Case:
    dataset: str
    camera: str
    frame: int
    object_points: list[list[float]]
    image_points: list[list[float]]
    camera_matrix: list[list[float]]
    distortion: list[float]


def load_toml(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def calibration_path(saved: dict, override: Path | None) -> Path:
    if override is not None:
        path = override.expanduser().resolve()
    else:
        path = Path(saved["calibration_toml"])
    if not path.is_file():
        raise FileNotFoundError(
            f"camera calibration not found: {path}; pass --camera-calibration"
        )
    return path


def rigid_body_points(config: dict) -> dict[int, np.ndarray]:
    half = float(config["meta"]["tag_size_m"]) / 2.0
    # RapidTag/OpenCV ArUco order: top-left, top-right, bottom-right, bottom-left.
    local = np.array(
        [[-half, half, 0.0], [half, half, 0.0],
         [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float64,
    )
    result = {}
    for marker_id, marker in config["markers"].items():
        rotation = np.asarray(marker["rotation_marker_to_reference"], dtype=np.float64)
        translation = np.asarray(marker["translation_marker_to_reference_m"], dtype=np.float64)
        result[int(marker_id)] = local @ rotation.T + translation
    return result


def build_cases(
    dataset_path: Path,
    marker_points: dict[int, np.ndarray],
    samples: int,
    calibration_override: Path | None,
) -> list[Case]:
    with dataset_path.open("rb") as f:
        saved = pickle.load(f)
    calibration = load_toml(calibration_path(saved, calibration_override))
    dataset = dataset_path.parent.name
    cases = []

    for camera, camera_data in saved["cameras"].items():
        detections = camera_data["detections"]
        frame_indices = np.linspace(
            0, len(detections) - 1, min(samples, len(detections)), dtype=int
        )
        intrinsics = calibration[camera]
        camera_matrix = intrinsics["camera_matrix"]
        distortion = intrinsics["dist_coeffs"][0]
        for frame in frame_indices:
            object_chunks = []
            image_chunks = []
            for marker_id, corners in detections[int(frame)].items():
                points = marker_points.get(int(marker_id))
                if points is None:
                    continue
                object_chunks.append(points)
                image_chunks.append(np.asarray(corners, dtype=np.float64))
            if len(object_chunks) < 2:
                # RANSAC is deliberately a redundant multi-marker solver.
                continue
            cases.append(
                Case(
                    dataset=dataset,
                    camera=camera,
                    frame=int(frame),
                    object_points=np.concatenate(object_chunks).tolist(),
                    image_points=np.concatenate(image_chunks).tolist(),
                    camera_matrix=camera_matrix,
                    distortion=distortion,
                )
            )
    return cases


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def summarize(name: str, latencies_us: list[float], successes: int, total: int,
              inlier_ratios: list[float], rmses: list[float]) -> None:
    print(
        f"  {name:9s} success={successes:4d}/{total:<4d} "
        f"median={statistics.median(latencies_us):7.2f}us "
        f"p95={percentile(latencies_us, 95):7.2f}us "
        f"mean={statistics.mean(latencies_us):7.2f}us "
        f"inliers={statistics.mean(inlier_ratios) * 100:5.1f}% "
        f"rmse={statistics.mean(rmses):5.2f}px"
    )


def benchmark_group(cases: list[Case], iterations: int, threshold: float,
                    confidence: float, seed: int) -> None:
    for case in cases[:20]:
        rapidtag.solve_pnp_ransac(
            case.object_points, case.image_points, case.camera_matrix, case.distortion,
            iterations=iterations, reprojection_error=threshold,
            confidence=confidence, seed=seed + case.frame,
        )

    rt_times = []
    rt_inlier_ratios = []
    rt_rmses = []
    rt_success = 0
    for case in cases:
        start = time.perf_counter_ns()
        result = rapidtag.solve_pnp_ransac(
            case.object_points, case.image_points, case.camera_matrix, case.distortion,
            iterations=iterations, reprojection_error=threshold,
            confidence=confidence, seed=seed + case.frame,
        )
        rt_times.append((time.perf_counter_ns() - start) / 1000.0)
        if result is not None:
            _, _, inliers, rmse = result
            rt_success += 1
            rt_inlier_ratios.append(len(inliers) / len(case.object_points))
            rt_rmses.append(rmse)

    cv_times = []
    cv_inlier_ratios = []
    cv_rmses = []
    cv_success = 0
    cv2.setRNGSeed(seed)
    for case in cases:
        obj = np.asarray(case.object_points, dtype=np.float64)
        img = np.asarray(case.image_points, dtype=np.float64)
        camera = np.asarray(case.camera_matrix, dtype=np.float64)
        distortion = np.asarray(case.distortion, dtype=np.float64)
        start = time.perf_counter_ns()
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj, img, camera, distortion,
            iterationsCount=iterations,
            reprojectionError=threshold,
            confidence=confidence,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        cv_times.append((time.perf_counter_ns() - start) / 1000.0)
        if ok and inliers is not None:
            indices = inliers.ravel()
            projected, _ = cv2.projectPoints(obj[indices], rvec, tvec, camera, distortion)
            residual = projected.reshape(-1, 2) - img[indices]
            cv_success += 1
            cv_inlier_ratios.append(len(indices) / len(obj))
            cv_rmses.append(float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))))

    summarize("rapidtag", rt_times, rt_success, len(cases), rt_inlier_ratios, rt_rmses)
    summarize("opencv", cv_times, cv_success, len(cases), cv_inlier_ratios, cv_rmses)
    print(f"  speedup:   {statistics.median(cv_times) / statistics.median(rt_times):.2f}x")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500,
                        help="evenly sampled frames per camera and recording")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=3.0,
                        help="RANSAC inlier threshold in pixels")
    parser.add_argument("--confidence", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--camera-calibration", type=Path)
    args = parser.parse_args()

    rigid_body = load_toml(RIGID_BODY)
    marker_points = rigid_body_points(rigid_body)
    total_cases = 0
    for dataset in DATASETS:
        cases = build_cases(dataset, marker_points, args.samples, args.camera_calibration)
        total_cases += len(cases)
        points = [len(case.object_points) for case in cases]
        print(
            f"\n{dataset.parent.name}: {len(cases)} camera-frames, "
            f"median={statistics.median(points):.0f} corners "
            f"({statistics.median(points) / 4:.0f} markers)"
        )
        benchmark_group(cases, args.iterations, args.threshold, args.confidence, args.seed)

    print(f"\nbenchmarked {total_cases} camera-frames across both recordings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
