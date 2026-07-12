"""Benchmark fasttag vs OpenCV aruco on real OV9281 dual-cam frames.

Reads frames through the msgpack + msgpack_numpy pipeline, runs both detectors,
and reports throughput (FPS), latency, detection parity and corner agreement.
"""
import argparse
import time
import numpy as np
import msgpack
import msgpack_numpy as mpn
import cv2
import fasttag

DICT = "DICT_APRILTAG_36h11"
CV_DICT = cv2.aruco.DICT_APRILTAG_36h11


def frames(path, limit):
    with open(path, "rb") as f:
        unp = msgpack.Unpacker(f, object_hook=mpn.decode, raw=False)
        for i, obj in enumerate(unp):
            if limit and i >= limit:
                break
            yield np.ascontiguousarray(obj)


def corner_err(a, b):
    a = np.asarray(a)
    b = np.asarray(b).reshape(4, 2)
    return min(np.abs(np.roll(a, k, axis=0) - b).max() for k in range(4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/dual_cam_single_aprl_50mm_t0/cam0_frame.msgpack")
    ap.add_argument("--n", type=int, default=300, help="frames to process")
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    print(f"Loading up to {args.n} frames from {args.path} ...")
    imgs = list(frames(args.path, args.n))
    print(f"Loaded {len(imgs)} frames of {imgs[0].shape} {imgs[0].dtype}\n")

    cv_detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(CV_DICT))

    def run_fasttag(img):
        c, i = fasttag.detect_markers(img, DICT)
        return {int(k): np.array(v) for k, v in zip(i, c)}

    def run_opencv(img):
        c, i, _ = cv_detector.detectMarkers(img)
        if i is None:
            return {}
        return {int(k): v.reshape(4, 2) for k, v in zip(i.flatten(), c)}

    engines = {"fasttag": run_fasttag, "opencv": run_opencv}

    # warmup
    for img in imgs[: args.warmup]:
        for fn in engines.values():
            fn(img)

    results = {}
    per_frame_ids = {name: [] for name in engines}
    for name, fn in engines.items():
        latencies = []
        t0 = time.perf_counter()
        for img in imgs:
            s = time.perf_counter()
            res = fn(img)
            latencies.append(time.perf_counter() - s)
            per_frame_ids[name].append(res)
        total = time.perf_counter() - t0
        lat = np.array(latencies) * 1e3
        results[name] = dict(
            fps=len(imgs) / total,
            mean_ms=lat.mean(),
            p50_ms=np.percentile(lat, 50),
            p99_ms=np.percentile(lat, 99),
            detections=sum(len(r) for r in per_frame_ids[name]),
        )

    # parity
    n_frames = len(imgs)
    both = agree_ids = ft_extra = cv_extra = 0
    errs = []
    for k in range(n_frames):
        fk = per_frame_ids["fasttag"][k]
        ck = per_frame_ids["opencv"][k]
        fk_ids, ck_ids = set(fk), set(ck)
        agree_ids += len(fk_ids & ck_ids)
        ft_extra += len(fk_ids - ck_ids)
        cv_extra += len(ck_ids - fk_ids)
        for mid in fk_ids & ck_ids:
            both += 1
            errs.append(corner_err(fk[mid], ck[mid]))

    print(f"{'engine':10s} {'FPS':>8s} {'mean ms':>9s} {'p50 ms':>8s} {'p99 ms':>8s} {'dets':>7s}")
    for name, r in results.items():
        print(f"{name:10s} {r['fps']:8.1f} {r['mean_ms']:9.2f} {r['p50_ms']:8.2f} {r['p99_ms']:8.2f} {r['detections']:7d}")

    speedup = results["fasttag"]["fps"] / results["opencv"]["fps"]
    print(f"\nfasttag / opencv throughput: {speedup:.2f}x")
    print(f"\nParity over {n_frames} frames:")
    print(f"  markers agreed (same id both):   {agree_ids}")
    print(f"  fasttag-only detections:         {ft_extra}")
    print(f"  opencv-only detections:          {cv_extra}")
    if errs:
        e = np.array(errs)
        print(f"  corner error vs opencv (px): mean={e.mean():.3f} p99={np.percentile(e,99):.3f} max={e.max():.3f}")


if __name__ == "__main__":
    main()
