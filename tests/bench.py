"""Benchmark fasttag vs OpenCV aruco on real OV9281 dual-cam frames.

Reads frames through the msgpack + msgpack_numpy pipeline and reports:
  - single-frame latency (realtime, one camera)
  - dual-camera pair latency (both cameras processed together via the batch API)
  - offline batch throughput (all cores)
  - detection parity and corner agreement vs OpenCV
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
DATA = "data/dual_cam_single_aprl_50mm_t0"


def load(cam, n):
    with open(f"{DATA}/{cam}_frame.msgpack", "rb") as f:
        unp = msgpack.Unpacker(f, object_hook=mpn.decode, raw=False)
        return [np.ascontiguousarray(o) for _, o in zip(range(n), unp)]


def lat_ms(fn, items):
    out = []
    for x in items:
        t = time.perf_counter()
        fn(x)
        out.append((time.perf_counter() - t) * 1e3)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()

    c0 = load("cam0", args.n)
    c1 = load("cam1", args.n)
    print(f"Loaded {len(c0)}+{len(c1)} frames of {c0[0].shape}\n")

    # warmup
    for im in c0[:20]:
        fasttag.detect_markers(im, DICT)
    fasttag.detect_markers_batch(c0[:8], DICT)

    cvd = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(CV_DICT))

    s = lat_ms(lambda im: fasttag.detect_markers(im, DICT), c0)
    print(f"fasttag single-frame:    {1000/s.mean():6.1f} FPS   mean={s.mean():.2f}ms  p99={np.percentile(s,99):.2f}ms")

    pairs = list(zip(c0, c1))
    d = lat_ms(lambda pr: fasttag.detect_markers_batch(list(pr), DICT), pairs)
    print(f"fasttag dual-cam pair:   {1000/d.mean():6.1f} pairs/s ({2000/d.mean():.0f} fps total)  mean={d.mean():.2f}ms/pair")

    allf = c0 + c1
    t = time.perf_counter()
    fasttag.detect_markers_batch(allf, DICT)
    bt = time.perf_counter() - t
    print(f"fasttag offline batch:   {len(allf)/bt:6.1f} FPS   ({len(allf)} frames, all cores)")

    o = lat_ms(lambda im: cvd.detectMarkers(im), c0)
    print(f"opencv  single-frame:    {1000/o.mean():6.1f} FPS   mean={o.mean():.2f}ms")

    # parity vs OpenCV
    res = fasttag.detect_markers_batch(c0, DICT)
    agree = ftonly = cvonly = 0
    errs = []
    for im, (cb, ib) in zip(c0, res):
        cc, ci, _ = cvd.detectMarkers(im)
        di = {} if ci is None else {int(k): v.reshape(4, 2) for k, v in zip(ci.flatten(), cc)}
        fi = {int(k): np.array(v) for k, v in zip(ib, cb)}
        for k in set(fi) & set(di):
            agree += 1
            errs.append(min(np.abs(np.roll(fi[k], r, axis=0) - di[k]).max() for r in range(4)))
        ftonly += len(set(fi) - set(di))
        cvonly += len(set(di) - set(fi))
    e = np.array(errs)
    print(f"\nParity vs OpenCV ({len(c0)} frames): agreed={agree} fasttag-only={ftonly} opencv-only={cvonly}")
    print(f"corner error vs OpenCV (px): mean={e.mean():.4f} p99={np.percentile(e,99):.3f} max={e.max():.3f}")


if __name__ == "__main__":
    main()
