"""OpenCV vs rapidtag: identical solvePnP pipeline, only the corner source differs.

For each camera we detect tag id 12 with both OpenCV's aruco detector and rapidtag,
run the SAME solvePnP (IPPE_SQUARE) on each, and overlay the recovered tag position.
Layout: 2 rows (cam0, cam1) x 3 cols (X, Y, Z vs frame). If rapidtag is a faithful
drop-in, the two lines sit on top of each other. Output: pnp_opencv_vs_rapidtag.png.
"""
import tomllib
import numpy as np
import cv2
import msgpack
import msgpack_numpy as mpn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rapidtag

DATA = "data/dual_cam_single_aprl_50mm_t0"
DICT = "DICT_APRILTAG_36h11"
CV_DICT = cv2.aruco.DICT_APRILTAG_36h11
TAG_ID = 12
L = 50.0  # marker side length in mm

cal = tomllib.load(open(f"{DATA}/stereo_calibration.toml", "rb"))
K = {c: np.array(cal[c]["camera_matrix"]) for c in ("cam0", "cam1")}
D = {c: np.array(cal[c]["dist_coeffs"][0]) for c in ("cam0", "cam1")}

obj = np.array([[-L / 2, L / 2, 0], [L / 2, L / 2, 0],
                [L / 2, -L / 2, 0], [-L / 2, -L / 2, 0]], dtype=np.float64)

cvd = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(CV_DICT))


def load(cam):
    with open(f"{DATA}/{cam}_frame.msgpack", "rb") as f:
        unp = msgpack.Unpacker(f, object_hook=mpn.decode, raw=False)
        return [np.ascontiguousarray(o) for o in unp]


def corners_rapidtag(im):
    cs, ids = rapidtag.detect_markers(im, DICT)
    for c, i in zip(cs, ids):
        if i == TAG_ID:
            return np.array(c, dtype=np.float64)
    return None


def corners_opencv(im):
    cs, ids, _ = cvd.detectMarkers(im)
    if ids is None:
        return None
    for c, i in zip(cs, ids.flatten()):
        if int(i) == TAG_ID:
            return c.reshape(4, 2).astype(np.float64)
    return None


def pnp_pos(q, cam):
    ok, _, tv = cv2.solvePnP(obj, q, K[cam], D[cam], flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return tv.ravel() if ok else np.full(3, np.nan)


print("loading frames...")
frames = {c: load(c) for c in ("cam0", "cam1")}
N = min(len(frames["cam0"]), len(frames["cam1"]))
print(f"{N} frames")

# pos[cam][method] -> (N,3)
pos = {c: {"rapidtag": np.full((N, 3), np.nan), "opencv": np.full((N, 3), np.nan)}
       for c in ("cam0", "cam1")}

for cam in ("cam0", "cam1"):
    for k in range(N):
        im = frames[cam][k]
        qr = corners_rapidtag(im)
        qc = corners_opencv(im)
        if qr is not None:
            pos[cam]["rapidtag"][k] = pnp_pos(qr, cam)
        if qc is not None:
            pos[cam]["opencv"][k] = pnp_pos(qc, cam)

# agreement stats where both methods produced a pose
for cam in ("cam0", "cam1"):
    a, b = pos[cam]["rapidtag"], pos[cam]["opencv"]
    m = np.all(np.isfinite(a), 1) & np.all(np.isfinite(b), 1)
    d = np.linalg.norm(a[m] - b[m], axis=1)
    print(f"{cam}: both={m.sum()}  |rapidtag-opencv| PnP pos (mm): "
          f"mean={d.mean():.3f} median={np.median(d):.3f} p99={np.percentile(d,99):.3f} max={d.max():.3f}")

# =========================== plot ===========================
fr = np.arange(N)
C_RT, C_CV = "#2563eb", "#f59e0b"  # rapidtag blue, opencv amber
fig, axes = plt.subplots(2, 3, figsize=(17, 8), sharex=True)
fig.suptitle("solvePnP tag position — OpenCV corners vs rapidtag corners (same pipeline)",
             fontsize=15, fontweight="bold")
labels = ["X (mm)", "Y (mm)", "Z depth (mm)"]

for r, cam in enumerate(("cam0", "cam1")):
    a, b = pos[cam]["rapidtag"], pos[cam]["opencv"]
    m = np.all(np.isfinite(a), 1) & np.all(np.isfinite(b), 1)
    d = np.linalg.norm(a[m] - b[m], axis=1)
    for c in range(3):
        ax = axes[r][c]
        ax.plot(fr, b[:, c], color=C_CV, lw=1.4, label="OpenCV", alpha=0.9)
        ax.plot(fr, a[:, c], color=C_RT, lw=0.9, label="rapidtag", alpha=0.9)
        if r == 1:
            ax.set_xlabel("frame")
        ax.set_ylabel(labels[c])
        ax.grid(alpha=0.3)
        if r == 0 and c == 0:
            ax.legend(fontsize=9, loc="upper right")
        if c == 0:
            ax.text(0.02, 0.97, f"{cam}", transform=ax.transAxes, va="top", fontsize=12,
                    fontweight="bold", bbox=dict(boxstyle="round", fc="w", alpha=0.8))
    axes[r][1].set_title(f"{cam}:  |rapidtag − OpenCV| pos  mean={d.mean():.2f}mm  median={np.median(d):.2f}mm",
                         fontsize=10)

fig.tight_layout(rect=[0, 0, 1, 0.96])
out = f"{DATA}/pnp_opencv_vs_rapidtag.png"
fig.savefig(out, dpi=130)
print(f"\nsaved {out}")
