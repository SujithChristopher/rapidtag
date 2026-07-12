"""Sanity check: recover the AprilTag trajectory with solvePnP from each camera.

For every synchronized frame pair we:
  1. detect tag id 12 with rapidtag in cam0 and cam1,
  2. solvePnP (IPPE_SQUARE, the planar-square solver OpenCV's
     estimatePoseSingleMarkers uses) -> tag position in each camera frame,
  3. map cam1's estimate into cam0's frame via the stereo extrinsics (R, T),
  4. independently triangulate the 4 corners (pure stereo geometry, no PnP) as a
     cross-check.

If the library is working, all three trajectories overlap, the path is smooth,
and reprojection error is sub-pixel. Output: pnp_sanity_trajectory.png.
"""
import tomllib
import numpy as np
import cv2
import msgpack
import msgpack_numpy as mpn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # 3D projection auto-registers on import
import rapidtag

DATA = "data/dual_cam_single_aprl_50mm_t0"
DICT = "DICT_APRILTAG_36h11"
TAG_ID = 12
L = 50.0  # marker side length in mm (matches stereo T units)

# --- calibration ---
cal = tomllib.load(open(f"{DATA}/stereo_calibration.toml", "rb"))
K0 = np.array(cal["cam0"]["camera_matrix"])
D0 = np.array(cal["cam0"]["dist_coeffs"][0])
K1 = np.array(cal["cam1"]["camera_matrix"])
D1 = np.array(cal["cam1"]["dist_coeffs"][0])
R = np.array(cal["stereo"]["R"])          # X_cam1 = R X_cam0 + T
T = np.array(cal["stereo"]["T"]).reshape(3, 1)

# object points in aruco corner order (TL, TR, BR, BL), tag centred at origin
obj = np.array([[-L / 2, L / 2, 0], [L / 2, L / 2, 0],
                [L / 2, -L / 2, 0], [-L / 2, -L / 2, 0]], dtype=np.float64)

# projection matrices for triangulation, operating on normalized (undistorted) rays
P0 = np.hstack([np.eye(3), np.zeros((3, 1))])
P1 = np.hstack([R, T])


def load(cam):
    with open(f"{DATA}/{cam}_frame.msgpack", "rb") as f:
        unp = msgpack.Unpacker(f, object_hook=mpn.decode, raw=False)
        return [np.ascontiguousarray(o) for o in unp]


def detect(im):
    corners, ids = rapidtag.detect_markers(im, DICT)
    for c, i in zip(corners, ids):
        if i == TAG_ID:
            return np.array(c, dtype=np.float64)
    return None


def pnp(q, K, D):
    ok, rv, tv = cv2.solvePnP(obj, q, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None, None, None
    proj, _ = cv2.projectPoints(obj, rv, tv, K, D)
    rep = np.linalg.norm(proj.reshape(-1, 2) - q, axis=1).mean()
    return tv.ravel(), rv, rep


print("loading frames...")
c0, c1 = load("cam0"), load("cam1")
N = min(len(c0), len(c1))
print(f"{N} synchronized frame pairs")

nan3 = np.full(3, np.nan)
pnp0 = np.full((N, 3), np.nan)
pnp1_0 = np.full((N, 3), np.nan)   # cam1 PnP mapped into cam0 frame
tri = np.full((N, 3), np.nan)
rep0, rep1 = [], []
det0 = det1 = both = 0

for k in range(N):
    q0, q1 = detect(c0[k]), detect(c1[k])
    if q0 is not None:
        det0 += 1
        t0, _, r0 = pnp(q0, K0, D0)
        if t0 is not None:
            pnp0[k] = t0
            rep0.append(r0)
    if q1 is not None:
        det1 += 1
        t1, _, r1 = pnp(q1, K1, D1)
        if t1 is not None:
            pnp1_0[k] = (R.T @ (t1.reshape(3, 1) - T)).ravel()  # -> cam0 frame
            rep1.append(r1)
    if q0 is not None and q1 is not None:
        both += 1
        u0 = cv2.undistortPoints(q0.reshape(-1, 1, 2), K0, D0).reshape(-1, 2).T
        u1 = cv2.undistortPoints(q1.reshape(-1, 1, 2), K1, D1).reshape(-1, 2).T
        Xh = cv2.triangulatePoints(P0, P1, u0, u1)
        tri[k] = (Xh[:3] / Xh[3]).T.mean(axis=0)   # 4 corners -> tag centre, cam0 frame

# --- agreement stats (mm) ---
m_pnp = np.all(np.isfinite(pnp0), 1) & np.all(np.isfinite(pnp1_0), 1)
m_tri = np.all(np.isfinite(pnp0), 1) & np.all(np.isfinite(tri), 1)
d_cam = np.linalg.norm(pnp0[m_pnp] - pnp1_0[m_pnp], axis=1)
d_tri = np.linalg.norm(pnp0[m_tri] - tri[m_tri], axis=1)

print(f"\ndetections: cam0={det0}/{N}  cam1={det1}/{N}  both={both}")
print(f"reprojection err (px):  cam0 mean={np.mean(rep0):.3f}  cam1 mean={np.mean(rep1):.3f}")
print(f"cam0-PnP vs cam1-PnP (mm): mean={d_cam.mean():.2f}  median={np.median(d_cam):.2f}  p95={np.percentile(d_cam,95):.2f}")
print(f"cam0-PnP vs triangulation (mm): mean={d_tri.mean():.2f}  median={np.median(d_tri):.2f}  p95={np.percentile(d_tri,95):.2f}")
depth = pnp0[np.isfinite(pnp0[:, 2]), 2]
print(f"tag depth Z (mm): min={depth.min():.0f}  max={depth.max():.0f}")

# =========================== plots ===========================
fr = np.arange(N)
C0, C1, CT = "#2563eb", "#dc2626", "#059669"  # cam0, cam1, triangulated
fig = plt.figure(figsize=(16, 10))
fig.suptitle("rapidtag AprilTag (id 12) — solvePnP trajectory sanity check", fontsize=15, fontweight="bold")

# (1) 3D trajectory
ax = fig.add_subplot(2, 3, 1, projection="3d")
ax.plot(*pnp0[m_pnp | np.isfinite(pnp0[:, 0])].T, color=C0, lw=0.8, label="cam0 PnP")
ax.plot(*pnp1_0[np.isfinite(pnp1_0[:, 0])].T, color=C1, lw=0.8, alpha=0.7, label="cam1 PnP→cam0")
ax.plot(*tri[np.isfinite(tri[:, 0])].T, color=CT, lw=0.8, alpha=0.7, label="stereo triangulation")
ax.set_title("3D trajectory (cam0 frame, mm)")
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z (depth)")
ax.legend(fontsize=8)

# (2) top-down X-Z path
ax = fig.add_subplot(2, 3, 2)
ax.plot(pnp0[:, 0], pnp0[:, 2], color=C0, lw=0.8, label="cam0 PnP")
ax.plot(pnp1_0[:, 0], pnp1_0[:, 2], color=C1, lw=0.8, alpha=0.7, label="cam1 PnP→cam0")
ax.plot(tri[:, 0], tri[:, 2], color=CT, lw=0.8, alpha=0.6, label="triangulation")
ax.scatter([0], [0], c="k", marker="^", s=60, label="cam0 origin")
ax.set_title("top-down (X vs Z depth)"); ax.set_xlabel("X (mm)"); ax.set_ylabel("Z (mm)")
ax.axis("equal"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# (3-5) per-axis vs frame
for i, (axis, lab) in enumerate(zip(range(3), ["X", "Y", "Z (depth)"])):
    ax = fig.add_subplot(2, 3, 3 + i)
    ax.plot(fr, pnp0[:, axis], color=C0, lw=0.7, label="cam0 PnP")
    ax.plot(fr, pnp1_0[:, axis], color=C1, lw=0.7, alpha=0.7, label="cam1 PnP→cam0")
    ax.plot(fr, tri[:, axis], color=CT, lw=0.7, alpha=0.6, label="triangulation")
    ax.set_title(f"{lab} vs frame"); ax.set_xlabel("frame"); ax.set_ylabel(f"{lab} (mm)")
    ax.grid(alpha=0.3)
    if i == 0:
        ax.legend(fontsize=8)

# (6) agreement + reprojection error
ax = fig.add_subplot(2, 3, 6)
ax.hist(d_cam, bins=60, color=C1, alpha=0.6, label=f"|cam0−cam1| PnP (μ={d_cam.mean():.1f}mm)")
ax.hist(d_tri, bins=60, color=CT, alpha=0.6, label=f"|cam0 PnP−tri| (μ={d_tri.mean():.1f}mm)")
ax.set_title("inter-estimate agreement"); ax.set_xlabel("position difference (mm)"); ax.set_ylabel("frames")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
txt = (f"detections cam0 {det0}/{N}, cam1 {det1}/{N}\n"
       f"reproj err  cam0 {np.mean(rep0):.3f}px  cam1 {np.mean(rep1):.3f}px")
ax.text(0.97, 0.55, txt, transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round", fc="w", alpha=0.8))

fig.tight_layout(rect=[0, 0, 1, 0.97])
out = f"{DATA}/pnp_sanity_trajectory.png"
fig.savefig(out, dpi=130)
print(f"\nsaved {out}")
