"""Cross-check rapidtag's solvePnP / projectPoints / Rodrigues against OpenCV.

Pose is a converged optimum, unlike cornerSubPix's early-terminating refinement,
so an independent Levenberg-Marquardt should land on the same minimum even though
the path there differs (OpenCV 5 uses a geodesic-accelerated LevMarq; we use a
textbook damped Gauss-Newton, and our homography init is a plain DLT where OpenCV
additionally LM-refines it). We therefore compare the converged pose and the
reprojection error, not the arithmetic.
"""
import numpy as np
import cv2
import rapidtag

rng = np.random.default_rng(7)

K = np.array([[800.0, 0.0, 320.0],
              [0.0, 810.0, 240.0],
              [0.0, 0.0, 1.0]])

DISTS = {
    "none": np.zeros(5),
    "k1k2p1p2k3": np.array([-0.28, 0.12, 0.001, -0.0007, 0.02]),
    "8-coeff": np.array([-0.3, 0.15, 0.001, -0.001, 0.03, 0.0005, -0.0002, 0.0001]),
}

POSES = [
    (np.array([0.05, -0.03, 0.02]), np.array([0.02, -0.01, 0.60])),
    (np.array([0.4, 0.25, -0.15]), np.array([-0.05, 0.03, 0.80])),
    (np.array([-0.9, 0.5, 0.3]), np.array([0.10, -0.06, 1.20])),
    (np.array([1.4, -0.2, 0.7]), np.array([0.00, 0.00, 0.45])),
]


def planar_points(n=5):
    xs, ys = np.meshgrid(np.linspace(-0.06, 0.06, n), np.linspace(-0.04, 0.04, n))
    return np.stack([xs.ravel(), ys.ravel(), np.zeros(n * n)], axis=1)


def cube_points(n=12):
    return rng.uniform(-0.06, 0.06, (n, 3))


def rot_err(r1, r2):
    """Geodesic angle between two rotations, in degrees."""
    R1, _ = cv2.Rodrigues(np.asarray(r1, float).reshape(3, 1))
    R2, _ = cv2.Rodrigues(np.asarray(r2, float).reshape(3, 1))
    c = (np.trace(R1.T @ R2) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def reproj_err(obj, img, rvec, tvec, dist):
    p, _ = cv2.projectPoints(obj, np.asarray(rvec, float).reshape(3, 1),
                             np.asarray(tvec, float).reshape(3, 1), K, dist)
    return float(np.abs(p.reshape(-1, 2) - img).max())


def charuco_pose_end_to_end():
    """Render a ChArUco board at a known pose, detect it, recover the pose.

    This is the whole stack at once: detectMarkers -> interpolate -> cornerSubPix
    -> solvePnP. Ground truth is known exactly, so it catches sign/axis/ordering
    mistakes that a solvePnP-vs-solvePnP comparison would agree on.
    """
    print("\n--- ChArUco board pose, end to end ---")
    sx, sy, sq, ml = 5, 7, 0.04, 0.02
    cv_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    cv_board = cv2.aruco.CharucoBoard((sx, sy), sq, ml, cv_dict)
    rt_board = rapidtag.CharucoBoard(sx, sy, sq, ml, "DICT_4X4_50")
    px = 100
    flat = cv_board.generateImage((sx * px, sy * px), marginSize=0)

    # A dedicated camera/canvas: the board must land large enough that its markers
    # clear the detector's min_side_length_canonical_img=32 gate, and the poses
    # must keep it off edge-on. POSES above include a ~90deg rotation, which would
    # render the board as a sliver.
    global K
    K_saved = K
    K = np.array([[900.0, 0.0, 450.0], [0.0, 900.0, 400.0], [0.0, 0.0, 1.0]])
    canvas = (900, 800)
    e2e_poses = [
        (np.array([0.05, -0.03, 0.02]), np.array([0.0, 0.0, 0.40])),
        (np.array([0.30, 0.20, -0.10]), np.array([0.02, -0.01, 0.45])),
        (np.array([-0.50, 0.30, 0.15]), np.array([-0.02, 0.01, 0.42])),
        (np.array([0.60, -0.25, 0.20]), np.array([0.01, 0.02, 0.50])),
    ]

    worst_rot, worst_t, tested = 0.0, 0.0, 0
    for rvec, tvec in e2e_poses:
        # Board plane (z=0) spans [0,sx*sq] x [0,sy*sq]; centre it on the origin
        # so the poses above frame it sensibly.
        w, h = sx * sq, sy * sq
        corners3d = np.array([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0],
                              [w / 2, h / 2, 0], [-w / 2, h / 2, 0]], dtype=np.float64)
        proj, _ = cv2.projectPoints(corners3d, rvec.reshape(3, 1), tvec.reshape(3, 1),
                                    K, np.zeros(5))
        proj = proj.reshape(-1, 2).astype(np.float32)
        src = np.array([[0, 0], [sx * px, 0], [sx * px, sy * px], [0, sy * px]], np.float32)
        M = cv2.getPerspectiveTransform(src, proj)
        img = cv2.warpPerspective(flat, M, canvas, borderValue=255)

        c, i, _, _ = rapidtag.detect_charuco_board(img, rt_board)
        if len(i) < 4:
            print(f"  rvec={np.round(rvec,2)}: FAIL - only {len(i)} corners detected")
            K = K_saved
            return False
        tested += 1

        # rapidtag's board corners are in [0,w]x[0,h]; shift to the centred frame
        # used for the ground-truth render.
        cc = np.array(rt_board.chessboard_corners)
        obj = np.array([[cc[k][0] - w / 2, cc[k][1] - h / 2, 0.0] for k in i])
        res = rapidtag.solve_pnp(obj.tolist(), [list(p) for p in c], K.tolist(), [0.0] * 5)
        assert res is not None
        rt_r, rt_t = res
        dr = rot_err(rvec, rt_r)
        dt = float(np.abs(tvec - np.array(rt_t)).max())
        worst_rot = max(worst_rot, dr)
        worst_t = max(worst_t, dt)
        print(f"  rvec={np.round(rvec,2)} corners={len(i):2d}: "
              f"rot_err={dr:.4f}deg t_err={dt*1000:.4f}mm")

    K = K_saved
    print(f"  worst over {tested} poses: rot={worst_rot:.4f}deg  t={worst_t*1000:.4f}mm")
    # Recovered from rendered pixels, so this is limited by rasterisation, not
    # maths. Requiring all 4 poses to have been tested keeps the check from
    # passing vacuously when nothing is detected.
    return tested == len(e2e_poses) and worst_rot < 0.5 and worst_t < 0.002


def main():
    ok = True

    # --- Rodrigues ---
    e = 0.0
    for _ in range(200):
        r = rng.uniform(-np.pi, np.pi, 3)
        r = r / max(np.linalg.norm(r), 1e-12) * rng.uniform(0, np.pi * 0.99)
        cvR, _ = cv2.Rodrigues(r.reshape(3, 1))
        rtR = np.array(rapidtag.rodrigues([[v] for v in r]))
        e = max(e, float(np.abs(cvR - rtR).max()))
        back = np.array(rapidtag.rodrigues([list(row) for row in cvR])).ravel()
        cvback, _ = cv2.Rodrigues(cvR)
        e = max(e, float(np.abs(cvback.ravel() - back).max()))
    print(f"Rodrigues (both directions, 200 random rotations): max_err={e:.2e}")
    ok &= e < 1e-9

    # --- projectPoints ---
    e = 0.0
    for name, dist in DISTS.items():
        for rvec, tvec in POSES:
            obj = cube_points(30)
            cvp, _ = cv2.projectPoints(obj, rvec.reshape(3, 1), tvec.reshape(3, 1), K, dist)
            rtp = np.array(rapidtag.project_points(
                obj.tolist(), rvec.tolist(), tvec.tolist(), K.tolist(), dist.tolist()))
            e = max(e, float(np.abs(cvp.reshape(-1, 2) - rtp).max()))
    print(f"projectPoints (3 distortion models x 4 poses): max_err={e:.2e}px")
    ok &= e < 1e-9

    # --- solvePnP ---
    print()
    worst_rot, worst_t, worst_reproj = 0.0, 0.0, 0.0
    for name, dist in DISTS.items():
        for geom, obj in (("planar", planar_points()), ("non-planar", cube_points(12))):
            for rvec, tvec in POSES:
                img, _ = cv2.projectPoints(obj, rvec.reshape(3, 1), tvec.reshape(3, 1), K, dist)
                img = img.reshape(-1, 2)
                img_noisy = img + rng.normal(0, 0.05, img.shape)  # sub-pixel detector noise

                cv_ok, cv_r, cv_t = cv2.solvePnP(obj, img_noisy, K, dist,
                                                 flags=cv2.SOLVEPNP_ITERATIVE)
                res = rapidtag.solve_pnp(obj.tolist(), img_noisy.tolist(),
                                         K.tolist(), dist.tolist())
                if not cv_ok or res is None:
                    print(f"  {name:11s} {geom:10s}: SOLVE FAILED (cv={cv_ok}, rt={res is not None})")
                    ok = False
                    continue
                rt_r, rt_t = res
                dr = rot_err(cv_r.ravel(), rt_r)
                dt = float(np.abs(cv_t.ravel() - np.array(rt_t)).max())
                # What matters most: does our pose explain the image as well as cv's?
                e_cv = reproj_err(obj, img_noisy, cv_r.ravel(), cv_t.ravel(), dist)
                e_rt = reproj_err(obj, img_noisy, rt_r, rt_t, dist)
                worst_rot = max(worst_rot, dr)
                worst_t = max(worst_t, dt)
                worst_reproj = max(worst_reproj, abs(e_cv - e_rt))
                print(f"  {name:11s} {geom:10s} rvec={np.round(rvec,2)}: "
                      f"rot_diff={dr:.2e}deg t_diff={dt:.2e}m "
                      f"reproj cv={e_cv:.4f} rt={e_rt:.4f}")

    print(f"\nWorst rotation difference : {worst_rot:.3e} deg")
    print(f"Worst translation difference: {worst_t:.3e} m")
    print(f"Worst reprojection gap      : {worst_reproj:.3e} px")
    ok &= worst_rot < 1e-3 and worst_t < 1e-6 and worst_reproj < 1e-4

    ok &= charuco_pose_end_to_end()
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
