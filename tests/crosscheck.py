"""Cross-check fasttag against OpenCV's aruco on synthetic scenes."""
import numpy as np
import cv2
import fasttag

DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


def make_scene(cv_dict, ids, marker_px=80, pad=40, cols=3):
    """Tile markers on a white canvas at known positions."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv_dict)
    rows = (len(ids) + cols - 1) // cols
    cell = marker_px + 2 * pad
    canvas = np.full((rows * cell, cols * cell), 255, np.uint8)
    placed = {}
    for k, mid in enumerate(ids):
        r, c = divmod(k, cols)
        img = cv2.aruco.generateImageMarker(dictionary, mid, marker_px)
        y0, x0 = r * cell + pad, c * cell + pad
        canvas[y0:y0 + marker_px, x0:x0 + marker_px] = img
        placed[mid] = (x0, y0)
    return canvas, placed


def cv_detect(canvas, cv_dict):
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv_dict))
    corners, ids, _ = detector.detectMarkers(canvas)
    out = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            out[int(i)] = c.reshape(4, 2)
    return out


def main():
    total_ok = 0
    total = 0
    max_corner_err = 0.0
    for name, cv_dict in DICTS.items():
        n = 40 if "50" not in name else 30
        ids = list(range(0, n, 3))[:9]
        canvas, _ = make_scene(cv_dict, ids)

        cv_res = cv_detect(canvas, cv_dict)
        ft_corners, ft_ids = fasttag.detect_markers(canvas, name)
        ft_res = {int(i): np.array(c) for i, c in zip(ft_ids, ft_corners)}

        expected = set(ids)
        ft_found = set(ft_res.keys())
        cv_found = set(cv_res.keys())

        for mid in expected:
            total += 1
            if mid in ft_found:
                total_ok += 1
                if mid in cv_res:
                    # align corners (both clockwise, may differ by rotation start)
                    a = ft_res[mid]
                    b = cv_res[mid]
                    best = min(
                        np.abs(np.roll(a, k, axis=0) - b).max() for k in range(4)
                    )
                    max_corner_err = max(max_corner_err, best)
        print(
            f"{name:22s} expected={len(expected):2d} "
            f"fasttag={len(ft_found & expected):2d} opencv={len(cv_found & expected):2d} "
            f"false+={len(ft_found - expected)}"
        )

    print(f"\nTotal detected: {total_ok}/{total}")
    print(f"Max corner disagreement vs OpenCV (px): {max_corner_err:.3f}")


if __name__ == "__main__":
    main()
