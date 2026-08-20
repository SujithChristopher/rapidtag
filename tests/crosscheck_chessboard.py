"""Cross-check RapidTag's generic chessboard detector against OpenCV.

The cases cover different rectangular grids, a frontal board, perspective
distortion, rotation, blur, uneven illumination, and negative images. OpenCV's
classic detector can reverse the entire grid for some even-by-even patterns, so
the accuracy comparison accepts that equivalent ordering while still requiring
RapidTag's output itself to form the requested row-major grid.
"""

import cv2
import numpy as np
import rapidtag


CASES = [(9, 6), (7, 5), (4, 4)]
SQUARE = 58
MARGIN = 70


def make_board(pattern_size):
    width, height = pattern_size
    cols, rows = width + 1, height + 1
    board = np.full((rows * SQUARE, cols * SQUARE), 255, np.uint8)
    for y in range(rows):
        for x in range(cols):
            if (x + y) % 2 == 0:
                board[y * SQUARE : (y + 1) * SQUARE, x * SQUARE : (x + 1) * SQUARE] = 0
    return cv2.copyMakeBorder(board, MARGIN, MARGIN, MARGIN, MARGIN, cv2.BORDER_CONSTANT, value=255)


def perspective(img):
    h, w = img.shape
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([[45, 30], [w - 70, 5], [w - 15, h - 50], [20, h - 5]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, matrix, (w, h), borderValue=255)


def rotated(img):
    h, w = img.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 17.0, 0.90)
    return cv2.warpAffine(img, matrix, (w, h), borderValue=255)


def uneven_light(img):
    h, w = img.shape
    shade = np.linspace(0.55, 1.0, w, dtype=np.float32)[None, :]
    lit = np.clip(img.astype(np.float32) * shade + 22.0, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(lit, (5, 5), 1.0)


def opencv_detect(img, pattern_size):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(img, pattern_size, flags)
    if not found:
        return False, np.empty((0, 2), np.float32)
    return True, corners.reshape(-1, 2)


def rapidtag_detect(img, pattern_size):
    found, corners = rapidtag.find_chessboard_corners(img, pattern_size)
    return found, np.asarray(corners, np.float32).reshape(-1, 2)


def ordering_is_grid(corners, pattern_size):
    width, height = pattern_size
    grid = corners.reshape(height, width, 2)
    horizontal = np.linalg.norm(np.diff(grid, axis=1), axis=2)
    vertical = np.linalg.norm(np.diff(grid, axis=0), axis=2)
    if np.any(horizontal < 3) or np.any(vertical < 3):
        return False
    # Each row and column must move consistently rather than doubling back.
    for lines in (grid, np.swapaxes(grid, 0, 1)):
        for line in lines:
            axis = line[-1] - line[0]
            steps = np.diff(line, axis=0)
            if np.any(steps @ axis <= 0):
                return False
    return True


def main():
    failures = []
    worst_error = 0.0
    matched = 0
    total = 0

    for pattern_size in CASES:
        base = make_board(pattern_size)
        views = [
            ("frontal", base),
            ("perspective", perspective(base)),
            ("rotated", rotated(base)),
            ("uneven", uneven_light(perspective(base))),
            ("bgr", cv2.cvtColor(perspective(base), cv2.COLOR_GRAY2BGR)),
        ]
        for name, img in views:
            cv_found, cv_corners = opencv_detect(img, pattern_size)
            rt_found, rt_corners = rapidtag_detect(img, pattern_size)
            total += 1
            error = float("inf")
            grid_ok = False
            if rt_found and len(rt_corners) == pattern_size[0] * pattern_size[1]:
                grid_ok = ordering_is_grid(rt_corners, pattern_size)
            if cv_found and rt_found and len(cv_corners) == len(rt_corners):
                direct = np.max(np.abs(cv_corners - rt_corners))
                reverse = np.max(np.abs(cv_corners - rt_corners[::-1]))
                error = float(min(direct, reverse))
                worst_error = max(worst_error, error)
                matched += 1
            # On blurred, unevenly lit boards OpenCV's 2px cornerSubPix pass is
            # start-dependent; both detectors remain within two pixels.
            ok = cv_found and rt_found and grid_ok and error < 2.0
            if not ok:
                failures.append(
                    f"{pattern_size} {name}: cv={cv_found} rt={rt_found} "
                    f"count={len(rt_corners)} grid={grid_ok} error={error:.4f}px"
                )
            print(
                f"board={pattern_size[0]}x{pattern_size[1]} {name:11s} "
                f"cv={cv_found} rt={rt_found} corners={len(rt_corners):2d} "
                f"grid={grid_ok} max_err={error:.4f}px"
            )

    negatives = {
        "blank": np.full((480, 640), 180, np.uint8),
        "noise": np.random.default_rng(4).integers(0, 256, (480, 640), np.uint8),
    }
    for name, img in negatives.items():
        found, corners = rapidtag_detect(img, (9, 6))
        print(f"negative={name:5s} found={found} corners={len(corners)}")
        if found or len(corners):
            failures.append(f"negative {name}: false positive with {len(corners)} corners")

    print(f"\nMatched positive cases: {matched}/{total}")
    print(f"Worst corner disagreement vs OpenCV: {worst_error:.4f}px")
    if failures:
        print("\nFailures:")
        for failure in failures:
            print("  " + failure)
    ok = not failures and matched == total
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
