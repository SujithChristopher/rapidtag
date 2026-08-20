//! Port of cv::aruco::CharucoDetector::detectBoard (charuco_detector.cpp).
//!
//! The chessboard corners are never searched for directly. Instead each detected
//! ArUco marker gives an exact board->image homography (4 point pairs), and every
//! chessboard corner is predicted by the homographies of the markers touching it.
//! Those predictions are then refined to sub-pixel accuracy against the image.
//! This is why ChArUco needs no chessboard finder, and why it tolerates occlusion:
//! any corner whose neighbouring markers were seen can still be recovered.

use image::GrayImage;

use crate::board::CharucoBoard;
use crate::cornersubpix::corner_sub_pix;
use crate::detector::{self, DetectorParameters};
use crate::dictionary::Dictionary;
use crate::imgproc::{self, Pt};

/// Sentinel written by the interpolation step for corners it could not predict.
const NOT_INTERPOLATED: Pt = (-1.0, -1.0);

#[derive(Clone)]
pub struct CharucoParameters {
    /// Adjacent markers that must be detected to accept a corner (0..=2).
    pub min_markers: i32,
    /// Verify the detected markers really belong to this board.
    pub check_markers: bool,
}

impl Default for CharucoParameters {
    fn default() -> Self {
        CharucoParameters {
            min_markers: 2,
            check_markers: true,
        }
        // NOTE: OpenCV also has tryRefineMarkers (refineDetectedMarkers, default
        // off) and the cameraMatrix/distCoeffs approxCalib path. Neither is ported
        // yet; the local-homography path below is the no-camera default.
    }
}

pub struct CharucoResult {
    /// Refined chessboard corners, in image coordinates.
    pub corners: Vec<Pt>,
    /// Index of each corner within the board's chessboard_corners.
    pub ids: Vec<usize>,
    /// The ArUco markers detected along the way.
    pub marker_corners: Vec<[Pt; 4]>,
    pub marker_ids: Vec<i32>,
}

/// cv::Rect::contains(Point2f) rounds the point to int first (Point_ converts via
/// saturate_cast). cvRound is round-half-to-even.
fn cv_round(v: f32) -> i64 {
    (v as f64).round_ties_even() as i64
}

/// 3x3 determinant of a row-major homography.
fn det3(m: &[f64; 9]) -> f64 {
    m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
}

/// cv::perspectiveTransform for a single 2D point. A vanishing denominator maps to
/// the origin, matching OpenCV.
fn perspective_transform(p: Pt, m: &[f64; 9]) -> Pt {
    let (x, y) = (p.0 as f64, p.1 as f64);
    let w = x * m[6] + y * m[7] + m[8];
    if w.abs() > f32::EPSILON as f64 {
        let w = 1.0 / w;
        (
            ((x * m[0] + y * m[1] + m[2]) * w) as f32,
            ((x * m[3] + y * m[4] + m[5]) * w) as f32,
        )
    } else {
        (0.0, 0.0)
    }
}

fn dist(a: Pt, b: Pt) -> f64 {
    (((a.0 - b.0) as f64).powi(2) + ((a.1 - b.1) as f64).powi(2)).sqrt()
}

/// Find the index of `marker_id` within the detected marker ids.
fn find_marker(marker_ids: &[i32], marker_id: usize) -> Option<usize> {
    marker_ids.iter().position(|&k| k == marker_id as i32)
}

/// interpolateCornersCharucoLocalHom: predict every chessboard corner from the
/// homographies of the markers nearest to it.
fn interpolate_local_hom(
    board: &CharucoBoard,
    marker_corners: &[[Pt; 4]],
    marker_ids: &[i32],
) -> Vec<Pt> {
    let n_markers = marker_ids.len();
    // Per-marker board->image homography. Exact, not fitted: 4 correspondences.
    let mut transforms: Vec<[f64; 9]> = vec![[0.0; 9]; n_markers];
    let mut valid = vec![false; n_markers];

    for i in 0..n_markers {
        let Some(board_idx) = board.ids.iter().position(|&id| id as i32 == marker_ids[i]) else {
            continue; // detected a marker that isn't on this board
        };
        let op = &board.obj_points[board_idx];
        let src: [Pt; 4] = [
            (op[0].0, op[0].1),
            (op[1].0, op[1].1),
            (op[2].0, op[2].1),
            (op[3].0, op[3].1),
        ];
        let h = imgproc::get_perspective_transform(&src, &marker_corners[i]);
        valid[i] = det3(&h).abs() > 1e-6; // reject singular transforms
        transforms[i] = h;
    }

    let mut out = vec![NOT_INTERPOLATED; board.chessboard_corners.len()];
    for i in 0..board.chessboard_corners.len() {
        let cc = board.chessboard_corners[i];
        let obj2d = (cc.0, cc.1);
        let mut predictions: Vec<Pt> = Vec::new();
        for j in 0..board.nearest_marker_idx[i].len() {
            let marker_id = board.ids[board.nearest_marker_idx[i][j]];
            let Some(idx) = find_marker(marker_ids, marker_id) else {
                continue;
            };
            if valid[idx] {
                predictions.push(perspective_transform(obj2d, &transforms[idx]));
            }
        }
        if predictions.is_empty() {
            continue; // none of this corner's markers were detected
        }
        // Two markers touch an interior corner; averaging their predictions cancels
        // some of each one's error.
        out[i] = if predictions.len() > 1 {
            (
                (predictions[0].0 + predictions[1].0) / 2.0,
                (predictions[0].1 + predictions[1].1) / 2.0,
            )
        } else {
            predictions[0]
        };
    }
    out
}

/// getMaximumSubPixWindowSizes: cap each corner's refinement window at its distance
/// to the nearest marker corner, so refinement cannot be pulled onto a marker edge.
/// `-1` means "no constraint, use the detector default".
fn max_sub_pix_win_sizes(
    board: &CharucoBoard,
    marker_corners: &[[Pt; 4]],
    marker_ids: &[i32],
    charuco_corners: &[Pt],
) -> Vec<i32> {
    let mut win = vec![-1i32; charuco_corners.len()];
    for i in 0..charuco_corners.len() {
        if charuco_corners[i] == NOT_INTERPOLATED || board.nearest_marker_idx[i].is_empty() {
            continue;
        }
        let mut min_dist = -1f64;
        let mut counter = 0;
        for j in 0..board.nearest_marker_idx[i].len() {
            let marker_id = board.ids[board.nearest_marker_idx[i][j]];
            let Some(idx) = find_marker(marker_ids, marker_id) else {
                continue;
            };
            let mc = marker_corners[idx][board.nearest_marker_corners[i][j]];
            let d = dist(mc, charuco_corners[i]);
            if min_dist == -1.0 {
                min_dist = d;
            }
            min_dist = d.min(min_dist);
            counter += 1;
        }
        if counter == 0 {
            continue;
        }
        // 2px of safety margin, then clamp to OpenCV's 1..=10.
        win[i] = ((min_dist - 2.0) as i32).clamp(1, 10);
    }
    win
}

/// selectAndRefineChessboardCorners: drop corners outside the image, then refine.
fn select_and_refine(
    gray: &GrayImage,
    all_corners: &[Pt],
    win_sizes: &[i32],
    p: &DetectorParameters,
) -> (Vec<Pt>, Vec<usize>) {
    const MIN_DIST_TO_BORDER: i64 = 2;
    let w = gray.width() as i64;
    let h = gray.height() as i64;

    let mut corners = Vec::new();
    let mut ids = Vec::new();
    let mut wins = Vec::new();
    for (i, &c) in all_corners.iter().enumerate() {
        // cv::Rect(2, 2, cols-4, rows-4).contains() on a rounded point.
        let (x, y) = (cv_round(c.0), cv_round(c.1));
        if x >= MIN_DIST_TO_BORDER
            && x < w - MIN_DIST_TO_BORDER
            && y >= MIN_DIST_TO_BORDER
            && y < h - MIN_DIST_TO_BORDER
        {
            corners.push(c);
            ids.push(i);
            wins.push(win_sizes[i]);
        }
    }
    if corners.is_empty() {
        return (corners, ids);
    }

    for (c, &win) in corners.iter_mut().zip(wins.iter()) {
        let win = if win == -1 {
            p.corner_refinement_win_size
        } else {
            win
        };
        *c = corner_sub_pix(
            gray,
            *c,
            win as usize,
            p.corner_refinement_max_iterations,
            p.corner_refinement_min_accuracy,
        );
    }
    (corners, ids)
}

/// filterCornersWithoutMinMarkers: require `min_markers` of a corner's neighbouring
/// markers to have actually been detected.
fn filter_min_markers(
    board: &CharucoBoard,
    corners: &[Pt],
    ids: &[usize],
    marker_ids: &[i32],
    min_markers: i32,
) -> (Vec<Pt>, Vec<usize>) {
    let mut out_c = Vec::new();
    let mut out_i = Vec::new();
    for (k, &ch_id) in ids.iter().enumerate() {
        let mut total = 0;
        for m in 0..board.nearest_marker_idx[ch_id].len() {
            let marker_id = board.ids[board.nearest_marker_idx[ch_id][m]];
            if find_marker(marker_ids, marker_id).is_some() {
                total += 1;
            }
        }
        if total >= min_markers {
            out_i.push(ch_id);
            out_c.push(corners[k]);
        }
    }
    (out_c, out_i)
}

/// checkBoard: reject a board whose corners sit closer to markers that should not
/// form them than to the ones that should — i.e. the markers aren't this board.
fn check_board(
    board: &CharucoBoard,
    marker_corners: &[[Pt; 4]],
    marker_ids: &[i32],
    ch_corners: &[Pt],
    ch_ids: &[usize],
) -> bool {
    // distance[i].0: max distance from corner i to its corner-forming markers
    // distance[i].1: min distance from corner i to any other marker
    let mut distance = vec![(0f32, f32::MAX); board.nearest_marker_idx.len()];

    for (i, &ch_id) in ch_ids.iter().enumerate() {
        let cc = ch_corners[i];
        for (j, &m_id) in marker_ids.iter().enumerate() {
            if !board.ids.iter().any(|&b| b as i32 == m_id) {
                continue; // marker isn't on this board
            }
            let mc = &marker_corners[j];
            let center = (
                (mc[0].0 + mc[1].0 + mc[2].0 + mc[3].0) / 4.0,
                (mc[0].1 + mc[1].1 + mc[2].1 + mc[3].1) / 4.0,
            );
            let d = dist(center, cc) as f32;

            // Every interior chessboard corner touches exactly two marker squares,
            // so OpenCV indexes [1] unchecked. Fall back rather than panic.
            let near1 = board.ids[board.nearest_marker_idx[ch_id][0]] as i32;
            let near2 = board
                .nearest_marker_idx[ch_id]
                .get(1)
                .map(|&b| board.ids[b] as i32)
                .unwrap_or(near1);
            if near1 == m_id || near2 == m_id {
                let nearest_corner_id = if near1 == m_id {
                    board.nearest_marker_corners[ch_id][0]
                } else {
                    board.nearest_marker_corners[ch_id][1]
                };
                let nearest_corner = mc[nearest_corner_id];
                let dist_to_nearest = dist(nearest_corner, cc) as f32;
                distance[ch_id].0 = distance[ch_id].0.max(dist_to_nearest);
                // The corner should be nearer to the marker's corner than to the
                // midpoints of its adjacent edges; otherwise the layout is wrong.
                let mid1 = (
                    (mc[(nearest_corner_id + 1) % 4].0 + nearest_corner.0) * 0.5,
                    (mc[(nearest_corner_id + 1) % 4].1 + nearest_corner.1) * 0.5,
                );
                let mid2 = (
                    (mc[(nearest_corner_id + 3) % 4].0 + nearest_corner.0) * 0.5,
                    (mc[(nearest_corner_id + 3) % 4].1 + nearest_corner.1) * 0.5,
                );
                let tmp = (dist(mid1, cc) as f32).min(dist(mid2, cc) as f32);
                if tmp < dist_to_nearest {
                    return false;
                }
            } else {
                distance[ch_id].1 = distance[ch_id].1.min(d);
            }
        }
        if distance[ch_id].0 > 0.0 && distance[ch_id].1 < f32::MAX && distance[ch_id].0 > distance[ch_id].1 {
            return false;
        }
    }
    true
}

/// Test hook: the interpolated corners and window sizes, before refinement.
pub fn debug_predict(
    gray: &GrayImage,
    board: &CharucoBoard,
    dict: &Dictionary,
    p: &DetectorParameters,
) -> (Vec<Pt>, Vec<i32>) {
    let (detections, _) = detector::detect_markers(gray, dict, p);
    let mc: Vec<[Pt; 4]> = detections.iter().map(|d| d.corners).collect();
    let mi: Vec<i32> = detections.iter().map(|d| d.id).collect();
    let predicted = interpolate_local_hom(board, &mc, &mi);
    let wins = max_sub_pix_win_sizes(board, &mc, &mi, &predicted);
    (predicted, wins)
}

/// Detect a ChArUco board: find its markers, interpolate the chessboard corners,
/// and refine them to sub-pixel accuracy.
pub fn detect_board(
    gray: &GrayImage,
    board: &CharucoBoard,
    dict: &Dictionary,
    p: &DetectorParameters,
    cp: &CharucoParameters,
) -> CharucoResult {
    let (detections, _) = detector::detect_markers(gray, dict, p);
    let marker_corners: Vec<[Pt; 4]> = detections.iter().map(|d| d.corners).collect();
    let marker_ids: Vec<i32> = detections.iter().map(|d| d.id).collect();

    let empty = CharucoResult {
        corners: Vec::new(),
        ids: Vec::new(),
        marker_corners: marker_corners.clone(),
        marker_ids: marker_ids.clone(),
    };
    if marker_ids.is_empty() {
        return empty;
    }

    let predicted = interpolate_local_hom(board, &marker_corners, &marker_ids);
    let win_sizes = max_sub_pix_win_sizes(board, &marker_corners, &marker_ids, &predicted);
    let (corners, ids) = select_and_refine(gray, &predicted, &win_sizes, p);
    let (corners, ids) = filter_min_markers(board, &corners, &ids, &marker_ids, cp.min_markers);

    if cp.check_markers && !check_board(board, &marker_corners, &marker_ids, &corners, &ids) {
        return empty;
    }

    CharucoResult {
        corners,
        ids,
        marker_corners,
        marker_ids,
    }
}
