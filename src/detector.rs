//! Port of the cv::aruco::ArucoDetector::detectMarkers pipeline (CORNER_REFINE_NONE).

use crate::contours::for_each_contour;
use crate::dictionary::{Dictionary, DEFAULT_VALID_BIT_ID_THRESHOLD};
use crate::imgproc::{self, Pt};
use image::GrayImage;
use rayon::prelude::*;

#[derive(Clone)]
pub struct DetectorParameters {
    pub adaptive_thresh_win_size_min: i32,
    pub adaptive_thresh_win_size_max: i32,
    pub adaptive_thresh_win_size_step: i32,
    pub adaptive_thresh_constant: f64,
    pub min_marker_perimeter_rate: f64,
    pub max_marker_perimeter_rate: f64,
    pub polygonal_approx_accuracy_rate: f64,
    pub min_corner_distance_rate: f64,
    pub min_distance_to_border: i32,
    pub min_marker_distance_rate: f64,
    pub marker_border_bits: i32,
    pub perspective_remove_pixel_per_cell: i32,
    pub perspective_remove_ignored_margin_per_cell: f64,
    pub max_erroneous_bits_in_border_rate: f64,
    pub min_otsu_std_dev: f64,
    pub error_correction_rate: f64,
    pub detect_inverted_marker: bool,
    pub min_side_length_canonical_img: i32,
    pub valid_bit_id_threshold: f32,
}

impl Default for DetectorParameters {
    fn default() -> Self {
        DetectorParameters {
            adaptive_thresh_win_size_min: 3,
            adaptive_thresh_win_size_max: 23,
            adaptive_thresh_win_size_step: 10,
            adaptive_thresh_constant: 7.0,
            min_marker_perimeter_rate: 0.03,
            max_marker_perimeter_rate: 4.0,
            polygonal_approx_accuracy_rate: 0.03,
            min_corner_distance_rate: 0.05,
            min_distance_to_border: 3,
            min_marker_distance_rate: 0.125,
            marker_border_bits: 1,
            perspective_remove_pixel_per_cell: 4,
            perspective_remove_ignored_margin_per_cell: 0.13,
            max_erroneous_bits_in_border_rate: 0.35,
            min_otsu_std_dev: 5.0,
            error_correction_rate: 0.6,
            detect_inverted_marker: false,
            min_side_length_canonical_img: 32,
            valid_bit_id_threshold: DEFAULT_VALID_BIT_ID_THRESHOLD,
        }
    }
}

type Quad = [Pt; 4];

/// Extract square marker candidates from an already-thresholded image.
fn find_marker_contours(thresh: &GrayImage, p: &DetectorParameters) -> Vec<Quad> {
    let (w, h) = (thresh.width() as f64, thresh.height() as f64);
    let max_wh = w.max(h);

    let mut min_perimeter = (p.min_marker_perimeter_rate * max_wh) as usize;
    let max_perimeter = (p.max_marker_perimeter_rate * max_wh) as usize;
    // OpenCV overrides the min with 4*minSideLengthCanonicalImg when non-zero.
    if p.min_side_length_canonical_img != 0 {
        min_perimeter = 4 * p.min_side_length_canonical_img as usize;
    }

    let mut out = Vec::new();
    let mut pts: Vec<Pt> = Vec::new(); // reused f32 buffer
    // Trace + filter fused: most contours are tiny noise and are rejected by the
    // size test before we ever allocate for them.
    for_each_contour(thresh, |contour| {
        let n = contour.len();
        if n < min_perimeter || n > max_perimeter {
            return;
        }
        pts.clear();
        pts.extend(contour.iter().map(|&(x, y)| (x as f32, y as f32)));
        let eps = (n as f64 * p.polygonal_approx_accuracy_rate) as f32;
        let approx = imgproc::approx_poly_dp_closed(&pts, eps);
        if approx.len() != 4 || !imgproc::is_convex(&approx) {
            return;
        }
        // min distance between corners
        let mut min_dist_sq = (max_wh * max_wh) as f32;
        for j in 0..4 {
            let a = approx[j];
            let b = approx[(j + 1) % 4];
            let d = (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2);
            min_dist_sq = min_dist_sq.min(d);
        }
        let min_corner = (n as f64 * p.min_corner_distance_rate) as f32;
        if min_dist_sq < min_corner * min_corner {
            return;
        }
        out.push([approx[0], approx[1], approx[2], approx[3]]);
    });
    out
}

/// Ensure the candidate corners run clockwise (positive cross product).
fn reorder_corners(c: &mut Quad) {
    let dx1 = c[1].0 - c[0].0;
    let dy1 = c[1].1 - c[0].1;
    let dx2 = c[2].0 - c[0].0;
    let dy2 = c[2].1 - c[0].1;
    if (dx1 * dy2 - dy1 * dx2) < 0.0 {
        c.swap(1, 3);
    }
}

fn perimeter(c: &Quad) -> f32 {
    let mut p = 0.0;
    for i in 0..4 {
        let a = c[i];
        let b = c[(i + 1) % 4];
        p += ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt();
    }
    p
}

/// Average corner distance minimised over the 4 possible corner alignments.
fn average_distance(a: &Quad, b: &Quad) -> f32 {
    let mut min_sq = f32::MAX;
    for fc in 0..4 {
        let mut d = 0.0;
        for c in 0..4 {
            let m = (c + fc) % 4;
            d += (a[m].0 - b[c].0).powi(2) + (a[m].1 - b[c].1).powi(2);
        }
        d /= 4.0;
        min_sq = min_sq.min(d);
    }
    min_sq.sqrt()
}

/// Group near-duplicate candidates, keep the largest of each group, drop those
/// touching the border. Mirrors filterTooCloseCandidates (non-inverted case).
fn filter_too_close(
    candidates: &[Quad],
    p: &DetectorParameters,
    w: f32,
    h: f32,
) -> Vec<Quad> {
    let n = candidates.len();
    // sort indices by perimeter descending (stable)
    let perims: Vec<f32> = candidates.iter().map(perimeter).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| perims[b].partial_cmp(&perims[a]).unwrap());
    // ct[k] is the k-th biggest candidate
    let ct: Vec<Quad> = order.iter().map(|&i| candidates[i]).collect();
    let ct_perim: Vec<f32> = order.iter().map(|&i| perims[i]).collect();

    let mut group_id = vec![-1i32; n];
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut selected_alone = vec![true; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let dist = average_distance(&ct[i], &ct[j]);
            if dist < ct_perim[j] * p.min_marker_distance_rate as f32 {
                selected_alone[i] = false;
                selected_alone[j] = false;
                if group_id[i] < 0 && group_id[j] < 0 {
                    group_id[i] = groups.len() as i32;
                    group_id[j] = groups.len() as i32;
                    groups.push(vec![i, j]);
                } else if group_id[i] > -1 && group_id[j] == -1 {
                    let g = group_id[i] as usize;
                    group_id[j] = g as i32;
                    groups[g].push(j);
                } else if group_id[j] > -1 && group_id[i] == -1 {
                    let g = group_id[j] as usize;
                    group_id[i] = g as i32;
                    groups[g].push(i);
                }
            }
        }
        if selected_alone[i] {
            selected_alone[i] = false;
            group_id[i] = groups.len() as i32;
            groups.push(vec![i]);
        }
    }

    let mut out = Vec::new();
    for mut grouped in groups {
        // non-inverted: pick largest -> smallest ct index
        grouped.sort_unstable();
        let cur = ct[grouped[0]];
        let too_near = cur.iter().any(|&(x, y)| {
            x < p.min_distance_to_border as f32
                || y < p.min_distance_to_border as f32
                || x > w - 1.0 - p.min_distance_to_border as f32
                || y > h - 1.0 - p.min_distance_to_border as f32
        });
        if !too_near {
            out.push(cur);
        }
    }
    out
}

/// Extract per-cell white-pixel ratios (including border cells) via perspective removal.
fn extract_cell_ratio(gray: &GrayImage, corners: &Quad, marker_size: usize, p: &DetectorParameters) -> Vec<f32> {
    let border = p.marker_border_bits as usize;
    let cell = p.perspective_remove_pixel_per_cell as usize;
    let size_with_borders = marker_size + 2 * border;
    let result_size = size_with_borders * cell;
    let margin = (p.perspective_remove_ignored_margin_per_cell * cell as f64) as usize;

    let result = imgproc::warp_marker(gray, corners, result_size);

    // inner region for the Otsu std-dev check
    let lo = cell / 2;
    let hi = result_size - cell / 2;
    let mut inner = Vec::with_capacity((hi - lo) * (hi - lo));
    for y in lo..hi {
        for x in lo..hi {
            inner.push(result[y * result_size + x]);
        }
    }
    let (mean, std) = imgproc::mean_std(&inner);

    let mut ratios = vec![0f32; size_with_borders * size_with_borders];
    if std < p.min_otsu_std_dev {
        let v = if mean > 127.0 { 1.0 } else { 0.0 };
        ratios.iter_mut().for_each(|r| *r = v);
        return ratios;
    }

    let level = imgproc::otsu_level(&result);
    let bin: Vec<bool> = result.iter().map(|&v| v > level).collect();

    for y in 0..size_with_borders {
        for x in 0..size_with_borders {
            let xstart = x * cell + margin;
            let ystart = y * cell + margin;
            let csz = cell - 2 * margin;
            let mut nz = 0usize;
            for dy in 0..csz {
                for dx in 0..csz {
                    if bin[(ystart + dy) * result_size + (xstart + dx)] {
                        nz += 1;
                    }
                }
            }
            ratios[y * size_with_borders + x] = nz as f32 / (csz * csz) as f32;
        }
    }
    ratios
}

/// Count erroneous border cells (black-border and inverted-border variants).
fn border_errors(ratios: &[f32], marker_size: usize, border: usize, thr: f32) -> (i32, i32) {
    let s = marker_size + 2 * border;
    let inv = 1.0 - thr;
    let (mut be, mut ibe) = (0i32, 0i32);
    let mut count = |r: f32| {
        if r > thr {
            be += 1;
        }
        if r < inv {
            ibe += 1;
        }
    };
    for y in 0..s {
        for k in 0..border {
            count(ratios[y * s + k]);
            count(ratios[y * s + (s - 1 - k)]);
        }
    }
    for x in border..(s - border) {
        for k in 0..border {
            count(ratios[k * s + x]);
            count(ratios[(s - 1 - k) * s + x]);
        }
    }
    (be, ibe)
}

/// Try to identify one candidate. Returns (id, rotation) if valid.
fn identify_one(
    dict: &Dictionary,
    gray: &GrayImage,
    corners: &Quad,
    p: &DetectorParameters,
) -> Option<(usize, usize)> {
    let ms = dict.marker_size;
    let border = p.marker_border_bits as usize;
    let mut ratios = extract_cell_ratio(gray, corners, ms, p);

    let max_border_errors = (ms * ms) as f64 * p.max_erroneous_bits_in_border_rate;
    let (mut be, ibe) = border_errors(&ratios, ms, border, p.valid_bit_id_threshold);

    if p.detect_inverted_marker && ibe < be {
        be = ibe;
        ratios.iter_mut().for_each(|r| *r = 1.0 - *r);
    }
    if be as f64 > max_border_errors {
        return None;
    }

    // inner cells only
    let s = ms + 2 * border;
    let mut inner = vec![0f32; ms * ms];
    for y in 0..ms {
        for x in 0..ms {
            inner[y * ms + x] = ratios[(y + border) * s + (x + border)];
        }
    }
    dict.identify(&inner, p.error_correction_rate, p.valid_bit_id_threshold)
}

pub struct Detection {
    pub corners: Quad,
    pub id: i32,
}

/// Detect and identify markers of `dict` in `gray`.
///
/// `parallel_scales` runs the adaptive-threshold scales across threads. Enable it
/// for single-frame calls; disable it when the caller already parallelizes across
/// frames (batch mode) to avoid nested-rayon oversubscription. Output is identical
/// either way — scale results are always concatenated in scale order.
fn n_scales(p: &DetectorParameters) -> i32 {
    (p.adaptive_thresh_win_size_max - p.adaptive_thresh_win_size_min) / p.adaptive_thresh_win_size_step
        + 1
}

/// Candidate quads found at one threshold scale (STEP 1 for a single scale).
/// `parallel_threshold` parallelizes the integral+compare internally; use it when
/// scales are computed sequentially (single frame), off when the caller already
/// parallelizes across frames/scales (batch).
fn candidates_for_scale(
    gray: &GrayImage,
    p: &DetectorParameters,
    scale_i: i32,
    parallel_threshold: bool,
) -> Vec<Quad> {
    let win = p.adaptive_thresh_win_size_min + scale_i * p.adaptive_thresh_win_size_step;
    let thresh = imgproc::adaptive_threshold_mean_inv(
        gray,
        win as u32,
        p.adaptive_thresh_constant,
        parallel_threshold,
    );
    find_marker_contours(&thresh, p)
}

/// STEP 2 + 3: reorder, drop near-duplicates, identify. `candidates` must already
/// be concatenated in scale order to match OpenCV's output.
fn finalize(
    gray: &GrayImage,
    mut candidates: Vec<Quad>,
    dict: &Dictionary,
    p: &DetectorParameters,
) -> (Vec<Detection>, Vec<Quad>) {
    let (w, h) = (gray.width() as f32, gray.height() as f32);
    for c in candidates.iter_mut() {
        reorder_corners(c);
    }
    let selected = filter_too_close(&candidates, p, w, h);

    let mut detections = Vec::new();
    let mut rejected = Vec::new();
    for cand in selected {
        match identify_one(dict, gray, &cand, p) {
            Some((id, rotation)) => {
                let mut c = cand;
                c.rotate_left((4 - rotation) % 4); // correctCornerPosition
                detections.push(Detection { corners: c, id: id as i32 });
            }
            None => rejected.push(cand),
        }
    }
    (detections, rejected)
}

/// Detect markers in a single frame. The threshold scales run on separate cores.
pub fn detect_markers(
    gray: &GrayImage,
    dict: &Dictionary,
    p: &DetectorParameters,
) -> (Vec<Detection>, Vec<Quad>) {
    let per_scale: Vec<Vec<Quad>> = (0..n_scales(p))
        .into_par_iter()
        .map(|i| candidates_for_scale(gray, p, i, false))
        .collect();
    let candidates: Vec<Quad> = per_scale.into_iter().flatten().collect();
    finalize(gray, candidates, dict, p)
}

/// Detect markers across many frames using flat (frame × scale) parallelism.
///
/// This one scheme is optimal at every batch size: 1 frame keeps the 3 scales on
/// 3 cores, a dual-camera pair uses 6, and a large offline batch fills all cores —
/// with no nested rayon. Output order and values match `detect_markers` per frame.
pub fn detect_markers_multi(
    grays: Vec<GrayImage>,
    dict: &Dictionary,
    p: &DetectorParameters,
) -> Vec<(Vec<Detection>, Vec<Quad>)> {
    let ns = n_scales(p);
    // Flat work list ordered (frame, scale) with scale innermost, so per-frame
    // results stay in scale order when regrouped.
    let work: Vec<(usize, i32)> = (0..grays.len())
        .flat_map(|f| (0..ns).map(move |s| (f, s)))
        .collect();
    let partial: Vec<Vec<Quad>> = work
        .par_iter()
        .map(|&(f, s)| candidates_for_scale(&grays[f], p, s, false))
        .collect();

    // Regroup candidates per frame (partial is in the same (frame, scale) order).
    let mut per_frame: Vec<Vec<Quad>> = vec![Vec::new(); grays.len()];
    for (idx, quads) in partial.into_iter().enumerate() {
        per_frame[work[idx].0].extend(quads);
    }

    // Finalize each frame in parallel.
    per_frame
        .into_par_iter()
        .zip(grays.into_par_iter())
        .map(|(cands, gray)| finalize(&gray, cands, dict, p))
        .collect()
}
