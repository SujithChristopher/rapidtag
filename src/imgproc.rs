//! Image-processing primitives ported to match the OpenCV routines the aruco
//! detector relies on: adaptive mean threshold, Otsu, polygon approximation,
//! convexity test, and perspective removal.

use image::GrayImage;
use nalgebra::{SMatrix, SVector};
use rayon::prelude::*;

type Matrix8 = SMatrix<f64, 8, 8>;
type Vector8 = SVector<f64, 8>;

pub type Pt = (f32, f32);

/// Build a GrayImage from an interleaved BGR (h, w, 3) or gray (h, w) u8 buffer.
/// For grayscale the contiguous buffer is used directly (zero conversion).
/// BGR->gray uses OpenCV's coefficients (0.114 B + 0.587 G + 0.299 R).
pub fn to_gray(data: Vec<u8>, h: usize, w: usize, channels: usize) -> GrayImage {
    if channels == 1 {
        // reuse the buffer as-is
        return GrayImage::from_raw(w as u32, h as u32, data).expect("gray buffer size");
    }
    let mut out = vec![0u8; w * h];
    for (px, chunk) in out.iter_mut().zip(data.chunks_exact(channels)) {
        let b = chunk[0] as f32;
        let g = chunk[1] as f32;
        let r = chunk[2] as f32;
        // OpenCV rounds: CV_DESCALE with 0.5 offset
        *px = ((0.114 * b + 0.587 * g + 0.299 * r + 0.5) as u32).min(255) as u8;
    }
    GrayImage::from_raw(w as u32, h as u32, out).expect("gray buffer size")
}

/// Adaptive threshold, ADAPTIVE_THRESH_MEAN_C + THRESH_BINARY_INV.
/// For each pixel: T = mean(win x win neighbourhood) - c; out = src > T ? 0 : 255.
/// Uses a clamped-window mean (border pixels divide by the true covered area).
/// `parallel` row-parallelizes the compare loop (use for single-frame; leave off
/// when the caller already parallelizes across frames).
pub fn adaptive_threshold_mean_inv(src: &GrayImage, mut win: u32, c: f64, parallel: bool) -> GrayImage {
    if win < 3 {
        win = 3;
    }
    if win % 2 == 0 {
        win += 1;
    }
    let (w, h) = (src.width() as i64, src.height() as i64);
    let radius = (win / 2) as i64;
    let sp = src.as_raw();

    // Integral image (w+1)x(h+1). i32 holds sums up to 255*W*H (~2.1e9), which
    // covers images up to ~8.4 MP — well beyond typical marker-detection frames.
    let iw = (w + 1) as usize;
    let ih = (h + 1) as usize;
    let mut integral = vec![0i32; iw * ih];
    for y in 0..h as usize {
        let mut row_sum = 0i32;
        let row = &sp[y * w as usize..(y + 1) * w as usize];
        let (head, tail) = integral.split_at_mut((y + 1) * iw);
        let prev = &head[y * iw..(y + 1) * iw];
        let dst = &mut tail[0..iw];
        for x in 0..w as usize {
            row_sum += row[x] as i32;
            dst[x + 1] = prev[x + 1] + row_sum;
        }
    }
    // Integer form of `src > round(mean) - C`, with mean = sum/area:
    //   dst=0  ⟺  src > floor(sum/area + 0.5) - C  ⟺  2*sum + area < 2*area*(src + C)
    // No per-pixel division → autovectorizes under target-cpu=native.
    let c_int = c as i32;
    let wu = w as usize;
    let mut out = vec![0u8; wu * h as usize];

    // Per-output-row compare against the integral image (read-only, shared).
    let compute_row = |y: i64, out_row: &mut [u8]| {
        let y0 = (y - radius).max(0);
        let y1 = (y + radius).min(h - 1);
        let yspan = (y1 - y0 + 1) as i32;
        let iy0 = y0 as usize * iw;
        let iy1 = (y1 + 1) as usize * iw;
        let sp_row = &sp[(y * w) as usize..(y * w) as usize + wu];
        for x in 0..wu {
            let xi = x as i64;
            let x0 = (xi - radius).max(0) as usize;
            let x1 = (xi + radius).min(w - 1) as usize;
            let area = (x1 - x0 + 1) as i32 * yspan;
            let sum = integral[iy1 + x1 + 1] - integral[iy0 + x1 + 1] - integral[iy1 + x0]
                + integral[iy0 + x0];
            let v = sp_row[x] as i32;
            out_row[x] = if 2 * sum + area < 2 * area * (v + c_int) { 0u8 } else { 255u8 };
        }
    };

    if parallel {
        out.par_chunks_mut(wu)
            .enumerate()
            .for_each(|(y, row)| compute_row(y as i64, row));
    } else {
        out.chunks_mut(wu)
            .enumerate()
            .for_each(|(y, row)| compute_row(y as i64, row));
    }
    GrayImage::from_raw(w as u32, h as u32, out).expect("threshold buffer size")
}

/// Otsu threshold level over a grayscale buffer (0..=255 histogram).
pub fn otsu_level(pixels: &[u8]) -> u8 {
    let mut hist = [0u64; 256];
    for &p in pixels {
        hist[p as usize] += 1;
    }
    let total = pixels.len() as f64;
    let sum: f64 = (0..256).map(|i| i as f64 * hist[i] as f64).sum();
    let mut sum_b = 0.0;
    let mut w_b = 0.0;
    let mut max_var = -1.0;
    let mut thresh = 0u8;
    for t in 0..256 {
        w_b += hist[t] as f64;
        if w_b == 0.0 {
            continue;
        }
        let w_f = total - w_b;
        if w_f == 0.0 {
            break;
        }
        sum_b += t as f64 * hist[t] as f64;
        let m_b = sum_b / w_b;
        let m_f = (sum - sum_b) / w_f;
        let var_between = w_b * w_f * (m_b - m_f) * (m_b - m_f);
        if var_between > max_var {
            max_var = var_between;
            thresh = t as u8;
        }
    }
    thresh
}

/// Population mean and std-dev of a grayscale buffer.
pub fn mean_std(pixels: &[u8]) -> (f64, f64) {
    let n = pixels.len() as f64;
    if n == 0.0 {
        return (0.0, 0.0);
    }
    let mean = pixels.iter().map(|&p| p as f64).sum::<f64>() / n;
    let var = pixels.iter().map(|&p| (p as f64 - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

/// Perpendicular distance from point p to the line through a-b.
fn perp_dist(p: Pt, a: Pt, b: Pt) -> f32 {
    let dx = b.0 - a.0;
    let dy = b.1 - a.1;
    let denom = (dx * dx + dy * dy).sqrt();
    if denom < 1e-9 {
        let ex = p.0 - a.0;
        let ey = p.1 - a.1;
        return (ex * ex + ey * ey).sqrt();
    }
    ((p.0 - a.0) * dy - (p.1 - a.1) * dx).abs() / denom
}

/// Douglas-Peucker on an open chain (inclusive endpoints).
fn dp_open(pts: &[Pt], eps: f32, out: &mut Vec<Pt>) {
    if pts.len() < 2 {
        return;
    }
    let (a, b) = (pts[0], pts[pts.len() - 1]);
    let mut idx = 0;
    let mut dmax = 0.0;
    for i in 1..pts.len() - 1 {
        let d = perp_dist(pts[i], a, b);
        if d > dmax {
            dmax = d;
            idx = i;
        }
    }
    if dmax > eps {
        dp_open(&pts[..=idx], eps, out);
        out.pop(); // avoid duplicating the split point
        dp_open(&pts[idx..], eps, out);
    } else {
        out.push(a);
        out.push(b);
    }
}

/// Polygon approximation of a *closed* contour, akin to cv::approxPolyDP(closed=true).
/// Splits the loop at its two most distant vertices then runs DP on each half.
pub fn approx_poly_dp_closed(pts: &[Pt], eps: f32) -> Vec<Pt> {
    let n = pts.len();
    if n < 3 {
        return pts.to_vec();
    }
    // farthest point from pts[0]
    let far = |from: Pt| -> usize {
        let mut bi = 0;
        let mut bd = -1.0;
        for (i, &q) in pts.iter().enumerate() {
            let d = (q.0 - from.0).powi(2) + (q.1 - from.1).powi(2);
            if d > bd {
                bd = d;
                bi = i;
            }
        }
        bi
    };
    let s = far(pts[0]);
    let e = far(pts[s]);
    let (lo, hi) = (s.min(e), s.max(e));

    // chain lo..=hi and hi..=end+0..=lo (wrap)
    let chain1: Vec<Pt> = pts[lo..=hi].to_vec();
    let mut chain2: Vec<Pt> = pts[hi..n].to_vec();
    chain2.extend_from_slice(&pts[0..=lo]);

    let mut res = Vec::new();
    let mut a = Vec::new();
    dp_open(&chain1, eps, &mut a);
    let mut b = Vec::new();
    dp_open(&chain2, eps, &mut b);

    // stitch, dropping shared endpoints
    if !a.is_empty() {
        res.extend_from_slice(&a[..a.len() - 1]);
    }
    if !b.is_empty() {
        res.extend_from_slice(&b[..b.len() - 1]);
    }
    res
}

/// True if the 4 (or more) vertex polygon is convex (consistent turn direction).
pub fn is_convex(pts: &[Pt]) -> bool {
    let n = pts.len();
    if n < 3 {
        return false;
    }
    let mut sign = 0i32;
    for i in 0..n {
        let a = pts[i];
        let b = pts[(i + 1) % n];
        let c = pts[(i + 2) % n];
        let cross = (b.0 - a.0) * (c.1 - b.1) - (b.1 - a.1) * (c.0 - b.0);
        let s = if cross > 0.0 {
            1
        } else if cross < 0.0 {
            -1
        } else {
            0
        };
        if s != 0 {
            if sign == 0 {
                sign = s;
            } else if sign != s {
                return false;
            }
        }
    }
    true
}

/// Solve for the 3x3 perspective transform mapping src[i] -> dst[i]. Returns row-major H (h8=1).
pub fn get_perspective_transform(src: &[Pt; 4], dst: &[Pt; 4]) -> [f64; 9] {
    let mut a = Matrix8::zeros();
    let mut b = Vector8::zeros();
    for i in 0..4 {
        let (x, y) = (src[i].0 as f64, src[i].1 as f64);
        let (u, v) = (dst[i].0 as f64, dst[i].1 as f64);
        let r0 = 2 * i;
        let r1 = 2 * i + 1;
        a[(r0, 0)] = x;
        a[(r0, 1)] = y;
        a[(r0, 2)] = 1.0;
        a[(r0, 6)] = -x * u;
        a[(r0, 7)] = -y * u;
        b[r0] = u;

        a[(r1, 3)] = x;
        a[(r1, 4)] = y;
        a[(r1, 5)] = 1.0;
        a[(r1, 6)] = -x * v;
        a[(r1, 7)] = -y * v;
        b[r1] = v;
    }
    let h = a.lu().solve(&b).expect("perspective solve failed");
    [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0]
}

/// Warp `src` into a `size`x`size` image using perspective removal, INTER_NEAREST.
/// `corners` are the marker corners in the source image (clockwise).
pub fn warp_marker(src: &GrayImage, corners: &[Pt; 4], size: usize) -> Vec<u8> {
    let s = (size - 1) as f32;
    let dst_corners = [(0.0, 0.0), (s, 0.0), (s, s), (0.0, s)];
    // map result-space -> image-space directly (no matrix inversion needed)
    let h = get_perspective_transform(&dst_corners, corners);
    let (w, ht) = (src.width() as i32, src.height() as i32);
    let sp = src.as_raw();
    let mut out = vec![0u8; size * size];
    for y in 0..size {
        for x in 0..size {
            let fx = x as f64;
            let fy = y as f64;
            let denom = h[6] * fx + h[7] * fy + h[8];
            let sx = (h[0] * fx + h[1] * fy + h[2]) / denom;
            let sy = (h[3] * fx + h[4] * fy + h[5]) / denom;
            let ix = sx.round() as i32;
            let iy = sy.round() as i32;
            if ix >= 0 && iy >= 0 && ix < w && iy < ht {
                out[y * size + x] = sp[(iy * w + ix) as usize];
            }
        }
    }
    out
}
