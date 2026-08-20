//! Port of cv::getRectSubPix (8u -> 32f) and cv::cornerSubPix.
//!
//! ChArUco predicts each chessboard corner from marker homographies, which lands
//! within a pixel or two of the truth. `corner_sub_pix` does the rest: it fits the
//! local intensity gradient field, exploiting the fact that at a saddle-shaped
//! chessboard corner every nearby gradient is perpendicular to the vector pointing
//! at the corner. Solving that least-squares system gives sub-pixel position.

use image::GrayImage;

use crate::imgproc::Pt;

const MAX_ITERS: i32 = 100;

/// Bilinear patch extraction with replicated borders, matching getRectSubPix_8u32f.
///
/// Writes a `win_w` x `win_h` float patch centred on `center` into `dst`.
fn get_rect_sub_pix(src: &GrayImage, win_w: usize, win_h: usize, center: Pt, dst: &mut [f32]) {
    let w = src.width() as i64;
    let h = src.height() as i64;
    let sp = src.as_raw();

    // The patch is centred, so shift to its top-left sample position.
    let cx = center.0 - (win_w as f32 - 1.0) * 0.5;
    let cy = center.1 - (win_h as f32 - 1.0) * 0.5;
    let ipx = cx.floor() as i64;
    let ipy = cy.floor() as i64;
    let mut a = cx - ipx as f32;
    let b = cy - ipy as f32;

    let inside = ipx >= 0 && ipx + (win_w as i64) < w && ipy >= 0 && ipy + (win_h as i64) < h;
    if inside {
        // getRectSubPix_8u32f's fast path floors `a` away from zero. It does this
        // so its running-sum trick can divide by `a`; we don't use that trick, but
        // we keep the clamp because it perturbs the result and we match OpenCV.
        a = a.max(0.0001);
    }

    let a11 = (1.0 - a) * (1.0 - b);
    let a12 = a * (1.0 - b);
    let a21 = (1.0 - a) * b;
    let a22 = a * b;
    let b1 = 1.0 - b;
    let b2 = b;

    let at = |x: i64, y: i64| -> f32 { sp[(y * w + x) as usize] as f32 };

    if inside {
        for i in 0..win_h as i64 {
            let y0 = ipy + i;
            for j in 0..win_w as i64 {
                let x0 = ipx + j;
                dst[(i as usize) * win_w + j as usize] = at(x0, y0) * a11
                    + at(x0 + 1, y0) * a12
                    + at(x0, y0 + 1) * a21
                    + at(x0 + 1, y0 + 1) * a22;
            }
        }
        return;
    }

    // Border path. OpenCV walks this with adjusted pointers and a clamped rect;
    // clamping the sample coordinates instead is equivalent and far clearer. Where
    // both x-neighbours clamp to the same column the bilinear weights collapse to
    // the vertical pair (b1, b2), which is exactly what OpenCV's pad regions emit.
    let clamp = |v: i64, hi: i64| v.max(0).min(hi - 1);
    for i in 0..win_h as i64 {
        let y0 = clamp(ipy + i, h);
        let y1 = clamp(ipy + i + 1, h);
        for j in 0..win_w as i64 {
            let x0 = clamp(ipx + j, w);
            let x1 = clamp(ipx + j + 1, w);
            let v = if x0 == x1 {
                at(x0, y0) * b1 + at(x0, y1) * b2
            } else {
                at(x0, y0) * a11 + at(x1, y0) * a12 + at(x0, y1) * a21 + at(x1, y1) * a22
            };
            dst[(i as usize) * win_w + j as usize] = v;
        }
    }
}

/// Refine `corner` to sub-pixel accuracy against the local gradient field.
///
/// `win` is the half-size of the search window; `max_iters` / `eps` are the
/// termination criteria (`eps` in pixels). Returns the refined point. If the
/// refinement drifts further than the window, OpenCV discards it and keeps the
/// input, so we do too.
pub fn corner_sub_pix(
    src: &GrayImage,
    corner: Pt,
    win: usize,
    max_iters: i32,
    eps: f64,
) -> Pt {
    let win_w = win * 2 + 1;
    let win_h = win * 2 + 1;
    let max_iters = max_iters.clamp(1, MAX_ITERS);
    let eps = eps.max(0.0);
    let eps = eps * eps; // compared against squared step length

    // Gaussian weighting: centre samples count most.
    let mut mask = vec![0f32; win_w * win_h];
    for i in 0..win_h {
        let y = (i as f32 - win as f32) / win as f32;
        let vy = (-y * y).exp();
        for j in 0..win_w {
            let x = (j as f32 - win as f32) / win as f32;
            mask[i * win_w + j] = vy * (-x * x).exp();
        }
    }
    // NOTE: OpenCV also supports a zeroZone that blanks the mask centre. ChArUco
    // always passes Size() (disabled), so it is not ported.

    let ct = corner;
    let mut ci = corner;
    // Patch is one pixel larger on each side so central differences have neighbours.
    let stride = win_w + 2;
    let mut buf = vec![0f32; stride * (win_h + 2)];

    let (w, h) = (src.width() as f32, src.height() as f32);
    let contains = |p: Pt| p.0 >= 0.0 && p.1 >= 0.0 && p.0 < w && p.1 < h;
    if !contains(ct) {
        return ct;
    }

    let mut iter = 0;
    loop {
        get_rect_sub_pix(src, win_w + 2, win_h + 2, ci, &mut buf);

        // Accumulate the normal equations of the gradient-perpendicularity system.
        let (mut a, mut b, mut c, mut bb1, mut bb2) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for i in 0..win_h {
            let py = i as f64 - win as f64;
            for j in 0..win_w {
                let m = mask[i * win_w + j] as f64;
                let tgx = (buf[(i + 1) * stride + j + 2] - buf[(i + 1) * stride + j]) as f64;
                let tgy = (buf[(i + 2) * stride + j + 1] - buf[i * stride + j + 1]) as f64;
                let gxx = tgx * tgx * m;
                let gxy = tgx * tgy * m;
                let gyy = tgy * tgy * m;
                let px = j as f64 - win as f64;
                a += gxx;
                b += gxy;
                c += gyy;
                bb1 += gxx * px + gxy * py;
                bb2 += gxy * px + gyy * py;
            }
        }

        let det = a * c - b * b;
        if det.abs() <= f64::EPSILON * f64::EPSILON {
            break; // degenerate neighbourhood (flat or 1-D structure)
        }
        let scale = 1.0 / det;
        let ci2 = (
            (ci.0 as f64 + c * scale * bb1 - b * scale * bb2) as f32,
            (ci.1 as f64 - b * scale * bb1 + a * scale * bb2) as f32,
        );
        let err = ((ci2.0 - ci.0) as f64).powi(2) + ((ci2.1 - ci.1) as f64).powi(2);
        if !contains(ci2) {
            break;
        }
        ci = ci2;

        iter += 1;
        if iter >= max_iters || err <= eps {
            break;
        }
    }

    // A refinement that ran away is worse than none.
    if (ci.0 - ct.0).abs() > win as f32 || (ci.1 - ct.1).abs() > win as f32 {
        return ct;
    }
    ci
}
