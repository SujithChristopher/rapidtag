//! Contour-line refinement of a detected marker quadrilateral.

use crate::imgproc::{self, Pt};

type Quad = [Pt; 4];

fn fit_line(points: &[Pt]) -> Option<(f32, f32, f32)> {
    if points.len() < 3 {
        return None;
    }
    let n = points.len() as f32;
    let cx = points.iter().map(|p| p.0).sum::<f32>() / n;
    let cy = points.iter().map(|p| p.1).sum::<f32>() / n;
    let (mut xx, mut xy, mut yy) = (0.0, 0.0, 0.0);
    for &(x, y) in points {
        let dx = x - cx;
        let dy = y - cy;
        xx += dx * dx;
        xy += dx * dy;
        yy += dy * dy;
    }
    if xx + yy < 1e-4 {
        return None;
    }
    // Dominant eigenvector of the covariance matrix is the line direction.
    let angle = 0.5 * (2.0 * xy).atan2(xx - yy);
    let (a, b) = (-angle.sin(), angle.cos());
    Some((a, b, -(a * cx + b * cy)))
}

fn intersect(a: (f32, f32, f32), b: (f32, f32, f32)) -> Option<Pt> {
    let det = a.0 * b.1 - a.1 * b.0;
    if det.abs() < 0.1 {
        return None;
    }
    Some(((a.1 * b.2 - a.2 * b.1) / det, (a.2 * b.0 - a.0 * b.2) / det))
}

fn refine_from_edges(corners: &Quad, edges: &[Vec<Pt>; 4], max_move: f32) -> Quad {
    let lines: Option<Vec<_>> = edges.iter().map(|e| fit_line(e)).collect();
    let Some(lines) = lines else { return *corners };
    let mut out = *corners;
    for i in 0..4 {
        let Some(p) = intersect(lines[(i + 3) % 4], lines[i]) else {
            return *corners;
        };
        if !p.0.is_finite()
            || !p.1.is_finite()
            || (p.0 - corners[i].0).hypot(p.1 - corners[i].1) > max_move
        {
            return *corners;
        }
        out[i] = p;
    }
    if !imgproc::is_convex(&out) {
        return *corners;
    }
    out
}

/// Fit side lines to contour pixels near the middle of each polygon side.
pub fn contour(corners: &Quad, contour: &[(i32, i32)]) -> Quad {
    let mut edges: [Vec<Pt>; 4] = std::array::from_fn(|_| Vec::new());
    for &(xi, yi) in contour {
        let (x, y) = (xi as f32, yi as f32);
        let mut best = (f32::MAX, 4usize);
        for i in 0..4 {
            let a = corners[i];
            let b = corners[(i + 1) % 4];
            let (dx, dy) = (b.0 - a.0, b.1 - a.1);
            let len2 = dx * dx + dy * dy;
            if len2 < 1.0 {
                continue;
            }
            let t = ((x - a.0) * dx + (y - a.1) * dy) / len2;
            if !(0.1..=0.9).contains(&t) {
                continue;
            }
            let distance = ((x - a.0) * dy - (y - a.1) * dx).abs() / len2.sqrt();
            if distance < best.0 {
                best = (distance, i);
            }
        }
        if best.0 <= 3.0 {
            edges[best.1].push((x, y));
        }
    }
    refine_from_edges(corners, &edges, 5.0)
}
