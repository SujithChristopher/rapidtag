//! AprilTag 2 quadrilateral candidate detection, ported from the algorithm used
//! by OpenCV's `CORNER_REFINE_APRILTAG` path. Dictionary decoding stays in the
//! shared ArUco pipeline.

use image::GrayImage;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::detector::DetectorParameters;
use crate::imgproc::{self, Pt};

type Quad = [Pt; 4];

#[derive(Clone, Copy)]
struct EdgePoint {
    x2: i32,
    y2: i32,
    gx: i32,
    gy: i32,
}

struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            size: vec![1; n],
        }
    }

    fn root(&mut self, mut i: usize) -> usize {
        let mut r = i;
        while self.parent[r] != r {
            r = self.parent[r];
        }
        while self.parent[i] != r {
            let next = self.parent[i];
            self.parent[i] = r;
            i = next;
        }
        r
    }

    fn connect(&mut self, a: usize, b: usize) {
        let a = self.root(a);
        let b = self.root(b);
        if a == b {
            return;
        }
        if self.size[a] > self.size[b] {
            self.parent[b] = a;
            self.size[a] += self.size[b];
        } else {
            self.parent[a] = b;
            self.size[b] += self.size[a];
        }
    }
}

/// AprilTag's 4-pixel tiles, expanded by the extrema of their 3x3 neighbours.
/// A value of 127 marks a region with insufficient local contrast.
fn threshold(gray: &GrayImage, p: &DetectorParameters) -> Vec<u8> {
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    let (tw, th) = (w / 4, h / 4);
    let src = gray.as_raw();
    let mut out = vec![127; w * h];
    if tw == 0 || th == 0 {
        return out;
    }
    let mut lo = vec![255u8; tw * th];
    let mut hi = vec![0u8; tw * th];
    for ty in 0..th {
        for tx in 0..tw {
            for y in ty * 4..ty * 4 + 4 {
                for x in tx * 4..tx * 4 + 4 {
                    let v = src[y * w + x];
                    lo[ty * tw + tx] = lo[ty * tw + tx].min(v);
                    hi[ty * tw + tx] = hi[ty * tw + tx].max(v);
                }
            }
        }
    }
    let (mut expanded_lo, mut expanded_hi) = (vec![255u8; tw * th], vec![0u8; tw * th]);
    for ty in 0..th {
        for tx in 0..tw {
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let (nx, ny) = (tx as i32 + dx, ty as i32 + dy);
                    if nx >= 0 && nx < tw as i32 && ny >= 0 && ny < th as i32 {
                        let src_i = ny as usize * tw + nx as usize;
                        let dst_i = ty * tw + tx;
                        expanded_lo[dst_i] = expanded_lo[dst_i].min(lo[src_i]);
                        expanded_hi[dst_i] = expanded_hi[dst_i].max(hi[src_i]);
                    }
                }
            }
        }
    }
    for y in 0..h {
        for x in 0..w {
            let ti = (y / 4).min(th - 1) * tw + (x / 4).min(tw - 1);
            let min = expanded_lo[ti] as i32;
            let max = expanded_hi[ti] as i32;
            if max - min >= p.april_tag_min_white_black_diff {
                out[y * w + x] = if src[y * w + x] as i32 > (max + min) / 2 {
                    255
                } else {
                    0
                };
            }
        }
    }
    if p.april_tag_deglitch {
        let mut tmp = out.clone();
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                tmp[y * w + x] = (-1i32..=1)
                    .flat_map(|dy| (-1i32..=1).map(move |dx| (dx, dy)))
                    .map(|(dx, dy)| out[(y as i32 + dy) as usize * w + (x as i32 + dx) as usize])
                    .max()
                    .unwrap();
            }
        }
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                out[y * w + x] = (-1i32..=1)
                    .flat_map(|dy| (-1i32..=1).map(move |dx| (dx, dy)))
                    .map(|(dx, dy)| tmp[(y as i32 + dy) as usize * w + (x as i32 + dx) as usize])
                    .min()
                    .unwrap();
            }
        }
    }
    out
}

fn boundary_clusters(binary: &[u8], w: usize, h: usize, min_points: usize) -> Vec<Vec<EdgePoint>> {
    let mut uf = UnionFind::new(w * h);
    for y in 0..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let i = y * w + x;
            let v = binary[i];
            if v == 127 {
                continue;
            }
            let neighbors: &[(isize, isize)] = if v == 255 {
                &[(1, 0), (0, 1), (-1, 1), (1, 1)]
            } else {
                &[(1, 0), (0, 1)]
            };
            for &(dx, dy) in neighbors {
                let j = (y as isize + dy) as usize * w + (x as isize + dx) as usize;
                if binary[j] == v {
                    uf.connect(i, j);
                }
            }
        }
    }
    let mut clusters: FxHashMap<(usize, usize), Vec<EdgePoint>> = FxHashMap::default();
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let i = y * w + x;
            let v0 = binary[i];
            if v0 == 127 {
                continue;
            }
            let rep0 = uf.root(i);
            for (dx, dy) in [(1isize, 0isize), (0, 1), (-1, 1), (1, 1)] {
                let j = (y as isize + dy) as usize * w + (x as isize + dx) as usize;
                let v1 = binary[j];
                if v0 as u16 + v1 as u16 != 255 {
                    continue;
                }
                let rep1 = uf.root(j);
                let key = if rep0 < rep1 {
                    (rep0, rep1)
                } else {
                    (rep1, rep0)
                };
                clusters.entry(key).or_default().push(EdgePoint {
                    x2: 2 * x as i32 + dx as i32,
                    y2: 2 * y as i32 + dy as i32,
                    gx: dx as i32 * (v1 as i32 - v0 as i32),
                    gy: dy as i32 * (v1 as i32 - v0 as i32),
                });
            }
        }
    }
    let mut ordered: Vec<_> = clusters.into_iter().collect();
    ordered.retain(|(_, points)| points.len() >= min_points);
    ordered.sort_unstable_by_key(|(key, _)| *key);
    ordered.into_iter().map(|(_, points)| points).collect()
}

#[derive(Clone, Copy)]
struct Line {
    nx: f64,
    ny: f64,
    c: f64,
    mse: f64,
    error: f64,
}

struct LineFits {
    prefix: Vec<[f64; 6]>, // weight, x, y, xx, xy, yy
}

impl LineFits {
    fn new(points: &[EdgePoint], gray: &GrayImage) -> Self {
        let (w, h) = (gray.width() as i32, gray.height() as i32);
        let mut prefix = Vec::with_capacity(points.len() + 1);
        prefix.push([0.0; 6]);
        for p in points {
            let (x, y) = (p.x2 as f64 * 0.5 + 0.5, p.y2 as f64 * 0.5 + 0.5);
            let (ix, iy) = (x.floor() as i32, y.floor() as i32);
            let weight = if ix > 0 && ix + 1 < w && iy > 0 && iy + 1 < h {
                let px = |x: i32, y: i32| gray.get_pixel(x as u32, y as u32).0[0] as f64;
                (px(ix + 1, iy) - px(ix - 1, iy)).hypot(px(ix, iy + 1) - px(ix, iy - 1)) + 1.0
            } else {
                1.0
            };
            let value = [
                weight,
                weight * x,
                weight * y,
                weight * x * x,
                weight * x * y,
                weight * y * y,
            ];
            let previous = *prefix.last().unwrap();
            prefix.push(std::array::from_fn(|i| previous[i] + value[i]));
        }
        Self { prefix }
    }

    fn fit(&self, start: usize, end: usize) -> Option<Line> {
        let n = self.prefix.len() - 1;
        let count = if end >= start {
            end - start + 1
        } else {
            n - start + end + 1
        };
        if count < 3 {
            return None;
        }
        let sums: [f64; 6] = if end >= start {
            std::array::from_fn(|i| self.prefix[end + 1][i] - self.prefix[start][i])
        } else {
            std::array::from_fn(|i| {
                self.prefix[n][i] - self.prefix[start][i] + self.prefix[end + 1][i]
            })
        };
        if sums[0] <= 0.0 {
            return None;
        }
        let (cx, cy) = (sums[1] / sums[0], sums[2] / sums[0]);
        let (xx, xy, yy) = (
            sums[3] / sums[0] - cx * cx,
            sums[4] / sums[0] - cx * cy,
            sums[5] / sums[0] - cy * cy,
        );
        let angle = 0.5 * (2.0 * xy).atan2(xx - yy);
        let (nx, ny) = (-angle.sin(), angle.cos());
        let mse = (nx * nx * xx + 2.0 * nx * ny * xy + ny * ny * yy).max(0.0);
        Some(Line {
            nx,
            ny,
            c: -(nx * cx + ny * cy),
            mse,
            error: mse * count as f64,
        })
    }
}

fn intersection(a: Line, b: Line) -> Option<Pt> {
    let det = a.nx * b.ny - a.ny * b.nx;
    if det.abs() < 0.001 {
        return None;
    }
    let x = (a.ny * b.c - a.c * b.ny) / det;
    let y = (a.c * b.nx - a.nx * b.c) / det;
    if x.is_finite() && y.is_finite() {
        Some((x as f32, y as f32))
    } else {
        None
    }
}

fn fit_quad(mut points: Vec<EdgePoint>, gray: &GrayImage, p: &DetectorParameters) -> Option<Quad> {
    if points.len() < 24usize.max(p.april_tag_min_cluster_pixels as usize)
        || points.len() > 3 * (2 * gray.width() as usize + 2 * gray.height() as usize)
    {
        return None;
    }
    let (mut min_x, mut max_x, mut min_y, mut max_y) = (i32::MAX, i32::MIN, i32::MAX, i32::MIN);
    for q in &points {
        min_x = min_x.min(q.x2);
        max_x = max_x.max(q.x2);
        min_y = min_y.min(q.y2);
        max_y = max_y.max(q.y2);
    }
    let (cx, cy) = (
        (min_x + max_x) as f64 * 0.5 + 0.05118,
        (min_y + max_y) as f64 * 0.5 - 0.028581,
    );
    let polarity: f64 = points
        .iter()
        .map(|q| (q.x2 as f64 - cx) * q.gx as f64 + (q.y2 as f64 - cy) * q.gy as f64)
        .sum();
    if polarity < 0.0 {
        return None;
    }
    let mut angular: Vec<(f64, EdgePoint)> = points
        .into_iter()
        .map(|q| ((q.y2 as f64 - cy).atan2(q.x2 as f64 - cx), q))
        .collect();
    angular.sort_by(|a, b| a.0.total_cmp(&b.0));
    points = angular.into_iter().map(|(_, q)| q).collect();
    points.dedup_by_key(|q| (q.x2, q.y2));
    let n = points.len();
    let k = (n / 12).min(20);
    if k < 2 {
        return None;
    }
    let fits = LineFits::new(&points, gray);

    let mut errors = vec![0.0; n];
    for i in 0..n {
        errors[i] = fits.fit((i + n - k) % n, (i + k) % n)?.error;
    }
    let mut smooth = vec![0.0; n];
    let weights = [
        0.011108996,
        0.135335283,
        0.606530659,
        1.0,
        0.606530659,
        0.135335283,
        0.011108996,
    ];
    for i in 0..n {
        for j in 0..7 {
            smooth[i] += errors[(i + n + j - 3) % n] * weights[j];
        }
    }
    let mut maxima: Vec<usize> = (0..n)
        .filter(|&i| smooth[i] > smooth[(i + 1) % n] && smooth[i] > smooth[(i + n - 1) % n])
        .collect();
    if maxima.len() < 4 {
        return None;
    }
    if maxima.len() > p.april_tag_max_nmaxima as usize {
        maxima.sort_by(|&a, &b| smooth[b].total_cmp(&smooth[a]));
        maxima.truncate(p.april_tag_max_nmaxima as usize);
        maxima.sort_unstable();
    }
    let max_dot = p.april_tag_critical_rad.cos();
    let mut best: Option<(f64, [Line; 4])> = None;
    for a in 0..maxima.len() - 3 {
        for b in a + 1..maxima.len() - 2 {
            let Some(l0) = fits.fit(maxima[a], maxima[b]) else {
                continue;
            };
            if l0.mse > p.april_tag_max_line_fit_mse {
                continue;
            }
            for c in b + 1..maxima.len() - 1 {
                let Some(l1) = fits.fit(maxima[b], maxima[c]) else {
                    continue;
                };
                if l1.mse > p.april_tag_max_line_fit_mse
                    || (l0.nx * l1.nx + l0.ny * l1.ny).abs() > max_dot
                {
                    continue;
                }
                for d in c + 1..maxima.len() {
                    let Some(l2) = fits.fit(maxima[c], maxima[d]) else {
                        continue;
                    };
                    let Some(l3) = fits.fit(maxima[d], maxima[a]) else {
                        continue;
                    };
                    if l2.mse > p.april_tag_max_line_fit_mse
                        || l3.mse > p.april_tag_max_line_fit_mse
                    {
                        continue;
                    }
                    let error = l0.error + l1.error + l2.error + l3.error;
                    if best.is_none_or(|(old, _)| error < old) {
                        best = Some((error, [l0, l1, l2, l3]));
                    }
                }
            }
        }
    }
    let (error, lines) = best?;
    if error / n as f64 >= p.april_tag_max_line_fit_mse {
        return None;
    }
    let mut quad = [(0.0, 0.0); 4];
    for i in 0..4 {
        quad[i] = intersection(lines[(i + 3) % 4], lines[i])?;
    }
    if !imgproc::is_convex(&quad) {
        return None;
    }
    let area = (0..4)
        .map(|i| {
            let a = quad[i];
            let b = quad[(i + 1) % 4];
            a.0 * b.1 - a.1 * b.0
        })
        .sum::<f32>()
        .abs()
        * 0.5;
    if area < 64.0 {
        return None;
    }
    for i in 0..4 {
        let prev = quad[(i + 3) % 4];
        let cur = quad[i];
        let next = quad[(i + 1) % 4];
        let u = (prev.0 - cur.0, prev.1 - cur.1);
        let v = (next.0 - cur.0, next.1 - cur.1);
        let angle = ((u.0 * v.0 + u.1 * v.1) / (u.0.hypot(u.1) * v.0.hypot(v.1)))
            .clamp(-1.0, 1.0)
            .acos();
        if angle < p.april_tag_critical_rad as f32
            || angle > std::f32::consts::PI - p.april_tag_critical_rad as f32
        {
            return None;
        }
    }
    Some(quad)
}

pub fn candidates(gray: &GrayImage, p: &DetectorParameters) -> Vec<Quad> {
    let scale = p.april_tag_quad_decimate;
    let mut work = if scale > 1.0 {
        let w = ((gray.width() as f64 / scale).round() as u32).max(1);
        let h = ((gray.height() as f64 / scale).round() as u32).max(1);
        image::imageops::resize(gray, w, h, image::imageops::FilterType::Triangle)
    } else {
        gray.clone()
    };
    if p.april_tag_quad_sigma != 0.0 {
        let sigma = p.april_tag_quad_sigma.abs() as f32;
        let blurred = image::imageops::blur(&work, sigma);
        if p.april_tag_quad_sigma > 0.0 {
            work = blurred;
        } else {
            for (dst, low) in work.as_mut().iter_mut().zip(blurred.as_raw()) {
                *dst = (2 * *dst as i32 - *low as i32).clamp(0, 255) as u8;
            }
        }
    }
    let binary = threshold(&work, p);
    let clusters = boundary_clusters(
        &binary,
        work.width() as usize,
        work.height() as usize,
        24usize.max(p.april_tag_min_cluster_pixels as usize),
    );
    let mut quads: Vec<Quad> = clusters
        .into_par_iter()
        .map(|points| fit_quad(points, &work, p))
        .collect::<Vec<_>>()
        .into_iter()
        .flatten()
        .collect();
    if scale > 1.0 {
        for quad in &mut quads {
            for corner in quad {
                corner.0 *= scale as f32;
                corner.1 *= scale as f32;
            }
        }
    }
    quads
}
