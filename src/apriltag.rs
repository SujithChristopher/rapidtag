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

struct BoundaryEdge {
    a: u32,
    b: u32,
    point: EdgePoint,
}

struct UnionFind {
    parent: Vec<u32>,
    size: Vec<u32>,
}

impl UnionFind {
    fn new() -> Self {
        Self {
            parent: Vec::new(),
            size: Vec::new(),
        }
    }

    fn add(&mut self) -> u32 {
        let label = self.parent.len() as u32;
        self.parent.push(label);
        self.size.push(1);
        label
    }

    fn root(&mut self, mut i: u32) -> u32 {
        let mut r = i;
        while self.parent[r as usize] != r {
            r = self.parent[r as usize];
        }
        while self.parent[i as usize] != r {
            let next = self.parent[i as usize];
            self.parent[i as usize] = r;
            i = next;
        }
        r
    }

    fn connect(&mut self, a: u32, b: u32) {
        if self.parent[a as usize] == self.parent[b as usize] {
            return;
        }
        let a = self.root(a);
        let b = self.root(b);
        if a == b {
            return;
        }
        if self.size[a as usize] > self.size[b as usize] {
            self.parent[b as usize] = a;
            self.size[a as usize] += self.size[b as usize];
        } else {
            self.parent[a as usize] = b;
            self.size[b as usize] += self.size[a as usize];
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
    for ty in 0..th {
        for tx in 0..tw {
            let (mut min, mut max) = (255u8, 0u8);
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let (nx, ny) = (tx as i32 + dx, ty as i32 + dy);
                    if nx >= 0 && nx < tw as i32 && ny >= 0 && ny < th as i32 {
                        let src_i = ny as usize * tw + nx as usize;
                        min = min.min(lo[src_i]);
                        max = max.max(hi[src_i]);
                    }
                }
            }
            if max as i32 - min as i32 >= p.april_tag_min_white_black_diff {
                let cutoff = (max as i32 + min as i32) / 2;
                let y_end = if ty + 1 == th { h } else { (ty + 1) * 4 };
                let x_end = if tx + 1 == tw { w } else { (tx + 1) * 4 };
                for y in ty * 4..y_end {
                    for x in tx * 4..x_end {
                        let i = y * w + x;
                        out[i] = if src[i] as i32 > cutoff { 255 } else { 0 };
                    }
                }
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
    assert!(w * h < u32::MAX as usize);
    let mut uf = UnionFind::new();
    let mut labels = vec![u32::MAX; w * h];
    let mut edges = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let v = binary[i];
            if v == 127 {
                continue;
            }

            // Each horizontal run receives one label. The last row and first
            // column have no horizontal outgoing links in AprilTag's graph.
            let left_connected = y + 1 < h && x > 1 && binary[i - 1] == v;
            labels[i] = if left_connected {
                labels[i - 1]
            } else {
                uf.add()
            };

            // Reverse the original downward links. Only pixels in columns
            // 1..w-1 on the preceding row can be their source.
            if y > 0 {
                let start = if v == 255 { x.saturating_sub(1) } else { x };
                let end = if v == 255 { (x + 1).min(w - 1) } else { x };
                // A continuing run has already visited the overlap with the
                // preceding pixel's vertical neighborhood.
                for nx in (if left_connected { end } else { start })..=end {
                    if nx < 1 || nx + 1 >= w {
                        continue;
                    }
                    let j = (y - 1) * w + nx;
                    if binary[j] == v {
                        if left_connected && nx > 0 && labels[j - 1] == labels[j] {
                            continue;
                        }
                        uf.connect(labels[i], labels[j]);
                    }
                }
            }

            if y > 0 && y + 1 < h && x > 0 && x + 1 < w {
                for (dx, dy) in [(1isize, 0isize), (0, 1), (-1, 1), (1, 1)] {
                    let j = (y as isize + dy) as usize * w + (x as isize + dx) as usize;
                    let other = binary[j];
                    if v as u16 + other as u16 == 255 {
                        edges.push(BoundaryEdge {
                            a: i as u32,
                            b: j as u32,
                            point: EdgePoint {
                                x2: 2 * x as i32 + dx as i32,
                                y2: 2 * y as i32 + dy as i32,
                                gx: dx as i32 * (other as i32 - v as i32),
                                gy: dy as i32 * (other as i32 - v as i32),
                            },
                        });
                    }
                }
            }
        }
    }
    let mut clusters: FxHashMap<u64, Vec<EdgePoint>> = FxHashMap::default();
    for edge in edges {
        let rep0 = uf.root(labels[edge.a as usize]);
        let rep1 = uf.root(labels[edge.b as usize]);
        let key = ((rep0.min(rep1) as u64) << 32) | rep0.max(rep1) as u64;
        clusters.entry(key).or_default().push(edge.point);
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

/// Refit one known marker: fit only the edge cluster that traces the outer border
/// of `coarse` (in `gray` coordinates) instead of every cluster in the image, and
/// skip decoding. `modules` is the marker width in modules (data bits plus both
/// borders). Returns the fitted corners in `coarse`'s order.
pub fn refit_quad(
    gray: &GrayImage,
    coarse: &Quad,
    p: &DetectorParameters,
    modules: usize,
) -> Option<Quad> {
    let (outer, inner) = border_band(coarse, modules, p.marker_border_bits);
    let mut binary = threshold(gray, p);
    mask_outside_border_band(&mut binary, gray.width() as usize, coarse, outer, inner);
    let clusters = boundary_clusters(
        &binary,
        gray.width() as usize,
        gray.height() as usize,
        24usize.max(p.april_tag_min_cluster_pixels as usize),
    );
    // Inner bit edges sit at least one module inside the outer border.
    let tol = (0.5 * module_size(coarse, modules)).max(2.0);
    let near_border = |q: &EdgePoint| {
        let (x, y) = (q.x2 as f32 * 0.5 + 0.5, q.y2 as f32 * 0.5 + 0.5);
        (0..4).any(|i| segment_distance((x, y), coarse[i], coarse[(i + 1) % 4]) < tol)
    };
    let (near, border) = clusters
        .into_iter()
        .map(|points| (points.iter().filter(|q| near_border(q)).count(), points))
        .max_by_key(|(near, _)| *near)?;
    if near < 24 {
        return None;
    }
    let quad = fit_quad(border, gray, p)?;
    // Match the fitted corners to the coarse ordering (the fit's start is arbitrary).
    let cost = |r: usize| {
        (0..4)
            .map(|i| {
                let (f, c) = (quad[(i + r) % 4], coarse[i]);
                (f.0 - c.0).powi(2) + (f.1 - c.1).powi(2)
            })
            .sum::<f32>()
    };
    let rotation = (0..4).min_by(|&a, &b| cost(a).total_cmp(&cost(b)))?;
    Some(std::array::from_fn(|i| quad[(i + rotation) % 4]))
}

/// Longest side of `coarse` divided by the marker width in modules.
fn module_size(coarse: &Quad, modules: usize) -> f32 {
    let side = (0..4)
        .map(|i| {
            let (a, b) = (coarse[i], coarse[(i + 1) % 4]);
            (a.0 - b.0).hypot(a.1 - b.1)
        })
        .fold(0.0, f32::max);
    side / modules.max(1) as f32
}

/// Width of the band kept around a coarse border, in pixels outside and inside:
/// one module of quiet zone, the black border and half the first data row, plus
/// slack for the coarse corners' error.
pub fn border_band(coarse: &Quad, modules: usize, border_bits: i32) -> (f32, f32) {
    let module = module_size(coarse, modules);
    (module + 4.0, (border_bits.max(1) as f32 + 0.5) * module + 4.0)
}

/// Mark pixels away from the coarse border as "no contrast" (127) so the
/// clustering skips the tag interior and the background.
fn mask_outside_border_band(binary: &mut [u8], w: usize, coarse: &Quad, outer: f32, inner: f32) {
    let area2: f32 = (0..4)
        .map(|i| {
            let (a, b) = (coarse[i], coarse[(i + 1) % 4]);
            a.0 * b.1 - a.1 * b.0
        })
        .sum();
    let orient = if area2 >= 0.0 { 1.0 } else { -1.0 };
    // Signed distance to each edge line, positive inside the (convex) quad.
    let lines: [(f32, f32, f32); 4] = std::array::from_fn(|i| {
        let (a, b) = (coarse[i], coarse[(i + 1) % 4]);
        let len = (b.0 - a.0).hypot(b.1 - a.1).max(1e-6);
        let (nx, ny) = (-(b.1 - a.1) * orient / len, (b.0 - a.0) * orient / len);
        (nx, ny, -(nx * a.0 + ny * a.1))
    });
    for (y, row) in binary.chunks_exact_mut(w).enumerate() {
        let py = y as f32 + 0.5;
        for (x, v) in row.iter_mut().enumerate() {
            let px = x as f32 + 0.5;
            let d = lines
                .iter()
                .map(|&(nx, ny, c)| nx * px + ny * py + c)
                .fold(f32::MAX, f32::min);
            if d < -outer || d > inner {
                *v = 127;
            }
        }
    }
}

fn segment_distance(p: Pt, a: Pt, b: Pt) -> f32 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len2 = dx * dx + dy * dy;
    let t = if len2 > 0.0 {
        (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (p.0 - a.0 - t * dx).hypot(p.1 - a.1 - t * dy)
}
