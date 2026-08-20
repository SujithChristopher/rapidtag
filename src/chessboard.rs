//! Generic chessboard corner detection, compatible with the default
//! `cv::findChessboardCorners` workflow.
//!
//! The detector follows the same high-level idea as OpenCV's classic detector:
//! threshold the image, recover the dark square quadrilaterals, connect shared
//! square vertices into a grid, validate that grid against the requested pattern
//! size, and finally refine every intersection with `cornerSubPix`.

use std::collections::{HashMap, HashSet, VecDeque};

use image::GrayImage;

use crate::contours::{self, FG};
use crate::cornersubpix::corner_sub_pix;
use crate::imgproc::{self, Pt};

#[derive(Clone, Debug)]
struct Quad {
    corners: [Pt; 4],
    min_edge: f32,
}

#[derive(Debug)]
struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DisjointSet {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        if self.rank[ra] == self.rank[rb] {
            self.rank[ra] += 1;
        }
    }
}

#[inline]
fn sqr_dist(a: Pt, b: Pt) -> f32 {
    (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)
}

fn polygon_area(points: &[Pt]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        sum += a.0 * b.1 - a.1 * b.0;
    }
    0.5 * sum.abs()
}

fn order_quad(mut points: Vec<Pt>) -> Option<[Pt; 4]> {
    if points.len() != 4 {
        return None;
    }
    let center = (
        points.iter().map(|p| p.0).sum::<f32>() * 0.25,
        points.iter().map(|p| p.1).sum::<f32>() * 0.25,
    );
    points.sort_by(|a, b| {
        (a.1 - center.1)
            .atan2(a.0 - center.0)
            .total_cmp(&(b.1 - center.1).atan2(b.0 - center.0))
    });
    let q = [points[0], points[1], points[2], points[3]];
    if imgproc::is_convex(&q) {
        Some(q)
    } else {
        None
    }
}

/// Erode a padded foreground-label image. Dark chessboard squares touch only at
/// a point; one erosion separates those diagonal contacts so each square gets its
/// own contour. The coordinates remain symmetric around the true intersection,
/// so averaging the two diagonal vertices recovers the corner.
fn erode_labels(src: &[i8], w: usize, h: usize) -> Vec<i8> {
    let stride = w + 2;
    let mut dst = vec![0i8; src.len()];
    if w < 3 || h < 3 {
        return dst;
    }
    for y in 1..h - 1 {
        let base = (y + 1) * stride + 1;
        for x in 1..w - 1 {
            let i = base + x;
            let mut keep = true;
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    let j = (i as isize + dy * stride as isize + dx) as usize;
                    keep &= src[j] != 0;
                }
            }
            if keep {
                dst[i] = FG;
            }
        }
    }
    dst
}

fn global_threshold_labels(gray: &GrayImage) -> Vec<i8> {
    let w = gray.width() as usize;
    let h = gray.height() as usize;
    let stride = w + 2;
    let threshold = imgproc::otsu_level(gray.as_raw());
    let mut labels = vec![0i8; stride * (h + 2)];
    for y in 0..h {
        let row = &gray.as_raw()[y * w..(y + 1) * w];
        let out = (y + 1) * stride + 1;
        for (x, &v) in row.iter().enumerate() {
            if v <= threshold {
                labels[out + x] = FG;
            }
        }
    }
    labels
}

fn extract_quads(mut labels: Vec<i8>, w: usize, h: usize) -> Vec<Quad> {
    let min_area = 25.0f32;
    let max_area = (w * h) as f32 * 0.35;
    let mut quads = Vec::new();

    contours::for_each_contour(&mut labels, w as i32, h as i32, |contour| {
        if contour.len() < 12 {
            return;
        }
        let points: Vec<Pt> = contour.iter().map(|&(x, y)| (x as f32, y as f32)).collect();
        let mut approx = Vec::new();
        for epsilon in 1..=7 {
            approx = imgproc::approx_poly_dp_closed(&points, epsilon as f32);
            if approx.len() <= 4 {
                break;
            }
        }
        let Some(corners) = order_quad(approx) else {
            return;
        };
        let area = polygon_area(&corners);
        if area < min_area || area > max_area {
            return;
        }

        let mut sides = [0.0f32; 4];
        for i in 0..4 {
            sides[i] = sqr_dist(corners[i], corners[(i + 1) & 3]).sqrt();
        }
        let min_edge = sides.iter().copied().fold(f32::INFINITY, f32::min);
        let max_edge = sides.iter().copied().fold(0.0f32, f32::max);
        if min_edge < 4.0 || max_edge > min_edge * 4.0 {
            return;
        }
        let d0 = sqr_dist(corners[0], corners[2]).sqrt();
        let d1 = sqr_dist(corners[1], corners[3]).sqrt();
        let perimeter: f32 = sides.iter().sum();
        if d0 < perimeter * 0.15 || d1 < perimeter * 0.15 || sides[0] * sides[1] >= area * 1.8 {
            return;
        }
        quads.push(Quad { corners, min_edge });
    });
    quads
}

fn bfs_distances(
    adjacency: &[HashSet<usize>],
    start: usize,
    allowed: &HashSet<usize>,
) -> Vec<usize> {
    let mut dist = vec![usize::MAX; adjacency.len()];
    let mut queue = VecDeque::new();
    dist[start] = 0;
    queue.push_back(start);
    while let Some(v) = queue.pop_front() {
        for &next in &adjacency[v] {
            if allowed.contains(&next) && dist[next] == usize::MAX {
                dist[next] = dist[v] + 1;
                queue.push_back(next);
            }
        }
    }
    dist
}

fn validate_order(
    adjacency: &[HashSet<usize>],
    component: &HashSet<usize>,
    width: usize,
    height: usize,
    a: usize,
    b: usize,
) -> Option<Vec<usize>> {
    let da = bfs_distances(adjacency, a, component);
    let db = bfs_distances(adjacency, b, component);
    let mut grid = vec![usize::MAX; width * height];
    for &v in component {
        if da[v] == usize::MAX || db[v] == usize::MAX {
            return None;
        }
        let numerator = da[v] as isize - db[v] as isize + width as isize - 1;
        if numerator < 0 || numerator % 2 != 0 {
            return None;
        }
        let x = (numerator / 2) as usize;
        if x >= width || da[v] < x {
            return None;
        }
        let y = da[v] - x;
        if y >= height || grid[y * width + x] != usize::MAX {
            return None;
        }
        grid[y * width + x] = v;
    }
    if grid.contains(&usize::MAX) {
        return None;
    }

    for y in 0..height {
        for x in 0..width {
            let v = grid[y * width + x];
            let expected = (x > 0) as usize
                + (x + 1 < width) as usize
                + (y > 0) as usize
                + (y + 1 < height) as usize;
            if adjacency[v]
                .iter()
                .filter(|n| component.contains(n))
                .count()
                != expected
            {
                return None;
            }
            if x + 1 < width && !adjacency[v].contains(&grid[y * width + x + 1]) {
                return None;
            }
            if y + 1 < height && !adjacency[v].contains(&grid[(y + 1) * width + x]) {
                return None;
            }
        }
    }
    Some(grid)
}

fn canonical_score(
    order: &[usize],
    points: &[Pt],
    width: usize,
    height: usize,
) -> (i32, i32, i32, i32) {
    let p0 = points[order[0]];
    let pr = points[order[width - 1]];
    let pd = points[order[(height - 1) * width]];
    let cross = (pr.0 - p0.0) * (pd.1 - p0.1) - (pr.1 - p0.1) * (pd.0 - p0.0);
    let orientation_penalty = if cross >= 0.0 { 0 } else { 1 };
    let right_penalty = if pr.0 >= p0.0 { 0 } else { 1 };
    (
        right_penalty,
        orientation_penalty,
        (p0.1 * 1024.0) as i32,
        (p0.0 * 1024.0) as i32,
    )
}

fn order_component(
    adjacency: &[HashSet<usize>],
    component: &[usize],
    points: &[Pt],
    width: usize,
    height: usize,
) -> Option<Vec<usize>> {
    if component.len() != width * height {
        return None;
    }
    let allowed: HashSet<usize> = component.iter().copied().collect();
    let edge_count: usize = component
        .iter()
        .map(|&v| adjacency[v].iter().filter(|n| allowed.contains(n)).count())
        .sum::<usize>()
        / 2;
    if edge_count != width * (height - 1) + height * (width - 1) {
        return None;
    }
    let corners: Vec<usize> = component
        .iter()
        .copied()
        .filter(|&v| adjacency[v].iter().filter(|n| allowed.contains(n)).count() == 2)
        .collect();
    if corners.len() != 4 {
        return None;
    }

    type ScoredOrder = ((i32, i32, i32, i32), Vec<usize>);
    let mut best: Option<ScoredOrder> = None;
    for &a in &corners {
        let da = bfs_distances(adjacency, a, &allowed);
        for &b in &corners {
            if a == b || da[b] != width - 1 {
                continue;
            }
            let Some(order) = validate_order(adjacency, &allowed, width, height, a, b) else {
                continue;
            };
            let score = canonical_score(&order, points, width, height);
            if best.as_ref().is_none_or(|(old, _)| score < *old) {
                best = Some((score, order));
            }
        }
    }
    best.map(|(_, order)| order)
}

fn grid_from_quads(quads: &[Quad], width: usize, height: usize) -> Option<Vec<Pt>> {
    if quads.len() < 2 {
        return None;
    }
    let vertex_count = quads.len() * 4;
    let mut sets = DisjointSet::new(vertex_count);
    for qa in 0..quads.len() {
        for qb in qa + 1..quads.len() {
            let radius = quads[qa].min_edge.min(quads[qb].min_edge) * 0.32;
            let radius2 = radius * radius;
            for a in 0..4 {
                for b in 0..4 {
                    if sqr_dist(quads[qa].corners[a], quads[qb].corners[b]) <= radius2 {
                        sets.union(qa * 4 + a, qb * 4 + b);
                    }
                }
            }
        }
    }

    let mut members: HashMap<usize, Vec<usize>> = HashMap::new();
    for v in 0..vertex_count {
        let root = sets.find(v);
        members.entry(root).or_default().push(v);
    }

    let mut root_to_corner = HashMap::new();
    let mut points = Vec::new();
    for (root, vertices) in &members {
        let unique_quads: HashSet<usize> = vertices.iter().map(|v| v / 4).collect();
        // A true interior chessboard intersection is shared by exactly two
        // diagonally opposite dark squares.
        if vertices.len() != 2 || unique_quads.len() != 2 {
            continue;
        }
        let mut p = (0.0f32, 0.0f32);
        for &v in vertices {
            let q = v / 4;
            let c = v & 3;
            p.0 += quads[q].corners[c].0;
            p.1 += quads[q].corners[c].1;
        }
        p.0 *= 0.5;
        p.1 *= 0.5;
        root_to_corner.insert(*root, points.len());
        points.push(p);
    }
    if points.len() < width * height {
        return None;
    }

    let mut adjacency = vec![HashSet::new(); points.len()];
    for q in 0..quads.len() {
        let mut ids = [None; 4];
        for (c, slot) in ids.iter_mut().enumerate() {
            let root = sets.find(q * 4 + c);
            *slot = root_to_corner.get(&root).copied();
        }
        for c in 0..4 {
            if let (Some(a), Some(b)) = (ids[c], ids[(c + 1) & 3]) {
                if a != b {
                    adjacency[a].insert(b);
                    adjacency[b].insert(a);
                }
            }
        }
    }

    let mut seen = vec![false; points.len()];
    let mut best = None;
    for start in 0..points.len() {
        if seen[start] || adjacency[start].is_empty() {
            continue;
        }
        let mut component = Vec::new();
        let mut queue = VecDeque::from([start]);
        seen[start] = true;
        while let Some(v) = queue.pop_front() {
            component.push(v);
            for &next in &adjacency[v] {
                if !seen[next] {
                    seen[next] = true;
                    queue.push_back(next);
                }
            }
        }
        if let Some(order) = order_component(&adjacency, &component, &points, width, height) {
            best = Some(order.into_iter().map(|i| points[i]).collect());
            break;
        }
    }
    best
}

fn detect_thresholded(
    gray: &GrayImage,
    labels: Vec<i8>,
    width: usize,
    height: usize,
) -> Option<Vec<Pt>> {
    let w = gray.width() as usize;
    let h = gray.height() as usize;
    let quads = extract_quads(erode_labels(&labels, w, h), w, h);
    let mut corners = grid_from_quads(&quads, width, height)?;
    if corners
        .iter()
        .any(|p| p.0 <= 8.0 || p.1 <= 8.0 || p.0 > w as f32 - 8.0 || p.1 > h as f32 - 8.0)
    {
        return None;
    }
    for corner in &mut corners {
        *corner = corner_sub_pix(gray, *corner, 2, 15, 0.1);
    }
    Some(corners)
}

/// Find the requested `width x height` grid of interior chessboard corners.
/// Returns row-major, sub-pixel refined image coordinates on success.
pub fn find_chessboard_corners(gray: &GrayImage, width: usize, height: usize) -> Option<Vec<Pt>> {
    if width < 3 || height < 3 || gray.width() < 16 || gray.height() < 16 {
        return None;
    }
    if let Some(corners) = detect_thresholded(gray, global_threshold_labels(gray), width, height) {
        return Some(corners);
    }

    let min_dim = gray.width().min(gray.height());
    let mut windows = vec![
        ((min_dim as f32 * 0.10).round() as u32) | 1,
        ((min_dim as f32 * 0.20).round() as u32) | 1,
        31,
    ];
    windows.iter_mut().for_each(|w| *w = (*w).max(3));
    windows.sort_unstable();
    windows.dedup();
    for window in windows {
        for constant in [0.0, 5.0, 10.0] {
            let labels = imgproc::adaptive_threshold_labels(gray, window, constant);
            if let Some(corners) = detect_thresholded(gray, labels, width, height) {
                return Some(corners);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orders_a_rectangular_grid_graph() {
        let (width, height) = (4, 3);
        let points: Vec<Pt> = (0..height)
            .flat_map(|y| (0..width).map(move |x| (20.0 + x as f32 * 10.0, 30.0 + y as f32 * 12.0)))
            .collect();
        let mut adjacency = vec![HashSet::new(); points.len()];
        for y in 0..height {
            for x in 0..width {
                let v = y * width + x;
                if x + 1 < width {
                    adjacency[v].insert(v + 1);
                    adjacency[v + 1].insert(v);
                }
                if y + 1 < height {
                    adjacency[v].insert(v + width);
                    adjacency[v + width].insert(v);
                }
            }
        }
        let component: Vec<usize> = (0..points.len()).collect();
        let order = order_component(&adjacency, &component, &points, width, height).unwrap();
        assert_eq!(order, component);
    }
}
