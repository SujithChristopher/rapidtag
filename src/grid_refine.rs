//! Grid refinement of tag corners: one homography fitted to every black/white
//! transition of the tag's known bit pattern, the interior cell boundaries as
//! well as the four outer edges.
//!
//! Port of NOARK's `grid_refine.py`. Edge refinement places the outer edges from
//! ~128 crossings; the pattern holds two to three times more straight
//! transitions, all on the grid lines u = k, v = k of the tag, and a homography
//! ties them together, so every crossing informs every corner.
//!
//! Per tag, starting from refined corners:
//!   1. runs: maximal straight pieces of each grid line along which the colours
//!      on both sides stay the same; PER_CELL profiles per cell of run, END_MARGIN
//!      cells kept free at each run end;
//!   2. NO bilinear samples per profile along the image normal, black to white,
//!      half a cell across for interior runs and clip(0.6 cell, 2.5, 4) px for
//!      outer runs; interior runs only where the cell is >= MIN_CELL_PX;
//!   3. black / white per run from the profile ends, halfway crossing nearest the
//!      middle;
//!   4. Gauss-Newton on the image -> tag homography (8) and optionally the print
//!      bleed delta (1), residual = image distance from each crossing to its
//!      line, Huber weights; profiles re-sampled once around the first fit.
//! A tag keeps its input corners when too few crossings survive, the residual is
//! large, delta is implausible, or a corner would move too far.

use crate::edge_refine::GrayView;
use crate::imgproc::{self, Pt};
use nalgebra::{Matrix3, Vector3};

pub type QuadF64 = [(f64, f64); 4];

const PER_CELL: f64 = 5.0; // profiles per cell of run length
const NO: usize = 13; // samples per profile
const END_MARGIN: f64 = 0.35; // cells kept free at each run end
const MIN_CELL_PX: f64 = 4.0; // interior runs need cells at least this wide across them
const OUTER_R: (f64, f64) = (2.5, 4.0); // px; outer-run profile half-width range
const MIN_CONTRAST: f64 = 20.0;
const MIN_POINTS: usize = 40;
const ROUNDS: usize = 2; // profile sampling rounds
const ITERS: usize = 6; // Gauss-Newton steps per round
const HUBER_PX: f64 = 0.25;
const MAX_RMS_PX: f64 = 0.6;
const MAX_DELTA: f64 = 0.25; // cells
const MAX_SHIFT_PX: f64 = 1.5;
const MAX_SHIFT_CELLS: f64 = 0.3;
const MIN_PATTERN_STD: f64 = 1e-6;

/// A straight transition run: on line u = k (axis 0) or v = k (axis 1), white
/// side towards +u / +v when `s` = +1, extent t0..t1 cells along the line.
#[derive(Clone, Copy)]
struct Run {
    axis: u8,
    k: f64,
    s: f64,
    t0: f64,
    t1: f64,
    outer: bool,
}

/// A tag's colours and transition runs. `cells` = marker size + 2 (black border).
pub struct Pattern {
    cells: usize,
    /// (cells + 2)^2 colours, 1 = white, including the white quiet zone.
    grid: Vec<u8>,
    runs: Vec<Run>,
}

impl Pattern {
    /// From a marker's bits (row-major, 1 = white), one-cell black border.
    pub fn new(bits: &[u8], marker_size: usize) -> Pattern {
        let cells = marker_size + 2;
        assert!(cells < MAX_LINES, "marker too large for grid refinement");
        let g = cells + 2;
        let mut grid = vec![1u8; g * g];
        for r in 1..g - 1 {
            for c in 1..g - 1 {
                let inner = r >= 2 && r < g - 2 && c >= 2 && c < g - 2;
                grid[r * g + c] = if inner { bits[(r - 2) * marker_size + (c - 2)] } else { 0 };
            }
        }
        let at = |r: usize, c: usize| grid[r * g + c];
        let mut runs = Vec::new();
        for axis in 0..2u8 {
            for k in 0..=cells {
                let mut cur: Option<Run> = None;
                for t in 0..cells {
                    // cells on either side of the line: left / right of u = k, above / below v = k
                    let (a, b) = if axis == 0 { (at(t + 1, k), at(t + 1, k + 1)) } else { (at(k, t + 1), at(k + 1, t + 1)) };
                    let s = if a == b { 0.0 } else if b == 1 { 1.0 } else { -1.0 };
                    if let Some(c) = cur.as_mut() {
                        if c.s == s {
                            c.t1 = (t + 1) as f64;
                            continue;
                        }
                        if c.s != 0.0 {
                            runs.push(*c);
                        }
                    }
                    cur = Some(Run { axis, k: k as f64, s, t0: t as f64, t1: (t + 1) as f64, outer: k == 0 || k == cells });
                }
                if let Some(c) = cur {
                    if c.s != 0.0 {
                        runs.push(c);
                    }
                }
            }
        }
        Pattern { cells, grid, runs }
    }

    /// Colours of the tag's own cells (border + data), row-major, 1 = white.
    fn tag_cells(&self) -> impl Iterator<Item = f64> + '_ {
        let g = self.cells + 2;
        (1..g - 1).flat_map(move |r| (1..g - 1).map(move |c| self.grid[r * g + c] as f64))
    }

    fn square(&self) -> [Pt; 4] {
        let n = self.cells as f32;
        [(0.0, 0.0), (n, 0.0), (n, n), (0.0, n)]
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GridInfo {
    pub ok: bool,
    pub points: usize,
    pub interior: usize,
    pub delta: f64,
    pub rms_px: f64,
}

fn mat3(h: &[f64; 9]) -> Matrix3<f64> {
    Matrix3::from_row_slice(h)
}

fn apply(h: &Matrix3<f64>, x: f64, y: f64) -> (f64, f64) {
    let v = h * Vector3::new(x, y, 1.0);
    (v[0] / v[2], v[1] / v[2])
}

/// One profile across a run: base point, unit normal (black -> white), half-width.
struct Profile {
    base: (f64, f64),
    nrm: (f64, f64),
    r: f64,
    run: usize,
}

/// All profiles of the tag under homography `h` (tag -> image) whose samples lie
/// inside the image.
fn profiles(img: &GrayView, h: &Matrix3<f64>, delta: f64, pat: &Pattern, outer_only: bool, out: &mut Vec<Profile>) {
    out.clear();
    let (w1, h1) = ((img.w - 1) as f64, (img.h - 1) as f64);
    for (i, run) in pat.runs.iter().enumerate() {
        if outer_only && !run.outer {
            continue;
        }
        let (a, b) = (run.t0 + END_MARGIN, run.t1 - END_MARGIN);
        let n = ((PER_CELL * (b - a)).round_ties_even() as usize).max(2);
        let step = (b - a) / (n - 1) as f64;
        let c = run.k + run.s * delta;
        for j in 0..n {
            let t = if j == n - 1 { b } else { a + j as f64 * step };
            let (u, v) = if run.axis == 0 { (c, t) } else { (t, c) };
            // image point and Jacobian d(image)/d(tag)
            let x = h * Vector3::new(u, v, 1.0);
            let wz = x[2];
            let (ix, iy) = (x[0] / wz, x[1] / wz);
            let jac = |r: usize, col: usize, img_r: f64| (h[(r, col)] - img_r * h[(2, col)]) / wz;
            let du = (jac(0, 0, ix), jac(1, 0, iy));
            let dv = (jac(0, 1, ix), jac(1, 1, iy));
            let (across, along) = if run.axis == 0 { (du, dv) } else { (dv, du) };
            let len = along.0.hypot(along.1);
            let mut nrm = (-along.1 / len, along.0 / len);
            let sgn = ((nrm.0 * across.0 + nrm.1 * across.1) * run.s).signum();
            nrm = (nrm.0 * sgn, nrm.1 * sgn);
            let cell = (nrm.0 * across.0 + nrm.1 * across.1).abs();
            if !run.outer && !(cell >= MIN_CELL_PX) {
                continue;
            }
            let r = if run.outer { (0.6 * cell).clamp(OUTER_R.0, OUTER_R.1) } else { 0.5 * cell };
            let inside = [-r, r].iter().all(|&o| {
                let (px, py) = (ix + o * nrm.0, iy + o * nrm.1);
                px >= 0.0 && py >= 0.0 && px <= w1 && py <= h1
            });
            if inside {
                out.push(Profile { base: (ix, iy), nrm, r, run: i });
            }
        }
    }
}

/// Halfway crossing per profile (px along the normal), black / white per run from
/// the profile ends. Returns crossings and validity.
fn crossings(img: &GrayView, profs: &[Profile], nrun: usize, x: &mut Vec<f64>, ok: &mut Vec<bool>) {
    let lin: [f64; NO] = std::array::from_fn(|i| -1.0 + 2.0 * i as f64 / (NO - 1) as f64);
    let mut prof = vec![0.0f64; profs.len() * NO];
    let (mut black, mut white, mut cnt) = (vec![0.0; nrun], vec![0.0; nrun], vec![0usize; nrun]);
    for (p, row) in profs.iter().zip(prof.chunks_exact_mut(NO)) {
        for (j, v) in row.iter_mut().enumerate() {
            let o = p.r * lin[j];
            *v = img.bilinear_inside((p.base.0 + o * p.nrm.0) as f32, (p.base.1 + o * p.nrm.1) as f32) as f64;
        }
        black[p.run] += (row[0] + row[1]) / 2.0;
        white[p.run] += (row[NO - 2] + row[NO - 1]) / 2.0;
        cnt[p.run] += 1;
    }
    for i in 0..nrun {
        let c = cnt[i].max(1) as f64;
        black[i] /= c;
        white[i] /= c;
    }
    let centre = (NO - 2) as f64 / 2.0;
    x.clear();
    ok.clear();
    for (p, row) in profs.iter().zip(prof.chunks_exact(NO)) {
        let mid = (black[p.run] + white[p.run]) / 2.0;
        let mut best: Option<usize> = None;
        let mut best_rank = f64::MAX;
        for j in 0..NO - 1 {
            if row[j] - mid < 0.0 && row[j + 1] - mid >= 0.0 {
                let rank = (j as f64 - centre).abs();
                if rank < best_rank {
                    best_rank = rank;
                    best = Some(j);
                }
            }
        }
        let step = p.r * (lin[1] - lin[0]);
        match best {
            Some(j) if white[p.run] - black[p.run] >= MIN_CONTRAST => {
                let (s0, s1) = (row[j] - mid, row[j + 1] - mid);
                let v = p.r * lin[j] + s0 / (s0 - s1) * step;
                x.push(v);
                ok.push(v.is_finite());
            }
            _ => {
                x.push(0.0);
                ok.push(false);
            }
        }
    }
}

/// Upper-triangle (u <= v) index pairs of a 6 x 6 symmetric block.
const PAIRS: [(usize, usize); 21] = {
    let mut out = [(0, 0); 21];
    let (mut u, mut t) = (0, 0);
    while u < 6 {
        let mut v = u;
        while v < 6 {
            out[t] = (u, v);
            t += 1;
            v += 1;
        }
        u += 1;
    }
    out
};

/// acc[t] += wv[u] * val[v] over PAIRS, unrolled so the accumulators stay in registers.
#[inline(always)]
fn accumulate(acc: &mut [f64; 21], wv: &[f64; 6], val: &[f64; 6]) {
    macro_rules! acc {
        ($($t:literal: $u:literal $v:literal),*) => { $( acc[$t] += wv[$u] * val[$v]; )* };
    }
    acc!(0: 0 0, 1: 0 1, 2: 0 2, 3: 0 3, 4: 0 4, 5: 0 5, 6: 1 1, 7: 1 2, 8: 1 3, 9: 1 4, 10: 1 5,
         11: 2 2, 12: 2 3, 13: 2 4, 14: 2 5, 15: 3 3, 16: 3 4, 17: 3 5, 18: 4 4, 19: 4 5, 20: 5 5);
}

/// Solve the symmetric positive definite m x m system `a x = b` (Cholesky, in place).
fn cholesky_solve(a: &mut [[f64; 9]; 9], b: &mut [f64; 9], m: usize) -> bool {
    for j in 0..m {
        let mut d = a[j][j];
        for k in 0..j {
            d -= a[j][k] * a[j][k];
        }
        if !(d > 0.0) {
            return false;
        }
        let d = d.sqrt();
        a[j][j] = d;
        for i in j + 1..m {
            let mut v = a[i][j];
            for k in 0..j {
                v -= a[i][k] * a[j][k];
            }
            a[i][j] = v / d;
        }
    }
    for i in 0..m {
        let mut v = b[i];
        for k in 0..i {
            v -= a[i][k] * b[k];
        }
        b[i] = v / a[i][i];
    }
    for i in (0..m).rev() {
        let mut v = b[i];
        for k in i + 1..m {
            v -= a[k][i] * b[k];
        }
        b[i] = v / a[i][i];
    }
    true
}

/// A crossing in normalised image coordinates, on line `k` of its family with the
/// white side towards + (`pos`) or - .
#[derive(Clone, Copy)]
struct Crossing {
    p: (f64, f64),
    k: u8,
    pos: bool,
}

const MAX_LINES: usize = 16; // grid lines per family (cells + 1); 7x7 dictionaries need 10

/// Per line (k, white side) of family `r` (= 3 axis): c = k + s delta and the line's
/// image-space normaliser. With L = (1, 0, -c) on lines u = c and (0, 1, -c) on v = c,
/// a point's distance is d = (L . G p) / |b|, b = (G^T L)_xy, and b depends only on
/// the line. Entries: (c, b0, b1, 1 / |b|), index 2 k + pos, for the first `n` lines.
fn line_table(g: &[f64; 9], r: usize, delta: f64, n: usize) -> [[f64; 4]; 2 * MAX_LINES] {
    std::array::from_fn(|li| {
        if li >= 2 * n {
            return [0.0; 4];
        }
        let c = (li / 2) as f64 + if li % 2 == 1 { delta } else { -delta };
        let (b0, b1) = (g[r] - c * g[6], g[r + 1] - c * g[7]);
        [c, b0, b1, 1.0 / (b0 * b0 + b1 * b1).sqrt()]
    })
}

/// Gauss-Newton on G (image -> tag, normalised image coordinates) and delta.
/// `groups[a]` holds the crossings on line family a (u = k for 0, v = k for 1),
/// k < `nlines`.
/// Returns the final residuals (normalised units), group 0 first.
fn fit(groups: &[Vec<Crossing>; 2], nlines: usize, gm: &mut Matrix3<f64>, delta: &mut f64, fit_delta: bool, huber: f64) -> Vec<f64> {
    let m = if fit_delta { 9 } else { 8 };
    // G row-major; G[2][2] stays fixed, so the unknowns are g[0..8] (+ delta). Only G's
    // row `axis` and row 2 enter a point's distance, so each Jacobian row has 5
    // non-zeros in g (+ delta): unknowns (g[3a], g[3a+1], g[3a+2], g[6], g[7], delta),
    // accumulated per family in a symmetric 6 x 6 block.
    let mut g: [f64; 9] = std::array::from_fn(|i| gm[(i / 3, i % 3)]);
    for _ in 0..ITERS {
        let mut a = [[0.0f64; 9]; 9];
        let mut rhs = [0.0f64; 9];
        for (ax, pts) in groups.iter().enumerate() {
            let r = 3 * ax;
            let tab = line_table(&g, r, *delta, nlines);
            let mut acc = [0.0f64; 21];
            let mut rb = [0.0f64; 6];
            for x in pts {
                let [c, b0, b1, inv] = tab[2 * x.k as usize + x.pos as usize];
                let p = x.p;
                let q2 = g[6] * p.0 + g[7] * p.1 + g[8];
                let num = g[r] * p.0 + g[r + 1] * p.1 + g[r + 2] - c * q2;
                let d = num * inv;
                let e = d * inv * inv;
                // dd/dG[i][j] = L_i (P_j / nn - e bpad_j), bpad = (b0, b1, 0)
                let (jx, jy) = (p.0 * inv - e * b0, p.1 * inv - e * b1);
                let jd = if fit_delta {
                    let s = if x.pos { 1.0 } else { -1.0 };
                    let dnn = (b0 * -s * g[6] + b1 * -s * g[7]) * inv;
                    -s * q2 * inv - num * dnn * inv * inv
                } else {
                    0.0
                };
                let val = [jx, jy, inv, -c * jx, -c * jy, jd];
                let ad = d.abs();
                let w = if ad <= huber { 1.0 } else { huber / ad.max(1e-12) };
                let wv: [f64; 6] = std::array::from_fn(|u| w * val[u]);
                for u in 0..6 {
                    rb[u] -= wv[u] * d;
                }
                accumulate(&mut acc, &wv, &val);
            }
            let vi = [r, r + 1, r + 2, 6, 7, 8];
            for u in 0..6 {
                rhs[vi[u]] += rb[u];
            }
            for (t, &(u, v)) in PAIRS.iter().enumerate() {
                // lower triangle, as cholesky_solve reads it (vi is increasing)
                a[vi[v]][vi[u]] += acc[t];
            }
        }
        let tr: f64 = (0..m).map(|i| a[i][i]).sum();
        for (i, row) in a.iter_mut().enumerate().take(m) {
            row[i] += 1e-9 * tr;
        }
        if !cholesky_solve(&mut a, &mut rhs, m) {
            break;
        }
        let step = rhs; // step[8] stays 0 when delta is fixed (m = 8)
        for (gi, si) in g.iter_mut().zip(step.iter()).take(8) {
            *gi += si;
        }
        if fit_delta {
            *delta += step[8];
        }
        if step.iter().take(m).fold(0.0f64, |acc, v| acc.max(v.abs())) < 1e-10 {
            break;
        }
    }
    *gm = Matrix3::from_row_slice(&g);
    let mut res = Vec::with_capacity(groups[0].len() + groups[1].len());
    for (ax, pts) in groups.iter().enumerate() {
        let r = 3 * ax;
        let tab = line_table(&g, r, *delta, nlines);
        for x in pts {
            let [c, _, _, inv] = tab[2 * x.k as usize + x.pos as usize];
            let p = x.p;
            res.push((g[r] * p.0 + g[r + 1] * p.1 + g[r + 2] - c * (g[6] * p.0 + g[7] * p.1 + g[8])) * inv);
        }
    }
    res
}

/// Grid-refined corners of one tag from refined `corners`, and fit info. On
/// failure the input corners are returned with `ok = false`.
pub fn refine_tag(img: &GrayView, corners: &QuadF64, pat: &Pattern, fit_delta: bool, outer_only: bool) -> (QuadF64, GridInfo) {
    let q = *corners;
    let mut info = GridInfo { delta: f64::NAN, rms_px: f64::NAN, ..Default::default() };
    let side: f64 = (0..4).map(|i| (q[(i + 1) % 4].0 - q[i].0).hypot(q[(i + 1) % 4].1 - q[i].1)).sum();
    let cell = side / 4.0 / pat.cells as f64;
    let centre = ((q[0].0 + q[1].0 + q[2].0 + q[3].0) / 4.0, (q[0].1 + q[1].1 + q[2].1 + q[3].1) / 4.0);
    let t = Matrix3::new(1.0 / cell, 0.0, -centre.0 / cell, 0.0, 1.0 / cell, -centre.1 / cell, 0.0, 0.0, 1.0);
    let t_inv = Matrix3::new(cell, 0.0, centre.0, 0.0, cell, centre.1, 0.0, 0.0, 1.0);
    let qf: [Pt; 4] = std::array::from_fn(|i| (q[i].0 as f32, q[i].1 as f32));
    let square = pat.square();
    let h = mat3(&imgproc::get_perspective_transform(&square, &qf));
    let Some(mut g) = (t * h).try_inverse() else { return (q, info) };
    g /= g[(2, 2)];
    let mut delta = 0.0;
    let mut profs = Vec::new();
    let (mut x, mut ok) = (Vec::new(), Vec::new());
    let mut res = Vec::new();
    let mut groups: [Vec<Crossing>; 2] = [Vec::new(), Vec::new()];
    for _ in 0..ROUNDS {
        let Some(g_inv) = g.try_inverse() else { return (q, info) };
        profiles(img, &(t_inv * g_inv), delta, pat, outer_only, &mut profs);
        if profs.len() < MIN_POINTS {
            return (q, info);
        }
        crossings(img, &profs, pat.runs.len(), &mut x, &mut ok);
        groups.iter_mut().for_each(Vec::clear);
        for (i, p) in profs.iter().enumerate() {
            if ok[i] {
                let (px, py) = (p.base.0 + x[i] * p.nrm.0, p.base.1 + x[i] * p.nrm.1);
                let run = &pat.runs[p.run];
                let p = ((px - centre.0) / cell, (py - centre.1) / cell);
                groups[run.axis as usize].push(Crossing { p, k: run.k as u8, pos: run.s > 0.0 });
            }
        }
        if groups[0].len() + groups[1].len() < MIN_POINTS {
            return (q, info);
        }
        res = fit(&groups, pat.cells + 1, &mut g, &mut delta, fit_delta, HUBER_PX / cell);
        g /= g[(2, 2)];
    }
    let Some(g_inv) = g.try_inverse() else { return (q, info) };
    let hh = t_inv * g_inv;
    let new: QuadF64 = std::array::from_fn(|i| apply(&hh, square[i].0 as f64, square[i].1 as f64));
    let (mut sw, mut swr) = (0.0, 0.0);
    for r in &res {
        let rp = r * cell;
        let w = (HUBER_PX / rp.abs().max(1e-12)).min(1.0);
        sw += w;
        swr += w * rp * rp;
    }
    let rms = (swr / sw).sqrt();
    let shift = (0..4).map(|i| (new[i].0 - q[i].0).hypot(new[i].1 - q[i].1)).fold(0.0f64, f64::max);
    info.points = res.len();
    info.interior = profs.iter().zip(&ok).filter(|(p, &o)| o && !pat.runs[p.run].outer).count();
    info.delta = delta;
    info.rms_px = rms;
    let finite = new.iter().all(|p| p.0.is_finite() && p.1.is_finite());
    if !(finite && rms <= MAX_RMS_PX && delta.abs() <= MAX_DELTA && shift <= MAX_SHIFT_PX.max(MAX_SHIFT_CELLS * cell)) {
        return (q, info);
    }
    info.ok = true;
    (new, info)
}

/// Correlation between the grey levels at the tag's cell centres (through the
/// corners' homography) and its pattern: ~1 for the right tag in the right
/// orientation, ~0 otherwise, -1 when a cell centre falls outside the image.
pub fn pattern_score(img: &GrayView, corners: &QuadF64, pat: &Pattern) -> f64 {
    let qf: [Pt; 4] = std::array::from_fn(|i| (corners[i].0 as f32, corners[i].1 as f32));
    let h = mat3(&imgproc::get_perspective_transform(&pat.square(), &qf));
    let n = pat.cells;
    let mut v = Vec::with_capacity(n * n);
    for r in 0..n {
        for c in 0..n {
            let (x, y) = apply(&h, c as f64 + 0.5, r as f64 + 0.5);
            let (x, y) = (x as f32, y as f32);
            if !img.contains(x, y) {
                return -1.0;
            }
            v.push(img.bilinear(x, y) as f64);
        }
    }
    let p: Vec<f64> = pat.tag_cells().collect();
    let k = v.len() as f64;
    let (mv, mp) = (v.iter().sum::<f64>() / k, p.iter().sum::<f64>() / k);
    let (mut svv, mut spp, mut svp) = (0.0, 0.0, 0.0);
    for (a, b) in v.iter().zip(&p) {
        svv += (a - mv) * (a - mv);
        spp += (b - mp) * (b - mp);
        svp += (a - mv) * (b - mp);
    }
    if (svv / k).sqrt() < MIN_PATTERN_STD {
        return 0.0;
    }
    svp / (svv * spp).sqrt()
}
