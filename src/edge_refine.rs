//! Edge refinement of tag corners: each outer edge of a detected tag is placed
//! where the grey level crosses halfway between black and white, and the corners
//! are the intersections of those edges.
//!
//! Port of the NOARK tracker's `edge_refine.py`. Contour refinement fits lines
//! through the outermost dark pixels of the thresholded image, which sit ~0.5 px
//! inside the true edge plus the threshold's own offset, so every side comes out
//! ~1.3 px short. Here the grey levels decide.
//!
//! Per tag, a fine pass: NS2 profiles of NO bilinear samples across each edge,
//! +-r px around PRESHIFT px outside the detector's edge; black / white levels
//! from the profile ends; per profile the halfway crossing nearest the middle;
//! a total-least-squares line through the crossings. Only for tags whose edges
//! the window did not hold, a coarse pass first (NS1 profiles across
//! +-COARSE_CELLS cells) finds where each edge is. A tag keeps its input corners
//! when an edge leaves the image, has too few crossings or too little contrast,
//! or a corner would move more than max(MAX_SHIFT_PX, 0.7 cell).

use crate::imgproc::Pt;

pub type Quad = [Pt; 4];

const NS1: usize = 8; // profiles per edge, coarse pass
const NS2: usize = 32; // profiles per edge, fine pass
const NO: usize = 17; // samples per profile
const R_FINE: f32 = 4.0; // px; fine profile half-width
const R_MIN: f32 = 2.5; // px; smallest half-width (far tags, cell ~5 px)
const COARSE_CELLS: f32 = 0.6; // coarse half-width, cells
const PRESHIFT: f32 = 0.6; // px outward: where the true edge usually is
const END_MARGIN: f32 = 0.12; // of the edge length kept free at each end (>= 3 px)
const ENDS: usize = 3; // samples at each profile end that give black / white
const MIN_CONTRAST: f32 = 20.0;
const MIN_POINTS: usize = 8; // valid crossings per edge in the fine pass
const MAX_SHIFT_PX: f32 = 3.0;

/// Borrowed 8-bit grey image with bilinear sampling.
#[derive(Clone, Copy)]
pub struct GrayView<'a> {
    pub data: &'a [u8],
    pub w: usize,
    pub h: usize,
}

impl GrayView<'_> {
    /// Bilinear grey level at (x, y), border pixels replicated.
    #[inline]
    pub fn bilinear(&self, x: f32, y: f32) -> f32 {
        let (xf, yf) = (x.floor(), y.floor());
        let (fx, fy) = (x - xf, y - yf);
        let (wm, hm) = (self.w as i32 - 1, self.h as i32 - 1);
        let x0 = (xf as i32).clamp(0, wm) as usize;
        let x1 = (xf as i32 + 1).clamp(0, wm) as usize;
        let y0 = (yf as i32).clamp(0, hm) as usize;
        let y1 = (yf as i32 + 1).clamp(0, hm) as usize;
        let r0 = &self.data[y0 * self.w..];
        let r1 = &self.data[y1 * self.w..];
        let top = r0[x0] as f32 + fx * (r0[x1] as f32 - r0[x0] as f32);
        let bot = r1[x0] as f32 + fx * (r1[x1] as f32 - r1[x0] as f32);
        top + fy * (bot - top)
    }

    /// Bilinear grey level at (x, y) already known to lie in [0, w-1] x [0, h-1].
    #[inline]
    pub fn bilinear_inside(&self, x: f32, y: f32) -> f32 {
        let (x0, y0) = ((x.max(0.0) as usize).min(self.w - 1), (y.max(0.0) as usize).min(self.h - 1));
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let i00 = y0 * self.w + x0;
        let dx = usize::from(x0 + 1 < self.w);
        let dy = if y0 + 1 < self.h { self.w } else { 0 };
        let px = |i: usize| self.data[i] as f32;
        let (p00, p01, p10, p11) = (px(i00), px(i00 + dx), px(i00 + dy), px(i00 + dy + dx));
        let top = p00 + fx * (p01 - p00);
        let bot = p10 + fx * (p11 - p10);
        top + fy * (bot - top)
    }

    #[inline]
    pub fn contains(&self, x: f32, y: f32) -> bool {
        x >= 0.0 && y >= 0.0 && x <= (self.w - 1) as f32 && y <= (self.h - 1) as f32
    }
}

fn unit_profile() -> [f32; NO] {
    std::array::from_fn(|i| -1.0 + 2.0 * i as f32 / (NO - 1) as f32)
}

/// Per edge k (corner k -> corner k+1): start point, unit direction, outward unit
/// normal, length; plus the tag's cell size (mean side / 8).
struct Geometry {
    a: [Pt; 4],
    d: [Pt; 4],
    n: [Pt; 4],
    length: [f32; 4],
    cell: f32,
}

fn geometry(q: &Quad) -> Geometry {
    let cx = (q[0].0 + q[1].0 + q[2].0 + q[3].0) / 4.0;
    let cy = (q[0].1 + q[1].1 + q[2].1 + q[3].1) / 4.0;
    let mut g = Geometry { a: *q, d: [(0.0, 0.0); 4], n: [(0.0, 0.0); 4], length: [0.0; 4], cell: 0.0 };
    for k in 0..4 {
        let (a, b) = (q[k], q[(k + 1) % 4]);
        let len = (b.0 - a.0).hypot(b.1 - a.1);
        let d = ((b.0 - a.0) / len, (b.1 - a.1) / len);
        let mut n = (d.1, -d.0);
        let mid_out = ((a.0 + b.0) / 2.0 - cx, (a.1 + b.1) / 2.0 - cy);
        let sgn = (n.0 * mid_out.0 + n.1 * mid_out.1).signum();
        n = (n.0 * sgn, n.1 * sgn);
        g.d[k] = d;
        g.n[k] = n;
        g.length[k] = len;
    }
    g.cell = g.length.iter().sum::<f32>() / 4.0 / 8.0;
    g
}

/// Base points of `s` profiles along edge k, END_MARGIN of the length free at each end.
fn along(g: &Geometry, k: usize, s: usize, out: &mut [Pt]) {
    let margin = (END_MARGIN * g.length[k]).max(3.0);
    let span = g.length[k] - 2.0 * margin;
    for (i, p) in out.iter_mut().take(s).enumerate() {
        let t = margin + (i as f32 / (s - 1) as f32) * span;
        *p = (g.a[k].0 + t * g.d[k].0, g.a[k].1 + t * g.d[k].1);
    }
}

/// Halfway crossings of one edge's profiles. `prof` holds `s` profiles of NO
/// samples at offsets `offs` (px along the normal). Black / white are edge-level
/// means of the ENDS samples at each end. Returns (black, white); `x[i]` is the
/// crossing of profile i and `ok[i]` whether it exists.
fn crossings(prof: &[f32], s: usize, offs: &[f32; NO], x: &mut [f32], ok: &mut [bool]) -> (f32, f32) {
    let (mut black, mut white) = (0.0f32, 0.0f32);
    for p in prof.chunks_exact(NO).take(s) {
        for e in 0..ENDS {
            black += p[e];
            white += p[NO - 1 - e];
        }
    }
    black /= (s * ENDS) as f32;
    white /= (s * ENDS) as f32;
    let mid = (black + white) / 2.0;
    let step = offs[1] - offs[0];
    let centre = (NO - 2) as f32 / 2.0;
    for (i, p) in prof.chunks_exact(NO).take(s).enumerate() {
        // the black -> white crossing nearest the middle of the profile
        let mut best: Option<usize> = None;
        let mut best_rank = f32::MAX;
        for j in 0..NO - 1 {
            if p[j] - mid < 0.0 && p[j + 1] - mid >= 0.0 {
                let rank = (j as f32 - centre).abs();
                if rank < best_rank {
                    best_rank = rank;
                    best = Some(j);
                }
            }
        }
        ok[i] = false;
        if let Some(j) = best {
            let (s0, s1) = (p[j] - mid, p[j + 1] - mid);
            let v = offs[0] + (j as f32 + s0 / (s0 - s1)) * step;
            if v.is_finite() {
                x[i] = v;
                ok[i] = true;
            }
        }
    }
    (black, white)
}

/// Where each edge really is, px out from the input edge: lower median crossing of
/// NS1 profiles across +-COARSE_CELLS cells (PRESHIFT when none is found).
fn coarse(img: &GrayView, q: &Quad) -> [f32; 4] {
    let g = geometry(q);
    let u = unit_profile();
    let r1 = (COARSE_CELLS * g.cell).max(R_MIN);
    let offs: [f32; NO] = std::array::from_fn(|i| PRESHIFT + r1 * u[i]);
    let mut out = [PRESHIFT; 4];
    let mut base = [(0.0, 0.0); NS1];
    let mut prof = [0.0f32; NS1 * NO];
    let (mut x, mut ok) = ([0.0f32; NS1], [false; NS1]);
    for k in 0..4 {
        along(&g, k, NS1, &mut base);
        let n = g.n[k];
        for (i, b) in base.iter().enumerate() {
            for (j, o) in offs.iter().enumerate() {
                prof[i * NO + j] = img.bilinear(b.0 + o * n.0, b.1 + o * n.1);
            }
        }
        crossings(&prof, NS1, &offs, &mut x, &mut ok);
        let mut v: Vec<f32> = (0..NS1).filter(|&i| ok[i]).map(|i| x[i]).collect();
        if !v.is_empty() {
            v.sort_by(|a, b| a.total_cmp(b));
            out[k] = v[(v.len() - 1) / 2];
        }
    }
    out
}

/// Fine pass around `centre` (px out from each input edge): refined corners,
/// whether the tag is usable, and whether all four edges were found.
fn fine(img: &GrayView, q: &Quad, centre: &[f32; 4]) -> (Quad, bool, bool) {
    let g = geometry(q);
    let u = unit_profile();
    let r2 = (0.6 * g.cell).clamp(R_MIN, R_FINE);
    let mut base = [(0.0, 0.0); NS2];
    let mut prof = [0.0f32; NS2 * NO];
    let (mut x, mut ok) = ([0.0f32; NS2], [false; NS2]);
    let mut found = true;
    let mut in_image = true;
    let mut lines = [(0.0f64, 0.0f64, 0.0f64); 4]; // (nx, ny, rho): nx x + ny y = rho
    for k in 0..4 {
        let offs: [f32; NO] = std::array::from_fn(|i| centre[k] + r2 * u[i]);
        along(&g, k, NS2, &mut base);
        let n = g.n[k];
        for (i, b) in base.iter().enumerate() {
            for (j, o) in offs.iter().enumerate() {
                let (px, py) = (b.0 + o * n.0, b.1 + o * n.1);
                in_image &= img.contains(px, py);
                prof[i * NO + j] = img.bilinear(px, py);
            }
        }
        let (black, white) = crossings(&prof, NS2, &offs, &mut x, &mut ok);

        // total-least-squares line through the valid crossings (closed form, 2 x 2)
        let (mut cnt, mut mx, mut my, mut xm) = (0usize, 0.0f64, 0.0f64, 0.0f64);
        let mut pts = [(0.0f64, 0.0f64); NS2];
        for i in 0..NS2 {
            if ok[i] {
                let p = ((base[i].0 + x[i] * n.0) as f64, (base[i].1 + x[i] * n.1) as f64);
                pts[cnt] = p;
                mx += p.0;
                my += p.1;
                xm += x[i] as f64;
                cnt += 1;
            }
        }
        let c = cnt.max(1) as f64;
        let (mx, my, xm) = (mx / c, my / c, xm / c);
        let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
        for p in &pts[..cnt] {
            let (dx, dy) = (p.0 - mx, p.1 - my);
            sxx += dx * dx;
            syy += dy * dy;
            sxy += dx * dy;
        }
        let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
        let (nx, ny) = (-theta.sin(), theta.cos());
        lines[k] = (nx, ny, nx * mx + ny * my);

        // found = the window held the whole edge: enough crossings, contrast between
        // the ends, and the crossings near its middle (else they are biased)
        let centred = (xm - centre[k] as f64).abs() <= (r2 as f64 - 2.5).max(1.0);
        found &= cnt >= MIN_POINTS && white - black >= MIN_CONTRAST && centred;
    }

    // corner k joins edge k-1 and edge k
    let mut new = *q;
    let mut usable = found && in_image;
    let limit = MAX_SHIFT_PX.max(0.7 * g.cell) as f64;
    for k in 0..4 {
        let (n1x, n1y, r1) = lines[(k + 3) % 4];
        let (nx, ny, rho) = lines[k];
        let det = n1x * ny - n1y * nx;
        if det.abs() < 1e-6 {
            usable = false;
            continue;
        }
        let p = ((r1 * ny - rho * n1y) / det, (n1x * rho - nx * r1) / det);
        if !((p.0 - q[k].0 as f64).hypot(p.1 - q[k].1 as f64) <= limit) {
            usable = false;
        }
        new[k] = (p.0 as f32, p.1 as f32);
    }
    (new, usable, found)
}

/// Edge-refined corners of one tag and whether the refinement held; on failure
/// the input corners are returned.
pub fn refine_tag(img: &GrayView, q: &Quad) -> (Quad, bool) {
    let (mut new, mut usable, found) = fine(img, q, &[PRESHIFT; 4]);
    if !found {
        let c = coarse(img, q);
        (new, usable, _) = fine(img, q, &c);
    }
    if usable {
        (new, true)
    } else {
        (*q, false)
    }
}
