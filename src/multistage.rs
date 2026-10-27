//! Multistage corner refinement for one camera: edge refinement, a track layer
//! that keeps tags measured through detector misses, and grid refinement.
//!
//! Port of NOARK's `detect_grid.py` ("ud_grid_trk"). Per frame:
//!   1. edge-refine every detected tag; a tag whose edge refinement fails is
//!      treated as missed (its detector corners are biased);
//!   2. track layer: a tag seen in the last `max_gap` samples but missing now is
//!      re-found by edge refinement seeded with its predicted corners (constant
//!      velocity from its two last sightings), accepted when all four edges are
//!      found and the cell-centre grey levels match its pattern
//!      (`pattern_score >= min_score`);
//!   3. grid refinement of every kept tag.
//! Tracks are kept in full-image coordinates, so the image may be a moving crop
//! whose origin is passed as `offset`. With `parallel`, a frame's tags are refined
//! in parallel on the rayon pool (pinned to the fast cores, see affinity.rs).

use crate::dictionary::Dictionary;
use crate::edge_refine::{self, GrayView, Quad};
use crate::grid_refine::{self, Pattern, QuadF64};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

#[derive(Clone, Copy)]
pub struct Params {
    pub max_gap: i64,
    pub min_score: f64,
    pub box_margin: f64,
    pub grid: bool,
    pub fit_delta: bool,
    /// Refine a frame's tags in parallel on the rayon pool.
    pub parallel: bool,
}

impl Default for Params {
    fn default() -> Self {
        Params { max_gap: 4, min_score: 0.7, box_margin: 3.0, grid: true, fit_delta: false, parallel: false }
    }
}

/// Where a kept tag came from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum How {
    Detected = 0,
    Tracked = 1,
}

pub struct TagOut {
    pub id: i32,
    pub corners: QuadF64,
    pub how: How,
    pub grid_ok: bool,
}

type Sighting = (i64, QuadF64);

struct Track {
    last: Sighting,
    prev: Option<Sighting>,
}

pub struct MultiStage {
    dict: Dictionary,
    patterns: FxHashMap<i32, Pattern>,
    marker_ids: Option<FxHashSet<i32>>,
    tracks: FxHashMap<i32, Track>,
    pub params: Params,
}

fn to_f32(q: &QuadF64) -> Quad {
    std::array::from_fn(|i| (q[i].0 as f32, q[i].1 as f32))
}

fn to_f64(q: &Quad) -> QuadF64 {
    std::array::from_fn(|i| (q[i].0 as f64, q[i].1 as f64))
}

fn shifted(q: &QuadF64, dx: f64, dy: f64) -> QuadF64 {
    std::array::from_fn(|i| (q[i].0 + dx, q[i].1 + dy))
}

impl MultiStage {
    pub fn new(dict: Dictionary, marker_ids: Option<Vec<i32>>, params: Params) -> MultiStage {
        MultiStage {
            dict,
            patterns: FxHashMap::default(),
            marker_ids: marker_ids.map(|v| v.into_iter().collect()),
            tracks: FxHashMap::default(),
            params,
        }
    }

    pub fn reset(&mut self) {
        self.tracks.clear();
    }

    /// Build the grid pattern of `id` once; false when the id is not in the dictionary.
    fn ensure_pattern(&mut self, id: i32) -> bool {
        if self.patterns.contains_key(&id) {
            return true;
        }
        let Some(bits) = usize::try_from(id).ok().and_then(|i| self.dict.marker_bits(i)) else { return false };
        self.patterns.insert(id, Pattern::new(&bits, self.dict.marker_size));
        true
    }

    /// Seed corners (full image) for `sample` from a track's two last sightings.
    fn predict(&self, tr: &Track, sample: i64) -> QuadF64 {
        let (s1, q1) = tr.last;
        match tr.prev {
            Some((s0, q0)) if s1 - s0 <= 2 * self.params.max_gap => {
                let f = (sample - s1) as f64 / (s1 - s0) as f64;
                std::array::from_fn(|i| (q1[i].0 + (q1[i].0 - q0[i].0) * f, q1[i].1 + (q1[i].1 - q0[i].1) * f))
            }
            _ => q1,
        }
    }

    /// Refine one frame's detections (`(id, corners)` in `img` pixels; `offset` =
    /// position of `img` in the full image). Returns the kept tags sorted by id,
    /// corners in `img` pixels.
    pub fn refine(&mut self, img: &GrayView, detections: &[(i32, Quad)], sample: i64, offset: (f64, f64)) -> Vec<TagOut> {
        let (ox, oy) = offset;
        let p = self.params;

        // 1. edge refinement of the detected tags; failed ones count as missed
        let dets: Vec<(i32, Quad)> = detections
            .iter()
            .filter(|(id, _)| self.marker_ids.as_ref().map_or(true, |s| s.contains(id)))
            .copied()
            .collect();
        let edge = |(_, q): &(i32, Quad)| edge_refine::refine_tag(img, q);
        let edges: Vec<(Quad, bool)> =
            if p.parallel { dets.par_iter().map(edge).collect() } else { dets.iter().map(edge).collect() };
        let mut kept: FxHashMap<i32, (QuadF64, How)> = FxHashMap::default();
        let mut failed: FxHashSet<i32> = FxHashSet::default();
        for (&(id, _), (e, ok)) in dets.iter().zip(&edges) {
            kept.insert(id, (to_f64(e), How::Detected));
            if !ok {
                failed.insert(id);
            }
        }
        kept.retain(|id, _| !failed.contains(id));

        // 2. track layer: re-find recently seen tags from their predicted corners
        let (bw, bh) = (img.w as f64, img.h as f64);
        let mut seeds: Vec<(i32, QuadF64)> = Vec::new();
        for (&id, tr) in &self.tracks {
            if kept.contains_key(&id) || sample - tr.last.0 > p.max_gap {
                continue;
            }
            let q = shifted(&self.predict(tr, sample), -ox, -oy);
            let inside = q.iter().all(|c| {
                c.0 >= p.box_margin && c.1 >= p.box_margin && c.0 <= bw - 1.0 - p.box_margin && c.1 <= bh - 1.0 - p.box_margin
            });
            if inside {
                seeds.push((id, q));
            }
        }
        seeds.retain(|&(id, _)| self.ensure_pattern(id));
        let patterns = &self.patterns;
        let refind = |(id, seed): &(i32, QuadF64)| {
            let (e, ok) = edge_refine::refine_tag(img, &to_f32(seed));
            let e = to_f64(&e);
            (ok && grid_refine::pattern_score(img, &e, &patterns[id]) >= p.min_score).then_some(e)
        };
        let found: Vec<Option<QuadF64>> =
            if p.parallel { seeds.par_iter().map(refind).collect() } else { seeds.iter().map(refind).collect() };
        for (&(id, _), e) in seeds.iter().zip(found) {
            if let Some(e) = e {
                kept.insert(id, (e, How::Tracked));
            }
        }
        for (&id, &(q, _)) in &kept {
            let full = shifted(&q, ox, oy);
            let prev = self.tracks.get(&id).map(|t| t.last);
            self.tracks.insert(id, Track { last: (sample, full), prev });
        }

        // 3. grid refinement
        let mut out: Vec<TagOut> = kept
            .into_iter()
            .map(|(id, (q, how))| TagOut { id, corners: q, how, grid_ok: false })
            .collect();
        out.sort_by_key(|t| t.id);
        if p.grid {
            for t in &out {
                self.ensure_pattern(t.id);
            }
            let patterns = &self.patterns;
            let grid = |t: &mut TagOut| {
                if let Some(pat) = patterns.get(&t.id) {
                    let (g, info) = grid_refine::refine_tag(img, &t.corners, pat, p.fit_delta, false);
                    t.corners = g;
                    t.grid_ok = info.ok;
                }
            };
            if p.parallel {
                out.par_iter_mut().for_each(grid);
            } else {
                out.iter_mut().for_each(grid);
            }
        }
        out
    }
}
