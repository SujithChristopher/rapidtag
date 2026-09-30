//! Suzuki-Abe border following (RETR_LIST, CHAIN_APPROX_NONE).
//!
//! Produces the same ordered contours as `imageproc::find_contours`, but faster:
//!   - a 1-pixel zero-padded label buffer removes all neighbor bounds checks
//!   - integer direction indices replace the VecDeque rotate-and-search
//!   - no hierarchy/parent bookkeeping (unused by the aruco pipeline)
//!
//! The neighbor ordering, labeling (positive NBD / negative right-edge), and
//! border-start conditions match imageproc exactly, so downstream output is
//! identical.

// Label alphabet (i8, not i32): the Suzuki-Abe NBD *magnitude* is only used for
// hierarchy, which the aruco pipeline discards — so we never need distinct
// per-contour ids, only to mark visited pixels and keep the +/- sign that the
// border-start test relies on. Collapsing to {0,FG,POS,NEG} shrinks this buffer
// 4x (the dominant cost is streaming it, not the tracing).
//
// FG is `pub` because the adaptive threshold writes the foreground mask straight
// into this same padded buffer (fused, no separate 0/255 image), so it must agree
// on the foreground marker value.
pub const FG: i8 = 1; // unlabeled foreground
const POS: i8 = 2; // traced, positive (interior-capable)
const NEG: i8 = -2; // traced, negative (right edge)

// 8-neighborhood offsets, in imageproc's order (index == search position).
const DIRS: [(i32, i32); 8] = [
    (-1, 0),  // 0 W
    (-1, -1), // 1 NW
    (0, -1),  // 2 N
    (1, -1),  // 3 NE
    (1, 0),   // 4 E
    (1, 1),   // 5 SE
    (0, 1),   // 6 S
    (-1, 1),  // 7 SW
];

/// Trace all borders of foreground regions, invoking `f` once per contour with the
/// ordered pixel coordinates in a *reused* buffer.
///
/// `lbl` is a 1-pixel zero-padded label buffer (stride `w+2`, height `h+2`) that the
/// adaptive threshold has already filled: `FG` at foreground pixels, `0` everywhere
/// else including the border ring. Fusing the binarize into the threshold this way
/// saves a whole full-image pass and the separate 0/255 image. The buffer is
/// consumed (mutated in place with trace labels); the caller hands over ownership.
///
/// The per-contour buffer is borrowed only for the duration of each call — the caller
/// copies out what it needs. This avoids allocating a `Vec` per contour, which matters
/// because a thresholded frame contains hundreds of tiny noise contours that the
/// aruco size filter immediately discards.
pub fn for_each_contour<F: FnMut(&[(i32, i32)])>(lbl: &mut [i8], w: i32, h: i32, mut f: F) {
    if w == 0 || h == 0 {
        return;
    }
    let stride = (w + 2) as usize;
    debug_assert_eq!(lbl.len(), stride * (h + 2) as usize);

    let max_points = (w as usize) * (h as usize) + 1; // hang guard
    let mut points: Vec<(i32, i32)> = Vec::new(); // reused across contours

    // DIRS as offsets into the padded buffer, so the tracer steps by index.
    let s = stride as isize;
    let offs: [isize; 8] = [-1, -s - 1, -s, -s + 1, 1, s + 1, s, s - 1];
    const EAST: usize = 4;

    for y in 0..h {
        // Row base for real x=0. The 1px pad guarantees `row_base-1` and
        // `row_base + (w-1) + 1` are in-bounds, so left/right neighbours need no
        // edge test.
        let row_base = (y + 1) as usize * stride + 1;
        let mut x = 0usize;
        while x < w as usize {
            // Skip background 8 pixels at a time.
            if x + 8 <= w as usize {
                let word: [i8; 8] = lbl[row_base + x..row_base + x + 8].try_into().unwrap();
                if u64::from_ne_bytes(word.map(|b| b as u8)) == 0 {
                    x += 8;
                    continue;
                }
            }
            let v = lbl[row_base + x];
            if v == 0 {
                x += 1;
                continue;
            }
            // border start: outer (unlabeled fg with background to the left) or
            // hole (fg with background to the right). The x>0 / x+1<w guards match
            // imageproc exactly — the image edge itself is NOT treated as a
            // background neighbour here, so we cannot lean on the zero pad.
            let d0 = if v == FG && x > 0 && lbl[row_base + x - 1] == 0 {
                0 // W
            } else if v > 0 && x + 1 < w as usize && lbl[row_base + x + 1] == 0 {
                EAST
            } else {
                x += 1;
                continue;
            };
            let curr = (x as i32, y);
            let curr_i = row_base + x;

            // pos1: first non-zero neighbor, searching clockwise from the
            // background neighbour.
            let mut pos1 = None;
            for k in 0..8 {
                let d = (d0 + k) & 7;
                if lbl[(curr_i as isize + offs[d]) as usize] != 0 {
                    pos1 = Some(d);
                    break;
                }
            }

            points.clear();
            match pos1 {
                None => {
                    // isolated pixel
                    points.push(curr);
                    lbl[curr_i] = NEG;
                }
                Some(d1) => {
                    let pos1_i = (curr_i as isize + offs[d1]) as usize;
                    // pos3 is the current border pixel; `base` is the direction
                    // from pos3 back to pos2 (the previous pixel).
                    let (mut p3x, mut p3y) = curr;
                    let mut p3_i = curr_i;
                    let mut base = d1;
                    loop {
                        points.push((p3x, p3y));

                        // pos4: first non-zero neighbor scanning counter-clockwise
                        // from just before `base`.
                        let mut k4 = None;
                        for k in (0..8).rev() {
                            let d = (base + k) & 7;
                            if lbl[(p3_i as isize + offs[d]) as usize] != 0 {
                                k4 = Some(k);
                                break;
                            }
                        }

                        // right-edge test: did the CCW scan pass East before pos4?
                        // The scan visits k = 7, 6, ..., so East came first iff its
                        // k is larger than pos4's (always, if nothing was found).
                        let k_east = (EAST + 8 - base) & 7;
                        let is_right_edge = match k4 {
                            Some(k) => k_east > k,
                            None => true,
                        };

                        if p3x + 1 == w || is_right_edge {
                            lbl[p3_i] = NEG;
                        } else if lbl[p3_i] == FG {
                            lbl[p3_i] = POS;
                        }

                        let (p4_i, d4) = match k4 {
                            Some(k) => {
                                let d = (base + k) & 7;
                                ((p3_i as isize + offs[d]) as usize, Some(d))
                            }
                            None => (p3_i, None),
                        };
                        if p4_i == curr_i && p3_i == pos1_i {
                            break;
                        }
                        match d4 {
                            Some(d) => {
                                p3x += DIRS[d].0;
                                p3y += DIRS[d].1;
                                p3_i = p4_i;
                                base = (d + 4) & 7;
                            }
                            // pos4 == pos3: pos2 becomes pos3 itself. Unreachable
                            // for a traced border (pos1 exists), but keep the
                            // original semantics: the direction to pos2 is then
                            // undefined, so stop.
                            None => break,
                        }

                        if points.len() > max_points {
                            break; // defensive; should never trigger
                        }
                    }
                }
            }
            f(&points);
            x += 1;
        }
    }
}
