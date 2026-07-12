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

use image::GrayImage;

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

#[inline(always)]
fn dir_index(d: (i32, i32)) -> usize {
    match d {
        (-1, 0) => 0,
        (-1, -1) => 1,
        (0, -1) => 2,
        (1, -1) => 3,
        (1, 0) => 4,
        (1, 1) => 5,
        (0, 1) => 6,
        (-1, 1) => 7,
        _ => unreachable!(),
    }
}

/// Trace all borders of foreground (non-zero) regions in a binary image, invoking
/// `f` once per contour with the ordered pixel coordinates in a *reused* buffer.
///
/// The buffer is borrowed only for the duration of each call — the caller copies
/// out what it needs. This avoids allocating a `Vec` per contour, which matters
/// because a thresholded frame contains hundreds of tiny noise contours that the
/// aruco size filter immediately discards.
pub fn for_each_contour<F: FnMut(&[(i32, i32)])>(thresh: &GrayImage, mut f: F) {
    let w = thresh.width() as i32;
    let h = thresh.height() as i32;
    if w == 0 || h == 0 {
        return;
    }
    let stride = (w + 2) as usize;
    let mut lbl = vec![0i32; stride * (h + 2) as usize];
    let src = thresh.as_raw();

    // padded index for a real coordinate; valid for x in [-1, w], y in [-1, h]
    let idx = |x: i32, y: i32| -> usize { (y + 1) as usize * stride + (x + 1) as usize };

    // binarize into the padded buffer (1 = unlabeled foreground)
    for y in 0..h {
        let row = &src[(y * w) as usize..(y * w + w) as usize];
        let base = (y + 1) as usize * stride + 1;
        for x in 0..w as usize {
            if row[x] > 0 {
                lbl[base + x] = 1;
            }
        }
    }

    let max_points = (w as usize) * (h as usize) + 1; // hang guard
    let mut points: Vec<(i32, i32)> = Vec::new(); // reused across contours
    let mut nbd = 1i32;

    for y in 0..h {
        for x in 0..w {
            let v = lbl[idx(x, y)];
            if v == 0 {
                continue;
            }
            // border start: outer (unlabeled fg with background to the left) or
            // hole (fg with background to the right)
            let adj = if v == 1 && x > 0 && lbl[idx(x - 1, y)] == 0 {
                (x - 1, y)
            } else if v > 0 && x + 1 < w && lbl[idx(x + 1, y)] == 0 {
                (x + 1, y)
            } else {
                continue;
            };
            nbd += 1;
            let curr = (x, y);

            // pos1: first non-zero neighbor, searching clockwise from `adj`
            let d0 = dir_index((adj.0 - curr.0, adj.1 - curr.1));
            let mut pos1 = None;
            for k in 0..8 {
                let d = DIRS[(d0 + k) & 7];
                if lbl[idx(curr.0 + d.0, curr.1 + d.1)] != 0 {
                    pos1 = Some((curr.0 + d.0, curr.1 + d.1));
                    break;
                }
            }

            points.clear();
            match pos1 {
                None => {
                    // isolated pixel
                    points.push((x, y));
                    lbl[idx(x, y)] = -nbd;
                }
                Some(pos1) => {
                    let mut pos2 = pos1;
                    let mut pos3 = curr;
                    loop {
                        points.push(pos3);
                        let base = dir_index((pos2.0 - pos3.0, pos2.1 - pos3.1));

                        // pos4: first non-zero neighbor scanning counter-clockwise
                        let mut pos4 = pos3;
                        for k in (0..8).rev() {
                            let d = DIRS[(base + k) & 7];
                            if lbl[idx(pos3.0 + d.0, pos3.1 + d.1)] != 0 {
                                pos4 = (pos3.0 + d.0, pos3.1 + d.1);
                                break;
                            }
                        }

                        // right-edge test: did the CCW scan pass East before pos4?
                        let target = (pos4.0 - pos3.0, pos4.1 - pos3.1);
                        let mut is_right_edge = false;
                        for k in (0..8).rev() {
                            let d = DIRS[(base + k) & 7];
                            if d == target {
                                break;
                            }
                            if d == (1, 0) {
                                is_right_edge = true;
                                break;
                            }
                        }

                        if pos3.0 + 1 == w || is_right_edge {
                            lbl[idx(pos3.0, pos3.1)] = -nbd;
                        } else if lbl[idx(pos3.0, pos3.1)] == 1 {
                            lbl[idx(pos3.0, pos3.1)] = nbd;
                        }

                        if pos4 == curr && pos3 == pos1 {
                            break;
                        }
                        pos2 = pos3;
                        pos3 = pos4;

                        if points.len() > max_points {
                            break; // defensive; should never trigger
                        }
                    }
                }
            }
            f(&points);
        }
    }
}
