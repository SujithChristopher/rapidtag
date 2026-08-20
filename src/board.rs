//! Port of cv::aruco::CharucoBoard geometry (aruco_board.cpp, CharucoBoardImpl).
//!
//! A ChArUco board is a chessboard with ArUco markers in the white squares. The
//! detector needs three precomputed tables from the board layout:
//!   * `obj_points`      — the 4 corners of each marker, in board coordinates
//!   * `chessboard_corners` — the interior chessboard corners we want to find
//!   * `nearest_marker_idx` / `nearest_marker_corners` — for each chessboard
//!     corner, which markers touch it and at which of their 4 corners
//!
//! The last pair is what makes marker-driven interpolation possible: a detected
//! marker's homography is only used to predict the corners it actually borders.

/// A point in board space (z is always 0 for a planar board).
pub type Pt3 = (f32, f32, f32);

pub struct CharucoBoard {
    /// Chessboard size in squares (width = squares_x, height = squares_y).
    pub squares_x: usize,
    pub squares_y: usize,
    /// Physical side length of a chessboard square (normally metres).
    pub square_length: f32,
    /// Physical side length of a marker (normally metres).
    pub marker_length: f32,
    /// Pre-4.6.0 pattern: even-row-count boards put a white box top-left.
    pub legacy_pattern: bool,
    /// Marker ids, one per entry of `obj_points`.
    pub ids: Vec<usize>,
    /// Marker corners in board space, 4 per marker.
    pub obj_points: Vec<[Pt3; 4]>,
    /// Interior chessboard corners, (squares_x-1) * (squares_y-1) of them.
    pub chessboard_corners: Vec<Pt3>,
    /// For each chessboard corner, the indices of the nearest markers.
    pub nearest_marker_idx: Vec<Vec<usize>>,
    /// For each chessboard corner, the nearest corner (0..4) of each nearest marker.
    pub nearest_marker_corners: Vec<Vec<usize>>,
    pub right_bottom_border: Pt3,
}

impl CharucoBoard {
    /// Build a board. `ids` defaults to `0..n_markers` (the first ids of the dictionary).
    pub fn new(
        squares_x: usize,
        squares_y: usize,
        square_length: f32,
        marker_length: f32,
        ids: Option<Vec<usize>>,
        legacy_pattern: bool,
    ) -> Result<CharucoBoard, String> {
        let mut b = CharucoBoard {
            squares_x,
            squares_y,
            square_length,
            marker_length,
            legacy_pattern,
            ids: ids.unwrap_or_default(),
            obj_points: Vec::new(),
            chessboard_corners: Vec::new(),
            nearest_marker_idx: Vec::new(),
            nearest_marker_corners: Vec::new(),
            right_bottom_border: (0.0, 0.0, 0.0),
        };
        b.create()?;
        Ok(b)
    }

    /// createCharucoBoard(): lay out markers on the white squares, then the
    /// interior chessboard corners.
    fn create(&mut self) -> Result<(), String> {
        let diff = (self.square_length - self.marker_length) / 2.0;
        let total_markers = self.ids.len();
        let mut next_id = 0usize;
        self.obj_points.clear();

        for y in 0..self.squares_y {
            for x in 0..self.squares_x {
                // Which squares carry a marker. The legacy pattern flips the
                // parity, but only for even row counts.
                if self.legacy_pattern && self.squares_y % 2 == 0 {
                    if (y + 1) % 2 == x % 2 {
                        continue; // black corner, no marker here
                    }
                } else if y % 2 == x % 2 {
                    continue; // black corner, no marker here
                }

                let c0 = (
                    x as f32 * self.square_length + diff,
                    y as f32 * self.square_length + diff,
                    0.0,
                );
                let m = self.marker_length;
                self.obj_points.push([
                    c0,
                    (c0.0 + m, c0.1, 0.0),
                    (c0.0 + m, c0.1 + m, 0.0),
                    (c0.0, c0.1 + m, 0.0),
                ]);
                if total_markers == 0 {
                    self.ids.push(next_id);
                }
                next_id += 1;
            }
        }
        if total_markers > 0 && next_id != total_markers {
            return Err(format!(
                "Size of ids must be equal to the number of markers: {next_id}"
            ));
        }

        self.chessboard_corners.clear();
        for y in 0..self.squares_y.saturating_sub(1) {
            for x in 0..self.squares_x.saturating_sub(1) {
                self.chessboard_corners.push((
                    (x + 1) as f32 * self.square_length,
                    (y + 1) as f32 * self.square_length,
                    0.0,
                ));
            }
        }
        self.right_bottom_border = (
            self.squares_x as f32 * self.square_length,
            self.squares_y as f32 * self.square_length,
            0.0,
        );
        self.calc_nearest_marker_corners();
        Ok(())
    }

    /// calcNearestMarkerCorners(): for each chessboard corner find the closest
    /// marker(s) and, within each, the closest of its 4 corners.
    ///
    /// Ties matter: a chessboard corner in the board interior is equidistant from
    /// two diagonal markers, and both are kept so interpolation can average them.
    /// The tie window is 1% of a square, matching OpenCV.
    fn calc_nearest_marker_corners(&mut self) {
        let n_corners = self.chessboard_corners.len();
        let n_markers = self.obj_points.len();
        self.nearest_marker_idx = vec![Vec::new(); n_corners];
        self.nearest_marker_corners = vec![Vec::new(); n_corners];
        let tie_tol = (0.01 * self.square_length).powi(2) as f64;

        for i in 0..n_corners {
            let cc = self.chessboard_corners[i];
            let mut min_dist = -1f64;
            for j in 0..n_markers {
                // distance from marker centre to chessboard corner (xy only)
                let mut center = (0f32, 0f32);
                for k in 0..4 {
                    center.0 += self.obj_points[j][k].0;
                    center.1 += self.obj_points[j][k].1;
                }
                center.0 /= 4.0;
                center.1 /= 4.0;
                let dx = (cc.0 - center.0) as f64;
                let dy = (cc.1 - center.1) as f64;
                let sq = dx * dx + dy * dy;

                if j == 0 || (sq - min_dist).abs() < tie_tol {
                    // first marker, or tied with the current best: keep both
                    self.nearest_marker_idx[i].push(j);
                    min_dist = sq;
                } else if sq < min_dist {
                    // strictly closer: this marker replaces the previous set
                    self.nearest_marker_idx[i].clear();
                    self.nearest_marker_idx[i].push(j);
                    min_dist = sq;
                }
            }

            // within each nearest marker, pick the corner closest to the chessboard corner
            let n_near = self.nearest_marker_idx[i].len();
            self.nearest_marker_corners[i] = vec![0usize; n_near];
            for j in 0..n_near {
                let mj = self.nearest_marker_idx[i][j];
                let mut min_corner = -1f64;
                for k in 0..4 {
                    let p = self.obj_points[mj][k];
                    let dx = (cc.0 - p.0) as f64;
                    let dy = (cc.1 - p.1) as f64;
                    let sq = dx * dx + dy * dy;
                    if k == 0 || sq < min_corner {
                        min_corner = sq;
                        self.nearest_marker_corners[i][j] = k;
                    }
                }
            }
        }
    }
}
