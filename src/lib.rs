//! RapidTag — fast, pure-Rust fiducial marker detection for realtime use,
//! exposed to Python via PyO3/maturin.
//!
//! Ported so far: detectMarkers (CORNER_REFINE_NONE) and CharucoDetector::detectBoard
//! (local-homography path). Not yet ported: solvePnP and the approxCalib path that
//! needs it, refineDetectedMarkers, findChessboardCorners, and calibration.

mod affinity;
mod board;
mod charuco;
mod contours;
mod cornersubpix;
mod detector;
#[allow(non_upper_case_globals)]
mod dictionaries_data;

// A multithread-friendly allocator; glibc malloc serializes the large
// integral-image allocations under the batch detector's 24-way parallelism.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
mod dictionary;
mod imgproc;

use detector::DetectorParameters;
use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Detector parameters (subset relevant to marker detection). Fields mirror
/// cv::aruco::DetectorParameters and default to the same values.
#[pyclass(name = "DetectorParameters")]
#[derive(Clone)]
struct PyDetectorParameters {
    inner: DetectorParameters,
}

#[pymethods]
impl PyDetectorParameters {
    #[new]
    fn new() -> Self {
        PyDetectorParameters {
            inner: DetectorParameters::default(),
        }
    }

    #[getter]
    fn adaptive_thresh_win_size_min(&self) -> i32 {
        self.inner.adaptive_thresh_win_size_min
    }
    #[setter]
    fn set_adaptive_thresh_win_size_min(&mut self, v: i32) {
        self.inner.adaptive_thresh_win_size_min = v;
    }
    #[getter]
    fn adaptive_thresh_win_size_max(&self) -> i32 {
        self.inner.adaptive_thresh_win_size_max
    }
    #[setter]
    fn set_adaptive_thresh_win_size_max(&mut self, v: i32) {
        self.inner.adaptive_thresh_win_size_max = v;
    }
    #[getter]
    fn adaptive_thresh_win_size_step(&self) -> i32 {
        self.inner.adaptive_thresh_win_size_step
    }
    #[setter]
    fn set_adaptive_thresh_win_size_step(&mut self, v: i32) {
        self.inner.adaptive_thresh_win_size_step = v;
    }
    #[getter]
    fn adaptive_thresh_constant(&self) -> f64 {
        self.inner.adaptive_thresh_constant
    }
    #[setter]
    fn set_adaptive_thresh_constant(&mut self, v: f64) {
        self.inner.adaptive_thresh_constant = v;
    }
    #[getter]
    fn polygonal_approx_accuracy_rate(&self) -> f64 {
        self.inner.polygonal_approx_accuracy_rate
    }
    #[setter]
    fn set_polygonal_approx_accuracy_rate(&mut self, v: f64) {
        self.inner.polygonal_approx_accuracy_rate = v;
    }
    #[getter]
    fn error_correction_rate(&self) -> f64 {
        self.inner.error_correction_rate
    }
    #[setter]
    fn set_error_correction_rate(&mut self, v: f64) {
        self.inner.error_correction_rate = v;
    }
    #[getter]
    fn detect_inverted_marker(&self) -> bool {
        self.inner.detect_inverted_marker
    }
    #[setter]
    fn set_detect_inverted_marker(&mut self, v: bool) {
        self.inner.detect_inverted_marker = v;
    }
    #[getter]
    fn min_side_length_canonical_img(&self) -> i32 {
        self.inner.min_side_length_canonical_img
    }
    #[setter]
    fn set_min_side_length_canonical_img(&mut self, v: i32) {
        self.inner.min_side_length_canonical_img = v;
    }
}

/// List the names of the predefined dictionaries this build supports.
#[pyfunction]
fn predefined_dictionaries() -> Vec<String> {
    [
        "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
        "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
        "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
        "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
        "DICT_ARUCO_ORIGINAL", "DICT_ARUCO_MIP_36h12",
        "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9",
        "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// Raw frame pixels extracted from a numpy array (owned, GIL not required to use).
struct FrameData {
    data: Vec<u8>,
    h: usize,
    w: usize,
    ch: usize,
}

type MarkerResult = (Vec<[[f32; 2]; 4]>, Vec<i32>);

/// Pull a contiguous u8 buffer + shape out of a numpy array (requires the GIL).
fn extract_frame(arr: &PyReadonlyArrayDyn<u8>) -> PyResult<FrameData> {
    let view = arr.as_array();
    let shape = view.shape();
    let (h, w, ch) = match shape.len() {
        2 => (shape[0], shape[1], 1),
        3 => (shape[0], shape[1], shape[2]),
        _ => return Err(PyValueError::new_err("image must be 2D (gray) or 3D (BGR)")),
    };
    if ch != 1 && ch != 3 {
        return Err(PyValueError::new_err("image must have 1 or 3 channels"));
    }
    let data: Vec<u8> = match view.as_slice() {
        Some(s) => s.to_vec(),
        None => view.iter().copied().collect(),
    };
    Ok(FrameData { data, h, w, ch })
}

/// Convert internal detections into the Python-facing (corners, ids) shape.
fn to_result(detections: Vec<detector::Detection>) -> MarkerResult {
    let mut corners = Vec::with_capacity(detections.len());
    let mut ids = Vec::with_capacity(detections.len());
    for d in detections {
        corners.push([
            [d.corners[0].0, d.corners[0].1],
            [d.corners[1].0, d.corners[1].1],
            [d.corners[2].0, d.corners[2].1],
            [d.corners[3].0, d.corners[3].1],
        ]);
        ids.push(d.id);
    }
    (corners, ids)
}

/// Detect ArUco markers in `image` (HxW grayscale or HxWx3 BGR, uint8).
///
/// Returns `(corners, ids)`:
///   - `corners`: list of markers, each a 4x2 list of (x, y) float corners
///   - `ids`: list of integer marker ids, aligned with `corners`
#[pyfunction]
#[pyo3(signature = (image, dictionary, parameters=None))]
fn detect_markers(
    py: Python<'_>,
    image: PyReadonlyArrayDyn<u8>,
    dictionary: &str,
    parameters: Option<PyDetectorParameters>,
) -> PyResult<MarkerResult> {
    let dict = dictionary::get_predefined_dictionary(dictionary)
        .ok_or_else(|| PyValueError::new_err(format!("unknown dictionary: {dictionary}")))?;
    let params = parameters.map(|p| p.inner).unwrap_or_default();
    let fd = extract_frame(&image)?;
    let gray = imgproc::to_gray(fd.data, fd.h, fd.w, fd.ch);
    let (detections, _) = py.allow_threads(|| detector::detect_markers(&gray, &dict, &params));
    Ok(to_result(detections))
}

/// Detect markers across a batch of frames using flat (frame × scale) parallelism
/// with the GIL released. Optimal at any batch size — a single frame, a dual-camera
/// pair, or a large offline batch all keep the cores busy without nested threading.
///
/// Returns a list of `(corners, ids)`, one per input frame, in the same order.
#[pyfunction]
#[pyo3(signature = (images, dictionary, parameters=None))]
fn detect_markers_batch(
    py: Python<'_>,
    images: Vec<PyReadonlyArrayDyn<u8>>,
    dictionary: &str,
    parameters: Option<PyDetectorParameters>,
) -> PyResult<Vec<MarkerResult>> {
    let dict = dictionary::get_predefined_dictionary(dictionary)
        .ok_or_else(|| PyValueError::new_err(format!("unknown dictionary: {dictionary}")))?;
    let params = parameters.map(|p| p.inner).unwrap_or_default();

    // Copy pixels out of Python objects while we hold the GIL...
    let grays: Vec<image::GrayImage> = images
        .iter()
        .map(|arr| extract_frame(arr).map(|fd| imgproc::to_gray(fd.data, fd.h, fd.w, fd.ch)))
        .collect::<PyResult<Vec<_>>>()?;

    // ...then detect in parallel with the GIL released.
    let results = py.allow_threads(|| detector::detect_markers_multi(grays, &dict, &params));
    Ok(results.into_iter().map(|(dets, _)| to_result(dets)).collect())
}

/// A ChArUco board: a chessboard with ArUco markers in its white squares.
///
/// The board precomputes its own geometry, so build it once and reuse it across
/// frames rather than constructing it per call.
#[pyclass(name = "CharucoBoard")]
struct PyCharucoBoard {
    inner: board::CharucoBoard,
    dictionary: String,
}

#[pymethods]
impl PyCharucoBoard {
    #[new]
    #[pyo3(signature = (squares_x, squares_y, square_length, marker_length, dictionary, ids=None, legacy_pattern=false))]
    fn new(
        squares_x: usize,
        squares_y: usize,
        square_length: f32,
        marker_length: f32,
        dictionary: &str,
        ids: Option<Vec<usize>>,
        legacy_pattern: bool,
    ) -> PyResult<Self> {
        if dictionary::get_predefined_dictionary(dictionary).is_none() {
            return Err(PyValueError::new_err(format!(
                "unknown dictionary: {dictionary}"
            )));
        }
        let inner = board::CharucoBoard::new(
            squares_x,
            squares_y,
            square_length,
            marker_length,
            ids,
            legacy_pattern,
        )
        .map_err(PyValueError::new_err)?;
        Ok(PyCharucoBoard {
            inner,
            dictionary: dictionary.to_string(),
        })
    }

    /// Interior chessboard corners in board coordinates, as (x, y, z) triples.
    #[getter]
    fn chessboard_corners(&self) -> Vec<[f32; 3]> {
        self.inner
            .chessboard_corners
            .iter()
            .map(|c| [c.0, c.1, c.2])
            .collect()
    }

    /// Marker ids laid out on the board.
    #[getter]
    fn ids(&self) -> Vec<usize> {
        self.inner.ids.clone()
    }
}

/// Detect a ChArUco board in `image` (HxW grayscale or HxWx3 BGR, uint8).
///
/// Returns `(charuco_corners, charuco_ids, marker_corners, marker_ids)`:
///   - `charuco_corners`: Nx2 sub-pixel refined chessboard corners
///   - `charuco_ids`: index of each corner within `board.chessboard_corners`
///   - `marker_corners` / `marker_ids`: the ArUco markers found along the way
#[pyfunction]
#[pyo3(signature = (image, board, parameters=None, min_markers=2, check_markers=true))]
fn detect_charuco_board(
    py: Python<'_>,
    image: PyReadonlyArrayDyn<u8>,
    board: &PyCharucoBoard,
    parameters: Option<PyDetectorParameters>,
    min_markers: i32,
    check_markers: bool,
) -> PyResult<(Vec<[f32; 2]>, Vec<usize>, Vec<[[f32; 2]; 4]>, Vec<i32>)> {
    if !(0..=2).contains(&min_markers) {
        return Err(PyValueError::new_err("min_markers must be 0, 1 or 2"));
    }
    let dict = dictionary::get_predefined_dictionary(&board.dictionary)
        .ok_or_else(|| PyValueError::new_err(format!("unknown dictionary: {}", board.dictionary)))?;
    let params = parameters.map(|p| p.inner).unwrap_or_default();
    let cp = charuco::CharucoParameters {
        min_markers,
        check_markers,
    };
    let fd = extract_frame(&image)?;
    let gray = imgproc::to_gray(fd.data, fd.h, fd.w, fd.ch);
    let res = py.allow_threads(|| charuco::detect_board(&gray, &board.inner, &dict, &params, &cp));

    let corners = res.corners.iter().map(|c| [c.0, c.1]).collect();
    let marker_corners = res
        .marker_corners
        .iter()
        .map(|q| [[q[0].0, q[0].1], [q[1].0, q[1].1], [q[2].0, q[2].1], [q[3].0, q[3].1]])
        .collect();
    Ok((corners, res.ids, marker_corners, res.marker_ids))
}

/// Test hook: interpolated charuco corners and window sizes, pre-refinement.
#[pyfunction]
fn _charuco_predict(
    image: PyReadonlyArrayDyn<u8>,
    board: &PyCharucoBoard,
) -> PyResult<(Vec<[f32; 2]>, Vec<i32>)> {
    let dict = dictionary::get_predefined_dictionary(&board.dictionary)
        .ok_or_else(|| PyValueError::new_err("unknown dictionary"))?;
    let params = DetectorParameters::default();
    let fd = extract_frame(&image)?;
    let gray = imgproc::to_gray(fd.data, fd.h, fd.w, fd.ch);
    let (pts, wins) = charuco::debug_predict(&gray, &board.inner, &dict, &params);
    Ok((pts.iter().map(|p| [p.0, p.1]).collect(), wins))
}

/// Test hook: run cornerSubPix directly so it can be compared against
/// cv2.cornerSubPix on identical inputs. Not part of the public API.
#[pyfunction]
fn _corner_sub_pix(
    image: PyReadonlyArrayDyn<u8>,
    corners: Vec<[f32; 2]>,
    win: usize,
    max_iters: i32,
    eps: f64,
) -> PyResult<Vec<[f32; 2]>> {
    let fd = extract_frame(&image)?;
    let gray = imgproc::to_gray(fd.data, fd.h, fd.w, fd.ch);
    Ok(corners
        .iter()
        .map(|c| {
            let r = cornersubpix::corner_sub_pix(&gray, (c[0], c[1]), win, max_iters, eps);
            [r.0, r.1]
        })
        .collect())
}

#[pymodule]
fn rapidtag(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // On big.LITTLE ARM, pin the detect worker pool to the fast cores before
    // rayon spins up — see affinity.rs (RAPIDTAG_CORES overrides/disables).
    affinity::init_pool();
    m.add_class::<PyDetectorParameters>()?;
    m.add_class::<PyCharucoBoard>()?;
    m.add_function(wrap_pyfunction!(detect_markers, m)?)?;
    m.add_function(wrap_pyfunction!(detect_markers_batch, m)?)?;
    m.add_function(wrap_pyfunction!(detect_charuco_board, m)?)?;
    m.add_function(wrap_pyfunction!(predefined_dictionaries, m)?)?;
    m.add_function(wrap_pyfunction!(_corner_sub_pix, m)?)?;
    m.add_function(wrap_pyfunction!(_charuco_predict, m)?)?;
    Ok(())
}
