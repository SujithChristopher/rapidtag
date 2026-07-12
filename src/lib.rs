//! RapidTag — fast, pure-Rust fiducial marker detection for realtime use,
//! exposed to Python via PyO3/maturin. v1: detectMarkers (CORNER_REFINE_NONE).

mod contours;
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

#[pymodule]
fn rapidtag(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDetectorParameters>()?;
    m.add_function(wrap_pyfunction!(detect_markers, m)?)?;
    m.add_function(wrap_pyfunction!(detect_markers_batch, m)?)?;
    m.add_function(wrap_pyfunction!(predefined_dictionaries, m)?)?;
    Ok(())
}
