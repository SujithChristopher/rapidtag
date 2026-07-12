//! Port of cv::aruco::Dictionary — marker identification via Hamming distance
//! over cell pixel-ratios, matching the CellBitMasks logic in aruco_dictionary.cpp.

use crate::dictionaries_data as data;

pub const DEFAULT_VALID_BIT_ID_THRESHOLD: f32 = 0.5;

pub struct Dictionary {
    pub marker_size: usize,
    pub max_correction_bits: i32,
    /// One row per marker; each row is `4 * nbytes` bytes: rot0|rot1|rot2|rot3.
    pub bytes_list: Vec<Vec<u8>>,
}

impl Dictionary {
    fn from_table(rows: &[&[u8]], marker_size: usize, max_correction_bits: i32) -> Dictionary {
        Dictionary {
            marker_size,
            max_correction_bits,
            bytes_list: rows.iter().map(|r| r.to_vec()).collect(),
        }
    }

    /// bytes per rotation = ceil(markerSize^2 / 8)
    fn s(&self) -> usize {
        (self.marker_size * self.marker_size + 7) / 8
    }

    /// Identify a marker from its per-cell white-pixel ratios (row-major, len = markerSize^2).
    /// Returns (marker id, rotation 0..4) or None.
    pub fn identify(
        &self,
        cell_ratio: &[f32],
        max_correction_rate: f64,
        valid_bit_threshold: f32,
    ) -> Option<(usize, usize)> {
        let masks = CellBitMasks::new(cell_ratio, self.marker_size, valid_bit_threshold);
        let max_corr = (self.max_correction_bits as f64 * max_correction_rate) as i32;
        let s = self.s();
        for (m, row) in self.bytes_list.iter().enumerate() {
            let (dist, rot) = masks.hamming_to_id(row, s);
            if dist as i32 <= max_corr {
                return Some((m, rot));
            }
        }
        None
    }
}

/// Bit masks of cells that are "not black" (not0) and "not white" (not1),
/// packed row-major exactly like Dictionary::getByteListFromBits.
struct CellBitMasks {
    not0: Vec<u8>,
    notxor: Vec<u8>,
    total_cells: usize,
}

impl CellBitMasks {
    fn new(cell_ratio: &[f32], marker_size: usize, valid_bit_threshold: f32) -> CellBitMasks {
        let s = (marker_size * marker_size + 7) / 8;
        let mut not0 = vec![0u8; s];
        let mut not1 = vec![0u8; s];
        let inv = 1.0 - valid_bit_threshold;

        let mut not0b = 0u8;
        let mut not1b = 0u8;
        let mut byte = 0usize;
        let mut bit = 0u8;
        for j in 0..marker_size {
            for i in 0..marker_size {
                not0b <<= 1;
                not1b <<= 1;
                let r = cell_ratio[j * marker_size + i];
                if r > valid_bit_threshold {
                    not0b |= 1;
                }
                if r < inv {
                    not1b |= 1;
                }
                bit += 1;
                if bit == 8 {
                    not0[byte] = not0b;
                    not1[byte] = not1b;
                    not0b = 0;
                    not1b = 0;
                    byte += 1;
                    bit = 0;
                }
            }
        }
        if bit != 0 {
            not0[byte] = not0b;
            not1[byte] = not1b;
        }
        let notxor: Vec<u8> = (0..s).map(|k| not0[k] ^ not1[k]).collect();
        CellBitMasks {
            not0,
            notxor,
            total_cells: marker_size * marker_size,
        }
    }

    /// Smallest Hamming distance to marker `row` over its 4 rotations.
    fn hamming_to_id(&self, row: &[u8], s: usize) -> (u32, usize) {
        let mut min = self.total_cells as u32 + 1;
        let mut best_rot = 0usize;
        for r in 0..4usize {
            let base = r * s;
            let mut d = 0u32;
            for k in 0..s {
                // not0 ^ ((not0 ^ not1) & bytesRot)
                let t = self.notxor[k] & row[base + k];
                d += (self.not0[k] ^ t).count_ones();
            }
            if d < min {
                min = d;
                best_rot = r;
                if d == 0 {
                    break;
                }
            }
        }
        (min, best_rot)
    }
}

fn slice_rows<const N: usize>(table: &'static [[u8; N]], count: usize) -> Vec<&'static [u8]> {
    table[..count].iter().map(|r| r.as_slice()).collect()
}

/// Predefined dictionaries, mirroring cv::aruco::getPredefinedDictionary.
/// Smaller dicts subset the 1000-marker base tables, as in OpenCV.
pub fn get_predefined_dictionary(name: &str) -> Option<Dictionary> {
    let key = name.to_ascii_uppercase();
    let key = key.strip_prefix("DICT_").unwrap_or(&key);
    let mk = |rows: Vec<&'static [u8]>, ms: usize, mc: i32| {
        Some(Dictionary::from_table(&rows, ms, mc))
    };
    match key {
        "4X4_50" => mk(slice_rows(&data::DICT_4X4_1000, 50), 4, (4 - 1) / 2),
        "4X4_100" => mk(slice_rows(&data::DICT_4X4_1000, 100), 4, (3 - 1) / 2),
        "4X4_250" => mk(slice_rows(&data::DICT_4X4_1000, 250), 4, (3 - 1) / 2),
        "4X4_1000" => mk(slice_rows(&data::DICT_4X4_1000, 1000), 4, (2 - 1) / 2),
        "5X5_50" => mk(slice_rows(&data::DICT_5X5_1000, 50), 5, (8 - 1) / 2),
        "5X5_100" => mk(slice_rows(&data::DICT_5X5_1000, 100), 5, (7 - 1) / 2),
        "5X5_250" => mk(slice_rows(&data::DICT_5X5_1000, 250), 5, (6 - 1) / 2),
        "5X5_1000" => mk(slice_rows(&data::DICT_5X5_1000, 1000), 5, (5 - 1) / 2),
        "6X6_50" => mk(slice_rows(&data::DICT_6X6_1000, 50), 6, (13 - 1) / 2),
        "6X6_100" => mk(slice_rows(&data::DICT_6X6_1000, 100), 6, (12 - 1) / 2),
        "6X6_250" => mk(slice_rows(&data::DICT_6X6_1000, 250), 6, (11 - 1) / 2),
        "6X6_1000" => mk(slice_rows(&data::DICT_6X6_1000, 1000), 6, (9 - 1) / 2),
        "7X7_50" => mk(slice_rows(&data::DICT_7X7_1000, 50), 7, (19 - 1) / 2),
        "7X7_100" => mk(slice_rows(&data::DICT_7X7_1000, 100), 7, (18 - 1) / 2),
        "7X7_250" => mk(slice_rows(&data::DICT_7X7_1000, 250), 7, (17 - 1) / 2),
        "7X7_1000" => mk(slice_rows(&data::DICT_7X7_1000, 1000), 7, (14 - 1) / 2),
        "ARUCO_ORIGINAL" => mk(slice_rows(&data::DICT_ARUCO, 1024), 5, (3 - 1) / 2),
        "ARUCO_MIP_36H12" => mk(slice_rows(&data::DICT_ARUCO_MIP_36h12, 250), 6, (12 - 1) / 2),
        "APRILTAG_16H5" => mk(slice_rows(&data::DICT_APRILTAG_16h5, 30), 4, (5 - 1) / 2),
        "APRILTAG_25H9" => mk(slice_rows(&data::DICT_APRILTAG_25h9, 35), 5, (9 - 1) / 2),
        "APRILTAG_36H10" => mk(slice_rows(&data::DICT_APRILTAG_36h10, 2320), 6, (10 - 1) / 2),
        "APRILTAG_36H11" => mk(slice_rows(&data::DICT_APRILTAG_36h11, 587), 6, (11 - 1) / 2),
        _ => None,
    }
}
