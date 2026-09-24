//! Rigid multi-marker geometry and correspondence assembly.

use std::collections::{HashMap, HashSet};

use crate::pnp::{Pt2d, Pt3d};

/// Precomputed marker corners expressed in a shared rigid-body reference frame.
pub struct RigidBody {
    tag_size_m: f64,
    marker_ids: Vec<i32>,
    marker_index: HashMap<i32, usize>,
    object_corners: Vec<[Pt3d; 4]>,
}

/// Flattened correspondences built from one frame of marker detections.
pub struct Correspondences {
    pub object_points: Vec<Pt3d>,
    pub image_points: Vec<Pt2d>,
    pub marker_ids_per_corner: Vec<i32>,
    pub used_marker_ids: Vec<i32>,
}

impl RigidBody {
    pub fn new(
        tag_size_m: f64,
        marker_ids: Vec<i32>,
        rotations_marker_to_reference: Vec<[[f64; 3]; 3]>,
        translations_marker_to_reference: Vec<[f64; 3]>,
    ) -> Result<Self, String> {
        if !tag_size_m.is_finite() || tag_size_m <= 0.0 {
            return Err("tag_size_m must be finite and greater than zero".to_string());
        }
        if marker_ids.is_empty() {
            return Err("rigid body must contain at least one marker".to_string());
        }
        if marker_ids.len() != rotations_marker_to_reference.len()
            || marker_ids.len() != translations_marker_to_reference.len()
        {
            return Err(
                "marker_ids, rotations_marker_to_reference and translations_marker_to_reference must have the same length"
                    .to_string(),
            );
        }

        let mut marker_index = HashMap::with_capacity(marker_ids.len());
        let mut object_corners = Vec::with_capacity(marker_ids.len());
        let half = tag_size_m * 0.5;
        // Detector/OpenCV order: top-left, top-right, bottom-right, bottom-left.
        let local = [
            (-half, half, 0.0),
            (half, half, 0.0),
            (half, -half, 0.0),
            (-half, -half, 0.0),
        ];

        for (index, ((&marker_id, rotation), translation)) in marker_ids
            .iter()
            .zip(&rotations_marker_to_reference)
            .zip(&translations_marker_to_reference)
            .enumerate()
        {
            if marker_index.insert(marker_id, index).is_some() {
                return Err(format!("duplicate marker id {marker_id}"));
            }
            if rotation.iter().flatten().any(|v| !v.is_finite())
                || translation.iter().any(|v| !v.is_finite())
            {
                return Err(format!(
                    "marker {marker_id} transform must contain only finite values"
                ));
            }
            if !proper_rotation(rotation) {
                return Err(format!(
                    "marker {marker_id} rotation must be an orthonormal matrix with determinant +1"
                ));
            }

            let mut corners = [(0.0, 0.0, 0.0); 4];
            for (corner, &(x, y, z)) in corners.iter_mut().zip(&local) {
                // Column-vector convention: p_reference = R * p_marker + t.
                *corner = (
                    rotation[0][0] * x + rotation[0][1] * y + rotation[0][2] * z + translation[0],
                    rotation[1][0] * x + rotation[1][1] * y + rotation[1][2] * z + translation[1],
                    rotation[2][0] * x + rotation[2][1] * y + rotation[2][2] * z + translation[2],
                );
            }
            object_corners.push(corners);
        }

        Ok(Self {
            tag_size_m,
            marker_ids,
            marker_index,
            object_corners,
        })
    }

    pub fn tag_size_m(&self) -> f64 {
        self.tag_size_m
    }

    pub fn marker_ids(&self) -> &[i32] {
        &self.marker_ids
    }

    pub fn assemble(
        &self,
        marker_corners: &[[[f64; 2]; 4]],
        marker_ids: &[i32],
    ) -> Result<Correspondences, String> {
        if marker_corners.len() != marker_ids.len() {
            return Err("marker_corners and marker_ids must have the same length".to_string());
        }

        let mut seen = HashSet::with_capacity(marker_ids.len());
        let mut object_points = Vec::with_capacity(marker_ids.len() * 4);
        let mut image_points = Vec::with_capacity(marker_ids.len() * 4);
        let mut marker_ids_per_corner = Vec::with_capacity(marker_ids.len() * 4);
        let mut used_marker_ids = Vec::with_capacity(marker_ids.len());

        for (&marker_id, image_corners) in marker_ids.iter().zip(marker_corners) {
            if !seen.insert(marker_id) {
                return Err(format!("duplicate detected marker id {marker_id}"));
            }
            let Some(&index) = self.marker_index.get(&marker_id) else {
                continue;
            };
            if image_corners.iter().flatten().any(|v| !v.is_finite()) {
                return Err(format!(
                    "marker {marker_id} corners must contain only finite values"
                ));
            }
            used_marker_ids.push(marker_id);
            for (object, image) in self.object_corners[index].iter().zip(image_corners) {
                object_points.push(*object);
                image_points.push((image[0], image[1]));
                marker_ids_per_corner.push(marker_id);
            }
        }

        Ok(Correspondences {
            object_points,
            image_points,
            marker_ids_per_corner,
            used_marker_ids,
        })
    }
}

fn proper_rotation(rotation: &[[f64; 3]; 3]) -> bool {
    const TOLERANCE: f64 = 1e-3;
    for i in 0..3 {
        let norm = rotation[i].iter().map(|v| v * v).sum::<f64>();
        if (norm - 1.0).abs() > TOLERANCE {
            return false;
        }
        for j in i + 1..3 {
            let dot = (0..3).map(|k| rotation[i][k] * rotation[j][k]).sum::<f64>();
            if dot.abs() > TOLERANCE {
                return false;
            }
        }
    }
    let determinant = rotation[0][0]
        * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0]);
    (determinant - 1.0).abs() <= TOLERANCE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transforms_marker_corners_into_reference_frame() {
        let body = RigidBody::new(
            0.2,
            vec![7],
            vec![[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
            vec![[1.0, 2.0, 3.0]],
        )
        .unwrap();
        let correspondences = body
            .assemble(
                &[[[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 40.0]]],
                &[7],
            )
            .unwrap();

        assert_eq!(
            correspondences.object_points,
            vec![
                (0.9, 1.9, 3.0),
                (0.9, 2.1, 3.0),
                (1.1, 2.1, 3.0),
                (1.1, 1.9, 3.0),
            ]
        );
        assert_eq!(correspondences.used_marker_ids, vec![7]);
    }

    #[test]
    fn skips_unknown_markers_and_rejects_duplicate_detections() {
        let body = RigidBody::new(
            0.1,
            vec![1],
            vec![[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
            vec![[0.0, 0.0, 0.0]],
        )
        .unwrap();
        let corners = [[[0.0; 2]; 4]; 2];
        assert!(body
            .assemble(&corners[..1], &[99])
            .unwrap()
            .object_points
            .is_empty());
        assert!(body.assemble(&corners, &[1, 1]).is_err());
    }
}
