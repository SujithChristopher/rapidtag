//! Port of cv::solvePnP (SOLVEPNP_ITERATIVE) and its supporting routines:
//! Rodrigues, projectPoints with Jacobians, undistortPoints, and the planar /
//! DLT pose initialisation. Mirrors findExtrinsicCameraParams2 in
//! modules/geometry/src/calibration_base.cpp (OpenCV 5.0.0).
//!
//! Structure: normalise the image points, build an initial pose (homography
//! decomposition when the object is planar, DLT otherwise), then refine with
//! Levenberg-Marquardt against the reprojection error.
//!
//! The initialisation only has to land in the right basin — the refinement runs
//! against the original, still-distorted image points through `project_points`,
//! so any initial guess that converges reaches the same optimum. That is why the
//! homography init below is a plain DLT where OpenCV additionally LM-refines it,
//! and why our LM is a textbook damped Gauss-Newton where OpenCV 5 uses its
//! geodesic-accelerated LevMarq. Both settle on the same minimum; only the path
//! there differs.

use nalgebra::{DMatrix, Matrix3, SMatrix, SVector, Vector3};

pub type Pt3d = (f64, f64, f64);
pub type Pt2d = (f64, f64);

/// Pinhole intrinsics (the 3x3 camera matrix, row-major).
#[derive(Clone, Copy)]
pub struct Camera {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

impl Camera {
    pub fn from_matrix(m: &[f64; 9]) -> Camera {
        Camera {
            fx: m[0],
            fy: m[4],
            cx: m[2],
            cy: m[5],
        }
    }
}

/// Distortion coefficients, laid out as OpenCV's `k[14]`:
/// [k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, taux, tauy].
/// Accepts 0, 4, 5, 8, 12 or 14 coefficients; the thin-prism (s1..s4) terms are
/// supported, the tilt (taux/tauy) terms are not.
#[derive(Clone, Copy, Default)]
pub struct Distortion {
    pub k: [f64; 14],
}

impl Distortion {
    pub fn from_slice(c: &[f64]) -> Result<Distortion, String> {
        if !matches!(c.len(), 0 | 4 | 5 | 8 | 12 | 14) {
            return Err(format!(
                "distortion must have 0, 4, 5, 8, 12 or 14 coefficients, got {}",
                c.len()
            ));
        }
        if c.len() == 14 && (c[12] != 0.0 || c[13] != 0.0) {
            return Err("tilted-sensor distortion (taux/tauy) is not supported".into());
        }
        let mut k = [0f64; 14];
        k[..c.len()].copy_from_slice(c);
        Ok(Distortion { k })
    }
}

/// Rodrigues, vector -> matrix. Returns the row-major 3x3 R and, optionally, the
/// 3x9 dR/dr Jacobian laid out as OpenCV's `dRdr[27]`.
pub fn rodrigues_v2m(r: [f64; 3], want_jac: bool) -> ([f64; 9], Option<[f64; 27]>) {
    let theta = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
    if theta < f64::EPSILON {
        let eye = [1., 0., 0., 0., 1., 0., 0., 0., 1.];
        if !want_jac {
            return (eye, None);
        }
        let mut j = [0f64; 27];
        j[5] = -1.;
        j[15] = -1.;
        j[19] = -1.;
        j[7] = 1.;
        j[11] = 1.;
        j[21] = 1.;
        return (eye, Some(j));
    }

    let c = theta.cos();
    let s = theta.sin();
    let c1 = 1.0 - c;
    let itheta = 1.0 / theta;
    let (rx, ry, rz) = (r[0] * itheta, r[1] * itheta, r[2] * itheta);

    let rrt = [
        rx * rx, rx * ry, rx * rz, rx * ry, ry * ry, ry * rz, rx * rz, ry * rz, rz * rz,
    ];
    let r_x = [0., -rz, ry, rz, 0., -rx, -ry, rx, 0.];
    let eye = [1., 0., 0., 0., 1., 0., 0., 0., 1.];

    // R = cos(t)*I + (1 - cos(t))*r*r^T + sin(t)*[r]_x
    let mut rot = [0f64; 9];
    for i in 0..9 {
        rot[i] = c * eye[i] + c1 * rrt[i] + s * r_x[i];
    }
    if !want_jac {
        return (rot, None);
    }

    let drrt = [
        rx + rx, ry, rz, ry, 0., 0., rz, 0., 0., 0., rx, 0., rx, ry + ry, rz, 0., rz, 0., 0., 0.,
        rx, 0., 0., ry, rx, ry, rz + rz,
    ];
    let d_r_x = [
        0., 0., 0., 0., 0., -1., 0., 1., 0., 0., 0., 1., 0., 0., 0., -1., 0., 0., 0., -1., 0., 1.,
        0., 0., 0., 0., 0.,
    ];
    let mut j = [0f64; 27];
    for i in 0..3 {
        let ri = [rx, ry, rz][i];
        let a0 = -s * ri;
        let a1 = (s - 2.0 * c1 * itheta) * ri;
        let a2 = c1 * itheta;
        let a3 = (c - s * itheta) * ri;
        let a4 = s * itheta;
        for k in 0..9 {
            j[i * 9 + k] =
                a0 * eye[k] + a1 * rrt[k] + a2 * drrt[i * 9 + k] + a3 * r_x[k] + a4 * d_r_x[i * 9 + k];
        }
    }
    (rot, Some(j))
}

/// Rodrigues, matrix -> vector. Re-orthogonalises via SVD first, as OpenCV does.
pub fn rodrigues_m2v(rot: &[f64; 9]) -> [f64; 3] {
    let m = Matrix3::new(
        rot[0], rot[1], rot[2], rot[3], rot[4], rot[5], rot[6], rot[7], rot[8],
    );
    let svd = m.svd(true, true);
    let m = svd.u.unwrap() * svd.v_t.unwrap();
    let at = |i: usize, j: usize| m[(i, j)];

    let mut r = [
        at(2, 1) - at(1, 2),
        at(0, 2) - at(2, 0),
        at(1, 0) - at(0, 1),
    ];
    let s = ((r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) * 0.25).sqrt();
    let c = ((at(0, 0) + at(1, 1) + at(2, 2)) - 1.0) * 0.5;
    let c = c.clamp(-1.0, 1.0);
    let mut theta = c.acos();

    if s < 1e-5 {
        if c > 0.0 {
            return [0.0, 0.0, 0.0];
        }
        // theta ~ pi: recover the axis from the diagonal, where the
        // antisymmetric part has vanished.
        let t = (at(0, 0) + 1.0) * 0.5;
        r[0] = t.max(0.0).sqrt();
        let t = (at(1, 1) + 1.0) * 0.5;
        r[1] = t.max(0.0).sqrt() * if at(0, 1) < 0.0 { -1.0 } else { 1.0 };
        let t = (at(2, 2) + 1.0) * 0.5;
        r[2] = t.max(0.0).sqrt() * if at(0, 2) < 0.0 { -1.0 } else { 1.0 };
        if r[0].abs() < r[1].abs()
            && r[0].abs() < r[2].abs()
            && (at(1, 2) > 0.0) != (r[1] * r[2] > 0.0)
        {
            r[2] = -r[2];
        }
        let n = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        theta /= n;
        return [r[0] * theta, r[1] * theta, r[2] * theta];
    }

    let vth = 1.0 / (2.0 * s) * theta;
    [r[0] * vth, r[1] * vth, r[2] * vth]
}

/// Project object points into the image, optionally with the dp/drvec and
/// dp/dtvec Jacobians (each 2N x 3, row-major per point: [dx/d0..2; dy/d0..2]).
#[allow(clippy::type_complexity)]
pub fn project_points(
    obj: &[Pt3d],
    rvec: [f64; 3],
    tvec: [f64; 3],
    cam: &Camera,
    dist: &Distortion,
    want_jac: bool,
) -> (Vec<Pt2d>, Option<(Vec<[f64; 3]>, Vec<[f64; 3]>)>) {
    let (rot, drdr) = rodrigues_v2m(rvec, want_jac);
    let k = &dist.k;
    let (fx, fy, cx, cy) = (cam.fx, cam.fy, cam.cx, cam.cy);

    let mut out = Vec::with_capacity(obj.len());
    let mut jr: Vec<[f64; 3]> = Vec::new();
    let mut jt: Vec<[f64; 3]> = Vec::new();
    if want_jac {
        jr.reserve(obj.len() * 2);
        jt.reserve(obj.len() * 2);
    }

    for &(bx, by, bz) in obj {
        let mut x = rot[0] * bx + rot[1] * by + rot[2] * bz + tvec[0];
        let mut y = rot[3] * bx + rot[4] * by + rot[5] * bz + tvec[1];
        let z0 = rot[6] * bx + rot[7] * by + rot[8] * bz + tvec[2];
        let z = if z0 != 0.0 { 1.0 / z0 } else { 1.0 };
        x *= z;
        y *= z;

        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        let a1 = 2.0 * x * y;
        let a2 = r2 + 2.0 * x * x;
        let a3 = r2 + 2.0 * y * y;
        let cdist = 1.0 + k[0] * r2 + k[1] * r4 + k[4] * r6;
        let icdist2 = 1.0 / (1.0 + k[5] * r2 + k[6] * r4 + k[7] * r6);
        // radial + tangential + thin-prism
        let xd = x * cdist * icdist2 + k[2] * a1 + k[3] * a2 + k[8] * r2 + k[9] * r4;
        let yd = y * cdist * icdist2 + k[2] * a3 + k[3] * a1 + k[10] * r2 + k[11] * r4;
        out.push((xd * fx + cx, yd * fy + cy));

        if !want_jac {
            continue;
        }
        let drdr = drdr.as_ref().unwrap();

        // d/dt
        let dxdt = [z, 0.0, -x * z];
        let dydt = [0.0, z, -y * z];
        let mut rowx = [0f64; 3];
        let mut rowy = [0f64; 3];
        for j in 0..3 {
            let dr2dt = 2.0 * x * dxdt[j] + 2.0 * y * dydt[j];
            let dcdist_dt = k[0] * dr2dt + 2.0 * k[1] * r2 * dr2dt + 3.0 * k[4] * r4 * dr2dt;
            let dicdist2_dt =
                -icdist2 * icdist2 * (k[5] * dr2dt + 2.0 * k[6] * r2 * dr2dt + 3.0 * k[7] * r4 * dr2dt);
            let da1dt = 2.0 * (x * dydt[j] + y * dxdt[j]);
            let dmxdt = dxdt[j] * cdist * icdist2
                + x * dcdist_dt * icdist2
                + x * cdist * dicdist2_dt
                + k[2] * da1dt
                + k[3] * (dr2dt + 4.0 * x * dxdt[j])
                + k[8] * dr2dt
                + 2.0 * r2 * k[9] * dr2dt;
            let dmydt = dydt[j] * cdist * icdist2
                + y * dcdist_dt * icdist2
                + y * cdist * dicdist2_dt
                + k[2] * (dr2dt + 4.0 * y * dydt[j])
                + k[3] * da1dt
                + k[10] * dr2dt
                + 2.0 * r2 * k[11] * dr2dt;
            rowx[j] = fx * dmxdt;
            rowy[j] = fy * dmydt;
        }
        jt.push(rowx);
        jt.push(rowy);

        // d/dr, chained through dR/dr
        let dx0dr = [
            bx * drdr[0] + by * drdr[1] + bz * drdr[2],
            bx * drdr[9] + by * drdr[10] + bz * drdr[11],
            bx * drdr[18] + by * drdr[19] + bz * drdr[20],
        ];
        let dy0dr = [
            bx * drdr[3] + by * drdr[4] + bz * drdr[5],
            bx * drdr[12] + by * drdr[13] + bz * drdr[14],
            bx * drdr[21] + by * drdr[22] + bz * drdr[23],
        ];
        let dz0dr = [
            bx * drdr[6] + by * drdr[7] + bz * drdr[8],
            bx * drdr[15] + by * drdr[16] + bz * drdr[17],
            bx * drdr[24] + by * drdr[25] + bz * drdr[26],
        ];
        let mut rowx = [0f64; 3];
        let mut rowy = [0f64; 3];
        for j in 0..3 {
            let dxdr = z * (dx0dr[j] - x * dz0dr[j]);
            let dydr = z * (dy0dr[j] - y * dz0dr[j]);
            let dr2dr = 2.0 * x * dxdr + 2.0 * y * dydr;
            let dcdist_dr = (k[0] + 2.0 * k[1] * r2 + 3.0 * k[4] * r4) * dr2dr;
            let dicdist2_dr = -icdist2 * icdist2 * (k[5] + 2.0 * k[6] * r2 + 3.0 * k[7] * r4) * dr2dr;
            let da1dr = 2.0 * (x * dydr + y * dxdr);
            let dmxdr = dxdr * cdist * icdist2
                + x * dcdist_dr * icdist2
                + x * cdist * dicdist2_dr
                + k[2] * da1dr
                + k[3] * (dr2dr + 4.0 * x * dxdr)
                + (k[8] + 2.0 * r2 * k[9]) * dr2dr;
            let dmydr = dydr * cdist * icdist2
                + y * dcdist_dr * icdist2
                + y * cdist * dicdist2_dr
                + k[2] * (dr2dr + 4.0 * y * dydr)
                + k[3] * da1dr
                + (k[10] + 2.0 * r2 * k[11]) * dr2dr;
            rowx[j] = fx * dmxdr;
            rowy[j] = fy * dmydr;
        }
        jr.push(rowx);
        jr.push(rowy);
    }

    if want_jac {
        (out, Some((jr, jt)))
    } else {
        (out, None)
    }
}

/// cv::undistortPoints: map pixel coords back to normalised camera coords,
/// iteratively inverting the distortion model.
pub fn undistort_points(pts: &[Pt2d], cam: &Camera, dist: &Distortion) -> Vec<Pt2d> {
    let k = &dist.k;
    let has_dist = k.iter().any(|&v| v != 0.0);
    pts.iter()
        .map(|&(u, v)| {
            let x0 = (u - cam.cx) / cam.fx;
            let y0 = (v - cam.cy) / cam.fy;
            if !has_dist {
                return (x0, y0);
            }
            let (mut x, mut y) = (x0, y0);
            // Fixed-point iteration; OpenCV uses 5 passes by default.
            for _ in 0..5 {
                let r2 = x * x + y * y;
                let icdist = (1.0 + ((k[7] * r2 + k[6]) * r2 + k[5]) * r2)
                    / (1.0 + ((k[4] * r2 + k[1]) * r2 + k[0]) * r2);
                if icdist < 0.0 {
                    return (x0, y0);
                }
                let dx = 2.0 * k[2] * x * y + k[3] * (r2 + 2.0 * x * x) + k[8] * r2 + k[9] * r2 * r2;
                let dy = k[2] * (r2 + 2.0 * y * y) + 2.0 * k[3] * x * y + k[10] * r2 + k[11] * r2 * r2;
                x = (x0 - dx) * icdist;
                y = (y0 - dy) * icdist;
            }
            (x, y)
        })
        .collect()
}

/// Least-squares DLT homography src -> dst, via the null space of the 2N x 9
/// design matrix. Points are Hartley-normalised first for conditioning.
fn find_homography_dlt(src: &[Pt2d], dst: &[Pt2d]) -> Option<[f64; 9]> {
    let n = src.len();
    if n < 4 {
        return None;
    }
    // Hartley normalisation: centre each set and scale to mean distance sqrt(2).
    let norm = |p: &[Pt2d]| -> ([f64; 9], Vec<Pt2d>) {
        let (mut mx, mut my) = (0.0, 0.0);
        for &(x, y) in p {
            mx += x;
            my += y;
        }
        mx /= n as f64;
        my /= n as f64;
        let mut d = 0.0;
        for &(x, y) in p {
            d += ((x - mx).powi(2) + (y - my).powi(2)).sqrt();
        }
        d /= n as f64;
        let s = if d > f64::EPSILON { 2f64.sqrt() / d } else { 1.0 };
        let t = [s, 0.0, -s * mx, 0.0, s, -s * my, 0.0, 0.0, 1.0];
        let q = p.iter().map(|&(x, y)| (s * (x - mx), s * (y - my))).collect();
        (t, q)
    };
    let (t1, sn) = norm(src);
    let (t2, dn) = norm(dst);

    let mut a = DMatrix::<f64>::zeros(2 * n, 9);
    for i in 0..n {
        let (x, y) = sn[i];
        let (u, v) = dn[i];
        a[(2 * i, 0)] = -x;
        a[(2 * i, 1)] = -y;
        a[(2 * i, 2)] = -1.0;
        a[(2 * i, 6)] = u * x;
        a[(2 * i, 7)] = u * y;
        a[(2 * i, 8)] = u;
        a[(2 * i + 1, 3)] = -x;
        a[(2 * i + 1, 4)] = -y;
        a[(2 * i + 1, 5)] = -1.0;
        a[(2 * i + 1, 6)] = v * x;
        a[(2 * i + 1, 7)] = v * y;
        a[(2 * i + 1, 8)] = v;
    }
    // With exactly four correspondences A is 8x9. nalgebra returns a thin
    // 8x9 V^T for that shape, which does not contain the one-dimensional
    // nullspace row. A^T A is always 9x9 and exposes that final singular vector.
    let ata = a.transpose() * &a;
    let svd = ata.svd(false, true);
    let vt = svd.v_t?;
    let h: Vec<f64> = (0..9).map(|i| vt[(8, i)]).collect();

    // Undo the normalisation: H = T2^-1 * Hn * T1
    let m = |v: &[f64]| Matrix3::new(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
    let hn = m(&h);
    let t1m = m(&t1);
    let t2m = m(&t2);
    let t2inv = t2m.try_inverse()?;
    let hh = t2inv * hn * t1m;
    let s = hh[(2, 2)];
    if s.abs() < f64::EPSILON {
        return None;
    }
    let mut out = [0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            out[i * 3 + j] = hh[(i, j)] / s;
        }
    }
    Some(out)
}

/// Initial pose for a planar object, by decomposing the plane->image homography.
fn init_planar(obj: &[Pt3d], normalised: &[Pt2d], vt: &Matrix3<f64>, mc: [f64; 3]) -> Option<([f64; 9], [f64; 3])> {
    let mut r_transform = *vt;
    if vt[(0, 2)] * vt[(0, 2)] + vt[(1, 2)] * vt[(1, 2)] < 1e-10 {
        r_transform = Matrix3::identity();
    }
    if r_transform.determinant() < 0.0 {
        r_transform = -r_transform;
    }
    // T = -R * Mc  (move the plane's centroid to the origin)
    let mcv = Vector3::new(mc[0], mc[1], mc[2]);
    let tt = -(r_transform * mcv);

    // Express the object points in the plane's own 2D frame.
    let mxy: Vec<Pt2d> = obj
        .iter()
        .map(|&(x, y, z)| {
            (
                r_transform[(0, 0)] * x + r_transform[(0, 1)] * y + r_transform[(0, 2)] * z + tt[0],
                r_transform[(1, 0)] * x + r_transform[(1, 1)] * y + r_transform[(1, 2)] * z + tt[1],
            )
        })
        .collect();

    let h = find_homography_dlt(&mxy, normalised)?;
    if !h.iter().all(|v| v.is_finite()) {
        return None;
    }

    // Columns of H are [r1*s, r2*s, t*s]; normalise to recover the rotation.
    let h1_norm = (h[0] * h[0] + h[3] * h[3] + h[6] * h[6]).sqrt();
    let h2_norm = (h[1] * h[1] + h[4] * h[4] + h[7] * h[7]).sqrt();
    let s1 = 1.0 / h1_norm.max(f64::EPSILON);
    let s2 = 1.0 / h2_norm.max(f64::EPSILON);
    let ts = 2.0 / (h1_norm + h2_norm).max(f64::EPSILON);

    let c1 = [h[0] * s1, h[3] * s1, h[6] * s1];
    let c2 = [h[1] * s2, h[4] * s2, h[7] * s2];
    let t = [h[2] * ts, h[5] * ts, h[8] * ts];
    // third column = c1 x c2, so the frame is right-handed
    let c3 = [
        c1[1] * c2[2] - c1[2] * c2[1],
        c1[2] * c2[0] - c1[0] * c2[2],
        c1[0] * c2[1] - c1[1] * c2[0],
    ];
    // H now holds an approximate rotation; round-trip through Rodrigues to make
    // it exactly orthonormal, exactly as OpenCV does.
    let approx = [c1[0], c2[0], c3[0], c1[1], c2[1], c3[1], c1[2], c2[2], c3[2]];
    let rv = rodrigues_m2v(&approx);
    let (rot, _) = rodrigues_v2m(rv, false);

    // Compose with the plane transform: R = Rh * R_transform, t = th + Rh * T
    let rm = Matrix3::new(
        rot[0], rot[1], rot[2], rot[3], rot[4], rot[5], rot[6], rot[7], rot[8],
    );
    let tv = Vector3::new(t[0], t[1], t[2]) + rm * tt;
    let rfin = rm * r_transform;
    let mut ro = [0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            ro[i * 3 + j] = rfin[(i, j)];
        }
    }
    Some((ro, [tv[0], tv[1], tv[2]]))
}

/// Initial pose for a non-planar object, by DLT on the 3x4 projection matrix.
fn init_dlt(obj: &[Pt3d], normalised: &[Pt2d]) -> Option<([f64; 9], [f64; 3])> {
    let n = obj.len();
    if n < 6 {
        return None;
    }
    let mut l = DMatrix::<f64>::zeros(2 * n, 12);
    for i in 0..n {
        let (bx, by, bz) = obj[i];
        let (x, y) = (-normalised[i].0, -normalised[i].1);
        let r0 = 2 * i;
        let r1 = 2 * i + 1;
        l[(r0, 0)] = bx;
        l[(r0, 1)] = by;
        l[(r0, 2)] = bz;
        l[(r0, 3)] = 1.0;
        l[(r0, 8)] = x * bx;
        l[(r0, 9)] = x * by;
        l[(r0, 10)] = x * bz;
        l[(r0, 11)] = x;
        l[(r1, 4)] = bx;
        l[(r1, 5)] = by;
        l[(r1, 6)] = bz;
        l[(r1, 7)] = 1.0;
        l[(r1, 8)] = y * bx;
        l[(r1, 9)] = y * by;
        l[(r1, 10)] = y * bz;
        l[(r1, 11)] = y;
    }
    let ll = l.transpose() * &l;
    let svd = ll.svd(false, true);
    let vt = svd.v_t?;
    let mut rrt = [0f64; 12];
    for i in 0..12 {
        rrt[i] = vt[(11, i)];
    }
    let mut rr = Matrix3::new(
        rrt[0], rrt[1], rrt[2], rrt[4], rrt[5], rrt[6], rrt[8], rrt[9], rrt[10],
    );
    let mut tt = [rrt[3], rrt[7], rrt[11]];
    if rr.determinant() < 0.0 {
        rr = -rr;
        tt = [-tt[0], -tt[1], -tt[2]];
    }
    let sc = rr.norm();
    if sc.abs() < f64::EPSILON {
        return None;
    }
    // Nearest orthonormal matrix to the DLT block.
    let svd = rr.svd(true, true);
    let rm = svd.u? * svd.v_t?;
    let scale = rm.norm() / sc;
    let mut ro = [0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            ro[i * 3 + j] = rm[(i, j)];
        }
    }
    Some((ro, [tt[0] * scale, tt[1] * scale, tt[2] * scale]))
}

/// Reprojection residual and Jacobian for the LM refinement.
fn residual(
    obj: &[Pt3d],
    img: &[Pt2d],
    param: &[f64; 6],
    cam: &Camera,
    dist: &Distortion,
    want_jac: bool,
) -> (Vec<f64>, Option<DMatrix<f64>>) {
    let rvec = [param[0], param[1], param[2]];
    let tvec = [param[3], param[4], param[5]];
    let (proj, jac) = project_points(obj, rvec, tvec, cam, dist, want_jac);
    let mut err = Vec::with_capacity(obj.len() * 2);
    for i in 0..obj.len() {
        err.push(proj[i].0 - img[i].0);
        err.push(proj[i].1 - img[i].1);
    }
    let j = jac.map(|(jr, jt)| {
        let mut m = DMatrix::<f64>::zeros(obj.len() * 2, 6);
        for row in 0..obj.len() * 2 {
            for c in 0..3 {
                m[(row, c)] = jr[row][c];
                m[(row, c + 3)] = jt[row][c];
            }
        }
        m
    });
    (err, j)
}

/// Levenberg-Marquardt refinement of the 6-DOF pose.
fn refine(
    obj: &[Pt3d],
    img: &[Pt2d],
    param: &mut [f64; 6],
    cam: &Camera,
    dist: &Distortion,
    max_iter: usize,
) {
    let sq = |e: &[f64]| e.iter().map(|v| v * v).sum::<f64>();
    let (mut err, _) = residual(obj, img, param, cam, dist, false);
    let mut cost = sq(&err);
    let mut lambda = 1e-3;

    for _ in 0..max_iter {
        let (e, j) = residual(obj, img, param, cam, dist, true);
        let j = j.unwrap();
        let ev = DMatrix::from_column_slice(e.len(), 1, &e);
        let jt = j.transpose();
        let h = &jt * &j; // 6x6 Gauss-Newton approximation
        let g = &jt * &ev;

        let mut improved = false;
        for _ in 0..10 {
            // Damp the diagonal; large lambda degrades to gradient descent.
            let mut hd = h.clone();
            for i in 0..6 {
                hd[(i, i)] += lambda * hd[(i, i)].max(1e-12);
            }
            let hs: SMatrix<f64, 6, 6> = SMatrix::from_iterator(hd.iter().copied());
            let gs: SVector<f64, 6> = SVector::from_iterator(g.iter().copied());
            let Some(step) = hs.lu().solve(&gs) else {
                lambda *= 10.0;
                continue;
            };
            let mut cand = *param;
            for i in 0..6 {
                cand[i] -= step[i];
            }
            let (ce, _) = residual(obj, img, &cand, cam, dist, false);
            let ccost = sq(&ce);
            if ccost < cost {
                *param = cand;
                cost = ccost;
                err = ce;
                lambda = (lambda * 0.1).max(1e-12);
                improved = true;
                break;
            }
            lambda *= 10.0;
            if lambda > 1e12 {
                break;
            }
        }
        if !improved {
            break;
        }
        // Converged: residual can no longer move meaningfully.
        if sq(&err) < 1e-24 {
            break;
        }
    }
}

/// Solve for the pose (rvec, tvec) mapping object points into the camera frame.
/// Mirrors cv::solvePnP with SOLVEPNP_ITERATIVE.
///
/// `guess` supplies an initial (rvec, tvec) — the equivalent of
/// `useExtrinsicGuess`. Without one, at least 4 points are required (6 if the
/// object is not planar).
pub fn solve_pnp(
    obj: &[Pt3d],
    img: &[Pt2d],
    cam: &Camera,
    dist: &Distortion,
    guess: Option<([f64; 3], [f64; 3])>,
) -> Option<([f64; 3], [f64; 3])> {
    solve_pnp_with_refinement(obj, img, cam, dist, guess, 20)
}

fn solve_pnp_with_refinement(
    obj: &[Pt3d],
    img: &[Pt2d],
    cam: &Camera,
    dist: &Distortion,
    guess: Option<([f64; 3], [f64; 3])>,
    refinement_iterations: usize,
) -> Option<([f64; 3], [f64; 3])> {
    if obj.len() != img.len() || obj.len() < 4 {
        return None;
    }
    let mut param = [0f64; 6];

    if let Some((r, t)) = guess {
        param[..3].copy_from_slice(&r);
        param[3..].copy_from_slice(&t);
    } else {
        // Normalised (undistorted) image points drive the initialisation only.
        let normalised = undistort_points(img, cam, dist);

        // Is the object planar? Look at the spread of the points about their
        // centroid: a vanishing third singular value means they lie in a plane.
        let n = obj.len() as f64;
        let mc = [
            obj.iter().map(|p| p.0).sum::<f64>() / n,
            obj.iter().map(|p| p.1).sum::<f64>() / n,
            obj.iter().map(|p| p.2).sum::<f64>() / n,
        ];
        let mut mm = Matrix3::zeros();
        for &(x, y, z) in obj {
            let d = Vector3::new(x - mc[0], y - mc[1], z - mc[2]);
            mm += d * d.transpose();
        }
        let svd = mm.svd(false, true);
        let w = svd.singular_values;
        let vt = svd.v_t?;

        let (rot, t) = if w[1] > f64::EPSILON && w[2] / w[1] < 1e-3 {
            init_planar(obj, &normalised, &vt, mc)?
        } else {
            init_dlt(obj, &normalised)?
        };
        let rv = rodrigues_m2v(&rot);
        param[..3].copy_from_slice(&rv);
        param[3..].copy_from_slice(&t);
    }

    refine(obj, img, &mut param, cam, dist, refinement_iterations);
    if !param.iter().all(|v| v.is_finite()) {
        return None;
    }
    Some((
        [param[0], param[1], param[2]],
        [param[3], param[4], param[5]],
    ))
}

/// Result of robust pose estimation with [`solve_pnp_ransac`].
#[derive(Clone, Debug)]
pub struct RansacPose {
    pub rvec: [f64; 3],
    pub tvec: [f64; 3],
    pub inliers: Vec<usize>,
    pub reprojection_rmse: f64,
}

/// Small deterministic PRNG used for RANSAC sampling.
///
/// Keeping this local avoids adding a runtime dependency and makes a supplied
/// seed reproduce exactly the same hypotheses on every platform.
struct RansacRng {
    state: u64,
}

impl RansacRng {
    fn new(seed: u64) -> Self {
        // xorshift64* cannot use the all-zero state.
        Self {
            state: if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn sample(&mut self, population: usize, count: usize) -> [usize; 6] {
        debug_assert!(count <= 6 && count <= population);
        let mut indices = [0usize; 6];
        for i in 0..count {
            loop {
                let candidate = self.next_u64() as usize % population;
                if !indices[..i].contains(&candidate) {
                    indices[i] = candidate;
                    break;
                }
            }
        }
        indices
    }
}

fn object_points_are_planar(obj: &[Pt3d]) -> bool {
    if obj.len() < 3 {
        return false;
    }
    let n = obj.len() as f64;
    let centroid = Vector3::new(
        obj.iter().map(|p| p.0).sum::<f64>() / n,
        obj.iter().map(|p| p.1).sum::<f64>() / n,
        obj.iter().map(|p| p.2).sum::<f64>() / n,
    );
    let mut scatter = Matrix3::zeros();
    for &(x, y, z) in obj {
        let d = Vector3::new(x, y, z) - centroid;
        scatter += d * d.transpose();
    }
    let singular = scatter.svd(false, false).singular_values;
    singular[1] > f64::EPSILON && singular[2] / singular[1] < 1e-3
}

fn classify_inliers(
    obj: &[Pt3d],
    img: &[Pt2d],
    rvec: [f64; 3],
    tvec: [f64; 3],
    cam: &Camera,
    dist: &Distortion,
    threshold_sq: f64,
    inliers: &mut Vec<usize>,
) -> f64 {
    let (projected, _) = project_points(obj, rvec, tvec, cam, dist, false);
    inliers.clear();
    let mut squared_error = 0.0;
    for (i, (&observed, &predicted)) in img.iter().zip(projected.iter()).enumerate() {
        let dx = observed.0 - predicted.0;
        let dy = observed.1 - predicted.1;
        let err_sq = dx * dx + dy * dy;
        if err_sq.is_finite() && err_sq <= threshold_sq {
            inliers.push(i);
            squared_error += err_sq;
        }
    }
    squared_error
}

fn points_in_front(obj: &[Pt3d], rvec: [f64; 3], tvec: [f64; 3]) -> bool {
    let (rot, _) = rodrigues_v2m(rvec, false);
    obj.iter().all(|&(x, y, z)| {
        let camera_z = rot[6] * x + rot[7] * y + rot[8] * z + tvec[2];
        camera_z.is_finite() && camera_z > f64::EPSILON
    })
}

/// Robustly estimate pose while rejecting bad 3D-2D correspondences.
///
/// This first-stage RANSAC implementation uses the existing iterative PnP
/// solver for each minimal hypothesis: five points for planar data and six for
/// general 3D data. Hypotheses are scored in distorted pixel coordinates, the
/// iteration limit is reduced adaptively as the inlier ratio improves, and the
/// winning pose is refined over its full consensus set.
///
/// This is intended for boards, multiple known markers, or feature matches. A
/// single four-corner marker has no redundant correspondence to reject and is
/// therefore deliberately not accepted here.
#[allow(clippy::too_many_arguments)]
pub fn solve_pnp_ransac(
    obj: &[Pt3d],
    img: &[Pt2d],
    cam: &Camera,
    dist: &Distortion,
    max_iterations: usize,
    reprojection_error: f64,
    confidence: f64,
    seed: u64,
) -> Option<RansacPose> {
    if obj.len() != img.len()
        || max_iterations == 0
        || !reprojection_error.is_finite()
        || reprojection_error <= 0.0
        || !confidence.is_finite()
        || confidence <= 0.0
        || confidence >= 1.0
        || obj
            .iter()
            .any(|p| !p.0.is_finite() || !p.1.is_finite() || !p.2.is_finite())
        || img.iter().any(|p| !p.0.is_finite() || !p.1.is_finite())
    {
        return None;
    }

    let sample_size = if object_points_are_planar(obj) { 5 } else { 6 };
    if obj.len() < sample_size {
        return None;
    }

    let threshold_sq = reprojection_error * reprojection_error;
    let mut rng = RansacRng::new(seed);
    let mut iteration_limit = max_iterations;
    let mut iteration = 0usize;
    let mut best_pose = None;
    let mut best_inliers = Vec::new();
    let mut candidate_inliers = Vec::with_capacity(obj.len());
    let mut best_error = f64::INFINITY;

    while iteration < iteration_limit {
        iteration += 1;
        let sample = rng.sample(obj.len(), sample_size);
        let mut sample_obj = [(0.0, 0.0, 0.0); 6];
        let mut sample_img = [(0.0, 0.0); 6];
        for i in 0..sample_size {
            sample_obj[i] = obj[sample[i]];
            sample_img[i] = img[sample[i]];
        }
        let sample_obj = &sample_obj[..sample_size];
        let sample_img = &sample_img[..sample_size];

        // A hypothesis only needs to be accurate enough for consensus scoring;
        // the winning pose receives the normal full refinement below.
        let Some((rvec, tvec)) =
            solve_pnp_with_refinement(sample_obj, sample_img, cam, dist, None, 6)
        else {
            continue;
        };
        if !points_in_front(sample_obj, rvec, tvec) {
            continue;
        }

        let error = classify_inliers(
            obj,
            img,
            rvec,
            tvec,
            cam,
            dist,
            threshold_sq,
            &mut candidate_inliers,
        );
        if candidate_inliers.len() < sample_size
            || candidate_inliers.len() < best_inliers.len()
            || (candidate_inliers.len() == best_inliers.len() && error >= best_error)
        {
            continue;
        }

        best_pose = Some((rvec, tvec));
        best_error = error;
        std::mem::swap(&mut best_inliers, &mut candidate_inliers);

        // N = log(1-confidence) / log(1-inlier_ratio^sample_size).
        // Clamp the probability away from 0 and 1 to keep the logarithms sane.
        let inlier_ratio = best_inliers.len() as f64 / obj.len() as f64;
        let all_inlier_sample = inlier_ratio.powi(sample_size as i32);
        if all_inlier_sample >= 1.0 - f64::EPSILON {
            iteration_limit = iteration;
        } else if all_inlier_sample > f64::EPSILON {
            let needed = ((1.0 - confidence).ln() / (1.0 - all_inlier_sample).ln())
                .ceil()
                .max(1.0) as usize;
            iteration_limit = iteration_limit.min(needed.max(iteration));
        }
    }

    let (mut rvec, mut tvec) = best_pose?;
    let mut consensus = best_inliers;

    // Local optimisation: refit and reclassify twice. The second pass matters
    // when the first all-inlier fit pulls a few borderline points into consensus.
    for _ in 0..2 {
        if consensus.len() < sample_size {
            break;
        }
        let inlier_obj: Vec<Pt3d> = consensus.iter().map(|&i| obj[i]).collect();
        let inlier_img: Vec<Pt2d> = consensus.iter().map(|&i| img[i]).collect();
        let Some((new_rvec, new_tvec)) =
            solve_pnp(&inlier_obj, &inlier_img, cam, dist, Some((rvec, tvec)))
        else {
            break;
        };
        if !points_in_front(&inlier_obj, new_rvec, new_tvec) {
            break;
        }
        rvec = new_rvec;
        tvec = new_tvec;
        let mut new_consensus = Vec::with_capacity(obj.len());
        classify_inliers(
            obj,
            img,
            rvec,
            tvec,
            cam,
            dist,
            threshold_sq,
            &mut new_consensus,
        );
        if new_consensus == consensus {
            break;
        }
        consensus = new_consensus;
    }

    let mut final_inliers = Vec::with_capacity(obj.len());
    let final_squared_error = classify_inliers(
        obj,
        img,
        rvec,
        tvec,
        cam,
        dist,
        threshold_sq,
        &mut final_inliers,
    );
    if final_inliers.len() < sample_size {
        return None;
    }
    let reprojection_rmse = (final_squared_error / final_inliers.len() as f64).sqrt();

    Some(RansacPose {
        rvec,
        tvec,
        inliers: final_inliers,
        reprojection_rmse,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera() -> Camera {
        Camera {
            fx: 800.0,
            fy: 810.0,
            cx: 320.0,
            cy: 240.0,
        }
    }

    fn distort() -> Distortion {
        Distortion::from_slice(&[-0.15, 0.04, 0.001, -0.0005, 0.01]).unwrap()
    }

    fn noisy_projection(obj: &[Pt3d], outliers: &[usize]) -> Vec<Pt2d> {
        let (mut img, _) = project_points(
            obj,
            [0.25, -0.12, 0.08],
            [0.015, -0.02, 0.7],
            &camera(),
            &distort(),
            false,
        );
        for (i, p) in img.iter_mut().enumerate() {
            let noise = ((i * 17 % 7) as f64 - 3.0) * 0.025;
            p.0 += noise;
            p.1 -= noise * 0.7;
        }
        for (j, &i) in outliers.iter().enumerate() {
            img[i] = (40.0 + j as f64 * 83.0, 700.0 - j as f64 * 61.0);
        }
        img
    }

    #[test]
    fn ransac_rejects_non_planar_outliers() {
        let obj: Vec<Pt3d> = (0..30)
            .map(|i| {
                let x = ((i * 37 % 101) as f64 / 100.0 - 0.5) * 0.16;
                let y = ((i * 53 % 97) as f64 / 96.0 - 0.5) * 0.12;
                let z = ((i * 29 % 89) as f64 / 88.0 - 0.5) * 0.05;
                (x, y, z)
            })
            .collect();
        let outliers = [1usize, 6, 11, 17, 22, 28];
        let img = noisy_projection(&obj, &outliers);
        let result = solve_pnp_ransac(&obj, &img, &camera(), &distort(), 300, 2.0, 0.99, 7)
            .expect("robust pose");

        assert_eq!(result.inliers.len(), obj.len() - outliers.len());
        assert!(outliers.iter().all(|i| !result.inliers.contains(i)));
        assert!(result.reprojection_rmse < 0.2);
        assert!((result.tvec[2] - 0.7).abs() < 1e-3);
    }

    #[test]
    fn ransac_rejects_planar_outliers_and_is_deterministic() {
        let obj: Vec<Pt3d> = (0..5)
            .flat_map(|y| {
                (0..6).map(move |x| {
                    (
                        (x as f64 - 2.5) * 0.025,
                        (y as f64 - 2.0) * 0.025,
                        0.0,
                    )
                })
            })
            .collect();
        let outliers = [0usize, 7, 14, 21, 29];
        let img = noisy_projection(&obj, &outliers);
        let a = solve_pnp_ransac(&obj, &img, &camera(), &distort(), 200, 2.0, 0.99, 123)
            .expect("first robust pose");
        let b = solve_pnp_ransac(&obj, &img, &camera(), &distort(), 200, 2.0, 0.99, 123)
            .expect("repeat robust pose");

        assert_eq!(a.inliers, b.inliers);
        assert_eq!(a.rvec, b.rvec);
        assert_eq!(a.tvec, b.tvec);
        assert_eq!(a.inliers.len(), obj.len() - outliers.len());
        assert!(outliers.iter().all(|i| !a.inliers.contains(i)));
        assert!(a.reprojection_rmse < 0.2);
    }

    #[test]
    fn ransac_rejects_invalid_parameters_and_four_point_input() {
        let obj = vec![(0.0, 0.0, 0.0); 4];
        let img = vec![(0.0, 0.0); 4];
        assert!(solve_pnp_ransac(&obj, &img, &camera(), &distort(), 100, 2.0, 0.99, 0).is_none());

        let obj = vec![(0.0, 0.0, 0.0); 6];
        let img = vec![(0.0, 0.0); 6];
        assert!(solve_pnp_ransac(&obj, &img, &camera(), &distort(), 0, 2.0, 0.99, 0).is_none());
        assert!(solve_pnp_ransac(&obj, &img, &camera(), &distort(), 10, 0.0, 0.99, 0).is_none());
        assert!(solve_pnp_ransac(&obj, &img, &camera(), &distort(), 10, 2.0, 1.0, 0).is_none());
    }
}
