//! Every signature of the stationary Gaussian process kernels in the Stan language, compared with
//! the kernel formula.
//!
//! Written by `tools/reference/gen_gp.py` from the CmdStan 2.40.0 signature dump: 28 Stan
//! signatures. `array[] real` points go through `gp_points_1d`, a second point set is the `_cross`
//! function, and one length scale per dimension is the `_ard` function. Each entry must equal
//! `sigma^2 k(d)`, with `d` the distance between the points after dividing each coordinate by its
//! length scale (the periodic kernel uses the plain distance).
#![allow(
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::needless_range_loop
)]

use stanmath_rs::prelude::*;

#[test]
fn gp_signatures() {
    // stan: gp_exp_quad_cov(array[] real, real, real)
    {
        let x1 = gp_points_1d(&[0.3, -0.8, 1.4]);
        let x2 = x1.clone();
        let got = gp_exp_quad_cov(&x1, 1.3, 0.7).unwrap();
        let scale = [0.7, 0.7];
        assert_eq!((got.nrows(), got.ncols()), (x1.len(), x2.len()));
        for i in 0..x1.len() {
            for j in 0..x2.len() {
                let d = x1[i]
                    .iter()
                    .zip(&x2[j])
                    .zip(&scale)
                    .map(|((a, b), s)| ((a - b) / s).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let want = 1.3 * 1.3 * (-d * d / 2.0).exp();
                assert!(
                    (got[(i, j)] - want).abs() <= 1e-12,
                    "gp_exp_quad_cov(array[] real, real, real) ({i}, {j}): {} vs {want}",
                    got[(i, j)]
                );
            }
        }
    }
    // stan: gp_exp_quad_cov(array[] real, array[] real, real, real)
    {
        let x1 = gp_points_1d(&[0.3, -0.8, 1.4]);
        let x2 = gp_points_1d(&[0.9, -0.1]);
        let got = gp_exp_quad_cov_cross(&x1, &x2, 1.3, 0.7).unwrap();
        let scale = [0.7, 0.7];
        assert_eq!((got.nrows(), got.ncols()), (x1.len(), x2.len()));
        for i in 0..x1.len() {
            for j in 0..x2.len() {
                let d = x1[i]
                    .iter()
                    .zip(&x2[j])
                    .zip(&scale)
                    .map(|((a, b), s)| ((a - b) / s).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let want = 1.3 * 1.3 * (-d * d / 2.0).exp();
                assert!(
                    (got[(i, j)] - want).abs() <= 1e-12,
                    "gp_exp_quad_cov(array[] real, array[] real, real, real) ({i}, {j}): {} vs {want}",
                    got[(i, j)]
                );
            }
        }
    }
    // stan: gp_exp_quad_cov(array[] vector, real, real)
    {
        let x1 = vec![vec![0.3, -0.8], vec![1.4, 0.2], vec![-0.5, 0.6]];
        let x2 = x1.clone();
        let got = gp_exp_quad_cov(&x1, 1.3, 0.7).unwrap();
        let scale = [0.7, 0.7];
        assert_eq!((got.nrows(), got.ncols()), (x1.len(), x2.len()));
        for i in 0..x1.len() {
            for j in 0..x2.len() {
                let d = x1[i]
                    .iter()
                    .zip(&x2[j])
                    .zip(&scale)
                    .map(|((a, b), s)| ((a - b) / s).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let want = 1.3 * 1.3 * (-d * d / 2.0).exp();
                assert!(
                    (got[(i, j)] - want).abs() <= 1e-12,
                    "gp_exp_quad_cov(array[] vector, real, real) ({i}, {j}): {} vs {want}",
                    got[(i, j)]
                );
            }
        }
    }
    // stan: gp_exp_quad_cov(array[] vector, real, array[] real)
    {
        let x1 = vec![vec![0.3, -0.8], vec![1.4, 0.2], vec![-0.5, 0.6]];
        let x2 = x1.clone();
        let l = [0.7, 1.6];
        let got = gp_exp_quad_cov_ard(&x1, 1.3, &l).unwrap();
        let scale = l;
        assert_eq!((got.nrows(), got.ncols()), (x1.len(), x2.len()));
        for i in 0..x1.len() {
            for j in 0..x2.len() {
                let d = x1[i]
                    .iter()
                    .zip(&x2[j])
                    .zip(&scale)
                    .map(|((a, b), s)| ((a - b) / s).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let want = 1.3 * 1.3 * (-d * d / 2.0).exp();
                assert!(
                    (got[(i, j)] - want).abs() <= 1e-12,
                    "gp_exp_quad_cov(array[] vector, real, array[] real) ({i}, {j}): {} vs {want}",
                    got[(i, j)]
                );
            }
        }
    }
    // stan: gp_exp_quad_cov(array[] vector, array[] vector, real, real)
    {
        let x1 = vec![vec![0.3, -0.8], vec![1.4, 0.2], vec![-0.5, 0.6]];
        let x2 = vec![vec![0.9, -0.1], vec![0.0, 1.1]];
        let got = gp_exp_quad_cov_cross(&x1, &x2, 1.3, 0.7).unwrap();
        let scale = [0.7, 0.7];
        assert_eq!((got.nrows(), got.ncols()), (x1.len(), x2.len()));
        for i in 0..x1.len() {
            for j in 0..x2.len() {
                let d = x1[i]
                    .iter()
                    .zip(&x2[j])
                    .zip(&scale)
                    .map(|((a, b), s)| ((a - b) / s).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let want = 1.3 * 1.3 * (-d * d / 2.0).exp();
                assert!(
                    (got[(i, j)] - want).abs() <= 1e-12,
                    "gp_exp_quad_cov(array[] vector, array[] vector, real, real) ({i}, {j}): {} vs {want}",
                    got[(i, j)]
                );
            }
        }
    }
    // stan: gp_exp_quad_cov(array[] vector, array[] vector, real, array[] real)
    {
        let x1 = vec![vec![0.3, -0.8], vec![1.4, 0.2], vec![-0.5, 0.6]];
        let x2 = vec![vec![0.9, -0.1], vec![0.0, 1.1]];
        let l = [0.7, 1.6];
        let got = gp_exp_quad_cov_ard_cross(&x1, &x2, 1.3, &l).unwrap();
        let scale = l;
        assert_eq!((got.nrows(), got.ncols()), (x1.len(), x2.len()));
        for i in 0..x1.len() {
            for j in 0..x2.len() {
                let d = x1[i]
                    .iter()
                    .zip(&x2[j])
                    .zip(&scale)
                    .map(|((a, b), s)| ((a - b) / s).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let want = 1.3 * 1.3 * (-d * d / 2.0).exp();
                assert!(
                    (got[(i, j)] - want).abs() <= 1e-12,
                    "gp_exp_quad_cov(array[] vector, array[] vector, real, array[] real) ({i}, {j}): {} vs {want}",
                    got[(i, j)]
                );
            }
        }
    }
    // stan: gp_exponential_cov(array[] real, real, real)
    {
        let x1 = gp_points_1d(&[0.3, -0.8, 1.4]);
        let x2 = x1.clone();
        let got = gp_exponential_cov(&x1, 1.3, 0.7).unwrap();
        let scale = [0.7, 0.7];
        assert_eq!((got.nrows(), got.ncols()), (x1.len(), x2.len()));
        for i in 0..x1.len() {
            for j in 0..x2.len() {
                let d = x1[i]
                    .iter()
                    .zip(&x2[j])
                    .zip(&scale)
                    .map(|((a, b), s)| ((a - b) / s).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let want = 1.3 * 1.3 * (-d).exp();
                assert!(
                    (got[(i, j)] - want).abs() <= 1e-12,
                    "gp_exponential_cov(array[] real, real, real) ({i}, {j}): {} vs {want}",
                    got[(i, j)]
                );
            }
        }
    }
    // stan: gp_exponential_cov(array[] real, array[] real, real, real)
    {
        let x1 = gp_points_1d(&[0.3, -0.8, 1.4]);
        let x2 = gp_points_1d(&[0.9, -0.1]);
        let got = gp_exponential_cov_cross(&x1, &x2, 1.3, 0.7).unwrap();
        let scale = [0.7, 0.7];
        assert_eq!((got.nrows(), got.ncols()), (x1.len(), x2.len()));
        for i in 0..x1.len() {
            for j in 0..x2.len() {
                let d = x1[i]
                    .iter()
                    .zip(&x2[j])
                    .zip(&scale)
                    .map(|((a, b), s)| ((a - b) / s).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let want = 1.3 * 1.3 * (-d).exp();
                assert!(
                    (got[(i, j)] - want).abs() <= 1e-12,
                    "gp_exponential_cov(array[] real, array[] real, real, real) ({i}, {j}): {} vs {want}",
                    got[(i, j)]
                );
            }
        }
    }
    // stan: gp_exponential_cov(array[] vector, real, real)
    {
        let x1 = vec![vec![0.3, -0.8], vec![1.4, 0.2], vec![-0.5, 0.6]];
        let x2 = x1.clone();
        let got = gp_exponential_cov(&x1, 1.3, 0.7).unwrap();
        let scale = [0.7, 0.7];
        assert_eq!((got.nrows(), got.ncols()), (x1.len(), x2.len()));
        for i in 0..x1.len() {
            for j in 0..x2.len() {
                let d = x1[i]
                    .iter()
                    .zip(&x2[j])
                    .zip(&scale)
                    .map(|((a, b), s)| ((a - b) / s).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let want = 1.3 * 1.3 * (-d).exp();
                assert!(
                    (got[(i, j)] - want).abs() <= 1e-12,
                    "gp_exponential_cov(array[] vector, real, real) ({i}, {j}): {} vs {want}",
                    got[(i, j)]
                );
            }
        }
    }
    // stan: gp_exponential_cov(array[] vector, real, array[] real)
    {
        let x1 = vec![vec![0.3, -0.8], vec![1.4, 0.2], vec![-0.5, 0.6]];
        let x2 = x1.clone();
        let l = [0.7, 1.6];
        let got = gp_exponential_cov_ard(&x1, 1.3, &l).unwrap();
        let scale = l;
        assert_eq!((got.nrows(), got.ncols()), (x1.len(), x2.len()));
        for i in 0..x1.len() {
            for j in 0..x2.len() {
                let d = x1[i]
                    .iter()
                    .zip(&x2[j])
                    .zip(&scale)
                    .map(|((a, b), s)| ((a - b) / s).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let want = 1.3 * 1.3 * (-d).exp();
                assert!(
                    (got[(i, j)] - want).abs() <= 1e-12,
                    "gp_exponential_cov(array[] vector, real, array[] real) ({i}, {j}): {} vs {want}",
                    got[(i, j)]
                );
            }
        }
    }
    // stan: gp_exponential_cov(array[] vector, array[] vector, real, real)
    {
        let x1 = vec![vec![0.3, -0.8], vec![1.4, 0.2], vec![-0.5, 0.6]];
        let x2 = vec![vec![0.9, -0.1], vec![0.0, 1.1]];
        let got = gp_exponential_cov_cross(&x1, &x2, 1.3, 0.7).unwrap();
        let scale = [0.7, 0.7];
        assert_eq!((got.nrows(), got.ncols()), (x1.len(), x2.len()));
        for i in 0..x1.len() {
            for j in 0..x2.len() {
                let d = x1[i]
                    .iter()
                    .zip(&x2[j])
                    .zip(&scale)
                    .map(|((a, b), s)| ((a - b) / s).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let want = 1.3 * 1.3 * (-d).exp();
                assert!(
                    (got[(i, j)] - want).abs() <= 1e-12,
                    "gp_exponential_cov(array[] vector, array[] vector, real, real) ({i}, {j}): {} vs {want}",
                    got[(i, j)]
                );
            }
        }
    }
    // stan: gp_exponential_cov(array[] vector, array[] vector, real, array[] real)
    {
        let x1 = vec![vec![0.3, -0.8], vec![1.4, 0.2], vec![-0.5, 0.6]];
        let x2 = vec![vec![0.9, -0.1], vec![0.0, 1.1]];
        let l = [0.7, 1.6];
        let got = gp_exponential_cov_ard_cross(&x1, &x2, 1.3, &l).unwrap();
        let scale = l;
        assert_eq!((got.nrows(), got.ncols()), (x1.len(), x2.len()));
        for i in 0..x1.len() {
            for j in 0..x2.len() {
                let d = x1[i]
                    .iter()
                    .zip(&x2[j])
                    .zip(&scale)
                    .map(|((a, b), s)| ((a - b) / s).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let want = 1.3 * 1.3 * (-d).exp();
                assert!(
                    (got[(i, j)] - want).abs() <= 1e-12,
                    "gp_exponential_cov(array[] vector, array[] vector, real, array[] real) ({i}, {j}): {} vs {want}",
                    got[(i, j)]
                );
            }
        }
    }
    // stan: gp_matern32_cov(array[] real, real, real)
    {
        let x1 = gp_points_1d(&[0.3, -0.8, 1.4]);
        let x2 = x1.clone();
        let got = gp_matern32_cov(&x1, 1.3, 0.7).unwrap();
        let scale = [0.7, 0.7];
        assert_eq!((got.nrows(), got.ncols()), (x1.len(), x2.len()));
        for i in 0..x1.len() {
            for j in 0..x2.len() {
                let d = x1[i]
                    .iter()
                    .zip(&x2[j])
                    .zip(&scale)
                    .map(|((a, b), s)| ((a - b) / s).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let want = 1.3 * 1.3 * (1.0 + 3f64.sqrt() * d) * (-(3f64.sqrt()) * d).exp();
                assert!(
                    (got[(i, j)] - want).abs() <= 1e-12,
                    "gp_matern32_cov(array[] real, real, real) ({i}, {j}): {} vs {want}",
                    got[(i, j)]
                );
            }
        }
    }
    // stan: gp_matern32_cov(array[] real, array[] real, real, real)
    {
        let x1 = gp_points_1d(&[0.3, -0.8, 1.4]);
        let x2 = gp_points_1d(&[0.9, -0.1]);
        let got = gp_matern32_cov_cross(&x1, &x2, 1.3, 0.7).unwrap();
        let scale = [0.7, 0.7];
        assert_eq!((got.nrows(), got.ncols()), (x1.len(), x2.len()));
        for i in 0..x1.len() {
            for j in 0..x2.len() {
                let d = x1[i]
                    .iter()
                    .zip(&x2[j])
                    .zip(&scale)
                    .map(|((a, b), s)| ((a - b) / s).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let want = 1.3 * 1.3 * (1.0 + 3f64.sqrt() * d) * (-(3f64.sqrt()) * d).exp();
                assert!(
                    (got[(i, j)] - want).abs() <= 1e-12,
                    "gp_matern32_cov(array[] real, array[] real, real, real) ({i}, {j}): {} vs {want}",
                    got[(i, j)]
                );
            }
        }
    }
    // stan: gp_matern32_cov(array[] vector, real, real)
    {
        let x1 = vec![vec![0.3, -0.8], vec![1.4, 0.2], vec![-0.5, 0.6]];
        let x2 = x1.clone();
        let got = gp_matern32_cov(&x1, 1.3, 0.7).unwrap();
        let scale = [0.7, 0.7];
        assert_eq!((got.nrows(), got.ncols()), (x1.len(), x2.len()));
        for i in 0..x1.len() {
            for j in 0..x2.len() {
                let d = x1[i]
                    .iter()
                    .zip(&x2[j])
                    .zip(&scale)
                    .map(|((a, b), s)| ((a - b) / s).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let want = 1.3 * 1.3 * (1.0 + 3f64.sqrt() * d) * (-(3f64.sqrt()) * d).exp();
                assert!(
                    (got[(i, j)] - want).abs() <= 1e-12,
                    "gp_matern32_cov(array[] vector, real, real) ({i}, {j}): {} vs {want}",
                    got[(i, j)]
                );
            }
        }
    }
    // stan: gp_matern32_cov(array[] vector, real, array[] real)
    {
        let x1 = vec![vec![0.3, -0.8], vec![1.4, 0.2], vec![-0.5, 0.6]];
        let x2 = x1.clone();
        let l = [0.7, 1.6];
        let got = gp_matern32_cov_ard(&x1, 1.3, &l).unwrap();
        let scale = l;
        assert_eq!((got.nrows(), got.ncols()), (x1.len(), x2.len()));
        for i in 0..x1.len() {
            for j in 0..x2.len() {
                let d = x1[i]
                    .iter()
                    .zip(&x2[j])
                    .zip(&scale)
                    .map(|((a, b), s)| ((a - b) / s).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let want = 1.3 * 1.3 * (1.0 + 3f64.sqrt() * d) * (-(3f64.sqrt()) * d).exp();
                assert!(
                    (got[(i, j)] - want).abs() <= 1e-12,
                    "gp_matern32_cov(array[] vector, real, array[] real) ({i}, {j}): {} vs {want}",
                    got[(i, j)]
                );
            }
        }
    }
    // stan: gp_matern32_cov(array[] vector, array[] vector, real, real)
    {
        let x1 = vec![vec![0.3, -0.8], vec![1.4, 0.2], vec![-0.5, 0.6]];
        let x2 = vec![vec![0.9, -0.1], vec![0.0, 1.1]];
        let got = gp_matern32_cov_cross(&x1, &x2, 1.3, 0.7).unwrap();
        let scale = [0.7, 0.7];
        assert_eq!((got.nrows(), got.ncols()), (x1.len(), x2.len()));
        for i in 0..x1.len() {
            for j in 0..x2.len() {
                let d = x1[i]
                    .iter()
                    .zip(&x2[j])
                    .zip(&scale)
                    .map(|((a, b), s)| ((a - b) / s).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let want = 1.3 * 1.3 * (1.0 + 3f64.sqrt() * d) * (-(3f64.sqrt()) * d).exp();
                assert!(
                    (got[(i, j)] - want).abs() <= 1e-12,
                    "gp_matern32_cov(array[] vector, array[] vector, real, real) ({i}, {j}): {} vs {want}",
                    got[(i, j)]
                );
            }
        }
    }
    // stan: gp_matern32_cov(array[] vector, array[] vector, real, array[] real)
    {
        let x1 = vec![vec![0.3, -0.8], vec![1.4, 0.2], vec![-0.5, 0.6]];
        let x2 = vec![vec![0.9, -0.1], vec![0.0, 1.1]];
        let l = [0.7, 1.6];
        let got = gp_matern32_cov_ard_cross(&x1, &x2, 1.3, &l).unwrap();
        let scale = l;
        assert_eq!((got.nrows(), got.ncols()), (x1.len(), x2.len()));
        for i in 0..x1.len() {
            for j in 0..x2.len() {
                let d = x1[i]
                    .iter()
                    .zip(&x2[j])
                    .zip(&scale)
                    .map(|((a, b), s)| ((a - b) / s).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let want = 1.3 * 1.3 * (1.0 + 3f64.sqrt() * d) * (-(3f64.sqrt()) * d).exp();
                assert!(
                    (got[(i, j)] - want).abs() <= 1e-12,
                    "gp_matern32_cov(array[] vector, array[] vector, real, array[] real) ({i}, {j}): {} vs {want}",
                    got[(i, j)]
                );
            }
        }
    }
    // stan: gp_matern52_cov(array[] real, real, real)
    {
        let x1 = gp_points_1d(&[0.3, -0.8, 1.4]);
        let x2 = x1.clone();
        let got = gp_matern52_cov(&x1, 1.3, 0.7).unwrap();
        let scale = [0.7, 0.7];
        assert_eq!((got.nrows(), got.ncols()), (x1.len(), x2.len()));
        for i in 0..x1.len() {
            for j in 0..x2.len() {
                let d = x1[i]
                    .iter()
                    .zip(&x2[j])
                    .zip(&scale)
                    .map(|((a, b), s)| ((a - b) / s).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let want = 1.3
                    * 1.3
                    * (1.0 + 5f64.sqrt() * d + 5.0 * d * d / 3.0)
                    * (-(5f64.sqrt()) * d).exp();
                assert!(
                    (got[(i, j)] - want).abs() <= 1e-12,
                    "gp_matern52_cov(array[] real, real, real) ({i}, {j}): {} vs {want}",
                    got[(i, j)]
                );
            }
        }
    }
    // stan: gp_matern52_cov(array[] real, array[] real, real, real)
    {
        let x1 = gp_points_1d(&[0.3, -0.8, 1.4]);
        let x2 = gp_points_1d(&[0.9, -0.1]);
        let got = gp_matern52_cov_cross(&x1, &x2, 1.3, 0.7).unwrap();
        let scale = [0.7, 0.7];
        assert_eq!((got.nrows(), got.ncols()), (x1.len(), x2.len()));
        for i in 0..x1.len() {
            for j in 0..x2.len() {
                let d = x1[i]
                    .iter()
                    .zip(&x2[j])
                    .zip(&scale)
                    .map(|((a, b), s)| ((a - b) / s).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let want = 1.3
                    * 1.3
                    * (1.0 + 5f64.sqrt() * d + 5.0 * d * d / 3.0)
                    * (-(5f64.sqrt()) * d).exp();
                assert!(
                    (got[(i, j)] - want).abs() <= 1e-12,
                    "gp_matern52_cov(array[] real, array[] real, real, real) ({i}, {j}): {} vs {want}",
                    got[(i, j)]
                );
            }
        }
    }
    // stan: gp_matern52_cov(array[] vector, real, real)
    {
        let x1 = vec![vec![0.3, -0.8], vec![1.4, 0.2], vec![-0.5, 0.6]];
        let x2 = x1.clone();
        let got = gp_matern52_cov(&x1, 1.3, 0.7).unwrap();
        let scale = [0.7, 0.7];
        assert_eq!((got.nrows(), got.ncols()), (x1.len(), x2.len()));
        for i in 0..x1.len() {
            for j in 0..x2.len() {
                let d = x1[i]
                    .iter()
                    .zip(&x2[j])
                    .zip(&scale)
                    .map(|((a, b), s)| ((a - b) / s).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let want = 1.3
                    * 1.3
                    * (1.0 + 5f64.sqrt() * d + 5.0 * d * d / 3.0)
                    * (-(5f64.sqrt()) * d).exp();
                assert!(
                    (got[(i, j)] - want).abs() <= 1e-12,
                    "gp_matern52_cov(array[] vector, real, real) ({i}, {j}): {} vs {want}",
                    got[(i, j)]
                );
            }
        }
    }
    // stan: gp_matern52_cov(array[] vector, real, array[] real)
    {
        let x1 = vec![vec![0.3, -0.8], vec![1.4, 0.2], vec![-0.5, 0.6]];
        let x2 = x1.clone();
        let l = [0.7, 1.6];
        let got = gp_matern52_cov_ard(&x1, 1.3, &l).unwrap();
        let scale = l;
        assert_eq!((got.nrows(), got.ncols()), (x1.len(), x2.len()));
        for i in 0..x1.len() {
            for j in 0..x2.len() {
                let d = x1[i]
                    .iter()
                    .zip(&x2[j])
                    .zip(&scale)
                    .map(|((a, b), s)| ((a - b) / s).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let want = 1.3
                    * 1.3
                    * (1.0 + 5f64.sqrt() * d + 5.0 * d * d / 3.0)
                    * (-(5f64.sqrt()) * d).exp();
                assert!(
                    (got[(i, j)] - want).abs() <= 1e-12,
                    "gp_matern52_cov(array[] vector, real, array[] real) ({i}, {j}): {} vs {want}",
                    got[(i, j)]
                );
            }
        }
    }
    // stan: gp_matern52_cov(array[] vector, array[] vector, real, real)
    {
        let x1 = vec![vec![0.3, -0.8], vec![1.4, 0.2], vec![-0.5, 0.6]];
        let x2 = vec![vec![0.9, -0.1], vec![0.0, 1.1]];
        let got = gp_matern52_cov_cross(&x1, &x2, 1.3, 0.7).unwrap();
        let scale = [0.7, 0.7];
        assert_eq!((got.nrows(), got.ncols()), (x1.len(), x2.len()));
        for i in 0..x1.len() {
            for j in 0..x2.len() {
                let d = x1[i]
                    .iter()
                    .zip(&x2[j])
                    .zip(&scale)
                    .map(|((a, b), s)| ((a - b) / s).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let want = 1.3
                    * 1.3
                    * (1.0 + 5f64.sqrt() * d + 5.0 * d * d / 3.0)
                    * (-(5f64.sqrt()) * d).exp();
                assert!(
                    (got[(i, j)] - want).abs() <= 1e-12,
                    "gp_matern52_cov(array[] vector, array[] vector, real, real) ({i}, {j}): {} vs {want}",
                    got[(i, j)]
                );
            }
        }
    }
    // stan: gp_matern52_cov(array[] vector, array[] vector, real, array[] real)
    {
        let x1 = vec![vec![0.3, -0.8], vec![1.4, 0.2], vec![-0.5, 0.6]];
        let x2 = vec![vec![0.9, -0.1], vec![0.0, 1.1]];
        let l = [0.7, 1.6];
        let got = gp_matern52_cov_ard_cross(&x1, &x2, 1.3, &l).unwrap();
        let scale = l;
        assert_eq!((got.nrows(), got.ncols()), (x1.len(), x2.len()));
        for i in 0..x1.len() {
            for j in 0..x2.len() {
                let d = x1[i]
                    .iter()
                    .zip(&x2[j])
                    .zip(&scale)
                    .map(|((a, b), s)| ((a - b) / s).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let want = 1.3
                    * 1.3
                    * (1.0 + 5f64.sqrt() * d + 5.0 * d * d / 3.0)
                    * (-(5f64.sqrt()) * d).exp();
                assert!(
                    (got[(i, j)] - want).abs() <= 1e-12,
                    "gp_matern52_cov(array[] vector, array[] vector, real, array[] real) ({i}, {j}): {} vs {want}",
                    got[(i, j)]
                );
            }
        }
    }
    // stan: gp_periodic_cov(array[] real, real, real, real)
    {
        let x1 = gp_points_1d(&[0.3, -0.8, 1.4]);
        let x2 = x1.clone();
        let got = gp_periodic_cov(&x1, 1.3, 0.7, 2.5).unwrap();
        let scale = [1.0, 1.0];
        assert_eq!((got.nrows(), got.ncols()), (x1.len(), x2.len()));
        for i in 0..x1.len() {
            for j in 0..x2.len() {
                let d = x1[i]
                    .iter()
                    .zip(&x2[j])
                    .zip(&scale)
                    .map(|((a, b), s)| ((a - b) / s).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let want = 1.3
                    * 1.3
                    * (-2.0 * (std::f64::consts::PI * d / 2.5).sin().powi(2) / (0.7 * 0.7)).exp();
                assert!(
                    (got[(i, j)] - want).abs() <= 1e-12,
                    "gp_periodic_cov(array[] real, real, real, real) ({i}, {j}): {} vs {want}",
                    got[(i, j)]
                );
            }
        }
    }
    // stan: gp_periodic_cov(array[] real, array[] real, real, real, real)
    {
        let x1 = gp_points_1d(&[0.3, -0.8, 1.4]);
        let x2 = gp_points_1d(&[0.9, -0.1]);
        let got = gp_periodic_cov_cross(&x1, &x2, 1.3, 0.7, 2.5).unwrap();
        let scale = [1.0, 1.0];
        assert_eq!((got.nrows(), got.ncols()), (x1.len(), x2.len()));
        for i in 0..x1.len() {
            for j in 0..x2.len() {
                let d = x1[i]
                    .iter()
                    .zip(&x2[j])
                    .zip(&scale)
                    .map(|((a, b), s)| ((a - b) / s).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let want = 1.3
                    * 1.3
                    * (-2.0 * (std::f64::consts::PI * d / 2.5).sin().powi(2) / (0.7 * 0.7)).exp();
                assert!(
                    (got[(i, j)] - want).abs() <= 1e-12,
                    "gp_periodic_cov(array[] real, array[] real, real, real, real) ({i}, {j}): {} vs {want}",
                    got[(i, j)]
                );
            }
        }
    }
    // stan: gp_periodic_cov(array[] vector, real, real, real)
    {
        let x1 = vec![vec![0.3, -0.8], vec![1.4, 0.2], vec![-0.5, 0.6]];
        let x2 = x1.clone();
        let got = gp_periodic_cov(&x1, 1.3, 0.7, 2.5).unwrap();
        let scale = [1.0, 1.0];
        assert_eq!((got.nrows(), got.ncols()), (x1.len(), x2.len()));
        for i in 0..x1.len() {
            for j in 0..x2.len() {
                let d = x1[i]
                    .iter()
                    .zip(&x2[j])
                    .zip(&scale)
                    .map(|((a, b), s)| ((a - b) / s).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let want = 1.3
                    * 1.3
                    * (-2.0 * (std::f64::consts::PI * d / 2.5).sin().powi(2) / (0.7 * 0.7)).exp();
                assert!(
                    (got[(i, j)] - want).abs() <= 1e-12,
                    "gp_periodic_cov(array[] vector, real, real, real) ({i}, {j}): {} vs {want}",
                    got[(i, j)]
                );
            }
        }
    }
    // stan: gp_periodic_cov(array[] vector, array[] vector, real, real, real)
    {
        let x1 = vec![vec![0.3, -0.8], vec![1.4, 0.2], vec![-0.5, 0.6]];
        let x2 = vec![vec![0.9, -0.1], vec![0.0, 1.1]];
        let got = gp_periodic_cov_cross(&x1, &x2, 1.3, 0.7, 2.5).unwrap();
        let scale = [1.0, 1.0];
        assert_eq!((got.nrows(), got.ncols()), (x1.len(), x2.len()));
        for i in 0..x1.len() {
            for j in 0..x2.len() {
                let d = x1[i]
                    .iter()
                    .zip(&x2[j])
                    .zip(&scale)
                    .map(|((a, b), s)| ((a - b) / s).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let want = 1.3
                    * 1.3
                    * (-2.0 * (std::f64::consts::PI * d / 2.5).sin().powi(2) / (0.7 * 0.7)).exp();
                assert!(
                    (got[(i, j)] - want).abs() <= 1e-12,
                    "gp_periodic_cov(array[] vector, array[] vector, real, real, real) ({i}, {j}): {} vs {want}",
                    got[(i, j)]
                );
            }
        }
    }
}
