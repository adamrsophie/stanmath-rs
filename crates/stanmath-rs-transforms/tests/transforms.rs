//! Round trips and log Jacobians of every transform.
//!
//! The log Jacobian added by each `*_constrain_lp` must equal `ln |det J|` of the map from the
//! unconstrained values to the free coordinates of the constrained value, computed here from a
//! forward-mode Jacobian and an LU factorization.

use stanmath_rs_autodiff::{Dual64, Real, grad, jacobian_forward};
use stanmath_rs_core::Matrix;
use stanmath_rs_transforms::*;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + b.abs())
}

/// `ln |det A|` by LU with partial pivoting.
fn log_abs_det(a: &Matrix<f64>) -> f64 {
    let n = a.nrows();
    let mut m = a.to_rows();
    let mut acc = 0.0;
    for c in 0..n {
        let p = (c..n)
            .max_by(|&i, &j| m[i][c].abs().total_cmp(&m[j][c].abs()))
            .unwrap();
        m.swap(c, p);
        let pivot = m[c][c];
        acc += pivot.abs().ln();
        for r in c + 1..n {
            let f = m[r][c] / pivot;
            let (top, bottom) = m.split_at_mut(r);
            for (target, source) in bottom[0][c..].iter_mut().zip(&top[c][c..]) {
                *target -= f * source;
            }
        }
    }
    acc
}

/// Checks `lp(y) = ln |det d free(constrain(y)) / dy| + offset`.
fn check_jacobian(
    name: &str,
    y: &[f64],
    free_coords: impl Fn(&[Dual64]) -> Vec<Dual64>,
    lp: impl Fn(&[f64]) -> f64,
    offset: f64,
) {
    let (_, j) = jacobian_forward(free_coords, y);
    assert_eq!(j.nrows(), j.ncols(), "{name}: Jacobian not square");
    let want = log_abs_det(&j) + offset;
    let got = lp(y);
    assert!(
        close(got, want, 1e-12),
        "{name}: lp {got} vs ln|det J| {want}"
    );
}

#[test]
fn scalar_transforms() {
    for &y in &[-3.0, 0.0, 0.4, 5.0] {
        let x = lb_constrain(y, 2.0);
        assert!(close(lb_free(x, 2.0).unwrap(), y, 1e-14));
        let x = ub_constrain(y, -1.0);
        assert!(close(ub_free(x, -1.0).unwrap(), y, 1e-14));
        let x = lub_constrain(y, -1.0, 3.0).unwrap();
        assert!(close(lub_free(x, -1.0, 3.0).unwrap(), y, 1e-12));
        let x = offset_multiplier_constrain(y, 1.5, 0.3).unwrap();
        assert!(close(
            offset_multiplier_free(x, 1.5, 0.3).unwrap(),
            y,
            1e-14
        ));
        // ln |dx/dy| from forward mode.
        type Case = (&'static str, fn(Dual64) -> Dual64, fn(f64) -> f64);
        let cases: [Case; 4] = [
            (
                "lb",
                |v| lb_constrain(v, Dual64::constant(2.0)),
                |v| {
                    let mut lp = 0.0;
                    lb_constrain_lp(v, 2.0, &mut lp);
                    lp
                },
            ),
            (
                "ub",
                |v| ub_constrain(v, Dual64::constant(-1.0)),
                |v| {
                    let mut lp = 0.0;
                    ub_constrain_lp(v, -1.0, &mut lp);
                    lp
                },
            ),
            (
                "lub",
                |v| lub_constrain(v, Dual64::constant(-1.0), Dual64::constant(3.0)).unwrap(),
                |v| {
                    let mut lp = 0.0;
                    lub_constrain_lp(v, -1.0, 3.0, &mut lp).unwrap();
                    lp
                },
            ),
            (
                "offset_multiplier",
                |v| {
                    offset_multiplier_constrain(v, Dual64::constant(1.5), Dual64::constant(0.3))
                        .unwrap()
                },
                |v| {
                    let mut lp = 0.0;
                    offset_multiplier_constrain_lp(v, 1.5, 0.3, &mut lp).unwrap();
                    lp
                },
            ),
        ];
        for (name, f, lp) in cases {
            let d = f(Dual64::variable(y)).eps;
            assert!(close(lp(y), d.abs().ln(), 1e-13), "{name} at {y}");
        }
    }
    // Infinite bounds reduce to the one-sided or identity transforms.
    assert_eq!(
        lub_constrain(0.7, f64::NEG_INFINITY, f64::INFINITY).unwrap(),
        0.7
    );
    assert_eq!(
        lub_constrain(0.7, 1.0, f64::INFINITY).unwrap(),
        lb_constrain(0.7, 1.0)
    );
    // Extreme y stays finite and inside the bounds, and the Jacobian term stays finite.
    let mut lp = 0.0;
    let x = lub_constrain_lp(800.0, 0.0, 1.0, &mut lp).unwrap();
    assert!(x <= 1.0 && lp.is_finite());
    assert!(lub_constrain(0.0, 1.0, 1.0).is_err());
    assert!(lb_free(1.0, 2.0).is_err());
}

const Y: [f64; 4] = [0.3, -1.2, 0.8, 0.05];

#[test]
fn ordered_transforms() {
    let x = ordered_constrain(&Y);
    assert!(x.windows(2).all(|w| w[1] > w[0]));
    let back = ordered_free(&x).unwrap();
    assert!(Y.iter().zip(&back).all(|(a, b)| close(*a, *b, 1e-14)));
    check_jacobian(
        "ordered",
        &Y,
        ordered_constrain,
        |y| {
            let mut lp = 0.0;
            ordered_constrain_lp(y, &mut lp);
            lp
        },
        0.0,
    );
    let x = positive_ordered_constrain(&Y);
    assert!(x[0] > 0.0);
    let back = positive_ordered_free(&x).unwrap();
    assert!(Y.iter().zip(&back).all(|(a, b)| close(*a, *b, 1e-14)));
    check_jacobian(
        "positive_ordered",
        &Y,
        positive_ordered_constrain,
        |y| {
            let mut lp = 0.0;
            positive_ordered_constrain_lp(y, &mut lp);
            lp
        },
        0.0,
    );
    assert!(ordered_free(&[1.0, 1.0]).is_err());
}

#[test]
fn sum_to_zero_and_simplex() {
    let n = Y.len() as f64;
    let z = sum_to_zero_constrain(&Y);
    assert_eq!(z.len(), Y.len() + 1);
    assert!(z.iter().sum::<f64>().abs() < 1e-15);
    // Isometric: |z| = |y|.
    let nz: f64 = z.iter().map(|v| v * v).sum();
    let ny: f64 = Y.iter().map(|v| v * v).sum();
    assert!(close(nz, ny, 1e-14));
    let back = sum_to_zero_free(&z).unwrap();
    assert!(Y.iter().zip(&back).all(|(a, b)| close(*a, *b, 1e-14)));
    // No Jacobian term; relative to the first N coordinates the surface measure adds ln(N+1)/2.
    check_jacobian(
        "sum_to_zero",
        &Y,
        |y| {
            let mut z = sum_to_zero_constrain(y);
            z.pop();
            z
        },
        |_| 0.0,
        0.5 * (n + 1.0).ln(),
    );

    let x = simplex_constrain(&Y);
    assert!(close(x.iter().sum::<f64>(), 1.0, 1e-15) && x.iter().all(|&v| v > 0.0));
    let back = simplex_free(&x).unwrap();
    assert!(Y.iter().zip(&back).all(|(a, b)| close(*a, *b, 1e-13)));
    check_jacobian(
        "simplex",
        &Y,
        |y| {
            let mut x = simplex_constrain(y);
            x.pop();
            x
        },
        |y| {
            let mut lp = 0.0;
            simplex_constrain_lp(y, &mut lp);
            lp
        },
        // The simplex term is relative to Lebesgue measure on the first N coordinates, the
        // measure the Dirichlet density uses.
        0.0,
    );
    assert!(simplex_free(&[0.5, 0.6]).is_err());
    assert_eq!(simplex_constrain::<f64>(&[]), vec![1.0]);
}

#[test]
fn unit_vector() {
    let x = unit_vector_constrain(&Y).unwrap();
    assert!(close(x.iter().map(|v| v * v).sum(), 1.0, 1e-15));
    let mut lp = 0.0;
    unit_vector_constrain_lp(&Y, &mut lp).unwrap();
    assert!(close(
        lp,
        -0.5 * Y.iter().map(|v| v * v).sum::<f64>(),
        1e-15
    ));
    assert!(unit_vector_constrain(&[0.0, 0.0]).is_err());
    assert_eq!(unit_vector_free(&x).unwrap(), x);
}

fn strictly_lower<T: Real>(m: &Matrix<T>, include_diag: bool) -> Vec<T> {
    let mut v = Vec::new();
    for i in 0..m.nrows() {
        for j in 0..m.ncols().min(if include_diag { i + 1 } else { i }) {
            v.push(m[(i, j)]);
        }
    }
    v
}

#[test]
fn correlation_cholesky_factor() {
    let k = 4;
    let y = [0.3, -1.2, 0.8, 0.05, 2.0, -0.4];
    let l = cholesky_factor_corr_constrain(&y, k).unwrap();
    for i in 0..k {
        let norm: f64 = (0..k).map(|j| l[(i, j)].powi(2)).sum();
        assert!(close(norm, 1.0, 1e-15), "row {i} norm {norm}");
    }
    let back = cholesky_factor_corr_free(&l).unwrap();
    assert!(y.iter().zip(&back).all(|(a, b)| close(*a, *b, 1e-12)));
    check_jacobian(
        "cholesky_factor_corr",
        &y,
        |v| strictly_lower(&cholesky_factor_corr_constrain(v, k).unwrap(), false),
        |v| {
            let mut lp = 0.0;
            cholesky_factor_corr_constrain_lp(v, k, &mut lp).unwrap();
            lp
        },
        0.0,
    );
    assert!(cholesky_factor_corr_constrain(&y[..5], k).is_err());
}

#[test]
fn covariance_transforms() {
    let k = 3;
    let x = [0.2, -0.5, 0.1, 0.9, 0.3, -0.2];
    let l = cholesky_factor_cov_constrain(&x, k, k).unwrap();
    let back = cholesky_factor_cov_free(&l).unwrap();
    assert!(x.iter().zip(&back).all(|(a, b)| close(*a, *b, 1e-14)));
    check_jacobian(
        "cholesky_factor_cov",
        &x,
        |v| strictly_lower(&cholesky_factor_cov_constrain(v, k, k).unwrap(), true),
        |v| {
            let mut lp = 0.0;
            cholesky_factor_cov_constrain_lp(v, k, k, &mut lp).unwrap();
            lp
        },
        0.0,
    );
    // Rectangular factor: 4 by 2.
    let xr = [0.1, 0.4, -0.3, 0.2, 0.5, -0.6, 0.7];
    let lr = cholesky_factor_cov_constrain(&xr, 4, 2).unwrap();
    let back = cholesky_factor_cov_free(&lr).unwrap();
    assert!(xr.iter().zip(&back).all(|(a, b)| close(*a, *b, 1e-14)));

    let sigma = cov_matrix_constrain(&x, k).unwrap();
    let back = cov_matrix_free(&sigma).unwrap();
    assert!(x.iter().zip(&back).all(|(a, b)| close(*a, *b, 1e-13)));
    check_jacobian(
        "cov_matrix",
        &x,
        |v| strictly_lower(&cov_matrix_constrain(v, k).unwrap(), true),
        |v| {
            let mut lp = 0.0;
            cov_matrix_constrain_lp(v, k, &mut lp).unwrap();
            lp
        },
        0.0,
    );
}

#[test]
fn log_jacobians_differentiate_in_reverse_mode() {
    // Gradients of a transformed log density agree with forward mode.
    let y = [0.3, -1.2, 0.8];
    let g = grad(
        |p| {
            let mut lp = p[0] * 0.0;
            let x = simplex_constrain_lp(p, &mut lp);
            lp + x[0].ln() * 2.0
        },
        &y,
    );
    let (_, j) = jacobian_forward(
        |p| {
            let mut lp = p[0] * 0.0;
            let x = simplex_constrain_lp(p, &mut lp);
            vec![lp + x[0].ln() * 2.0]
        },
        &y,
    );
    for k in 0..3 {
        assert!(close(g[k], j[(0, k)], 1e-14));
    }
}

#[test]
fn stan_named_scalar_transforms() {
    type Case = (
        &'static str,
        fn(Dual64) -> Dual64,
        fn(f64, &mut f64) -> f64,
        fn(f64) -> f64,
    );
    let cases: [Case; 3] = [
        ("positive", positive_constrain, positive_constrain_lp, |x| {
            positive_free(x).unwrap()
        }),
        ("prob", prob_constrain, prob_constrain_lp, |x| {
            prob_free(x).unwrap()
        }),
        ("corr", corr_constrain, corr_constrain_lp, |x| {
            corr_free(x).unwrap()
        }),
    ];
    for (name, f, f_lp, free) in cases {
        for y in [-2.3, -0.1, 0.0, 0.7, 3.1] {
            let mut lp = 0.0;
            let x = f_lp(y, &mut lp);
            assert!(close(free(x), y, 1e-12), "{name}: round trip at {y}");
            check_jacobian(
                name,
                &[y],
                |v| vec![f(v[0])],
                |v| {
                    let mut lp = 0.0;
                    f_lp(v[0], &mut lp);
                    lp
                },
                0.0,
            );
        }
    }
    assert!(positive_free(-1.0).is_err());
    assert!(prob_free(1.5).is_err());
    assert!(corr_free(-1.0).is_err());
    assert_eq!(identity_free(identity_constrain(0.4)), 0.4);
}

#[test]
fn stan_named_cholesky_transforms() {
    let k = 4;
    let y = [0.3, -1.2, 0.8, 0.05, 2.0, -0.4];
    let l = cholesky_corr_constrain(&y, k).unwrap();
    assert_eq!(l, cholesky_factor_corr_constrain(&y, k).unwrap());
    let back = cholesky_corr_free(&l).unwrap();
    assert!(y.iter().zip(&back).all(|(a, b)| close(*a, *b, 1e-12)));

    let xr = [0.1, 0.4, -0.3, 0.2, 0.5, -0.6, 0.7];
    let lr = cholesky_factor_constrain(&xr, 4, 2).unwrap();
    let back = cholesky_factor_free(&lr).unwrap();
    assert!(xr.iter().zip(&back).all(|(a, b)| close(*a, *b, 1e-14)));
    check_jacobian(
        "cholesky_factor 4 by 2",
        &xr,
        |v| strictly_lower(&cholesky_factor_constrain(v, 4, 2).unwrap(), true),
        |v| {
            let mut lp = 0.0;
            cholesky_factor_constrain_lp(v, 4, 2, &mut lp).unwrap();
            lp
        },
        0.0,
    );
}

/// `L Lᵀ` for a lower triangular `L`.
fn self_outer<T: Real>(l: &Matrix<T>) -> Matrix<T> {
    let n = l.nrows();
    Matrix::from_fn(n, n, |i, j| {
        (0..l.ncols()).fold(T::from_f64(0.0), |s, c| s + l[(i, c)] * l[(j, c)])
    })
}

#[test]
fn canonical_partial_correlations() {
    let k = 4;
    let cpcs = [0.3, -0.6, 0.15, 0.45, -0.2, 0.7];
    let l = read_corr_L(&cpcs, k).unwrap();
    let r = read_corr_matrix(&cpcs, k).unwrap();
    assert!(r.max_abs_diff(&self_outer(&l)) < 1e-14);
    for i in 0..k {
        assert!(close(r[(i, i)], 1.0, 1e-14), "diagonal {i}");
    }
    check_jacobian(
        "read_corr_matrix",
        &cpcs,
        |v| strictly_lower(&read_corr_matrix(v, k).unwrap(), false),
        |v| {
            let mut lp = 0.0;
            read_corr_matrix_lp(v, k, &mut lp).unwrap();
            lp
        },
        0.0,
    );
    // factor_U inverts tanh followed by read_corr_L, up to the transpose.
    let z = factor_U(&l.transpose()).unwrap();
    assert!(z.iter().zip(&cpcs).all(|(a, b)| close(a.tanh(), *b, 1e-12)));

    let sds = [0.5, 1.3, 2.0, 0.8];
    let sigma = read_cov_matrix(&cpcs, &sds).unwrap();
    let lc = read_cov_L(&cpcs, &sds).unwrap();
    assert!(sigma.max_abs_diff(&self_outer(&lc)) < 1e-13);
    for i in 0..k {
        assert!(close(sigma[(i, i)], sds[i] * sds[i], 1e-14));
    }
    let (z, log_sds) = factor_cov_matrix(&sigma).unwrap();
    assert!(z.iter().zip(&cpcs).all(|(a, b)| close(a.tanh(), *b, 1e-12)));
    assert!(
        log_sds
            .iter()
            .zip(&sds)
            .all(|(a, b)| close(a.exp(), *b, 1e-12))
    );

    // The full LKJ covariance transform and its inverse.
    let x: Vec<f64> = z.iter().chain(&log_sds).copied().collect();
    let back = cov_matrix_constrain_lkj(&x, k).unwrap();
    assert!(back.max_abs_diff(&sigma) < 1e-12);
    let free = cov_matrix_free_lkj(&sigma).unwrap();
    assert!(free.iter().zip(&x).all(|(a, b)| close(*a, *b, 1e-11)));
    check_jacobian(
        "cov_matrix_lkj",
        &x,
        |v| strictly_lower(&cov_matrix_constrain_lkj(v, k).unwrap(), true),
        |v| {
            let mut lp = 0.0;
            cov_matrix_constrain_lkj_lp(v, k, &mut lp).unwrap();
            lp
        },
        0.0,
    );

    let nu = make_nu(1.5, k).unwrap();
    assert_eq!(nu.len(), k * (k - 1) / 2);
    assert!(make_nu(0.0, k).is_err());
}
