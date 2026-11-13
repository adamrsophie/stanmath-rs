//! The Laplace approximation against closed forms: with a normal likelihood the approximation is
//! exact, so the marginal and the latent posterior are those of a Gaussian process regression.

// Dense matrix loops read more clearly with indices.
#![allow(clippy::needless_range_loop)]

use stanmath_rs_autodiff::functions::lgamma;
use stanmath_rs_autodiff::{Real, grad};
use stanmath_rs_core::{Matrix, Result};
use stanmath_rs_solvers::*;

const X: [f64; 3] = [0.0, 0.5, 1.3];
const Y: [f64; 3] = [0.4, -0.2, 1.1];
const HALF_LN_2PI: f64 = 0.918_938_533_204_672_8;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + b.abs())
}

/// Squared exponential covariance `alpha^2 exp(-(x_i - x_j)^2 / (2 ell^2))` with
/// `phi = [alpha, ell]`.
struct SqExp;

impl LaplaceCovariance for SqExp {
    fn covariance<S: Real>(&self, phi: &[S]) -> Result<Matrix<S>> {
        let (a, l) = (phi[0], phi[1]);
        Ok(Matrix::from_fn(X.len(), X.len(), |i, j| {
            let d = X[i] - X[j];
            a * a * (-(S::from_f64(d * d) / (l * l * 2.0))).exp()
        }))
    }
}

/// `y_i ~ normal(theta_i, sigma)` with `eta = [sigma]`.
struct Gauss;

impl LaplaceLikelihood for Gauss {
    fn log_likelihood<S: Real>(&self, theta: &[S], eta: &[S]) -> S {
        let sigma = eta[0];
        let mut lp = S::zero();
        for (&t, &y) in theta.iter().zip(&Y) {
            let z = (S::from_f64(y) - t) / sigma;
            lp += -(z * z * 0.5) - sigma.ln() - HALF_LN_2PI;
        }
        lp
    }
}

fn kernel(alpha: f64, ell: f64) -> Vec<Vec<f64>> {
    X.iter()
        .map(|&xi| {
            X.iter()
                .map(|&xj| alpha * alpha * (-(xi - xj).powi(2) / (2.0 * ell * ell)).exp())
                .collect()
        })
        .collect()
}

fn cholesky(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
            l[i][j] = if i == j {
                (a[i][i] - s).sqrt()
            } else {
                (a[i][j] - s) / l[j][j]
            };
        }
    }
    l
}

/// `a^(-1) b` for symmetric positive definite `a`.
fn solve(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let l = cholesky(a);
    let n = b.len();
    let mut z = vec![0.0; n];
    for i in 0..n {
        z[i] = (b[i] - (0..i).map(|k| l[i][k] * z[k]).sum::<f64>()) / l[i][i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = (z[i] - (i + 1..n).map(|k| l[k][i] * x[k]).sum::<f64>()) / l[i][i];
    }
    x
}

/// `K + sigma^2 I`.
fn marginal_cov(alpha: f64, ell: f64, sigma: f64) -> Vec<Vec<f64>> {
    let mut c = kernel(alpha, ell);
    for (i, row) in c.iter_mut().enumerate() {
        row[i] += sigma * sigma;
    }
    c
}

/// `log normal(y | 0, K + sigma^2 I)`.
fn exact(alpha: f64, ell: f64, sigma: f64) -> f64 {
    let c = marginal_cov(alpha, ell, sigma);
    let l = cholesky(&c);
    let log_det: f64 = (0..Y.len()).map(|i| 2.0 * l[i][i].ln()).sum();
    let quad: f64 = solve(&c, &Y).iter().zip(&Y).map(|(a, b)| a * b).sum();
    -0.5 * quad - 0.5 * log_det - Y.len() as f64 * HALF_LN_2PI
}

#[test]
fn normal_likelihood_marginal_is_exact() {
    let (alpha, ell, sigma) = (1.2, 0.7, 0.5);
    let v = laplace_marginal(&Gauss, &[sigma], &SqExp, &[alpha, ell]).unwrap();
    let want = exact(alpha, ell, sigma);
    assert!(close(v, want, 1e-8), "{v} vs {want}");
    let g = grad(
        |p| laplace_marginal(&Gauss, &p[2..], &SqExp, &p[..2]).unwrap(),
        &[alpha, ell, sigma],
    );
    let h = 1e-5;
    let fd = [
        (exact(alpha + h, ell, sigma) - exact(alpha - h, ell, sigma)) / (2.0 * h),
        (exact(alpha, ell + h, sigma) - exact(alpha, ell - h, sigma)) / (2.0 * h),
        (exact(alpha, ell, sigma + h) - exact(alpha, ell, sigma - h)) / (2.0 * h),
    ];
    for (j, (a, b)) in g.iter().zip(&fd).enumerate() {
        assert!(close(*a, *b, 1e-6), "parameter {j}: {a} vs {b}");
    }
}

#[test]
fn normal_likelihood_latent_posterior_is_exact() {
    let (alpha, ell, sigma) = (1.2, 0.7, 0.5);
    let (mean, l) = laplace_latent_solve(&Gauss, &[sigma], &SqExp, &[alpha, ell]).unwrap();
    // Posterior mean K C^(-1) y and covariance K - K C^(-1) K, with C = K + sigma^2 I.
    let k = kernel(alpha, ell);
    let c = marginal_cov(alpha, ell, sigma);
    let c_inv_y = solve(&c, &Y);
    let n = Y.len();
    for i in 0..n {
        let want: f64 = (0..n).map(|j| k[i][j] * c_inv_y[j]).sum();
        assert!(
            close(mean[i], want, 1e-8),
            "mean {i}: {} vs {want}",
            mean[i]
        );
    }
    let c_inv_k: Vec<Vec<f64>> = (0..n)
        .map(|j| solve(&c, &(0..n).map(|i| k[i][j]).collect::<Vec<_>>()))
        .collect();
    for i in 0..n {
        for j in 0..n {
            let got: f64 = (0..n).map(|m| l[(i, m)] * l[(j, m)]).sum();
            let want = k[i][j] - (0..n).map(|m| k[i][m] * c_inv_k[j][m]).sum::<f64>();
            assert!(close(got, want, 1e-8), "cov {i},{j}: {got} vs {want}");
        }
    }
}

/// `y_i ~ poisson_log(theta_i + m)`, written as a user likelihood with `eta = [m]`.
struct Poisson(Vec<i64>);

impl LaplaceLikelihood for Poisson {
    fn log_likelihood<S: Real>(&self, theta: &[S], eta: &[S]) -> S {
        let mut lp = S::zero();
        for (&t, &y) in theta.iter().zip(&self.0) {
            let e = t + eta[0];
            let yf = y as f64;
            lp += e * yf - e.exp() - lgamma(yf + 1.0);
        }
        lp
    }
}

#[test]
fn builtin_poisson_matches_the_user_likelihood() {
    let y = vec![2i64, 0, 5];
    let index = [0usize, 1, 2];
    let (alpha, ell, m) = (0.9, 0.6, 0.3);
    let builtin =
        laplace_marginal_poisson_log_lpmf(&y, &index, &[m; 3], &SqExp, &[alpha, ell]).unwrap();
    let general = laplace_marginal(&Poisson(y.clone()), &[m], &SqExp, &[alpha, ell]).unwrap();
    assert!(close(builtin, general, 1e-8), "{builtin} vs {general}");
    let gb = grad(
        |p| {
            let offsets = [p[2], p[2], p[2]];
            laplace_marginal_poisson_log_lpmf(&y, &index, &offsets, &SqExp, &p[..2]).unwrap()
        },
        &[alpha, ell, m],
    );
    let gg = grad(
        |p| laplace_marginal(&Poisson(y.clone()), &p[2..], &SqExp, &p[..2]).unwrap(),
        &[alpha, ell, m],
    );
    for (j, (a, b)) in gb.iter().zip(&gg).enumerate() {
        assert!(close(*a, *b, 1e-7), "parameter {j}: {a} vs {b}");
    }
}

/// Stan's probe covariance: `gp_exp_quad_cov(x, alpha, rho)` plus `1e-8` on the diagonal, at
/// `x = [0, 0.5, 1.2]`, with `phi = [alpha, rho]`.
struct ProbeCov;

impl LaplaceCovariance for ProbeCov {
    fn covariance<S: Real>(&self, phi: &[S]) -> Result<Matrix<S>> {
        const PX: [f64; 3] = [0.0, 0.5, 1.2];
        let (a, rho) = (phi[0], phi[1]);
        Ok(Matrix::from_fn(3, 3, |i, j| {
            let d = PX[i] - PX[j];
            let k = a * a * (-(S::from_f64(d * d) / (rho * rho * 2.0))).exp();
            if i == j { k + 1e-8 } else { k }
        }))
    }
}

/// Values from `tools/stan-reference/stan_probe.cpp` against Stan Math 5.4.0, called as stanc 2.40
/// calls them (one based `y_index` there, zero based here). The `_lupmf` forms equal the `_lpmf`
/// forms in Stan, whether or not an argument is a parameter.
#[test]
fn builtin_laplace_marginals_match_stan() {
    let phi = [1.2, 0.7];
    let index = [0usize, 1, 2];
    let m = [0.0; 3];
    let poisson = [1i64, 0, 3];
    let bernoulli = [1i64, 0, 1];
    let cases = [
        (
            "poisson_log",
            laplace_marginal_poisson_log_lpmf(&poisson, &index, &m, &ProbeCov, &phi).unwrap(),
            laplace_marginal_poisson_log_lupmf(&poisson, &index, &m, &ProbeCov, &phi).unwrap(),
            -5.2519293722403537,
        ),
        (
            "bernoulli_logit",
            laplace_marginal_bernoulli_logit_lpmf(&bernoulli, &index, &m, &ProbeCov, &phi).unwrap(),
            laplace_marginal_bernoulli_logit_lupmf(&bernoulli, &index, &m, &ProbeCov, &phi)
                .unwrap(),
            -2.4047765826747565,
        ),
    ];
    let permuted =
        laplace_marginal_poisson_log_lpmf(&poisson, &[2, 0, 1], &m, &ProbeCov, &phi).unwrap();
    assert!(
        close(permuted, -5.5979215135734739, 1e-7),
        "permuted: {permuted}"
    );
    for (name, lpmf, lupmf, want) in cases {
        assert!(close(lpmf, want, 1e-7), "{name}: {lpmf} vs {want}");
        assert_eq!(lpmf, lupmf, "{name}");
    }
}

#[test]
fn laplace_options_match_stan_defaults() {
    // generate_laplace_options(3) in Stan Math 5.4.0: (zeros(3), 1.49012e-08, 500, 1, 1000, 1).
    let o = generate_laplace_options(3);
    assert_eq!(o.theta_init.as_deref(), Some(&[0.0; 3][..]));
    assert!(close(o.tolerance, 1.49012e-08, 1e-5));
    assert_eq!(
        (
            o.max_num_steps,
            o.solver,
            o.max_steps_line_search,
            o.allow_fallback
        ),
        (500, 1, 1000, true)
    );
}
