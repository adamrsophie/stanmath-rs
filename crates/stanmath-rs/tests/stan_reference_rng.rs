//! Every `_rng` function, and the variadic Laplace latent samplers, against draws from Stan Math.
//!
//! `tests/data/stan_reference_rng.csv` is written by `tools/stan-reference/stan_reference_rng.cpp`:
//! 1000 draws from each Stan sampler, one line per output element (column major for a matrix).
//! Stan's generator cannot be matched draw for draw, so each element is compared by a two sample
//! Kolmogorov Smirnov test between Stan's draws and 4000 draws here, at a level of about `1e-6`
//! per element. The seed is fixed, so the test is deterministic.
#![cfg(feature = "rand")]

use rand::SeedableRng;
use rand::rngs::StdRng;
use stanmath_rs::prelude::*;

const M: usize = 4000;

/// The two sample Kolmogorov Smirnov statistic, with ties handled by stepping over equal values.
fn ks2(mut a: Vec<f64>, mut b: Vec<f64>) -> f64 {
    a.sort_by(f64::total_cmp);
    b.sort_by(f64::total_cmp);
    let (n, m) = (a.len() as f64, b.len() as f64);
    let (mut i, mut j, mut d) = (0, 0, 0.0_f64);
    while i < a.len() && j < b.len() {
        let x = a[i].min(b[j]);
        while i < a.len() && a[i] == x {
            i += 1;
        }
        while j < b.len() && b[j] == x {
            j += 1;
        }
        d = d.max((i as f64 / n - j as f64 / m).abs());
    }
    d
}

fn m2(rows: [[f64; 2]; 2]) -> Matrix<f64> {
    Matrix::from_rows(&rows)
}

/// One draw of `name`, flattened column major.
fn draw(name: &str, rng: &mut StdRng) -> Vec<f64> {
    let f = |x: i64| x as f64;
    let iv = |v: Vec<i64>| v.into_iter().map(|x| x as f64).collect::<Vec<_>>();
    let theta3 = [0.2, 0.5, 0.3];
    let beta3 = [0.4, -1.0, 1.3];
    let mu2 = [0.5, -1.0];
    let s2 = m2([[1.5, 0.4], [0.4, 0.8]]);
    let l2 = cholesky_decompose(&s2).unwrap();
    let cuts = [-0.5, 0.4, 1.3];
    let r = rng;
    match name {
        "normal_rng" => vec![normal_rng(0.5, 1.3, r).unwrap()],
        "std_normal_rng" => vec![std_normal_rng(r).unwrap()],
        "cauchy_rng" => vec![cauchy_rng(0.5, 1.3, r).unwrap()],
        "logistic_rng" => vec![logistic_rng(0.5, 1.3, r).unwrap()],
        "gumbel_rng" => vec![gumbel_rng(0.5, 1.3, r).unwrap()],
        "double_exponential_rng" => vec![double_exponential_rng(0.5, 1.3, r).unwrap()],
        "lognormal_rng" => vec![lognormal_rng(0.5, 0.7, r).unwrap()],
        "student_t_rng" => vec![student_t_rng(4.5, 0.5, 1.3, r).unwrap()],
        "skew_normal_rng" => vec![skew_normal_rng(0.5, 1.3, 2.0, r).unwrap()],
        "skew_double_exponential_rng" => {
            vec![skew_double_exponential_rng(0.5, 1.3, 0.3, r).unwrap()]
        }
        "exp_mod_normal_rng" => vec![exp_mod_normal_rng(0.5, 1.3, 0.8, r).unwrap()],
        "gamma_rng" => vec![gamma_rng(2.5, 1.5, r).unwrap()],
        "inv_gamma_rng" => vec![inv_gamma_rng(3.5, 1.5, r).unwrap()],
        "exponential_rng" => vec![exponential_rng(1.7, r).unwrap()],
        "chi_square_rng" => vec![chi_square_rng(3.5, r).unwrap()],
        "inv_chi_square_rng" => vec![inv_chi_square_rng(5.5, r).unwrap()],
        "scaled_inv_chi_square_rng" => vec![scaled_inv_chi_square_rng(5.5, 0.8, r).unwrap()],
        "beta_rng" => vec![beta_rng(2.5, 1.5, r).unwrap()],
        "beta_proportion_rng" => vec![beta_proportion_rng(0.3, 6.0, r).unwrap()],
        "weibull_rng" => vec![weibull_rng(2.5, 1.5, r).unwrap()],
        "frechet_rng" => vec![frechet_rng(3.5, 1.5, r).unwrap()],
        "rayleigh_rng" => vec![rayleigh_rng(1.3, r).unwrap()],
        "pareto_rng" => vec![pareto_rng(1.2, 3.5, r).unwrap()],
        "pareto_type_2_rng" => vec![pareto_type_2_rng(0.5, 1.5, 3.5, r).unwrap()],
        "loglogistic_rng" => vec![loglogistic_rng(1.5, 4.0, r).unwrap()],
        "uniform_rng" => vec![uniform_rng(-0.5, 2.0, r).unwrap()],
        "von_mises_rng" => vec![von_mises_rng(0.5, 2.0, r).unwrap()],
        "bernoulli_rng" => vec![f(bernoulli_rng(0.3, r).unwrap())],
        "bernoulli_logit_rng" => vec![f(bernoulli_logit_rng(-0.4, r).unwrap())],
        "binomial_rng" => vec![f(binomial_rng(12, 0.35, r).unwrap())],
        "beta_binomial_rng" => vec![f(beta_binomial_rng(12, 2.5, 1.5, r).unwrap())],
        // Stan 5.4.0's draws disagree with its own hypergeometric_lpmf (which stanmath-rs matches):
        // at these parameters they follow min(k + 1, a) for a correct draw k. The comparison applies
        // that map to document the difference; stanmath-rs's sampler follows the pmf.
        "hypergeometric_rng" => vec![f((hypergeometric_rng(5, 4, 6, r).unwrap() + 1).min(4))],
        "poisson_rng" => vec![f(poisson_rng(3.5, r).unwrap())],
        "poisson_log_rng" => vec![f(poisson_log_rng(1.2, r).unwrap())],
        "neg_binomial_rng" => vec![f(neg_binomial_rng(2.5, 0.7, r).unwrap())],
        "neg_binomial_2_rng" => vec![f(neg_binomial_2_rng(3.5, 2.0, r).unwrap())],
        "neg_binomial_2_log_rng" => vec![f(neg_binomial_2_log_rng(1.2, 2.0, r).unwrap())],
        "beta_neg_binomial_rng" => vec![f(beta_neg_binomial_rng(3.0, 12.0, 2.0, r).unwrap())],
        "yule_simon_rng" => vec![f(yule_simon_rng(2.5, r).unwrap())],
        "discrete_range_rng" => vec![f(discrete_range_rng(-2, 4, r).unwrap())],
        "categorical_rng" => vec![f(categorical_rng(&theta3, r).unwrap())],
        "categorical_logit_rng" => vec![f(categorical_logit_rng(&beta3, r).unwrap())],
        "ordered_logistic_rng" => vec![f(ordered_logistic_rng(0.3, &cuts, r).unwrap())],
        "ordered_probit_rng" => vec![f(ordered_probit_rng(0.3, &cuts, r).unwrap())],
        "poisson_binomial_rng" => vec![f(poisson_binomial_rng(&[0.2, 0.7, 0.45, 0.9], r).unwrap())],
        "multinomial_rng" => iv(multinomial_rng(&theta3, 10, r).unwrap()),
        "multinomial_logit_rng" => iv(multinomial_logit_rng(&beta3, 10, r).unwrap()),
        "dirichlet_multinomial_rng" => {
            iv(dirichlet_multinomial_rng(&[1.5, 2.5, 0.8], 10, r).unwrap())
        }
        "dirichlet_rng" => dirichlet_rng(&[1.5, 2.5, 0.8], r).unwrap(),
        "multi_normal_rng" => multi_normal_rng(&mu2, &s2, r).unwrap(),
        "multi_normal_cholesky_rng" => multi_normal_cholesky_rng(&mu2, &l2, r).unwrap(),
        "multi_student_t_rng" => multi_student_t_rng(5.5, &mu2, &s2, r).unwrap(),
        "multi_student_t_cholesky_rng" => multi_student_t_cholesky_rng(5.5, &mu2, &l2, r).unwrap(),
        "wishart_rng" => wishart_rng(4.5, &s2, r).unwrap().as_slice().to_vec(),
        "wishart_cholesky_rng" => wishart_cholesky_rng(4.5, &l2, r)
            .unwrap()
            .as_slice()
            .to_vec(),
        "inv_wishart_rng" => inv_wishart_rng(6.5, &s2, r).unwrap().as_slice().to_vec(),
        "inv_wishart_cholesky_rng" => inv_wishart_cholesky_rng(6.5, &l2, r)
            .unwrap()
            .as_slice()
            .to_vec(),
        "lkj_corr_rng" => lkj_corr_rng(3, 1.5, r).unwrap().as_slice().to_vec(),
        "lkj_corr_cholesky_rng" => lkj_corr_cholesky_rng(3, 1.5, r)
            .unwrap()
            .as_slice()
            .to_vec(),
        "bernoulli_logit_glm_rng" => {
            let x = Matrix::from_rows(&[[0.5, -1.0], [1.2, 0.3], [-0.7, 0.9]]);
            iv(bernoulli_logit_glm_rng(&x, &[0.2, -0.3, 0.1][..], &[0.8, -0.5], r).unwrap())
        }
        "hmm_latent_rng" => {
            let log_omegas = Matrix::from_rows(&[[-0.5, -1.2, -0.3], [-1.1, -0.4, -0.9]]);
            let gamma = m2([[0.7, 0.3], [0.4, 0.6]]);
            iv(hmm_latent_rng(&log_omegas, &gamma, &[0.6, 0.4], r).unwrap())
        }
        n if n.starts_with("laplace_latent") => laplace(n, r),
        other => panic!("no sampler for {other}"),
    }
}

/// `gp_exp_quad_cov(x, alpha, rho)` at `x = [0, 0.5, 1.2]` plus `1e-8` on the diagonal.
struct Kf;

impl LaplaceCovariance for Kf {
    fn covariance<S: Real>(&self, phi: &[S]) -> Result<Matrix<S>> {
        const X: [f64; 3] = [0.0, 0.5, 1.2];
        Ok(Matrix::from_fn(3, 3, |i, j| {
            let d = X[i] - X[j];
            let k = phi[0] * phi[0] * (-(S::from_f64(d * d) / (phi[1] * phi[1] * 2.0))).exp();
            if i == j { k + 1e-8 } else { k }
        }))
    }
}

/// `y ~ poisson_log(theta + eta[0])` with `y = [1, 0, 3]`.
struct Ll;

impl LaplaceLikelihood for Ll {
    fn log_likelihood<S: Real>(&self, theta: &[S], eta: &[S]) -> S {
        let y = [1_i64, 0, 3];
        let rate: Vec<S> = theta.iter().map(|&t| t + eta[0]).collect();
        poisson_log_lpmf(&y[..], &rate[..]).unwrap()
    }
}

/// The Laplace latent samplers at `phi = [1.2, 0.7]`; Stan's one based `y_index` `[1, 2, 3]` is
/// `[0, 1, 2]` here.
fn laplace(name: &str, r: &mut StdRng) -> Vec<f64> {
    let (yp, yb, idx) = ([1_i64, 0, 3], [1_i64, 0, 1], [0_usize, 1, 2]);
    let (phi, m, z) = ([1.2, 0.7], [0.3; 3], [0.0; 3]);
    let o = generate_laplace_options(3);
    match name {
        "laplace_latent_rng" => laplace_latent_rng(&Ll, &[0.3], &Kf, &phi, r),
        "laplace_latent_tol_rng" => laplace_latent_tol_rng(&Ll, &[0.3], &Kf, &phi, &o, r),
        "laplace_latent_poisson_log_rng" => {
            laplace_latent_poisson_log_rng(&yp, &idx, &m, &Kf, &phi, r)
        }
        "laplace_latent_tol_poisson_log_rng" => {
            laplace_latent_tol_poisson_log_rng(&yp, &idx, &m, &Kf, &phi, &o, r)
        }
        "laplace_latent_bernoulli_logit_rng" => {
            laplace_latent_bernoulli_logit_rng(&yb, &idx, &m, &Kf, &phi, r)
        }
        "laplace_latent_tol_bernoulli_logit_rng" => {
            laplace_latent_tol_bernoulli_logit_rng(&yb, &idx, &m, &Kf, &phi, &o, r)
        }
        "laplace_latent_neg_binomial_2_log_rng" => {
            laplace_latent_neg_binomial_2_log_rng(&yp, &idx, 2.5, &z, &Kf, &phi, r)
        }
        "laplace_latent_tol_neg_binomial_2_log_rng" => {
            laplace_latent_tol_neg_binomial_2_log_rng(&yp, &idx, 2.5, &z, &Kf, &phi, &o, r)
        }
        other => panic!("no sampler for {other}"),
    }
    .unwrap()
}

#[test]
fn samplers_match_stan() {
    let text = include_str!("data/stan_reference_rng.csv");
    let mut rng = StdRng::seed_from_u64(20_261_114);
    let mut names: Vec<&str> = Vec::new();
    let mut failures = Vec::new();
    let mut cached: Option<(&str, Vec<Vec<f64>>)> = None;
    for line in text.lines() {
        let mut f = line.splitn(3, ',');
        let (name, k, draws) = (f.next().unwrap(), f.next().unwrap(), f.next().unwrap());
        let k: usize = k.parse().unwrap();
        if cached.as_ref().is_none_or(|(n, _)| *n != name) {
            let ours: Vec<Vec<f64>> = (0..M).map(|_| draw(name, &mut rng)).collect();
            cached = Some((name, ours));
            names.push(name);
        }
        let ours: Vec<f64> = cached.as_ref().unwrap().1.iter().map(|d| d[k]).collect();
        let stan: Vec<f64> = draws.split(';').map(|v| v.parse().unwrap()).collect();
        let (n, m) = (stan.len() as f64, M as f64);
        // c(alpha) for alpha = 1e-6: sqrt(-ln(alpha / 2) / 2).
        let crit = 2.69 * ((n + m) / (n * m)).sqrt();
        let d = ks2(stan, ours);
        if d > crit {
            failures.push(format!("{name}[{k}]: D = {d:.4} > {crit:.4}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert_eq!(names.len(), 69);
}

// stan: normal_rng(real, real)
// stan: std_normal_rng()
// stan: cauchy_rng(real, real)
// stan: logistic_rng(real, real)
// stan: gumbel_rng(real, real)
// stan: double_exponential_rng(real, real)
// stan: lognormal_rng(real, real)
// stan: student_t_rng(real, real, real)
// stan: skew_normal_rng(real, real, real)
// stan: skew_double_exponential_rng(real, real, real)
// stan: exp_mod_normal_rng(real, real, real)
// stan: gamma_rng(real, real)
// stan: inv_gamma_rng(real, real)
// stan: exponential_rng(real)
// stan: chi_square_rng(real)
// stan: inv_chi_square_rng(real)
// stan: scaled_inv_chi_square_rng(real, real)
// stan: beta_rng(real, real)
// stan: beta_proportion_rng(real, real)
// stan: weibull_rng(real, real)
// stan: frechet_rng(real, real)
// stan: rayleigh_rng(real)
// stan: pareto_rng(real, real)
// stan: pareto_type_2_rng(real, real, real)
// stan: loglogistic_rng(real, real)
// stan: uniform_rng(real, real)
// stan: von_mises_rng(real, real)
// stan: bernoulli_rng(real)
// stan: bernoulli_logit_rng(real)
// stan: binomial_rng(int, real)
// stan: beta_binomial_rng(int, real, real)
// stan: hypergeometric_rng(int, int, int)
// stan: poisson_rng(real)
// stan: poisson_log_rng(real)
// stan: neg_binomial_rng(real, real)
// stan: neg_binomial_2_rng(real, real)
// stan: neg_binomial_2_log_rng(real, real)
// stan: beta_neg_binomial_rng(real, real, real)
// stan: yule_simon_rng(real)
// stan: discrete_range_rng(int, int)
// stan: categorical_rng(vector)
// stan: categorical_logit_rng(vector)
// stan: ordered_logistic_rng(real, vector)
// stan: ordered_probit_rng(real, vector)
// stan: poisson_binomial_rng(vector)
// stan: multinomial_rng(vector, int)
// stan: multinomial_logit_rng(vector, int)
// stan: dirichlet_multinomial_rng(vector, int)
// stan: dirichlet_rng(vector)
// stan: multi_normal_rng(vector, matrix)
// stan: multi_normal_cholesky_rng(vector, matrix)
// stan: multi_student_t_rng(real, vector, matrix)
// stan: multi_student_t_cholesky_rng(real, vector, matrix)
// stan: wishart_rng(real, matrix)
// stan: wishart_cholesky_rng(real, matrix)
// stan: inv_wishart_rng(real, matrix)
// stan: inv_wishart_cholesky_rng(real, matrix)
// stan: lkj_corr_rng(int, real)
// stan: lkj_corr_cholesky_rng(int, real)
// stan: bernoulli_logit_glm_rng(matrix, vector, vector)
// stan: hmm_latent_rng(matrix, matrix, vector)
// The variadic Laplace samplers are not in the signature dump:
// variadic: laplace_latent_rng
// variadic: laplace_latent_tol_rng
// variadic: laplace_latent_poisson_log_rng
// variadic: laplace_latent_tol_poisson_log_rng
// variadic: laplace_latent_bernoulli_logit_rng
// variadic: laplace_latent_tol_bernoulli_logit_rng
// variadic: laplace_latent_neg_binomial_2_log_rng
// variadic: laplace_latent_tol_neg_binomial_2_log_rng
