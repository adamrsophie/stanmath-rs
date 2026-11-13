//! Values and gradients against Stan Math 5.4.0.
//!
//! `tests/data/stan_reference.csv` is produced by `tools/stan-reference`; see its makefile. Each
//! case is evaluated in reverse mode with stanmath-rs and compared with Stan's value and gradient.
//! Rows where Stan throws must make stanmath-rs return an error too.

use stanmath_rs_autodiff::{Var, try_value_and_grad};
use stanmath_rs_core::{Matrix, Result};
use stanmath_rs_distributions::*;

fn eval<'t>(name: &str, n: &[i64], x: &[Var<'t>]) -> Result<Var<'t>> {
    match name {
        "normal_lpdf" => normal_lpdf(x[0], x[1], x[2]),
        "normal_lcdf" => normal_lcdf(x[0], x[1], x[2]),
        "normal_lccdf" => normal_lccdf(x[0], x[1], x[2]),
        "std_normal_lpdf" => std_normal_lpdf(x[0]),
        "std_normal_lcdf" => std_normal_lcdf(x[0]),
        "std_normal_lccdf" => std_normal_lccdf(x[0]),
        "student_t_lpdf" => student_t_lpdf(x[0], x[1], x[2], x[3]),
        "student_t_lcdf" => student_t_lcdf(x[0], x[1], x[2], x[3]),
        "student_t_lccdf" => student_t_lccdf(x[0], x[1], x[2], x[3]),
        "cauchy_lpdf" => cauchy_lpdf(x[0], x[1], x[2]),
        "cauchy_lcdf" => cauchy_lcdf(x[0], x[1], x[2]),
        "cauchy_lccdf" => cauchy_lccdf(x[0], x[1], x[2]),
        "logistic_lpdf" => logistic_lpdf(x[0], x[1], x[2]),
        "logistic_lcdf" => logistic_lcdf(x[0], x[1], x[2]),
        "logistic_lccdf" => logistic_lccdf(x[0], x[1], x[2]),
        "exponential_lpdf" => exponential_lpdf(x[0], x[1]),
        "exponential_lcdf" => exponential_lcdf(x[0], x[1]),
        "exponential_lccdf" => exponential_lccdf(x[0], x[1]),
        "gamma_lpdf" => gamma_lpdf(x[0], x[1], x[2]),
        "gamma_lcdf" => gamma_lcdf(x[0], x[1], x[2]),
        "gamma_lccdf" => gamma_lccdf(x[0], x[1], x[2]),
        "inv_gamma_lpdf" => inv_gamma_lpdf(x[0], x[1], x[2]),
        "inv_gamma_lcdf" => inv_gamma_lcdf(x[0], x[1], x[2]),
        "inv_gamma_lccdf" => inv_gamma_lccdf(x[0], x[1], x[2]),
        "beta_lpdf" => beta_lpdf(x[0], x[1], x[2]),
        "beta_lcdf" => beta_lcdf(x[0], x[1], x[2]),
        "beta_lccdf" => beta_lccdf(x[0], x[1], x[2]),
        "lognormal_lpdf" => lognormal_lpdf(x[0], x[1], x[2]),
        "lognormal_lcdf" => lognormal_lcdf(x[0], x[1], x[2]),
        "lognormal_lccdf" => lognormal_lccdf(x[0], x[1], x[2]),
        "uniform_lpdf" => uniform_lpdf(x[0], x[1], x[2]),
        "uniform_lcdf" => uniform_lcdf(x[0], x[1], x[2]),
        "uniform_lccdf" => uniform_lccdf(x[0], x[1], x[2]),
        "weibull_lpdf" => weibull_lpdf(x[0], x[1], x[2]),
        "weibull_lcdf" => weibull_lcdf(x[0], x[1], x[2]),
        "weibull_lccdf" => weibull_lccdf(x[0], x[1], x[2]),
        "bernoulli_lpmf" => bernoulli_lpmf(n[0], x[0]),
        "bernoulli_lcdf" => bernoulli_lcdf(n[0], x[0]),
        "bernoulli_lccdf" => bernoulli_lccdf(n[0], x[0]),
        "bernoulli_logit_lpmf" => bernoulli_logit_lpmf(n[0], x[0]),
        "binomial_lpmf" => binomial_lpmf(n[0], n[1], x[0]),
        "binomial_lcdf" => binomial_lcdf(n[0], n[1], x[0]),
        "binomial_lccdf" => binomial_lccdf(n[0], n[1], x[0]),
        "binomial_logit_lpmf" => binomial_logit_lpmf(n[0], n[1], x[0]),
        "poisson_lpmf" => poisson_lpmf(n[0], x[0]),
        "poisson_lcdf" => poisson_lcdf(n[0], x[0]),
        "poisson_lccdf" => poisson_lccdf(n[0], x[0]),
        "poisson_log_lpmf" => poisson_log_lpmf(n[0], x[0]),
        "neg_binomial_2_lpmf" => neg_binomial_2_lpmf(n[0], x[0], x[1]),
        "neg_binomial_2_lcdf" => neg_binomial_2_lcdf(n[0], x[0], x[1]),
        "neg_binomial_2_lccdf" => neg_binomial_2_lccdf(n[0], x[0], x[1]),
        "neg_binomial_2_log_lpmf" => neg_binomial_2_log_lpmf(n[0], x[0], x[1]),
        "categorical_lpmf" => categorical_lpmf(n[0], x),
        "categorical_logit_lpmf" => categorical_logit_lpmf(n[0], x),
        "multi_normal_cholesky_lpdf" => {
            multi_normal_cholesky_lpdf(&x[0..3], &x[3..6], &lower(&x[6..12]))
        }
        "multi_normal_lpdf" => multi_normal_lpdf(&x[0..3], &x[3..6], &symmetric(&x[6..12])),
        "lkj_corr_cholesky_lpdf" => lkj_corr_cholesky_lpdf(&lower(&x[0..6]), x[6]),
        "dirichlet_lpdf" => dirichlet_lpdf(&x[0..3], &x[3..6]),
        other => panic!("no dispatch for {other}"),
    }
}

/// A 3 by 3 lower triangular matrix from its lower triangle, row by row.
fn lower<'t>(v: &[Var<'t>]) -> Matrix<Var<'t>> {
    let z = Var::constant(0.0);
    Matrix::from_rows(&[[v[0], z, z], [v[1], v[2], z], [v[3], v[4], v[5]]])
}

/// A 3 by 3 symmetric matrix from its lower triangle, row by row.
fn symmetric<'t>(v: &[Var<'t>]) -> Matrix<Var<'t>> {
    Matrix::from_rows(&[[v[0], v[1], v[3]], [v[1], v[2], v[4]], [v[3], v[4], v[5]]])
}

fn list(s: &str) -> Vec<f64> {
    if s.is_empty() {
        Vec::new()
    } else {
        s.split(';').map(|v| v.parse().unwrap()).collect()
    }
}

struct Case {
    name: String,
    ints: Vec<i64>,
    reals: Vec<f64>,
    /// `None` when the reference rejects the arguments.
    value: Option<f64>,
    grad: Vec<f64>,
}

fn load(file: &str) -> Vec<Case> {
    let path = format!("{}/tests/data/{file}", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    text.lines()
        .filter(|l| !l.starts_with('#'))
        .map(|line| {
            let f: Vec<&str> = line.split(',').collect();
            Case {
                name: f[0].to_string(),
                ints: list(f[1]).iter().map(|&v| v as i64).collect(),
                reals: list(f[2]),
                value: if f[3] == "error" {
                    None
                } else {
                    Some(f[3].parse().unwrap())
                },
                grad: list(f[4]),
            }
        })
        .collect()
}

fn close(a: f64, b: f64, rtol: f64) -> bool {
    a == b || (a - b).abs() <= rtol * b.abs()
}

/// Evaluates a case with stanmath-rs: `(value, gradient)`, or `None` on error.
fn evaluate(c: &Case) -> Option<(f64, Vec<f64>)> {
    try_value_and_grad(|x| eval(&c.name, &c.ints, x), &c.reals)
        .ok()
        .map(|r| (r.value, r.gradient))
}

/// Tolerances of stanmath-rs against the high-precision references: `(value rtol, gradient rtol)`.
///
/// * Densities: a few ulps.
/// * CDFs: shape derivatives of the incomplete gamma and beta functions come from forward-mode
///   differentiation of their series and continued fractions, which sum dozens of terms; 1e-11.
/// * Student's t CDF at `nu >= 1000` differentiates the incomplete beta function at shapes of
///   order 500 to 5000, where the continued fraction needs hundreds of terms; 1e-9.
fn mpmath_tolerance(c: &Case) -> (f64, f64) {
    if c.name.starts_with("student_t_lc") && c.reals[1] >= 1e3 {
        (1e-13, 1e-9)
    } else if c.name.ends_with("cdf") {
        (1e-13, 1e-11)
    } else {
        (1e-13, 1e-12)
    }
}

/// Absolute error floor for a case: a few ulps of the largest term that can cancel, so that
/// exactly zero gradients such as `1 / beta - y` at `beta y = 1` compare equal to the reference
/// noise around zero.
fn abs_floor(c: &Case) -> f64 {
    let scale = c
        .reals
        .iter()
        .filter(|x| **x != 0.0)
        .map(|x| x.abs().max(1.0 / x.abs()))
        .fold(1.0, f64::max);
    1e-14 * scale
}

#[test]
fn distributions_match_high_precision_references() {
    let cases = load("mpmath_reference.csv");
    assert!(cases.len() > 140);
    let mut failures = Vec::new();
    for c in cases.iter().filter(|c| c.value.is_some()) {
        let Some((v, g)) = evaluate(c) else {
            failures.push(format!(
                "{}({:?}, {:?}): unexpected error",
                c.name, c.ints, c.reals
            ));
            continue;
        };
        let (tv, tg) = mpmath_tolerance(c);
        let floor = abs_floor(c);
        let want = c.value.unwrap();
        if !close(v, want, tv) && (v - want).abs() > floor {
            failures.push(format!(
                "{}({:?}, {:?}) value: {v:e} vs {want:e}",
                c.name, c.ints, c.reals
            ));
        }
        for (k, (a, b)) in g.iter().zip(&c.grad).enumerate() {
            if !close(*a, *b, tg) && (a - b).abs() > floor {
                failures.push(format!(
                    "{}({:?}, {:?}) d/dx{k}: {a:e} vs {b:e}, rel {:.1e}",
                    c.name,
                    c.ints,
                    c.reals,
                    ((a - b) / b).abs()
                ));
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// Agreement with Stan Math to 1e-12, except where the two disagree; then stanmath-rs must be the
/// one closer to the 80-digit reference. The disagreements found are listed in
/// `docs/compatibility.md`.
#[test]
fn distributions_match_stan_or_beat_it() {
    let stan = load("stan_reference.csv");
    let exact = load("mpmath_reference.csv");
    assert_eq!(stan.len(), exact.len());
    let mut failures = Vec::new();
    let mut adjudicated = Vec::new();
    for (s, m) in stan.iter().zip(&exact) {
        assert_eq!(
            (&s.name, &s.reals),
            (&m.name, &m.reals),
            "fixtures out of order"
        );
        let ours = evaluate(s);
        let (Some(sv), Some((v, g))) = (s.value, &ours) else {
            if s.value.is_none() && ours.is_some() {
                failures.push(format!(
                    "{}({:?}, {:?}): Stan rejects, stanmath-rs accepts",
                    s.name, s.ints, s.reals
                ));
            }
            if s.value.is_some() && ours.is_none() {
                failures.push(format!(
                    "{}({:?}, {:?}): stanmath-rs rejects, Stan accepts",
                    s.name, s.ints, s.reals
                ));
            }
            continue;
        };
        let mv = m.value.unwrap();
        let pairs = core::iter::once((*v, sv, mv)).chain(
            g.iter()
                .zip(&s.grad)
                .zip(&m.grad)
                .map(|((a, b), c)| (*a, *b, *c)),
        );
        for (k, (pg, st, mp)) in pairs.enumerate() {
            if close(pg, st, 1e-12) || (pg - st).abs() < 1e-300 {
                continue;
            }
            let what = if k == 0 {
                "value".to_string()
            } else {
                format!("d/dx{}", k - 1)
            };
            if (pg - mp).abs() <= (st - mp).abs().max(abs_floor(s)) && close(pg, mp, 1e-9) {
                adjudicated.push(format!(
                    "{}({:?}, {:?}) {what}: stanmath-rs error {:.1e}, Stan error {:.1e}",
                    s.name,
                    s.ints,
                    s.reals,
                    ((pg - mp) / mp).abs(),
                    ((st - mp) / mp).abs()
                ));
            } else {
                failures.push(format!(
                    "{}({:?}, {:?}) {what}: stanmath-rs {pg:e}, Stan {st:e}, mpmath {mp:e}",
                    s.name, s.ints, s.reals
                ));
            }
        }
    }
    println!(
        "{} disagreements with Stan resolved in stanmath-rs's favor:",
        adjudicated.len()
    );
    for a in &adjudicated {
        println!("  {a}");
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
