//! Every elementary function, in every differentiation mode, against 60-digit mpmath references.
//!
//! The fixtures in `tests/data` are produced by `tools/reference/elementary.py`. For each function
//! and point they hold the value and the first and second derivatives. This test checks:
//!
//! * the `f64` value,
//! * the first derivative by forward mode (`Dual<f64>`) and by reverse mode (`Var`),
//! * the second derivative by forward over reverse (`Dual<Var>`) and forward over forward
//!   (`Dual<Dual<f64>>`).

use stanmath_rs_autodiff::functions::*;
use stanmath_rs_autodiff::{Real, derivative, grad, hessian, hessian_forward, second_derivative};

fn read(name: &str) -> Vec<(String, Vec<f64>)> {
    let path = format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    text.lines()
        .skip(1)
        .map(|line| {
            let mut fields = line.split(',');
            let f = fields.next().unwrap().to_string();
            let nums = fields.map(|v| v.parse::<f64>().unwrap()).collect();
            (f, nums)
        })
        .collect()
}

fn close_abs(actual: f64, expected: f64, rtol: f64, atol: f64) -> bool {
    if expected.is_infinite() || actual.is_infinite() {
        return actual == expected;
    }
    (actual - expected).abs() <= rtol * expected.abs() + atol + 1e-300
}

fn unary<T: Real>(name: &str, x: T) -> T {
    match name {
        "exp" => exp(x),
        "exp2" => x.exp2(),
        "expm1" => expm1(x),
        "log" => log(x),
        "log1p" => log1p(x),
        "log1m" => log1m(x),
        "log2" => log2(x),
        "log10" => log10(x),
        "sqrt" => sqrt(x),
        "cbrt" => cbrt(x),
        "square" => square(x),
        "inv" => inv(x),
        "inv_sqrt" => inv_sqrt(x),
        "inv_square" => inv_square(x),
        "abs" => abs(x),
        "sin" => sin(x),
        "cos" => cos(x),
        "tan" => tan(x),
        "asin" => asin(x),
        "acos" => acos(x),
        "atan" => atan(x),
        "sinh" => sinh(x),
        "cosh" => cosh(x),
        "tanh" => tanh(x),
        "asinh" => asinh(x),
        "acosh" => acosh(x),
        "atanh" => atanh(x),
        "log1p_exp" => log1p_exp(x),
        "log1m_exp" => log1m_exp(x),
        "inv_logit" => inv_logit(x),
        "logit" => logit(x),
        "log_inv_logit" => log_inv_logit(x),
        "log1m_inv_logit" => log1m_inv_logit(x),
        "lgamma" => lgamma(x),
        "tgamma" => tgamma(x),
        "digamma" => digamma(x),
        "trigamma" => trigamma(x),
        "polygamma2" => polygamma(2, x),
        "polygamma3" => polygamma(3, x),
        "erf" => erf(x),
        "erfc" => erfc(x),
        "erfcx" => erfcx(x),
        "Phi" => Phi(x),
        "log_Phi" => log_Phi(x),
        "inv_Phi" => inv_Phi(x),
        "log1pmx" => log1pmx(x),
        "lgamma_stirling_diff" => lgamma_stirling_diff(x),
        other => panic!("no dispatch for {other}"),
    }
}

fn binary<T: Real>(name: &str, a: T, b: T) -> T {
    match name {
        "pow" => pow(a, b),
        "atan2" => atan2(a, b),
        "hypot" => hypot(a, b),
        "log_sum_exp" => log_sum_exp(a, b),
        "log_diff_exp" => log_diff_exp(a, b),
        "lbeta" => lbeta(a, b),
        "gamma_p" => gamma_p(a, b),
        "gamma_q" => gamma_q(a, b),
        other => panic!("no dispatch for {other}"),
    }
}

/// Relative tolerances: values and first derivatives to a few ulps, second derivatives a little
/// looser because their rules chain more operations.
const RTOL_VALUE: f64 = 4e-15;
const RTOL_D1: f64 = 8e-15;
const RTOL_D2: f64 = 3e-14;

/// Per-function relative tolerances `(value, first derivative, second derivative)`.
///
/// The defaults hold for the elementary functions. Looser entries are documented in
/// `docs/numerical-accuracy.md` with the reason for each.
fn tolerance(name: &str) -> (f64, f64, f64, f64) {
    match name {
        // Sums of O(1) terms (recurrence plus asymptotic series): the absolute error is a few
        // ulps of 1, which is large relative to values near the roots of digamma.
        "lgamma" | "digamma" | "trigamma" | "polygamma2" | "polygamma3" => {
            (1e-14, 1e-14, 3e-14, 2e-15)
        }
        // The derivative is 1 / φ(Φ⁻¹(p)), whose relative condition number with respect to the
        // computed quantile y is y^2 (about 1400 at p = 1e-300).
        "inv_Phi" => (RTOL_VALUE, 3e-13, 3e-13, 0.0),
        // Iterative algorithms and recurrences: a few dozen operations per result.
        "lgamma_stirling_diff" | "lbeta" | "gamma_p" | "gamma_q" | "beta_inc" => {
            (1e-14, 1e-13, 1e-12, 0.0)
        }
        _ => (RTOL_VALUE, RTOL_D1, RTOL_D2, 0.0),
    }
}

fn check_unary(file: &str) {
    let mut failures = Vec::new();
    let rows = read(file);
    assert!(!rows.is_empty());
    for (name, v) in rows {
        let (x, value, d1, d2) = (v[0], v[1], v[2], v[3]);
        let (tv, t1, t2, atol) = tolerance(&name);
        let mut checks = vec![("value", unary::<f64>(&name, x), value, tv)];
        if !d1.is_nan() {
            checks.push(("forward d1", derivative(|d| unary(&name, d), x), d1, t1));
            checks.push(("reverse d1", grad(|p| unary(&name, p[0]), &[x])[0], d1, t1));
        }
        if !d2.is_nan() {
            checks.push((
                "fwd-over-rev d2",
                hessian(|p| unary(&name, p[0]), &[x])[(0, 0)],
                d2,
                t2,
            ));
            checks.push((
                "fwd-over-fwd d2",
                second_derivative(|d| unary(&name, d), x),
                d2,
                t2,
            ));
        }
        for (mode, got, want, rtol) in checks {
            if !close_abs(got, want, rtol, atol) {
                failures.push(format!(
                    "{name}({x}) {mode}: got {got:e}, want {want:e}, rel err {:.2e}",
                    ((got - want) / want).abs()
                ));
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

fn check_binary(file: &str) {
    let mut failures = Vec::new();
    let rows = read(file);
    for (name, v) in rows {
        let (a, b, value) = (v[0], v[1], v[2]);
        let (da, db, daa, dab, dbb) = (v[3], v[4], v[5], v[6], v[7]);
        let (tv, t1, t2, atol) = tolerance(&name);
        let g = grad(|p| binary(&name, p[0], p[1]), &[a, b]);
        let h = hessian(|p| binary(&name, p[0], p[1]), &[a, b]);
        let hf = hessian_forward(|p| binary(&name, p[0], p[1]), &[a, b]);
        let checks = [
            ("value", binary::<f64>(&name, a, b), value, tv),
            ("reverse da", g[0], da, t1),
            ("reverse db", g[1], db, t1),
            ("fwd-over-rev daa", h[(0, 0)], daa, t2),
            ("fwd-over-rev dab", h[(0, 1)], dab, t2),
            ("fwd-over-rev dba", h[(1, 0)], dab, t2),
            ("fwd-over-rev dbb", h[(1, 1)], dbb, t2),
            ("fwd-over-fwd daa", hf[(0, 0)], daa, t2),
            ("fwd-over-fwd dab", hf[(0, 1)], dab, t2),
            ("fwd-over-fwd dbb", hf[(1, 1)], dbb, t2),
        ];
        for (mode, got, want, rtol) in checks {
            if !close_abs(got, want, rtol, atol) {
                failures.push(format!(
                    "{name}({a}, {b}) {mode}: got {got:e}, want {want:e}, rel err {:.2e}",
                    ((got - want) / want).abs()
                ));
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn unary_functions_match_mpmath_in_every_mode() {
    check_unary("unary_reference.csv");
}

#[test]
fn binary_functions_match_mpmath_in_every_mode() {
    check_binary("binary_reference.csv");
}

#[test]
fn special_unary_functions_match_mpmath_in_every_mode() {
    check_unary("special_unary_reference.csv");
}

#[test]
fn special_binary_functions_match_mpmath_in_every_mode() {
    check_binary("special_binary_reference.csv");
}

#[test]
fn incomplete_beta_matches_mpmath() {
    let mut failures = Vec::new();
    let path = format!(
        "{}/tests/data/beta_inc_reference.csv",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(path).unwrap();
    for line in text.lines().skip(1) {
        let v: Vec<f64> = line.split(',').map(|x| x.parse().unwrap()).collect();
        let (a, b, x) = (v[0], v[1], v[2]);
        let (tv, t1, _, atol) = tolerance("beta_inc");
        let g = grad(|p| beta_inc(p[0], p[1], p[2]), &[a, b, x]);
        let checks = [
            ("value", beta_inc(a, b, x), v[3], tv),
            ("da", g[0], v[4], t1),
            ("db", g[1], v[5], t1),
            ("dx", g[2], v[6], t1),
        ];
        for (mode, got, want, rtol) in checks {
            if !close_abs(got, want, rtol, atol) {
                failures.push(format!(
                    "beta_inc({a}, {b}, {x}) {mode}: got {got:e}, want {want:e}, rel err {:.2e}",
                    ((got - want) / want).abs()
                ));
            }
        }
        // Hessian modes must agree with each other.
        let h = hessian(|p| beta_inc(p[0], p[1], p[2]), &[a, b, x]);
        let hf = hessian_forward(|p| beta_inc(p[0], p[1], p[2]), &[a, b, x]);
        let scale = hf.as_slice().iter().fold(0.0f64, |m, v| m.max(v.abs()));
        if h.max_abs_diff(&hf) > 1e-9 * scale {
            failures.push(format!("beta_inc({a}, {b}, {x}) Hessian modes disagree"));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
