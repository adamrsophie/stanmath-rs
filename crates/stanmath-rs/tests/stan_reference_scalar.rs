//! Values and gradients against Stan Math 5.4.0 for every function of one or two reals in the Stan
//! language except the `_rng` and bound transform functions. `tests/data/stan_reference_scalar.csv`
//! is produced by `tools/stan-reference/stan_reference_scalar.cpp` (written by `gen_scalar.py`
//! from the CmdStan 2.40.0 signature dump). Where Stan and stanmath-rs disagree, stanmath-rs must
//! match the 40 digit values of `tests/data/mpmath_reference_scalar.csv`
//! (`tools/reference/scalar.py`); those rows are Stan's inaccuracies.

use std::collections::HashMap;

use stanmath_rs::prelude::*;

/// The functions return either a scalar or a `Result`.
trait IntoResult<'t> {
    fn into_result(self) -> Result<Var<'t>>;
}

impl<'t> IntoResult<'t> for Var<'t> {
    fn into_result(self) -> Result<Var<'t>> {
        Ok(self)
    }
}

impl<'t> IntoResult<'t> for Result<Var<'t>> {
    fn into_result(self) -> Result<Var<'t>> {
        self
    }
}

/// The scalar forms of `add`, `subtract`, `multiply`, `elt_multiply`, and `elt_divide` are Rust's
/// operators; `squared_distance` of two reals is its slice form with length one.
fn eval<'t>(name: &str, x: &[Var<'t>]) -> Result<Var<'t>> {
    match name {
        "Phi" => Phi(x[0]).into_result(),
        "Phi_approx" => Phi_approx(x[0]).into_result(),
        "abs" => abs(x[0]).into_result(),
        "acos" => acos(x[0]).into_result(),
        "acosh" => acosh(x[0]).into_result(),
        "asin" => asin(x[0]).into_result(),
        "asinh" => asinh(x[0]).into_result(),
        "atan" => atan(x[0]).into_result(),
        "atanh" => atanh(x[0]).into_result(),
        "cbrt" => cbrt(x[0]).into_result(),
        "ceil" => ceil(x[0]).into_result(),
        "cos" => cos(x[0]).into_result(),
        "cosh" => cosh(x[0]).into_result(),
        "digamma" => digamma(x[0]).into_result(),
        "erf" => erf(x[0]).into_result(),
        "erfc" => erfc(x[0]).into_result(),
        "exp" => exp(x[0]).into_result(),
        "exp2" => exp2(x[0]).into_result(),
        "expm1" => expm1(x[0]).into_result(),
        "floor" => floor(x[0]).into_result(),
        "inv" => inv(x[0]).into_result(),
        "inv_Phi" => inv_Phi(x[0]).into_result(),
        "inv_cloglog" => inv_cloglog(x[0]).into_result(),
        "inv_erfc" => inv_erfc(x[0]).into_result(),
        "inv_logit" => inv_logit(x[0]).into_result(),
        "inv_sqrt" => inv_sqrt(x[0]).into_result(),
        "inv_square" => inv_square(x[0]).into_result(),
        "lambert_w0" => lambert_w0(x[0]).into_result(),
        "lambert_wm1" => lambert_wm1(x[0]).into_result(),
        "lgamma" => lgamma(x[0]).into_result(),
        "log" => log(x[0]).into_result(),
        "log10" => log10(x[0]).into_result(),
        "log1m" => log1m(x[0]).into_result(),
        "log1m_exp" => log1m_exp(x[0]).into_result(),
        "log1m_inv_logit" => log1m_inv_logit(x[0]).into_result(),
        "log1p" => log1p(x[0]).into_result(),
        "log1p_exp" => log1p_exp(x[0]).into_result(),
        "log2" => log2(x[0]).into_result(),
        "log_inv_logit" => log_inv_logit(x[0]).into_result(),
        "logit" => logit(x[0]).into_result(),
        "minus" => minus(x[0]).into_result(),
        "plus" => plus(x[0]).into_result(),
        "round" => round(x[0]).into_result(),
        "sin" => sin(x[0]).into_result(),
        "sinh" => sinh(x[0]).into_result(),
        "sqrt" => sqrt(x[0]).into_result(),
        "square" => square(x[0]).into_result(),
        "std_normal_cdf" => std_normal_cdf(x[0]).into_result(),
        "std_normal_lccdf" => std_normal_lccdf(x[0]).into_result(),
        "std_normal_lcdf" => std_normal_lcdf(x[0]).into_result(),
        "std_normal_log_qf" => std_normal_log_qf(x[0]).into_result(),
        "std_normal_lpdf" => std_normal_lpdf(x[0]).into_result(),
        "std_normal_qf" => std_normal_qf(x[0]).into_result(),
        "step" => step(x[0]).into_result(),
        "tan" => tan(x[0]).into_result(),
        "tanh" => tanh(x[0]).into_result(),
        "tgamma" => tgamma(x[0]).into_result(),
        "trigamma" => trigamma(x[0]).into_result(),
        "trunc" => trunc(x[0]).into_result(),
        "add" => Ok(x[0] + x[1]),
        "atan2" => atan2(x[0], x[1]).into_result(),
        "beta" => beta(x[0], x[1]).into_result(),
        "chi_square_cdf" => chi_square_cdf(x[0], x[1]).into_result(),
        "chi_square_lccdf" => chi_square_lccdf(x[0], x[1]).into_result(),
        "chi_square_lcdf" => chi_square_lcdf(x[0], x[1]).into_result(),
        "chi_square_lpdf" => chi_square_lpdf(x[0], x[1]).into_result(),
        "divide" => divide(x[0], x[1]).into_result(),
        "elt_divide" => Ok(x[0] / x[1]),
        "elt_multiply" => Ok(x[0] * x[1]),
        "exponential_cdf" => exponential_cdf(x[0], x[1]).into_result(),
        "exponential_lccdf" => exponential_lccdf(x[0], x[1]).into_result(),
        "exponential_lcdf" => exponential_lcdf(x[0], x[1]).into_result(),
        "exponential_lpdf" => exponential_lpdf(x[0], x[1]).into_result(),
        "fdim" => fdim(x[0], x[1]).into_result(),
        "fmax" => fmax(x[0], x[1]).into_result(),
        "fmin" => fmin(x[0], x[1]).into_result(),
        "fmod" => fmod(x[0], x[1]).into_result(),
        "gamma_p" => gamma_p(x[0], x[1]).into_result(),
        "gamma_q" => gamma_q(x[0], x[1]).into_result(),
        "hypergeometric_1F0" => hypergeometric_1F0(x[0], x[1]).into_result(),
        "hypot" => hypot(x[0], x[1]).into_result(),
        "inv_chi_square_cdf" => inv_chi_square_cdf(x[0], x[1]).into_result(),
        "inv_chi_square_lccdf" => inv_chi_square_lccdf(x[0], x[1]).into_result(),
        "inv_chi_square_lcdf" => inv_chi_square_lcdf(x[0], x[1]).into_result(),
        "inv_chi_square_lpdf" => inv_chi_square_lpdf(x[0], x[1]).into_result(),
        "lbeta" => lbeta(x[0], x[1]).into_result(),
        "lchoose" => lchoose(x[0], x[1]).into_result(),
        "lmultiply" => lmultiply(x[0], x[1]).into_result(),
        "log_diff_exp" => log_diff_exp(x[0], x[1]).into_result(),
        "log_falling_factorial" => log_falling_factorial(x[0], x[1]).into_result(),
        "log_inv_logit_diff" => log_inv_logit_diff(x[0], x[1]).into_result(),
        "log_modified_bessel_first_kind" => {
            log_modified_bessel_first_kind(x[0], x[1]).into_result()
        }
        "log_rising_factorial" => log_rising_factorial(x[0], x[1]).into_result(),
        "log_sum_exp" => log_sum_exp(x[0], x[1]).into_result(),
        "multiply" => Ok(x[0] * x[1]),
        "owens_t" => owens_t(x[0], x[1]).into_result(),
        "pow" => pow(x[0], x[1]).into_result(),
        "rayleigh_cdf" => rayleigh_cdf(x[0], x[1]).into_result(),
        "rayleigh_lccdf" => rayleigh_lccdf(x[0], x[1]).into_result(),
        "rayleigh_lcdf" => rayleigh_lcdf(x[0], x[1]).into_result(),
        "rayleigh_lpdf" => rayleigh_lpdf(x[0], x[1]).into_result(),
        "squared_distance" => squared_distance(&x[..1], &x[1..]),
        "subtract" => Ok(x[0] - x[1]),
        other => panic!("no dispatch for {other}"),
    }
}

fn list(s: &str) -> Vec<f64> {
    if s.is_empty() {
        Vec::new()
    } else {
        s.split(';').map(|v| v.parse().unwrap()).collect()
    }
}

fn close(a: f64, b: f64, rtol: f64) -> bool {
    a == b || (a.is_nan() && b.is_nan()) || (a - b).abs() <= rtol * b.abs().max(1.0)
}

fn agrees(ours: (f64, &[f64]), want: (f64, &[f64]), rtol: f64) -> bool {
    close(ours.0, want.0, rtol)
        && ours.1.len() == want.1.len()
        && ours.1.iter().zip(want.1).all(|(a, b)| close(*a, *b, rtol))
}

fn read(file: &str) -> String {
    let path = format!("{}/tests/data/{file}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

#[test]
fn scalar_functions_match_stan_or_beat_it() {
    let text = read("stan_reference_scalar.csv");
    let exact: HashMap<String, (f64, Vec<f64>)> = read("mpmath_reference_scalar.csv")
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|line| {
            let f: Vec<&str> = line.split(',').collect();
            (f[..3].join(","), (f[3].parse().unwrap(), list(f[4])))
        })
        .collect();
    let mut failures = Vec::new();
    let mut adjudicated = 0;
    let mut count = 0;
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        count += 1;
        let f: Vec<&str> = line.split(',').collect();
        let reals = list(f[2]);
        let ours = try_value_and_grad(|x| eval(f[0], x), &reals);
        match (f[3], ours) {
            // Stan throws where stanmath-rs returns NaN or an error: both reject the argument.
            ("error", Err(_)) => {}
            ("error", Ok(r)) if r.value.is_nan() => {}
            ("error", Ok(r)) => failures.push(format!("{line}: Stan rejects, we give {}", r.value)),
            (_, Err(e)) => failures.push(format!("{line}: we reject: {e}")),
            (v, Ok(r)) => {
                let ours = (r.value, &r.gradient[..]);
                if agrees(ours, (v.parse().unwrap(), &list(f[4])), 1e-10) {
                    continue;
                }
                match exact.get(&f[..3].join(",")) {
                    Some((v, g)) if agrees(ours, (*v, g), 1e-12) => adjudicated += 1,
                    Some((v, g)) => {
                        failures.push(format!("{line}: ours {ours:?}, mpmath ({v:e}, {g:?})"))
                    }
                    None => failures.push(format!("{line}: ours {ours:?}, no mpmath row")),
                }
            }
        }
    }
    println!("{adjudicated} disagreements with Stan resolved in stanmath-rs's favor");
    assert!(count > 500);
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
