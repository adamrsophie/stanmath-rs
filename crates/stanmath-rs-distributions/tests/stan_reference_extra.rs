//! Values and gradients against Stan Math 5.4.0 for the families that `stan_reference.rs` does
//! not cover. `tests/data/stan_reference_extra.csv` is produced by
//! `tools/stan-reference/stan_reference_extra.cpp`. Rows where Stan throws must make stanmath-rs
//! return an error too. Where the two disagree, stanmath-rs must match the 40 digit values of
//! `tests/data/mpmath_reference_extra.csv` (`tools/reference/extra.py`); those rows are Stan's
//! inaccuracies, listed in `docs/compatibility.md`.

use std::collections::HashMap;

use stanmath_rs_autodiff::{Var, try_value_and_grad};
use stanmath_rs_core::Result;
use stanmath_rs_distributions::*;

fn eval<'t>(name: &str, n: &[i64], x: &[Var<'t>]) -> Result<Var<'t>> {
    match name {
        "chi_square_lpdf" => chi_square_lpdf(x[0], x[1]),
        "chi_square_lcdf" => chi_square_lcdf(x[0], x[1]),
        "chi_square_lccdf" => chi_square_lccdf(x[0], x[1]),
        "inv_chi_square_lpdf" => inv_chi_square_lpdf(x[0], x[1]),
        "inv_chi_square_lcdf" => inv_chi_square_lcdf(x[0], x[1]),
        "inv_chi_square_lccdf" => inv_chi_square_lccdf(x[0], x[1]),
        "scaled_inv_chi_square_lpdf" => scaled_inv_chi_square_lpdf(x[0], x[1], x[2]),
        "scaled_inv_chi_square_lcdf" => scaled_inv_chi_square_lcdf(x[0], x[1], x[2]),
        "scaled_inv_chi_square_lccdf" => scaled_inv_chi_square_lccdf(x[0], x[1], x[2]),
        "double_exponential_lpdf" => double_exponential_lpdf(x[0], x[1], x[2]),
        "double_exponential_lcdf" => double_exponential_lcdf(x[0], x[1], x[2]),
        "double_exponential_lccdf" => double_exponential_lccdf(x[0], x[1], x[2]),
        "exp_mod_normal_lpdf" => exp_mod_normal_lpdf(x[0], x[1], x[2], x[3]),
        "exp_mod_normal_lcdf" => exp_mod_normal_lcdf(x[0], x[1], x[2], x[3]),
        "exp_mod_normal_lccdf" => exp_mod_normal_lccdf(x[0], x[1], x[2], x[3]),
        "frechet_lpdf" => frechet_lpdf(x[0], x[1], x[2]),
        "frechet_lcdf" => frechet_lcdf(x[0], x[1], x[2]),
        "frechet_lccdf" => frechet_lccdf(x[0], x[1], x[2]),
        "gumbel_lpdf" => gumbel_lpdf(x[0], x[1], x[2]),
        "gumbel_lcdf" => gumbel_lcdf(x[0], x[1], x[2]),
        "gumbel_lccdf" => gumbel_lccdf(x[0], x[1], x[2]),
        "pareto_lpdf" => pareto_lpdf(x[0], x[1], x[2]),
        "pareto_lcdf" => pareto_lcdf(x[0], x[1], x[2]),
        "pareto_lccdf" => pareto_lccdf(x[0], x[1], x[2]),
        "pareto_type_2_lpdf" => pareto_type_2_lpdf(x[0], x[1], x[2], x[3]),
        "pareto_type_2_lcdf" => pareto_type_2_lcdf(x[0], x[1], x[2], x[3]),
        "pareto_type_2_lccdf" => pareto_type_2_lccdf(x[0], x[1], x[2], x[3]),
        "rayleigh_lpdf" => rayleigh_lpdf(x[0], x[1]),
        "rayleigh_lcdf" => rayleigh_lcdf(x[0], x[1]),
        "rayleigh_lccdf" => rayleigh_lccdf(x[0], x[1]),
        "skew_normal_lpdf" => skew_normal_lpdf(x[0], x[1], x[2], x[3]),
        "skew_normal_lcdf" => skew_normal_lcdf(x[0], x[1], x[2], x[3]),
        "skew_normal_lccdf" => skew_normal_lccdf(x[0], x[1], x[2], x[3]),
        "skew_double_exponential_lpdf" => skew_double_exponential_lpdf(x[0], x[1], x[2], x[3]),
        "skew_double_exponential_lcdf" => skew_double_exponential_lcdf(x[0], x[1], x[2], x[3]),
        "skew_double_exponential_lccdf" => skew_double_exponential_lccdf(x[0], x[1], x[2], x[3]),
        "von_mises_lpdf" => von_mises_lpdf(x[0], x[1], x[2]),
        "von_mises_lcdf" => von_mises_lcdf(x[0], x[1], x[2]),
        "von_mises_lccdf" => von_mises_lccdf(x[0], x[1], x[2]),
        "beta_proportion_lpdf" => beta_proportion_lpdf(x[0], x[1], x[2]),
        "beta_proportion_lcdf" => beta_proportion_lcdf(x[0], x[1], x[2]),
        "beta_proportion_lccdf" => beta_proportion_lccdf(x[0], x[1], x[2]),
        "loglogistic_lpdf" => loglogistic_lpdf(x[0], x[1], x[2]),
        "neg_binomial_lpmf" => neg_binomial_lpmf(n[0], x[0], x[1]),
        "neg_binomial_lcdf" => neg_binomial_lcdf(n[0], x[0], x[1]),
        "neg_binomial_lccdf" => neg_binomial_lccdf(n[0], x[0], x[1]),
        "beta_binomial_lpmf" => beta_binomial_lpmf(n[0], n[1], x[0], x[1]),
        "beta_binomial_lcdf" => beta_binomial_lcdf(n[0], n[1], x[0], x[1]),
        "beta_binomial_lccdf" => beta_binomial_lccdf(n[0], n[1], x[0], x[1]),
        "beta_neg_binomial_lpmf" => beta_neg_binomial_lpmf(n[0], x[0], x[1], x[2]),
        "beta_neg_binomial_lcdf" => beta_neg_binomial_lcdf(n[0], x[0], x[1], x[2]),
        "beta_neg_binomial_lccdf" => beta_neg_binomial_lccdf(n[0], x[0], x[1], x[2]),
        "poisson_binomial_lpmf" => poisson_binomial_lpmf(n[0], x),
        "poisson_binomial_lcdf" => poisson_binomial_lcdf(n[0], x),
        "poisson_binomial_lccdf" => poisson_binomial_lccdf(n[0], x),
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
    a == b || (a - b).abs() <= rtol * b.abs().max(1.0)
}

/// `name,ints,reals` to `(value, gradient)` from a fixture.
fn table(file: &str) -> HashMap<String, (f64, Vec<f64>)> {
    let path = format!("{}/tests/data/{file}", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    text.lines()
        .filter(|l| !l.starts_with('#'))
        .map(|line| {
            let f: Vec<&str> = line.split(',').collect();
            let value = if f[3] == "error" {
                f64::NAN
            } else {
                f[3].parse().unwrap()
            };
            (f[..3].join(","), (value, list(f[4])))
        })
        .collect()
}

fn agrees(ours: (f64, &[f64]), want: (f64, &[f64]), rtol: f64) -> bool {
    close(ours.0, want.0, rtol)
        && ours.1.len() == want.1.len()
        && ours.1.iter().zip(want.1).all(|(a, b)| close(*a, *b, rtol))
}

#[test]
fn extra_distributions_match_stan_or_beat_it() {
    let path = format!(
        "{}/tests/data/stan_reference_extra.csv",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).unwrap();
    let exact = table("mpmath_reference_extra.csv");
    let mut failures = Vec::new();
    let mut adjudicated = 0;
    let mut count = 0;
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        count += 1;
        let f: Vec<&str> = line.split(',').collect();
        let ints: Vec<i64> = list(f[1]).iter().map(|&v| v as i64).collect();
        let reals = list(f[2]);
        let ours = try_value_and_grad(|x| eval(f[0], &ints, x), &reals);
        match (f[3], ours) {
            ("error", Ok(_)) => failures.push(format!("{line}: Stan rejects, we accept")),
            ("error", Err(_)) => {}
            (_, Err(e)) => failures.push(format!("{line}: we reject: {e}")),
            (v, Ok(r)) => {
                let stan: f64 = v.parse().unwrap();
                let ours = (r.value, &r.gradient[..]);
                if agrees(ours, (stan, &list(f[4])), 1e-10) {
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
    assert!(count > 150);
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
