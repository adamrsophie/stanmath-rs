//! Values and gradients against Stan Math 5.4.0 for the scalar signatures of the Stan language
//! that `stan_reference_scalar.rs` does not reach: constants, functions with an int argument,
//! three and four argument functions, the scalar bound transforms, and the 6 to 9 argument Wiener
//! forms. `tests/data/stan_reference_scalar_extra.csv` is produced by
//! `tools/stan-reference/stan_reference_scalar_extra.cpp`. A `_jacobian` case is the constrained
//! value plus the log Jacobian. Functions of ints return an int in Stan; here they return an
//! integer valued `f64` or `i64`, compared by value.
#![allow(clippy::cast_precision_loss)]

use stanmath_rs::prelude::*;

// stan: e()
// stan: pi()
// stan: sqrt2()
// stan: machine_precision()
// stan: not_a_number()
// stan: positive_infinity()
// stan: negative_infinity()
// stan: bessel_first_kind(int, real)
// stan: bessel_second_kind(int, real)
// stan: modified_bessel_first_kind(int, real)
// stan: modified_bessel_second_kind(int, real)
// stan: binary_log_loss(int, real)
// stan: lmgamma(int, real)
// stan: ldexp(real, int)
// stan: falling_factorial(real, int)
// stan: rising_factorial(real, int)
// stan: falling_factorial(int, int)
// stan: rising_factorial(int, int)
// stan: choose(int, int)
// stan: modulus(int, int)
// stan: logical_and(int, int)
// stan: logical_or(int, int)
// stan: logical_lt(int, int)
// stan: logical_lt(int, real)
// stan: logical_lt(real, int)
// stan: logical_lt(real, real)
// stan: is_nan(real)
// stan: to_int(real)
// stan: to_int(int)
// stan: hypergeometric_2F1(real, real, real, real)
// stan: inc_beta(real, real, real)
// stan: inv_inc_beta(real, real, real)
// stan: log_mix(real, real, real)
// stan: lower_bound_constrain(real, real)
// stan: lower_bound_jacobian(real, real)
// stan: lower_bound_unconstrain(real, real)
// stan: upper_bound_constrain(real, real)
// stan: upper_bound_jacobian(real, real)
// stan: upper_bound_unconstrain(real, real)
// stan: lower_upper_bound_constrain(real, real, real)
// stan: lower_upper_bound_jacobian(real, real, real)
// stan: lower_upper_bound_unconstrain(real, real, real)
// stan: offset_multiplier_constrain(real, real, real)
// stan: offset_multiplier_jacobian(real, real, real)
// stan: offset_multiplier_unconstrain(real, real, real)
// stan: wiener_lpdf(real, real, real, real, real, real)
// stan: wiener_lpdf(real, real, real, real, real, real, real)
// stan: wiener_lpdf(real, real, real, real, real, real, real, real)
// stan: wiener_lpdf(real, real, real, real, real, real, real, real, real)

/// The precision Stan was given for the Wiener forms that take one.
const PRECISION: f64 = 1e-8;

fn k<'t>(v: f64) -> Var<'t> {
    Var::constant(v)
}

fn b<'t>(v: bool) -> Var<'t> {
    k(f64::from(u8::from(v)))
}

fn eval<'t>(name: &str, n: &[i64], x: &[Var<'t>]) -> Result<Var<'t>> {
    let i = |j: usize| i32::try_from(n[j]).unwrap();
    let mut lp = x.first().map_or(k(0.0), |v| *v * 0.0);
    Ok(match name {
        "e" => k(e()),
        "pi" => k(pi()),
        "sqrt2" => k(sqrt2()),
        "machine_precision" => k(machine_precision()),
        "not_a_number" => k(not_a_number()),
        "positive_infinity" => k(positive_infinity()),
        "negative_infinity" => k(negative_infinity()),
        "bessel_first_kind" => bessel_first_kind(i(0), x[0]),
        "bessel_second_kind" => bessel_second_kind(i(0), x[0]),
        "modified_bessel_first_kind" => modified_bessel_first_kind(i(0), x[0]),
        "modified_bessel_second_kind" => modified_bessel_second_kind(i(0), x[0]),
        "binary_log_loss" => binary_log_loss(n[0], x[0]),
        "lmgamma" => lmgamma(i(0), x[0]),
        "ldexp" => ldexp(x[0], i(0)),
        "falling_factorial" => falling_factorial(x[0], i(0)),
        "rising_factorial" => rising_factorial(x[0], i(0)),
        "falling_factorial_int" => k(falling_factorial(n[0] as f64, i(1))),
        "rising_factorial_int" => k(rising_factorial(n[0] as f64, i(1))),
        "choose" => k(choose(n[0], n[1])? as f64),
        "modulus" => k(modulus(n[0], n[1])? as f64),
        "logical_and" => b(logical_and(n[0], n[1])),
        "logical_or" => b(logical_or(n[0], n[1])),
        "logical_lt_int_int" => b(logical_lt(n[0], n[1])),
        "logical_lt_int_real" => b(logical_lt(n[0], x[0])),
        "logical_lt_real_int" => b(logical_lt(x[0], n[0])),
        "logical_lt_real_real" => b(logical_lt(x[0], x[1])),
        "is_nan" => b(is_nan(x[0])),
        "to_int" => k(to_int(x[0].value())? as f64),
        "to_int_int" => k(n[0] as f64),
        "hypergeometric_2F1" => hypergeometric_2F1(x[0], x[1], x[2], x[3]),
        "inc_beta" => inc_beta(x[0], x[1], x[2]),
        "inv_inc_beta" => inv_inc_beta(x[0], x[1], x[2]),
        "log_mix" => log_mix(x[0], x[1], x[2]),
        "lower_bound_constrain" => lower_bound_constrain(x[0], x[1]),
        "lower_bound_jacobian" => lower_bound_jacobian(x[0], x[1], &mut lp) + lp,
        "lower_bound_unconstrain" => lower_bound_unconstrain(x[0], x[1])?,
        "upper_bound_constrain" => upper_bound_constrain(x[0], x[1]),
        "upper_bound_jacobian" => upper_bound_jacobian(x[0], x[1], &mut lp) + lp,
        "upper_bound_unconstrain" => upper_bound_unconstrain(x[0], x[1])?,
        "lower_upper_bound_constrain" => lower_upper_bound_constrain(x[0], x[1], x[2])?,
        "lower_upper_bound_jacobian" => lower_upper_bound_jacobian(x[0], x[1], x[2], &mut lp)? + lp,
        "lower_upper_bound_unconstrain" => lower_upper_bound_unconstrain(x[0], x[1], x[2])?,
        "offset_multiplier_constrain" => offset_multiplier_constrain(x[0], x[1], x[2])?,
        "offset_multiplier_jacobian" => offset_multiplier_jacobian(x[0], x[1], x[2], &mut lp)? + lp,
        "offset_multiplier_unconstrain" => offset_multiplier_unconstrain(x[0], x[1], x[2])?,
        "wiener_lpdf_6" => wiener5_lpdf(x[0], x[1], x[2], x[3], x[4], x[5])?,
        "wiener_lpdf_7" => wiener5_prec_lpdf(x[0], x[1], x[2], x[3], x[4], x[5], PRECISION)?,
        "wiener_lpdf_8" => wiener_full_lpdf(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7])?,
        "wiener_lpdf_9" => {
            wiener_full_prec_lpdf(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], PRECISION)?
        }
        other => panic!("no dispatch for {other}"),
    })
}

fn list<T: std::str::FromStr>(s: &str) -> Vec<T>
where
    T::Err: std::fmt::Debug,
{
    if s.is_empty() {
        Vec::new()
    } else {
        s.split(';').map(|v| v.parse().unwrap()).collect()
    }
}

fn close(a: f64, b: f64, rtol: f64) -> bool {
    a == b || (a.is_nan() && b.is_nan()) || (a - b).abs() <= rtol * b.abs().max(1.0)
}

#[test]
fn scalar_extra_functions_match_stan() {
    let data = include_str!("data/stan_reference_scalar_extra.csv");
    let mut failures = Vec::new();
    let mut count = 0;
    for line in data.lines().filter(|l| !l.starts_with('#')) {
        count += 1;
        let f: Vec<&str> = line.split(',').collect();
        let (name, ints, reals) = (f[0], list::<i64>(f[1]), list::<f64>(f[2]));
        // The Wiener forms integrate numerically. Stan's eight argument form uses its default
        // precision, 1e-4, and differs from ours by about 2e-6; the nine argument form at the same
        // points, with precision 1e-8, agrees to 1e-15.
        let rtol = match name {
            "wiener_lpdf_8" => 1e-5,
            n if n.starts_with("wiener_lpdf_") => 1e-6,
            _ => 1e-10,
        };
        let ours = try_value_and_grad(|x| eval(name, &ints, x), &reals);
        match (f[3], ours) {
            ("error", Err(_)) => {}
            ("error", Ok(r)) if r.value.is_nan() => {}
            ("error", Ok(r)) => failures.push(format!("{line}: Stan rejects, we give {}", r.value)),
            (_, Err(e)) => failures.push(format!("{line}: we reject: {e}")),
            (v, Ok(r)) => {
                let want: f64 = v.parse().unwrap();
                let grad = list::<f64>(f[4]);
                let grad_ok = grad.is_empty()
                    || (r.gradient.len() == grad.len()
                        && r.gradient
                            .iter()
                            .zip(&grad)
                            .all(|(a, b)| close(*a, *b, rtol)));
                if !close(r.value, want, rtol) || !grad_ok {
                    failures.push(format!("{line}: ours {} {:?}", r.value, r.gradient));
                }
            }
        }
    }
    assert!(count > 90);
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
