//! Values and gradients against Stan Math 5.4.0 for the deprecated functional solvers.
//! `tests/data/stan_reference_solvers.csv` is written by
//! `tools/stan-reference/stan_reference_solvers.cpp`, which documents the systems. Each solution is
//! reduced to `sum_k (1 + k / 10) r_k` over its entries (time major for the ODEs). Both sides solve
//! at tight tolerances, so the comparison allows a relative difference of `1e-7`.

use stanmath_rs::prelude::*;

// stan: integrate_ode_rk45((real, array[] real, array[] real, data array[] real, data array[] int) => array[] real, array[] real, real, array[] real, array[] real, array[] real, array[] int, real, real, real)
// stan: integrate_ode_adams((real, array[] real, array[] real, data array[] real, data array[] int) => array[] real, array[] real, real, array[] real, array[] real, array[] real, array[] int, real, real, real)
// stan: integrate_ode_bdf((real, array[] real, array[] real, data array[] real, data array[] int) => array[] real, array[] real, real, array[] real, array[] real, array[] real, array[] int, real, real, real)
// stan: algebra_solver((vector, vector, data array[] real, data array[] int) => vector, vector, vector, array[] real, array[] int, real, real, real)
// stan: algebra_solver_newton((vector, vector, data array[] real, data array[] int) => vector, vector, vector, array[] real, array[] int, real, real, real)
// stan: integrate_1d((real, real, array[] real, data array[] real, data array[] int) => real, real, real, array[] real, array[] real, array[] int, real)

/// Lotka Volterra: `y0' = t0 y0 - t1 y0 y1`, `y1' = -t2 y1 + t1 y0 y1`.
struct Lv;

impl OdeSystemData for Lv {
    fn eval<S: Real>(&self, _t: S, y: &[S], th: &[S], _x_r: &[f64], _x_i: &[i64]) -> Vec<S> {
        vec![
            th[0] * y[0] - th[1] * y[0] * y[1],
            -(th[2] * y[1]) + th[1] * y[0] * y[1],
        ]
    }
}

/// `x0^2 + x1 - t0 = 0`, `x0 - x1^3 + t1 = 0`.
struct Sys;

impl AlgebraSystemData for Sys {
    fn eval<S: Real>(&self, x: &[S], th: &[S], _x_r: &[f64], _x_i: &[i64]) -> Vec<S> {
        vec![
            x[0] * x[0] + x[1] - th[0],
            x[0] - x[1] * x[1] * x[1] + th[1],
        ]
    }
}

/// `exp(-t0 x) cos(t1 x)`.
struct Damped;

impl Integrand for Damped {
    fn eval<S: Real>(&self, x: S, th: &[S]) -> S {
        exp(-(th[0] * x)) * cos(th[1] * x)
    }
}

fn wsum<'t>(e: impl IntoIterator<Item = Var<'t>>) -> Var<'t> {
    e.into_iter()
        .enumerate()
        .map(|(k, v)| v * (1.0 + 0.1 * k as f64))
        .sum()
}

fn eval<'t>(name: &str, x: &[Var<'t>]) -> Result<Var<'t>> {
    let ts = [0.5, 1.0, 2.0];
    let ode = OdeOptions {
        relative_tolerance: 1e-10,
        absolute_tolerance: 1e-10,
        max_num_steps: 1_000_000,
    };
    let sys = SystemOptions {
        tolerance: 1e-12,
        function_tolerance: 1e-12,
        max_iterations: 1000,
    };
    let (y0, th) = (&x[..2.min(x.len())], &x[2.min(x.len())..]);
    let flat = |r: Vec<Vec<Var<'t>>>| wsum(r.into_iter().flatten());
    match name {
        "integrate_ode_rk45" => {
            integrate_ode_rk45(&Lv, y0, 0.0, &ts, th, &[], &[], Some(&ode)).map(flat)
        }
        "integrate_ode_adams" => {
            integrate_ode_adams(&Lv, y0, 0.0, &ts, th, &[], &[], Some(&ode)).map(flat)
        }
        "integrate_ode_bdf" => {
            integrate_ode_bdf(&Lv, y0, 0.0, &ts, th, &[], &[], Some(&ode)).map(flat)
        }
        "algebra_solver" => algebra_solver(&Sys, &[1.0, 1.0], x, &[], &[], Some(&sys)).map(wsum),
        "algebra_solver_newton" => {
            algebra_solver_newton(&Sys, &[1.0, 1.0], x, &[], &[], Some(&sys)).map(wsum)
        }
        "integrate_1d" => integrate_1d(&Damped, Var::constant(0.0), x[0], &x[1..3], 1e-10),
        other => panic!("no dispatch for {other}"),
    }
}

fn list(s: &str) -> Vec<f64> {
    s.split(';').map(|v| v.parse().unwrap()).collect()
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-7 * b.abs().max(1.0)
}

#[test]
fn solvers_match_stan() {
    let path = format!(
        "{}/tests/data/stan_reference_solvers.csv",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).unwrap();
    let mut failures = Vec::new();
    let mut count = 0;
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        count += 1;
        let f: Vec<&str> = line.split(',').collect();
        match try_value_and_grad(|x| eval(f[0], x), &list(f[2])) {
            Err(e) => failures.push(format!("{}: we reject: {e}", f[0])),
            Ok(r) => {
                let want: f64 = f[3].parse().unwrap();
                let grad = list(f[4]);
                if !close(r.value, want)
                    || r.gradient.len() != grad.len()
                    || r.gradient.iter().zip(&grad).any(|(a, b)| !close(*a, *b))
                {
                    failures.push(format!(
                        "{}: ours {} {:?}, Stan {want} {grad:?}",
                        f[0], r.value, r.gradient
                    ));
                }
            }
        }
    }
    assert_eq!(count, 6);
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
