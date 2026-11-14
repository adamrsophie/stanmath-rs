//! Values and gradients against Stan Math 5.4.0 for the deprecated functional solvers.
//! `tests/data/stan_reference_solvers.csv` is written by
//! `tools/stan-reference/stan_reference_solvers.cpp`, which documents the systems. Each solution is
//! reduced to `sum_k (1 + k / 10) r_k` over its entries (time major for the ODEs). Both sides solve
//! at tight tolerances, so the comparison allows a relative difference of `1e-7` (`1e-4` for
//! `integrate_ode`, which uses the default tolerances).

use stanmath_rs::prelude::*;

// stan: integrate_ode_rk45((real, array[] real, array[] real, data array[] real, data array[] int) => array[] real, array[] real, real, array[] real, array[] real, array[] real, array[] int, real, real, real)
// stan: integrate_ode_adams((real, array[] real, array[] real, data array[] real, data array[] int) => array[] real, array[] real, real, array[] real, array[] real, array[] real, array[] int, real, real, real)
// stan: integrate_ode_bdf((real, array[] real, array[] real, data array[] real, data array[] int) => array[] real, array[] real, real, array[] real, array[] real, array[] real, array[] int, real, real, real)
// stan: algebra_solver((vector, vector, data array[] real, data array[] int) => vector, vector, vector, array[] real, array[] int, real, real, real)
// stan: algebra_solver_newton((vector, vector, data array[] real, data array[] int) => vector, vector, vector, array[] real, array[] int, real, real, real)
// stan: integrate_ode((real, array[] real, array[] real, data array[] real, data array[] int) => array[] real, array[] real, real, array[] real, array[] real, array[] real, array[] int)
// stan: map_rect((vector, vector, data array[] real, data array[] int) => vector, vector, array[] vector, array[,] real, array[,] int)
// stan: generate_laplace_options(int)
// stan: generate_laplace_options(vector)
// stan: integrate_1d((real, real, array[] real, data array[] real, data array[] int) => real, real, real, array[] real, array[] real, array[] int, real)

// The variadic functions are not in the signature dump; they are compared here as Stan Math
// calls them:
// variadic: ode_rk45
// variadic: ode_ckrk
// variadic: ode_adams
// variadic: ode_bdf
// variadic: ode_rk45_tol
// variadic: ode_ckrk_tol
// variadic: ode_adams_tol
// variadic: ode_bdf_tol
// variadic: ode_adjoint_tol_ctl
// variadic: dae
// variadic: dae_tol
// variadic: solve_newton
// variadic: solve_powell
// variadic: solve_newton_tol
// variadic: solve_powell_tol
// variadic: reduce_sum
// variadic: reduce_sum_static
// variadic: laplace_marginal
// variadic: laplace_marginal_tol
// variadic: laplace_marginal_poisson_log_lpmf
// variadic: laplace_marginal_tol_poisson_log_lpmf
// variadic: laplace_marginal_poisson_log_lupmf
// variadic: laplace_marginal_bernoulli_logit_lpmf
// variadic: laplace_marginal_tol_bernoulli_logit_lpmf
// variadic: laplace_marginal_bernoulli_logit_lupmf
// variadic: laplace_marginal_neg_binomial_2_log_lpmf
// variadic: laplace_marginal_tol_neg_binomial_2_log_lpmf
// variadic: laplace_marginal_neg_binomial_2_log_lupmf

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

/// The variadic Lotka Volterra system, as [`Lv`].
struct LvV;

impl OdeSystem for LvV {
    fn eval<S: Real>(&self, _t: S, y: &[S], th: &[S]) -> Vec<S> {
        Lv.eval(_t, y, th, &[], &[])
    }
}

/// The variadic algebraic system, as [`Sys`].
struct SysV;

impl EquationSystem for SysV {
    fn eval<S: Real>(&self, x: &[S], th: &[S]) -> Vec<S> {
        Sys.eval(x, th, &[], &[])
    }
}

/// DAE residual `y0' + k y0 = 0`, `y1 - 2 y0 = 0`.
struct Res;

impl DaeSystem for Res {
    fn eval<S: Real>(&self, _t: S, y: &[S], yp: &[S], th: &[S]) -> Vec<S> {
        vec![yp[0] + th[0] * y[0], y[1] - y[0] * 2.0]
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

/// The `map_rect` job, as a type for the threaded form.
struct Job;

impl MapRectJob for Job {
    fn eval<S: Real>(&self, phi: &[S], t: &[S], xr: &[f64], xi: &[i64]) -> Result<Vec<S>> {
        Ok(vec![
            phi[0] * t[0] + xr[0],
            phi[1] * t[1].exp() * xi[0] as f64,
        ])
    }
}

/// The `reduce_sum` partial sum `normal_lpdf(slice | mu, sigma)`, as a type for the threaded form.
struct Part;

impl PartialSum<f64> for Part {
    fn eval<S: Real>(&self, s: &[f64], _: usize, _: usize, p: &[S]) -> Result<S> {
        normal_lpdf(s, p[0], p[1])
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
        "integrate_ode" => integrate_ode(&Lv, y0, 0.0, &ts, th, &[], &[], None).map(flat),
        "map_rect" => map_rect(
            |phi: &[Var<'t>], t: &[Var<'t>], xr: &[f64], xi: &[i64]| {
                Ok(vec![
                    phi[0] * t[0] + xr[0],
                    phi[1] * exp(t[1]) * xi[0] as f64,
                ])
            },
            &x[0..2],
            &[x[2..4].to_vec(), x[4..6].to_vec()],
            &[vec![0.5], vec![0.25]],
            &[vec![2], vec![3]],
        )
        .map(wsum),
        "ode_rk45" => ode_rk45(&LvV, y0, 0.0, &ts, th).map(flat),
        "ode_ckrk" => ode_ckrk(&LvV, y0, 0.0, &ts, th).map(flat),
        "ode_adams" => ode_adams(&LvV, y0, 0.0, &ts, th).map(flat),
        "ode_bdf" => ode_bdf(&LvV, y0, 0.0, &ts, th).map(flat),
        "ode_rk45_tol" => ode_rk45_tol(&LvV, y0, 0.0, &ts, th, &ode).map(flat),
        "ode_ckrk_tol" => ode_ckrk_tol(&LvV, y0, 0.0, &ts, th, &ode).map(flat),
        "ode_adams_tol" => ode_adams_tol(&LvV, y0, 0.0, &ts, th, &ode).map(flat),
        "ode_bdf_tol" => ode_bdf_tol(&LvV, y0, 0.0, &ts, th, &ode).map(flat),
        "ode_adjoint_tol_ctl" => {
            let o = AdjointOptions {
                rel_tol_forward: 1e-10,
                abs_tol_forward: vec![1e-10; 2],
                rel_tol_backward: 1e-10,
                abs_tol_backward: vec![1e-10; 2],
                rel_tol_quadrature: 1e-10,
                abs_tol_quadrature: 1e-10,
                max_num_steps: 1_000_000,
                num_steps_between_checkpoints: 150,
                interpolation_polynomial: 1,
                solver_forward: 2,
                solver_backward: 2,
            };
            ode_adjoint_tol_ctl(&LvV, y0, 0.0, &ts, &o, th).map(flat)
        }
        "dae" | "dae_tol" => {
            let yy0 = [Var::constant(1.0), Var::constant(2.0)];
            let yp0 = [-x[0], -(x[0] * 2.0)];
            if name == "dae" {
                dae(&Res, &yy0, &yp0, 0.0, &ts, &x[..1]).map(flat)
            } else {
                dae_tol(&Res, &yy0, &yp0, 0.0, &ts, &x[..1], &ode).map(flat)
            }
        }
        "solve_newton" => solve_newton(&SysV, &[1.0, 1.0], x).map(wsum),
        "solve_powell" => solve_powell(&SysV, &[1.0, 1.0], x).map(wsum),
        "solve_newton_tol" => solve_newton_tol(&SysV, &[1.0, 1.0], x, &sys).map(wsum),
        "solve_powell_tol" => solve_powell_tol(&SysV, &[1.0, 1.0], x, &sys).map(wsum),
        "reduce_sum" | "reduce_sum_static" => {
            let s = [0.1, 0.4, -0.3, 1.2, 0.8];
            let part = |sl: &[f64], _: usize, _: usize| normal_lpdf(sl, x[0], x[1]);
            if name == "reduce_sum" {
                reduce_sum(part, &s, 2)
            } else {
                reduce_sum_static(part, &s, 2)
            }
        }
        // The threaded forms are compared with the same Stan rows as the serial ones.
        "map_rect_parallel" => map_rect_parallel(
            Job,
            &x[0..2],
            &[x[2..4].to_vec(), x[4..6].to_vec()],
            &[vec![0.5], vec![0.25]],
            &[vec![2], vec![3]],
        )
        .map(wsum),
        "reduce_sum_parallel" => reduce_sum_parallel(Part, &[0.1, 0.4, -0.3, 1.2, 0.8], 2, x),
        n if n.starts_with("laplace_marginal") => laplace(n, x),
        "integrate_1d" => integrate_1d(&Damped, Var::constant(0.0), x[0], &x[1..3], 1e-10),
        other => panic!("no dispatch for {other}"),
    }
}

/// The Laplace marginals: `phi = x[0..2]`, then the offset (or the dispersion for the negative
/// binomial). Stan's `y_index` is one based, so `[1, 2, 3]` there is `[0, 1, 2]` here.
fn laplace<'t>(name: &str, x: &[Var<'t>]) -> Result<Var<'t>> {
    let (yp, yb, idx) = ([1_i64, 0, 3], [1_i64, 0, 1], [0_usize, 1, 2]);
    let o = generate_laplace_options(3);
    let (phi, m) = (&x[..2], [x[2]; 3]);
    let zero = [Var::constant(0.0); 3];
    match name {
        "laplace_marginal_poisson_log_lpmf" => {
            laplace_marginal_poisson_log_lpmf(&yp, &idx, &m, &Kf, phi)
        }
        "laplace_marginal_poisson_log_lupmf" => {
            laplace_marginal_poisson_log_lupmf(&yp, &idx, &m, &Kf, phi)
        }
        "laplace_marginal_tol_poisson_log_lpmf" => {
            laplace_marginal_tol_poisson_log_lpmf(&yp, &idx, &m, &Kf, phi, &o)
        }
        "laplace_marginal_bernoulli_logit_lpmf" => {
            laplace_marginal_bernoulli_logit_lpmf(&yb, &idx, &m, &Kf, phi)
        }
        "laplace_marginal_bernoulli_logit_lupmf" => {
            laplace_marginal_bernoulli_logit_lupmf(&yb, &idx, &m, &Kf, phi)
        }
        "laplace_marginal_tol_bernoulli_logit_lpmf" => {
            laplace_marginal_tol_bernoulli_logit_lpmf(&yb, &idx, &m, &Kf, phi, &o)
        }
        "laplace_marginal_neg_binomial_2_log_lpmf" => {
            laplace_marginal_neg_binomial_2_log_lpmf(&yp, &idx, x[2], &zero, &Kf, phi)
        }
        "laplace_marginal_neg_binomial_2_log_lupmf" => {
            laplace_marginal_neg_binomial_2_log_lupmf(&yp, &idx, x[2], &zero, &Kf, phi)
        }
        "laplace_marginal_tol_neg_binomial_2_log_lpmf" => {
            laplace_marginal_tol_neg_binomial_2_log_lpmf(&yp, &idx, x[2], &zero, &Kf, phi, &o)
        }
        "laplace_marginal" => laplace_marginal(&Ll, &x[2..], &Kf, phi),
        "laplace_marginal_tol" => laplace_marginal_tol(&Ll, &x[2..], &Kf, phi, &o),
        other => panic!("no dispatch for {other}"),
    }
}

fn list(s: &str) -> Vec<f64> {
    s.split(';').map(|v| v.parse().unwrap()).collect()
}

/// `integrate_ode` and the variadic solvers without `_tol` run at the default tolerances (`1e-6`),
/// where the two step size controllers give sensitivities that differ by up to `1e-4`; the others
/// run at `1e-10`.
fn close(name: &str, a: f64, b: f64) -> bool {
    let rtol = match name {
        "integrate_ode" | "ode_rk45" | "ode_ckrk" | "ode_adams" | "ode_bdf" => 1e-4,
        "solve_newton" | "solve_powell" => 1e-5,
        _ => 1e-7,
    };
    (a - b).abs() <= rtol * b.abs().max(1.0)
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
        if let Some(rest) = line.strip_prefix("generate_laplace_options,") {
            laplace_options(rest, &mut failures);
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        let threaded = match f[0] {
            "map_rect" => Some("map_rect_parallel"),
            "reduce_sum" => Some("reduce_sum_parallel"),
            _ => None,
        };
        for name in std::iter::once(f[0]).chain(threaded) {
            check(name, &f, &mut failures);
        }
    }
    assert_eq!(count, 38);
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

fn check(name: &str, f: &[&str], failures: &mut Vec<String>) {
    match try_value_and_grad(|x| eval(name, x), &list(f[2])) {
        Err(e) => failures.push(format!("{name}: we reject: {e}")),
        Ok(r) => {
            let want: f64 = f[3].parse().unwrap();
            let grad = list(f[4]);
            if !close(f[0], r.value, want)
                || r.gradient.len() != grad.len()
                || r.gradient
                    .iter()
                    .zip(&grad)
                    .any(|(a, b)| !close(f[0], *a, *b))
            {
                failures.push(format!(
                    "{name}: ours {} {:?}, Stan {want} {grad:?}",
                    r.value, r.gradient
                ));
            }
        }
    }
}

/// A `generate_laplace_options` row: `dimension,,weighted theta_init,tolerance;max_num_steps;
/// solver;max_steps_line_search;allow_fallback`. Dimension 3 is the int form, 2 the vector
/// form with `[1.5, 0.2]`.
fn laplace_options(rest: &str, failures: &mut Vec<String>) {
    let f: Vec<&str> = rest.split(',').collect();
    let o = if f[0] == "3" {
        generate_laplace_options(3)
    } else {
        generate_laplace_options_from(&[1.5, 0.2])
    };
    let init = o.theta_init.clone().unwrap_or_default();
    let got = vec![
        init.iter()
            .enumerate()
            .map(|(k, v)| v * (1.0 + 0.1 * k as f64))
            .sum::<f64>(),
        o.tolerance,
        o.max_num_steps as f64,
        f64::from(o.solver),
        o.max_steps_line_search as f64,
        f64::from(u8::from(o.allow_fallback)),
    ];
    let mut want = vec![f[2].parse::<f64>().unwrap()];
    want.extend(list(f[3]));
    if init.len().to_string() != f[0] || got != want {
        failures.push(format!(
            "generate_laplace_options({}): {got:?} vs {want:?}",
            f[0]
        ));
    }
}
