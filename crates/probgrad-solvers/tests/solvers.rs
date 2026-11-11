//! Solutions and their derivatives against closed forms, in every differentiation mode.

use probgrad_autodiff::{Real, finite_diff_gradient, grad, hessian, hessian_forward};
use probgrad_solvers::*;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + b.abs())
}

struct Cubic;

impl Equation for Cubic {
    fn eval<S: Real>(&self, x: S, theta: &[S]) -> S {
        x.powi(3) + x - theta[0]
    }
}

#[test]
fn root_derivatives_are_exact_to_second_order() {
    let opts = RootOptions::bracket(-10.0, 10.0);
    let x = solve_root(&Cubic, &[2.0], &opts).unwrap();
    assert!(close(x, 1.0, 1e-15));
    // x^3 + x = theta: dx/dtheta = 1 / (3x^2 + 1), d2x/dtheta2 = -6x / (3x^2 + 1)^3.
    let g = grad(|t| solve_root(&Cubic, t, &opts).unwrap(), &[2.0]);
    assert!(close(g[0], 0.25, 1e-15));
    let h = hessian(|t| solve_root(&Cubic, t, &opts).unwrap(), &[2.0]);
    let hf = hessian_forward(|t| solve_root(&Cubic, t, &opts).unwrap(), &[2.0]);
    assert!(close(h[(0, 0)], -6.0 / 64.0, 1e-14), "{}", h[(0, 0)]);
    assert!(close(hf[(0, 0)], -6.0 / 64.0, 1e-14), "{}", hf[(0, 0)]);
}

/// Kepler's equation E - e sin E = M for the eccentric anomaly E.
struct Kepler;

impl Equation for Kepler {
    fn eval<S: Real>(&self, e_anom: S, theta: &[S]) -> S {
        e_anom - theta[0] * e_anom.sin() - theta[1]
    }
}

#[test]
fn kepler_equation_gradient() {
    let opts = RootOptions::bracket(0.0, 7.0);
    let theta = [0.3, 1.2];
    let e = solve_root(&Kepler, &theta, &opts).unwrap();
    assert!(close(e - 0.3 * e.sin(), 1.2, 1e-15));
    let g = grad(|t| solve_root(&Kepler, t, &opts).unwrap(), &theta);
    // Implicit derivatives: dE/de = sin E / (1 - e cos E), dE/dM = 1 / (1 - e cos E).
    let denom = 1.0 - 0.3 * e.cos();
    assert!(close(g[0], e.sin() / denom, 1e-14));
    assert!(close(g[1], 1.0 / denom, 1e-14));
    assert!(solve_root(&Kepler, &theta, &RootOptions::bracket(2.0, 7.0)).is_err());
}

/// x^2 + y^2 = r2, x - y = d, solved for (x, y) with x, y > 0.
struct Circle;

impl EquationSystem for Circle {
    fn eval<S: Real>(&self, x: &[S], theta: &[S]) -> Vec<S> {
        vec![
            x[0].square() + x[1].square() - theta[0],
            x[0] - x[1] - theta[1],
        ]
    }
}

fn circle_solution(r2: f64, d: f64) -> f64 {
    // x = (d + sqrt(2 r2 - d^2)) / 2
    0.5 * (d + (2.0 * r2 - d * d).sqrt())
}

#[test]
fn system_solution_and_derivatives() {
    let opts = SystemOptions::default();
    let theta = [5.0, 1.0];
    let x = solve_system(&Circle, &[1.0, 1.0], &theta, &opts).unwrap();
    assert!(close(x[0], 2.0, 1e-14) && close(x[1], 1.0, 1e-14));
    let g = grad(
        |t| solve_system(&Circle, &[1.0, 1.0], t, &opts).unwrap()[0],
        &theta,
    );
    let fd = finite_diff_gradient(|t| circle_solution(t[0], t[1]), &theta, 1e-3);
    for k in 0..2 {
        assert!(close(g[k], fd[k], 1e-9), "{k}: {} vs {}", g[k], fd[k]);
    }
    let h = hessian(
        |t| solve_system(&Circle, &[1.0, 1.0], t, &opts).unwrap()[0],
        &theta,
    );
    let hf = hessian_forward(
        |t| solve_system(&Circle, &[1.0, 1.0], t, &opts).unwrap()[0],
        &theta,
    );
    assert!(h.max_abs_diff(&hf) < 1e-12);
    // Second derivative of the closed form in r2: -(2 r2 - d^2)^(-3/2) / 2.
    let q: f64 = 2.0 * 5.0 - 1.0;
    assert!(close(h[(0, 0)], -0.5 * q.powf(-1.5), 1e-12));
}

struct Power;

impl Integrand for Power {
    fn eval<S: Real>(&self, x: S, theta: &[S]) -> S {
        x.pow(theta[0])
    }
}

struct Decay;

impl Integrand for Decay {
    fn eval<S: Real>(&self, x: S, theta: &[S]) -> S {
        (-(theta[0] * x)).exp()
    }
}

struct Gaussian;

impl Integrand for Gaussian {
    fn eval<S: Real>(&self, x: S, theta: &[S]) -> S {
        (-(x.square()) / (theta[0].square() * 2.0)).exp()
    }
}

struct Log;

impl Integrand for Log {
    fn eval<S: Real>(&self, x: S, _theta: &[S]) -> S {
        x.ln()
    }
}

#[test]
fn quadrature_values_and_derivatives() {
    let o = QuadratureOptions::default();
    // Endpoint singularity: int_0^1 x^theta dx = 1 / (theta + 1), here with theta = -0.5.
    let v = integrate(&Power, 0.0, 1.0, &[-0.5], &o).unwrap();
    assert!(close(v, 2.0, 1e-12), "{v}");
    let g = grad(|t| integrate(&Power, 0.0, 1.0, t, &o).unwrap(), &[1.5]);
    assert!(close(g[0], -1.0 / 2.5f64.powi(2), 1e-11), "{}", g[0]);
    // Semi-infinite: int_0^inf exp(-theta x) dx = 1 / theta.
    let v = integrate(&Decay, 0.0, f64::INFINITY, &[2.0], &o).unwrap();
    assert!(close(v, 0.5, 1e-12));
    let h = hessian(
        |t| integrate(&Decay, 0.0, f64::INFINITY, t, &o).unwrap(),
        &[2.0],
    );
    assert!(close(h[(0, 0)], 2.0 / 8.0, 1e-10), "{}", h[(0, 0)]);
    // Doubly infinite: int exp(-x^2 / (2 s^2)) dx = s sqrt(2 pi).
    let g = grad(
        |t| integrate(&Gaussian, f64::NEG_INFINITY, f64::INFINITY, t, &o).unwrap(),
        &[1.3],
    );
    assert!(close(g[0], (2.0 * core::f64::consts::PI).sqrt(), 1e-11));
    // Logarithmic singularity: int_0^1 ln x dx = -1.
    let v: f64 = integrate(&Log, 0.0, 1.0, &[], &o).unwrap();
    assert!(close(v, -1.0, 1e-12));
    assert!(integrate(&Log, 1.0, 0.0, &[0.0], &o).is_err());
}
