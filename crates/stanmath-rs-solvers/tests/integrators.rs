//! ODE, DAE, quadrature, cubature, and Powell solutions against closed forms, with their
//! derivatives with respect to the parameters.

use stanmath_rs_autodiff::{Real, grad};
use stanmath_rs_solvers::*;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + b.abs())
}

/// `y'' = -omega^2 y` as a first order system; `y(0) = 1, y'(0) = 0` gives `y = cos(omega t)`.
struct Oscillator;

impl OdeSystem for Oscillator {
    fn eval<S: Real>(&self, _t: S, y: &[S], theta: &[S]) -> Vec<S> {
        vec![y[1], -(theta[0] * theta[0] * y[0])]
    }
}

const TS: [f64; 3] = [0.5, 1.0, 2.0];

type Solve = fn(&[f64], &[f64]) -> Vec<Vec<f64>>;

#[test]
fn oscillator_values_for_every_method() {
    let solvers: [(&str, Solve); 4] = [
        ("ode_rk45", |y0, th| {
            ode_rk45(&Oscillator, y0, 0.0, &TS, th).unwrap()
        }),
        ("ode_ckrk", |y0, th| {
            ode_ckrk(&Oscillator, y0, 0.0, &TS, th).unwrap()
        }),
        ("ode_adams", |y0, th| {
            ode_adams(&Oscillator, y0, 0.0, &TS, th).unwrap()
        }),
        ("ode_bdf", |y0, th| {
            ode_bdf(&Oscillator, y0, 0.0, &TS, th).unwrap()
        }),
    ];
    let omega = 1.3;
    for (name, solve) in solvers {
        let y = solve(&[1.0, 0.0], &[omega]);
        assert_eq!(y.len(), TS.len(), "{name}");
        for (yt, &t) in y.iter().zip(&TS) {
            let (pos, vel) = ((omega * t).cos(), -omega * (omega * t).sin());
            assert!(close(yt[0], pos, 1e-4), "{name} t={t}: {} vs {pos}", yt[0]);
            assert!(close(yt[1], vel, 1e-4), "{name} t={t}: {} vs {vel}", yt[1]);
        }
    }
}

macro_rules! oscillator_gradient {
    ($solver:ident) => {{
        let omega = 1.3;
        let t = 2.0;
        let g = grad(
            |th| {
                let y0 = [Real::from_f64(1.0), Real::from_f64(0.0)];
                $solver(&Oscillator, &y0, 0.0, &[t], th).unwrap()[0][0]
            },
            &[omega],
        );
        // d cos(omega t) / d omega = -t sin(omega t).
        let want = -t * (omega * t).sin();
        assert!(
            close(g[0], want, 1e-4),
            "{}: {} vs {want}",
            stringify!($solver),
            g[0]
        );
    }};
}

#[test]
fn oscillator_sensitivities() {
    oscillator_gradient!(ode_rk45);
    oscillator_gradient!(ode_ckrk);
    oscillator_gradient!(ode_adams);
    oscillator_gradient!(ode_bdf);
}

/// `y' = -lambda (y - sin t) + cos t` with `y(0) = 1`: `y = sin t + exp(-lambda t)`, stiff for
/// large `lambda`.
struct Stiff;

impl OdeSystem for Stiff {
    fn eval<S: Real>(&self, t: S, y: &[S], theta: &[S]) -> Vec<S> {
        vec![-(theta[0] * (y[0] - t.sin())) + t.cos()]
    }
}

#[test]
fn stiff_problem_with_bdf() {
    let lambda = 1000.0;
    let ts = [0.001, 0.1, 1.0, 3.0];
    let y = ode_bdf(&Stiff, &[1.0], 0.0, &ts, &[lambda]).unwrap();
    for (yt, &t) in y.iter().zip(&ts) {
        let want = t.sin() + (-lambda * t).exp();
        assert!(close(yt[0], want, 1e-4), "t={t}: {} vs {want}", yt[0]);
    }
}

/// `y0' = -theta y0` with the algebraic constraint `y1 = 2 y0`.
struct Constrained;

impl DaeSystem for Constrained {
    fn eval<S: Real>(&self, _t: S, y: &[S], yp: &[S], theta: &[S]) -> Vec<S> {
        vec![yp[0] + theta[0] * y[0], y[1] - y[0] * 2.0]
    }
}

#[test]
fn dae_value_and_sensitivity() {
    let theta = 0.8;
    let y = dae(
        &Constrained,
        &[1.0, 2.0],
        &[-theta, -2.0 * theta],
        0.0,
        &[1.0, 2.0],
        &[theta],
    )
    .unwrap();
    for (yt, t) in y.iter().zip([1.0, 2.0]) {
        let e = (-theta * t).exp();
        assert!(close(yt[0], e, 1e-4), "t={t}: {}", yt[0]);
        assert!(close(yt[1], 2.0 * e, 1e-4), "t={t}: {}", yt[1]);
    }
    let g = grad(
        |th| {
            let y0 = [Real::from_f64(1.0), Real::from_f64(2.0)];
            let yp0 = [-th[0], -(th[0] * 2.0)];
            dae(&Constrained, &y0, &yp0, 0.0, &[1.0], th).unwrap()[0][1]
        },
        &[theta],
    );
    // d (2 exp(-theta)) / d theta = -2 exp(-theta).
    let want = -2.0 * (-theta).exp();
    assert!(close(g[0], want, 1e-4), "{} vs {want}", g[0]);
}

/// `x^theta`.
struct Power;

impl Integrand for Power {
    fn eval<S: Real>(&self, x: S, theta: &[S]) -> S {
        x.pow(theta[0])
    }
}

/// `exp(-theta x)`.
struct Decay;

impl Integrand for Decay {
    fn eval<S: Real>(&self, x: S, theta: &[S]) -> S {
        (-(theta[0] * x)).exp()
    }
}

/// `exp(-x^2)`.
struct Bell;

impl Integrand for Bell {
    fn eval<S: Real>(&self, x: S, _theta: &[S]) -> S {
        (-(x * x)).exp()
    }
}

#[test]
fn gauss_kronrod_values_and_derivatives() {
    // int_0^b x^a dx = b^(a + 1) / (a + 1), with derivatives in a and in the limit b.
    let (a, b) = (1.5, 2.0);
    let v = integrate_1d_gauss_kronrod(&Power, 0.0, b, &[a], 1e-10).unwrap();
    assert!(close(v, b.powf(a + 1.0) / (a + 1.0), 1e-9), "{v}");
    let g = grad(
        |p| integrate_1d_gauss_kronrod(&Power, Real::from_f64(0.0), p[1], &p[..1], 1e-10).unwrap(),
        &[a, b],
    );
    let da = b.powf(a + 1.0) * (b.ln() / (a + 1.0) - 1.0 / ((a + 1.0) * (a + 1.0)));
    assert!(close(g[0], da, 1e-8), "{} vs {da}", g[0]);
    assert!(close(g[1], b.powf(a), 1e-9), "{}", g[1]);
    // int_0^inf exp(-lambda x) dx = 1 / lambda.
    let lambda = 2.5;
    let g = grad(
        |p| {
            integrate_1d_gauss_kronrod(
                &Decay,
                Real::from_f64(0.0),
                Real::from_f64(f64::INFINITY),
                p,
                1e-10,
            )
            .unwrap()
        },
        &[lambda],
    );
    assert!(close(g[0], -1.0 / (lambda * lambda), 1e-8), "{}", g[0]);
    assert!(integrate_1d_gauss_kronrod(&Power, 1.0, 0.0, &[a], 1e-8).is_err());
}

#[test]
fn double_exponential_values_and_derivatives() {
    let inf = f64::INFINITY;
    let v = integrate_1d_double_exponential(&Bell, -inf, inf, &[], 1e-10).unwrap();
    assert!(close(v, core::f64::consts::PI.sqrt(), 1e-9), "{v}");
    let v = integrate_1d_double_exponential(&Power, 0.0, 1.0, &[-0.5], 1e-10).unwrap();
    assert!(close(v, 2.0, 1e-8), "{v}");
    let lambda = 0.7;
    let g = grad(
        |p| {
            integrate_1d_double_exponential(
                &Decay,
                Real::from_f64(0.0),
                Real::from_f64(inf),
                p,
                1e-10,
            )
            .unwrap()
        },
        &[lambda],
    );
    assert!(close(g[0], -1.0 / (lambda * lambda), 1e-8), "{}", g[0]);
}

/// `exp(theta (x0 + x1))` on the unit square.
struct Plane;

impl CubatureIntegrand for Plane {
    fn eval<S: Real>(&self, x: &[S], theta: &[S]) -> S {
        (theta[0] * (x[0] + x[1])).exp()
    }
}

#[test]
fn hcubature_value_and_derivative() {
    let theta = 0.7;
    let (lo, hi) = ([0.0, 0.0], [1.0, 1.0]);
    // int exp(theta (x + y)) over the unit square = q^2 with q = (e^theta - 1) / theta.
    let q = (theta.exp() - 1.0) / theta;
    let v = hcubature(&Plane, &[theta], 2, &lo, &hi, 100_000, 1e-12, 1e-10).unwrap();
    assert!(close(v, q * q, 1e-8), "{v} vs {}", q * q);
    let g = grad(
        |p| hcubature(&Plane, p, 2, &lo, &hi, 100_000, 1e-12, 1e-10).unwrap(),
        &[theta],
    );
    let dq = (theta * theta.exp() - (theta.exp() - 1.0)) / (theta * theta);
    assert!(
        close(g[0], 2.0 * q * dq, 1e-7),
        "{} vs {}",
        g[0],
        2.0 * q * dq
    );
    assert!(hcubature(&Plane, &[theta], 2, &lo, &[1.0], 1000, 1e-8, 1e-8).is_err());
}

/// `x^2 + y^2 = r2`, `x - y = d`.
struct Circle;

impl EquationSystem for Circle {
    fn eval<S: Real>(&self, x: &[S], theta: &[S]) -> Vec<S> {
        vec![x[0] * x[0] + x[1] * x[1] - theta[0], x[0] - x[1] - theta[1]]
    }
}

#[test]
fn powell_solution_and_implicit_derivatives() {
    let (r2, d) = (2.0, 0.5);
    let s = (2.0 * r2 - d * d).sqrt();
    let x = solve_powell(&Circle, &[1.0, 1.0], &[r2, d]).unwrap();
    assert!(close(x[0], 0.5 * (d + s), 1e-9), "{}", x[0]);
    assert!(close(x[1], 0.5 * (s - d), 1e-9), "{}", x[1]);
    let g = grad(
        |p| solve_powell(&Circle, &[1.0, 1.0], p).unwrap()[0],
        &[r2, d],
    );
    // x = (d + s) / 2 with s = sqrt(2 r2 - d^2).
    assert!(close(g[0], 0.5 / s, 1e-8), "{}", g[0]);
    assert!(close(g[1], 0.5 * (1.0 - d / s), 1e-8), "{}", g[1]);
}
