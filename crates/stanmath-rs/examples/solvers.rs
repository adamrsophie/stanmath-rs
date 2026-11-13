//! Differentiating through solvers: a root, a nonlinear system, and an integral.
//!
//! Run with `cargo run --example solvers`.

use stanmath_rs::prelude::*;

/// Kepler's equation E - e sin(E) = M for the eccentric anomaly E.
struct Kepler;

impl Equation for Kepler {
    fn eval<S: Real>(&self, e_anom: S, theta: &[S]) -> S {
        e_anom - theta[0] * e_anom.sin() - theta[1]
    }
}

/// Intersection of the circle x^2 + y^2 = theta0 with the line x = theta1 y.
struct Intersection;

impl EquationSystem for Intersection {
    fn eval<S: Real>(&self, p: &[S], theta: &[S]) -> Vec<S> {
        vec![
            p[0].square() + p[1].square() - theta[0],
            p[0] - theta[1] * p[1],
        ]
    }
}

/// A Gaussian bump with scale theta0, integrated over the whole line.
struct Bump;

impl Integrand for Bump {
    fn eval<S: Real>(&self, x: S, theta: &[S]) -> S {
        (-(x.square()) / (theta[0].square() * 2.0)).exp()
    }
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let opts = RootOptions::bracket(0.0, 7.0);
    let r = try_value_and_grad(|t| solve_root(&Kepler, t, &opts), &[0.3, 1.2])?;
    println!(
        "eccentric anomaly {:.12}, gradient (e, M) {:?}",
        r.value, r.gradient
    );
    let h = hessian(|t| solve_root(&Kepler, t, &opts).unwrap(), &[0.3, 1.2]);
    println!("Hessian of the root {:?}", h.to_rows());

    let sys = SystemOptions::default();
    let (point, jac) = jacobian(
        |t| solve_system(&Intersection, &[1.0, 1.0], t, &sys).unwrap(),
        &[5.0, 2.0],
    );
    println!("intersection {point:.12?} (exact [2, 1])");
    println!("sensitivity to (theta0, theta1) {:?}", jac.to_rows());

    let q = QuadratureOptions::default();
    let r = try_value_and_grad(
        |t| integrate(&Bump, f64::NEG_INFINITY, f64::INFINITY, t, &q),
        &[1.3],
    )?;
    println!(
        "integral {:.12} (exact {:.12}), derivative {:.12} (exact {:.12})",
        r.value,
        1.3 * (2.0 * std::f64::consts::PI).sqrt(),
        r.gradient[0],
        (2.0 * std::f64::consts::PI).sqrt()
    );
    Ok(())
}
