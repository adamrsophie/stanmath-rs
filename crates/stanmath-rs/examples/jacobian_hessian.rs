//! Jacobians, Hessians, and Hessian-vector products.
//!
//! Run with `cargo run --example jacobian_hessian`.

use stanmath_rs::prelude::*;

/// Rosenbrock's function, written once and reused by every driver.
fn rosenbrock<T: Real>(x: &[T]) -> T {
    (-x[0] + 1.0).square() + (x[1] - x[0].square()).square() * 100.0
}

/// A map from R^2 to R^3.
fn polar_features<T: Real>(x: &[T]) -> Vec<T> {
    vec![x[0].hypot(x[1]), x[1].atan2(x[0]), x[0] * x[1]]
}

fn main() {
    let x = [-1.2, 1.0];

    let h = hessian(|v| rosenbrock(v), &x);
    println!("Hessian of Rosenbrock at {x:?}:");
    for row in h.to_rows() {
        println!("  {row:?}");
    }

    let hv = hessian_vector_product(|v| rosenbrock(v), &x, &[1.0, 0.0]);
    println!("H e1 = {hv:?} (first column, one evaluation)");

    let (values, jac) = jacobian(|v| polar_features(v), &[3.0, 4.0]);
    println!("f(3, 4) = {values:?}");
    println!("Jacobian (reverse mode):");
    for row in jac.to_rows() {
        println!("  {row:?}");
    }
    let (_, jac_fwd) = stanmath_rs::autodiff::jacobian_forward(polar_features, &[3.0, 4.0]);
    println!(
        "max difference to forward mode: {:e}",
        jac.max_abs_diff(&jac_fwd)
    );
}
