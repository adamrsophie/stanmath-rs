//! Forward-mode differentiation with dual numbers.
//!
//! Run with `cargo run --example forward_mode`.

use stanmath_rs::prelude::*;

/// A generic function: works for f64, dual numbers, and reverse-mode variables.
fn f<T: Real>(x: T) -> T {
    x.sin() * x.exp() / (x.square() + 1.0)
}

fn main() {
    let x = 0.8;

    // First derivative: pass the generic function directly.
    let d = derivative(f, x);
    println!("f({x}) = {:.12}", f(x));
    println!("f'({x}) = {d:.12}");

    // The same by hand: seed the tangent with 1.
    let y = f(Dual::variable(x));
    assert_eq!(y.eps, d);

    // Second derivative with nested duals.
    let d2 = stanmath_rs::autodiff::second_derivative(f, x);
    println!("f''({x}) = {d2:.12}");

    // Directional derivative of a function of several variables.
    let g = |v: &[Dual64]| v[0] * v[1].cos() + v[2].ln();
    let dir = [1.0, -0.5, 2.0];
    let dd = stanmath_rs::autodiff::directional_derivative(g, &[1.0, 0.3, 2.0], &dir);
    println!("directional derivative along {dir:?} = {dd:.12}");

    // Jacobian-vector product of a vector function.
    let (values, jv) = jvp(
        |v| vec![v[0] * v[1], v[0].exp() + v[1]],
        &[1.0, 2.0],
        &[1.0, 0.0],
    );
    println!("f(x) = {values:?}, J v = {jv:?}");
}
