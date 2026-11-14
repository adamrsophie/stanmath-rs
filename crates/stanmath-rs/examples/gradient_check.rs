//! Checking a gradient against finite differences.
//!
//! Run with `cargo run --example gradient_check`.

use stanmath_rs::prelude::*;

fn model<T: Real>(p: &[T]) -> T {
    let mu = p[0];
    let log_sigma = p[1];
    let sigma = log_sigma.exp();
    let data = [0.3, -1.2, 2.2, 0.9];
    let mut lp = -(mu.square()) * 0.5 - log_sigma.square() * 0.5;
    for y in data {
        let z = (-mu + y) / sigma;
        lp += -(z * z) * 0.5 - log_sigma;
    }
    lp
}

fn main() {
    let check = check_gradient(|p| model(p), &[0.4, -0.3]);
    println!("value       {:.12}", check.value);
    println!("autodiff    {:?}", check.autodiff);
    println!("finite diff {:?}", check.finite_diff);
    println!("max abs err {:e}", check.max_abs_error());
    assert!(check.passes(1e-8, 1e-10), "gradient check failed");
    println!("gradient check passed");
}
