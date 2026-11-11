//! Logistic regression: log posterior, gradient, and a few steps of gradient ascent.
//!
//! Run with `cargo run --example logistic_regression`.

use probgrad::prelude::*;
use rand::SeedableRng;

/// Log posterior of logistic regression with normal(0, 2.5) priors on the coefficients.
fn log_posterior<T: Real>(beta: &[T], x: &[Vec<f64>], y: &[i64]) -> Result<T> {
    let mut eta = Vec::with_capacity(x.len());
    for row in x {
        let consts: Vec<T> = row.iter().map(|&v| T::from_f64(v)).collect();
        eta.push(dot_product(&consts, beta));
    }
    Ok(bernoulli_logit_lpmf(y, &eta)? + normal_lpdf(beta, 0.0, 2.5)?)
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Simulate data with an intercept and two covariates.
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let truth = [-0.5, 1.2, -2.0];
    let n = 500;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let row = vec![
            1.0,
            normal_rng(0.0, 1.0, &mut rng)?,
            normal_rng(0.0, 1.0, &mut rng)?,
        ];
        let eta: f64 = row.iter().zip(&truth).map(|(a, b)| a * b).sum();
        y.push(bernoulli_logit_rng(eta, &mut rng)?);
        x.push(row);
    }

    // Gradient ascent to the posterior mode, reusing one tape.
    let mut workspace = GradientWorkspace::new();
    let mut beta = [0.0; 3];
    let mut gradient = [0.0; 3];
    for step in 0..200 {
        let lp = workspace.try_value_and_grad_into(
            |b| log_posterior(b, &x, &y),
            &beta,
            &mut gradient,
        )?;
        for (b, g) in beta.iter_mut().zip(&gradient) {
            *b += 0.002 * g;
        }
        if step % 50 == 0 {
            println!("step {step:3}: log posterior {lp:.4}");
        }
    }
    println!("posterior mode {beta:.3?} (truth {truth:?})");

    // Laplace approximation: standard errors from the Hessian at the mode.
    let h = hessian(|b| log_posterior(b, &x, &y).unwrap(), &beta);
    let diag: Vec<f64> = (0..3).map(|i| (-1.0 / h[(i, i)]).sqrt()).collect();
    println!("approximate marginal standard deviations (diagonal only) {diag:.3?}");
    Ok(())
}
