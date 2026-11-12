//! Gradient of a normal log likelihood with priors, the example from the Stan Math paper.
//!
//! Run with `cargo run --example normal_log_likelihood`.

use stanmath_rs::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Single observation, as in Carpenter et al. (2015), section 2.1.
    let observation = 1.3;
    let r = try_value_and_grad(|p| normal_lpdf(observation, p[0], p[1]), &[0.5, 1.2])?;
    println!("log density {:.15}", r.value);
    println!("gradient    {:?}", r.gradient);

    // A vectorized likelihood with priors. `sigma` is sampled on the log scale, so the model adds
    // the log Jacobian of exp.
    let y = [2.3, 1.9, 3.1, 2.8, 0.7, 2.2, 1.5, 3.4];
    fn log_density<T: Real>(p: &[T], y: &[f64]) -> Result<T> {
        let mu = p[0];
        let log_sigma = p[1];
        let sigma = log_sigma.exp();
        Ok(normal_lpdf(y, mu, sigma)?
            + normal_lpdf(mu, 0.0, 10.0)?
            + exponential_lpdf(sigma, 1.0)?
            + log_sigma)
    }
    let theta = [2.0, 0.0];
    let r = try_value_and_grad(|p| log_density(p, &y), &theta)?;
    println!(
        "posterior log density {:.10} gradient {:?}",
        r.value, r.gradient
    );

    // The same function in forward over reverse mode gives the Hessian.
    let h = hessian(|p| log_density(p, &y).unwrap(), &theta);
    println!("Hessian {:?}", h.to_rows());

    // Errors distinguish bad parameters from bad data.
    match normal_lpdf(1.0, 0.0, -2.0) {
        Err(e) => println!("rejected: {e}"),
        Ok(_) => unreachable!(),
    }
    Ok(())
}
