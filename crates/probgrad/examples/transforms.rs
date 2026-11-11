//! Constraining transforms: round trips and log Jacobians.
//!
//! Run with `cargo run --example transforms`.

use probgrad::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // A probability parameter on the logit scale.
    let mut lp = 0.0;
    let theta = lub_constrain_lp(0.4, 0.0, 1.0, &mut lp)?;
    println!(
        "theta = {theta:.6}, log Jacobian = {lp:.6}, back = {:.6}",
        lub_free(theta, 0.0, 1.0)?
    );

    // A simplex of length 4 from 3 free values (Stan's isometric log ratio form).
    let y = [0.3, -1.2, 0.8];
    let mut lp = 0.0;
    let x = simplex_constrain_lp(&y, &mut lp);
    println!(
        "simplex {x:.4?} sums to {:.15}, log Jacobian {lp:.4}",
        x.iter().sum::<f64>()
    );
    println!("free again {:.6?}", simplex_free(&x)?);

    // An ordered vector and a covariance matrix.
    println!("ordered {:.4?}", ordered_constrain(&y));
    let sigma = cov_matrix_constrain(&[0.2, -0.5, 0.1], 2)?;
    println!("covariance {:?}", sigma.to_rows());

    // Gradients flow through transforms and their Jacobians.
    let g = grad(
        |p| {
            let mut lp = p[0] * 0.0;
            let x = simplex_constrain_lp(p, &mut lp);
            lp + dirichlet_lpdf(&x, &[2.0, 3.0, 1.5, 4.0]).unwrap()
        },
        &y,
    );
    println!("gradient of a Dirichlet log density on the unconstrained scale {g:.6?}");
    Ok(())
}
