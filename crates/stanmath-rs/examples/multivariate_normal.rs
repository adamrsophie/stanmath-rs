//! Multivariate normal likelihood with an unknown covariance, parameterized through a correlation
//! Cholesky factor and scales with an LKJ prior.
//!
//! Run with `cargo run --example multivariate_normal`.

use stanmath_rs::prelude::*;

/// Unconstrained parameters: mu (2), log scales (2), one correlation parameter.
fn log_posterior<T: Real>(p: &[T], ys: &[[f64; 2]]) -> Result<T> {
    let mut lp = T::zero();
    let mu = &p[0..2];
    let scales: Vec<T> = p[2..4]
        .iter()
        .map(|&s| lb_constrain_lp(s, 0.0, &mut lp))
        .collect();
    let l_corr = cholesky_factor_corr_constrain_lp(&p[4..5], 2, &mut lp)?;
    // L = diag(scales) L_corr
    let l = Matrix::from_fn(2, 2, |i, j| scales[i] * l_corr[(i, j)]);
    lp += lkj_corr_cholesky_lpdf(&l_corr, 2.0)?;
    lp += normal_lpdf(mu, 0.0, 5.0)? + exponential_lpdf(&scales[..], 1.0)?;
    for y in ys {
        lp += multi_normal_cholesky_lpdf(y, mu, &l)?;
    }
    Ok(lp)
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let ys = [
        [0.4, 1.1],
        [1.2, 2.3],
        [-0.3, 0.2],
        [0.9, 1.9],
        [0.1, 0.8],
        [1.5, 2.2],
    ];
    let mut theta = vec![0.0; 5];
    let mut workspace = GradientWorkspace::new();
    let mut g = vec![0.0; 5];
    for _ in 0..3000 {
        workspace.try_value_and_grad_into(|p| log_posterior(p, &ys), &theta, &mut g)?;
        for (t, gi) in theta.iter_mut().zip(&g) {
            *t += 0.01 * gi;
        }
    }
    let lp = log_posterior(&theta, &ys)?;
    let corr = cholesky_factor_corr_constrain(&theta[4..5], 2)?;
    println!("posterior mode (unconstrained) {theta:.3?}, log density {lp:.4}");
    println!("mu {:.3?}", &theta[0..2]);
    println!("scales {:.3?}", [theta[2].exp(), theta[3].exp()]);
    println!("correlation {:.3}", corr[(1, 0)]);
    let h = hessian(|p| log_posterior(p, &ys).unwrap(), &theta);
    println!(
        "Hessian diagonal at the mode {:.3?}",
        (0..5).map(|i| h[(i, i)]).collect::<Vec<_>>()
    );
    Ok(())
}
