//! End-to-end check with an external sampler: NUTS from `nuts-rs` draws from posteriors written
//! with probgrad (densities, transforms with Jacobians, and reverse-mode gradients), and the draws
//! must reproduce known moments.

use std::collections::HashMap;

use nuts_rs::{
    Chain, CpuLogpFunc, CpuMath, CpuMathError, DiagNutsSettings, HasDims, LogpError, Settings,
};
use probgrad::prelude::*;
use rand::SeedableRng;

#[derive(Debug)]
struct DensityError(Error);

impl std::fmt::Display for DensityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for DensityError {}

impl LogpError for DensityError {
    fn is_recoverable(&self) -> bool {
        true
    }
}

/// Adapter from a probgrad log density to `nuts-rs`.
struct Density<F> {
    dim: usize,
    f: F,
    workspace: GradientWorkspace,
}

impl<F> HasDims for Density<F> {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        HashMap::from([("unconstrained_parameter".to_string(), self.dim as u64)])
    }
}

impl<F> CpuLogpFunc for Density<F>
where
    F: for<'t> Fn(&[Var<'t>]) -> Result<Var<'t>>,
{
    type LogpError = DensityError;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;

    fn dim(&self) -> usize {
        self.dim
    }

    fn logp(
        &mut self,
        position: &[f64],
        gradient: &mut [f64],
    ) -> std::result::Result<f64, DensityError> {
        let f = &self.f;
        self.workspace
            .try_value_and_grad_into(|x| f(x), position, gradient)
            .map_err(DensityError)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        position: &[f64],
    ) -> std::result::Result<Vec<f64>, CpuMathError> {
        Ok(position.to_vec())
    }
}

fn sample<F>(dim: usize, f: F, draws: usize, seed: u64) -> Vec<Vec<f64>>
where
    F: for<'t> Fn(&[Var<'t>]) -> Result<Var<'t>>,
{
    let settings = DiagNutsSettings {
        num_tune: 1000,
        num_draws: draws as u64,
        ..Default::default()
    };
    let math = CpuMath::new(Density {
        dim,
        f,
        workspace: GradientWorkspace::new(),
    });
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut chain = settings
        .new_chain(0, math, &mut rng)
        .expect("valid settings");
    chain
        .set_position(&vec![0.1; dim])
        .expect("finite initial density");
    for _ in 0..settings.num_tune {
        chain.draw().expect("tuning draw");
    }
    (0..draws)
        .map(|_| chain.draw().expect("draw").0.to_vec())
        .collect()
}

fn mean_and_var(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let m = xs.iter().sum::<f64>() / n;
    (
        m,
        xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0),
    )
}

#[test]
fn nuts_recovers_a_correlated_normal() {
    // Target: N(mu, L L^T) in 3 dimensions with correlation 0.8 between the first two.
    let mu = [1.0, -2.0, 0.5];
    let l = Matrix::from_rows(&[[1.0, 0.0, 0.0], [1.6, 1.2, 0.0], [0.0, 0.0, 0.5]]);
    let draws = sample(
        3,
        move |x| {
            let lv = l.map(|&v| Var::constant(v));
            multi_normal_cholesky_lpdf(x, &mu, &lv)
        },
        3000,
        11,
    );
    let sd = [1.0, 2.0, 0.5];
    for d in 0..3 {
        let col: Vec<f64> = draws.iter().map(|x| x[d]).collect();
        let (m, v) = mean_and_var(&col);
        // NUTS draws are autocorrelated; allow 6 naive standard errors.
        assert!(
            (m - mu[d]).abs() < 6.0 * sd[d] / (col.len() as f64).sqrt(),
            "dim {d}: mean {m}"
        );
        assert!(
            (v / (sd[d] * sd[d]) - 1.0).abs() < 0.15,
            "dim {d}: variance {v}"
        );
    }
}

#[test]
fn nuts_recovers_a_constrained_posterior() {
    // theta in (0, 1) with a beta(3, 5) density, sampled on the logit scale with the Jacobian.
    let draws = sample(
        1,
        |x| {
            let mut lp = x[0] * 0.0;
            let theta = lub_constrain_lp(x[0], 0.0, 1.0, &mut lp)?;
            Ok(lp + beta_lpdf(theta, 3.0, 5.0)?)
        },
        4000,
        12,
    );
    let thetas: Vec<f64> = draws
        .iter()
        .map(|x| probgrad::special::inv_logit(x[0]))
        .collect();
    let (m, v) = mean_and_var(&thetas);
    let (a, b) = (3.0, 5.0);
    let exact_mean = a / (a + b);
    let exact_var = a * b / ((a + b) * (a + b) * (a + b + 1.0));
    assert!(
        (m - exact_mean).abs() < 6.0 * exact_var.sqrt() / (thetas.len() as f64).sqrt(),
        "mean {m}"
    );
    assert!(
        (v / exact_var - 1.0).abs() < 0.15,
        "variance {v} vs {exact_var}"
    );
}
