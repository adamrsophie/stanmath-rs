//! Hidden Markov models with `K` discrete states.
//!
//! The arguments follow Stan's conventions:
//!
//! * `log_omegas` is `K` by `N`; entry `(k, n)` is the log density of observation `n` given that
//!   the chain is in state `k` at time `n`;
//! * `gamma` is the `K` by `K` transition matrix; row `i` is the distribution of the next state
//!   given state `i`, so every row is a simplex;
//! * `rho` is the distribution of the initial state, a simplex of length `K`.

// Dense state loops read more clearly with indices.
#![allow(clippy::needless_range_loop)]

use stanmath_rs_autodiff::Real;
use stanmath_rs_core::{Error, Matrix, Result};

/// Tolerance for simplex checks, as Stan's `CONSTRAINT_TOLERANCE`.
const TOL: f64 = 1e-8;

fn is_simplex(values: impl Iterator<Item = f64>) -> bool {
    let mut sum = 0.0;
    for v in values {
        if v.is_nan() || v < 0.0 {
            return false;
        }
        sum += v;
    }
    (sum - 1.0).abs() <= TOL
}

fn check_hmm<T: Real>(
    function: &'static str,
    log_omegas: &Matrix<T>,
    gamma: &Matrix<T>,
    rho: &[T],
) -> Result<usize> {
    let k = gamma.nrows();
    if !gamma.is_square() {
        return Err(Error::dimension(
            function,
            format!("Gamma is {}x{}, expected a square matrix", k, gamma.ncols()),
        ));
    }
    if log_omegas.nrows() != k || rho.len() != k {
        return Err(Error::dimension(
            function,
            format!(
                "log_omegas has {} rows and rho has length {}, expected {k}",
                log_omegas.nrows(),
                rho.len()
            ),
        ));
    }
    if !is_simplex(rho.iter().map(Real::value)) {
        return Err(Error::parameter(function, "rho", f64::NAN, "a simplex"));
    }
    for i in 0..k {
        if !is_simplex(gamma.row(i).map(Real::value)) {
            return Err(Error::parameter(
                function,
                "Gamma",
                i as f64,
                "a matrix whose rows are simplexes",
            ));
        }
    }
    if let Some(v) = log_omegas.as_slice().iter().find(|v| v.is_nan()) {
        return Err(Error::parameter(
            function,
            "log_omegas",
            v.value(),
            "not NaN",
        ));
    }
    Ok(k)
}

/// Scaled forward algorithm.
///
/// Returns the filtered state probabilities `P(z_n | y_1, ..., y_n)` for every `n` (when `keep` is
/// true) and the log marginal likelihood. Each step works with `exp(log_omegas - c_n)` for the
/// constant `c_n = max_k log_omegas(k, n)` and adds `c_n` back to the log likelihood, so nothing
/// underflows and no logarithm of a zero transition probability is taken.
fn forward<T: Real>(
    log_omegas: &Matrix<T>,
    gamma: &Matrix<T>,
    rho: &[T],
    keep: bool,
) -> (Vec<Vec<T>>, T) {
    let (k, n) = log_omegas.shape();
    let mut filtered = Vec::new();
    let mut log_lik = T::zero();
    let mut alpha: Vec<T> = rho.to_vec();
    for t in 0..n {
        let col = log_omegas.column(t);
        let shift = col.iter().fold(f64::NEG_INFINITY, |m, v| m.max(v.value()));
        if !(shift > f64::NEG_INFINITY) {
            return (filtered, T::from_f64(f64::NEG_INFINITY));
        }
        let predicted: Vec<T> = if t == 0 {
            alpha
        } else {
            (0..k)
                .map(|j| (0..k).map(|i| alpha[i] * gamma[(i, j)]).sum::<T>())
                .collect()
        };
        let joint: Vec<T> = (0..k)
            .map(|j| predicted[j] * (col[j] - shift).exp())
            .collect();
        let total: T = joint.iter().copied().sum();
        if !(total.value() > 0.0) {
            return (filtered, T::from_f64(f64::NEG_INFINITY));
        }
        log_lik += total.ln() + shift;
        alpha = joint.into_iter().map(|v| v / total).collect();
        if keep {
            filtered.push(alpha.clone());
        }
    }
    (filtered, log_lik)
}

/// Log marginal density of the observations under a hidden Markov model, summing over every
/// path of hidden states with the forward algorithm.
///
/// Differentiable in all three arguments. Returns zero when there are no observations and `-inf`
/// when the observations are impossible under the model.
///
/// # Errors
///
/// Dimension mismatches, a `rho` or a row of `gamma` that is not a simplex (within `1e-8`), and
/// NaN in `log_omegas`.
pub fn hmm_marginal<T: Real>(log_omegas: &Matrix<T>, gamma: &Matrix<T>, rho: &[T]) -> Result<T> {
    check_hmm("hmm_marginal", log_omegas, gamma, rho)?;
    Ok(forward(log_omegas, gamma, rho, false).1)
}

/// Posterior probabilities of the hidden states: entry `(k, n)` is `P(z_n = k | y_1, ..., y_N)`,
/// computed with the forward backward algorithm.
///
/// # Errors
///
/// The checks of [`hmm_marginal`], and an error when the observations are impossible under the
/// model.
pub fn hmm_hidden_state_prob(
    log_omegas: &Matrix<f64>,
    gamma: &Matrix<f64>,
    rho: &[f64],
) -> Result<Matrix<f64>> {
    const F: &str = "hmm_hidden_state_prob";
    let k = check_hmm(F, log_omegas, gamma, rho)?;
    let n = log_omegas.ncols();
    let (filtered, log_lik) = forward(log_omegas, gamma, rho, true);
    if n == 0 {
        return Ok(Matrix::zeros(k, 0));
    }
    if !(log_lik > f64::NEG_INFINITY) {
        return Err(Error::parameter(
            F,
            "log_omegas",
            log_lik,
            "possible under Gamma and rho",
        ));
    }
    let mut probs = Matrix::zeros(k, n);
    // beta(i) is proportional to p(y_{t+1}, ..., y_N | z_t = i).
    let mut beta = vec![1.0; k];
    for t in (0..n).rev() {
        let post: Vec<f64> = (0..k).map(|i| filtered[t][i] * beta[i]).collect();
        let total: f64 = post.iter().sum();
        for i in 0..k {
            probs[(i, t)] = post[i] / total;
        }
        if t > 0 {
            let col = log_omegas.column(t);
            let shift = col.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v));
            let next: Vec<f64> = (0..k)
                .map(|i| {
                    (0..k)
                        .map(|j| gamma[(i, j)] * (col[j] - shift).exp() * beta[j])
                        .sum()
                })
                .collect();
            let scale: f64 = next.iter().sum();
            beta = next.into_iter().map(|v| v / scale).collect();
        }
    }
    Ok(probs)
}

/// Draws a path of hidden states from its posterior given the observations, by forward
/// filtering and backward sampling. States are numbered `1..=K`, as in Stan.
///
/// # Errors
///
/// The checks of [`hmm_marginal`], and an error when the observations are impossible under the
/// model.
#[cfg(feature = "rand")]
pub fn hmm_latent_rng<R: rand::Rng + ?Sized>(
    log_omegas: &Matrix<f64>,
    gamma: &Matrix<f64>,
    rho: &[f64],
    rng: &mut R,
) -> Result<Vec<i64>> {
    const F: &str = "hmm_latent_rng";
    let k = check_hmm(F, log_omegas, gamma, rho)?;
    let n = log_omegas.ncols();
    let (filtered, log_lik) = forward(log_omegas, gamma, rho, true);
    if n == 0 {
        return Ok(Vec::new());
    }
    if !(log_lik > f64::NEG_INFINITY) {
        return Err(Error::parameter(
            F,
            "log_omegas",
            log_lik,
            "possible under Gamma and rho",
        ));
    }
    let mut z = vec![0usize; n];
    z[n - 1] = draw(F, &filtered[n - 1], rng)?;
    for t in (0..n - 1).rev() {
        // P(z_t = i | z_{t+1}, y_1, ..., y_t) is proportional to alpha_t(i) Gamma(i, z_{t+1}).
        let weights: Vec<f64> = (0..k)
            .map(|i| filtered[t][i] * gamma[(i, z[t + 1])])
            .collect();
        z[t] = draw(F, &weights, rng)?;
    }
    Ok(z.into_iter().map(|s| s as i64 + 1).collect())
}

#[cfg(feature = "rand")]
fn draw<R: rand::Rng + ?Sized>(
    function: &'static str,
    weights: &[f64],
    rng: &mut R,
) -> Result<usize> {
    use rand::distr::{Distribution, weighted::WeightedIndex};
    let d = WeightedIndex::new(weights).map_err(|_| {
        Error::parameter(function, "state weights", f64::NAN, "positive and finite")
    })?;
    Ok(d.sample(rng))
}
