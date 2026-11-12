//! The embedded Laplace approximation for latent Gaussian models.
//!
//! The model is `theta ~ normal(0, K(phi))` for an `n` dimensional latent vector and
//! `y ~ p(y | theta, eta)`. The marginal log density `log p(y | phi, eta)`, with `theta`
//! integrated out, is approximated by Laplace's method around the mode `theta_hat` of
//!
//! `psi(theta) = log p(y | theta, eta) - theta' K^(-1) theta / 2`.
//!
//! With `a = K^(-1) theta_hat` and `W` the negative Hessian of the log likelihood in `theta` at
//! the mode, the approximation is (Rasmussen and Williams 2006, eq. 3.32)
//!
//! `log p(y | phi, eta) = log p(y | theta_hat, eta) - a' theta_hat / 2 - log det(B) / 2`,
//!
//! where `B = I + W^(1/2) K W^(1/2)`. When `W` is not diagonal, or has a negative entry, the same
//! determinant is computed as `det(I + W K)`.
//!
//! # Mode and derivatives
//!
//! The mode is found in `f64` by Newton's method on `psi` written in terms of `a` (Rasmussen and
//! Williams 2006, Algorithm 3.1): `a_new = (I + W K)^(-1) (W theta + grad log p)` and
//! `theta_new = K a_new`, halving the step in `a` until `psi` does not decrease. The search stops
//! when `psi` changes by less than the tolerance. The `f64` mode then seeds Newton steps taken in
//! the caller's scalar type, as for [`solve_system`](crate::solve_system), so the mode carries
//! exact derivatives with respect to `phi` and `eta` up to the order of the scalar type, and the
//! approximate marginal is differentiated exactly (Margossian et al. 2020 obtain the same
//! gradient with an adjoint method).
//!
//! Latent indices (`y_index`) are zero based.

use stanmath_rs_autodiff::functions::{inv_logit, lgamma, log1p_exp};
use stanmath_rs_autodiff::{Dual, Real};
use stanmath_rs_core::{Error, Matrix, Result};

use crate::newton_steps;

/// A log likelihood `log p(y | theta, eta)` of the latent vector `theta`; the data `y` live in
/// `self`.
pub trait LaplaceLikelihood {
    /// Evaluates the log likelihood at latent values `theta` and likelihood parameters `eta`.
    fn log_likelihood<S: Real>(&self, theta: &[S], eta: &[S]) -> S;
}

/// The prior covariance `K(phi)` of the latent vector.
pub trait LaplaceCovariance {
    /// Evaluates the `n` by `n` covariance matrix at covariance parameters `phi`.
    ///
    /// # Errors
    ///
    /// Any error the covariance function reports, for example a nonpositive length scale.
    fn covariance<S: Real>(&self, phi: &[S]) -> Result<Matrix<S>>;
}

/// Control parameters of the mode search, the `_tol` arguments of Stan's Laplace functions.
#[derive(Clone, Debug)]
pub struct LaplaceOptions {
    /// Starting value of the mode search; `None` starts at zero.
    pub theta_init: Option<Vec<f64>>,
    /// Stop when the objective changes by less than this between Newton iterations.
    pub tolerance: f64,
    /// Maximum number of Newton iterations.
    pub max_num_steps: usize,
    /// Size of the diagonal blocks of the Hessian of a user supplied log likelihood with respect
    /// to `theta`; entries outside the blocks are taken to be zero. One means a diagonal Hessian;
    /// a value of at least `n` computes the dense Hessian. The built in likelihoods ignore it.
    pub hessian_block_size: usize,
    /// Maximum number of step halvings in each Newton iteration.
    pub max_steps_line_search: usize,
}

impl Default for LaplaceOptions {
    fn default() -> Self {
        LaplaceOptions {
            theta_init: None,
            tolerance: f64::EPSILON.sqrt(),
            max_num_steps: 500,
            hessian_block_size: 1,
            max_steps_line_search: 1000,
        }
    }
}

/// The negative Hessian `W` of the log likelihood with respect to `theta`.
pub(crate) enum Curvature<S> {
    Diagonal(Vec<S>),
    Dense(Vec<Vec<S>>),
}

impl<S: Real> Curvature<S> {
    /// `W v`.
    fn apply(&self, v: &[S]) -> Vec<S> {
        match self {
            Curvature::Diagonal(d) => d.iter().zip(v).map(|(&x, &y)| x * y).collect(),
            Curvature::Dense(w) => w.iter().map(|row| dot(row, v)).collect(),
        }
    }

    /// Rows of `I + W K`.
    pub(crate) fn i_plus_w_k(&self, k: &Matrix<S>) -> Vec<Vec<S>> {
        let n = k.nrows();
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let wk = match self {
                            Curvature::Diagonal(d) => d[i] * k[(i, j)],
                            Curvature::Dense(w) => (0..n).map(|l| w[i][l] * k[(l, j)]).sum(),
                        };
                        if i == j { wk + 1.0 } else { wk }
                    })
                    .collect()
            })
            .collect()
    }
}

/// A log likelihood together with its gradient and curvature in `theta`.
pub(crate) trait Model {
    /// Value, gradient, and negative Hessian of the log likelihood at `theta`.
    fn local<S: Real>(&self, theta: &[S], params: &[S]) -> (S, Vec<S>, Curvature<S>);

    /// The latent dimension implied by the model's own arguments, if any.
    fn latent_size(&self) -> Option<usize> {
        None
    }
}

/// A user supplied likelihood, differentiated in `theta` by nested forward mode.
pub(crate) struct General<'a, L> {
    pub(crate) likelihood: &'a L,
    pub(crate) block: usize,
}

impl<L: LaplaceLikelihood> Model for General<'_, L> {
    #[allow(clippy::needless_range_loop)] // dense matrix loops read more clearly with indices
    fn local<S: Real>(&self, theta: &[S], eta: &[S]) -> (S, Vec<S>, Curvature<S>) {
        let n = theta.len();
        let block = self.block.clamp(1, n.max(1));
        let lifted: Vec<Dual<Dual<S>>> = eta
            .iter()
            .map(|&e| Dual::constant(Dual::constant(e)))
            .collect();
        let mut value = None;
        let mut grad = vec![S::zero(); n];
        let mut w = vec![vec![S::zero(); n]; n];
        for start in (0..n).step_by(block) {
            let end = (start + block).min(n);
            for i in start..end {
                for j in i..end {
                    // Inner tangent along e_j, outer tangent along e_i: the outer tangent of the
                    // inner tangent is the second derivative in theta_i and theta_j.
                    let x: Vec<Dual<Dual<S>>> = theta
                        .iter()
                        .enumerate()
                        .map(|(k, &t)| {
                            let inner = if k == j {
                                Dual::variable(t)
                            } else {
                                Dual::constant(t)
                            };
                            let outer = if k == i { S::one() } else { S::zero() };
                            Dual::new(inner, Dual::constant(outer))
                        })
                        .collect();
                    let r = self.likelihood.log_likelihood(&x, &lifted);
                    value = Some(r.re.re);
                    if i == j {
                        grad[i] = r.eps.re;
                    }
                    let h = -r.eps.eps;
                    w[i][j] = h;
                    w[j][i] = h;
                }
            }
        }
        let value = value.unwrap_or_else(|| self.likelihood.log_likelihood(theta, eta));
        let curvature = if block == 1 {
            Curvature::Diagonal((0..n).map(|i| w[i][i]).collect())
        } else {
            Curvature::Dense(w)
        };
        (value, grad, curvature)
    }
}

/// The likelihoods Stan builds in.
#[derive(Clone, Copy)]
pub(crate) enum Family {
    PoissonLog,
    NegBinomial2Log,
    BernoulliLogit,
}

/// A built in likelihood. Observation `i` has linear predictor `theta[g] + m[g]` with
/// `g = y_index[i]`. The parameters are `m`, followed by the overdispersion for the negative
/// binomial.
pub(crate) struct Builtin<'a> {
    family: Family,
    y: &'a [i64],
    y_index: &'a [usize],
    n: usize,
}

impl<'a> Builtin<'a> {
    /// Validates the data and assembles the parameter vector.
    pub(crate) fn new<S: Real>(
        function: &'static str,
        family: Family,
        y: &'a [i64],
        y_index: &'a [usize],
        m: &[S],
        overdispersion: Option<S>,
    ) -> Result<(Self, Vec<S>)> {
        if y.len() != y_index.len() {
            return Err(Error::dimension(
                function,
                format!(
                    "y has {} elements but y_index has {}",
                    y.len(),
                    y_index.len()
                ),
            ));
        }
        let n = m.len();
        if let Some(&g) = y_index.iter().find(|&&g| g >= n) {
            return Err(Error::parameter(
                function,
                "y_index",
                g as f64,
                "a zero based index less than the length of m",
            ));
        }
        for &v in y {
            let ok = match family {
                Family::BernoulliLogit => v == 0 || v == 1,
                _ => v >= 0,
            };
            if !ok {
                let requirement = match family {
                    Family::BernoulliLogit => "0 or 1",
                    _ => "nonnegative",
                };
                return Err(Error::observation(function, "y", v as f64, requirement));
            }
        }
        if let Some(v) = m.iter().find(|v| !v.is_finite()) {
            return Err(Error::parameter(function, "m", v.value(), "finite"));
        }
        let mut params = m.to_vec();
        if let Some(phi) = overdispersion {
            if !(phi.value() > 0.0 && phi.is_finite()) {
                return Err(Error::parameter(
                    function,
                    "eta",
                    phi.value(),
                    "positive finite",
                ));
            }
            params.push(phi);
        }
        Ok((
            Builtin {
                family,
                y,
                y_index,
                n,
            },
            params,
        ))
    }
}

impl Model for Builtin<'_> {
    fn local<S: Real>(&self, theta: &[S], params: &[S]) -> (S, Vec<S>, Curvature<S>) {
        let n = self.n;
        let m = &params[..n];
        let mut value = S::zero();
        let mut grad = vec![S::zero(); n];
        let mut w = vec![S::zero(); n];
        for (&count, &g) in self.y.iter().zip(self.y_index) {
            let eta = theta[g] + m[g];
            let y = count as f64;
            match self.family {
                Family::PoissonLog => {
                    let mu = eta.exp();
                    value += eta * y - mu;
                    value -= lgamma(y + 1.0);
                    grad[g] += S::from_f64(y) - mu;
                    w[g] += mu;
                }
                Family::NegBinomial2Log => {
                    let phi = params[n];
                    let mu = eta.exp();
                    let log_mu_phi = (mu + phi).ln();
                    value += lgamma(phi + y) - lgamma(phi)
                        + phi * (phi.ln() - log_mu_phi)
                        + (eta - log_mu_phi) * y;
                    value -= lgamma(y + 1.0);
                    grad[g] += S::from_f64(y) - (phi + y) * mu / (mu + phi);
                    w[g] += (phi + y) * mu * phi / (mu + phi).square();
                }
                Family::BernoulliLogit => {
                    let p = inv_logit(eta);
                    value += eta * y - log1p_exp(eta);
                    grad[g] += S::from_f64(y) - p;
                    w[g] += p * (S::one() - p);
                }
            }
        }
        (value, grad, Curvature::Diagonal(w))
    }

    fn latent_size(&self) -> Option<usize> {
        Some(self.n)
    }
}

fn dot<S: Real>(a: &[S], b: &[S]) -> S {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

fn k_times<S: Real>(k: &Matrix<S>, a: &[S]) -> Vec<S> {
    let n = k.nrows();
    (0..n)
        .map(|i| (0..n).map(|j| k[(i, j)] * a[j]).sum())
        .collect()
}

/// Gaussian elimination with partial pivoting chosen on the values.
///
/// Solves `a x = b` for every right hand side `b` in `rhs` and returns the solutions, `ln |det a|`,
/// and the sign of `det a`.
#[allow(clippy::needless_range_loop)] // dense matrix loops read more clearly with indices
pub(crate) fn eliminate<S: Real>(
    function: &'static str,
    mut a: Vec<Vec<S>>,
    mut rhs: Vec<Vec<S>>,
) -> Result<(Vec<Vec<S>>, S, f64)> {
    let n = a.len();
    let mut log_det = S::zero();
    let mut sign = 1.0;
    for c in 0..n {
        let p = (c..n)
            .max_by(|&i, &j| a[i][c].value().abs().total_cmp(&a[j][c].value().abs()))
            .unwrap_or(c);
        let pivot = a[p][c].value();
        if pivot == 0.0 || !pivot.is_finite() {
            return Err(Error::parameter(function, "I + W K", pivot, "nonsingular"));
        }
        if p != c {
            a.swap(c, p);
            for b in rhs.iter_mut() {
                b.swap(c, p);
            }
            sign = -sign;
        }
        if pivot < 0.0 {
            sign = -sign;
        }
        log_det += a[c][c].abs().ln();
        for r in c + 1..n {
            let factor = a[r][c] / a[c][c];
            for k in c..n {
                let v = a[c][k];
                a[r][k] -= factor * v;
            }
            for b in rhs.iter_mut() {
                let v = b[c];
                b[r] -= factor * v;
            }
        }
    }
    for b in rhs.iter_mut() {
        for r in (0..n).rev() {
            let mut s = b[r];
            for k in r + 1..n {
                s -= a[r][k] * b[k];
            }
            b[r] = s / a[r][r];
        }
    }
    Ok((rhs, log_det, sign))
}

/// Lower Cholesky factor of a symmetric positive definite matrix given by rows.
#[allow(clippy::needless_range_loop)] // dense matrix loops read more clearly with indices
pub(crate) fn cholesky<S: Real>(function: &'static str, a: Vec<Vec<S>>) -> Result<Vec<Vec<S>>> {
    let n = a.len();
    let mut l = vec![vec![S::zero(); n]; n];
    for j in 0..n {
        let mut d = a[j][j];
        for k in 0..j {
            d -= l[j][k].square();
        }
        if !(d.value() > 0.0) {
            return Err(Error::NotPositiveDefinite { function });
        }
        let ljj = d.sqrt();
        l[j][j] = ljj;
        for i in j + 1..n {
            let mut s = a[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            l[i][j] = s / ljj;
        }
    }
    Ok(l)
}

/// `ln det(I + W^(1/2) K W^(1/2))`.
fn log_det_b<S: Real>(function: &'static str, w: &Curvature<S>, k: &Matrix<S>) -> Result<S> {
    match w {
        Curvature::Diagonal(d) if d.iter().all(|v| v.value() >= 0.0) => {
            // A latent component without observations has W = 0 exactly; skip its square root,
            // whose derivative is infinite there.
            let s: Vec<S> = d
                .iter()
                .map(|&v| if v.value() > 0.0 { v.sqrt() } else { S::zero() })
                .collect();
            let n = s.len();
            let b: Vec<Vec<S>> = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| {
                            let v = s[i] * k[(i, j)] * s[j];
                            if i == j { v + 1.0 } else { v }
                        })
                        .collect()
                })
                .collect();
            let l = cholesky(function, b)?;
            Ok((0..n).map(|j| l[j][j].ln()).sum::<S>() * 2.0)
        }
        _ => {
            let (_, log_det, sign) = eliminate(function, w.i_plus_w_k(k), Vec::new())?;
            if sign > 0.0 {
                Ok(log_det)
            } else {
                Err(Error::NotPositiveDefinite { function })
            }
        }
    }
}

/// One Newton step on `psi` in the `a` parameterization; returns `(a, theta)`.
fn newton_step<S: Real>(
    function: &'static str,
    k: &Matrix<S>,
    theta: &[S],
    grad: Vec<S>,
    w: &Curvature<S>,
) -> Result<(Vec<S>, Vec<S>)> {
    let b: Vec<S> = w
        .apply(theta)
        .into_iter()
        .zip(grad)
        .map(|(x, y)| x + y)
        .collect();
    let (mut solution, _, _) = eliminate(function, w.i_plus_w_k(k), vec![b])?;
    let a = solution.swap_remove(0);
    let theta = k_times(k, &a);
    Ok((a, theta))
}

fn check_options(function: &'static str, o: &LaplaceOptions) -> Result<()> {
    if !(o.tolerance > 0.0 && o.tolerance.is_finite()) {
        return Err(Error::parameter(
            function,
            "tolerance",
            o.tolerance,
            "positive finite",
        ));
    }
    if o.max_num_steps == 0 {
        return Err(Error::parameter(function, "max_num_steps", 0.0, "positive"));
    }
    if o.hessian_block_size == 0 {
        return Err(Error::parameter(
            function,
            "hessian_block_size",
            0.0,
            "positive",
        ));
    }
    Ok(())
}

fn covariance_matrix<S: Real, K: LaplaceCovariance>(
    function: &'static str,
    covariance: &K,
    phi: &[S],
    n: Option<usize>,
) -> Result<Matrix<S>> {
    let k = covariance.covariance(phi)?;
    if !k.is_square() || n.is_some_and(|n| n != k.nrows()) {
        return Err(Error::dimension(
            function,
            format!(
                "covariance is {}x{}, expected a square matrix of the latent size",
                k.nrows(),
                k.ncols()
            ),
        ));
    }
    if let Some(v) = k.as_slice().iter().find(|v| !v.is_finite()) {
        return Err(Error::parameter(
            function,
            "covariance",
            v.value(),
            "finite",
        ));
    }
    Ok(k)
}

/// Damped Newton search for the mode in `f64`; returns `(theta_hat, a)` with `theta_hat = K a`.
fn find_mode<M: Model>(
    function: &'static str,
    model: &M,
    k: &Matrix<f64>,
    params: &[f64],
    o: &LaplaceOptions,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let n = k.nrows();
    let (mut theta, mut a) = match &o.theta_init {
        None => (vec![0.0; n], vec![0.0; n]),
        Some(t) => {
            if t.len() != n {
                return Err(Error::dimension(
                    function,
                    format!("theta_init has {} elements, expected {n}", t.len()),
                ));
            }
            let rows: Vec<Vec<f64>> = (0..n)
                .map(|i| (0..n).map(|j| k[(i, j)]).collect())
                .collect();
            let (mut solution, _, _) = eliminate(function, rows, vec![t.clone()])?;
            let a = solution.swap_remove(0);
            (k_times(k, &a), a)
        }
    };
    let mut local = model.local(&theta, params);
    let mut psi = local.0 - 0.5 * dot(&a, &theta);
    if !psi.is_finite() {
        return Err(Error::parameter(
            function,
            "theta_init",
            psi,
            "a point with finite log density",
        ));
    }
    for _ in 0..o.max_num_steps {
        let (_, grad, w) = local;
        let (mut a_new, mut theta_new) = newton_step(function, k, &theta, grad, &w)?;
        let mut next = model.local(&theta_new, params);
        let mut psi_new = next.0 - 0.5 * dot(&a_new, &theta_new);
        let mut halvings = 0;
        while !(psi_new >= psi) && halvings < o.max_steps_line_search {
            for (x, &y) in a_new.iter_mut().zip(&a) {
                *x = 0.5 * (*x + y);
            }
            theta_new = k_times(k, &a_new);
            next = model.local(&theta_new, params);
            psi_new = next.0 - 0.5 * dot(&a_new, &theta_new);
            halvings += 1;
        }
        if !psi_new.is_finite() {
            break;
        }
        let change = (psi_new - psi).abs();
        theta = theta_new;
        a = a_new;
        psi = psi_new;
        local = next;
        if change < o.tolerance {
            return Ok((theta, a));
        }
    }
    Err(Error::NoConvergence {
        function,
        iterations: o.max_num_steps,
    })
}

/// Validates the options and covariance and finds the mode in `f64`; returns
/// `(theta_hat, a, K)`.
pub(crate) fn fit<M: Model, K: LaplaceCovariance>(
    function: &'static str,
    model: &M,
    params: &[f64],
    covariance: &K,
    phi: &[f64],
    o: &LaplaceOptions,
) -> Result<(Vec<f64>, Vec<f64>, Matrix<f64>)> {
    check_options(function, o)?;
    let k = covariance_matrix(function, covariance, phi, model.latent_size())?;
    let (theta, a) = find_mode(function, model, &k, params, o)?;
    Ok((theta, a, k))
}

fn marginal<T: Real, M: Model, K: LaplaceCovariance>(
    function: &'static str,
    model: &M,
    params: &[T],
    covariance: &K,
    phi: &[T],
    o: &LaplaceOptions,
) -> Result<T> {
    let params_f: Vec<f64> = params.iter().map(Real::value).collect();
    let phi_f: Vec<f64> = phi.iter().map(Real::value).collect();
    let (theta_f, a_f, _) = fit(function, model, &params_f, covariance, &phi_f, o)?;
    let k = covariance_matrix(function, covariance, phi, model.latent_size())?;
    let mut theta: Vec<T> = theta_f.into_iter().map(T::from_f64).collect();
    let mut a: Vec<T> = a_f.into_iter().map(T::from_f64).collect();
    for _ in 0..newton_steps(T::ORDER) {
        let (_, grad, w) = model.local(&theta, params);
        (a, theta) = newton_step(function, &k, &theta, grad, &w)?;
    }
    let (value, _, w) = model.local(&theta, params);
    let log_det = log_det_b(function, &w, &k)?;
    Ok(value - dot(&a, &theta) * 0.5 - log_det * 0.5)
}

/// Laplace approximation of `log p(y | phi, eta)` for a user supplied likelihood, with default
/// [`LaplaceOptions`]. Stan's `laplace_marginal`.
///
/// # Errors
///
/// See [`laplace_marginal_tol`].
pub fn laplace_marginal<T: Real, L: LaplaceLikelihood, K: LaplaceCovariance>(
    likelihood: &L,
    eta: &[T],
    covariance: &K,
    phi: &[T],
) -> Result<T> {
    laplace_marginal_tol(likelihood, eta, covariance, phi, &LaplaceOptions::default())
}

/// Laplace approximation of `log p(y | phi, eta)` for a user supplied likelihood. Stan's
/// `laplace_marginal_tol`.
///
/// The Hessian of the log likelihood in `theta` is computed by nested forward mode within diagonal
/// blocks of size [`LaplaceOptions::hessian_block_size`]; set it to the latent size when the
/// Hessian is not diagonal.
///
/// # Errors
///
/// [`Error::NoConvergence`] when the mode search does not converge,
/// [`Error::NotPositiveDefinite`] when the mode is not a maximum, and dimension or parameter errors
/// for invalid options or a covariance that is not square and finite.
pub fn laplace_marginal_tol<T: Real, L: LaplaceLikelihood, K: LaplaceCovariance>(
    likelihood: &L,
    eta: &[T],
    covariance: &K,
    phi: &[T],
    options: &LaplaceOptions,
) -> Result<T> {
    let model = General {
        likelihood,
        block: options.hessian_block_size,
    };
    marginal("laplace_marginal", &model, eta, covariance, phi, options)
}

/// Mean and lower Cholesky factor of the Gaussian approximation `normal(theta_hat, (K^(-1) + W)^(-1))`
/// at the mode. The covariance is formed as `(I + K W)^(-1) K`, which needs no inverse of `K`.
pub(crate) fn latent_gaussian<M: Model, K: LaplaceCovariance>(
    function: &'static str,
    model: &M,
    params: &[f64],
    covariance: &K,
    phi: &[f64],
    o: &LaplaceOptions,
) -> Result<(Vec<f64>, Matrix<f64>)> {
    let (theta, _, k) = fit(function, model, params, covariance, phi, o)?;
    let (_, _, w) = model.local(&theta, params);
    let n = theta.len();
    // I + K W is the transpose of I + W K, because K and W are symmetric.
    let i_plus_w_k = w.i_plus_w_k(&k);
    let i_plus_k_w: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| i_plus_w_k[j][i]).collect())
        .collect();
    let columns: Vec<Vec<f64>> = (0..n).map(|j| k.column(j).to_vec()).collect();
    let (sigma, _, _) = eliminate(function, i_plus_k_w, columns)?;
    // `sigma[j]` is column j; symmetrize away rounding before the Cholesky factorization.
    let symmetric: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| 0.5 * (sigma[j][i] + sigma[i][j])).collect())
        .collect();
    let l = cholesky(function, symmetric)?;
    Ok((theta, Matrix::from_fn(n, n, |i, j| l[i][j])))
}

/// Mean and lower Cholesky factor of the covariance of the Laplace approximation to
/// `p(theta | y, phi)` for a user supplied likelihood, with default [`LaplaceOptions`]. Stan's
/// `laplace_latent_solve`.
///
/// # Errors
///
/// As [`laplace_marginal_tol`], plus [`Error::NotPositiveDefinite`] when the approximate covariance
/// is not positive definite.
pub fn laplace_latent_solve<L: LaplaceLikelihood, K: LaplaceCovariance>(
    likelihood: &L,
    eta: &[f64],
    covariance: &K,
    phi: &[f64],
) -> Result<(Vec<f64>, Matrix<f64>)> {
    laplace_latent_solve_tol(likelihood, eta, covariance, phi, &LaplaceOptions::default())
}

/// [`laplace_latent_solve`] with explicit options. Stan's `laplace_latent_solve_tol`.
///
/// # Errors
///
/// As [`laplace_latent_solve`].
pub fn laplace_latent_solve_tol<L: LaplaceLikelihood, K: LaplaceCovariance>(
    likelihood: &L,
    eta: &[f64],
    covariance: &K,
    phi: &[f64],
    options: &LaplaceOptions,
) -> Result<(Vec<f64>, Matrix<f64>)> {
    let model = General {
        likelihood,
        block: options.hessian_block_size,
    };
    latent_gaussian(
        "laplace_latent_solve",
        &model,
        eta,
        covariance,
        phi,
        options,
    )
}

/// Default options for a latent vector of size `dimension`, with a zero starting value. Stan's
/// `generate_laplace_options(int)`. Set `hessian_block_size` before use with a likelihood whose
/// Hessian is not diagonal.
pub fn generate_laplace_options(dimension: usize) -> LaplaceOptions {
    LaplaceOptions {
        theta_init: Some(vec![0.0; dimension]),
        ..LaplaceOptions::default()
    }
}

/// Default options with the starting value `theta_init`. Stan's `generate_laplace_options(vector)`.
pub fn generate_laplace_options_from(theta_init: &[f64]) -> LaplaceOptions {
    LaplaceOptions {
        theta_init: Some(theta_init.to_vec()),
        ..LaplaceOptions::default()
    }
}

/// Laplace approximation of the marginal log probability of counts `y` with
/// `y[i] ~ poisson_log(theta[y_index[i]] + m[y_index[i]])` and `theta ~ normal(0, K(phi))`.
/// Stan's `laplace_marginal_poisson_log_lpmf`, with default [`LaplaceOptions`].
///
/// # Errors
///
/// See [`laplace_marginal_tol_poisson_log_lpmf`].
pub fn laplace_marginal_poisson_log_lpmf<T: Real, K: LaplaceCovariance>(
    y: &[i64],
    y_index: &[usize],
    m: &[T],
    covariance: &K,
    phi: &[T],
) -> Result<T> {
    let o = LaplaceOptions::default();
    laplace_marginal_tol_poisson_log_lpmf(y, y_index, m, covariance, phi, &o)
}

/// [`laplace_marginal_poisson_log_lpmf`] with explicit options. Stan's
/// `laplace_marginal_tol_poisson_log_lpmf`.
///
/// # Errors
///
/// As [`laplace_marginal_tol`], plus an observation error for a negative count and a dimension or
/// parameter error when `y_index` does not match `y` or `m`.
pub fn laplace_marginal_tol_poisson_log_lpmf<T: Real, K: LaplaceCovariance>(
    y: &[i64],
    y_index: &[usize],
    m: &[T],
    covariance: &K,
    phi: &[T],
    options: &LaplaceOptions,
) -> Result<T> {
    const F: &str = "laplace_marginal_poisson_log_lpmf";
    let (model, params) = Builtin::new(F, Family::PoissonLog, y, y_index, m, None)?;
    marginal(F, &model, &params, covariance, phi, options)
}

/// Laplace approximation of the marginal log probability of counts `y` with
/// `y[i] ~ neg_binomial_2_log(theta[g] + m[g], eta)`, `g = y_index[i]`, and
/// `theta ~ normal(0, K(phi))`. Stan's `laplace_marginal_neg_binomial_2_log_lpmf`, with default
/// [`LaplaceOptions`].
///
/// # Errors
///
/// See [`laplace_marginal_tol_neg_binomial_2_log_lpmf`].
pub fn laplace_marginal_neg_binomial_2_log_lpmf<T: Real, K: LaplaceCovariance>(
    y: &[i64],
    y_index: &[usize],
    eta: T,
    m: &[T],
    covariance: &K,
    phi: &[T],
) -> Result<T> {
    let o = LaplaceOptions::default();
    laplace_marginal_tol_neg_binomial_2_log_lpmf(y, y_index, eta, m, covariance, phi, &o)
}

/// [`laplace_marginal_neg_binomial_2_log_lpmf`] with explicit options. Stan's
/// `laplace_marginal_tol_neg_binomial_2_log_lpmf`.
///
/// # Errors
///
/// As [`laplace_marginal_tol_poisson_log_lpmf`], plus a parameter error unless `eta` is positive
/// and finite.
pub fn laplace_marginal_tol_neg_binomial_2_log_lpmf<T: Real, K: LaplaceCovariance>(
    y: &[i64],
    y_index: &[usize],
    eta: T,
    m: &[T],
    covariance: &K,
    phi: &[T],
    options: &LaplaceOptions,
) -> Result<T> {
    const F: &str = "laplace_marginal_neg_binomial_2_log_lpmf";
    let (model, params) = Builtin::new(F, Family::NegBinomial2Log, y, y_index, m, Some(eta))?;
    marginal(F, &model, &params, covariance, phi, options)
}

/// Laplace approximation of the marginal log probability of binary outcomes `y` with
/// `y[i] ~ bernoulli_logit(theta[y_index[i]] + m[y_index[i]])` and `theta ~ normal(0, K(phi))`.
/// Stan's `laplace_marginal_bernoulli_logit_lpmf`, with default [`LaplaceOptions`].
///
/// # Errors
///
/// See [`laplace_marginal_tol_bernoulli_logit_lpmf`].
pub fn laplace_marginal_bernoulli_logit_lpmf<T: Real, K: LaplaceCovariance>(
    y: &[i64],
    y_index: &[usize],
    m: &[T],
    covariance: &K,
    phi: &[T],
) -> Result<T> {
    let o = LaplaceOptions::default();
    laplace_marginal_tol_bernoulli_logit_lpmf(y, y_index, m, covariance, phi, &o)
}

/// [`laplace_marginal_bernoulli_logit_lpmf`] with explicit options. Stan's
/// `laplace_marginal_tol_bernoulli_logit_lpmf`.
///
/// # Errors
///
/// As [`laplace_marginal_tol_poisson_log_lpmf`], with outcomes required to be 0 or 1.
pub fn laplace_marginal_tol_bernoulli_logit_lpmf<T: Real, K: LaplaceCovariance>(
    y: &[i64],
    y_index: &[usize],
    m: &[T],
    covariance: &K,
    phi: &[T],
    options: &LaplaceOptions,
) -> Result<T> {
    const F: &str = "laplace_marginal_bernoulli_logit_lpmf";
    let (model, params) = Builtin::new(F, Family::BernoulliLogit, y, y_index, m, None)?;
    marginal(F, &model, &params, covariance, phi, options)
}
