//! Random number generation for the Yule-Simon distribution, the Bernoulli logit GLM, the
//! Gaussian dynamic linear model, and the precision parameterized normal distributions.
//!
//! Each `*_rng` function validates its parameters like the corresponding log density and draws
//! one value. Pass any `rand::Rng`, for example a seeded `rand::rngs::StdRng`.

use rand::Rng;
use rand::distr::{Distribution, Uniform};
use rand_distr as rd;
use stanmath_rs_autodiff::Arg;
use stanmath_rs_core::special::{inv_logit, log1m_exp};
use stanmath_rs_core::{Error, Matrix, Result};
use stanmath_rs_linalg::cholesky_decompose;

use crate::check::param;

fn std_normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    rd::StandardNormal.sample(rng)
}

fn unit<R: Rng + ?Sized>(f: &'static str, rng: &mut R) -> Result<f64> {
    let d = Uniform::new(0.0, 1.0)
        .map_err(|_| Error::parameter(f, "parameters", f64::NAN, "valid for the sampler"))?;
    Ok(d.sample(rng))
}

fn dims(f: &'static str, what: &str, got: usize, expected: usize) -> Result<()> {
    if got == expected {
        Ok(())
    } else {
        Err(Error::dimension(
            f,
            format!("{what} has size {got}, expected {expected}"),
        ))
    }
}

fn square(f: &'static str, name: &str, m: &Matrix<f64>) -> Result<usize> {
    if m.is_square() {
        Ok(m.nrows())
    } else {
        Err(Error::dimension(
            f,
            format!(
                "{name} is {}x{}, expected a square matrix",
                m.nrows(),
                m.ncols()
            ),
        ))
    }
}

fn finite(f: &'static str, name: &'static str, xs: &[f64]) -> Result<()> {
    match xs.iter().find(|v| !v.is_finite()) {
        Some(&v) => Err(Error::parameter(f, name, v, "finite")),
        None => Ok(()),
    }
}

/// Solves `Lᵀ x = b` for lower triangular `l` by back substitution.
fn solve_lower_transpose(l: &Matrix<f64>, b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = b.to_vec();
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|j| l[(j, i)] * x[j]).sum();
        x[i] = (x[i] - s) / l[(i, i)];
    }
    x
}

/// `L z` for standard normal `z`, reading only the lower triangle of `l`.
fn lower_times_normal<R: Rng + ?Sized>(l: &Matrix<f64>, rng: &mut R) -> Vec<f64> {
    let k = l.nrows();
    let z: Vec<f64> = (0..k).map(|_| std_normal(rng)).collect();
    (0..k)
        .map(|i| (0..=i).map(|j| l[(i, j)] * z[j]).sum::<f64>())
        .collect()
}

/// A lower triangular `L` with `L Lᵀ = a` for symmetric positive semidefinite `a`.
///
/// Cholesky without pivoting, where a pivot within a relative tolerance of zero gives a zero
/// column, so singular covariance matrices (for example a zero state noise) are accepted.
fn psd_factor(f: &'static str, name: &'static str, a: &Matrix<f64>) -> Result<Matrix<f64>> {
    let n = square(f, name, a)?;
    finite(f, name, a.as_slice())?;
    let scale = (0..n).map(|i| a[(i, i)].abs()).fold(0.0f64, f64::max);
    let tol = 1e-8 * scale.max(f64::MIN_POSITIVE);
    for i in 0..n {
        for j in 0..i {
            let (x, y) = (a[(i, j)], a[(j, i)]);
            if (x - y).abs() > 1e-8 * x.abs().max(y.abs()).max(1.0) {
                return Err(Error::parameter(f, name, x, "symmetric"));
            }
        }
    }
    // ponytail: no pivoting; a nearly singular matrix with a late zero pivot loses accuracy.
    let mut l = Matrix::from_fn(n, n, |_, _| 0.0);
    for j in 0..n {
        let d = a[(j, j)] - (0..j).map(|k| l[(j, k)] * l[(j, k)]).sum::<f64>();
        if d < -tol {
            return Err(Error::parameter(f, name, d, "positive semidefinite"));
        }
        if d <= tol {
            continue;
        }
        let ljj = d.sqrt();
        l[(j, j)] = ljj;
        for i in j + 1..n {
            let s = a[(i, j)] - (0..j).map(|k| l[(i, k)] * l[(j, k)]).sum::<f64>();
            l[(i, j)] = s / ljj;
        }
    }
    Ok(l)
}

/// Draw from `yule_simon(alpha)`.
///
/// Uses the mixture representation: `W ~ exponential(alpha)`, then a geometric count on
/// `1, 2, ...` with success probability `exp(-W)`, drawn by inversion.
///
/// # Errors
///
/// `alpha` not positive finite, or a draw too large for `i64` (possible when `alpha` is tiny).
pub fn yule_simon_rng<R: Rng + ?Sized>(alpha: f64, rng: &mut R) -> Result<i64> {
    const F: &str = "yule_simon_rng";
    param::positive_finite(F, "alpha", alpha)?;
    let e: f64 = rd::Exp1.sample(rng);
    let w = e / alpha;
    // ln of the failure probability, ln(1 - exp(-w)).
    let ln_fail = log1m_exp(-w);
    if ln_fail == f64::NEG_INFINITY {
        return Ok(1);
    }
    // A uniform on (0, 1], so its log is finite.
    let u = 1.0 - unit(F, rng)?;
    let k = (u.ln() / ln_fail).floor() + 1.0;
    if !(k < i64::MAX as f64) {
        return Err(Error::parameter(
            F,
            "alpha",
            alpha,
            "large enough that draws fit in a 64 bit integer",
        ));
    }
    Ok(k as i64)
}

/// Draw from `bernoulli_logit_glm(x, alpha, beta)`: one outcome per row of `x`, each Bernoulli
/// with success probability `inv_logit(alpha_i + x_i beta)`.
///
/// `alpha` is a scalar or has one entry per row of `x`.
///
/// # Errors
///
/// Dimension mismatches and non-finite `x`, `alpha`, or `beta`.
pub fn bernoulli_logit_glm_rng<A, R>(
    x: &Matrix<f64>,
    alpha: A,
    beta: &[f64],
    rng: &mut R,
) -> Result<Vec<i64>>
where
    A: Arg<Elem = f64>,
    R: Rng + ?Sized,
{
    const F: &str = "bernoulli_logit_glm_rng";
    let n = x.nrows();
    dims(F, "beta", beta.len(), x.ncols())?;
    if let Some(len) = alpha.arg_len() {
        dims(F, "alpha", len, n)?;
    }
    finite(F, "x", x.as_slice())?;
    finite(F, "beta", beta)?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let a = alpha.arg_at(i);
        if !a.is_finite() {
            return Err(Error::parameter(F, "alpha", a, "finite"));
        }
        let eta = a + (0..beta.len()).map(|j| x[(i, j)] * beta[j]).sum::<f64>();
        let u = unit(F, rng)?;
        out.push(i64::from(u < inv_logit(eta)));
    }
    Ok(out)
}

/// Draw from `gaussian_dlm_obs(F, G, V, W, m0, C0)` over `t` time steps: the `r` by `t` matrix
/// whose columns are `y_s = Fᵀ θ_s + v_s`, with `θ_s = G θ_{s-1} + w_s`, `θ_0 ~ N(m0, C0)`,
/// `v_s ~ N(0, V)`, and `w_s ~ N(0, W)`.
///
/// `F` is `n` by `r`, `G` and `W` are `n` by `n`, `V` is `r` by `r`, `m0` has length `n`, and
/// `C0` is `n` by `n`. The covariance matrices may be singular (positive semidefinite).
///
/// # Errors
///
/// Dimension mismatches, non-finite `F`, `G`, or `m0`, and covariance matrices that are not
/// symmetric positive semidefinite.
#[allow(clippy::too_many_arguments)]
pub fn gaussian_dlm_obs_rng<R: Rng + ?Sized>(
    f: &Matrix<f64>,
    g: &Matrix<f64>,
    v: &Matrix<f64>,
    w: &Matrix<f64>,
    m0: &[f64],
    c0: &Matrix<f64>,
    t: usize,
    rng: &mut R,
) -> Result<Matrix<f64>> {
    const F: &str = "gaussian_dlm_obs_rng";
    let (n, r) = f.shape();
    dims(F, "G", square(F, "G", g)?, n)?;
    dims(F, "V", square(F, "V", v)?, r)?;
    dims(F, "W", square(F, "W", w)?, n)?;
    dims(F, "m0", m0.len(), n)?;
    dims(F, "C0", square(F, "C0", c0)?, n)?;
    finite(F, "F", f.as_slice())?;
    finite(F, "G", g.as_slice())?;
    finite(F, "m0", m0)?;
    let lv = psd_factor(F, "V", v)?;
    let lw = psd_factor(F, "W", w)?;
    let lc = psd_factor(F, "C0", c0)?;
    let mut theta: Vec<f64> = lower_times_normal(&lc, rng)
        .iter()
        .zip(m0)
        .map(|(e, m)| m + e)
        .collect();
    let mut y = Matrix::from_fn(r, t, |_, _| 0.0);
    for s in 0..t {
        let noise = lower_times_normal(&lw, rng);
        theta = (0..n)
            .map(|i| noise[i] + (0..n).map(|j| g[(i, j)] * theta[j]).sum::<f64>())
            .collect();
        let obs_noise = lower_times_normal(&lv, rng);
        for (k, e) in obs_noise.iter().enumerate() {
            y[(k, s)] = e + (0..n).map(|i| f[(i, k)] * theta[i]).sum::<f64>();
        }
    }
    Ok(y)
}

/// Draw from `multi_normal_prec(mu, omega)` with precision matrix `omega`: `mu + L⁻ᵀ z` for
/// standard normal `z`, where `omega = L Lᵀ`.
///
/// # Errors
///
/// Dimension mismatches, a non-finite `mu`, and an `omega` that is not symmetric positive
/// definite.
pub fn multi_normal_prec_rng<R: Rng + ?Sized>(
    mu: &[f64],
    omega: &Matrix<f64>,
    rng: &mut R,
) -> Result<Vec<f64>> {
    const F: &str = "multi_normal_prec_rng";
    dims(F, "Omega", square(F, "Omega", omega)?, mu.len())?;
    finite(F, "mu", mu)?;
    let l = cholesky_decompose(omega)?;
    let z: Vec<f64> = (0..mu.len()).map(|_| std_normal(rng)).collect();
    let x = solve_lower_transpose(&l, &z);
    Ok(mu.iter().zip(x).map(|(m, e)| m + e).collect())
}

/// Draw from `matrix_normal_prec(mu, sigma, d)`: the `N` by `P` matrix `mu + Lσ⁻ᵀ Z Ld⁻¹` for a
/// standard normal `Z`, where `sigma = Lσ Lσᵀ` is the row precision (`N` by `N`) and
/// `d = Ld Ldᵀ` the column precision (`P` by `P`).
///
/// # Errors
///
/// Dimension mismatches, a non-finite `mu`, and precision matrices that are not symmetric
/// positive definite.
pub fn matrix_normal_prec_rng<R: Rng + ?Sized>(
    mu: &Matrix<f64>,
    sigma: &Matrix<f64>,
    d: &Matrix<f64>,
    rng: &mut R,
) -> Result<Matrix<f64>> {
    const F: &str = "matrix_normal_prec_rng";
    let (n, p) = mu.shape();
    dims(F, "Sigma", square(F, "Sigma", sigma)?, n)?;
    dims(F, "D", square(F, "D", d)?, p)?;
    finite(F, "mu", mu.as_slice())?;
    let ls = cholesky_decompose(sigma)?;
    let ld = cholesky_decompose(d)?;
    // X = Lσ⁻ᵀ Z, one column at a time.
    let mut x = Matrix::from_fn(n, p, |_, _| 0.0);
    for j in 0..p {
        let z: Vec<f64> = (0..n).map(|_| std_normal(rng)).collect();
        for (i, v) in solve_lower_transpose(&ls, &z).into_iter().enumerate() {
            x[(i, j)] = v;
        }
    }
    // Y = X Ld⁻¹, one row at a time: each row y solves Ldᵀ yᵀ = xᵀ.
    let mut y = mu.clone();
    for i in 0..n {
        let row: Vec<f64> = (0..p).map(|j| x[(i, j)]).collect();
        for (j, v) in solve_lower_transpose(&ld, &row).into_iter().enumerate() {
            y[(i, j)] += v;
        }
    }
    Ok(y)
}
