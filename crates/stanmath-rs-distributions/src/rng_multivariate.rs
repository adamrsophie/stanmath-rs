//! Random number generation for multivariate distributions.
//!
//! Each `*_rng` function validates its parameters like the corresponding log density and draws
//! one value. Pass any `rand::Rng`, for example a seeded `rand::rngs::StdRng`.

use rand::Rng;
use rand::distr::Distribution;
use rand_distr as rd;
use stanmath_rs_core::{Error, Matrix, Result};
use stanmath_rs_linalg::{cholesky_decompose, inverse_spd, multiply};

fn std_normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    rd::StandardNormal.sample(rng)
}

/// Draw from `gamma(shape, scale)`.
fn gamma<R: Rng + ?Sized>(
    function: &'static str,
    shape: f64,
    scale: f64,
    rng: &mut R,
) -> Result<f64> {
    let d = rd::Gamma::new(shape, scale)
        .map_err(|_| Error::parameter(function, "shape", shape, "positive finite"))?;
    Ok(d.sample(rng))
}

fn beta<R: Rng + ?Sized>(function: &'static str, a: f64, b: f64, rng: &mut R) -> Result<f64> {
    let d = rd::Beta::new(a, b)
        .map_err(|_| Error::parameter(function, "shape", a, "positive finite"))?;
    Ok(d.sample(rng))
}

fn square(function: &'static str, name: &str, m: &Matrix<f64>) -> Result<usize> {
    if m.is_square() {
        Ok(m.nrows())
    } else {
        Err(Error::dimension(
            function,
            format!(
                "{name} is {}x{}, expected a square matrix",
                m.nrows(),
                m.ncols()
            ),
        ))
    }
}

fn dims(function: &'static str, what: &str, got: usize, expected: usize) -> Result<()> {
    if got == expected {
        Ok(())
    } else {
        Err(Error::dimension(
            function,
            format!("{what} has size {got}, expected {expected}"),
        ))
    }
}

fn finite(function: &'static str, name: &'static str, xs: &[f64]) -> Result<()> {
    match xs.iter().find(|v| !v.is_finite()) {
        Some(&v) => Err(Error::parameter(function, name, v, "finite")),
        None => Ok(()),
    }
}

fn symmetrize(m: &Matrix<f64>) -> Matrix<f64> {
    let n = m.nrows();
    Matrix::from_fn(n, n, |i, j| 0.5 * (m[(i, j)] + m[(j, i)]))
}

/// `mu + L z` for standard normal `z`, reading only the lower triangle of `l`.
fn normal_from_cholesky<R: Rng + ?Sized>(
    mu: &[f64],
    l: &Matrix<f64>,
    scale: f64,
    rng: &mut R,
) -> Vec<f64> {
    let k = mu.len();
    let z: Vec<f64> = (0..k).map(|_| std_normal(rng)).collect();
    (0..k)
        .map(|i| mu[i] + scale * (0..=i).map(|j| l[(i, j)] * z[j]).sum::<f64>())
        .collect()
}

/// Draw from `multi_normal_cholesky(mu, L)`: `mu + L z` with standard normal `z`.
pub fn multi_normal_cholesky_rng<R: Rng + ?Sized>(
    mu: &[f64],
    l: &Matrix<f64>,
    rng: &mut R,
) -> Result<Vec<f64>> {
    const F: &str = "multi_normal_cholesky_rng";
    dims(F, "L", square(F, "L", l)?, mu.len())?;
    finite(F, "mu", mu)?;
    Ok(normal_from_cholesky(mu, l, 1.0, rng))
}

/// Draw from `multi_normal(mu, sigma)`.
///
/// # Errors
///
/// Dimension mismatches, a non-finite `mu`, and a `sigma` that is not symmetric positive
/// definite.
pub fn multi_normal_rng<R: Rng + ?Sized>(
    mu: &[f64],
    sigma: &Matrix<f64>,
    rng: &mut R,
) -> Result<Vec<f64>> {
    const F: &str = "multi_normal_rng";
    dims(F, "Sigma", square(F, "Sigma", sigma)?, mu.len())?;
    finite(F, "mu", mu)?;
    let l = cholesky_decompose(sigma)?;
    Ok(normal_from_cholesky(mu, &l, 1.0, rng))
}

fn student_t_from_cholesky<R: Rng + ?Sized>(
    function: &'static str,
    nu: f64,
    mu: &[f64],
    l: &Matrix<f64>,
    rng: &mut R,
) -> Result<Vec<f64>> {
    if nu.is_nan() || !(nu > 0.0) {
        return Err(Error::parameter(function, "nu", nu, "positive"));
    }
    finite(function, "mu", mu)?;
    if nu.is_infinite() {
        return Ok(normal_from_cholesky(mu, l, 1.0, rng));
    }
    // w = chi_square(nu) / nu
    let w = gamma(function, 0.5 * nu, 2.0 / nu, rng)?;
    Ok(normal_from_cholesky(mu, l, 1.0 / w.sqrt(), rng))
}

/// Draw from `multi_student_t(nu, mu, sigma)`: a multivariate normal draw divided by the square
/// root of an independent `chi_square(nu) / nu` draw.
pub fn multi_student_t_rng<R: Rng + ?Sized>(
    nu: f64,
    mu: &[f64],
    sigma: &Matrix<f64>,
    rng: &mut R,
) -> Result<Vec<f64>> {
    const F: &str = "multi_student_t_rng";
    dims(F, "Sigma", square(F, "Sigma", sigma)?, mu.len())?;
    let l = cholesky_decompose(sigma)?;
    student_t_from_cholesky(F, nu, mu, &l, rng)
}

/// Draw from `multi_student_t_cholesky(nu, mu, L)`.
pub fn multi_student_t_cholesky_rng<R: Rng + ?Sized>(
    nu: f64,
    mu: &[f64],
    l: &Matrix<f64>,
    rng: &mut R,
) -> Result<Vec<f64>> {
    const F: &str = "multi_student_t_cholesky_rng";
    dims(F, "L", square(F, "L", l)?, mu.len())?;
    student_t_from_cholesky(F, nu, mu, l, rng)
}

/// Draw from `dirichlet(alpha)`.
///
/// Gamma draws are formed on the log scale, `ln G_α = ln G_{α+1} + ln(U) / α`, so concentrations
/// far below one do not underflow to an all zero draw.
pub fn dirichlet_rng<R: Rng + ?Sized>(alpha: &[f64], rng: &mut R) -> Result<Vec<f64>> {
    const F: &str = "dirichlet_rng";
    if let Some(&a) = alpha.iter().find(|a| !(**a > 0.0 && a.is_finite())) {
        return Err(Error::parameter(F, "alpha", a, "positive finite"));
    }
    let mut log_g = Vec::with_capacity(alpha.len());
    for &a in alpha {
        let g = gamma(F, a + 1.0, 1.0, rng)?;
        // ln(U) for a uniform U is minus a standard exponential draw.
        let e: f64 = rd::Exp1.sample(rng);
        log_g.push(g.ln() - e / a);
    }
    let lse = stanmath_rs_core::special::log_sum_exp_slice(&log_g);
    Ok(log_g.into_iter().map(|v| (v - lse).exp()).collect())
}

/// Draw the Cholesky factor of a `k` by `k` correlation matrix from `lkj_corr_cholesky(eta)`.
///
/// Uses the C-vine construction of Lewandowski, Kurowicka, and Joe (2009): the partial
/// correlations at level `i` (counting from 1) are independent and `2 Beta(b_i, b_i) - 1` with
/// `b_i = eta + (k - 1 - i) / 2`, and they map to the rows of the factor.
pub fn lkj_corr_cholesky_rng<R: Rng + ?Sized>(
    k: usize,
    eta: f64,
    rng: &mut R,
) -> Result<Matrix<f64>> {
    const F: &str = "lkj_corr_cholesky_rng";
    if !(eta > 0.0 && eta.is_finite()) {
        return Err(Error::parameter(F, "eta", eta, "positive finite"));
    }
    let mut partial = Matrix::zeros(k, k);
    let mut b = eta + 0.5 * (k as f64 - 1.0);
    for i in 0..k.saturating_sub(1) {
        b -= 0.5;
        for j in i + 1..k {
            partial[(i, j)] = 2.0 * beta(F, b, b, rng)? - 1.0;
        }
    }
    let mut l = Matrix::zeros(k, k);
    if k == 0 {
        return Ok(l);
    }
    l[(0, 0)] = 1.0;
    for j in 1..k {
        // remaining = 1 - Σ_{m < i} L(j, m)^2
        let mut remaining = 1.0;
        for i in 0..j {
            let v = partial[(i, j)] * remaining.sqrt();
            l[(j, i)] = v;
            remaining -= v * v;
        }
        l[(j, j)] = remaining.max(0.0).sqrt();
    }
    Ok(l)
}

/// Draw a `k` by `k` correlation matrix from `lkj_corr(eta)`.
pub fn lkj_corr_rng<R: Rng + ?Sized>(k: usize, eta: f64, rng: &mut R) -> Result<Matrix<f64>> {
    let l = lkj_corr_cholesky_rng(k, eta, rng)?;
    let mut c = multiply(&l, &l.transpose())?;
    for i in 0..k {
        c[(i, i)] = 1.0;
    }
    Ok(symmetrize(&c))
}

fn check_nu(function: &'static str, nu: f64, k: usize) -> Result<()> {
    if nu > k as f64 - 1.0 && nu.is_finite() {
        Ok(())
    } else {
        Err(Error::parameter(
            function,
            "nu",
            nu,
            "finite and greater than K - 1",
        ))
    }
}

/// `L_S A`, where `A` is the lower triangular Bartlett factor: `A(i, i)^2` is
/// `chi_square(nu - i)` (counting `i` from 0) and the entries below the diagonal are standard
/// normal. The result is the Cholesky factor of a `wishart(nu, L_S L_Sᵀ)` draw.
fn bartlett<R: Rng + ?Sized>(
    function: &'static str,
    nu: f64,
    l_s: &Matrix<f64>,
    rng: &mut R,
) -> Result<Matrix<f64>> {
    let k = l_s.nrows();
    let mut a = Matrix::zeros(k, k);
    for i in 0..k {
        a[(i, i)] = gamma(function, 0.5 * (nu - i as f64), 2.0, rng)?.sqrt();
        for j in 0..i {
            a[(i, j)] = std_normal(rng);
        }
    }
    let lower = Matrix::from_fn(k, k, |i, j| if i >= j { l_s[(i, j)] } else { 0.0 });
    multiply(&lower, &a)
}

/// Draw from `wishart(nu, s)` by the Bartlett decomposition. Requires `nu > K - 1` and `s`
/// symmetric positive definite.
pub fn wishart_rng<R: Rng + ?Sized>(nu: f64, s: &Matrix<f64>, rng: &mut R) -> Result<Matrix<f64>> {
    const F: &str = "wishart_rng";
    let k = square(F, "S", s)?;
    check_nu(F, nu, k)?;
    let l = bartlett(F, nu, &cholesky_decompose(s)?, rng)?;
    Ok(symmetrize(&multiply(&l, &l.transpose())?))
}

/// Draw the Cholesky factor of a `wishart(nu, L_S L_Sᵀ)` matrix, given the lower Cholesky factor
/// `l_s` of the scale (only its lower triangle is read).
pub fn wishart_cholesky_rng<R: Rng + ?Sized>(
    nu: f64,
    l_s: &Matrix<f64>,
    rng: &mut R,
) -> Result<Matrix<f64>> {
    const F: &str = "wishart_cholesky_rng";
    let k = square(F, "L_S", l_s)?;
    check_nu(F, nu, k)?;
    bartlett(F, nu, l_s, rng)
}

/// Draw from `inv_wishart(nu, s)` as the inverse of a `wishart(nu, s⁻¹)` draw.
pub fn inv_wishart_rng<R: Rng + ?Sized>(
    nu: f64,
    s: &Matrix<f64>,
    rng: &mut R,
) -> Result<Matrix<f64>> {
    const F: &str = "inv_wishart_rng";
    let k = square(F, "S", s)?;
    check_nu(F, nu, k)?;
    let s_inv = symmetrize(&inverse_spd(s)?);
    let l = bartlett(F, nu, &cholesky_decompose(&s_inv)?, rng)?;
    let w = symmetrize(&multiply(&l, &l.transpose())?);
    Ok(symmetrize(&inverse_spd(&w)?))
}

/// Draw the Cholesky factor of an `inv_wishart(nu, L_S L_Sᵀ)` matrix, given the lower Cholesky
/// factor `l_s` of the scale (only its lower triangle is read).
pub fn inv_wishart_cholesky_rng<R: Rng + ?Sized>(
    nu: f64,
    l_s: &Matrix<f64>,
    rng: &mut R,
) -> Result<Matrix<f64>> {
    const F: &str = "inv_wishart_cholesky_rng";
    let k = square(F, "L_S", l_s)?;
    let lower = Matrix::from_fn(k, k, |i, j| if i >= j { l_s[(i, j)] } else { 0.0 });
    let s = symmetrize(&multiply(&lower, &lower.transpose())?);
    check_nu(F, nu, k)?;
    cholesky_decompose(&inv_wishart_rng(nu, &s, rng)?)
}
