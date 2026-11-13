//! More multivariate distributions: the precision and Student t forms of the multivariate normal,
//! Gaussian process likelihoods, the Gaussian dynamic linear model, the Wishart and inverse
//! Wishart families, LKJ densities of correlation and covariance matrices, and the matrix normal.
//!
//! Each density has a full form and a `propto` form (`_lupdf`). Matrices share the scalar type
//! `T`, so they are parameters whenever `T` is differentiable; vector arguments that are often
//! data accept any element type that lifts to `T`.

use stanmath_rs_autodiff::functions::{lgamma, log1p};
use stanmath_rs_autodiff::{Lift, Real};
use stanmath_rs_core::special::{HALF_LN_2PI, LN_PI};
use stanmath_rs_core::{Error, Matrix, Result};
use stanmath_rs_linalg::{
    add, cholesky_decompose, log_determinant_spd, mdivide_left_tri_low,
    mdivide_left_tri_low_transpose, multiply, quad_form, subtract,
};

use crate::kernel::{is_param, keep};
use crate::multivariate::{lkj_constant, multi_normal_cholesky_core};

const LN_2: f64 = core::f64::consts::LN_2;

/// Tolerance for unit diagonals and simplexes, as Stan's `CONSTRAINT_TOLERANCE`.
const TOL: f64 = 1e-8;

fn lift_all<T: Real, E: Lift<T> + Copy>(xs: &[E]) -> Vec<T> {
    xs.iter().map(|&x| x.lift()).collect()
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

fn square<T>(function: &'static str, name: &str, m: &Matrix<T>) -> Result<usize> {
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

fn obs_not_nan<T: Real>(function: &'static str, name: &'static str, xs: &[T]) -> Result<()> {
    match xs.iter().find(|v| v.is_nan()) {
        Some(v) => Err(Error::observation(function, name, v.value(), "not NaN")),
        None => Ok(()),
    }
}

fn param_finite<T: Real>(function: &'static str, name: &'static str, xs: &[T]) -> Result<()> {
    match xs.iter().find(|v| !v.is_finite()) {
        Some(v) => Err(Error::parameter(function, name, v.value(), "finite")),
        None => Ok(()),
    }
}

fn param_symmetric<T: Real>(
    function: &'static str,
    name: &'static str,
    m: &Matrix<T>,
) -> Result<()> {
    let n = square(function, name, m)?;
    for j in 0..n {
        for i in j + 1..n {
            if !((m[(i, j)].value() - m[(j, i)].value()).abs() <= TOL) {
                return Err(Error::NotSymmetric {
                    function,
                    row: i,
                    col: j,
                });
            }
        }
    }
    Ok(())
}

/// Checks that `l` is square, lower triangular, and has a positive diagonal.
fn check_cholesky_factor<T: Real>(
    function: &'static str,
    name: &'static str,
    l: &Matrix<T>,
    observation: bool,
) -> Result<usize> {
    let k = square(function, name, l)?;
    for j in 0..k {
        let d = l[(j, j)].value();
        let upper_zero = (0..j).all(|i| l[(i, j)].value() == 0.0);
        if !(d > 0.0) || !upper_zero {
            return Err(if observation {
                Error::observation(function, name, d, "a Cholesky factor")
            } else {
                Error::parameter(function, name, d, "a Cholesky factor")
            });
        }
    }
    Ok(k)
}

/// `ln det(L Lᵀ)` from the Cholesky factor `L`.
fn log_det_cholesky<T: Real>(l: &Matrix<T>) -> T {
    (0..l.nrows()).map(|i| l[(i, i)].ln()).sum::<T>() * 2.0
}

fn sum_squares<T: Real>(m: &Matrix<T>) -> T {
    m.as_slice().iter().map(|v| v.square()).sum()
}

fn symmetrize<T: Real>(m: &Matrix<T>) -> Matrix<T> {
    let n = m.nrows();
    Matrix::from_fn(n, n, |i, j| (m[(i, j)] + m[(j, i)]) * 0.5)
}

/// Log multivariate gamma `ln Γ_k(a) = k (k - 1) / 4 ln π + Σ_{j=1}^k ln Γ(a + (1 - j) / 2)`.
fn lmgamma<T: Real>(k: usize, a: T) -> T {
    let kf = k as f64;
    let mut r = T::from_f64(kf * (kf - 1.0) * 0.25 * LN_PI);
    for j in 1..=k {
        r += lgamma(a + (1.0 - j as f64) * 0.5);
    }
    r
}

fn check_nu<T: Real>(function: &'static str, nu: T, k: usize) -> Result<()> {
    if nu.value() > k as f64 - 1.0 {
        Ok(())
    } else {
        Err(Error::parameter(
            function,
            "nu",
            nu.value(),
            "greater than K - 1",
        ))
    }
}

propto_pair! {
    /// Multivariate normal log density with location `mu` and precision matrix `omega`.
    ///
    /// `ln N(y | mu, omega⁻¹) = ½ ln det omega - ½ (y - mu)ᵀ omega (y - mu) - k/2 ln(2π)`.
    ///
    /// # Errors
    ///
    /// Dimension mismatches, a NaN in `y`, a non-finite `mu`, and an `omega` that is not
    /// symmetric positive definite.
    multi_normal_prec_lpdf, multi_normal_prec_lupdf <T, Y, M>(
        y: &[Y],
        mu: &[M],
        omega: &Matrix<T>,
    ) -> Result<T>
    where { T: Real, Y: Lift<T> + Copy, M: Lift<T> + Copy }
    |propto| { multi_normal_prec_with(y, mu, omega, propto) }
}

fn multi_normal_prec_with<T, Y, M>(y: &[Y], mu: &[M], omega: &Matrix<T>, propto: bool) -> Result<T>
where
    T: Real,
    Y: Lift<T> + Copy,
    M: Lift<T> + Copy,
{
    const F: &str = "multi_normal_prec_lpdf";
    let (py, pm, pl) = (is_param::<T, Y>(), is_param::<T, M>(), is_param::<T, T>());
    let k = y.len();
    dims(F, "mu", mu.len(), k)?;
    dims(F, "Omega", square(F, "Omega", omega)?, k)?;
    let y: Vec<T> = lift_all(y);
    let mu: Vec<T> = lift_all(mu);
    obs_not_nan(F, "y", &y)?;
    param_finite(F, "mu", &mu)?;
    if k == 0 {
        return Ok(T::zero());
    }
    let log_det = log_determinant_spd(omega)?;
    let mut lp = T::zero();
    if keep(propto, &[pl]) {
        lp += log_det * 0.5;
    }
    if keep(propto, &[py, pm, pl]) {
        let diff: Vec<T> = y.iter().zip(&mu).map(|(&a, &b)| a - b).collect();
        lp -= quad_form(omega, &diff)? * 0.5;
    }
    if keep(propto, &[]) {
        lp -= HALF_LN_2PI * k as f64;
    }
    Ok(lp)
}

/// Sum of multivariate normal log densities of the columns of a `k` by `n` matrix of differences
/// `y - mu`, with covariance `L Lᵀ`.
fn multi_normal_cholesky_many<T: Real>(
    function: &'static str,
    ys: &[Vec<T>],
    mus: &[Vec<T>],
    l: &Matrix<T>,
    propto: bool,
    [py, pm]: [bool; 2],
) -> Result<T> {
    let n = ys.len();
    if !(mus.len() == 1 || mus.len() == n) {
        return Err(Error::dimension(
            function,
            format!("{} locations for {} observations", mus.len(), n),
        ));
    }
    let k = square(function, "L", l)?;
    for y in ys {
        dims(function, "y", y.len(), k)?;
        obs_not_nan(function, "y", y)?;
    }
    for mu in mus {
        dims(function, "mu", mu.len(), k)?;
        param_finite(function, "mu", mu)?;
    }
    if n == 0 || k == 0 {
        return Ok(T::zero());
    }
    let pl = is_param::<T, T>();
    let mut lp = T::zero();
    if keep(propto, &[py, pm, pl]) {
        let mut data = Vec::with_capacity(k * n);
        for (i, y) in ys.iter().enumerate() {
            let mu = &mus[if mus.len() == 1 { 0 } else { i }];
            data.extend(y.iter().zip(mu).map(|(&a, &b)| a - b));
        }
        let z = mdivide_left_tri_low(l, &Matrix::from_column_major(k, n, data))?;
        lp -= sum_squares(&z) * 0.5;
    }
    if keep(propto, &[pl]) {
        lp -= (0..k).map(|i| l[(i, i)].ln()).sum::<T>() * n as f64;
    }
    if keep(propto, &[]) {
        lp -= HALF_LN_2PI * (k * n) as f64;
    }
    Ok(lp)
}

/// `y` and `mu` lifted to `T`, with flags for whether each is a parameter.
type Lifted<T> = (Vec<Vec<T>>, Vec<Vec<T>>, [bool; 2]);

fn lift_arrays<T, Y, M>(y: &[Vec<Y>], mu: &[Vec<M>]) -> Lifted<T>
where
    T: Real,
    Y: Lift<T> + Copy,
    M: Lift<T> + Copy,
{
    let ys = y.iter().map(|v| lift_all(v)).collect();
    let mus = mu.iter().map(|v| lift_all(v)).collect();
    (ys, mus, [is_param::<T, Y>(), is_param::<T, M>()])
}

propto_pair! {
    /// Sum of multivariate normal log densities over an array of observations `y`, with
    /// covariance `L Lᵀ` given by its lower Cholesky factor.
    ///
    /// `mu` holds either one location shared by every observation or one location per
    /// observation. The triangular solve is done once for all observations. Returns zero for an
    /// empty array.
    multi_normal_cholesky_lpdf_array, multi_normal_cholesky_lupdf_array <T, Y, M>(
        y: &[Vec<Y>],
        mu: &[Vec<M>],
        l: &Matrix<T>,
    ) -> Result<T>
    where { T: Real, Y: Lift<T> + Copy, M: Lift<T> + Copy }
    |propto| {
        let (ys, mus, params) = lift_arrays(y, mu);
        multi_normal_cholesky_many("multi_normal_cholesky_lpdf", &ys, &mus, l, propto, params)
    }
}

propto_pair! {
    /// Sum of multivariate normal log densities over an array of observations `y`, with
    /// covariance `sigma`, decomposed once.
    ///
    /// `mu` holds either one location shared by every observation or one location per
    /// observation.
    multi_normal_lpdf_array, multi_normal_lupdf_array <T, Y, M>(
        y: &[Vec<Y>],
        mu: &[Vec<M>],
        sigma: &Matrix<T>,
    ) -> Result<T>
    where { T: Real, Y: Lift<T> + Copy, M: Lift<T> + Copy }
    |propto| {
        const F: &str = "multi_normal_lpdf";
        square(F, "Sigma", sigma)?;
        let l = cholesky_decompose(sigma)?;
        let (ys, mus, params) = lift_arrays(y, mu);
        multi_normal_cholesky_many(F, &ys, &mus, &l, propto, params)
    }
}

/// The multivariate Student t log density with scale `L Lᵀ`; `params` flags whether `y`, `nu`,
/// and `mu` are parameters.
pub(crate) fn multi_student_t_core<T: Real>(
    function: &'static str,
    y: &[T],
    nu: T,
    mu: &[T],
    l: &Matrix<T>,
    propto: bool,
    [py, pn, pm]: [bool; 3],
) -> Result<T> {
    let k = y.len();
    dims(function, "mu", mu.len(), k)?;
    dims(function, "scale matrix", l.nrows(), k)?;
    if nu.is_nan() || !(nu.value() > 0.0) {
        return Err(Error::parameter(function, "nu", nu.value(), "positive"));
    }
    obs_not_nan(function, "y", y)?;
    param_finite(function, "mu", mu)?;
    if k == 0 {
        return Ok(T::zero());
    }
    if !nu.is_finite() {
        return multi_normal_cholesky_core(function, y, mu, l, propto, [py, pm]);
    }
    let kf = k as f64;
    let pl = is_param::<T, T>();
    let mut lp = T::zero();
    if keep(propto, &[pn]) {
        lp += lgamma((nu + kf) * 0.5) - lgamma(nu * 0.5) - nu.ln() * (kf * 0.5);
    }
    if keep(propto, &[]) {
        lp -= LN_PI * kf * 0.5;
    }
    if keep(propto, &[pl]) {
        lp -= (0..k).map(|i| l[(i, i)].ln()).sum::<T>();
    }
    if keep(propto, &[py, pn, pm, pl]) {
        let diff: Vec<T> = y.iter().zip(mu).map(|(&a, &b)| a - b).collect();
        let z = mdivide_left_tri_low(l, &Matrix::column_vector(diff))?;
        lp -= (nu + kf) * 0.5 * log1p(sum_squares(&z) / nu);
    }
    Ok(lp)
}

/// Parameter flags of `y`, `nu`, and `mu` for [`multi_student_t_core`].
fn student_t_params<T, Y: Lift<T>, N: Lift<T>, M: Lift<T>>() -> [bool; 3] {
    [is_param::<T, Y>(), is_param::<T, N>(), is_param::<T, M>()]
}

propto_pair! {
    /// Multivariate Student t log density with `nu` degrees of freedom, location `mu`, and scale
    /// matrix `sigma`.
    ///
    /// `ln Γ((ν + k)/2) - ln Γ(ν/2) - k/2 ln(νπ) - ½ ln det Σ
    /// - (ν + k)/2 ln(1 + (y - μ)ᵀ Σ⁻¹ (y - μ) / ν)`. An infinite `nu` gives the multivariate
    /// normal density. Requires `nu > 0`.
    multi_student_t_lpdf, multi_student_t_lupdf <T, Y, N, M>(
        y: &[Y],
        nu: N,
        mu: &[M],
        sigma: &Matrix<T>,
    ) -> Result<T>
    where { T: Real, Y: Lift<T> + Copy, N: Lift<T>, M: Lift<T> + Copy }
    |propto| {
        const F: &str = "multi_student_t_lpdf";
        dims(F, "Sigma", square(F, "Sigma", sigma)?, y.len())?;
        let l = cholesky_decompose(sigma)?;
        let params = student_t_params::<T, Y, N, M>();
        multi_student_t_core(F, &lift_all(y), nu.lift(), &lift_all(mu), &l, propto, params)
    }
}

propto_pair! {
    /// Multivariate Student t log density with the scale matrix given by its lower Cholesky
    /// factor `l` (only the lower triangle is read). See [`multi_student_t_lpdf`].
    multi_student_t_cholesky_lpdf, multi_student_t_cholesky_lupdf <T, Y, N, M>(
        y: &[Y],
        nu: N,
        mu: &[M],
        l: &Matrix<T>,
    ) -> Result<T>
    where { T: Real, Y: Lift<T> + Copy, N: Lift<T>, M: Lift<T> + Copy }
    |propto| {
        const F: &str = "multi_student_t_cholesky_lpdf";
        dims(F, "L", square(F, "L", l)?, y.len())?;
        let params = student_t_params::<T, Y, N, M>();
        multi_student_t_core(F, &lift_all(y), nu.lift(), &lift_all(mu), l, propto, params)
    }
}

fn multi_gp_core<T: Real>(
    function: &'static str,
    y: &Matrix<T>,
    l: &Matrix<T>,
    w: Vec<T>,
    propto: bool,
    pw: bool,
) -> Result<T> {
    let (k, n) = y.shape();
    dims(function, "w", w.len(), k)?;
    dims(function, "kernel matrix", l.nrows(), n)?;
    obs_not_nan(function, "y", y.as_slice())?;
    if let Some(v) = w.iter().find(|v| !(v.value() > 0.0 && v.is_finite())) {
        return Err(Error::parameter(
            function,
            "w",
            v.value(),
            "positive finite",
        ));
    }
    if k == 0 || n == 0 {
        return Ok(T::zero());
    }
    // Columns of z are L⁻¹ y_i for the rows y_i of y.
    let z = mdivide_left_tri_low(l, &y.transpose())?;
    let pl = is_param::<T, T>();
    let mut lp = T::zero();
    if keep(propto, &[pl]) {
        lp -= (0..n).map(|i| l[(i, i)].ln()).sum::<T>() * k as f64;
    }
    if keep(propto, &[]) {
        lp -= HALF_LN_2PI * (k * n) as f64;
    }
    for (i, &wi) in w.iter().enumerate() {
        if keep(propto, &[pw]) {
            lp += wi.ln() * (n as f64 * 0.5);
        }
        if keep(propto, &[pl, pw]) {
            lp -= wi * z.column(i).iter().map(|v| v.square()).sum::<T>() * 0.5;
        }
    }
    Ok(lp)
}

propto_pair! {
    /// Log density of a multi-output Gaussian process: each row `y_i` of the `k` by `n` matrix
    /// `y` is independently `N(0, sigma / w_i)`.
    ///
    /// Requires `sigma` symmetric positive definite and every `w_i` positive finite.
    multi_gp_lpdf, multi_gp_lupdf <T, W>(y: &Matrix<T>, sigma: &Matrix<T>, w: &[W]) -> Result<T>
    where { T: Real, W: Lift<T> + Copy }
    |propto| {
        const F: &str = "multi_gp_lpdf";
        dims(F, "Sigma", square(F, "Sigma", sigma)?, y.ncols())?;
        let l = cholesky_decompose(sigma)?;
        multi_gp_core(F, y, &l, lift_all(w), propto, is_param::<T, W>())
    }
}

propto_pair! {
    /// [`multi_gp_lpdf`] with the kernel matrix given by its lower Cholesky factor `l`.
    multi_gp_cholesky_lpdf, multi_gp_cholesky_lupdf <T, W>(
        y: &Matrix<T>,
        l: &Matrix<T>,
        w: &[W],
    ) -> Result<T>
    where { T: Real, W: Lift<T> + Copy }
    |propto| {
        const F: &str = "multi_gp_cholesky_lpdf";
        dims(F, "L", square(F, "L", l)?, y.ncols())?;
        multi_gp_core(F, y, l, lift_all(w), propto, is_param::<T, W>())
    }
}

propto_pair! {
    /// Log likelihood of a Gaussian dynamic linear model, computed with the Kalman filter.
    ///
    /// The model is `y_t ~ N(Fᵀ θ_t, V)` and `θ_t ~ N(G θ_{t-1}, W)`, with `θ_0 ~ N(m0, C0)`.
    /// The observations are the columns of the `r` by `T` matrix `y`; `F` is `n` by `r`, `G` and
    /// `W` are `n` by `n`, `V` is `r` by `r`, `m0` has length `n`, and `C0` is `n` by `n`.
    ///
    /// # Errors
    ///
    /// Dimension mismatches, a NaN observation, non-finite `F`, `G`, or `m0`, asymmetric `V`,
    /// `W`, or `C0`, and a one step ahead covariance that is not positive definite.
    gaussian_dlm_obs_lpdf, gaussian_dlm_obs_lupdf <T>(
        y: &Matrix<T>,
        f: &Matrix<T>,
        g: &Matrix<T>,
        v: &Matrix<T>,
        w: &Matrix<T>,
        m0: &[T],
        c0: &Matrix<T>,
    ) -> Result<T>
    where { T: Real }
    |propto| { gaussian_dlm_obs_with(y, f, g, v, w, m0, c0, propto) }
}

#[allow(clippy::too_many_arguments)]
fn gaussian_dlm_obs_with<T: Real>(
    y: &Matrix<T>,
    f: &Matrix<T>,
    g: &Matrix<T>,
    v: &Matrix<T>,
    w: &Matrix<T>,
    m0: &[T],
    c0: &Matrix<T>,
    propto: bool,
) -> Result<T> {
    const F: &str = "gaussian_dlm_obs_lpdf";
    let (r, steps) = y.shape();
    let n = f.nrows();
    dims(F, "F columns", f.ncols(), r)?;
    dims(F, "G", square(F, "G", g)?, n)?;
    dims(F, "V", square(F, "V", v)?, r)?;
    dims(F, "W", square(F, "W", w)?, n)?;
    dims(F, "m0", m0.len(), n)?;
    dims(F, "C0", square(F, "C0", c0)?, n)?;
    obs_not_nan(F, "y", y.as_slice())?;
    param_finite(F, "F", f.as_slice())?;
    param_finite(F, "G", g.as_slice())?;
    param_finite(F, "m0", m0)?;
    param_symmetric(F, "V", v)?;
    param_symmetric(F, "W", w)?;
    param_symmetric(F, "C0", c0)?;
    if r == 0 || steps == 0 {
        return Ok(T::zero());
    }
    let ft = f.transpose();
    let gt = g.transpose();
    let mut m = Matrix::column_vector(m0.to_vec());
    let mut c = c0.clone();
    let mut lp = T::zero();
    for t in 0..steps {
        // Prediction: a = G m, R = G C Gᵀ + W.
        let a = multiply(g, &m)?;
        let rr = symmetrize(&add(&multiply(&multiply(g, &c)?, &gt)?, w)?);
        // One step ahead forecast: f = Fᵀ a, Q = Fᵀ R F + V.
        let fa = multiply(&ft, &a)?;
        let rf = multiply(&rr, f)?;
        let q = symmetrize(&add(&multiply(&ft, &rf)?, v)?);
        let e: Vec<T> = (0..r).map(|i| y[(i, t)] - fa[(i, 0)]).collect();
        let lq = cholesky_decompose(&q)?;
        let z = mdivide_left_tri_low(&lq, &Matrix::column_vector(e.clone()))?;
        if keep(propto, &[is_param::<T, T>()]) {
            lp -= sum_squares(&z) * 0.5 + (0..r).map(|i| lq[(i, i)].ln()).sum::<T>();
        }
        if keep(propto, &[]) {
            lp -= HALF_LN_2PI * r as f64;
        }
        // Update: Aᵀ = Q⁻¹ (R F)ᵀ, m = a + A e, C = R - (R F) Q⁻¹ (R F)ᵀ.
        let at = mdivide_left_tri_low_transpose(&lq, &mdivide_left_tri_low(&lq, &rf.transpose())?)?;
        m = add(&a, &multiply(&at.transpose(), &Matrix::column_vector(e))?)?;
        c = symmetrize(&subtract(&rr, &multiply(&rf, &at)?)?);
    }
    Ok(lp)
}

propto_pair! {
    /// Wishart log density of a symmetric positive definite `w` with `nu` degrees of freedom and
    /// scale matrix `s`.
    ///
    /// `(ν - K - 1)/2 ln det W - ½ tr(S⁻¹ W) - νK/2 ln 2 - ν/2 ln det S - ln Γ_K(ν/2)`. Requires
    /// `nu > K - 1`.
    wishart_lpdf, wishart_lupdf <T, N>(w: &Matrix<T>, nu: N, s: &Matrix<T>) -> Result<T>
    where { T: Real, N: Lift<T> }
    |propto| { wishart_with(w, nu, s, propto) }
}

/// The terms of the Wishart family that depend only on `nu`: `-νK/2 ln 2 - ln Γ_K(ν/2)`.
fn wishart_nu_terms<T: Real>(nu: T, k: usize) -> T {
    -(nu * (k as f64 * 0.5 * LN_2)) - lmgamma(k, nu * 0.5)
}

fn wishart_with<T: Real, N: Lift<T>>(
    w: &Matrix<T>,
    nu: N,
    s: &Matrix<T>,
    propto: bool,
) -> Result<T> {
    const F: &str = "wishart_lpdf";
    let (pn, pm) = (is_param::<T, N>(), is_param::<T, T>());
    let k = square(F, "W", w)?;
    dims(F, "S", square(F, "S", s)?, k)?;
    let nu: T = nu.lift();
    check_nu(F, nu, k)?;
    let lw = cholesky_decompose(w)?;
    let ls = cholesky_decompose(s)?;
    if k == 0 {
        return Ok(T::zero());
    }
    let kf = k as f64;
    let mut lp = T::zero();
    if keep(propto, &[pm, pn]) {
        lp +=
            log_det_cholesky(&lw) * ((nu - (kf + 1.0)) * 0.5) - log_det_cholesky(&ls) * (nu * 0.5);
    }
    if keep(propto, &[pm]) {
        lp -= sum_squares(&mdivide_left_tri_low(&ls, &lw)?) * 0.5;
    }
    if keep(propto, &[pn]) {
        lp += wishart_nu_terms(nu, k);
    }
    Ok(lp)
}

propto_pair! {
    /// Log density of the lower Cholesky factor `l_w` of a Wishart matrix `W = L_W L_Wᵀ` with
    /// `nu` degrees of freedom and scale `L_S L_Sᵀ`.
    ///
    /// This is [`wishart_lpdf`] of `W` plus the log Jacobian `K ln 2 + Σ_i (K - i + 1) ln L_ii`
    /// of the map from `L_W` to `W`. Requires `nu > K - 1` and both arguments lower triangular
    /// with positive diagonals.
    wishart_cholesky_lpdf, wishart_cholesky_lupdf <T, N>(
        l_w: &Matrix<T>,
        nu: N,
        l_s: &Matrix<T>,
    ) -> Result<T>
    where { T: Real, N: Lift<T> }
    |propto| { wishart_cholesky_with(l_w, nu, l_s, propto) }
}

fn wishart_cholesky_with<T: Real, N: Lift<T>>(
    l_w: &Matrix<T>,
    nu: N,
    l_s: &Matrix<T>,
    propto: bool,
) -> Result<T> {
    const F: &str = "wishart_cholesky_lpdf";
    let (pn, pm) = (is_param::<T, N>(), is_param::<T, T>());
    let k = check_cholesky_factor(F, "L_W", l_w, true)?;
    dims(F, "L_S", check_cholesky_factor(F, "L_S", l_s, false)?, k)?;
    let nu: T = nu.lift();
    check_nu(F, nu, k)?;
    if k == 0 {
        return Ok(T::zero());
    }
    let mut lp = T::zero();
    if keep(propto, &[]) {
        lp += k as f64 * LN_2;
    }
    if keep(propto, &[pm]) {
        lp -= sum_squares(&mdivide_left_tri_low(l_s, l_w)?) * 0.5;
    }
    if keep(propto, &[pn]) {
        lp += wishart_nu_terms(nu, k);
    }
    if keep(propto, &[pm, pn]) {
        lp -= log_det_cholesky(l_s) * (nu * 0.5);
        for i in 0..k {
            lp += l_w[(i, i)].ln() * (nu - (i + 1) as f64);
        }
    }
    Ok(lp)
}

propto_pair! {
    /// Inverse Wishart log density of a symmetric positive definite `w` with `nu` degrees of
    /// freedom and scale matrix `s`.
    ///
    /// `ν/2 ln det S - νK/2 ln 2 - ln Γ_K(ν/2) - (ν + K + 1)/2 ln det W - ½ tr(S W⁻¹)`. Requires
    /// `nu > K - 1`.
    inv_wishart_lpdf, inv_wishart_lupdf <T, N>(w: &Matrix<T>, nu: N, s: &Matrix<T>) -> Result<T>
    where { T: Real, N: Lift<T> }
    |propto| { inv_wishart_with(w, nu, s, propto) }
}

fn inv_wishart_with<T: Real, N: Lift<T>>(
    w: &Matrix<T>,
    nu: N,
    s: &Matrix<T>,
    propto: bool,
) -> Result<T> {
    const F: &str = "inv_wishart_lpdf";
    let (pn, pm) = (is_param::<T, N>(), is_param::<T, T>());
    let k = square(F, "W", w)?;
    dims(F, "S", square(F, "S", s)?, k)?;
    let nu: T = nu.lift();
    check_nu(F, nu, k)?;
    let lw = cholesky_decompose(w)?;
    let ls = cholesky_decompose(s)?;
    if k == 0 {
        return Ok(T::zero());
    }
    let kf = k as f64;
    let mut lp = T::zero();
    if keep(propto, &[pm, pn]) {
        lp +=
            log_det_cholesky(&ls) * (nu * 0.5) - log_det_cholesky(&lw) * ((nu + (kf + 1.0)) * 0.5);
    }
    if keep(propto, &[pn]) {
        lp += wishart_nu_terms(nu, k);
    }
    if keep(propto, &[pm]) {
        lp -= sum_squares(&mdivide_left_tri_low(&lw, &ls)?) * 0.5;
    }
    Ok(lp)
}

propto_pair! {
    /// Log density of the lower Cholesky factor `l_w` of an inverse Wishart matrix
    /// `W = L_W L_Wᵀ` with `nu` degrees of freedom and scale `L_S L_Sᵀ`.
    ///
    /// This is [`inv_wishart_lpdf`] of `W` plus the log Jacobian of the map from `L_W` to `W`.
    inv_wishart_cholesky_lpdf, inv_wishart_cholesky_lupdf <T, N>(
        l_w: &Matrix<T>,
        nu: N,
        l_s: &Matrix<T>,
    ) -> Result<T>
    where { T: Real, N: Lift<T> }
    |propto| { inv_wishart_cholesky_with(l_w, nu, l_s, propto) }
}

fn inv_wishart_cholesky_with<T: Real, N: Lift<T>>(
    l_w: &Matrix<T>,
    nu: N,
    l_s: &Matrix<T>,
    propto: bool,
) -> Result<T> {
    const F: &str = "inv_wishart_cholesky_lpdf";
    let (pn, pm) = (is_param::<T, N>(), is_param::<T, T>());
    let k = check_cholesky_factor(F, "L_W", l_w, true)?;
    dims(F, "L_S", check_cholesky_factor(F, "L_S", l_s, false)?, k)?;
    let nu: T = nu.lift();
    check_nu(F, nu, k)?;
    if k == 0 {
        return Ok(T::zero());
    }
    let mut lp = T::zero();
    if keep(propto, &[]) {
        lp += k as f64 * LN_2;
    }
    if keep(propto, &[pn]) {
        lp += wishart_nu_terms(nu, k);
    }
    if keep(propto, &[pm]) {
        lp -= sum_squares(&mdivide_left_tri_low(l_w, l_s)?) * 0.5;
    }
    if keep(propto, &[pm, pn]) {
        lp += log_det_cholesky(l_s) * (nu * 0.5);
        for i in 0..k {
            lp -= l_w[(i, i)].ln() * (nu + (i + 1) as f64);
        }
    }
    Ok(lp)
}

/// LKJ log density of a correlation matrix whose diagonal has already been checked; `pe` flags
/// whether `eta` is a parameter.
fn lkj_corr_core<T: Real>(
    function: &'static str,
    y: &Matrix<T>,
    eta: T,
    propto: bool,
    pe: bool,
) -> Result<T> {
    if !(eta.value() > 0.0) {
        return Err(Error::parameter(function, "eta", eta.value(), "positive"));
    }
    let k = y.nrows();
    if k == 0 {
        return Ok(T::zero());
    }
    let l = cholesky_decompose(y)?;
    let mut lp = T::zero();
    if keep(propto, &[pe]) {
        lp += lkj_constant(eta, k);
    }
    if keep(propto, &[is_param::<T, T>(), pe]) {
        // (η - 1) ln det y, with ln det y = 2 Σ ln L_ii and L_00 = 1.
        for i in 1..k {
            lp += l[(i, i)].ln() * (eta * 2.0 - 2.0);
        }
    }
    Ok(lp)
}

propto_pair! {
    /// LKJ log density of a correlation matrix `y` with shape `eta`: `(η - 1) ln det y` plus the
    /// normalizing constant.
    ///
    /// Requires `eta > 0` and `y` symmetric positive definite with unit diagonal (within `1e-8`).
    lkj_corr_lpdf, lkj_corr_lupdf <T, E>(y: &Matrix<T>, eta: E) -> Result<T>
    where { T: Real, E: Lift<T> }
    |propto| { lkj_corr_with(y, eta, propto) }
}

fn lkj_corr_with<T: Real, E: Lift<T>>(y: &Matrix<T>, eta: E, propto: bool) -> Result<T> {
    const F: &str = "lkj_corr_lpdf";
    let k = square(F, "y", y)?;
    for i in 0..k {
        let d = y[(i, i)].value();
        if !((d - 1.0).abs() <= TOL) {
            return Err(Error::observation(F, "y", d, "a correlation matrix"));
        }
    }
    lkj_corr_core(F, y, eta.lift(), propto, is_param::<T, E>())
}

propto_pair! {
    /// LKJ log density of a covariance matrix `y`: the standard deviations `sqrt(diag(y))` are
    /// lognormal with locations `mu` and scales `sigma`, and the implied correlation matrix is
    /// LKJ with shape `eta`.
    ///
    /// `mu` and `sigma` have length `K`, or length 1 to share one value across the diagonal.
    lkj_cov_lpdf, lkj_cov_lupdf <T, M, S, E>(y: &Matrix<T>, mu: &[M], sigma: &[S], eta: E)
        -> Result<T>
    where { T: Real, M: Lift<T> + Copy, S: Lift<T> + Copy, E: Lift<T> }
    |propto| { lkj_cov_with(y, mu, sigma, eta, propto) }
}

fn lkj_cov_with<T, M, S, E>(y: &Matrix<T>, mu: &[M], sigma: &[S], eta: E, propto: bool) -> Result<T>
where
    T: Real,
    M: Lift<T> + Copy,
    S: Lift<T> + Copy,
    E: Lift<T>,
{
    const F: &str = "lkj_cov_lpdf";
    let (py, ps) = (is_param::<T, T>(), is_param::<T, S>());
    let k = square(F, "y", y)?;
    let mu: Vec<T> = lift_all(mu);
    let sigma: Vec<T> = lift_all(sigma);
    for (name, len) in [("mu", mu.len()), ("sigma", sigma.len())] {
        if !(len == k || len == 1) {
            return Err(Error::dimension(
                F,
                format!("{name} has length {len}, expected {k} or 1"),
            ));
        }
    }
    let eta: T = eta.lift();
    if !(eta.value() > 0.0) {
        return Err(Error::parameter(F, "eta", eta.value(), "positive"));
    }
    param_finite(F, "mu", &mu)?;
    if let Some(s) = sigma.iter().find(|s| !(s.value() > 0.0 && s.is_finite())) {
        return Err(Error::parameter(F, "sigma", s.value(), "positive finite"));
    }
    if k == 0 {
        return Ok(T::zero());
    }
    let pick = |v: &[T], i: usize| if v.len() == 1 { v[0] } else { v[i] };
    let mut sd = Vec::with_capacity(k);
    let mut lp = T::zero();
    for i in 0..k {
        let var = y[(i, i)];
        if !(var.value() > 0.0) {
            return Err(Error::observation(
                F,
                "y",
                var.value(),
                "a covariance matrix",
            ));
        }
        let s = var.sqrt();
        let ln_s = s.ln();
        let (m, sg) = (pick(&mu, i), pick(&sigma, i));
        // The lognormal terms of the standard deviation, gated as in lognormal_lupdf.
        // Stan keeps the constant of the lognormal under propto unless every argument is data,
        // and y is data only when every argument is.
        if keep(propto, &[py]) {
            let z = (ln_s - m) / sg;
            lp -= z.square() * 0.5 + HALF_LN_2PI;
        }
        if keep(propto, &[ps]) {
            lp -= sg.ln();
        }
        if keep(propto, &[py]) {
            lp -= ln_s;
        }
        sd.push(s);
    }
    let corr = Matrix::from_fn(k, k, |i, j| {
        if i == j {
            T::one()
        } else {
            y[(i, j)] / (sd[i] * sd[j])
        }
    });
    Ok(lp + lkj_corr_core(F, &corr, eta, propto, is_param::<T, E>())?)
}

propto_pair! {
    /// Matrix normal log density of the `N` by `P` matrix `y` with mean `mu`, row precision
    /// `sigma` (`N` by `N`), and column precision `d` (`P` by `P`).
    ///
    /// `P/2 ln det Σ + N/2 ln det D - ½ tr(D (y - μ)ᵀ Σ (y - μ)) - NP/2 ln(2π)`.
    matrix_normal_prec_lpdf, matrix_normal_prec_lupdf <T>(
        y: &Matrix<T>,
        mu: &Matrix<T>,
        sigma: &Matrix<T>,
        d: &Matrix<T>,
    ) -> Result<T>
    where { T: Real }
    |propto| { matrix_normal_prec_with(y, mu, sigma, d, propto) }
}

fn matrix_normal_prec_with<T: Real>(
    y: &Matrix<T>,
    mu: &Matrix<T>,
    sigma: &Matrix<T>,
    d: &Matrix<T>,
    propto: bool,
) -> Result<T> {
    const F: &str = "matrix_normal_prec_lpdf";
    let (n, p) = y.shape();
    if mu.shape() != (n, p) {
        return Err(Error::dimension(
            F,
            format!("Mu is {:?}, expected {:?}", mu.shape(), (n, p)),
        ));
    }
    dims(F, "Sigma", square(F, "Sigma", sigma)?, n)?;
    dims(F, "D", square(F, "D", d)?, p)?;
    obs_not_nan(F, "y", y.as_slice())?;
    param_finite(F, "Mu", mu.as_slice())?;
    let ld_sigma = log_determinant_spd(sigma)?;
    let ld_d = log_determinant_spd(d)?;
    if n == 0 || p == 0 || !keep(propto, &[is_param::<T, T>()]) {
        return Ok(T::zero());
    }
    let e = Matrix::from_fn(n, p, |i, j| y[(i, j)] - mu[(i, j)]);
    let m = multiply(&e.transpose(), &multiply(sigma, &e)?)?;
    let mut trace = T::zero();
    for i in 0..p {
        for j in 0..p {
            trace += d[(i, j)] * m[(j, i)];
        }
    }
    let mut lp = ld_sigma * (p as f64 * 0.5) + ld_d * (n as f64 * 0.5) - trace * 0.5;
    if keep(propto, &[]) {
        lp -= HALF_LN_2PI * (n * p) as f64;
    }
    Ok(lp)
}
