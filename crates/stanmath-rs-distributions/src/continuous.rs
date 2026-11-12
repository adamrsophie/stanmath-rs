//! Log densities of continuous distributions.
//!
//! Every function accepts scalars or sequences for each argument (see
//! [`Broadcast`](stanmath_rs_autodiff::Broadcast)), returns the sum of the log densities, and returns
//! zero when a sequence argument is empty. Argument checks follow Stan Math; the module
//! documentation of each function lists them and the boundary behavior.

use stanmath_rs_autodiff::functions::{digamma, inv_logit, lbeta, lgamma, log1m, log1p, log1p_exp};
use stanmath_rs_autodiff::special::digamma_diff;
use stanmath_rs_autodiff::{Broadcast, Promoted, Real};
use stanmath_rs_core::Result;
use stanmath_rs_core::special::{HALF_LN_2PI, LN_PI};

use crate::check::{obs, param};
use crate::kernel::{Kernel, sum_kernel};

/// `(a - 1) ln y` with the convention `0 * ln 0 = 0`.
#[inline]
fn multiply_log<S: Real>(a: S, ln_y: S) -> S {
    if a.is_const_zero() || (a.value() == 0.0 && ln_y.value() == f64::NEG_INFINITY) {
        S::zero()
    } else {
        a * ln_y
    }
}

pub(crate) struct Normal;

impl Kernel<0, 3> for Normal {
    const NAME: &'static str = "normal_lpdf";

    fn check(_: [i64; 0], [y, mu, sigma]: [f64; 3]) -> Result<()> {
        obs::not_nan(Self::NAME, "y", y)?;
        param::finite(Self::NAME, "mu", mu)?;
        param::positive(Self::NAME, "sigma", sigma)
    }

    fn eval<S: Real>(_: [i64; 0], [y, mu, sigma]: [S; 3], grad: Option<&mut [S; 3]>) -> S {
        let inv_sigma = sigma.recip();
        let z = (y - mu) * inv_sigma;
        if let Some(g) = grad {
            let dz = z * inv_sigma;
            g[0] = -dz;
            g[1] = dz;
            g[2] = (z.square() - 1.0) * inv_sigma;
        }
        -(z.square() * 0.5) - sigma.ln() - HALF_LN_2PI
    }
}

/// Normal log density `ln N(y | mu, sigma)` with location `mu` and scale `sigma`.
///
/// Requires `y` not NaN, `mu` finite, and `sigma > 0`.
///
/// ```
/// use stanmath_rs_distributions::normal_lpdf;
///
/// let lp: f64 = normal_lpdf(1.3, 0.5, 1.2).unwrap();
/// assert!((lp - (-1.3234823122208494)).abs() < 1e-15);
/// ```
pub fn normal_lpdf<Y, M, S>(y: Y, mu: M, sigma: S) -> Result<Promoted<(Y, M, S), 3>>
where
    (Y, M, S): Broadcast<3>,
{
    sum_kernel::<Normal, _, 0, 3>([], (y, mu, sigma))
}

pub(crate) struct StdNormal;

impl Kernel<0, 1> for StdNormal {
    const NAME: &'static str = "std_normal_lpdf";

    fn check(_: [i64; 0], [y]: [f64; 1]) -> Result<()> {
        obs::not_nan(Self::NAME, "y", y)
    }

    fn eval<S: Real>(_: [i64; 0], [y]: [S; 1], grad: Option<&mut [S; 1]>) -> S {
        if let Some(g) = grad {
            g[0] = -y;
        }
        -(y.square() * 0.5) - HALF_LN_2PI
    }
}

/// Standard normal log density. Requires `y` not NaN.
pub fn std_normal_lpdf<Y>(y: Y) -> Result<Promoted<(Y,), 1>>
where
    (Y,): Broadcast<1>,
{
    sum_kernel::<StdNormal, _, 0, 1>([], (y,))
}

pub(crate) struct StudentT;

impl Kernel<0, 4> for StudentT {
    const NAME: &'static str = "student_t_lpdf";

    fn check(_: [i64; 0], [y, nu, mu, sigma]: [f64; 4]) -> Result<()> {
        obs::not_nan(Self::NAME, "y", y)?;
        param::positive_finite(Self::NAME, "nu", nu)?;
        param::finite(Self::NAME, "mu", mu)?;
        param::positive_finite(Self::NAME, "sigma", sigma)
    }

    fn eval<S: Real>(_: [i64; 0], [y, nu, mu, sigma]: [S; 4], grad: Option<&mut [S; 4]>) -> S {
        let half_nu = nu * 0.5;
        let z = (y - mu) / sigma;
        let r = z.square() / nu;
        let log1p_r = log1p(r);
        if let Some(g) = grad {
            // d/dy: -(nu + 1) (y - mu) / (nu sigma^2 + (y - mu)^2)
            let diff = y - mu;
            let dy = -((nu + 1.0) * diff) / (nu * sigma.square() + diff.square());
            g[0] = dy;
            g[2] = -dy;
            let frac = r / (r + 1.0);
            g[3] = (-sigma.recip()) + (nu + 1.0) * frac / sigma;
            // 0.5 (psi((nu + 1) / 2) - psi(nu / 2)) - 1 / (2 nu) - 0.5 ln(1 + r)
            //   + 0.5 (nu + 1) r / (nu (1 + r))
            g[1] = -(digamma_diff(half_nu, S::from_f64(0.5)) * 0.5)
                - (nu * 2.0).recip()
                - log1p_r * 0.5
                + (nu + 1.0) * frac / (nu * 2.0);
        }
        // ln Γ((nu + 1) / 2) - ln Γ(nu / 2) - ln(pi) / 2 = -ln B(nu / 2, 1 / 2), which does not
        // cancel for large nu.
        -lbeta(half_nu, S::from_f64(0.5)) - nu.ln() * 0.5 - sigma.ln() - (half_nu + 0.5) * log1p_r
    }
}

/// Student's t log density with `nu` degrees of freedom, location `mu`, and scale `sigma`.
///
/// Requires `y` not NaN, `nu` and `sigma` positive finite, `mu` finite.
pub fn student_t_lpdf<Y, N, M, S>(y: Y, nu: N, mu: M, sigma: S) -> Result<Promoted<(Y, N, M, S), 4>>
where
    (Y, N, M, S): Broadcast<4>,
{
    sum_kernel::<StudentT, _, 0, 4>([], (y, nu, mu, sigma))
}

pub(crate) struct Cauchy;

impl Kernel<0, 3> for Cauchy {
    const NAME: &'static str = "cauchy_lpdf";

    fn check(_: [i64; 0], [y, mu, sigma]: [f64; 3]) -> Result<()> {
        obs::not_nan(Self::NAME, "y", y)?;
        param::finite(Self::NAME, "mu", mu)?;
        param::positive_finite(Self::NAME, "sigma", sigma)
    }

    fn eval<S: Real>(_: [i64; 0], [y, mu, sigma]: [S; 3], grad: Option<&mut [S; 3]>) -> S {
        let z = (y - mu) / sigma;
        let z2 = z.square();
        if let Some(g) = grad {
            let dy = -(z * 2.0) / (sigma * (z2 + 1.0));
            g[0] = dy;
            g[1] = -dy;
            g[2] = (z2 - 1.0) / (sigma * (z2 + 1.0));
        }
        -sigma.ln() - log1p(z2) - LN_PI
    }
}

/// Cauchy log density with location `mu` and scale `sigma`.
///
/// Requires `y` not NaN, `mu` finite, `sigma` positive finite.
pub fn cauchy_lpdf<Y, M, S>(y: Y, mu: M, sigma: S) -> Result<Promoted<(Y, M, S), 3>>
where
    (Y, M, S): Broadcast<3>,
{
    sum_kernel::<Cauchy, _, 0, 3>([], (y, mu, sigma))
}

pub(crate) struct Logistic;

impl Kernel<0, 3> for Logistic {
    const NAME: &'static str = "logistic_lpdf";

    fn check(_: [i64; 0], [y, mu, sigma]: [f64; 3]) -> Result<()> {
        obs::finite(Self::NAME, "y", y)?;
        param::finite(Self::NAME, "mu", mu)?;
        param::positive_finite(Self::NAME, "sigma", sigma)
    }

    fn eval<S: Real>(_: [i64; 0], [y, mu, sigma]: [S; 3], grad: Option<&mut [S; 3]>) -> S {
        let inv_sigma = sigma.recip();
        let z = (y - mu) * inv_sigma;
        if let Some(g) = grad {
            // d/dz [-z - 2 log1p_exp(-z)] = 2 inv_logit(-z) - 1
            let gz = inv_logit(-z) * 2.0 - 1.0;
            g[0] = gz * inv_sigma;
            g[1] = -(gz * inv_sigma);
            g[2] = -(gz * z + 1.0) * inv_sigma;
        }
        -z - sigma.ln() - log1p_exp(-z) * 2.0
    }
}

/// Logistic log density with location `mu` and scale `sigma`.
///
/// Requires `y` and `mu` finite, `sigma` positive finite.
pub fn logistic_lpdf<Y, M, S>(y: Y, mu: M, sigma: S) -> Result<Promoted<(Y, M, S), 3>>
where
    (Y, M, S): Broadcast<3>,
{
    sum_kernel::<Logistic, _, 0, 3>([], (y, mu, sigma))
}

pub(crate) struct Exponential;

impl Kernel<0, 2> for Exponential {
    const NAME: &'static str = "exponential_lpdf";

    fn check(_: [i64; 0], [y, beta]: [f64; 2]) -> Result<()> {
        obs::nonnegative(Self::NAME, "y", y)?;
        param::positive_finite(Self::NAME, "beta", beta)
    }

    fn eval<S: Real>(_: [i64; 0], [y, beta]: [S; 2], grad: Option<&mut [S; 2]>) -> S {
        if let Some(g) = grad {
            g[0] = -beta;
            g[1] = beta.recip() - y;
        }
        beta.ln() - beta * y
    }
}

/// Exponential log density with rate (inverse scale) `beta`.
///
/// Requires `y >= 0` and `beta` positive finite.
pub fn exponential_lpdf<Y, B>(y: Y, beta: B) -> Result<Promoted<(Y, B), 2>>
where
    (Y, B): Broadcast<2>,
{
    sum_kernel::<Exponential, _, 0, 2>([], (y, beta))
}

pub(crate) struct Gamma;

impl Kernel<0, 3> for Gamma {
    const NAME: &'static str = "gamma_lpdf";

    fn check(_: [i64; 0], [y, alpha, beta]: [f64; 3]) -> Result<()> {
        obs::positive_finite(Self::NAME, "y", y)?;
        param::positive_finite(Self::NAME, "alpha", alpha)?;
        param::positive_finite(Self::NAME, "beta", beta)
    }

    fn eval<S: Real>(_: [i64; 0], [y, alpha, beta]: [S; 3], grad: Option<&mut [S; 3]>) -> S {
        let ln_y = y.ln();
        let ln_beta = beta.ln();
        if let Some(g) = grad {
            g[0] = (alpha - 1.0) / y - beta;
            g[1] = ln_beta - digamma(alpha) + ln_y;
            g[2] = alpha / beta - y;
        }
        alpha * ln_beta - lgamma(alpha) + (alpha - 1.0) * ln_y - beta * y
    }
}

/// Gamma log density with shape `alpha` and rate (inverse scale) `beta`.
///
/// Requires `y`, `alpha`, and `beta` positive finite. For the shape and scale parameterization
/// pass `beta = 1 / scale`.
pub fn gamma_lpdf<Y, A, B>(y: Y, alpha: A, beta: B) -> Result<Promoted<(Y, A, B), 3>>
where
    (Y, A, B): Broadcast<3>,
{
    sum_kernel::<Gamma, _, 0, 3>([], (y, alpha, beta))
}

pub(crate) struct InvGamma;

impl Kernel<0, 3> for InvGamma {
    const NAME: &'static str = "inv_gamma_lpdf";

    fn check(_: [i64; 0], [y, alpha, beta]: [f64; 3]) -> Result<()> {
        obs::not_nan(Self::NAME, "y", y)?;
        param::positive_finite(Self::NAME, "alpha", alpha)?;
        param::positive_finite(Self::NAME, "beta", beta)
    }

    fn eval<S: Real>(_: [i64; 0], [y, alpha, beta]: [S; 3], grad: Option<&mut [S; 3]>) -> S {
        if y.value() <= 0.0 {
            if let Some(g) = grad {
                *g = [S::zero(); 3];
            }
            return S::from_f64(f64::NEG_INFINITY);
        }
        let ln_y = y.ln();
        let ln_beta = beta.ln();
        let inv_y = y.recip();
        if let Some(g) = grad {
            g[0] = -(alpha + 1.0) * inv_y + beta * inv_y.square();
            g[1] = ln_beta - digamma(alpha) - ln_y;
            g[2] = alpha / beta - inv_y;
        }
        alpha * ln_beta - lgamma(alpha) - (alpha + 1.0) * ln_y - beta * inv_y
    }
}

/// Inverse gamma log density with shape `alpha` and scale `beta`.
///
/// Requires `y` not NaN, `alpha` and `beta` positive finite; `y <= 0` gives `-inf`.
pub fn inv_gamma_lpdf<Y, A, B>(y: Y, alpha: A, beta: B) -> Result<Promoted<(Y, A, B), 3>>
where
    (Y, A, B): Broadcast<3>,
{
    sum_kernel::<InvGamma, _, 0, 3>([], (y, alpha, beta))
}

pub(crate) struct Beta;

impl Kernel<0, 3> for Beta {
    const NAME: &'static str = "beta_lpdf";

    fn check(_: [i64; 0], [y, a, b]: [f64; 3]) -> Result<()> {
        param::positive_finite(Self::NAME, "alpha", a)?;
        param::positive_finite(Self::NAME, "beta", b)?;
        obs::unit_interval(Self::NAME, "y", y)
    }

    fn eval<S: Real>(_: [i64; 0], [y, a, b]: [S; 3], grad: Option<&mut [S; 3]>) -> S {
        let ln_y = y.ln();
        let ln_1my = log1m(y);
        if let Some(g) = grad {
            g[0] = (a - 1.0) / y - (b - 1.0) / (-y + 1.0);
            g[1] = ln_y - digamma_diff(a, b);
            g[2] = ln_1my - digamma_diff(b, a);
        }
        multiply_log(a - 1.0, ln_y) + multiply_log(b - 1.0, ln_1my) - lbeta(a, b)
    }
}

/// Beta log density with shapes `alpha` and `beta`.
///
/// Requires `y` in `[0, 1]` and both shapes positive finite. Uses `0 ln 0 = 0`, so `y = 0` with
/// `alpha = 1` is finite.
pub fn beta_lpdf<Y, A, B>(y: Y, alpha: A, beta: B) -> Result<Promoted<(Y, A, B), 3>>
where
    (Y, A, B): Broadcast<3>,
{
    sum_kernel::<Beta, _, 0, 3>([], (y, alpha, beta))
}

pub(crate) struct LogNormal;

impl Kernel<0, 3> for LogNormal {
    const NAME: &'static str = "lognormal_lpdf";

    fn check(_: [i64; 0], [y, mu, sigma]: [f64; 3]) -> Result<()> {
        obs::nonnegative(Self::NAME, "y", y)?;
        param::finite(Self::NAME, "mu", mu)?;
        param::positive_finite(Self::NAME, "sigma", sigma)
    }

    fn eval<S: Real>(_: [i64; 0], [y, mu, sigma]: [S; 3], grad: Option<&mut [S; 3]>) -> S {
        if y.value() == 0.0 {
            if let Some(g) = grad {
                *g = [S::zero(); 3];
            }
            return S::from_f64(f64::NEG_INFINITY);
        }
        let ln_y = y.ln();
        let inv_sigma = sigma.recip();
        let z = (ln_y - mu) * inv_sigma;
        if let Some(g) = grad {
            g[0] = -(z * inv_sigma + 1.0) / y;
            g[1] = z * inv_sigma;
            g[2] = (z.square() - 1.0) * inv_sigma;
        }
        -(z.square() * 0.5) - sigma.ln() - HALF_LN_2PI - ln_y
    }
}

/// Lognormal log density with log-scale location `mu` and scale `sigma`.
///
/// Requires `y >= 0`, `mu` finite, `sigma` positive finite; `y = 0` gives `-inf`.
pub fn lognormal_lpdf<Y, M, S>(y: Y, mu: M, sigma: S) -> Result<Promoted<(Y, M, S), 3>>
where
    (Y, M, S): Broadcast<3>,
{
    sum_kernel::<LogNormal, _, 0, 3>([], (y, mu, sigma))
}

pub(crate) struct Uniform;

impl Kernel<0, 3> for Uniform {
    const NAME: &'static str = "uniform_lpdf";

    fn check(_: [i64; 0], [y, alpha, beta]: [f64; 3]) -> Result<()> {
        obs::not_nan(Self::NAME, "y", y)?;
        param::finite(Self::NAME, "alpha", alpha)?;
        param::finite(Self::NAME, "beta", beta)?;
        if beta > alpha {
            Ok(())
        } else {
            Err(stanmath_rs_core::Error::parameter(
                Self::NAME,
                "beta",
                beta,
                "greater than alpha",
            ))
        }
    }

    fn eval<S: Real>(_: [i64; 0], [y, alpha, beta]: [S; 3], grad: Option<&mut [S; 3]>) -> S {
        let inside = y.value() >= alpha.value() && y.value() <= beta.value();
        let width = beta - alpha;
        if let Some(g) = grad {
            if inside {
                let inv = width.recip();
                *g = [S::zero(), inv, -inv];
            } else {
                *g = [S::zero(); 3];
            }
        }
        if inside {
            -width.ln()
        } else {
            S::from_f64(f64::NEG_INFINITY)
        }
    }
}

/// Uniform log density on `[alpha, beta]`.
///
/// Requires `y` not NaN, `alpha` and `beta` finite, `beta > alpha`; `y` outside the interval gives
/// `-inf`.
pub fn uniform_lpdf<Y, A, B>(y: Y, alpha: A, beta: B) -> Result<Promoted<(Y, A, B), 3>>
where
    (Y, A, B): Broadcast<3>,
{
    sum_kernel::<Uniform, _, 0, 3>([], (y, alpha, beta))
}

pub(crate) struct Weibull;

impl Kernel<0, 3> for Weibull {
    const NAME: &'static str = "weibull_lpdf";

    fn check(_: [i64; 0], [y, alpha, sigma]: [f64; 3]) -> Result<()> {
        obs::finite(Self::NAME, "y", y)?;
        param::positive_finite(Self::NAME, "alpha", alpha)?;
        param::positive_finite(Self::NAME, "sigma", sigma)
    }

    fn eval<S: Real>(_: [i64; 0], [y, alpha, sigma]: [S; 3], grad: Option<&mut [S; 3]>) -> S {
        if y.value() < 0.0 {
            if let Some(g) = grad {
                *g = [S::zero(); 3];
            }
            return S::from_f64(f64::NEG_INFINITY);
        }
        let log_ratio = (y / sigma).ln();
        let pow = (log_ratio * alpha).exp(); // (y / sigma)^alpha
        if let Some(g) = grad {
            g[0] = ((alpha - 1.0) - alpha * pow) / y;
            g[1] = alpha.recip() + log_ratio * (-pow + 1.0);
            g[2] = alpha * (pow - 1.0) / sigma;
        }
        alpha.ln() - sigma.ln() + multiply_log(alpha - 1.0, log_ratio) - pow
    }
}

/// Weibull log density with shape `alpha` and scale `sigma`.
///
/// Requires `y` finite, `alpha` and `sigma` positive finite; `y < 0` gives `-inf`.
pub fn weibull_lpdf<Y, A, S>(y: Y, alpha: A, sigma: S) -> Result<Promoted<(Y, A, S), 3>>
where
    (Y, A, S): Broadcast<3>,
{
    sum_kernel::<Weibull, _, 0, 3>([], (y, alpha, sigma))
}
