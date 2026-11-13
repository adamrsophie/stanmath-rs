//! Further continuous distributions: log densities, log CDFs, and log complementary CDFs.
//!
//! Parameterizations and argument checks follow Stan Math. Every function accepts scalars or
//! sequences for each argument and returns the sum over elements. When a sequence argument is
//! empty the result is zero, after the scalar arguments are validated as Stan does. Log densities are fused kernels with analytic partials; the Wiener first
//! passage time density obtains its partials by forward mode instead.

use core::f64::consts::{LN_2, PI};

use stanmath_rs_autodiff::functions::{
    Phi, beta_inc, digamma, gamma_p, gamma_q, inv_logit, lbeta, lgamma, log_Phi, log_diff_exp,
    log_inv_logit, log_modified_bessel_first_kind, log_sum_exp, log1m, log1m_exp, log1m_inv_logit,
    log1p_exp, owens_t,
};
use stanmath_rs_autodiff::special::{InvMillsRatio, digamma_diff};
use stanmath_rs_autodiff::{Broadcast, Dual, Promoted, Real};
use stanmath_rs_core::special::{HALF_LN_2PI, LN_2PI};
use stanmath_rs_core::{Error, Result};

use crate::check::{obs, param};
use crate::kernel::{Keep, Kernel, sum_kernel, sum_kernel_propto, sum_terms};

/// `a ln y` with the convention `0 ln 0 = 0`.
#[inline]
fn multiply_log<S: Real>(a: S, ln_y: S) -> S {
    if a.is_const_zero() || (a.value() == 0.0 && ln_y.value() == f64::NEG_INFINITY) {
        S::zero()
    } else {
        a * ln_y
    }
}

/// `-inf` with zero partials, for points outside the support.
#[inline]
fn neg_inf<S: Real, const N: usize>(grad: Option<&mut [S; N]>) -> S {
    if let Some(g) = grad {
        *g = [S::zero(); N];
    }
    S::from_f64(f64::NEG_INFINITY)
}

/// `ln F` from closures for `F` and `1 - F`, choosing the one that keeps its digits.
fn log_lower<T: Real>(lower: impl FnOnce() -> T, upper: impl FnOnce() -> T) -> T {
    let p = lower();
    if p.value() <= 0.5 {
        p.ln()
    } else {
        log1m(upper())
    }
}

/// `ln(1 - F)` from closures for `F` and `1 - F`.
fn log_upper<T: Real>(lower: impl FnOnce() -> T, upper: impl FnOnce() -> T) -> T {
    let q = upper();
    if q.value() <= 0.5 {
        q.ln()
    } else {
        log1m(lower())
    }
}

/// A vectorized log density over kernel `$kernel` with `$n` real arguments.
macro_rules! lpdf {
    ($(#[$doc:meta])* $name:ident, $lupdf:ident, $kernel:ty, $n:literal,
     ($($arg:ident: $ty:ident),+)) => {
        $(#[$doc])*
        pub fn $name<$($ty),+>($($arg: $ty),+) -> Result<Promoted<($($ty,)+), $n>>
        where
            ($($ty,)+): Broadcast<$n>,
        {
            sum_kernel::<$kernel, _, 0, $n>([], ($($arg,)+))
        }

        #[doc = concat!(
            "[`", stringify!($name), "`] without the terms that depend only on data (Stan's `",
            stringify!($lupdf), "`, the `propto` form): zero when every argument is data, and ",
            "with the same derivatives."
        )]
        pub fn $lupdf<$($ty),+>($($arg: $ty),+) -> Result<Promoted<($($ty,)+), $n>>
        where
            ($($ty,)+): Broadcast<$n>,
        {
            sum_kernel_propto::<$kernel, _, 0, $n>([], ($($arg,)+))
        }
    };
}

/// A vectorized log CDF and log complementary CDF from per-element term functions.
macro_rules! cdfs {
    ($dist:literal, $n:literal, ($($arg:ident: $ty:ident),+), $lcdf:ident, $lccdf:ident,
     $check:expr, $lower:expr, $upper:expr) => {
        #[doc = concat!("Log CDF of the ", $dist, " distribution; arguments as for the log density.")]
        pub fn $lcdf<$($ty),+>($($arg: $ty),+) -> Result<Promoted<($($ty,)+), $n>>
        where
            ($($ty,)+): Broadcast<$n>,
        {
            const F: &str = stringify!($lcdf);
            sum_terms(F, ($($arg,)+), &[], |_, x| ($check)(F, x), |_, x| ($lower)(x))
        }

        #[doc = concat!(
            "Log complementary CDF of the ", $dist,
            " distribution; arguments as for the log density."
        )]
        pub fn $lccdf<$($ty),+>($($arg: $ty),+) -> Result<Promoted<($($ty,)+), $n>>
        where
            ($($ty,)+): Broadcast<$n>,
        {
            const F: &str = stringify!($lccdf);
            sum_terms(F, ($($arg,)+), &[], |_, x| ($check)(F, x), |_, x| ($upper)(x))
        }
    };
}

// Skew normal.

fn check_skew_normal(f: &'static str, [y, xi, omega, alpha]: [Option<f64>; 4]) -> Result<()> {
    obs::not_nan(f, "y", y)?;
    param::finite(f, "xi", xi)?;
    param::positive(f, "omega", omega)?;
    param::finite(f, "alpha", alpha)
}

pub(crate) struct SkewNormal;

impl Kernel<0, 4> for SkewNormal {
    const NAME: &'static str = "skew_normal_lpdf";

    fn check(_: [Option<i64>; 0], x: [Option<f64>; 4]) -> Result<()> {
        check_skew_normal(Self::NAME, x)
    }

    fn eval<S: Real>(_: [i64; 0], [y, xi, omega, alpha]: [S; 4], grad: Option<&mut [S; 4]>) -> S {
        let inv_omega = omega.recip();
        let z = (y - xi) * inv_omega;
        let az = alpha * z;
        if let Some(g) = grad {
            // d/dz of ln phi(z) + ln Phi(alpha z) is -z + alpha m with m the inverse Mills ratio.
            let m = az.map_unary(&InvMillsRatio);
            let dz = -z + alpha * m;
            g[0] = dz * inv_omega;
            g[1] = -(dz * inv_omega);
            g[2] = -(dz * z + 1.0) * inv_omega;
            g[3] = z * m;
        }
        -(z.square() * 0.5) - omega.ln() - HALF_LN_2PI + LN_2 + log_Phi(az)
    }
}

lpdf!(
    /// Skew normal log density with location `xi`, scale `omega`, and shape `alpha`:
    /// `2 / omega phi(z) Phi(alpha z)` with `z = (y - xi) / omega`.
    ///
    /// Requires `y` not NaN, `xi` and `alpha` finite, `omega > 0`.
    skew_normal_lpdf, skew_normal_lupdf, SkewNormal, 4, (y: Y, xi: X, omega: O, alpha: A)
);

/// `F = Phi(z) - 2 T(z, alpha)` and `1 - F = Phi(-z) + 2 T(z, alpha)`, with Owen's T.
fn skew_normal_lower<T: Real>([y, xi, omega, alpha]: [T; 4]) -> T {
    let z = (y - xi) / omega;
    log_lower(
        || Phi(z) - owens_t(z, alpha) * 2.0,
        || Phi(-z) + owens_t(z, alpha) * 2.0,
    )
}

fn skew_normal_upper<T: Real>([y, xi, omega, alpha]: [T; 4]) -> T {
    let z = (y - xi) / omega;
    log_upper(
        || Phi(z) - owens_t(z, alpha) * 2.0,
        || Phi(-z) + owens_t(z, alpha) * 2.0,
    )
}

cdfs!(
    "skew normal",
    4,
    (y: Y, xi: X, omega: O, alpha: A),
    skew_normal_lcdf,
    skew_normal_lccdf,
    check_skew_normal,
    skew_normal_lower,
    skew_normal_upper
);

// Double exponential (Laplace).

fn check_double_exponential(f: &'static str, [y, mu, sigma]: [Option<f64>; 3]) -> Result<()> {
    obs::finite(f, "y", y)?;
    param::finite(f, "mu", mu)?;
    param::positive_finite(f, "sigma", sigma)
}

pub(crate) struct DoubleExponential;

impl Kernel<0, 3> for DoubleExponential {
    const NAME: &'static str = "double_exponential_lpdf";

    fn check(_: [Option<i64>; 0], x: [Option<f64>; 3]) -> Result<()> {
        check_double_exponential(Self::NAME, x)
    }

    fn eval<S: Real>(_: [i64; 0], [y, mu, sigma]: [S; 3], grad: Option<&mut [S; 3]>) -> S {
        let inv_sigma = sigma.recip();
        let diff = y - mu;
        // The derivative of |y - mu| is taken to be zero at y = mu.
        let sign = if diff.value() > 0.0 {
            1.0
        } else if diff.value() < 0.0 {
            -1.0
        } else {
            0.0
        };
        let abs = diff * sign;
        if let Some(g) = grad {
            g[0] = inv_sigma * -sign;
            g[1] = inv_sigma * sign;
            g[2] = (abs * inv_sigma - 1.0) * inv_sigma;
        }
        -sigma.ln() - abs * inv_sigma - LN_2
    }
}

lpdf!(
    /// Double exponential (Laplace) log density with location `mu` and scale `sigma`.
    ///
    /// Requires `y` and `mu` finite, `sigma` positive finite.
    double_exponential_lpdf, double_exponential_lupdf, DoubleExponential, 3, (y: Y, mu: M, sigma: S)
);

fn double_exponential_lower<T: Real>([y, mu, sigma]: [T; 3]) -> T {
    let z = (y - mu) / sigma;
    if z.value() < 0.0 {
        z - LN_2
    } else {
        log1m_exp(-z - LN_2)
    }
}

fn double_exponential_upper<T: Real>([y, mu, sigma]: [T; 3]) -> T {
    let z = (y - mu) / sigma;
    if z.value() < 0.0 {
        log1m_exp(z - LN_2)
    } else {
        -z - LN_2
    }
}

cdfs!(
    "double exponential",
    3,
    (y: Y, mu: M, sigma: S),
    double_exponential_lcdf,
    double_exponential_lccdf,
    check_double_exponential,
    double_exponential_lower,
    double_exponential_upper
);

// Von Mises.

pub(crate) struct VonMises;

impl Kernel<0, 3> for VonMises {
    const NAME: &'static str = "von_mises_lpdf";

    fn check(_: [Option<i64>; 0], [y, mu, kappa]: [Option<f64>; 3]) -> Result<()> {
        obs::finite(Self::NAME, "y", y)?;
        param::finite(Self::NAME, "mu", mu)?;
        param::nonnegative_finite(Self::NAME, "kappa", kappa)
    }

    fn eval<S: Real>(_: [i64; 0], [y, mu, kappa]: [S; 3], grad: Option<&mut [S; 3]>) -> S {
        let d = y - mu;
        let c = d.cos();
        let log_i0 = log_modified_bessel_first_kind(S::zero(), kappa);
        if let Some(g) = grad {
            let s = d.sin();
            g[0] = -(kappa * s);
            g[1] = kappa * s;
            // d/dkappa ln I0(kappa) = I1(kappa) / I0(kappa)
            g[2] = c - (log_modified_bessel_first_kind(S::one(), kappa) - log_i0).exp();
        }
        kappa * c - log_i0 - LN_2PI
    }
}

lpdf!(
    /// Von Mises log density with location `mu` and concentration `kappa`:
    /// `exp(kappa cos(y - mu)) / (2 pi I0(kappa))`.
    ///
    /// Requires `y` and `mu` finite, `kappa` nonnegative finite. The density is periodic in `y`.
    von_mises_lpdf, von_mises_lupdf, VonMises, 3, (y: Y, mu: M, kappa: K)
);

/// An antiderivative of the von Mises density centered at zero, valid for every real `theta`:
/// `H(theta) = theta / (2 pi) + (1 / pi) sum_n (I_n(kappa) / I_0(kappa)) sin(n theta) / n`.
///
/// The Bessel ratios come from Miller's backward recurrence `I_{n-1} = I_{n+1} + (2n / kappa) I_n`,
/// which is stable in this direction and differentiable in `kappa`. For large `kappa` the ratios
/// decay like `exp(-n^2 / (2 kappa))`, so about `9 sqrt(kappa)` terms reach double precision.
fn von_mises_h<T: Real>(theta: T, kappa: T) -> T {
    let h = theta / (2.0 * PI);
    let k = kappa.value();
    if k == 0.0 {
        return h;
    }
    let terms = 20 + (9.0 * k.sqrt()).ceil() as usize;
    let start = 2 * terms + 10;
    let two_over_kappa = kappa.recip() * 2.0;
    let mut bessel = vec![T::zero(); terms + 1];
    // Unnormalized I_{n+1} and I_n, starting from I_{start+1} = 0, I_start = 1.
    let (mut next, mut cur) = (T::zero(), T::one());
    for n in (1..=start).rev() {
        let prev = next + cur * two_over_kappa * (n as f64);
        next = cur;
        cur = prev;
        if n - 1 <= terms {
            bessel[n - 1] = cur;
        }
        if cur.value() > 1e200 {
            next *= 1e-200;
            cur *= 1e-200;
            for b in &mut bessel {
                *b *= 1e-200;
            }
        }
    }
    let i0 = bessel[0];
    let mut series = T::zero();
    for (n, b) in bessel.iter().enumerate().skip(1) {
        let nf = n as f64;
        series += *b / i0 * (theta * nf).sin() / nf;
    }
    h + series / PI
}

fn check_von_mises_cdf(f: &'static str, [y, mu, kappa]: [Option<f64>; 3]) -> Result<()> {
    if let Some(y) = y.filter(|y| !(-PI..=PI).contains(y)) {
        return Err(Error::observation(f, "y", y, "in [-pi, pi]"));
    }
    param::finite(f, "mu", mu)?;
    param::positive_finite(f, "kappa", kappa)
}

/// `F(y) = H(y - mu) - H(-pi - mu)` on the support `[-pi, pi]`.
fn von_mises_lower<T: Real>([y, mu, kappa]: [T; 3]) -> T {
    (von_mises_h(y - mu, kappa) - von_mises_h(-mu - PI, kappa)).ln()
}

/// `1 - F(y) = H(pi - mu) - H(y - mu)`, since the density has unit mass over one period.
fn von_mises_upper<T: Real>([y, mu, kappa]: [T; 3]) -> T {
    (von_mises_h(-mu + PI, kappa) - von_mises_h(y - mu, kappa)).ln()
}

cdfs!(
    "von Mises (support [-pi, pi])",
    3,
    (y: Y, mu: M, kappa: K),
    von_mises_lcdf,
    von_mises_lccdf,
    check_von_mises_cdf,
    von_mises_lower,
    von_mises_upper
);

// Rayleigh.

fn check_rayleigh(f: &'static str, [y, sigma]: [Option<f64>; 2]) -> Result<()> {
    obs::nonnegative(f, "y", y)?;
    param::positive_finite(f, "sigma", sigma)
}

pub(crate) struct Rayleigh;

impl Kernel<0, 2> for Rayleigh {
    const NAME: &'static str = "rayleigh_lpdf";

    // Stan's density rejects y = 0; its CDFs use `check_rayleigh`.
    fn check(_: [Option<i64>; 0], [y, sigma]: [Option<f64>; 2]) -> Result<()> {
        obs::positive(Self::NAME, "y", y)?;
        param::positive_finite(Self::NAME, "sigma", sigma)
    }

    fn eval<S: Real>(_: [i64; 0], [y, sigma]: [S; 2], grad: Option<&mut [S; 2]>) -> S {
        let inv_sigma = sigma.recip();
        let z = y * inv_sigma;
        if let Some(g) = grad {
            g[0] = y.recip() - z * inv_sigma;
            g[1] = (z.square() - 2.0) * inv_sigma;
        }
        y.ln() - sigma.ln() * 2.0 - z.square() * 0.5
    }
}

lpdf!(
    /// Rayleigh log density with scale `sigma`.
    ///
    /// Requires `y` positive and `sigma` positive finite.
    rayleigh_lpdf, rayleigh_lupdf, Rayleigh, 2, (y: Y, sigma: S)
);

fn rayleigh_lower<T: Real>([y, sigma]: [T; 2]) -> T {
    log1m_exp(-((y / sigma).square() * 0.5))
}

fn rayleigh_upper<T: Real>([y, sigma]: [T; 2]) -> T {
    -((y / sigma).square() * 0.5)
}

cdfs!(
    "Rayleigh",
    2,
    (y: Y, sigma: S),
    rayleigh_lcdf,
    rayleigh_lccdf,
    check_rayleigh,
    rayleigh_lower,
    rayleigh_upper
);

// Pareto.

pub(crate) struct Pareto;

impl Kernel<0, 3> for Pareto {
    const NAME: &'static str = "pareto_lpdf";

    fn check(_: [Option<i64>; 0], [y, y_min, alpha]: [Option<f64>; 3]) -> Result<()> {
        obs::not_nan(Self::NAME, "y", y)?;
        param::positive_finite(Self::NAME, "y_min", y_min)?;
        param::positive_finite(Self::NAME, "alpha", alpha)
    }

    fn eval<S: Real>(_: [i64; 0], [y, y_min, alpha]: [S; 3], grad: Option<&mut [S; 3]>) -> S {
        if y.value() < y_min.value() {
            return neg_inf(grad);
        }
        let ln_y = y.ln();
        let ln_y_min = y_min.ln();
        if let Some(g) = grad {
            g[0] = -(alpha + 1.0) / y;
            g[1] = alpha / y_min;
            g[2] = alpha.recip() + ln_y_min - ln_y;
        }
        alpha.ln() + alpha * ln_y_min - (alpha + 1.0) * ln_y
    }
}

lpdf!(
    /// Pareto log density with minimum `y_min` and shape `alpha`.
    ///
    /// Requires `y` not NaN, `y_min` and `alpha` positive finite; `y < y_min` gives `-inf`.
    pareto_lpdf, pareto_lupdf, Pareto, 3, (y: Y, y_min: M, alpha: A)
);

fn check_pareto_cdf(f: &'static str, [y, y_min, alpha]: [Option<f64>; 3]) -> Result<()> {
    obs::not_nan(f, "y", y)?;
    obs::nonnegative(f, "y", y)?;
    param::positive_finite(f, "y_min", y_min)?;
    param::positive_finite(f, "alpha", alpha)
}

fn pareto_lower<T: Real>([y, y_min, alpha]: [T; 3]) -> T {
    if y.value() < y_min.value() {
        return T::from_f64(f64::NEG_INFINITY);
    }
    log1m_exp(alpha * (y_min / y).ln())
}

fn pareto_upper<T: Real>([y, y_min, alpha]: [T; 3]) -> T {
    if y.value() < y_min.value() {
        return T::zero();
    }
    alpha * (y_min / y).ln()
}

cdfs!(
    "Pareto",
    3,
    (y: Y, y_min: M, alpha: A),
    pareto_lcdf,
    pareto_lccdf,
    check_pareto_cdf,
    pareto_lower,
    pareto_upper
);

// Pareto type II (Lomax with location).

fn check_pareto_type_2(f: &'static str, [y, mu, lambda, alpha]: [Option<f64>; 4]) -> Result<()> {
    obs::not_nan(f, "y", y)?;
    param::finite(f, "mu", mu)?;
    param::positive_finite(f, "lambda", lambda)?;
    param::positive_finite(f, "alpha", alpha)?;
    match (y, mu) {
        (Some(y), Some(mu)) if !(y >= mu) => {
            Err(Error::observation(f, "y", y, "greater than or equal to mu"))
        }
        _ => Ok(()),
    }
}

pub(crate) struct ParetoType2;

impl Kernel<0, 4> for ParetoType2 {
    const NAME: &'static str = "pareto_type_2_lpdf";

    fn check(_: [Option<i64>; 0], x: [Option<f64>; 4]) -> Result<()> {
        check_pareto_type_2(Self::NAME, x)
    }

    fn eval<S: Real>(_: [i64; 0], [y, mu, lambda, alpha]: [S; 4], grad: Option<&mut [S; 4]>) -> S {
        let diff = y - mu;
        let l1p = (diff / lambda).ln_1p();
        if let Some(g) = grad {
            let a1 = alpha + 1.0;
            let inv = (lambda + diff).recip();
            g[0] = -(a1 * inv);
            g[1] = a1 * inv;
            g[2] = -lambda.recip() + a1 * diff * inv / lambda;
            g[3] = alpha.recip() - l1p;
        }
        alpha.ln() - lambda.ln() - (alpha + 1.0) * l1p
    }
}

lpdf!(
    /// Pareto type II log density with location `mu`, scale `lambda`, and shape `alpha`:
    /// `(alpha / lambda) (1 + (y - mu) / lambda)^(-(alpha + 1))`.
    ///
    /// Requires `y >= mu`, `mu` finite, `lambda` and `alpha` positive finite.
    pareto_type_2_lpdf, pareto_type_2_lupdf, ParetoType2, 4, (y: Y, mu: M, lambda: L, alpha: A)
);

fn pareto_type_2_lower<T: Real>([y, mu, lambda, alpha]: [T; 4]) -> T {
    log1m_exp(-(alpha * ((y - mu) / lambda).ln_1p()))
}

fn pareto_type_2_upper<T: Real>([y, mu, lambda, alpha]: [T; 4]) -> T {
    -(alpha * ((y - mu) / lambda).ln_1p())
}

cdfs!(
    "Pareto type II",
    4,
    (y: Y, mu: M, lambda: L, alpha: A),
    pareto_type_2_lcdf,
    pareto_type_2_lccdf,
    check_pareto_type_2,
    pareto_type_2_lower,
    pareto_type_2_upper
);

// Frechet.

fn check_frechet(f: &'static str, [y, alpha, sigma]: [Option<f64>; 3]) -> Result<()> {
    obs::positive(f, "y", y)?;
    param::positive_finite(f, "alpha", alpha)?;
    param::positive_finite(f, "sigma", sigma)
}

pub(crate) struct Frechet;

impl Kernel<0, 3> for Frechet {
    const NAME: &'static str = "frechet_lpdf";

    fn check(_: [Option<i64>; 0], x: [Option<f64>; 3]) -> Result<()> {
        check_frechet(Self::NAME, x)
    }

    fn eval<S: Real>(_: [i64; 0], [y, alpha, sigma]: [S; 3], grad: Option<&mut [S; 3]>) -> S {
        let log_ratio = (sigma / y).ln();
        let p = (alpha * log_ratio).exp(); // (sigma / y)^alpha
        if let Some(g) = grad {
            g[0] = (alpha * p - alpha - 1.0) / y;
            g[1] = alpha.recip() + log_ratio * (-p + 1.0);
            g[2] = alpha * (-p + 1.0) / sigma;
        }
        alpha.ln() + alpha * sigma.ln() - (alpha + 1.0) * y.ln() - p
    }
}

lpdf!(
    /// Frechet log density with shape `alpha` and scale `sigma`.
    ///
    /// Requires `y > 0`, `alpha` and `sigma` positive finite.
    frechet_lpdf, frechet_lupdf, Frechet, 3, (y: Y, alpha: A, sigma: S)
);

fn frechet_lower<T: Real>([y, alpha, sigma]: [T; 3]) -> T {
    -(alpha * (sigma / y).ln()).exp()
}

fn frechet_upper<T: Real>([y, alpha, sigma]: [T; 3]) -> T {
    log1m_exp(-(alpha * (sigma / y).ln()).exp())
}

cdfs!(
    "Frechet",
    3,
    (y: Y, alpha: A, sigma: S),
    frechet_lcdf,
    frechet_lccdf,
    check_frechet,
    frechet_lower,
    frechet_upper
);

// Gumbel.

fn check_gumbel(f: &'static str, [y, mu, beta]: [Option<f64>; 3]) -> Result<()> {
    obs::not_nan(f, "y", y)?;
    param::finite(f, "mu", mu)?;
    param::positive(f, "beta", beta)
}

pub(crate) struct Gumbel;

impl Kernel<0, 3> for Gumbel {
    const NAME: &'static str = "gumbel_lpdf";

    fn check(_: [Option<i64>; 0], x: [Option<f64>; 3]) -> Result<()> {
        check_gumbel(Self::NAME, x)
    }

    fn eval<S: Real>(ints: [i64; 0], x: [S; 3], grad: Option<&mut [S; 3]>) -> S {
        Self::eval_propto(ints, x, grad, Keep::ALL)
    }

    fn eval_propto<S: Real>(
        _: [i64; 0],
        [y, mu, beta]: [S; 3],
        grad: Option<&mut [S; 3]>,
        keep: Keep<3>,
    ) -> S {
        let inv_beta = beta.recip();
        let z = (y - mu) * inv_beta;
        let e = (-z).exp();
        if let Some(g) = grad {
            let dz = e - 1.0;
            g[0] = dz * inv_beta;
            g[1] = -(dz * inv_beta);
            g[2] = -(dz * z + 1.0) * inv_beta;
        }
        let lp = -z - e;
        if keep.term(&[2]) { lp - beta.ln() } else { lp }
    }
}

lpdf!(
    /// Gumbel log density with location `mu` and scale `beta`.
    ///
    /// Requires `y` not NaN, `mu` finite, `beta > 0`.
    gumbel_lpdf, gumbel_lupdf, Gumbel, 3, (y: Y, mu: M, beta: B)
);

fn gumbel_lower<T: Real>([y, mu, beta]: [T; 3]) -> T {
    -(-(y - mu) / beta).exp()
}

fn gumbel_upper<T: Real>([y, mu, beta]: [T; 3]) -> T {
    log1m_exp(-(-(y - mu) / beta).exp())
}

cdfs!(
    "Gumbel",
    3,
    (y: Y, mu: M, beta: B),
    gumbel_lcdf,
    gumbel_lccdf,
    check_gumbel,
    gumbel_lower,
    gumbel_upper
);

// Chi-square.

fn check_chi_square(f: &'static str, [y, nu]: [Option<f64>; 2]) -> Result<()> {
    obs::nonnegative(f, "y", y)?;
    param::positive_finite(f, "nu", nu)
}

pub(crate) struct ChiSquare;

impl Kernel<0, 2> for ChiSquare {
    const NAME: &'static str = "chi_square_lpdf";

    fn check(_: [Option<i64>; 0], x: [Option<f64>; 2]) -> Result<()> {
        check_chi_square(Self::NAME, x)
    }

    fn eval<S: Real>(_: [i64; 0], [y, nu]: [S; 2], grad: Option<&mut [S; 2]>) -> S {
        let half_nu = nu * 0.5;
        let ln_y = y.ln();
        if let Some(g) = grad {
            g[0] = (half_nu - 1.0) / y - 0.5;
            g[1] = (ln_y - digamma(half_nu) - LN_2) * 0.5;
        }
        multiply_log(half_nu - 1.0, ln_y) - y * 0.5 - half_nu * LN_2 - lgamma(half_nu)
    }
}

lpdf!(
    /// Chi-square log density with `nu` degrees of freedom.
    ///
    /// Requires `y >= 0` and `nu` positive finite.
    chi_square_lpdf, chi_square_lupdf, ChiSquare, 2, (y: Y, nu: N)
);

fn chi_square_lower<T: Real>([y, nu]: [T; 2]) -> T {
    let (a, x) = (nu * 0.5, y * 0.5);
    log_lower(|| gamma_p(a, x), || gamma_q(a, x))
}

fn chi_square_upper<T: Real>([y, nu]: [T; 2]) -> T {
    let (a, x) = (nu * 0.5, y * 0.5);
    log_upper(|| gamma_p(a, x), || gamma_q(a, x))
}

cdfs!(
    "chi-square",
    2,
    (y: Y, nu: N),
    chi_square_lcdf,
    chi_square_lccdf,
    check_chi_square,
    chi_square_lower,
    chi_square_upper
);

// Inverse chi-square.

pub(crate) struct InvChiSquare;

impl Kernel<0, 2> for InvChiSquare {
    const NAME: &'static str = "inv_chi_square_lpdf";

    fn check(_: [Option<i64>; 0], [y, nu]: [Option<f64>; 2]) -> Result<()> {
        obs::not_nan(Self::NAME, "y", y)?;
        param::positive_finite(Self::NAME, "nu", nu)
    }

    fn eval<S: Real>(_: [i64; 0], [y, nu]: [S; 2], grad: Option<&mut [S; 2]>) -> S {
        if y.value() <= 0.0 {
            return neg_inf(grad);
        }
        let half_nu = nu * 0.5;
        let ln_y = y.ln();
        let inv_y = y.recip();
        if let Some(g) = grad {
            g[0] = -(half_nu + 1.0) * inv_y + inv_y.square() * 0.5;
            g[1] = -(digamma(half_nu) + ln_y + LN_2) * 0.5;
        }
        -(half_nu * LN_2) - lgamma(half_nu) - (half_nu + 1.0) * ln_y - inv_y * 0.5
    }
}

lpdf!(
    /// Inverse chi-square log density with `nu` degrees of freedom.
    ///
    /// Requires `y` not NaN and `nu` positive finite; `y <= 0` gives `-inf`.
    inv_chi_square_lpdf, inv_chi_square_lupdf, InvChiSquare, 2, (y: Y, nu: N)
);

fn check_inv_chi_square_cdf(f: &'static str, [y, nu]: [Option<f64>; 2]) -> Result<()> {
    obs::nonnegative(f, "y", y)?;
    param::positive_finite(f, "nu", nu)
}

/// `F(y) = Q(nu / 2, 1 / (2 y))`.
fn inv_chi_square_lower<T: Real>([y, nu]: [T; 2]) -> T {
    if y.value() == 0.0 {
        return T::from_f64(f64::NEG_INFINITY);
    }
    let (a, x) = (nu * 0.5, (y * 2.0).recip());
    log_lower(|| gamma_q(a, x), || gamma_p(a, x))
}

fn inv_chi_square_upper<T: Real>([y, nu]: [T; 2]) -> T {
    if y.value() == 0.0 {
        return T::zero();
    }
    let (a, x) = (nu * 0.5, (y * 2.0).recip());
    log_upper(|| gamma_q(a, x), || gamma_p(a, x))
}

cdfs!(
    "inverse chi-square",
    2,
    (y: Y, nu: N),
    inv_chi_square_lcdf,
    inv_chi_square_lccdf,
    check_inv_chi_square_cdf,
    inv_chi_square_lower,
    inv_chi_square_upper
);

// Scaled inverse chi-square.

pub(crate) struct ScaledInvChiSquare;

impl Kernel<0, 3> for ScaledInvChiSquare {
    const NAME: &'static str = "scaled_inv_chi_square_lpdf";

    fn check(_: [Option<i64>; 0], [y, nu, s]: [Option<f64>; 3]) -> Result<()> {
        obs::not_nan(Self::NAME, "y", y)?;
        param::positive_finite(Self::NAME, "nu", nu)?;
        param::positive_finite(Self::NAME, "s", s)
    }

    fn eval<S: Real>(_: [i64; 0], [y, nu, s]: [S; 3], grad: Option<&mut [S; 3]>) -> S {
        if y.value() <= 0.0 {
            return neg_inf(grad);
        }
        let half_nu = nu * 0.5;
        let ln_y = y.ln();
        let ln_s = s.ln();
        let inv_y = y.recip();
        let s2 = s.square();
        if let Some(g) = grad {
            g[0] = -(half_nu + 1.0) * inv_y + half_nu * s2 * inv_y.square();
            g[1] = (half_nu.ln() - digamma(half_nu) + 1.0) * 0.5 + ln_s
                - ln_y * 0.5
                - s2 * inv_y * 0.5;
            g[2] = nu / s - nu * s * inv_y;
        }
        half_nu * half_nu.ln() - lgamma(half_nu) + nu * ln_s
            - (half_nu + 1.0) * ln_y
            - half_nu * s2 * inv_y
    }
}

lpdf!(
    /// Scaled inverse chi-square log density with `nu` degrees of freedom and scale `s`.
    ///
    /// Requires `y` not NaN, `nu` and `s` positive finite; `y <= 0` gives `-inf`.
    scaled_inv_chi_square_lpdf, scaled_inv_chi_square_lupdf, ScaledInvChiSquare, 3, (y: Y, nu: N, s: S)
);

fn check_scaled_inv_chi_square_cdf(f: &'static str, [y, nu, s]: [Option<f64>; 3]) -> Result<()> {
    obs::nonnegative(f, "y", y)?;
    param::positive_finite(f, "nu", nu)?;
    param::positive_finite(f, "s", s)
}

/// `F(y) = Q(nu / 2, nu s^2 / (2 y))`: an inverse gamma with shape `nu / 2`, scale `nu s^2 / 2`.
fn scaled_inv_chi_square_lower<T: Real>([y, nu, s]: [T; 3]) -> T {
    if y.value() == 0.0 {
        return T::from_f64(f64::NEG_INFINITY);
    }
    let a = nu * 0.5;
    let x = a * s.square() / y;
    log_lower(|| gamma_q(a, x), || gamma_p(a, x))
}

fn scaled_inv_chi_square_upper<T: Real>([y, nu, s]: [T; 3]) -> T {
    if y.value() == 0.0 {
        return T::zero();
    }
    let a = nu * 0.5;
    let x = a * s.square() / y;
    log_upper(|| gamma_q(a, x), || gamma_p(a, x))
}

cdfs!(
    "scaled inverse chi-square",
    3,
    (y: Y, nu: N, s: S),
    scaled_inv_chi_square_lcdf,
    scaled_inv_chi_square_lccdf,
    check_scaled_inv_chi_square_cdf,
    scaled_inv_chi_square_lower,
    scaled_inv_chi_square_upper
);

// Exponentially modified normal.

fn check_exp_mod_normal(f: &'static str, [y, mu, sigma, lambda]: [Option<f64>; 4]) -> Result<()> {
    obs::not_nan(f, "y", y)?;
    param::finite(f, "mu", mu)?;
    param::positive_finite(f, "sigma", sigma)?;
    param::positive_finite(f, "lambda", lambda)
}

pub(crate) struct ExpModNormal;

impl Kernel<0, 4> for ExpModNormal {
    const NAME: &'static str = "exp_mod_normal_lpdf";

    fn check(_: [Option<i64>; 0], x: [Option<f64>; 4]) -> Result<()> {
        check_exp_mod_normal(Self::NAME, x)
    }

    fn eval<S: Real>(_: [i64; 0], [y, mu, sigma, lambda]: [S; 4], grad: Option<&mut [S; 4]>) -> S {
        // ln f = ln lambda + lambda (mu - y) + lambda^2 sigma^2 / 2 + ln Phi(-v),
        // v = (mu + lambda sigma^2 - y) / sigma, using erfc(v / sqrt 2) = 2 Phi(-v).
        let inv_sigma = sigma.recip();
        let v = (mu + lambda * sigma.square() - y) * inv_sigma;
        if let Some(g) = grad {
            let m = (-v).map_unary(&InvMillsRatio); // d/dv ln Phi(-v) = -m
            g[0] = -lambda + m * inv_sigma;
            g[1] = lambda - m * inv_sigma;
            // dv/dsigma = 2 lambda - v / sigma
            g[2] = lambda.square() * sigma - m * (lambda * 2.0 - v * inv_sigma);
            g[3] = lambda.recip() + (mu - y) + lambda * sigma.square() - m * sigma;
        }
        lambda.ln() + lambda * (mu - y) + (lambda * sigma).square() * 0.5 + log_Phi(-v)
    }
}

lpdf!(
    /// Exponentially modified normal log density: a normal with location `mu` and scale `sigma`
    /// plus an independent exponential with rate `lambda`.
    ///
    /// Requires `y` not NaN, `mu` finite, `sigma` and `lambda` positive finite.
    exp_mod_normal_lpdf, exp_mod_normal_lupdf, ExpModNormal, 4, (y: Y, mu: M, sigma: S, lambda: L)
);

/// `ln Phi(u)` and `ln(exp(-lambda (y - mu) + lambda^2 sigma^2 / 2) Phi(u - lambda sigma))`, with
/// `F = Phi(u) - exp(...) Phi(u - lambda sigma)` and `u = (y - mu) / sigma`.
fn exp_mod_normal_parts<T: Real>([y, mu, sigma, lambda]: [T; 4]) -> (T, T) {
    let u = (y - mu) / sigma;
    let shifted =
        -(lambda * (y - mu)) + (lambda * sigma).square() * 0.5 + log_Phi(u - lambda * sigma);
    (u, shifted)
}

fn exp_mod_normal_lower<T: Real>(x: [T; 4]) -> T {
    match x[0].value() {
        v if v == f64::NEG_INFINITY => T::from_f64(f64::NEG_INFINITY),
        v if v == f64::INFINITY => T::zero(),
        _ => {
            let (u, shifted) = exp_mod_normal_parts(x);
            log_diff_exp(log_Phi(u), shifted)
        }
    }
}

fn exp_mod_normal_upper<T: Real>(x: [T; 4]) -> T {
    match x[0].value() {
        v if v == f64::NEG_INFINITY => T::zero(),
        v if v == f64::INFINITY => T::from_f64(f64::NEG_INFINITY),
        _ => {
            let (u, shifted) = exp_mod_normal_parts(x);
            log_sum_exp(log_Phi(-u), shifted)
        }
    }
}

cdfs!(
    "exponentially modified normal",
    4,
    (y: Y, mu: M, sigma: S, lambda: L),
    exp_mod_normal_lcdf,
    exp_mod_normal_lccdf,
    check_exp_mod_normal,
    exp_mod_normal_lower,
    exp_mod_normal_upper
);

// Skew double exponential.

fn check_skew_double_exponential(
    f: &'static str,
    [y, mu, sigma, tau]: [Option<f64>; 4],
) -> Result<()> {
    obs::finite(f, "y", y)?;
    param::finite(f, "mu", mu)?;
    param::positive_finite(f, "sigma", sigma)?;
    param::probability(f, "tau", tau)
}

pub(crate) struct SkewDoubleExponential;

impl Kernel<0, 4> for SkewDoubleExponential {
    const NAME: &'static str = "skew_double_exponential_lpdf";

    fn check(_: [Option<i64>; 0], x: [Option<f64>; 4]) -> Result<()> {
        check_skew_double_exponential(Self::NAME, x)
    }

    fn eval<S: Real>(_: [i64; 0], [y, mu, sigma, tau]: [S; 4], grad: Option<&mut [S; 4]>) -> S {
        // ln f = ln 2 + ln tau + ln(1 - tau) - ln sigma - 2 w k / sigma, where k = |y - mu| and
        // w = 1 - tau below mu, tau above.
        let inv_sigma = sigma.recip();
        let diff = y - mu;
        let below = diff.value() < 0.0;
        let (w, k, dir) = if below {
            (-tau + 1.0, -diff, -1.0)
        } else {
            (tau, diff, 1.0)
        };
        if let Some(g) = grad {
            // dk/dy = dir, dw/dtau = dir
            g[0] = w * inv_sigma * (-2.0 * dir);
            g[1] = w * inv_sigma * (2.0 * dir);
            g[2] = (w * k * inv_sigma * 2.0 - 1.0) * inv_sigma;
            g[3] = tau.recip() - (-tau + 1.0).recip() - k * inv_sigma * (2.0 * dir);
        }
        tau.ln() + log1m(tau) - sigma.ln() - w * k * inv_sigma * 2.0 + LN_2
    }
}

lpdf!(
    /// Skew double exponential log density with location `mu`, scale `sigma`, and skewness
    /// `tau`: `2 tau (1 - tau) / sigma exp(-2 / sigma ((1 - tau) (mu - y)^+ + tau (y - mu)^+))`.
    ///
    /// Requires `y` and `mu` finite, `sigma` positive finite, `tau` in `[0, 1]`.
    skew_double_exponential_lpdf, skew_double_exponential_lupdf, SkewDoubleExponential, 4, (y: Y, mu: M, sigma: S, tau: T)
);

fn skew_double_exponential_lower<T: Real>([y, mu, sigma, tau]: [T; 4]) -> T {
    let d = (y - mu) / sigma * 2.0;
    if d.value() < 0.0 {
        // tau exp(-2 (1 - tau) (mu - y) / sigma)
        tau.ln() + (-tau + 1.0) * d
    } else {
        // 1 - (1 - tau) exp(-2 tau (y - mu) / sigma)
        log1m_exp(log1m(tau) - tau * d)
    }
}

fn skew_double_exponential_upper<T: Real>([y, mu, sigma, tau]: [T; 4]) -> T {
    let d = (y - mu) / sigma * 2.0;
    if d.value() < 0.0 {
        log1m_exp(tau.ln() + (-tau + 1.0) * d)
    } else {
        log1m(tau) - tau * d
    }
}

cdfs!(
    "skew double exponential",
    4,
    (y: Y, mu: M, sigma: S, tau: T),
    skew_double_exponential_lcdf,
    skew_double_exponential_lccdf,
    check_skew_double_exponential,
    skew_double_exponential_lower,
    skew_double_exponential_upper
);

// Log-logistic.

fn check_loglogistic(f: &'static str, [y, alpha, beta]: [Option<f64>; 3]) -> Result<()> {
    obs::finite(f, "y", y)?;
    obs::nonnegative(f, "y", y)?;
    param::positive_finite(f, "alpha", alpha)?;
    param::positive_finite(f, "beta", beta)
}

pub(crate) struct LogLogistic;

impl Kernel<0, 3> for LogLogistic {
    const NAME: &'static str = "loglogistic_lpdf";

    // Stan's density requires y positive finite; its CDF uses `check_loglogistic`.
    fn check(_: [Option<i64>; 0], [y, alpha, beta]: [Option<f64>; 3]) -> Result<()> {
        obs::positive_finite(Self::NAME, "y", y)?;
        param::positive_finite(Self::NAME, "alpha", alpha)?;
        param::positive_finite(Self::NAME, "beta", beta)
    }

    fn eval<S: Real>(_: [i64; 0], [y, alpha, beta]: [S; 3], grad: Option<&mut [S; 3]>) -> S {
        let r = (y / alpha).ln();
        let br = beta * r;
        if let Some(g) = grad {
            let s = inv_logit(br);
            let dr = beta - 1.0 - beta * s * 2.0; // d ln f / d r
            g[0] = dr / y;
            g[1] = beta * (s * 2.0 - 1.0) / alpha;
            g[2] = beta.recip() + r * (-(s * 2.0) + 1.0);
        }
        beta.ln() - alpha.ln() + (beta - 1.0) * r - log1p_exp(br) * 2.0
    }
}

lpdf!(
    /// Log-logistic log density with scale `alpha` and shape `beta`:
    /// `(beta / alpha) (y / alpha)^(beta - 1) / (1 + (y / alpha)^beta)^2`.
    ///
    /// Requires `y`, `alpha`, and `beta` positive finite.
    loglogistic_lpdf, loglogistic_lupdf, LogLogistic, 3, (y: Y, alpha: A, beta: B)
);

fn loglogistic_lower<T: Real>([y, alpha, beta]: [T; 3]) -> T {
    if y.value() == 0.0 {
        return T::from_f64(f64::NEG_INFINITY);
    }
    log_inv_logit(beta * (y / alpha).ln())
}

fn loglogistic_upper<T: Real>([y, alpha, beta]: [T; 3]) -> T {
    if y.value() == 0.0 {
        return T::zero();
    }
    log1m_inv_logit(beta * (y / alpha).ln())
}

cdfs!(
    "log-logistic",
    3,
    (y: Y, alpha: A, beta: B),
    loglogistic_lcdf,
    loglogistic_lccdf,
    check_loglogistic,
    loglogistic_lower,
    loglogistic_upper
);

// Beta with mean and precision.

fn check_beta_proportion(f: &'static str, [y, mu, kappa]: [Option<f64>; 3]) -> Result<()> {
    obs::unit_interval(f, "y", y)?;
    param::open_unit(f, "mu", mu)?;
    param::positive_finite(f, "kappa", kappa)
}

pub(crate) struct BetaProportion;

impl Kernel<0, 3> for BetaProportion {
    const NAME: &'static str = "beta_proportion_lpdf";

    fn check(_: [Option<i64>; 0], x: [Option<f64>; 3]) -> Result<()> {
        check_beta_proportion(Self::NAME, x)
    }

    fn eval<S: Real>(_: [i64; 0], [y, mu, kappa]: [S; 3], grad: Option<&mut [S; 3]>) -> S {
        let a = mu * kappa;
        let b = (-mu + 1.0) * kappa;
        let ln_y = y.ln();
        let ln_1my = log1m(y);
        if let Some(g) = grad {
            let da = ln_y - digamma_diff(a, b);
            let db = ln_1my - digamma_diff(b, a);
            g[0] = (a - 1.0) / y - (b - 1.0) / (-y + 1.0);
            g[1] = kappa * (da - db);
            g[2] = mu * da + (-mu + 1.0) * db;
        }
        multiply_log(a - 1.0, ln_y) + multiply_log(b - 1.0, ln_1my) - lbeta(a, b)
    }
}

lpdf!(
    /// Beta log density with mean `mu` and precision `kappa`, that is
    /// `beta(mu kappa, (1 - mu) kappa)`.
    ///
    /// Requires `y` in `[0, 1]`, `mu` in `(0, 1)`, `kappa` positive finite.
    beta_proportion_lpdf, beta_proportion_lupdf, BetaProportion, 3, (y: Y, mu: M, kappa: K)
);

fn beta_proportion_lower<T: Real>([y, mu, kappa]: [T; 3]) -> T {
    let (a, b) = (mu * kappa, (-mu + 1.0) * kappa);
    log_lower(|| beta_inc(a, b, y), || beta_inc(b, a, -y + 1.0))
}

fn beta_proportion_upper<T: Real>([y, mu, kappa]: [T; 3]) -> T {
    let (a, b) = (mu * kappa, (-mu + 1.0) * kappa);
    log_upper(|| beta_inc(a, b, y), || beta_inc(b, a, -y + 1.0))
}

cdfs!(
    "beta proportion",
    3,
    (y: Y, mu: M, kappa: K),
    beta_proportion_lcdf,
    beta_proportion_lccdf,
    check_beta_proportion,
    beta_proportion_lower,
    beta_proportion_upper
);

// Wiener first passage time.

/// Target absolute error of the truncated series for the standardized density.
const WIENER_EPSILON: f64 = 1e-12;

/// Log density of the first passage time at the upper boundary of a Wiener diffusion with
/// boundary separation `alpha`, nondecision time `tau`, relative starting point `beta`, and drift
/// `delta`.
///
/// The upper boundary density equals the lower boundary density with `w = 1 - beta` and
/// `v = -delta`. With `x = y - tau` and `u = x / alpha^2`, Navarro and Fuss (2009) give
/// `f(x) = alpha^-2 exp(-v alpha w - v^2 x / 2) g(u | w)`, where `g` is either the small time
/// series `(2 pi u^3)^(-1/2) sum_k (w + 2k) exp(-(w + 2k)^2 / (2u))` or the large time series
/// `pi sum_k k exp(-k^2 pi^2 u / 2) sin(k pi w)`, whichever needs fewer terms. The term counts
/// depend on values only, so every derivative mode evaluates the same truncated series.
fn wiener_log_density<T: Real>([y, alpha, tau, beta, delta]: [T; 5]) -> T {
    let x = y - tau;
    let w = -beta + 1.0;
    let v = -delta;
    let u = x / alpha.square();
    let uv = u.value();

    // Terms needed by the large time series.
    let large = {
        let floor = 1.0 / (PI * uv.sqrt());
        if PI * uv * WIENER_EPSILON < 1.0 {
            (-2.0 * (PI * uv * WIENER_EPSILON).ln() / (PI * PI * uv))
                .sqrt()
                .max(floor)
        } else {
            floor
        }
    };
    // Terms needed by the small time series.
    let small = {
        let arg = 2.0 * (2.0 * PI * uv).sqrt() * WIENER_EPSILON;
        if arg < 1.0 {
            (2.0 + (-2.0 * uv * arg.ln()).sqrt()).max(uv.sqrt() + 1.0)
        } else {
            2.0
        }
    };

    let log_g = if small < large {
        // Factor out exp(-w^2 / (2u)); every other exponent is then negative.
        // k from -floor((K - 1) / 2) to ceil((K - 1) / 2).
        let k_count = small.ceil() as i64;
        let lo = -((k_count - 1) / 2);
        let hi = k_count / 2;
        let mut sum = T::zero();
        for k in lo..=hi {
            let wk = w + 2.0 * k as f64;
            sum += wk * (-(wk.square() - w.square()) / (u * 2.0)).exp();
        }
        sum.ln() - w.square() / (u * 2.0) - u.ln() * 1.5 - LN_2PI * 0.5
    } else {
        // Factor out exp(-pi^2 u / 2).
        let k_count = (large.ceil() as i64).max(1);
        let mut sum = T::zero();
        for k in 1..=k_count {
            let kf = k as f64;
            sum += (-(u * ((kf * kf - 1.0) * PI * PI * 0.5))).exp() * (w * (kf * PI)).sin() * kf;
        }
        sum.ln() - u * (PI * PI * 0.5) + PI.ln()
    };
    log_g - alpha.ln() * 2.0 - v * alpha * w - v.square() * x * 0.5
}

pub(crate) struct Wiener;

impl Kernel<0, 5> for Wiener {
    const NAME: &'static str = "wiener_lpdf";

    fn check(_: [Option<i64>; 0], [y, alpha, tau, beta, delta]: [Option<f64>; 5]) -> Result<()> {
        obs::positive_finite(Self::NAME, "y", y)?;
        param::positive_finite(Self::NAME, "alpha", alpha)?;
        param::nonnegative_finite(Self::NAME, "tau", tau)?;
        param::open_unit(Self::NAME, "beta", beta)?;
        param::finite(Self::NAME, "delta", delta)?;
        match (y, tau) {
            (Some(y), Some(tau)) if !(y > tau) => Err(Error::observation(
                Self::NAME,
                "y",
                y,
                "greater than the nondecision time tau",
            )),
            _ => Ok(()),
        }
    }

    fn eval<S: Real>(_: [i64; 0], x: [S; 5], grad: Option<&mut [S; 5]>) -> S {
        if let Some(g) = grad {
            for (j, gj) in g.iter_mut().enumerate() {
                let xd: [Dual<S>; 5] = core::array::from_fn(|k| {
                    if k == j {
                        Dual::variable(x[k])
                    } else {
                        Dual::constant(x[k])
                    }
                });
                *gj = wiener_log_density(xd).eps;
            }
        }
        wiener_log_density(x)
    }
}

lpdf!(
    /// Wiener first passage time log density at the upper boundary, with boundary separation
    /// `alpha`, nondecision time `tau`, a priori bias `beta`, and drift rate `delta`.
    ///
    /// Requires `y > tau`, `y` and `alpha` positive finite, `tau` nonnegative finite, `beta` in
    /// `(0, 1)`, `delta` finite. The series is truncated at an absolute error of about `1e-12` in
    /// the standardized density (Navarro and Fuss 2009).
    wiener_lpdf, wiener_lupdf, Wiener, 5, (y: Y, alpha: A, tau: T, beta: B, delta: D)
);
