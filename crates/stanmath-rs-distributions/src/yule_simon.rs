//! The Yule-Simon distribution.
//!
//! For a shape `α > 0`, the mass at `n = 1, 2, ...` is `α B(n, α + 1)`, and the upper tail is
//! `P(X > n) = n B(n, α + 1)`. Both follow from writing `B(n, α + 1)` as an integral over
//! `(0, 1)` and summing the geometric series in `n`.
//!
//! The log mass, log CDF, and log complementary CDF are fused kernels with analytic partials in
//! `α`, written generically so higher derivatives stay exact.

use stanmath_rs_autodiff::functions::{lbeta, log1m_exp};
use stanmath_rs_autodiff::special::digamma_diff;
use stanmath_rs_autodiff::{Broadcast, IntArg, Promoted, Real};
use stanmath_rs_core::{Error, Result};

use crate::check::param;
use crate::kernel::{Kernel, sum_kernel};

fn check_yule_simon(f: &'static str, n: i64, alpha: f64) -> Result<()> {
    if n < 1 {
        return Err(Error::observation(f, "n", n as f64, "positive"));
    }
    param::positive_finite(f, "alpha", alpha)
}

/// `ln P(X > n) = ln n + ln B(n, α + 1)` and its derivative `ψ(α + 1) - ψ(n + α + 1)`.
#[inline]
fn log_upper_tail<S: Real>(n: i64, alpha: S) -> (S, S) {
    let nf = n as f64;
    let value = lbeta(S::from_f64(nf), alpha + 1.0) + nf.ln();
    let d_alpha = digamma_diff(alpha + 1.0, S::from_f64(nf));
    (value, d_alpha)
}

pub(crate) struct YuleSimon;

impl Kernel<1, 1> for YuleSimon {
    const NAME: &'static str = "yule_simon_lpmf";

    fn check([n]: [i64; 1], [alpha]: [f64; 1]) -> Result<()> {
        check_yule_simon(Self::NAME, n, alpha)
    }

    fn eval<S: Real>([n]: [i64; 1], [alpha]: [S; 1], grad: Option<&mut [S; 1]>) -> S {
        let nf = n as f64;
        if let Some(g) = grad {
            // d/dα [ln α + ln Γ(α + 1) - ln Γ(n + α + 1)]
            g[0] = alpha.recip() + digamma_diff(alpha + 1.0, S::from_f64(nf));
        }
        alpha.ln() + lbeta(S::from_f64(nf), alpha + 1.0)
    }
}

pub(crate) struct YuleSimonLccdf;

impl Kernel<1, 1> for YuleSimonLccdf {
    const NAME: &'static str = "yule_simon_lccdf";

    fn check([n]: [i64; 1], [alpha]: [f64; 1]) -> Result<()> {
        check_yule_simon(Self::NAME, n, alpha)
    }

    fn eval<S: Real>([n]: [i64; 1], [alpha]: [S; 1], grad: Option<&mut [S; 1]>) -> S {
        let (value, d_alpha) = log_upper_tail(n, alpha);
        if let Some(g) = grad {
            g[0] = d_alpha;
        }
        value
    }
}

pub(crate) struct YuleSimonLcdf;

impl Kernel<1, 1> for YuleSimonLcdf {
    const NAME: &'static str = "yule_simon_lcdf";

    fn check([n]: [i64; 1], [alpha]: [f64; 1]) -> Result<()> {
        check_yule_simon(Self::NAME, n, alpha)
    }

    fn eval<S: Real>([n]: [i64; 1], [alpha]: [S; 1], grad: Option<&mut [S; 1]>) -> S {
        let (c, d_alpha) = log_upper_tail(n, alpha);
        if let Some(g) = grad {
            // d/dα ln(1 - e^c) = -c' e^c / (1 - e^c) = -c' / (e^(-c) - 1)
            g[0] = -(d_alpha / (-c).exp_m1());
        }
        log1m_exp(c)
    }
}

/// Yule-Simon log mass `ln α + ln B(n, α + 1)` of outcome `n >= 1` with shape `alpha > 0`.
pub fn yule_simon_lpmf<N, A>(n: N, alpha: A) -> Result<Promoted<(A,), 1>>
where
    N: IntArg,
    (A,): Broadcast<1>,
{
    sum_kernel::<YuleSimon, _, 1, 1>([&n], (alpha,))
}

/// Yule-Simon log CDF `ln(1 - n B(n, α + 1))`; arguments as for [`yule_simon_lpmf`].
pub fn yule_simon_lcdf<N, A>(n: N, alpha: A) -> Result<Promoted<(A,), 1>>
where
    N: IntArg,
    (A,): Broadcast<1>,
{
    sum_kernel::<YuleSimonLcdf, _, 1, 1>([&n], (alpha,))
}

/// Yule-Simon log complementary CDF `ln n + ln B(n, α + 1)`; arguments as for
/// [`yule_simon_lpmf`].
pub fn yule_simon_lccdf<N, A>(n: N, alpha: A) -> Result<Promoted<(A,), 1>>
where
    N: IntArg,
    (A,): Broadcast<1>,
{
    sum_kernel::<YuleSimonLccdf, _, 1, 1>([&n], (alpha,))
}

/// Yule-Simon CDF on the probability scale: the product over elements, computed as the
/// exponential of [`yule_simon_lcdf`].
pub fn yule_simon_cdf<N, A>(n: N, alpha: A) -> Result<Promoted<(A,), 1>>
where
    N: IntArg,
    (A,): Broadcast<1>,
{
    yule_simon_lcdf(n, alpha).map(|l| l.exp())
}
