//! The normal log density from sufficient statistics, and the Student t quantile function.

use stanmath_rs_autodiff::functions::inv_inc_beta;
use stanmath_rs_autodiff::{Broadcast, IntArg, Promoted, Real};
use stanmath_rs_core::special::HALF_LN_2PI;
use stanmath_rs_core::{Error, Result};

use crate::check::{obs, param};
use crate::kernel::{Keep, Kernel, sum_kernel, sum_kernel_propto};

pub(crate) struct NormalSufficient;

impl Kernel<1, 4> for NormalSufficient {
    const NAME: &'static str = "normal_sufficient_lpdf";

    fn check([n]: [Option<i64>; 1], [y_bar, s_squared, mu, sigma]: [Option<f64>; 4]) -> Result<()> {
        const F: &str = "normal_sufficient_lpdf";
        obs::finite(F, "y_bar", y_bar)?;
        obs::finite(F, "s_squared", s_squared)?;
        obs::nonnegative(F, "s_squared", s_squared)?;
        if let Some(n) = n.filter(|&n| n < 1) {
            return Err(Error::observation(F, "n_obs", n as f64, "positive"));
        }
        param::finite(F, "mu", mu)?;
        param::positive_finite(F, "sigma", sigma)
    }

    fn eval<S: Real>(ints: [i64; 1], x: [S; 4], grad: Option<&mut [S; 4]>) -> S {
        Self::eval_propto(ints, x, grad, Keep::ALL)
    }

    fn eval_propto<S: Real>(
        [n]: [i64; 1],
        [y_bar, s_squared, mu, sigma]: [S; 4],
        grad: Option<&mut [S; 4]>,
        keep: Keep<4>,
    ) -> S {
        let nf = n as f64;
        let diff = y_bar - mu;
        let inv_var = sigma.square().recip();
        // The sum over observations of (y_i - mu)^2, from the sufficient statistics.
        let ss = s_squared + diff.square() * nf;
        if let Some(g) = grad {
            let d_mean = diff * inv_var * nf;
            g[0] = -d_mean;
            g[1] = -(inv_var * 0.5);
            g[2] = d_mean;
            g[3] = (ss * inv_var - nf) / sigma;
        }
        let mut lp = -(ss * inv_var * 0.5);
        if keep.term(&[3]) {
            lp -= sigma.ln() * nf;
        }
        if keep.term(&[]) {
            lp -= HALF_LN_2PI * nf;
        }
        lp
    }
}

/// Normal log density of `n_obs` observations given only their sufficient statistics: the sample
/// mean `y_bar` and the sum of squared deviations from it, `s_squared = sum_i (y_i - y_bar)^2`.
///
/// Equals the sum of `normal_lpdf(y_i | mu, sigma)` over the observations:
/// `-n ln(2 pi) / 2 - n ln sigma - (s_squared + n (y_bar - mu)^2) / (2 sigma^2)`.
///
/// Requires `y_bar` finite, `s_squared` nonnegative finite, `n_obs >= 1`, `mu` finite, and
/// `sigma` positive finite. All arguments broadcast; the result is the sum over elements.
pub fn normal_sufficient_lpdf<Y, Q, N, M, G>(
    y_bar: Y,
    s_squared: Q,
    n_obs: N,
    mu: M,
    sigma: G,
) -> Result<Promoted<(Y, Q, M, G), 4>>
where
    N: IntArg,
    (Y, Q, M, G): Broadcast<4>,
{
    sum_kernel::<NormalSufficient, _, 1, 4>([&n_obs], (y_bar, s_squared, mu, sigma))
}

/// [`normal_sufficient_lpdf`] without the terms that depend only on data (Stan's `normal_sufficient_lupdf`, the
/// `propto` form): zero when every argument is data, and with the same derivatives.
pub fn normal_sufficient_lupdf<Y, Q, N, M, G>(
    y_bar: Y,
    s_squared: Q,
    n_obs: N,
    mu: M,
    sigma: G,
) -> Result<Promoted<(Y, Q, M, G), 4>>
where
    N: IntArg,
    (Y, Q, M, G): Broadcast<4>,
{
    sum_kernel_propto::<NormalSufficient, _, 1, 4>([&n_obs], (y_bar, s_squared, mu, sigma))
}

/// Quantile function of the Student t distribution with `nu` degrees of freedom, location `mu`,
/// and scale `sigma`: the `y` with `student_t_cdf(y | nu, mu, sigma) = p`.
///
/// Uses `P(T <= -t) = I_x(nu / 2, 1 / 2) / 2` with `x = nu / (nu + t^2)` and inverts the
/// regularized incomplete beta function for whichever of `x` and `1 - x` is smaller, so
/// `t^2 = nu (1 - x) / x` keeps full relative precision in both tails. Derivatives in all four
/// arguments come from the derivative rules of [`inv_inc_beta`].
///
/// Returns `-inf` at `p = 0` and `+inf` at `p = 1`.
///
/// # Errors
///
/// `p` outside `[0, 1]`, `nu` or `sigma` not positive finite, and `mu` not finite.
pub fn student_t_qf<T: Real>(p: T, nu: T, mu: T, sigma: T) -> Result<T> {
    const F: &str = "student_t_qf";
    let pv = p.value();
    if !(0.0..=1.0).contains(&pv) {
        return Err(Error::parameter(F, "p", pv, "in [0, 1]"));
    }
    param::positive_finite(F, "nu", nu.value())?;
    param::finite(F, "mu", mu.value())?;
    param::positive_finite(F, "sigma", sigma.value())?;
    if pv == 0.0 {
        return Ok(T::from_f64(f64::NEG_INFINITY));
    }
    if pv == 1.0 {
        return Ok(T::from_f64(f64::INFINITY));
    }
    if pv == 0.5 {
        return Ok(mu);
    }
    // q = 2 min(p, 1 - p) = I_x(nu / 2, 1 / 2); both 2 p and 1 - p are exact in floating point.
    let (q, sign) = if pv < 0.5 {
        (p * 2.0, -1.0)
    } else {
        ((-p + 1.0) * 2.0, 1.0)
    };
    let a = nu * 0.5;
    let half = T::from_f64(0.5);
    // Pick the branch on primal values, so only one inversion is recorded.
    let x0 = inv_inc_beta(a.value(), 0.5, q.value());
    let (x, one_minus_x) = if x0 <= 0.5 {
        let x = inv_inc_beta(a, half, q);
        (x, -x + 1.0)
    } else {
        // I_x(a, b) = 1 - I_{1 - x}(b, a)
        let y = inv_inc_beta(half, a, -q + 1.0);
        (-y + 1.0, y)
    };
    let t = (nu * one_minus_x / x).sqrt();
    Ok(mu + sigma * t * sign)
}
