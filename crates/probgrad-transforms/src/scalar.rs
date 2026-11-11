//! Scalar bounds and affine transforms.

use probgrad_autodiff::functions::{inv_logit, log1p_exp};
use probgrad_autodiff::{Lift, Real};
use probgrad_core::{Error, Result};

/// `x = lb + exp(y)`; with `lb = -inf` the identity.
pub fn lb_constrain<T: Real, L: Lift<T>>(y: T, lb: L) -> T {
    let lb: T = lb.lift();
    if lb.value() == f64::NEG_INFINITY {
        y
    } else {
        y.exp() + lb
    }
}

/// [`lb_constrain`] adding `ln |dx/dy| = y` to `lp`.
pub fn lb_constrain_lp<T: Real, L: Lift<T>>(y: T, lb: L, lp: &mut T) -> T {
    let lb: T = lb.lift();
    if lb.value() != f64::NEG_INFINITY {
        *lp += y;
    }
    lb_constrain(y, lb)
}

/// Inverse of [`lb_constrain`]: `y = ln(x - lb)`. Requires `x >= lb`.
pub fn lb_free<T: Real, L: Lift<T>>(x: T, lb: L) -> Result<T> {
    let lb: T = lb.lift();
    if lb.value() == f64::NEG_INFINITY {
        return Ok(x);
    }
    if !(x.value() >= lb.value()) {
        return Err(Error::parameter(
            "lb_free",
            "x",
            x.value(),
            "at least the lower bound",
        ));
    }
    Ok((x - lb).ln())
}

/// `x = ub - exp(y)`; with `ub = +inf` the identity.
pub fn ub_constrain<T: Real, U: Lift<T>>(y: T, ub: U) -> T {
    let ub: T = ub.lift();
    if ub.value() == f64::INFINITY {
        y
    } else {
        ub - y.exp()
    }
}

/// [`ub_constrain`] adding `ln |dx/dy| = y` to `lp`.
pub fn ub_constrain_lp<T: Real, U: Lift<T>>(y: T, ub: U, lp: &mut T) -> T {
    let ub: T = ub.lift();
    if ub.value() != f64::INFINITY {
        *lp += y;
    }
    ub_constrain(y, ub)
}

/// Inverse of [`ub_constrain`]: `y = ln(ub - x)`. Requires `x <= ub`.
pub fn ub_free<T: Real, U: Lift<T>>(x: T, ub: U) -> Result<T> {
    let ub: T = ub.lift();
    if ub.value() == f64::INFINITY {
        return Ok(x);
    }
    if !(x.value() <= ub.value()) {
        return Err(Error::parameter(
            "ub_free",
            "x",
            x.value(),
            "at most the upper bound",
        ));
    }
    Ok((ub - x).ln())
}

fn check_bounds<T: Real>(function: &'static str, lb: T, ub: T) -> Result<()> {
    if lb.value() < ub.value() {
        Ok(())
    } else {
        Err(Error::parameter(
            function,
            "ub",
            ub.value(),
            "greater than lb",
        ))
    }
}

/// `x = lb + (ub - lb) inv_logit(y)`. Infinite bounds reduce to [`lb_constrain`],
/// [`ub_constrain`], or the identity.
pub fn lub_constrain<T: Real, L: Lift<T>, U: Lift<T>>(y: T, lb: L, ub: U) -> Result<T> {
    let lb: T = lb.lift();
    let ub: T = ub.lift();
    check_bounds("lub_constrain", lb, ub)?;
    Ok(match (lb.is_finite(), ub.is_finite()) {
        (false, false) => y,
        (true, false) => lb_constrain(y, lb),
        (false, true) => ub_constrain(y, ub),
        (true, true) => (ub - lb) * inv_logit(y) + lb,
    })
}

/// [`lub_constrain`] adding `ln(ub - lb) + ln inv_logit(y) + ln(1 - inv_logit(y))` to `lp`,
/// evaluated as `ln(ub - lb) - |y| - 2 ln(1 + exp(-|y|))` for stability.
pub fn lub_constrain_lp<T: Real, L: Lift<T>, U: Lift<T>>(
    y: T,
    lb: L,
    ub: U,
    lp: &mut T,
) -> Result<T> {
    let lb: T = lb.lift();
    let ub: T = ub.lift();
    check_bounds("lub_constrain", lb, ub)?;
    Ok(match (lb.is_finite(), ub.is_finite()) {
        (false, false) => y,
        (true, false) => lb_constrain_lp(y, lb, lp),
        (false, true) => ub_constrain_lp(y, ub, lp),
        (true, true) => {
            let abs_y = y.abs();
            *lp += (ub - lb).ln() - abs_y - log1p_exp(-abs_y) * 2.0;
            (ub - lb) * inv_logit(y) + lb
        }
    })
}

/// Inverse of [`lub_constrain`]: `y = logit((x - lb) / (ub - lb))`. Requires `lb <= x <= ub`.
pub fn lub_free<T: Real, L: Lift<T>, U: Lift<T>>(x: T, lb: L, ub: U) -> Result<T> {
    let lb: T = lb.lift();
    let ub: T = ub.lift();
    check_bounds("lub_free", lb, ub)?;
    if !(x.value() >= lb.value() && x.value() <= ub.value()) {
        return Err(Error::parameter(
            "lub_free",
            "x",
            x.value(),
            "within the bounds",
        ));
    }
    Ok(match (lb.is_finite(), ub.is_finite()) {
        (false, false) => x,
        (true, false) => lb_free(x, lb)?,
        (false, true) => ub_free(x, ub)?,
        (true, true) => ((x - lb) / (ub - x)).ln(),
    })
}

fn check_offset_multiplier<T: Real>(function: &'static str, mu: T, sigma: T) -> Result<()> {
    if !mu.is_finite() {
        return Err(Error::parameter(function, "offset", mu.value(), "finite"));
    }
    if !(sigma.value() > 0.0 && sigma.is_finite()) {
        return Err(Error::parameter(
            function,
            "multiplier",
            sigma.value(),
            "positive finite",
        ));
    }
    Ok(())
}

/// `x = mu + sigma y` (the non-centered parameterization).
pub fn offset_multiplier_constrain<T: Real, M: Lift<T>, S: Lift<T>>(
    y: T,
    mu: M,
    sigma: S,
) -> Result<T> {
    let mu: T = mu.lift();
    let sigma: T = sigma.lift();
    check_offset_multiplier("offset_multiplier_constrain", mu, sigma)?;
    Ok(sigma * y + mu)
}

/// [`offset_multiplier_constrain`] adding `ln sigma` to `lp`.
pub fn offset_multiplier_constrain_lp<T: Real, M: Lift<T>, S: Lift<T>>(
    y: T,
    mu: M,
    sigma: S,
    lp: &mut T,
) -> Result<T> {
    let mu: T = mu.lift();
    let sigma: T = sigma.lift();
    check_offset_multiplier("offset_multiplier_constrain", mu, sigma)?;
    *lp += sigma.ln();
    Ok(sigma * y + mu)
}

/// Inverse of [`offset_multiplier_constrain`]: `y = (x - mu) / sigma`.
pub fn offset_multiplier_free<T: Real, M: Lift<T>, S: Lift<T>>(x: T, mu: M, sigma: S) -> Result<T> {
    let mu: T = mu.lift();
    let sigma: T = sigma.lift();
    check_offset_multiplier("offset_multiplier_free", mu, sigma)?;
    Ok((x - mu) / sigma)
}
