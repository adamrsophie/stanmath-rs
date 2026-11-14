//! One-dimensional integration with differentiable limits by adaptive Gauss-Kronrod quadrature and
//! by double exponential rules chosen by interval type.
//!
//! Both functions treat derivatives as [`integrate_1d`](crate::integrate_1d) does: nodes and
//! weights are constants, so derivatives with respect to `θ` come from the same rule applied to
//! the integrand in the caller's scalar type, and derivatives with respect to finite limits come
//! from the limit term in `limits.rs`.

use core::f64::consts::FRAC_PI_2;

use stanmath_rs_autodiff::Real;
use stanmath_rs_core::{Error, Result};

use crate::limits::limit_term;
use crate::quadrature::{Integrand, QuadratureOptions, integrate};

/// Kronrod nodes of the 15 point rule on `[-1, 1]`, largest first; the odd entries
/// (indices 1, 3, 5, 7) are the nodes of the embedded 7 point Gauss rule (Piessens et al. 1983).
#[allow(clippy::excessive_precision)] // published values, kept verbatim
const XGK: [f64; 8] = [
    0.991_455_371_120_812_639_206_854_697_526_329,
    0.949_107_912_342_758_524_526_189_684_047_851,
    0.864_864_423_359_769_072_789_712_788_640_926,
    0.741_531_185_599_394_439_863_864_773_280_788,
    0.586_087_235_467_691_130_294_144_845_693_013,
    0.405_845_151_377_397_166_906_606_412_076_961,
    0.207_784_955_007_898_467_600_689_403_773_245,
    0.0,
];

/// Kronrod weights matching [`XGK`].
#[allow(clippy::excessive_precision)] // published values, kept verbatim
const WGK: [f64; 8] = [
    0.022_935_322_010_529_224_963_732_008_058_970,
    0.063_092_092_629_978_553_290_700_663_189_204,
    0.104_790_010_322_250_183_839_876_322_541_518,
    0.140_653_259_715_525_918_745_189_590_510_238,
    0.169_004_726_639_267_902_826_583_426_598_550,
    0.190_350_578_064_785_409_913_256_402_421_014,
    0.204_432_940_075_298_892_414_161_999_234_649,
    0.209_482_141_084_727_828_012_999_174_891_714,
];

/// Gauss weights for the nodes `XGK[1]`, `XGK[3]`, `XGK[5]`, and `XGK[7]`.
#[allow(clippy::excessive_precision)] // published values, kept verbatim
const WG: [f64; 4] = [
    0.129_484_966_168_869_693_270_611_432_679_082,
    0.279_705_391_489_276_667_901_467_771_423_780,
    0.381_830_050_505_118_944_950_369_775_488_975,
    0.417_959_183_673_469_387_755_102_040_816_327,
];

/// Most panels the adaptive Gauss-Kronrod driver creates before giving up.
const MAX_PANELS: usize = 2000;

fn checked<T: Real>(name: &'static str, value: T) -> Result<T> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(Error::parameter(name, "integrand", value.value(), "finite"))
    }
}

/// One Gauss-Kronrod panel on `[lo, hi]`: the 15 point estimate, the magnitude of its difference
/// from the 7 point estimate, and the 15 point estimate of `∫ |g|` (primal values).
fn gk15<T: Real>(g: &impl Fn(f64) -> Result<T>, lo: f64, hi: f64) -> Result<(T, f64, f64)> {
    let center = 0.5 * (lo + hi);
    let half = 0.5 * (hi - lo);
    let mut kronrod = T::zero();
    let mut gauss = T::zero();
    let mut absolute = 0.0;
    for (i, (&x, &wk)) in XGK.iter().zip(&WGK).enumerate() {
        let offsets = [half * x, -(half * x)];
        let count = if x == 0.0 { 1 } else { 2 };
        for &d in &offsets[..count] {
            let v = g(center + d)?;
            kronrod += v * wk;
            absolute += wk * v.value().abs();
            if i % 2 == 1 {
                gauss += v * WG[i / 2];
            }
        }
    }
    let error = ((kronrod - gauss) * half).magnitude();
    Ok((kronrod * half, error, absolute * half))
}

/// Result of [`adaptive_gauss_kronrod`].
pub(crate) struct Adaptive<T> {
    /// Sum of the panel estimates.
    pub(crate) estimate: T,
    /// Whether the tolerance was met.
    pub(crate) converged: bool,
}

/// Globally adaptive Gauss-Kronrod quadrature of `g` on the finite interval `[a, b]`: the panel
/// with the largest error estimate is bisected until the summed error is below
/// `max(absolute_tolerance, relative_tolerance |I|)`, or at the rounding floor, or until
/// `max_panels` panels exist.
pub(crate) fn adaptive_gauss_kronrod<T: Real>(
    g: &impl Fn(f64) -> Result<T>,
    a: f64,
    b: f64,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    max_panels: usize,
) -> Result<Adaptive<T>> {
    // (lo, hi, estimate, error, integral of |g|)
    let (e, err, abs) = gk15(g, a, b)?;
    let mut panels = vec![(a, b, e, err, abs)];
    // ponytail: linear scan for the worst panel, a binary heap if panel counts grow large.
    let converged = loop {
        let value: f64 = panels.iter().map(|p| p.2.value()).sum();
        let error: f64 = panels.iter().map(|p| p.3).sum();
        let absolute: f64 = panels.iter().map(|p| p.4).sum();
        if error <= absolute_tolerance.max(relative_tolerance * value.abs())
            || error <= 50.0 * f64::EPSILON * absolute
        {
            break true;
        }
        if panels.len() >= max_panels {
            break false;
        }
        let worst = (0..panels.len())
            .max_by(|&i, &j| panels[i].3.total_cmp(&panels[j].3))
            .unwrap_or(0);
        let (lo, hi, ..) = panels[worst];
        let mid = 0.5 * (lo + hi);
        if !(mid > lo && mid < hi) {
            break false;
        }
        let (e1, r1, a1) = gk15(g, lo, mid)?;
        let (e2, r2, a2) = gk15(g, mid, hi)?;
        panels[worst] = (lo, mid, e1, r1, a1);
        panels.push((mid, hi, e2, r2, a2));
    };
    // Sum the estimates once, so a reverse-mode tape records one sum, not one per iteration.
    let mut estimate = T::zero();
    for p in &panels {
        estimate += p.2;
    }
    Ok(Adaptive {
        estimate,
        converged,
    })
}

fn check_limits(name: &'static str, a: f64, b: f64) -> Result<()> {
    if a.is_nan() || b.is_nan() || a > b {
        return Err(Error::parameter(
            name,
            "limits",
            b - a,
            "ordered and not NaN",
        ));
    }
    Ok(())
}

/// `∫_a^b f(x, θ) dx` by globally adaptive 7 point Gauss and 15 point Kronrod quadrature, with
/// derivatives with respect to `θ` and to the finite limits (Stan's
/// `integrate_1d_gauss_kronrod`). Infinite limits are mapped to a finite interval by
/// `x = a + t / (1 - t)`, `x = b - t / (1 - t)`, or `x = t / (1 - t^2)`.
///
/// # Errors
///
/// [`Error::InvalidParameter`] for NaN or reversed limits or an integrand that is not finite at a
/// node, and [`Error::NoConvergence`] when the tolerance is not met.
pub fn integrate_1d_gauss_kronrod<T: Real, F: Integrand>(
    f: &F,
    a: T,
    b: T,
    theta: &[T],
    relative_tolerance: f64,
) -> Result<T> {
    gauss_kronrod(f, a, b, theta, relative_tolerance, 0.0, MAX_PANELS)
}

/// [`integrate_1d_gauss_kronrod`] with Stan's control parameters (Stan's
/// `integrate_1d_gauss_kronrod_tol`): stop when the error estimate is below
/// `max(absolute_tolerance, relative_tolerance |I|)`, and allow at most `2^max_depth` panels per
/// mapped interval (capped at `2^20`).
///
/// # Errors
///
/// As [`integrate_1d_gauss_kronrod`], plus [`Error::InvalidParameter`] for a negative or NaN
/// tolerance.
pub fn integrate_1d_gauss_kronrod_tol<T: Real, F: Integrand>(
    f: &F,
    a: T,
    b: T,
    theta: &[T],
    relative_tolerance: f64,
    absolute_tolerance: f64,
    max_depth: usize,
) -> Result<T> {
    check_tolerances(
        "integrate_1d_gauss_kronrod_tol",
        relative_tolerance,
        absolute_tolerance,
    )?;
    let max_panels = 1usize << max_depth.min(20);
    gauss_kronrod(
        f,
        a,
        b,
        theta,
        relative_tolerance,
        absolute_tolerance,
        max_panels,
    )
}

fn check_tolerances(name: &'static str, relative: f64, absolute: f64) -> Result<()> {
    if !(relative >= 0.0) {
        return Err(Error::parameter(
            name,
            "relative_tolerance",
            relative,
            "nonnegative",
        ));
    }
    if !(absolute >= 0.0) {
        return Err(Error::parameter(
            name,
            "absolute_tolerance",
            absolute,
            "nonnegative",
        ));
    }
    Ok(())
}

fn gauss_kronrod<T: Real, F: Integrand>(
    f: &F,
    a: T,
    b: T,
    theta: &[T],
    relative_tolerance: f64,
    absolute_tolerance: f64,
    max_panels: usize,
) -> Result<T> {
    const NAME: &str = "integrate_1d_gauss_kronrod";
    let (a0, b0) = (a.value(), b.value());
    check_limits(NAME, a0, b0)?;
    let at = |x: f64, jacobian: f64| -> Result<T> {
        checked(NAME, f.eval(T::from_f64(x), theta) * jacobian)
    };
    let run = |g: &dyn Fn(f64) -> Result<T>, lo: f64, hi: f64| {
        adaptive_gauss_kronrod(
            &g,
            lo,
            hi,
            relative_tolerance,
            absolute_tolerance,
            max_panels,
        )
    };
    let core = if a0 == b0 {
        None
    } else {
        Some(match (a0.is_finite(), b0.is_finite()) {
            (true, true) => run(&|x: f64| at(x, 1.0), a0, b0)?,
            (true, false) => run(
                &|t: f64| {
                    let q = 1.0 - t;
                    at(a0 + t / q, 1.0 / (q * q))
                },
                0.0,
                1.0,
            )?,
            (false, true) => run(
                &|t: f64| {
                    let q = 1.0 - t;
                    at(b0 - t / q, 1.0 / (q * q))
                },
                0.0,
                1.0,
            )?,
            (false, false) => run(
                &|t: f64| {
                    let q = 1.0 - t * t;
                    at(t / q, (1.0 + t * t) / (q * q))
                },
                -1.0,
                1.0,
            )?,
        })
    };
    let core = match core {
        None => T::zero(),
        Some(r) if r.converged => r.estimate,
        Some(_) => {
            return Err(Error::NoConvergence {
                function: NAME,
                iterations: max_panels,
            });
        }
    };
    Ok(core + limit_term(f, b, theta) - limit_term(f, a, theta))
}

/// Trapezoidal sums of `w(t) g(x(t))` over `t = k h`, halving `h` until successive estimates agree
/// to `relative_tolerance`. `node(t)` returns `(x, w)`, or `None` once the node leaves the
/// representable range on that side.
fn double_exponential<T: Real>(
    name: &'static str,
    node: impl Fn(f64) -> Option<(f64, f64)>,
    g: impl Fn(f64) -> T,
    o: &QuadratureOptions,
    absolute_tolerance: f64,
) -> Result<T> {
    // Beyond |t| = 8 every rule here has left the range of f64.
    const T_MAX: f64 = 8.0;
    let level = |h: f64, odd_only: bool, scale: f64| -> Result<T> {
        let mut total = T::zero();
        let center = if odd_only { None } else { node(0.0) };
        if let Some((x, w)) = center {
            total += checked(name, g(x))? * w;
        }
        for side in [1.0, -1.0] {
            let mut k: i64 = 1;
            let mut small = 0;
            loop {
                let t = side * k as f64 * h;
                let Some((x, w)) = node(t) else { break };
                if t.abs() > T_MAX {
                    break;
                }
                let term = checked(name, g(x))? * w;
                // Three negligible terms in a row end the side where weights grow.
                if term.magnitude() <= f64::EPSILON * f64::EPSILON * total.magnitude().max(scale) {
                    small += 1;
                    if small == 3 {
                        break;
                    }
                } else {
                    small = 0;
                }
                total += term;
                k += if odd_only { 2 } else { 1 };
            }
        }
        Ok(total)
    };
    let mut h = 1.0;
    let mut estimate = level(h, false, 0.0)? * h;
    for _ in 0..o.max_levels {
        h *= 0.5;
        let refined = estimate * 0.5 + level(h, true, estimate.magnitude())? * h;
        let change = (refined - estimate).magnitude();
        estimate = refined;
        let relative = o.relative_tolerance * estimate.magnitude().max(f64::MIN_POSITIVE);
        if change <= absolute_tolerance.max(relative) {
            return Ok(estimate);
        }
    }
    Err(Error::NoConvergence {
        function: name,
        iterations: o.max_levels,
    })
}

/// `∫_a^b f(x, θ) dx` by a double exponential rule chosen by interval type, with derivatives with
/// respect to `θ` and to the finite limits (Stan's `integrate_1d_double_exponential`):
///
/// - finite `[a, b]`: tanh-sinh, `x = c + h tanh(pi/2 sinh t)` (the rule of [`integrate`]);
/// - `[a, ∞)` or `(-∞, b]`: exp-sinh, `x = a + exp(pi/2 sinh t)` or `x = b - exp(pi/2 sinh t)`;
/// - `(-∞, ∞)`: sinh-sinh, `x = sinh(pi/2 sinh t)`.
///
/// # Errors
///
/// As [`integrate_1d_gauss_kronrod`].
pub fn integrate_1d_double_exponential<T: Real, F: Integrand>(
    f: &F,
    a: T,
    b: T,
    theta: &[T],
    relative_tolerance: f64,
) -> Result<T> {
    let max_refinements = QuadratureOptions::default().max_levels;
    integrate_1d_double_exponential_tol(f, a, b, theta, relative_tolerance, 0.0, max_refinements)
}

/// [`integrate_1d_double_exponential`] with Stan's control parameters (Stan's
/// `integrate_1d_double_exponential_tol`): stop when successive refinements differ by less than
/// `max(absolute_tolerance, relative_tolerance |I|)`, after at most `max_refinements` halvings of
/// the step. A finite interval with `absolute_tolerance == 0` uses [`integrate`]; with a positive
/// absolute tolerance it uses the same tanh-sinh rule through the local driver.
///
/// # Errors
///
/// As [`integrate_1d_gauss_kronrod_tol`].
pub fn integrate_1d_double_exponential_tol<T: Real, F: Integrand>(
    f: &F,
    a: T,
    b: T,
    theta: &[T],
    relative_tolerance: f64,
    absolute_tolerance: f64,
    max_refinements: usize,
) -> Result<T> {
    const NAME: &str = "integrate_1d_double_exponential";
    check_tolerances(NAME, relative_tolerance, absolute_tolerance)?;
    let (a0, b0) = (a.value(), b.value());
    check_limits(NAME, a0, b0)?;
    let options = QuadratureOptions {
        relative_tolerance,
        max_levels: max_refinements,
    };
    let g = |x: f64| f.eval(T::from_f64(x), theta);
    let exp_sinh = |t: f64| {
        let e = (FRAC_PI_2 * t.sinh()).exp();
        let w = FRAC_PI_2 * t.cosh() * e;
        (e.is_finite() && w.is_finite() && w > 1e-300).then_some((e, w))
    };
    let core = if a0 == b0 {
        T::zero()
    } else {
        match (a0.is_finite(), b0.is_finite()) {
            (true, true) if absolute_tolerance == 0.0 => integrate(f, a0, b0, theta, &options)?,
            (true, true) => {
                let (c, h) = (0.5 * (a0 + b0), 0.5 * (b0 - a0));
                double_exponential(
                    NAME,
                    |t| {
                        let u = FRAC_PI_2 * t.sinh();
                        let x = c + h * u.tanh();
                        let w = h * FRAC_PI_2 * t.cosh() / u.cosh().powi(2);
                        (x > a0 && x < b0 && w.is_finite() && w > 0.0).then_some((x, w))
                    },
                    g,
                    &options,
                    absolute_tolerance,
                )?
            }
            (true, false) => double_exponential(
                NAME,
                |t| exp_sinh(t).and_then(|(e, w)| (a0 + e != a0).then_some((a0 + e, w))),
                g,
                &options,
                absolute_tolerance,
            )?,
            (false, true) => double_exponential(
                NAME,
                |t| exp_sinh(t).and_then(|(e, w)| (b0 - e != b0).then_some((b0 - e, w))),
                g,
                &options,
                absolute_tolerance,
            )?,
            (false, false) => double_exponential(
                NAME,
                |t| {
                    let s = FRAC_PI_2 * t.sinh();
                    let (x, w) = (s.sinh(), FRAC_PI_2 * t.cosh() * s.cosh());
                    (x.is_finite() && w.is_finite()).then_some((x, w))
                },
                g,
                &options,
                absolute_tolerance,
            )?,
        }
    };
    Ok(core + limit_term(f, b, theta) - limit_term(f, a, theta))
}
