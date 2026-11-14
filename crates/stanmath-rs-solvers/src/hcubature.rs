//! Adaptive integration over a hyperrectangle (Stan's `hcubature`).
//!
//! Each region is integrated by the degree 7 rule of Genz and Malik (1980) with its embedded
//! degree 5 rule; their difference estimates the error. The region with the largest error is
//! bisected along the coordinate with the largest fourth divided difference of the integrand, as in
//! Berntsen, Espelid and Genz (1991). One-dimensional problems use adaptive Gauss-Kronrod
//! quadrature instead. Nodes and weights are constants, so derivatives with respect to the
//! parameters come from the same rules applied in the caller's scalar type.

use stanmath_rs_autodiff::Real;
use stanmath_rs_core::{Error, Result};

use crate::quadrature_wave2::adaptive_gauss_kronrod;

/// An integrand `f(x, θ)` of a point `x` in a hyperrectangle.
pub trait CubatureIntegrand {
    /// Evaluates `f(x, θ)`.
    fn eval<S: Real>(&self, x: &[S], theta: &[S]) -> S;
}

/// A region with its rule estimate, error estimate, and preferred split coordinate.
struct Region<T> {
    center: Vec<f64>,
    half: Vec<f64>,
    estimate: T,
    error: f64,
    split: usize,
}

/// Applies the Genz and Malik rules to the region with `center` and half widths `half`.
#[allow(clippy::needless_range_loop)] // several arrays share each coordinate index
fn genz_malik<T: Real>(
    g: &impl Fn(&[f64]) -> Result<T>,
    center: Vec<f64>,
    half: Vec<f64>,
) -> Result<Region<T>> {
    let n = center.len();
    let nf = n as f64;
    let l2 = (9.0f64 / 70.0).sqrt();
    let l4 = (9.0f64 / 10.0).sqrt();
    let l5 = (9.0f64 / 19.0).sqrt();
    let mut x = center.clone();
    let fc = g(&x)?;
    let (mut sum2, mut sum3, mut sum4, mut sum5) = (T::zero(), T::zero(), T::zero(), T::zero());
    let mut fourth = vec![0.0; n];
    for i in 0..n {
        let mut pair = |lambda: f64| -> Result<T> {
            x[i] = center[i] + lambda * half[i];
            let p = g(&x)?;
            x[i] = center[i] - lambda * half[i];
            let m = g(&x)?;
            x[i] = center[i];
            Ok(p + m)
        };
        let s2 = pair(l2)?;
        let s3 = pair(l4)?;
        let twice_center = fc.value() * 2.0;
        fourth[i] = ((s2.value() - twice_center) - (s3.value() - twice_center) / 7.0).abs();
        sum2 += s2;
        sum3 += s3;
    }
    for i in 0..n {
        for j in i + 1..n {
            for (si, sj) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
                x[i] = center[i] + si * l4 * half[i];
                x[j] = center[j] + sj * l4 * half[j];
                sum4 += g(&x)?;
            }
            x[i] = center[i];
            x[j] = center[j];
        }
    }
    // ponytail: 2^n corner points, fine for the low dimensions Stan uses; a sparser rule if n grows.
    for signs in 0..1usize << n {
        for k in 0..n {
            let s = if (signs >> k) & 1 == 1 { -1.0 } else { 1.0 };
            x[k] = center[k] + s * l5 * half[k];
        }
        sum5 += g(&x)?;
    }
    let volume: f64 = half.iter().map(|h| 2.0 * h).product();
    let w1 = (12824.0 - 9120.0 * nf + 400.0 * nf * nf) / 19683.0;
    let w2 = 980.0 / 6561.0;
    let w3 = (1820.0 - 400.0 * nf) / 19683.0;
    let w4 = 200.0 / 19683.0;
    let w5 = 6859.0 / 19683.0 / 2f64.powi(n as i32);
    let v1 = (729.0 - 950.0 * nf + 50.0 * nf * nf) / 729.0;
    let v2 = 245.0 / 486.0;
    let v3 = (265.0 - 100.0 * nf) / 1458.0;
    let v4 = 25.0 / 729.0;
    let degree7 = (fc * w1 + sum2 * w2 + sum3 * w3 + sum4 * w4 + sum5 * w5) * volume;
    let degree5 = (fc * v1 + sum2 * v2 + sum3 * v3 + sum4 * v4) * volume;
    let error = (degree7 - degree5).magnitude();
    // Largest fourth difference; ties (for example all zero) go to the widest side.
    let split = (0..n)
        .max_by(|&i, &j| {
            fourth[i]
                .total_cmp(&fourth[j])
                .then(half[i].total_cmp(&half[j]))
        })
        .unwrap_or(0);
    Ok(Region {
        center,
        half,
        estimate: degree7,
        error,
        split,
    })
}

/// `∫ f(x, θ) dx` over the hyperrectangle `lower <= x <= upper` of dimension `dim`, with
/// derivatives with respect to `θ` (Stan's `hcubature`).
///
/// Subdivision stops when the summed error estimate is at most `absolute_tolerance` or
/// `relative_tolerance` times the magnitude of the integral, or when another bisection would
/// exceed `max_eval` integrand evaluations; in the last case the current estimate is returned, as
/// Stan does. A `max_eval` of zero means no limit.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when `dim` is zero or above 20, or the limits do not have length
/// `dim`; [`Error::InvalidParameter`] for limits that are not finite or not ordered, and for an
/// integrand that is not finite at a node.
#[allow(clippy::too_many_arguments)] // mirrors Stan's argument list
pub fn hcubature<T: Real, F: CubatureIntegrand>(
    f: &F,
    theta: &[T],
    dim: usize,
    lower: &[f64],
    upper: &[f64],
    max_eval: usize,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) -> Result<T> {
    const NAME: &str = "hcubature";
    // The degree 7 rule has 2^dim corner points.
    if dim == 0 || dim > 20 || lower.len() != dim || upper.len() != dim {
        return Err(Error::dimension(
            NAME,
            format!(
                "dimension {dim}, {} lower limits, {} upper limits",
                lower.len(),
                upper.len()
            ),
        ));
    }
    for (&a, &b) in lower.iter().zip(upper) {
        if !(a.is_finite() && b.is_finite() && a <= b) {
            return Err(Error::parameter(
                NAME,
                "limits",
                b - a,
                "finite with lower <= upper",
            ));
        }
    }
    let g = |x: &[f64]| -> Result<T> {
        let point: Vec<T> = x.iter().map(|&v| T::from_f64(v)).collect();
        let value = f.eval(&point, theta);
        if value.is_finite() {
            Ok(value)
        } else {
            Err(Error::parameter(NAME, "integrand", value.value(), "finite"))
        }
    };
    if dim == 1 {
        // A Gauss-Kronrod panel costs 15 evaluations, and p panels cost 15 (2 p - 1).
        let max_panels = if max_eval == 0 {
            usize::MAX
        } else {
            (max_eval / 15).div_ceil(2).max(1)
        };
        let r = adaptive_gauss_kronrod(
            &|x: f64| g(&[x]),
            lower[0],
            upper[0],
            relative_tolerance,
            absolute_tolerance,
            max_panels,
        )?;
        return Ok(r.estimate);
    }
    let per_region = 1 + 4 * dim + 2 * dim * (dim - 1) + (1usize << dim);
    let center: Vec<f64> = lower
        .iter()
        .zip(upper)
        .map(|(a, b)| 0.5 * (a + b))
        .collect();
    let half: Vec<f64> = lower
        .iter()
        .zip(upper)
        .map(|(a, b)| 0.5 * (b - a))
        .collect();
    let mut regions = vec![genz_malik(&g, center, half)?];
    let mut evaluations = per_region;
    // ponytail: linear scan for the worst region, a binary heap if region counts grow large.
    loop {
        let value: f64 = regions.iter().map(|r| r.estimate.value()).sum();
        let error: f64 = regions.iter().map(|r| r.error).sum();
        if error <= absolute_tolerance.max(relative_tolerance * value.abs()) {
            break;
        }
        if max_eval > 0 && evaluations + 2 * per_region > max_eval {
            break;
        }
        let worst = (0..regions.len())
            .max_by(|&i, &j| regions[i].error.total_cmp(&regions[j].error))
            .unwrap_or(0);
        let k = regions[worst].split;
        let (c, h) = (regions[worst].center[k], regions[worst].half[k] * 0.5);
        if !(c - h < c + h) {
            // The region can no longer be split in floating point.
            break;
        }
        let r = regions.swap_remove(worst);
        let mut half = r.half;
        half[k] = h;
        let (mut left, mut right) = (r.center.clone(), r.center);
        left[k] = c - h;
        right[k] = c + h;
        regions.push(genz_malik(&g, left, half.clone())?);
        regions.push(genz_malik(&g, right, half)?);
        evaluations += 2 * per_region;
    }
    // Sum the estimates once, so a reverse-mode tape records one sum, not one per iteration.
    let mut total = T::zero();
    for r in &regions {
        total += r.estimate;
    }
    Ok(total)
}
