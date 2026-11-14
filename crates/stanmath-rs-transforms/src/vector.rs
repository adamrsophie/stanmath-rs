//! Vector constraints: ordered, positive ordered, sum to zero, simplex, unit vector.

use stanmath_rs_autodiff::Real;
use stanmath_rs_autodiff::functions::log_softmax;
use stanmath_rs_core::{Error, Result};

/// Tolerance of membership checks in the `*_free` functions, as Stan's `CONSTRAINT_TOLERANCE`.
pub const CONSTRAINT_TOLERANCE: f64 = 1e-8;

/// Strictly increasing vector: `x[0] = y[0]`, `x[i] = x[i-1] + exp(y[i])`.
pub fn ordered_constrain<T: Real>(y: &[T]) -> Vec<T> {
    let mut x = Vec::with_capacity(y.len());
    for (i, &yi) in y.iter().enumerate() {
        x.push(if i == 0 { yi } else { x[i - 1] + yi.exp() });
    }
    x
}

/// [`ordered_constrain`] adding `sum_{i >= 1} y[i]` to `lp`.
pub fn ordered_constrain_lp<T: Real>(y: &[T], lp: &mut T) -> Vec<T> {
    for &yi in y.iter().skip(1) {
        *lp += yi;
    }
    ordered_constrain(y)
}

/// Inverse of [`ordered_constrain`]. Requires a strictly increasing `x`.
pub fn ordered_free<T: Real>(x: &[T]) -> Result<Vec<T>> {
    for w in x.windows(2) {
        if !(w[1].value() > w[0].value()) {
            return Err(Error::parameter(
                "ordered_free",
                "x",
                w[1].value(),
                "strictly increasing",
            ));
        }
    }
    Ok(x.iter()
        .enumerate()
        .map(|(i, &xi)| if i == 0 { xi } else { (xi - x[i - 1]).ln() })
        .collect())
}

/// Strictly increasing positive vector: `x[0] = exp(y[0])`, `x[i] = x[i-1] + exp(y[i])`.
pub fn positive_ordered_constrain<T: Real>(y: &[T]) -> Vec<T> {
    let mut x: Vec<T> = Vec::with_capacity(y.len());
    for (i, &yi) in y.iter().enumerate() {
        x.push(if i == 0 {
            yi.exp()
        } else {
            x[i - 1] + yi.exp()
        });
    }
    x
}

/// [`positive_ordered_constrain`] adding `sum_i y[i]` to `lp`.
pub fn positive_ordered_constrain_lp<T: Real>(y: &[T], lp: &mut T) -> Vec<T> {
    for &yi in y {
        *lp += yi;
    }
    positive_ordered_constrain(y)
}

/// Inverse of [`positive_ordered_constrain`]. Requires a positive, strictly increasing `x`.
pub fn positive_ordered_free<T: Real>(x: &[T]) -> Result<Vec<T>> {
    if let Some(first) = x.first() {
        if !(first.value() > 0.0) {
            return Err(Error::parameter(
                "positive_ordered_free",
                "x",
                first.value(),
                "positive",
            ));
        }
    }
    let mut y = ordered_free(x)?;
    if let Some(first) = y.first_mut() {
        *first = first.ln();
    }
    Ok(y)
}

/// Sum-to-zero vector of length `N + 1` from `N` unconstrained values, by Stan Math 5.4's
/// isometric transform (the Helmert-like orthonormal basis of the hyperplane). No Jacobian term.
pub fn sum_to_zero_constrain<T: Real>(y: &[T]) -> Vec<T> {
    let n = y.len();
    let mut z = vec![T::zero(); n + 1];
    let mut sum_w = T::zero();
    for i in (1..=n).rev() {
        let nf = i as f64;
        let w = y[i - 1] / (nf * (nf + 1.0)).sqrt();
        sum_w += w;
        z[i - 1] += sum_w;
        z[i] -= w * nf;
    }
    z
}

/// Inverse of [`sum_to_zero_constrain`]. Requires `|sum(z)| <= 1e-8`.
pub fn sum_to_zero_free<T: Real>(z: &[T]) -> Result<Vec<T>> {
    let total: f64 = z.iter().map(Real::value).sum();
    if z.is_empty() || !(total.abs() <= CONSTRAINT_TOLERANCE) {
        return Err(Error::parameter(
            "sum_to_zero_free",
            "z",
            total,
            "nonempty, summing to zero",
        ));
    }
    let n = z.len() - 1;
    let mut y = vec![T::zero(); n];
    if n == 0 {
        return Ok(y);
    }
    let nf = n as f64;
    y[n - 1] = -(z[n] * (nf * (nf + 1.0)).sqrt() / nf);
    let mut sum_w = T::zero();
    for i in (0..n - 1).rev() {
        let k = (i + 1) as f64;
        let w = y[i + 1] / ((k + 1.0) * (k + 2.0)).sqrt();
        sum_w += w;
        y[i] = (sum_w - z[i + 1]) * (k * (k + 1.0)).sqrt() / k;
    }
    Ok(y)
}

/// Simplex of length `N + 1` from `N` unconstrained values: the softmax of the sum-to-zero
/// transform, as in Stan Math 5.4.
pub fn simplex_constrain<T: Real>(y: &[T]) -> Vec<T> {
    log_softmax(&sum_to_zero_constrain(y))
        .into_iter()
        .map(Real::exp)
        .collect()
}

/// [`simplex_constrain`] adding `sum_i ln x[i] + ln(N + 1) / 2` to `lp`.
pub fn simplex_constrain_lp<T: Real>(y: &[T], lp: &mut T) -> Vec<T> {
    let log_x = log_softmax(&sum_to_zero_constrain(y));
    for &l in &log_x {
        *lp += l;
    }
    *lp += 0.5 * ((y.len() + 1) as f64).ln();
    log_x.into_iter().map(Real::exp).collect()
}

/// Inverse of [`simplex_constrain`]: the sum-to-zero inverse of the centered log ratio.
/// Requires positive entries summing to one within `1e-8` (a zero entry has no unconstrained
/// preimage).
pub fn simplex_free<T: Real>(x: &[T]) -> Result<Vec<T>> {
    let total: f64 = x.iter().map(Real::value).sum();
    if x.is_empty()
        || x.iter().any(|v| !(v.value() > 0.0))
        || !((total - 1.0).abs() <= CONSTRAINT_TOLERANCE)
    {
        return Err(Error::parameter(
            "simplex_free",
            "x",
            total,
            "positive entries summing to one",
        ));
    }
    let logs: Vec<T> = x.iter().map(|v| v.ln()).collect();
    let mean = logs.iter().copied().sum::<T>() / logs.len() as f64;
    let centered: Vec<T> = logs.iter().map(|&l| l - mean).collect();
    sum_to_zero_free(&centered)
}

/// Unit vector `y / |y|`. Requires `|y| > 0` and finite.
pub fn unit_vector_constrain<T: Real>(y: &[T]) -> Result<Vec<T>> {
    let sq: T = y.iter().map(|v| v.square()).sum();
    if !(sq.value() > 0.0 && sq.is_finite()) {
        return Err(Error::parameter(
            "unit_vector_constrain",
            "norm",
            sq.value(),
            "positive finite",
        ));
    }
    let norm = sq.sqrt();
    Ok(y.iter().map(|&v| v / norm).collect())
}

/// [`unit_vector_constrain`] adding `-|y|^2 / 2` to `lp`.
pub fn unit_vector_constrain_lp<T: Real>(y: &[T], lp: &mut T) -> Result<Vec<T>> {
    let x = unit_vector_constrain(y)?;
    let sq: T = y.iter().map(|v| v.square()).sum();
    *lp -= sq * 0.5;
    Ok(x)
}

/// Inverse of [`unit_vector_constrain`] (the identity). Requires `|x| = 1` within `1e-8`.
pub fn unit_vector_free<T: Real>(x: &[T]) -> Result<Vec<T>> {
    let sq: f64 = x.iter().map(|v| v.value() * v.value()).sum();
    if !((sq - 1.0).abs() <= CONSTRAINT_TOLERANCE) {
        return Err(Error::parameter(
            "unit_vector_free",
            "x",
            sq,
            "of unit length",
        ));
    }
    Ok(x.to_vec())
}
