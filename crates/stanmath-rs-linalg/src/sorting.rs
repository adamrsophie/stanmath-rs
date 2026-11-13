//! Sorting, ranking, sample quantiles, and reductions over containers, with Stan's names.
//!
//! The sorting functions take any `T: PartialOrd + Clone`, so the same function sorts integer
//! data (`i64`) and real values of every autodiff type; comparisons use primal values, and a
//! sorted vector of variables keeps each variable's derivative. Returned indices are one based,
//! as in Stan. Ties keep their original order (the sort is stable); Stan does not specify the
//! order of ties.

use core::cmp::Ordering;

use stanmath_rs_autodiff::Real;
use stanmath_rs_core::{Error, Result};

fn check_not_nan<T: PartialOrd>(function: &'static str, v: &[T]) -> Result<()> {
    // Only NaN fails to compare with itself.
    if v.iter().any(|x| x.partial_cmp(x).is_none()) {
        Err(Error::parameter(function, "v", f64::NAN, "free of NaN"))
    } else {
        Ok(())
    }
}

fn ascending<T: PartialOrd>(a: &T, b: &T) -> Ordering {
    a.partial_cmp(b).unwrap_or(Ordering::Equal)
}

/// The elements of `v` in ascending order.
///
/// # Errors
///
/// [`Error::InvalidParameter`] when `v` contains NaN.
pub fn sort_asc<T: PartialOrd + Clone>(v: &[T]) -> Result<Vec<T>> {
    check_not_nan("sort_asc", v)?;
    let mut out = v.to_vec();
    out.sort_by(ascending);
    Ok(out)
}

/// The elements of `v` in descending order.
///
/// # Errors
///
/// [`Error::InvalidParameter`] when `v` contains NaN.
pub fn sort_desc<T: PartialOrd + Clone>(v: &[T]) -> Result<Vec<T>> {
    check_not_nan("sort_desc", v)?;
    let mut out = v.to_vec();
    out.sort_by(|a, b| ascending(b, a));
    Ok(out)
}

fn sort_indices<T: PartialOrd>(
    function: &'static str,
    v: &[T],
    order: impl Fn(&T, &T) -> Ordering,
) -> Result<Vec<usize>> {
    check_not_nan(function, v)?;
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&i, &j| order(&v[i], &v[j]));
    Ok(idx.into_iter().map(|i| i + 1).collect())
}

/// One based indices that sort `v` in ascending order: `v[idx[0] - 1]` is the smallest element.
///
/// # Errors
///
/// [`Error::InvalidParameter`] when `v` contains NaN.
pub fn sort_indices_asc<T: PartialOrd>(v: &[T]) -> Result<Vec<usize>> {
    sort_indices("sort_indices_asc", v, ascending)
}

/// One based indices that sort `v` in descending order.
///
/// # Errors
///
/// [`Error::InvalidParameter`] when `v` contains NaN.
pub fn sort_indices_desc<T: PartialOrd>(v: &[T]) -> Result<Vec<usize>> {
    sort_indices("sort_indices_desc", v, |a, b| ascending(b, a))
}

/// Number of elements of `v` strictly less than `v[s]`, with `s` one based.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when `s` is not in `1..=v.len()`.
pub fn rank<T: PartialOrd>(v: &[T], s: usize) -> Result<usize> {
    if !(1..=v.len()).contains(&s) {
        return Err(Error::dimension(
            "rank",
            format!("index {s} is out of range for size {}", v.len()),
        ));
    }
    let pivot = &v[s - 1];
    Ok(v.iter().filter(|x| *x < pivot).count())
}

fn check_sample(x: &[f64]) -> Result<Vec<f64>> {
    if x.is_empty() {
        return Err(Error::dimension("quantile", "the sample is empty"));
    }
    if let Some(&bad) = x.iter().find(|v| v.is_nan()) {
        return Err(Error::observation("quantile", "x", bad, "not NaN"));
    }
    let mut sorted = x.to_vec();
    sorted.sort_by(f64::total_cmp);
    Ok(sorted)
}

/// Sample quantile type 7 of Hyndman and Fan (1996) on sorted data: linear interpolation
/// between order statistics at position `(n - 1) p`.
fn type7(sorted: &[f64], p: f64) -> Result<f64> {
    if !(0.0..=1.0).contains(&p) {
        return Err(Error::parameter("quantile", "p", p, "in [0, 1]"));
    }
    let h = (sorted.len() - 1) as f64 * p;
    let lo = h.floor() as usize;
    Ok(match sorted.get(lo + 1) {
        Some(&hi) => sorted[lo] + (h - lo as f64) * (hi - sorted[lo]),
        None => sorted[lo],
    })
}

/// Sample quantile of the data `x` at probability `p`, by linear interpolation between order
/// statistics (type 7 of Hyndman and Fan, the default of R and NumPy), as Stan computes it.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when `x` is empty, [`Error::InvalidObservation`] when it contains
/// NaN, and [`Error::InvalidParameter`] when `p` is not in `[0, 1]`.
pub fn quantile(x: &[f64], p: f64) -> Result<f64> {
    type7(&check_sample(x)?, p)
}

/// [`quantile`] at each probability in `p` (Stan's array overload); sorts `x` once.
///
/// # Errors
///
/// As [`quantile`].
pub fn quantile_array(x: &[f64], p: &[f64]) -> Result<Vec<f64>> {
    let sorted = check_sample(x)?;
    p.iter().map(|&pi| type7(&sorted, pi)).collect()
}

/// Product of the elements; one for an empty container.
pub fn prod<T: Real>(x: &[T]) -> T {
    x.iter().fold(T::one(), |acc, &v| acc * v)
}

/// Product of integer elements; one for an empty array.
pub fn prod_int(x: &[i64]) -> i64 {
    x.iter().product()
}

/// Largest element; negative infinity for an empty container, as in Stan. The derivative flows
/// to the first largest element.
pub fn max<T: Real>(x: &[T]) -> T {
    x.iter().fold(T::from_f64(f64::NEG_INFINITY), |acc, &v| {
        if v > acc { v } else { acc }
    })
}

/// Smallest element; positive infinity for an empty container, as in Stan. The derivative flows
/// to the first smallest element.
pub fn min<T: Real>(x: &[T]) -> T {
    x.iter().fold(
        T::from_f64(f64::INFINITY),
        |acc, &v| if v < acc { v } else { acc },
    )
}

/// Largest element of an integer array.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when `x` is empty (Stan rejects it, since integers have no
/// infinity).
pub fn max_int(x: &[i64]) -> Result<i64> {
    x.iter()
        .copied()
        .max()
        .ok_or_else(|| Error::dimension("max", "the array is empty"))
}

/// Smallest element of an integer array.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when `x` is empty.
pub fn min_int(x: &[i64]) -> Result<i64> {
    x.iter()
        .copied()
        .min()
        .ok_or_else(|| Error::dimension("min", "the array is empty"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorts_and_ranks() {
        let v = [3.0, 1.0, 2.0, 1.0];
        assert_eq!(sort_asc(&v).unwrap(), vec![1.0, 1.0, 2.0, 3.0]);
        assert_eq!(sort_desc(&[2, 5, 1]).unwrap(), vec![5, 2, 1]);
        assert_eq!(sort_indices_asc(&v).unwrap(), vec![2, 4, 3, 1]);
        assert_eq!(sort_indices_desc(&v).unwrap(), vec![1, 3, 2, 4]);
        assert_eq!(rank(&v, 3).unwrap(), 2);
        assert!(rank(&v, 0).is_err());
        assert!(sort_asc(&[1.0, f64::NAN]).is_err());
    }

    #[test]
    fn quantile_type_7() {
        let x = [4.0, 1.0, 3.0, 2.0];
        assert_eq!(quantile(&x, 0.0).unwrap(), 1.0);
        assert_eq!(quantile(&x, 1.0).unwrap(), 4.0);
        assert_eq!(quantile(&x, 0.5).unwrap(), 2.5);
        assert_eq!(quantile_array(&x, &[0.25, 0.75]).unwrap(), vec![1.75, 3.25]);
        assert!(quantile(&[], 0.5).is_err());
        assert!(quantile(&x, 1.5).is_err());
    }

    #[test]
    fn reductions() {
        assert_eq!(prod(&[2.0, 3.0, 4.0]), 24.0);
        assert_eq!(prod::<f64>(&[]), 1.0);
        assert_eq!(max(&[1.0, 5.0, 2.0]), 5.0);
        assert_eq!(min::<f64>(&[]), f64::INFINITY);
        assert!(max_int(&[]).is_err());
        assert_eq!(min_int(&[4, -1, 7]).unwrap(), -1);
    }
}
