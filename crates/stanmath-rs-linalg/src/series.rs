//! Sample autocovariance and autocorrelation of a sequence.

use stanmath_rs_autodiff::Real;
use stanmath_rs_autodiff::functions::{dot_product, sum};

/// Sample autocovariance at every lag `k = 0, ..., N - 1`:
/// `acov[k] = (1 / N) sum_{t < N - k} (y_t - ȳ)(y_{t+k} - ȳ)`.
///
/// The divisor is `N` at every lag (the biased estimator, as in Stan), so `acov[0]` is the
/// variance with divisor `N`. An empty input gives an empty result.
pub fn autocovariance<T: Real>(y: &[T]) -> Vec<T> {
    let n = y.len();
    if n == 0 {
        return Vec::new();
    }
    let nf = n as f64;
    let mean = sum(y) / nf;
    let d: Vec<T> = y.iter().map(|&v| v - mean).collect();
    // ponytail: direct O(N^2) sum; switch to a zero-padded FFT when long chains make it slow.
    (0..n)
        .map(|k| dot_product(&d[..n - k], &d[k..]) / nf)
        .collect()
}

/// Sample autocorrelation at every lag, `acov[k] / acov[0]` with [`autocovariance`].
///
/// A constant sequence has zero variance and gives `NaN` at every lag. An empty input gives an
/// empty result.
pub fn autocorrelation<T: Real>(y: &[T]) -> Vec<T> {
    let acov = autocovariance(y);
    let Some(&var) = acov.first() else {
        return acov;
    };
    acov.iter().map(|&c| c / var).collect()
}
