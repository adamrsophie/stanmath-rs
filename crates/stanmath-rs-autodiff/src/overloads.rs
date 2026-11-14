//! Structural overloads of scalar functions that take nested containers.

use crate::Real;
use crate::functions_extra::log_mix_slice;

/// `sum_n log_mix(theta, lps[n])`: Stan's `log_mix(thetas, lps)` when `lps` is an array of
/// vectors, row vectors, or arrays, one per observation. The Functions Reference allows `lps`
/// to be "an array of such" containers and returns a real; this reads it as the sum over the
/// array, the log density of independent observations. Zero for an empty `lps`.
///
/// # Panics
///
/// Panics if some `lps[n]` differs in length from `theta`, as [`log_mix_slice`].
pub fn log_mix_array<T: Real>(theta: &[T], lps: &[Vec<T>]) -> T {
    lps.iter().map(|lp| log_mix_slice(theta, lp)).sum()
}
