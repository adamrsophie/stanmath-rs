//! Argument validation following Stan Math's checks.
//!
//! Observations ("random variables" in Stan's messages) produce [`Error::InvalidObservation`];
//! everything else produces [`Error::InvalidParameter`].
//!
//! Every check accepts a value or an `Option`; `None` stands for an empty sequence argument and
//! passes. Stan validates the remaining arguments when a sequence is empty (checked against Stan
//! Math 5.4.0), so a density over an empty sequence still rejects an invalid scalar parameter.

use stanmath_rs_core::{Error, Result};

#[inline]
fn require(ok: bool, err: impl FnOnce() -> Error) -> Result<()> {
    if ok { Ok(()) } else { Err(err()) }
}

/// Runs `check` on a present value; an absent one passes.
#[inline]
fn on<T>(v: impl Into<Option<T>>, check: impl FnOnce(T) -> Result<()>) -> Result<()> {
    v.into().map_or(Ok(()), check)
}

/// Parameter checks.
pub(crate) mod param {
    use super::*;

    pub(crate) fn finite(
        f: &'static str,
        name: &'static str,
        v: impl Into<Option<f64>>,
    ) -> Result<()> {
        on::<f64>(v, |v| {
            require(v.is_finite(), || Error::parameter(f, name, v, "finite"))
        })
    }

    pub(crate) fn not_nan(
        f: &'static str,
        name: &'static str,
        v: impl Into<Option<f64>>,
    ) -> Result<()> {
        on::<f64>(v, |v| {
            require(!v.is_nan(), || Error::parameter(f, name, v, "not NaN"))
        })
    }

    pub(crate) fn positive(
        f: &'static str,
        name: &'static str,
        v: impl Into<Option<f64>>,
    ) -> Result<()> {
        on::<f64>(v, |v| {
            require(v > 0.0, || Error::parameter(f, name, v, "positive"))
        })
    }

    pub(crate) fn positive_finite(
        f: &'static str,
        name: &'static str,
        v: impl Into<Option<f64>>,
    ) -> Result<()> {
        on::<f64>(v, |v| {
            require(v > 0.0 && v.is_finite(), || {
                Error::parameter(f, name, v, "positive finite")
            })
        })
    }

    pub(crate) fn nonnegative(
        f: &'static str,
        name: &'static str,
        v: impl Into<Option<f64>>,
    ) -> Result<()> {
        on::<f64>(v, |v| {
            require(v >= 0.0, || Error::parameter(f, name, v, "nonnegative"))
        })
    }

    pub(crate) fn open_unit(
        f: &'static str,
        name: &'static str,
        v: impl Into<Option<f64>>,
    ) -> Result<()> {
        on::<f64>(v, |v| {
            require(v > 0.0 && v < 1.0, || {
                Error::parameter(f, name, v, "in (0, 1)")
            })
        })
    }

    pub(crate) fn nonnegative_finite(
        f: &'static str,
        name: &'static str,
        v: impl Into<Option<f64>>,
    ) -> Result<()> {
        on::<f64>(v, |v| {
            require(v >= 0.0 && v.is_finite(), || {
                Error::parameter(f, name, v, "nonnegative finite")
            })
        })
    }

    pub(crate) fn probability(
        f: &'static str,
        name: &'static str,
        v: impl Into<Option<f64>>,
    ) -> Result<()> {
        on::<f64>(v, |v| {
            require((0.0..=1.0).contains(&v), || {
                Error::parameter(f, name, v, "in [0, 1]")
            })
        })
    }
}

/// Observation checks.
pub(crate) mod obs {
    use super::*;

    pub(crate) fn not_nan(
        f: &'static str,
        name: &'static str,
        v: impl Into<Option<f64>>,
    ) -> Result<()> {
        on::<f64>(v, |v| {
            require(!v.is_nan(), || Error::observation(f, name, v, "not NaN"))
        })
    }

    pub(crate) fn finite(
        f: &'static str,
        name: &'static str,
        v: impl Into<Option<f64>>,
    ) -> Result<()> {
        on::<f64>(v, |v| {
            require(v.is_finite(), || Error::observation(f, name, v, "finite"))
        })
    }

    pub(crate) fn nonnegative(
        f: &'static str,
        name: &'static str,
        v: impl Into<Option<f64>>,
    ) -> Result<()> {
        on::<f64>(v, |v| {
            require(v >= 0.0, || Error::observation(f, name, v, "nonnegative"))
        })
    }

    pub(crate) fn positive(
        f: &'static str,
        name: &'static str,
        v: impl Into<Option<f64>>,
    ) -> Result<()> {
        on::<f64>(v, |v| {
            require(v > 0.0, || Error::observation(f, name, v, "positive"))
        })
    }

    pub(crate) fn positive_finite(
        f: &'static str,
        name: &'static str,
        v: impl Into<Option<f64>>,
    ) -> Result<()> {
        on::<f64>(v, |v| {
            require(v > 0.0 && v.is_finite(), || {
                Error::observation(f, name, v, "positive finite")
            })
        })
    }

    pub(crate) fn unit_interval(
        f: &'static str,
        name: &'static str,
        v: impl Into<Option<f64>>,
    ) -> Result<()> {
        on::<f64>(v, |v| {
            require((0.0..=1.0).contains(&v), || {
                Error::observation(f, name, v, "in [0, 1]")
            })
        })
    }

    pub(crate) fn count(
        f: &'static str,
        name: &'static str,
        n: impl Into<Option<i64>>,
    ) -> Result<()> {
        on::<i64>(n, |n| {
            require(n >= 0, || {
                Error::observation(f, name, n as f64, "nonnegative")
            })
        })
    }

    pub(crate) fn binary(
        f: &'static str,
        name: &'static str,
        n: impl Into<Option<i64>>,
    ) -> Result<()> {
        on::<i64>(n, |n| {
            require(n == 0 || n == 1, || {
                Error::observation(f, name, n as f64, "0 or 1")
            })
        })
    }
}
