//! Argument validation following Stan Math's checks.
//!
//! Observations ("random variables" in Stan's messages) produce [`Error::InvalidObservation`];
//! everything else produces [`Error::InvalidParameter`].

use stanmath_rs_core::{Error, Result};

#[inline]
fn require(ok: bool, err: impl FnOnce() -> Error) -> Result<()> {
    if ok { Ok(()) } else { Err(err()) }
}

/// Parameter checks.
pub(crate) mod param {
    use super::*;

    pub(crate) fn finite(f: &'static str, name: &'static str, v: f64) -> Result<()> {
        require(v.is_finite(), || Error::parameter(f, name, v, "finite"))
    }

    pub(crate) fn not_nan(f: &'static str, name: &'static str, v: f64) -> Result<()> {
        require(!v.is_nan(), || Error::parameter(f, name, v, "not NaN"))
    }

    pub(crate) fn positive(f: &'static str, name: &'static str, v: f64) -> Result<()> {
        require(v > 0.0, || Error::parameter(f, name, v, "positive"))
    }

    pub(crate) fn positive_finite(f: &'static str, name: &'static str, v: f64) -> Result<()> {
        require(v > 0.0 && v.is_finite(), || {
            Error::parameter(f, name, v, "positive finite")
        })
    }

    pub(crate) fn nonnegative(f: &'static str, name: &'static str, v: f64) -> Result<()> {
        require(v >= 0.0, || Error::parameter(f, name, v, "nonnegative"))
    }

    pub(crate) fn probability(f: &'static str, name: &'static str, v: f64) -> Result<()> {
        require((0.0..=1.0).contains(&v), || {
            Error::parameter(f, name, v, "in [0, 1]")
        })
    }
}

/// Observation checks.
pub(crate) mod obs {
    use super::*;

    pub(crate) fn not_nan(f: &'static str, name: &'static str, v: f64) -> Result<()> {
        require(!v.is_nan(), || Error::observation(f, name, v, "not NaN"))
    }

    pub(crate) fn finite(f: &'static str, name: &'static str, v: f64) -> Result<()> {
        require(v.is_finite(), || Error::observation(f, name, v, "finite"))
    }

    pub(crate) fn nonnegative(f: &'static str, name: &'static str, v: f64) -> Result<()> {
        require(v >= 0.0, || Error::observation(f, name, v, "nonnegative"))
    }

    pub(crate) fn positive(f: &'static str, name: &'static str, v: f64) -> Result<()> {
        require(v > 0.0, || Error::observation(f, name, v, "positive"))
    }

    pub(crate) fn positive_finite(f: &'static str, name: &'static str, v: f64) -> Result<()> {
        require(v > 0.0 && v.is_finite(), || {
            Error::observation(f, name, v, "positive finite")
        })
    }

    pub(crate) fn unit_interval(f: &'static str, name: &'static str, v: f64) -> Result<()> {
        require((0.0..=1.0).contains(&v), || {
            Error::observation(f, name, v, "in [0, 1]")
        })
    }

    pub(crate) fn count(f: &'static str, name: &'static str, n: i64) -> Result<()> {
        require(n >= 0, || {
            Error::observation(f, name, n as f64, "nonnegative")
        })
    }

    pub(crate) fn binary(f: &'static str, name: &'static str, n: i64) -> Result<()> {
        require(n == 0 || n == 1, || {
            Error::observation(f, name, n as f64, "0 or 1")
        })
    }
}
