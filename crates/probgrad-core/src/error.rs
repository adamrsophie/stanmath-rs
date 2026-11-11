//! Error type shared by all probgrad crates.

use core::fmt;

/// Result alias with [`Error`] as the default error type.
pub type Result<T, E = Error> = core::result::Result<T, E>;

/// Errors reported by probgrad functions.
///
/// Invalid parameters and invalid observations are separate variants so that callers can tell a
/// misspecified model (bad parameter) from bad data (observation outside the support).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// A parameter is outside the domain the function accepts.
    InvalidParameter {
        /// Function that rejected the argument.
        function: &'static str,
        /// Argument name, for example `"sigma"`.
        name: &'static str,
        /// Offending value.
        value: f64,
        /// Human readable requirement, for example `"positive finite"`.
        requirement: &'static str,
    },
    /// An observation (data value or outcome) is outside the support of the distribution.
    InvalidObservation {
        /// Function that rejected the argument.
        function: &'static str,
        /// Argument name, for example `"y"`.
        name: &'static str,
        /// Offending value.
        value: f64,
        /// Human readable requirement.
        requirement: &'static str,
    },
    /// Arguments have inconsistent sizes.
    DimensionMismatch {
        /// Function that rejected the arguments.
        function: &'static str,
        /// Description of the mismatch.
        details: String,
    },
    /// A matrix that must be symmetric positive definite is not.
    NotPositiveDefinite {
        /// Function that rejected the matrix.
        function: &'static str,
    },
    /// A matrix that must be symmetric is not.
    NotSymmetric {
        /// Function that rejected the matrix.
        function: &'static str,
        /// Row of the first asymmetric pair.
        row: usize,
        /// Column of the first asymmetric pair.
        col: usize,
    },
    /// An iterative algorithm did not reach its tolerance.
    NoConvergence {
        /// Function whose algorithm did not converge.
        function: &'static str,
        /// Iterations performed before giving up.
        iterations: usize,
    },
}

impl Error {
    /// Builds an [`Error::InvalidParameter`].
    pub fn parameter(
        function: &'static str,
        name: &'static str,
        value: f64,
        requirement: &'static str,
    ) -> Self {
        Error::InvalidParameter {
            function,
            name,
            value,
            requirement,
        }
    }

    /// Builds an [`Error::InvalidObservation`].
    pub fn observation(
        function: &'static str,
        name: &'static str,
        value: f64,
        requirement: &'static str,
    ) -> Self {
        Error::InvalidObservation {
            function,
            name,
            value,
            requirement,
        }
    }

    /// Builds an [`Error::DimensionMismatch`].
    pub fn dimension(function: &'static str, details: impl Into<String>) -> Self {
        Error::DimensionMismatch {
            function,
            details: details.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidParameter {
                function,
                name,
                value,
                requirement,
            } => write!(
                f,
                "{function}: parameter {name} is {value}, but must be {requirement}"
            ),
            Error::InvalidObservation {
                function,
                name,
                value,
                requirement,
            } => write!(
                f,
                "{function}: observation {name} is {value}, but must be {requirement}"
            ),
            Error::DimensionMismatch { function, details } => {
                write!(f, "{function}: inconsistent dimensions: {details}")
            }
            Error::NotPositiveDefinite { function } => {
                write!(f, "{function}: matrix is not symmetric positive definite")
            }
            Error::NotSymmetric { function, row, col } => {
                write!(f, "{function}: matrix is not symmetric at ({row}, {col})")
            }
            Error::NoConvergence {
                function,
                iterations,
            } => write!(
                f,
                "{function}: no convergence after {iterations} iterations"
            ),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_names_the_argument_kind() {
        let p = Error::parameter("normal_lpdf", "sigma", -1.0, "positive finite");
        let o = Error::observation("poisson_lpmf", "n", -2.0, "nonnegative");
        assert_eq!(
            p.to_string(),
            "normal_lpdf: parameter sigma is -1, but must be positive finite"
        );
        assert_eq!(
            o.to_string(),
            "poisson_lpmf: observation n is -2, but must be nonnegative"
        );
        assert_ne!(p, o);
    }
}
