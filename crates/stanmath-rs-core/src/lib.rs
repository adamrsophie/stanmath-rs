//! Foundation types and numerically stable `f64` numerics shared by the stanmath-rs crates.
//!
//! This crate contains no automatic differentiation. It provides:
//!
//! * [`Error`] and [`Result`], the error types used across the workspace;
//! * [`Matrix`], a small dense column-major matrix container that accepts any element type,
//!   including reverse-mode variables that borrow a tape;
//! * [`special`], `f64` special functions and stable log-domain helpers.
//!
//! The differentiable versions of these functions live in `stanmath-rs-autodiff`.

pub mod error;
pub mod matrix;
pub mod special;

pub use error::{Error, Result};
pub use matrix::Matrix;
