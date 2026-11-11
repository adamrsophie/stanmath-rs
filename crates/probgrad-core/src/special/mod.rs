//! Numerically stable `f64` special functions.
//!
//! Every function here works on plain `f64`. The differentiable versions, with derivative rules,
//! are in `probgrad-autodiff`. Functions return `NaN` outside their domain instead of panicking,
//! which matches IEEE 754 conventions; argument validation with informative errors happens in the
//! distribution layer.

mod gamma;
mod log_domain;
mod normal;

pub use gamma::*;
pub use log_domain::*;
pub use normal::*;

/// `ln(2 * pi)`.
pub const LN_2PI: f64 = 1.837_877_066_409_345_5;
/// `0.5 * ln(2 * pi)`.
pub const HALF_LN_2PI: f64 = 0.918_938_533_204_672_8;
/// `ln(pi)`.
pub const LN_PI: f64 = 1.144_729_885_849_400_2;
/// `sqrt(2)`.
pub const SQRT_2: f64 = core::f64::consts::SQRT_2;
/// `1 / sqrt(2 * pi)`.
pub const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
/// `sqrt(pi)`.
pub const SQRT_PI: f64 = 1.772_453_850_905_516;
