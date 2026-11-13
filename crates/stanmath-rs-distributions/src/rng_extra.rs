//! Random number generation for the distributions in `continuous_extra`.
//!
//! Each `*_rng` function validates its parameters with the same rules as the corresponding log
//! density and draws one value. Draws are built from standard uniform, normal, exponential,
//! chi-square, and beta variates.

use core::f64::consts::PI;

use rand::Rng;
use rand_distr::{Beta, ChiSquared, Distribution, Exp1, Open01, StandardNormal};
use stanmath_rs_core::{Error, Result};

use crate::check::param;

fn uniform<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    Open01.sample(rng)
}

fn std_normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    StandardNormal.sample(rng)
}

fn std_exponential<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    Exp1.sample(rng)
}

fn chi_square<R: Rng + ?Sized>(f: &'static str, nu: f64, rng: &mut R) -> Result<f64> {
    let d =
        ChiSquared::new(nu).map_err(|_| Error::parameter(f, "nu", nu, "valid for the sampler"))?;
    Ok(d.sample(rng))
}

/// Draw from `std_normal()`.
pub fn std_normal_rng<R: Rng + ?Sized>(rng: &mut R) -> Result<f64> {
    Ok(std_normal(rng))
}

/// Draw from `skew_normal(xi, omega, alpha)` by Azzalini's representation: with
/// `delta = alpha / sqrt(1 + alpha^2)` and independent standard normals `u0`, `v`, the value
/// `delta |u0| + sqrt(1 - delta^2) v` is standard skew normal.
pub fn skew_normal_rng<R: Rng + ?Sized>(
    xi: f64,
    omega: f64,
    alpha: f64,
    rng: &mut R,
) -> Result<f64> {
    const F: &str = "skew_normal_rng";
    param::finite(F, "xi", xi)?;
    param::positive_finite(F, "omega", omega)?;
    param::finite(F, "alpha", alpha)?;
    let delta = alpha / (1.0 + alpha * alpha).sqrt();
    let u0 = std_normal(rng);
    let v = std_normal(rng);
    let z = delta * u0.abs() + (1.0 - delta * delta).sqrt() * v;
    Ok(xi + omega * z)
}

/// Draw from `double_exponential(mu, sigma)` by inversion.
pub fn double_exponential_rng<R: Rng + ?Sized>(mu: f64, sigma: f64, rng: &mut R) -> Result<f64> {
    const F: &str = "double_exponential_rng";
    param::finite(F, "mu", mu)?;
    param::positive_finite(F, "sigma", sigma)?;
    let e = std_exponential(rng);
    let sign = if uniform(rng) < 0.5 { -1.0 } else { 1.0 };
    Ok(mu + sigma * sign * e)
}

/// Draw from `von_mises(mu, kappa)` by the Best and Fisher (1979) rejection algorithm.
///
/// The value lies in `[(mu mod 2 pi) - pi, (mu mod 2 pi) + pi]`, with `mod` taking the sign of
/// `mu`. For `kappa > 1e6` a normal approximation with standard deviation `1 / sqrt(kappa)` is used.
pub fn von_mises_rng<R: Rng + ?Sized>(mu: f64, kappa: f64, rng: &mut R) -> Result<f64> {
    const F: &str = "von_mises_rng";
    param::finite(F, "mu", mu)?;
    param::nonnegative_finite(F, "kappa", kappa)?;
    let center = mu % (2.0 * PI);
    if kappa < 1e-8 {
        return Ok(center + PI * (2.0 * uniform(rng) - 1.0));
    }
    if kappa > 1e6 {
        let theta = (std_normal(rng) / kappa.sqrt()).clamp(-PI, PI);
        return Ok(center + theta);
    }
    let tau = 1.0 + (1.0 + 4.0 * kappa * kappa).sqrt();
    let rho = (tau - (2.0 * tau).sqrt()) / (2.0 * kappa);
    let r = (1.0 + rho * rho) / (2.0 * rho);
    let f = loop {
        let z = (PI * uniform(rng)).cos();
        let f = (1.0 + r * z) / (r + z);
        let c = kappa * (r - f);
        let u2 = uniform(rng);
        if c * (2.0 - c) - u2 > 0.0 || (c / u2).ln() + 1.0 - c >= 0.0 {
            break f;
        }
    };
    let theta = f.clamp(-1.0, 1.0).acos();
    Ok(if uniform(rng) > 0.5 {
        center + theta
    } else {
        center - theta
    })
}

/// Draw from `rayleigh(sigma)` by inversion.
pub fn rayleigh_rng<R: Rng + ?Sized>(sigma: f64, rng: &mut R) -> Result<f64> {
    param::positive_finite("rayleigh_rng", "sigma", sigma)?;
    Ok(sigma * (2.0 * std_exponential(rng)).sqrt())
}

/// Draw from `pareto(y_min, alpha)`: `y_min exp(E / alpha)` with `E` standard exponential.
pub fn pareto_rng<R: Rng + ?Sized>(y_min: f64, alpha: f64, rng: &mut R) -> Result<f64> {
    const F: &str = "pareto_rng";
    param::positive_finite(F, "y_min", y_min)?;
    param::positive_finite(F, "alpha", alpha)?;
    Ok(y_min * (std_exponential(rng) / alpha).exp())
}

/// Draw from `pareto_type_2(mu, lambda, alpha)`: `mu + lambda (exp(E / alpha) - 1)`.
pub fn pareto_type_2_rng<R: Rng + ?Sized>(
    mu: f64,
    lambda: f64,
    alpha: f64,
    rng: &mut R,
) -> Result<f64> {
    const F: &str = "pareto_type_2_rng";
    param::finite(F, "mu", mu)?;
    param::positive_finite(F, "lambda", lambda)?;
    param::positive_finite(F, "alpha", alpha)?;
    Ok(mu + lambda * (std_exponential(rng) / alpha).exp_m1())
}

/// Draw from `frechet(alpha, sigma)`: `sigma E^(-1 / alpha)` with `E` standard exponential.
pub fn frechet_rng<R: Rng + ?Sized>(alpha: f64, sigma: f64, rng: &mut R) -> Result<f64> {
    const F: &str = "frechet_rng";
    param::positive_finite(F, "alpha", alpha)?;
    param::positive_finite(F, "sigma", sigma)?;
    Ok(sigma * std_exponential(rng).powf(-1.0 / alpha))
}

/// Draw from `gumbel(mu, beta)`: `mu - beta ln E` with `E` standard exponential.
pub fn gumbel_rng<R: Rng + ?Sized>(mu: f64, beta: f64, rng: &mut R) -> Result<f64> {
    const F: &str = "gumbel_rng";
    param::finite(F, "mu", mu)?;
    param::positive_finite(F, "beta", beta)?;
    Ok(mu - beta * std_exponential(rng).ln())
}

/// Draw from `chi_square(nu)`.
pub fn chi_square_rng<R: Rng + ?Sized>(nu: f64, rng: &mut R) -> Result<f64> {
    const F: &str = "chi_square_rng";
    param::positive_finite(F, "nu", nu)?;
    chi_square(F, nu, rng)
}

/// Draw from `inv_chi_square(nu)`: the reciprocal of a chi-square draw.
pub fn inv_chi_square_rng<R: Rng + ?Sized>(nu: f64, rng: &mut R) -> Result<f64> {
    const F: &str = "inv_chi_square_rng";
    param::positive_finite(F, "nu", nu)?;
    Ok(1.0 / chi_square(F, nu, rng)?)
}

/// Draw from `scaled_inv_chi_square(nu, s)`: `nu s^2 / X` with `X` chi-square on `nu` degrees of
/// freedom.
pub fn scaled_inv_chi_square_rng<R: Rng + ?Sized>(nu: f64, s: f64, rng: &mut R) -> Result<f64> {
    const F: &str = "scaled_inv_chi_square_rng";
    param::positive_finite(F, "nu", nu)?;
    param::positive_finite(F, "s", s)?;
    Ok(nu * s * s / chi_square(F, nu, rng)?)
}

/// Draw from `exp_mod_normal(mu, sigma, lambda)`: a normal plus an exponential with rate `lambda`.
pub fn exp_mod_normal_rng<R: Rng + ?Sized>(
    mu: f64,
    sigma: f64,
    lambda: f64,
    rng: &mut R,
) -> Result<f64> {
    const F: &str = "exp_mod_normal_rng";
    param::finite(F, "mu", mu)?;
    param::positive_finite(F, "sigma", sigma)?;
    param::positive_finite(F, "lambda", lambda)?;
    Ok(mu + sigma * std_normal(rng) + std_exponential(rng) / lambda)
}

/// Draw from `skew_double_exponential(mu, sigma, tau)` by inversion.
pub fn skew_double_exponential_rng<R: Rng + ?Sized>(
    mu: f64,
    sigma: f64,
    tau: f64,
    rng: &mut R,
) -> Result<f64> {
    const F: &str = "skew_double_exponential_rng";
    param::finite(F, "mu", mu)?;
    param::positive_finite(F, "sigma", sigma)?;
    param::open_unit(F, "tau", tau)?;
    let u = uniform(rng);
    Ok(if u < tau {
        mu + sigma / (2.0 * (1.0 - tau)) * (u / tau).ln()
    } else {
        mu - sigma / (2.0 * tau) * ((1.0 - u) / (1.0 - tau)).ln()
    })
}

/// Draw from `loglogistic(alpha, beta)` (scale, shape) by inversion.
pub fn loglogistic_rng<R: Rng + ?Sized>(alpha: f64, beta: f64, rng: &mut R) -> Result<f64> {
    const F: &str = "loglogistic_rng";
    param::positive_finite(F, "alpha", alpha)?;
    param::positive_finite(F, "beta", beta)?;
    let u = uniform(rng);
    Ok(alpha * (u / (1.0 - u)).powf(1.0 / beta))
}

/// Draw from `beta_proportion(mu, kappa)`, that is `beta(mu kappa, (1 - mu) kappa)`.
pub fn beta_proportion_rng<R: Rng + ?Sized>(mu: f64, kappa: f64, rng: &mut R) -> Result<f64> {
    const F: &str = "beta_proportion_rng";
    param::open_unit(F, "mu", mu)?;
    param::positive_finite(F, "kappa", kappa)?;
    let d = Beta::new(mu * kappa, (1.0 - mu) * kappa)
        .map_err(|_| Error::parameter(F, "kappa", kappa, "valid for the sampler"))?;
    Ok(d.sample(rng))
}
