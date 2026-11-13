//! Elementwise functions of one argument: Stan's OpenCL `exp`, `log`, `inv_logit`, and the rest.
//!
//! Each function is a row of one table: its Stan name, its value as an OpenCL C expression in `x`,
//! and its derivative as an expression in `x` and the value `y`. Each row becomes a program with
//! a value kernel and an adjoint kernel, built on first use.

use stanmath_rs_autodiff::functions as f;
use stanmath_rs_autodiff::{MultiFn, Real};
use stanmath_rs_core::Result;

use crate::device::{Arg, DeviceBuffer, launch};

macro_rules! unary_table {
    ($($variant:ident $name:literal $host:path, $value:literal, $deriv:literal;)*) => {
        /// An elementwise function of one argument with an OpenCL kernel.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum Unary {
            $(
                #[doc = concat!("Stan's `", $name, "`.")]
                $variant,
            )*
        }

        impl Unary {
            /// Every function, in table order.
            pub const ALL: &'static [Unary] = &[$(Unary::$variant),*];

            /// The Stan name.
            pub fn name(self) -> &'static str {
                match self {
                    $(Unary::$variant => $name,)*
                }
            }

            /// `(value, derivative)` as OpenCL C expressions.
            fn expressions(self) -> (&'static str, &'static str) {
                match self {
                    $(Unary::$variant => ($value, $deriv),)*
                }
            }

            /// The same function on the host, for scalars the device does not handle.
            fn host<S: Real>(self, x: S) -> S {
                match self {
                    $(Unary::$variant => $host(x),)*
                }
            }
        }
    };
}

unary_table! {
    Exp "exp" f::exp, "exp(x)", "y";
    Exp2 "exp2" f::exp2, "exp2(x)", "y * M_LN2";
    Expm1 "expm1" f::expm1, "expm1(x)", "y + 1.0";
    Log "log" f::log, "log(x)", "1.0 / x";
    Log2 "log2" f::log2, "log2(x)", "1.0 / (x * M_LN2)";
    Log10 "log10" f::log10, "log10(x)", "1.0 / (x * M_LN10)";
    Log1p "log1p" f::log1p, "log1p(x)", "1.0 / (1.0 + x)";
    Log1m "log1m" f::log1m, "log1p(-x)", "-1.0 / (1.0 - x)";
    Sqrt "sqrt" f::sqrt, "sqrt(x)", "0.5 / y";
    Cbrt "cbrt" f::cbrt, "cbrt(x)", "1.0 / (3.0 * y * y)";
    Square "square" f::square, "x * x", "2.0 * x";
    Inv "inv" f::inv, "1.0 / x", "-y * y";
    InvSqrt "inv_sqrt" f::inv_sqrt, "1.0 / sqrt(x)", "-0.5 * y / x";
    InvSquare "inv_square" f::inv_square, "1.0 / (x * x)", "-2.0 * y / x";
    Fabs "fabs" f::fabs, "fabs(x)", "x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0)";
    Sin "sin" f::sin, "sin(x)", "cos(x)";
    Cos "cos" f::cos, "cos(x)", "-sin(x)";
    Tan "tan" f::tan, "tan(x)", "1.0 + y * y";
    Asin "asin" f::asin, "asin(x)", "1.0 / sqrt(1.0 - x * x)";
    Acos "acos" f::acos, "acos(x)", "-1.0 / sqrt(1.0 - x * x)";
    Atan "atan" f::atan, "atan(x)", "1.0 / (1.0 + x * x)";
    Sinh "sinh" f::sinh, "sinh(x)", "cosh(x)";
    Cosh "cosh" f::cosh, "cosh(x)", "sinh(x)";
    Tanh "tanh" f::tanh, "tanh(x)", "1.0 - y * y";
    Asinh "asinh" f::asinh, "asinh(x)", "1.0 / sqrt(x * x + 1.0)";
    Acosh "acosh" f::acosh, "acosh(x)", "1.0 / sqrt(x * x - 1.0)";
    Atanh "atanh" f::atanh, "atanh(x)", "1.0 / (1.0 - x * x)";
    Erf "erf" f::erf, "erf(x)", "M_2_SQRTPI * exp(-x * x)";
    Erfc "erfc" f::erfc, "erfc(x)", "-M_2_SQRTPI * exp(-x * x)";
    Phi "Phi" f::Phi, "0.5 * erfc(-x * M_SQRT1_2)", "exp(-0.5 * x * x) * 0.5 * M_2_SQRTPI * M_SQRT1_2";
    PhiApprox "Phi_approx" f::Phi_approx,
        "1.0 / (1.0 + exp(-(0.07056 * x * x * x + 1.5976 * x)))",
        "y * (1.0 - y) * (3.0 * 0.07056 * x * x + 1.5976)";
    InvLogit "inv_logit" f::inv_logit, "1.0 / (1.0 + exp(-x))", "y * (1.0 - y)";
    Logit "logit" f::logit, "log(x / (1.0 - x))", "1.0 / (x - x * x)";
    LogInvLogit "log_inv_logit" f::log_inv_logit,
        "x < 0.0 ? x - log1p(exp(x)) : -log1p(exp(-x))", "1.0 / (1.0 + exp(x))";
    Log1mInvLogit "log1m_inv_logit" f::log1m_inv_logit,
        "x > 0.0 ? -x - log1p(exp(-x)) : -log1p(exp(x))", "-1.0 / (1.0 + exp(-x))";
    Log1pExp "log1p_exp" f::log1p_exp,
        "x > 0.0 ? x + log1p(exp(-x)) : log1p(exp(x))", "1.0 / (1.0 + exp(-x))";
    Log1mExp "log1m_exp" f::log1m_exp,
        "x > -M_LN2 ? log(-expm1(x)) : log1p(-exp(x))", "-1.0 / expm1(-x)";
    InvCloglog "inv_cloglog" f::inv_cloglog, "-expm1(-exp(x))", "exp(x - exp(x))";
    Floor "floor" f::floor, "floor(x)", "0.0";
    Ceil "ceil" f::ceil, "ceil(x)", "0.0";
    Round "round" f::round, "round(x)", "0.0";
    Trunc "trunc" f::trunc, "trunc(x)", "0.0";
}

fn source(op: Unary) -> String {
    let (value, deriv) = op.expressions();
    format!(
        "#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void value(__global const double* xs, __global double* ys) {{
  const size_t i = get_global_id(0);
  const double x = xs[i];
  ys[i] = {value};
}}
__kernel void adjoint(__global const double* xs, __global const double* ys,
                      __global const double* ya, __global double* xa) {{
  const size_t i = get_global_id(0);
  const double x = xs[i];
  const double y = ys[i];
  xa[i] = ya[i] * ({deriv});
}}
"
    )
}

fn device_value(op: Unary, x: &[f64]) -> Result<Vec<f64>> {
    let xs = DeviceBuffer::from_slice(x)?;
    let ys = DeviceBuffer::new(x.len())?;
    launch(
        op.name(),
        op.name(),
        "value",
        || source(op),
        &[Arg::Buffer(&xs), Arg::Buffer(&ys)],
        &[x.len()],
    )?;
    ys.to_vec()
}

fn device_adjoint(op: Unary, x: &[f64], y: &[f64], y_adj: &[f64]) -> Result<Vec<f64>> {
    let (xs, ys, ya) = (
        DeviceBuffer::from_slice(x)?,
        DeviceBuffer::from_slice(y)?,
        DeviceBuffer::from_slice(y_adj)?,
    );
    let xa = DeviceBuffer::new(x.len())?;
    launch(
        op.name(),
        op.name(),
        "adjoint",
        || source(op),
        &[
            Arg::Buffer(&xs),
            Arg::Buffer(&ys),
            Arg::Buffer(&ya),
            Arg::Buffer(&xa),
        ],
        &[x.len()],
    )?;
    xa.to_vec()
}

#[derive(Clone)]
struct Map(Unary);

impl MultiFn for Map {
    fn eval<S: Real>(&self, x: &[S]) -> Result<Vec<S>> {
        if S::ORDER > 0 {
            return Ok(x.iter().map(|&v| self.0.host(v)).collect());
        }
        let v: Vec<f64> = x.iter().map(Real::value).collect();
        Ok(device_value(self.0, &v)?
            .into_iter()
            .map(S::from_f64)
            .collect())
    }

    fn reverse(&self, x: &[f64], y: &[f64], y_adj: &[f64], needed: &[bool], x_adj: &mut [f64]) {
        // A failed kernel leaves NaN gradients.
        let g = device_adjoint(self.0, x, y, y_adj).unwrap_or_else(|_| vec![f64::NAN; x.len()]);
        for ((o, gi), &need) in x_adj.iter_mut().zip(g).zip(needed) {
            if need {
                *o += gi;
            }
        }
    }
}

/// `op` applied to every element of `x` on the device, in any scalar type. In reverse mode the
/// adjoint also runs on the device; forward-mode scalars are evaluated on the host.
///
/// # Errors
///
/// [`stanmath_rs_core::Error::Backend`] when the device cannot be opened or a kernel fails.
pub fn unary_on_device<T: Real>(op: Unary, x: &[T]) -> Result<Vec<T>> {
    if x.is_empty() {
        return Ok(Vec::new());
    }
    T::map_multi(x, &Map(op))
}

macro_rules! binary_table {
    ($($variant:ident $name:literal $host:path, $value:literal, $da:literal, $db:literal;)*) => {
        /// An elementwise function of two arguments of equal length with an OpenCL kernel.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum Binary {
            $(
                #[doc = concat!("Stan's `", $name, "`.")]
                $variant,
            )*
        }

        impl Binary {
            /// Every function, in table order.
            pub const ALL: &'static [Binary] = &[$(Binary::$variant),*];

            /// The Stan name.
            pub fn name(self) -> &'static str {
                match self {
                    $(Binary::$variant => $name,)*
                }
            }

            /// `(value, derivative in x, derivative in y)` as OpenCL C expressions in `x`, `y`,
            /// and the value `z`.
            fn expressions(self) -> (&'static str, &'static str, &'static str) {
                match self {
                    $(Binary::$variant => ($value, $da, $db),)*
                }
            }

            fn host<S: Real>(self, a: S, b: S) -> S {
                match self {
                    $(Binary::$variant => $host(a, b),)*
                }
            }
        }
    };
}

fn add<S: Real>(a: S, b: S) -> S {
    a + b
}

fn subtract<S: Real>(a: S, b: S) -> S {
    a - b
}

fn multiply<S: Real>(a: S, b: S) -> S {
    a * b
}

fn divide<S: Real>(a: S, b: S) -> S {
    a / b
}

binary_table! {
    Add "add" add, "x + y", "1.0", "1.0";
    Subtract "subtract" subtract, "x - y", "1.0", "-1.0";
    EltMultiply "elt_multiply" multiply, "x * y", "y", "x";
    EltDivide "elt_divide" divide, "x / y", "1.0 / y", "-z / y";
    Pow "pow" f::pow, "pow(x, y)", "y * pow(x, y - 1.0)", "z * log(x)";
    Hypot "hypot" f::hypot, "hypot(x, y)", "x / z", "y / z";
    LogDiffExp "log_diff_exp" f::log_diff_exp,
        "x + log(-expm1(y - x))", "-1.0 / expm1(y - x)", "-1.0 / expm1(x - y)";
}

fn binary_source(op: Binary) -> String {
    let (value, da, db) = op.expressions();
    format!(
        "#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void value(__global const double* xs, __global const double* ys, __global double* zs) {{
  const size_t i = get_global_id(0);
  const double x = xs[i];
  const double y = ys[i];
  zs[i] = {value};
}}
__kernel void adjoint(__global const double* xs, __global const double* ys,
                      __global const double* zs, __global const double* za,
                      __global double* xa, __global double* ya) {{
  const size_t i = get_global_id(0);
  const double x = xs[i];
  const double y = ys[i];
  const double z = zs[i];
  xa[i] = za[i] * ({da});
  ya[i] = za[i] * ({db});
}}
"
    )
}

#[derive(Clone)]
struct Map2(Binary);

impl Map2 {
    fn device_value(&self, x: &[f64]) -> Result<Vec<f64>> {
        let n = x.len() / 2;
        let (xs, ys) = (
            DeviceBuffer::from_slice(&x[..n])?,
            DeviceBuffer::from_slice(&x[n..])?,
        );
        let zs = DeviceBuffer::new(n)?;
        let op = self.0;
        launch(
            op.name(),
            op.name(),
            "value",
            || binary_source(op),
            &[Arg::Buffer(&xs), Arg::Buffer(&ys), Arg::Buffer(&zs)],
            &[n],
        )?;
        zs.to_vec()
    }

    fn device_adjoint(&self, x: &[f64], z: &[f64], z_adj: &[f64]) -> Result<Vec<f64>> {
        let n = z.len();
        let buffers = [&x[..n], &x[n..], z, z_adj]
            .into_iter()
            .map(DeviceBuffer::from_slice)
            .collect::<Result<Vec<_>>>()?;
        let (xa, ya) = (DeviceBuffer::new(n)?, DeviceBuffer::new(n)?);
        let op = self.0;
        launch(
            op.name(),
            op.name(),
            "adjoint",
            || binary_source(op),
            &[
                Arg::Buffer(&buffers[0]),
                Arg::Buffer(&buffers[1]),
                Arg::Buffer(&buffers[2]),
                Arg::Buffer(&buffers[3]),
                Arg::Buffer(&xa),
                Arg::Buffer(&ya),
            ],
            &[n],
        )?;
        let mut g = xa.to_vec()?;
        g.extend(ya.to_vec()?);
        Ok(g)
    }
}

impl MultiFn for Map2 {
    fn eval<S: Real>(&self, x: &[S]) -> Result<Vec<S>> {
        let n = x.len() / 2;
        if S::ORDER > 0 {
            return Ok((0..n).map(|i| self.0.host(x[i], x[n + i])).collect());
        }
        let v: Vec<f64> = x.iter().map(Real::value).collect();
        Ok(self
            .device_value(&v)?
            .into_iter()
            .map(S::from_f64)
            .collect())
    }

    fn reverse(&self, x: &[f64], z: &[f64], z_adj: &[f64], needed: &[bool], x_adj: &mut [f64]) {
        // A failed kernel leaves NaN gradients.
        let g = self
            .device_adjoint(x, z, z_adj)
            .unwrap_or_else(|_| vec![f64::NAN; x.len()]);
        for ((o, gi), &need) in x_adj.iter_mut().zip(g).zip(needed) {
            if need {
                *o += gi;
            }
        }
    }
}

/// `op(x[i], y[i])` for every `i` on the device, in any scalar type, with the adjoint on the
/// device in reverse mode and the host functions in forward mode.
///
/// # Errors
///
/// [`stanmath_rs_core::Error::DimensionMismatch`] unless `x` and `y` have the same length, and
/// [`stanmath_rs_core::Error::Backend`] when a kernel fails.
pub fn binary_on_device<T: Real>(op: Binary, x: &[T], y: &[T]) -> Result<Vec<T>> {
    if x.len() != y.len() {
        return Err(stanmath_rs_core::Error::dimension(
            op.name(),
            format!("x has {} elements but y has {}", x.len(), y.len()),
        ));
    }
    if x.is_empty() {
        return Ok(Vec::new());
    }
    let xy: Vec<T> = x.iter().chain(y).copied().collect();
    T::map_multi(&xy, &Map2(op))
}

#[cfg(test)]
mod tests {
    use super::*;
    use stanmath_rs_autodiff::{Dual, grad};

    /// Points inside the domain of `op`.
    fn points(op: Unary) -> Vec<f64> {
        let base = [0.05, 0.2, 0.37, 0.5, 0.71, 0.95];
        base.iter()
            .map(|&x| match op {
                Unary::Acosh => x + 1.5,
                Unary::Log1mExp => -x - 0.1,
                Unary::Floor | Unary::Ceil | Unary::Round | Unary::Trunc => 3.0 * x - 1.2,
                _ => x,
            })
            .collect()
    }

    #[test]
    fn every_function_matches_the_host() {
        for &op in Unary::ALL {
            let x = points(op);
            let got = unary_on_device(op, &x).unwrap();
            for (&xi, &g) in x.iter().zip(&got) {
                let want = op.host(xi);
                assert!(
                    (g - want).abs() <= 1e-13 * (1.0 + want.abs()),
                    "{}({xi}): {g} vs {want}",
                    op.name()
                );
            }
            let weights: Vec<f64> = (0..x.len()).map(|i| 1.0 + i as f64).collect();
            let g = grad(
                |v| {
                    let y = unary_on_device(op, v).unwrap();
                    y.iter().zip(&weights).map(|(&yi, &w)| yi * w).sum()
                },
                &x,
            );
            for (i, &xi) in x.iter().enumerate() {
                let want = op.host(Dual::variable(xi)).eps * weights[i];
                assert!(
                    (g[i] - want).abs() <= 1e-12 * (1.0 + want.abs()),
                    "d {}({xi}): {} vs {want}",
                    op.name(),
                    g[i]
                );
            }
        }
    }

    #[test]
    fn every_binary_function_matches_the_host() {
        let x = [0.3, 0.8, 1.7, 2.4];
        let y = [0.1, 0.5, 1.2, -0.6];
        for &op in Binary::ALL {
            // log_diff_exp needs x > y, pow a positive base; these points satisfy both.
            let got = binary_on_device(op, &x, &y).unwrap();
            for i in 0..x.len() {
                let want = op.host(x[i], y[i]);
                assert!(
                    (got[i] - want).abs() <= 1e-13 * (1.0 + want.abs()),
                    "{}({}, {}): {} vs {want}",
                    op.name(),
                    x[i],
                    y[i],
                    got[i]
                );
            }
            let xy: Vec<f64> = x.iter().chain(&y).copied().collect();
            let g = grad(
                |v| {
                    let z = binary_on_device(op, &v[..4], &v[4..]).unwrap();
                    z.iter()
                        .enumerate()
                        .map(|(i, &zi)| zi * (1.0 + i as f64))
                        .sum()
                },
                &xy,
            );
            for i in 0..4 {
                let w = 1.0 + i as f64;
                let da = op.host(Dual::variable(x[i]), Dual::constant(y[i])).eps * w;
                let db = op.host(Dual::constant(x[i]), Dual::variable(y[i])).eps * w;
                for (got, want) in [(g[i], da), (g[4 + i], db)] {
                    assert!(
                        (got - want).abs() <= 1e-12 * (1.0 + want.abs()),
                        "d {}: {got} vs {want}",
                        op.name()
                    );
                }
            }
        }
        assert!(binary_on_device(Binary::Add, &x, &y[..2]).is_err());
    }
}
