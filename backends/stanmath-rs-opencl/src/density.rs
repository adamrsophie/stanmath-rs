//! Log densities on the device: Stan's OpenCL `normal_lpdf`, `gamma_lpdf`, and the rest.
//!
//! Each density is a row of one table: its arguments with the check each must pass, OpenCL C
//! for the log density of one element, and its partial with respect to each argument. The device
//! computes every element and its partials; the host sums them. As in Stan's OpenCL support the
//! derivatives are reverse mode only: `f64` and `Var` arguments run on the device, and
//! forward-mode scalars are refused.

use stanmath_rs_autodiff::{MultiFn, Real};
use stanmath_rs_core::{Error, Result};

use crate::device::{Arg, DeviceBuffer, launch};
use crate::special::PRELUDE;

/// What an argument of a density must satisfy.
#[derive(Clone, Copy)]
enum Check {
    NotNan,
    Finite,
    Positive,
    PositiveFinite,
    Nonnegative,
    UnitInterval,
}

impl Check {
    fn holds(self, v: f64) -> bool {
        match self {
            Check::NotNan => !v.is_nan(),
            Check::Finite => v.is_finite(),
            Check::Positive => v > 0.0,
            Check::PositiveFinite => v > 0.0 && v.is_finite(),
            Check::Nonnegative => v >= 0.0,
            Check::UnitInterval => (0.0..=1.0).contains(&v),
        }
    }

    fn requirement(self) -> &'static str {
        match self {
            Check::NotNan => "not NaN",
            Check::Finite => "finite",
            Check::Positive => "positive",
            Check::PositiveFinite => "positive finite",
            Check::Nonnegative => "nonnegative",
            Check::UnitInterval => "in [0, 1]",
        }
    }
}

/// What an integer argument of a probability mass function must satisfy.
#[derive(Clone, Copy)]
enum IntCheck {
    Nonnegative,
    Binary,
}

impl IntCheck {
    fn holds(self, v: i64) -> bool {
        match self {
            IntCheck::Nonnegative => v >= 0,
            IntCheck::Binary => v == 0 || v == 1,
        }
    }

    fn requirement(self) -> &'static str {
        match self {
            IntCheck::Nonnegative => "nonnegative",
            IntCheck::Binary => "0 or 1",
        }
    }
}

macro_rules! density_table {
    ($(
        $variant:ident $name:literal ($($iarg:literal $icheck:ident),*) [$($arg:literal $check:ident),+]
        $prelude:literal, $lp:literal, [$($d:literal),+];
    )*) => {
        /// A log density with an OpenCL kernel.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum Density {
            $(
                #[doc = concat!("Stan's `", $name, "`.")]
                $variant,
            )*
        }

        impl Density {
            /// Every density, in table order.
            pub const ALL: &'static [Density] = &[$(Density::$variant),*];

            /// The Stan name.
            pub fn name(self) -> &'static str {
                match self {
                    $(Density::$variant => $name,)*
                }
            }

            /// The names of the integer arguments, outcome first, in Stan's order.
            pub fn int_args(self) -> &'static [&'static str] {
                match self {
                    $(Density::$variant => &[$($iarg),*],)*
                }
            }

            fn int_checks(self) -> &'static [IntCheck] {
                match self {
                    $(Density::$variant => &[$(IntCheck::$icheck),*],)*
                }
            }

            /// The names of the real arguments, outcome first for a density, in Stan's order.
            pub fn args(self) -> &'static [&'static str] {
                match self {
                    $(Density::$variant => &[$($arg),+],)*
                }
            }

            fn checks(self) -> &'static [Check] {
                match self {
                    $(Density::$variant => &[$(Check::$check),+],)*
                }
            }

            /// `(prelude, log density, partials)` as OpenCL C.
            fn code(self) -> (&'static str, &'static str, &'static [&'static str]) {
                match self {
                    $(Density::$variant => ($prelude, $lp, &[$($d),+]),)*
                }
            }
        }
    };
}

density_table! {
    StdNormal "std_normal_lpdf" () ["y" NotNan]
        "", "-0.5 * y * y - LOG_SQRT_2PI", ["-y"];
    Normal "normal_lpdf" () ["y" NotNan, "mu" Finite, "sigma" PositiveFinite]
        "const double z = (y - mu) / sigma;",
        "-0.5 * z * z - log(sigma) - LOG_SQRT_2PI",
        ["-z / sigma", "z / sigma", "(z * z - 1.0) / sigma"];
    Cauchy "cauchy_lpdf" () ["y" NotNan, "mu" Finite, "sigma" PositiveFinite]
        "const double z = (y - mu) / sigma; const double q = 1.0 + z * z;",
        "-log(M_PI) - log(sigma) - log1p(z * z)",
        ["-2.0 * z / (sigma * q)", "2.0 * z / (sigma * q)", "(z * z - 1.0) / (sigma * q)"];
    Logistic "logistic_lpdf" () ["y" Finite, "mu" Finite, "sigma" PositiveFinite]
        "const double z = (y - mu) / sigma; const double t = tanh(0.5 * z);",
        "-fabs(z) - 2.0 * log1p(exp(-fabs(z))) - log(sigma)",
        ["-t / sigma", "t / sigma", "(z * t - 1.0) / sigma"];
    Gumbel "gumbel_lpdf" () ["y" NotNan, "mu" Finite, "beta" PositiveFinite]
        "const double z = (y - mu) / beta; const double e = exp(-z);",
        "-log(beta) - z - e",
        ["(e - 1.0) / beta", "(1.0 - e) / beta", "(z - z * e - 1.0) / beta"];
    Lognormal "lognormal_lpdf" () ["y" Nonnegative, "mu" Finite, "sigma" PositiveFinite]
        "const double ly = log(y); const double z = (ly - mu) / sigma;",
        "-0.5 * z * z - log(sigma) - ly - LOG_SQRT_2PI",
        ["-(z / sigma + 1.0) / y", "z / sigma", "(z * z - 1.0) / sigma"];
    Exponential "exponential_lpdf" () ["y" Nonnegative, "beta" PositiveFinite]
        "", "log(beta) - beta * y", ["-beta", "1.0 / beta - y"];
    DoubleExponential "double_exponential_lpdf" () ["y" Finite, "mu" Finite, "sigma" PositiveFinite]
        "const double d = y - mu; const double s = d > 0.0 ? 1.0 : (d < 0.0 ? -1.0 : 0.0);",
        "-M_LN2 - log(sigma) - fabs(d) / sigma",
        ["-s / sigma", "s / sigma", "(fabs(d) / sigma - 1.0) / sigma"];
    Rayleigh "rayleigh_lpdf" () ["y" Nonnegative, "sigma" PositiveFinite]
        "const double s2 = sigma * sigma;",
        "log(y) - 2.0 * log(sigma) - 0.5 * y * y / s2",
        ["1.0 / y - y / s2", "(y * y / s2 - 2.0) / sigma"];
    Weibull "weibull_lpdf" () ["y" Nonnegative, "alpha" PositiveFinite, "sigma" PositiveFinite]
        "const double l = log(y / sigma); const double p = pow(y / sigma, alpha);",
        "log(alpha) - log(sigma) + (alpha - 1.0) * l - p",
        ["(alpha - 1.0 - alpha * p) / y", "1.0 / alpha + l - p * l", "alpha * (p - 1.0) / sigma"];
    Frechet "frechet_lpdf" () ["y" Positive, "alpha" PositiveFinite, "sigma" PositiveFinite]
        "const double l = log(y / sigma); const double p = pow(y / sigma, -alpha);",
        "log(alpha) - log(sigma) - (1.0 + alpha) * l - p",
        ["(alpha * p - 1.0 - alpha) / y", "1.0 / alpha - l + p * l", "alpha * (1.0 - p) / sigma"];
    Pareto "pareto_lpdf" () ["y" NotNan, "y_min" PositiveFinite, "alpha" PositiveFinite]
        "const int inside = y >= y_min;",
        "inside ? log(alpha) + alpha * log(y_min) - (alpha + 1.0) * log(y) : -INFINITY",
        ["inside ? -(alpha + 1.0) / y : 0.0", "inside ? alpha / y_min : 0.0",
         "inside ? 1.0 / alpha + log(y_min) - log(y) : 0.0"];
    ParetoType2 "pareto_type_2_lpdf" () ["y" NotNan, "mu" Finite, "lambda" PositiveFinite, "alpha" PositiveFinite]
        "const double t = (y - mu) / lambda; const double g = (alpha + 1.0) / (lambda * (1.0 + t));",
        "log(alpha) - log(lambda) - (alpha + 1.0) * log1p(t)",
        ["-g", "g", "t * g - 1.0 / lambda", "1.0 / alpha - log1p(t)"];
    SkewNormal "skew_normal_lpdf" () ["y" NotNan, "xi" Finite, "omega" PositiveFinite, "alpha" Finite]
        "const double z = (y - xi) / omega; const double u = alpha * z * M_SQRT1_2;
         const double r = M_2_SQRTPI * M_SQRT1_2 * exp(-u * u) / erfc(-u);",
        "-LOG_SQRT_2PI - log(omega) - 0.5 * z * z + log(erfc(-u))",
        ["(alpha * r - z) / omega", "(z - alpha * r) / omega",
         "(z * z - alpha * r * z - 1.0) / omega", "r * z"];
    Uniform "uniform_lpdf" () ["y" NotNan, "alpha" Finite, "beta" Finite]
        "const int inside = y >= alpha && y <= beta;",
        "inside ? -log(beta - alpha) : -INFINITY",
        ["0.0", "inside ? 1.0 / (beta - alpha) : 0.0", "inside ? -1.0 / (beta - alpha) : 0.0"];
    Gamma "gamma_lpdf" () ["y" Positive, "alpha" PositiveFinite, "beta" PositiveFinite]
        "",
        "alpha * log(beta) - lgamma(alpha) + (alpha - 1.0) * log(y) - beta * y",
        ["(alpha - 1.0) / y - beta", "log(beta) - digamma(alpha) + log(y)", "alpha / beta - y"];
    InvGamma "inv_gamma_lpdf" () ["y" Positive, "alpha" PositiveFinite, "beta" PositiveFinite]
        "",
        "alpha * log(beta) - lgamma(alpha) - (alpha + 1.0) * log(y) - beta / y",
        ["(beta / y - alpha - 1.0) / y", "log(beta) - digamma(alpha) - log(y)",
         "alpha / beta - 1.0 / y"];
    Beta "beta_lpdf" () ["y" UnitInterval, "alpha" PositiveFinite, "beta" PositiveFinite]
        "const double ab = digamma(alpha + beta);",
        "lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta)
         + (alpha - 1.0) * log(y) + (beta - 1.0) * log1p(-y)",
        ["(alpha - 1.0) / y - (beta - 1.0) / (1.0 - y)", "ab - digamma(alpha) + log(y)",
         "ab - digamma(beta) + log1p(-y)"];
    BetaProportion "beta_proportion_lpdf" () ["y" UnitInterval, "mu" Positive, "kappa" PositiveFinite]
        "const double a = mu * kappa; const double b = (1.0 - mu) * kappa;
         const double da = digamma(a); const double db = digamma(b);",
        "lgamma(kappa) - lgamma(a) - lgamma(b) + (a - 1.0) * log(y) + (b - 1.0) * log1p(-y)",
        ["(a - 1.0) / y - (b - 1.0) / (1.0 - y)", "kappa * (db - da + log(y) - log1p(-y))",
         "digamma(kappa) - mu * da - (1.0 - mu) * db + mu * log(y) + (1.0 - mu) * log1p(-y)"];
    ChiSquare "chi_square_lpdf" () ["y" Nonnegative, "nu" PositiveFinite]
        "",
        "(0.5 * nu - 1.0) * log(y) - 0.5 * y - 0.5 * nu * M_LN2 - lgamma(0.5 * nu)",
        ["(0.5 * nu - 1.0) / y - 0.5", "0.5 * (log(y) - M_LN2 - digamma(0.5 * nu))"];
    InvChiSquare "inv_chi_square_lpdf" () ["y" Positive, "nu" PositiveFinite]
        "",
        "-0.5 * nu * M_LN2 - lgamma(0.5 * nu) - (0.5 * nu + 1.0) * log(y) - 0.5 / y",
        ["(0.5 / y - 0.5 * nu - 1.0) / y", "-0.5 * (M_LN2 + digamma(0.5 * nu) + log(y))"];
    ScaledInvChiSquare "scaled_inv_chi_square_lpdf" () ["y" Positive, "nu" PositiveFinite, "s" PositiveFinite]
        "const double h = 0.5 * nu;",
        "h * log(h) - lgamma(h) + nu * log(s) - (h + 1.0) * log(y) - h * s * s / y",
        ["(h * s * s / y - h - 1.0) / y",
         "0.5 * (log(h) + 1.0 - digamma(h) - log(y) - s * s / y) + log(s)",
         "nu / s - nu * s / y"];
    StudentT "student_t_lpdf" () ["y" NotNan, "nu" PositiveFinite, "mu" Finite, "sigma" PositiveFinite]
        "const double z = (y - mu) / sigma; const double q = nu + z * z;",
        "lgamma(0.5 * (nu + 1.0)) - lgamma(0.5 * nu) - 0.5 * log(nu * M_PI) - log(sigma)
         - 0.5 * (nu + 1.0) * log1p(z * z / nu)",
        ["-(nu + 1.0) * z / (sigma * q)",
         "0.5 * (digamma(0.5 * (nu + 1.0)) - digamma(0.5 * nu) - 1.0 / nu - log1p(z * z / nu)
          + (nu + 1.0) * z * z / (nu * q))",
         "(nu + 1.0) * z / (sigma * q)", "((nu + 1.0) * z * z / q - 1.0) / sigma"];
    ExpModNormal "exp_mod_normal_lpdf" ()
        ["y" NotNan, "mu" Finite, "sigma" PositiveFinite, "lambda" PositiveFinite]
        "const double u = (y - mu) / sigma; const double w = (lambda * sigma - u) * M_SQRT1_2;
         const double r = -M_2_SQRTPI * exp(-w * w) / erfc(w);",
        "log(lambda) + lambda * (mu - y) + 0.5 * lambda * lambda * sigma * sigma + log(erfc(w)) - M_LN2",
        ["-lambda - r * M_SQRT1_2 / sigma", "lambda + r * M_SQRT1_2 / sigma",
         "lambda * lambda * sigma + r * M_SQRT1_2 * (lambda + u / sigma)",
         "1.0 / lambda + mu - y + lambda * sigma * sigma + r * M_SQRT1_2 * sigma"];
    Bernoulli "bernoulli_lpmf" ("n" Binary) ["theta" UnitInterval]
        "", "n == 1.0 ? log(theta) : log1p(-theta)",
        ["n == 1.0 ? 1.0 / theta : -1.0 / (1.0 - theta)"];
    BernoulliLogit "bernoulli_logit_lpmf" ("n" Binary) ["alpha" NotNan]
        "const double s = n == 1.0 ? alpha : -alpha;",
        "s < 0.0 ? s - log1p(exp(s)) : -log1p(exp(-s))", ["n - 1.0 / (1.0 + exp(-alpha))"];
    Binomial "binomial_lpmf" ("n" Nonnegative, "N" Nonnegative) ["theta" UnitInterval]
        "",
        "lgamma(N + 1.0) - lgamma(n + 1.0) - lgamma(N - n + 1.0)
         + (n > 0.0 ? n * log(theta) : 0.0) + (N > n ? (N - n) * log1p(-theta) : 0.0)",
        ["(n > 0.0 ? n / theta : 0.0) - (N > n ? (N - n) / (1.0 - theta) : 0.0)"];
    Poisson "poisson_lpmf" ("n" Nonnegative) ["lambda" Nonnegative]
        "", "(n > 0.0 ? n * log(lambda) : 0.0) - lambda - lgamma(n + 1.0)",
        ["(n > 0.0 ? n / lambda : 0.0) - 1.0"];
    PoissonLog "poisson_log_lpmf" ("n" Nonnegative) ["alpha" NotNan]
        "", "n * alpha - exp(alpha) - lgamma(n + 1.0)", ["n - exp(alpha)"];
    NegBinomial "neg_binomial_lpmf" ("n" Nonnegative) ["alpha" PositiveFinite, "beta" PositiveFinite]
        "const double lp1 = log1p(beta);",
        "lgamma(n + alpha) - lgamma(n + 1.0) - lgamma(alpha) + alpha * (log(beta) - lp1) - n * lp1",
        ["digamma(n + alpha) - digamma(alpha) + log(beta) - lp1",
         "alpha / beta - (alpha + n) / (1.0 + beta)"];
    NegBinomial2 "neg_binomial_2_lpmf" ("n" Nonnegative) ["mu" PositiveFinite, "phi" PositiveFinite]
        "const double s = mu + phi; const double ls = log(s);",
        "lgamma(n + phi) - lgamma(n + 1.0) - lgamma(phi) + (n > 0.0 ? n * (log(mu) - ls) : 0.0)
         + phi * (log(phi) - ls)",
        ["n / mu - (n + phi) / s",
         "digamma(n + phi) - digamma(phi) + log(phi) - ls + 1.0 - (n + phi) / s"];
    NegBinomial2Log "neg_binomial_2_log_lpmf" ("n" Nonnegative) ["eta" Finite, "phi" PositiveFinite]
        "const double mu = exp(eta); const double s = mu + phi; const double ls = log(s);",
        "lgamma(n + phi) - lgamma(n + 1.0) - lgamma(phi) + n * (eta - ls) + phi * (log(phi) - ls)",
        ["n - (n + phi) * mu / s",
         "digamma(n + phi) - digamma(phi) + log(phi) - ls + 1.0 - (n + phi) / s"];
}

/// Definitions shared by every density program.
const COMMON: &str = "#define LOG_SQRT_2PI 0.91893853320467274178\n";

fn source(op: Density) -> String {
    let (prelude, lp, partials) = op.code();
    let (ints, args) = (op.int_args(), op.args());
    let mut params: Vec<String> = (0..ints.len())
        .map(|k| format!("__global const double* q{k}, const ulong m{k}"))
        .collect();
    params
        .extend((0..args.len()).map(|k| format!("__global const double* p{k}, const ulong n{k}")));
    params.push("__global double* lp".into());
    params.extend((0..args.len()).map(|k| format!("__global double* d{k}")));
    let loads: String = ints
        .iter()
        .enumerate()
        .map(|(k, a)| format!("  const double {a} = q{k}[m{k} == 1 ? 0 : i];\n"))
        .chain(
            args.iter()
                .enumerate()
                .map(|(k, a)| format!("  const double {a} = p{k}[n{k} == 1 ? 0 : i];\n")),
        )
        .collect();
    let stores: String = partials
        .iter()
        .enumerate()
        .map(|(k, d)| format!("  d{k}[i] = {d};\n"))
        .collect();
    format!(
        "{PRELUDE}{COMMON}__kernel void eval({}) {{\n  const size_t i = get_global_id(0);\n{loads}  {prelude}\n  lp[i] = {lp};\n{stores}}}\n",
        params.join(", ")
    )
}

#[derive(Clone)]
struct Lpdf {
    op: Density,
    /// The integer arguments, as doubles; they carry no gradient.
    ints: Vec<Vec<f64>>,
    /// Length of each real argument: one, or the broadcast length `n`.
    lens: Vec<usize>,
    n: usize,
}

impl Lpdf {
    fn offsets(&self) -> Vec<usize> {
        let mut out = vec![0];
        for l in &self.lens {
            out.push(out.last().unwrap_or(&0) + l);
        }
        out
    }

    /// The log density of each element and the partials of each, per argument.
    fn device(&self, x: &[f64]) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
        let offsets = self.offsets();
        let inputs = (0..self.lens.len())
            .map(|k| DeviceBuffer::from_slice(&x[offsets[k]..offsets[k + 1]]))
            .collect::<Result<Vec<_>>>()?;
        let lp = DeviceBuffer::new(self.n)?;
        let partials = (0..self.lens.len())
            .map(|_| DeviceBuffer::new(self.n))
            .collect::<Result<Vec<_>>>()?;
        let int_inputs = self
            .ints
            .iter()
            .map(|v| DeviceBuffer::from_slice(v))
            .collect::<Result<Vec<_>>>()?;
        let mut args = Vec::new();
        for (b, v) in int_inputs.iter().zip(&self.ints) {
            args.push(Arg::Buffer(b));
            args.push(Arg::Size(v.len()));
        }
        for (b, &len) in inputs.iter().zip(&self.lens) {
            args.push(Arg::Buffer(b));
            args.push(Arg::Size(len));
        }
        args.push(Arg::Buffer(&lp));
        args.extend(partials.iter().map(Arg::Buffer));
        let op = self.op;
        launch(
            op.name(),
            op.name(),
            "eval",
            || source(op),
            &args,
            &[self.n],
        )?;
        Ok((
            lp.to_vec()?,
            partials
                .iter()
                .map(DeviceBuffer::to_vec)
                .collect::<Result<_>>()?,
        ))
    }
}

impl MultiFn for Lpdf {
    fn eval<S: Real>(&self, x: &[S]) -> Result<Vec<S>> {
        if S::ORDER > 0 {
            return Err(Error::Backend {
                function: self.op.name(),
                message: "OpenCL densities support reverse mode only, as in Stan; use \
                          stanmath-rs-distributions for forward mode"
                    .into(),
            });
        }
        let v: Vec<f64> = x.iter().map(Real::value).collect();
        let (lp, _) = self.device(&v)?;
        Ok(vec![S::from_f64(lp.iter().sum())])
    }

    fn reverse(&self, x: &[f64], _y: &[f64], y_adj: &[f64], needed: &[bool], x_adj: &mut [f64]) {
        let offsets = self.offsets();
        let partials = match self.device(x) {
            Ok((_, p)) => p,
            // A failed kernel leaves NaN gradients.
            Err(_) => vec![vec![f64::NAN; self.n]; self.lens.len()],
        };
        for (k, p) in partials.iter().enumerate() {
            let start = offsets[k];
            if self.lens[k] == 1 {
                if needed[start] {
                    x_adj[start] += y_adj[0] * p.iter().sum::<f64>();
                }
            } else {
                for (j, pj) in p.iter().enumerate() {
                    if needed[start + j] {
                        x_adj[start + j] += y_adj[0] * pj;
                    }
                }
            }
        }
    }
}

/// The log density `op` summed over the elements of its arguments, computed on the device.
/// `args` holds one slice per argument in Stan's order (outcome first); a slice of length one
/// is broadcast, as a scalar argument is in Stan. Works with `f64` and reverse-mode `Var`
/// arguments; forward-mode scalars are refused, as in Stan's OpenCL support. For a probability
/// mass function use [`lpmf_on_device`].
///
/// # Errors
///
/// As [`lpmf_on_device`].
pub fn lpdf_on_device<T: Real>(op: Density, args: &[&[T]]) -> Result<T> {
    lpmf_on_device(op, &[], args)
}

/// The log probability `op` summed over the elements of its arguments, computed on the device:
/// `ints` holds the integer arguments (the outcome first, then any such as binomial trials) and
/// `args` the real ones, each in Stan's order. A slice of length one is broadcast. Works with
/// `f64` and reverse-mode `Var` arguments; forward-mode scalars are refused, as in Stan's OpenCL
/// support.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] for the wrong number of arguments or lengths other than one and
/// the broadcast length; [`Error::InvalidParameter`] when an argument fails its check, such as a
/// scale that is not positive; and [`Error::Backend`] for forward-mode scalars or a kernel failure.
pub fn lpmf_on_device<T: Real>(op: Density, ints: &[&[i64]], args: &[&[T]]) -> Result<T> {
    let function = op.name();
    if args.len() != op.args().len() || ints.len() != op.int_args().len() {
        return Err(Error::dimension(
            function,
            format!(
                "{} integer and {} real arguments, expected {} and {}",
                ints.len(),
                args.len(),
                op.int_args().len(),
                op.args().len()
            ),
        ));
    }
    for ((a, &name), &check) in ints.iter().zip(op.int_args()).zip(op.int_checks()) {
        if let Some(&v) = a.iter().find(|&&v| !check.holds(v)) {
            return Err(Error::parameter(
                function,
                name,
                v as f64,
                check.requirement(),
            ));
        }
    }
    for ((a, &name), &check) in args.iter().zip(op.args()).zip(op.checks()) {
        if let Some(v) = a.iter().map(Real::value).find(|&v| !check.holds(v)) {
            return Err(Error::parameter(function, name, v, check.requirement()));
        }
    }
    if op == Density::Binomial {
        let at = |a: &[i64], i: usize| a[if a.len() == 1 { 0 } else { i }];
        let len = ints[0].len().max(ints[1].len());
        let (a, b) = (ints[0].len(), ints[1].len());
        if a > 0 && b > 0 && (a == b || a == 1 || b == 1) {
            if let Some(i) = (0..len).find(|&i| at(ints[0], i) > at(ints[1], i)) {
                let v = at(ints[0], i) as f64;
                return Err(Error::parameter(function, "n", v, "at most N"));
            }
        }
    }
    let lens: Vec<usize> = ints
        .iter()
        .map(|a| a.len())
        .chain(args.iter().map(|a| a.len()))
        .collect();
    // An empty argument makes the sum empty, as an empty vector does in Stan.
    if lens.contains(&0) {
        return Ok(T::zero());
    }
    let n = lens.iter().copied().max().unwrap_or(0);
    if let Some(a) = lens.iter().position(|&l| l != 1 && l != n) {
        let name = op
            .int_args()
            .iter()
            .chain(op.args())
            .nth(a)
            .copied()
            .unwrap_or("");
        return Err(Error::dimension(
            function,
            format!("{name} has {} elements; expected 1 or {n}", lens[a]),
        ));
    }
    let f = Lpdf {
        op,
        ints: ints
            .iter()
            .map(|a| a.iter().map(|&v| v as f64).collect())
            .collect(),
        lens: args.iter().map(|a| a.len()).collect(),
        n,
    };
    let x: Vec<T> = args.iter().flat_map(|a| a.iter().copied()).collect();
    Ok(T::map_multi(&x, &f)?[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use stanmath_rs_autodiff::{Dual, Var};
    use stanmath_rs_distributions as d;

    /// The host density, for comparison.
    fn host<'t>(op: Density, a: &[&[Var<'t>]]) -> Var<'t> {
        use Density::*;
        match op {
            StdNormal => d::std_normal_lpdf(a[0]),
            Normal => d::normal_lpdf(a[0], a[1], a[2]),
            Cauchy => d::cauchy_lpdf(a[0], a[1], a[2]),
            Logistic => d::logistic_lpdf(a[0], a[1], a[2]),
            Gumbel => d::gumbel_lpdf(a[0], a[1], a[2]),
            Lognormal => d::lognormal_lpdf(a[0], a[1], a[2]),
            Exponential => d::exponential_lpdf(a[0], a[1]),
            DoubleExponential => d::double_exponential_lpdf(a[0], a[1], a[2]),
            Rayleigh => d::rayleigh_lpdf(a[0], a[1]),
            Weibull => d::weibull_lpdf(a[0], a[1], a[2]),
            Frechet => d::frechet_lpdf(a[0], a[1], a[2]),
            Pareto => d::pareto_lpdf(a[0], a[1], a[2]),
            ParetoType2 => d::pareto_type_2_lpdf(a[0], a[1], a[2], a[3]),
            SkewNormal => d::skew_normal_lpdf(a[0], a[1], a[2], a[3]),
            Uniform => d::uniform_lpdf(a[0], a[1], a[2]),
            Gamma => d::gamma_lpdf(a[0], a[1], a[2]),
            InvGamma => d::inv_gamma_lpdf(a[0], a[1], a[2]),
            Beta => d::beta_lpdf(a[0], a[1], a[2]),
            BetaProportion => d::beta_proportion_lpdf(a[0], a[1], a[2]),
            ChiSquare => d::chi_square_lpdf(a[0], a[1]),
            InvChiSquare => d::inv_chi_square_lpdf(a[0], a[1]),
            ScaledInvChiSquare => d::scaled_inv_chi_square_lpdf(a[0], a[1], a[2]),
            StudentT => d::student_t_lpdf(a[0], a[1], a[2], a[3]),
            ExpModNormal => d::exp_mod_normal_lpdf(a[0], a[1], a[2], a[3]),
            _ => unreachable!("{} is a mass function", op.name()),
        }
        .unwrap()
    }

    /// Valid values: outcomes in `(0.1, 0.9)` and parameters in `(0.6, 2.2)`, with the uniform's
    /// and Pareto's bounds placed around the outcomes.
    fn point(op: Density) -> (Vec<f64>, Vec<usize>) {
        let ys = [0.15, 0.4, 0.55, 0.85];
        let mut x = ys.to_vec();
        let mut lens = vec![4];
        for (k, name) in op.args().iter().enumerate().skip(1) {
            let v = match (op, *name) {
                (Density::Uniform, "alpha") => vec![0.05],
                (Density::Uniform, "beta") => vec![1.3],
                (Density::Pareto, "y_min") => vec![0.1],
                (Density::ParetoType2, "mu") => vec![0.05],
                (Density::BetaProportion, "mu") => vec![0.3, 0.45, 0.6, 0.7],
                // Alternate between a broadcast scalar and a full vector.
                _ if k % 2 == 1 => vec![0.6 + 0.4 * k as f64],
                _ => (0..4)
                    .map(|i| 0.6 + 0.3 * k as f64 + 0.25 * i as f64)
                    .collect(),
            };
            lens.push(v.len());
            x.extend(v);
        }
        (x, lens)
    }

    fn split<T: Copy>(x: &[T], lens: &[usize]) -> Vec<Vec<T>> {
        let mut start = 0;
        lens.iter()
            .map(|&l| {
                let v = x[start..start + l].to_vec();
                start += l;
                v
            })
            .collect()
    }

    #[test]
    fn every_density_matches_the_host() {
        for &op in Density::ALL.iter().filter(|op| op.int_args().is_empty()) {
            let (x, lens) = point(op);
            let run = |on_device: bool| {
                stanmath_rs_autodiff::value_and_grad(
                    |v| {
                        let parts = split(v, &lens);
                        let refs: Vec<&[Var<'_>]> = parts.iter().map(Vec::as_slice).collect();
                        if on_device {
                            lpdf_on_device(op, &refs).unwrap()
                        } else {
                            // Stan does not broadcast a vector of length one, so repeat it.
                            let full: Vec<Vec<Var<'_>>> = parts
                                .iter()
                                .map(|p| p.iter().copied().cycle().take(4).collect())
                                .collect();
                            let refs: Vec<&[Var<'_>]> = full.iter().map(Vec::as_slice).collect();
                            host(op, &refs)
                        }
                    },
                    &x,
                )
            };
            let (dev, want) = (run(true), run(false));
            assert!(
                (dev.value - want.value).abs() <= 1e-12 * (1.0 + want.value.abs()),
                "{}: {} vs {}",
                op.name(),
                dev.value,
                want.value
            );
            for (k, (a, b)) in dev.gradient.iter().zip(&want.gradient).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-10 * (1.0 + b.abs()),
                    "{} gradient {k}: {a} vs {b}",
                    op.name()
                );
            }
        }
    }

    /// The host mass function, for comparison.
    fn host_lpmf<'t>(op: Density, n: &[i64], trials: &[i64], a: &[&[Var<'t>]]) -> Var<'t> {
        use Density::*;
        match op {
            Bernoulli => d::bernoulli_lpmf(n, a[0]),
            BernoulliLogit => d::bernoulli_logit_lpmf(n, a[0]),
            Binomial => d::binomial_lpmf(n, trials, a[0]),
            Poisson => d::poisson_lpmf(n, a[0]),
            PoissonLog => d::poisson_log_lpmf(n, a[0]),
            NegBinomial => d::neg_binomial_lpmf(n, a[0], a[1]),
            NegBinomial2 => d::neg_binomial_2_lpmf(n, a[0], a[1]),
            NegBinomial2Log => d::neg_binomial_2_log_lpmf(n, a[0], a[1]),
            _ => unreachable!("{} is a density", op.name()),
        }
        .unwrap()
    }

    #[test]
    fn every_mass_function_matches_the_host() {
        for &op in Density::ALL.iter().filter(|op| !op.int_args().is_empty()) {
            let binary = op.name().starts_with("bernoulli");
            let n: Vec<i64> = if binary {
                vec![1, 0, 0, 1]
            } else {
                vec![0, 3, 1, 6]
            };
            let trials = [2, 5, 4, 9];
            let probability = matches!(op, Density::Bernoulli | Density::Binomial);
            // The first real argument is a full vector, the second a broadcast scalar.
            let first: Vec<f64> = if probability {
                vec![0.2, 0.45, 0.6, 0.8]
            } else {
                vec![0.3, 1.1, 0.7, 2.4]
            };
            let x: Vec<f64> = first
                .iter()
                .copied()
                .chain([1.7])
                .take(4 + op.args().len() - 1)
                .collect();
            let run = |on_device: bool| {
                stanmath_rs_autodiff::value_and_grad(
                    |v| {
                        let (a, b) = v.split_at(4);
                        if on_device {
                            let ints: Vec<&[i64]> = if op == Density::Binomial {
                                vec![&n, &trials]
                            } else {
                                vec![&n]
                            };
                            let reals: Vec<&[Var<'_>]> =
                                if b.is_empty() { vec![a] } else { vec![a, b] };
                            lpmf_on_device(op, &ints, &reals).unwrap()
                        } else {
                            let full: Vec<Var<'_>> =
                                b.iter().copied().cycle().take(b.len() * 4).collect();
                            let reals: Vec<&[Var<'_>]> = if b.is_empty() {
                                vec![a]
                            } else {
                                vec![a, &full]
                            };
                            host_lpmf(op, &n, &trials, &reals)
                        }
                    },
                    &x,
                )
            };
            let (dev, want) = (run(true), run(false));
            assert!(
                (dev.value - want.value).abs() <= 1e-12 * (1.0 + want.value.abs()),
                "{}: {} vs {}",
                op.name(),
                dev.value,
                want.value
            );
            for (k, (a, b)) in dev.gradient.iter().zip(&want.gradient).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-10 * (1.0 + b.abs()),
                    "{} gradient {k}: {a} vs {b}",
                    op.name()
                );
            }
        }
        let theta = [0.5];
        assert!(lpmf_on_device(Density::Binomial, &[&[3], &[2]], &[&theta[..]]).is_err());
        assert!(lpmf_on_device(Density::Bernoulli, &[&[2]], &[&theta[..]]).is_err());
        assert!(lpdf_on_device(Density::Poisson, &[&theta[..]]).is_err());
    }

    #[test]
    fn arguments_are_checked_and_forward_mode_refused() {
        let (y, zero, one, bad) = ([0.5, 1.0], [0.0], [1.0], [-1.0]);
        let three = [0.0, 1.0, 2.0];
        let lpdf = |a: &[&[f64]]| lpdf_on_device(Density::Normal, a);
        assert!(lpdf(&[&y, &zero, &bad]).is_err());
        assert!(lpdf(&[&y, &three, &one]).is_err());
        assert!(lpdf(&[&y, &zero]).is_err());
        assert_eq!(lpdf(&[&[], &zero, &one]).unwrap(), 0.0);
        let (yd, c0, c1) = (
            [Dual::variable(0.5)],
            [Dual::constant(0.0)],
            [Dual::constant(1.0)],
        );
        assert!(lpdf_on_device(Density::Normal, &[&yd, &c0, &c1]).is_err());
    }
}
