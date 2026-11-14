//! Generalized linear models on the device: Stan's OpenCL `normal_id_glm_lpdf` and the rest.
//!
//! The linear predictor `x beta + alpha` is the device matrix product, and the elementwise
//! density runs on the device as [`lpmf_on_device`]; the categorical and ordered logistic
//! models have their own row-wise device kernels.

use stanmath_rs_autodiff::{MultiFn, Real};
use stanmath_rs_core::{Error, Matrix, Result};

use crate::device::{Arg, DeviceBuffer, launch};
use crate::special::PRELUDE;
use crate::{Density, lpmf_on_device, multiply_on_device};

/// `x beta + alpha` for an `n` by `k` matrix `x`, with `alpha` of length one or `n`.
fn predictor<T: Real>(
    function: &'static str,
    x: &Matrix<T>,
    alpha: &[T],
    beta: &Matrix<T>,
) -> Result<Matrix<T>> {
    let n = x.nrows();
    if alpha.len() != 1 && alpha.len() != beta.ncols() && !(beta.ncols() == 1 && alpha.len() == n) {
        return Err(Error::dimension(
            function,
            format!("alpha has {} elements", alpha.len()),
        ));
    }
    let eta = multiply_on_device(x, beta)?;
    let m = beta.ncols();
    let data = eta
        .as_slice()
        .iter()
        .enumerate()
        .map(|(idx, &e)| {
            // One column: alpha per row. Several (categorical): alpha per column.
            let a = match (alpha.len(), m) {
                (1, _) => alpha[0],
                (_, 1) => alpha[idx % n],
                _ => alpha[idx / n],
            };
            e + a
        })
        .collect();
    Ok(Matrix::from_column_major(n, m, data))
}

fn column<T: Real>(beta: &[T]) -> Matrix<T> {
    Matrix::from_column_major(beta.len(), 1, beta.to_vec())
}

/// Stan's `normal_id_glm_lpdf(y | x, alpha, beta, sigma)` on the device; `alpha` and `sigma`
/// have length one (broadcast) or the number of rows of `x`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] for inconsistent sizes and the errors of [`lpmf_on_device`].
pub fn normal_id_glm_lpdf_on_device<T: Real>(
    y: &[T],
    x: &Matrix<T>,
    alpha: &[T],
    beta: &[T],
    sigma: &[T],
) -> Result<T> {
    let eta = predictor("normal_id_glm_lpdf", x, alpha, &column(beta))?;
    lpmf_on_device(Density::Normal, &[], &[y, eta.as_slice(), sigma])
}

/// Stan's `bernoulli_logit_glm_lpmf(y | x, alpha, beta)` on the device.
///
/// # Errors
///
/// As [`normal_id_glm_lpdf_on_device`].
pub fn bernoulli_logit_glm_lpmf_on_device<T: Real>(
    y: &[i64],
    x: &Matrix<T>,
    alpha: &[T],
    beta: &[T],
) -> Result<T> {
    let eta = predictor("bernoulli_logit_glm_lpmf", x, alpha, &column(beta))?;
    lpmf_on_device(Density::BernoulliLogit, &[y], &[eta.as_slice()])
}

/// Stan's `poisson_log_glm_lpmf(y | x, alpha, beta)` on the device.
///
/// # Errors
///
/// As [`normal_id_glm_lpdf_on_device`].
pub fn poisson_log_glm_lpmf_on_device<T: Real>(
    y: &[i64],
    x: &Matrix<T>,
    alpha: &[T],
    beta: &[T],
) -> Result<T> {
    let eta = predictor("poisson_log_glm_lpmf", x, alpha, &column(beta))?;
    lpmf_on_device(Density::PoissonLog, &[y], &[eta.as_slice()])
}

/// Stan's `neg_binomial_2_log_glm_lpmf(y | x, alpha, beta, phi)` on the device.
///
/// # Errors
///
/// As [`normal_id_glm_lpdf_on_device`].
pub fn neg_binomial_2_log_glm_lpmf_on_device<T: Real>(
    y: &[i64],
    x: &Matrix<T>,
    alpha: &[T],
    beta: &[T],
    phi: &[T],
) -> Result<T> {
    let eta = predictor("neg_binomial_2_log_glm_lpmf", x, alpha, &column(beta))?;
    lpmf_on_device(Density::NegBinomial2Log, &[y], &[eta.as_slice(), phi])
}

/// Stan's `categorical_logit_glm_lpmf(y | x, alpha, beta)` with the `n` by `K` predictor
/// and the density computed on the device; `alpha` has length `K` and `beta` is `k` by `K`.
///
/// # Errors
///
/// As [`normal_id_glm_lpdf_on_device`], and [`Error::InvalidParameter`] for an outcome
/// outside `1..=K`.
pub fn categorical_logit_glm_lpmf_on_device<T: Real>(
    y: &[i64],
    x: &Matrix<T>,
    alpha: &[T],
    beta: &Matrix<T>,
) -> Result<T> {
    const F: &str = "categorical_logit_glm_lpmf";
    if alpha.len() != beta.ncols() {
        return Err(Error::dimension(
            F,
            format!(
                "alpha has {} elements, beta {} columns",
                alpha.len(),
                beta.ncols()
            ),
        ));
    }
    let eta = predictor(F, x, alpha, beta)?;
    let (n, k) = eta.shape();
    if y.len() != n {
        return Err(Error::dimension(
            F,
            format!("{} outcomes for {n} rows", y.len()),
        ));
    }
    if let Some(&bad) = y.iter().find(|&&v| v < 1 || v > k as i64) {
        return Err(Error::parameter(F, "y", bad as f64, "in 1..=K"));
    }
    if n == 0 || k == 0 {
        return Ok(T::zero());
    }
    let f = RowLogProb {
        kind: Kind::Categorical,
        y: y.to_vec(),
        n,
        k,
    };
    Ok(T::map_multi(eta.as_slice(), &f)?[0])
}

/// Stan's `ordered_logistic_glm_lpmf(y | x, beta, cuts)` with the predictor and the density
/// computed on the device.
///
/// # Errors
///
/// As [`normal_id_glm_lpdf_on_device`], and [`Error::InvalidParameter`] for an outcome
/// outside `1..=K` or cutpoints that are not finite and strictly increasing.
pub fn ordered_logistic_glm_lpmf_on_device<T: Real>(
    y: &[i64],
    x: &Matrix<T>,
    beta: &[T],
    cuts: &[T],
) -> Result<T> {
    const F: &str = "ordered_logistic_glm_lpmf";
    let eta = predictor(F, x, &[T::zero()], &column(beta))?;
    let (n, k) = (eta.nrows(), cuts.len() + 1);
    if y.len() != n {
        return Err(Error::dimension(
            F,
            format!("{} outcomes for {n} rows", y.len()),
        ));
    }
    if let Some(&bad) = y.iter().find(|&&v| v < 1 || v > k as i64) {
        return Err(Error::parameter(F, "y", bad as f64, "in 1..=K"));
    }
    let c: Vec<f64> = cuts.iter().map(Real::value).collect();
    if let Some(&bad) = c.iter().find(|v| !v.is_finite()) {
        return Err(Error::parameter(F, "cuts", bad, "finite"));
    }
    if c.windows(2).any(|w| w[0] >= w[1]) {
        return Err(Error::parameter(F, "cuts", c[0], "strictly increasing"));
    }
    if n == 0 {
        return Ok(T::zero());
    }
    let f = RowLogProb {
        kind: Kind::Ordered,
        y: y.to_vec(),
        n,
        k,
    };
    let inputs: Vec<T> = eta.as_slice().iter().chain(cuts).copied().collect();
    Ok(T::map_multi(&inputs, &f)?[0])
}

#[derive(Clone, Copy)]
enum Kind {
    /// Inputs: the n by K predictor.
    Categorical,
    /// Inputs: the n predictors, then the K - 1 cutpoints.
    Ordered,
}

/// The summed log probability of row-wise outcomes, with every row on its own work item.
#[derive(Clone)]
struct RowLogProb {
    kind: Kind,
    y: Vec<i64>,
    n: usize,
    k: usize,
}

const ROWS: &str = "__kernel void categorical(__global const double* eta, __global const double* y,
                          __global double* lp, __global double* g, const ulong n,
                          const ulong k) {
  const size_t i = get_global_id(0);
  const size_t c = (size_t) y[i] - 1;
  double m = eta[i];
  for (size_t j = 1; j < k; j++) m = fmax(m, eta[i + j * n]);
  double s = 0.0;
  for (size_t j = 0; j < k; j++) s += exp(eta[i + j * n] - m);
  lp[i] = eta[i + c * n] - m - log(s);
  for (size_t j = 0; j < k; j++) g[i + j * n] = (j == c ? 1.0 : 0.0) - exp(eta[i + j * n] - m) / s;
}
// Row i: lp, the predictor's gradient, and the gradients of the cutpoints below and above.
__kernel void ordered(__global const double* eta, __global const double* cuts,
                      __global const double* y, __global double* lp, __global double* ge,
                      __global double* glo, __global double* ghi, const ulong k) {
  const size_t i = get_global_id(0);
  const long c = (long) y[i];
  const double e = eta[i];
  if (c == 1) {
    const double a = e - cuts[0];
    lp[i] = a > 0.0 ? -a - log1p(exp(-a)) : -log1p(exp(a));
    ge[i] = -1.0 / (1.0 + exp(-a));
    glo[i] = 0.0;
    ghi[i] = -ge[i];
  } else if (c == (long) k) {
    const double a = e - cuts[k - 2];
    lp[i] = a < 0.0 ? a - log1p(exp(a)) : -log1p(exp(-a));
    ge[i] = 1.0 / (1.0 + exp(a));
    glo[i] = -ge[i];
    ghi[i] = 0.0;
  } else {
    const double a = e - cuts[c - 2], b = e - cuts[c - 1];
    lp[i] = a - log1p(exp(a)) - log1p(exp(b)) + log(-expm1(b - a));
    const double da = 1.0 / (1.0 + exp(a)) + 1.0 / expm1(a - b);
    const double db = -1.0 / (1.0 + exp(-b)) - 1.0 / expm1(a - b);
    ge[i] = da + db;
    glo[i] = -da;
    ghi[i] = -db;
  }
}
";

impl RowLogProb {
    /// The log probability of each row and the gradient with respect to every input.
    fn device(&self, x: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
        let n = self.n;
        let y: Vec<f64> = self.y.iter().map(|&v| v as f64).collect();
        let yb = DeviceBuffer::from_slice(&y)?;
        let lp = DeviceBuffer::new(n)?;
        let source = || format!("{PRELUDE}{ROWS}");
        match self.kind {
            Kind::Categorical => {
                let (eta, g) = (DeviceBuffer::from_slice(x)?, DeviceBuffer::new(x.len())?);
                launch(
                    "categorical_logit_glm_lpmf",
                    "glm_rows",
                    "categorical",
                    source,
                    &[
                        Arg::Buffer(&eta),
                        Arg::Buffer(&yb),
                        Arg::Buffer(&lp),
                        Arg::Buffer(&g),
                        Arg::Size(n),
                        Arg::Size(self.k),
                    ],
                    &[n],
                )?;
                Ok((lp.to_vec()?, g.to_vec()?))
            }
            Kind::Ordered => {
                let (eta, cuts) = (
                    DeviceBuffer::from_slice(&x[..n])?,
                    DeviceBuffer::from_slice(&x[n..])?,
                );
                let (ge, glo, ghi) = (
                    DeviceBuffer::new(n)?,
                    DeviceBuffer::new(n)?,
                    DeviceBuffer::new(n)?,
                );
                launch(
                    "ordered_logistic_glm_lpmf",
                    "glm_rows",
                    "ordered",
                    source,
                    &[
                        Arg::Buffer(&eta),
                        Arg::Buffer(&cuts),
                        Arg::Buffer(&yb),
                        Arg::Buffer(&lp),
                        Arg::Buffer(&ge),
                        Arg::Buffer(&glo),
                        Arg::Buffer(&ghi),
                        Arg::Size(self.k),
                    ],
                    &[n],
                )?;
                let mut g = ge.to_vec()?;
                g.resize(x.len(), 0.0);
                let (lo, hi) = (glo.to_vec()?, ghi.to_vec()?);
                for (i, &c) in self.y.iter().enumerate() {
                    let c = c as usize;
                    if c >= 2 {
                        g[n + c - 2] += lo[i];
                    }
                    if c < self.k {
                        g[n + c - 1] += hi[i];
                    }
                }
                Ok((lp.to_vec()?, g))
            }
        }
    }
}

impl MultiFn for RowLogProb {
    fn eval<S: Real>(&self, x: &[S]) -> Result<Vec<S>> {
        if S::ORDER > 0 {
            return Err(Error::Backend {
                function: "glm",
                message: "OpenCL GLMs support reverse mode only, as in Stan".into(),
            });
        }
        let v: Vec<f64> = x.iter().map(Real::value).collect();
        Ok(vec![S::from_f64(self.device(&v)?.0.iter().sum())])
    }

    fn reverse(&self, x: &[f64], _y: &[f64], y_adj: &[f64], needed: &[bool], x_adj: &mut [f64]) {
        // A failed kernel leaves NaN gradients.
        let g = self
            .device(x)
            .map(|(_, g)| g)
            .unwrap_or_else(|_| vec![f64::NAN; x.len()]);
        for ((o, gi), &need) in x_adj.iter_mut().zip(g).zip(needed) {
            if need {
                *o += y_adj[0] * gi;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stanmath_rs_autodiff::{Var, value_and_grad};
    use stanmath_rs_distributions as d;

    const N: usize = 5;
    const K: usize = 3;

    fn x_values() -> Vec<f64> {
        (0..N * K)
            .map(|i| ((i * 7 % 11) as f64 - 5.0) / 6.0)
            .collect()
    }

    /// Compares the device and host forms on `params` = x, then the model's own parameters.
    fn compare(name: &str, params: &[f64], f: impl for<'t> Fn(&[Var<'t>], bool) -> Var<'t>) {
        let dev = value_and_grad(|v| f(v, true), params);
        let host = value_and_grad(|v| f(v, false), params);
        assert!(
            (dev.value - host.value).abs() <= 1e-12 * (1.0 + host.value.abs()),
            "{name}: {} vs {}",
            dev.value,
            host.value
        );
        for (k, (a, b)) in dev.gradient.iter().zip(&host.gradient).enumerate() {
            assert!(
                (a - b).abs() <= 1e-10 * (1.0 + b.abs()),
                "{name} gradient {k}: {a} vs {b}"
            );
        }
    }

    fn split<'a, 't>(v: &'a [Var<'t>]) -> (Matrix<Var<'t>>, &'a [Var<'t>]) {
        (
            Matrix::from_column_major(N, K, v[..N * K].to_vec()),
            &v[N * K..],
        )
    }

    #[test]
    fn glms_match_the_host() {
        let ys = [0.3, -1.2, 0.8, 2.0, -0.4];
        let counts = [0, 2, 1, 5, 3];
        let binary = [1, 0, 0, 1, 1];
        let mut p = x_values();
        p.extend([0.2, 0.5, -0.3, 0.4, 1.3]); // alpha, beta (3), sigma or phi
        compare("normal_id_glm", &p, |v, dev| {
            let (x, r) = split(v);
            let y: Vec<Var<'_>> = ys.iter().map(|&y| v[0] * 0.0 + y).collect();
            if dev {
                normal_id_glm_lpdf_on_device(&y, &x, &r[..1], &r[1..4], &r[4..]).unwrap()
            } else {
                d::normal_id_glm_lpdf(&y, &x, r[0], &r[1..4], r[4]).unwrap()
            }
        });
        compare("bernoulli_logit_glm", &p[..p.len() - 1], |v, dev| {
            let (x, r) = split(v);
            if dev {
                bernoulli_logit_glm_lpmf_on_device(&binary, &x, &r[..1], &r[1..]).unwrap()
            } else {
                d::bernoulli_logit_glm_lpmf(&binary, &x, r[0], &r[1..]).unwrap()
            }
        });
        compare("poisson_log_glm", &p[..p.len() - 1], |v, dev| {
            let (x, r) = split(v);
            if dev {
                poisson_log_glm_lpmf_on_device(&counts, &x, &r[..1], &r[1..]).unwrap()
            } else {
                d::poisson_log_glm_lpmf(&counts, &x, r[0], &r[1..]).unwrap()
            }
        });
        compare("neg_binomial_2_log_glm", &p, |v, dev| {
            let (x, r) = split(v);
            if dev {
                neg_binomial_2_log_glm_lpmf_on_device(&counts, &x, &r[..1], &r[1..4], &r[4..])
                    .unwrap()
            } else {
                d::neg_binomial_2_log_glm_lpmf(&counts, &x, r[0], &r[1..4], r[4]).unwrap()
            }
        });
        let mut pc = x_values();
        pc.extend([0.1, -0.2, 0.3]); // alpha
        pc.extend((0..K * 3).map(|i| 0.1 * i as f64 - 0.4)); // beta, K by 3
        let classes = [1, 3, 2, 2, 1];
        compare("categorical_logit_glm", &pc, |v, dev| {
            let (x, r) = split(v);
            let beta = Matrix::from_column_major(K, 3, r[3..].to_vec());
            if dev {
                categorical_logit_glm_lpmf_on_device(&classes, &x, &r[..3], &beta).unwrap()
            } else {
                d::categorical_logit_glm_lpmf(&classes, &x, &r[..3], &beta).unwrap()
            }
        });
        let mut po = x_values();
        po.extend([0.4, -0.6, 0.9, -0.5, 0.7]); // beta (3), cuts (2)
        compare("ordered_logistic_glm", &po, |v, dev| {
            let (x, r) = split(v);
            if dev {
                ordered_logistic_glm_lpmf_on_device(&classes, &x, &r[..3], &r[3..]).unwrap()
            } else {
                d::ordered_logistic_glm_lpmf(&classes, &x, &r[..3], &r[3..]).unwrap()
            }
        });
    }
}
