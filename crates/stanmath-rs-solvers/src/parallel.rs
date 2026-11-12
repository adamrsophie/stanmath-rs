//! Parallel `reduce_sum` and `map_rect`.
//!
//! A reverse-mode tape is not thread-safe, so the parallel work is wrapped in one multi-output
//! operation ([`MultiFn`]). Its value is computed in `f64` across threads. In reverse mode its
//! adjoint is computed across threads as well, each thread recording the partial sums or jobs
//! assigned to it on a tape of its own and returning a gradient. Forward-mode scalars ([`Dual`](stanmath_rs_autodiff::Dual), and therefore nested higher
//! derivatives) are evaluated sequentially in the caller's scalar type, so every mode gets exact
//! derivatives.
//!
//! Threads come from [`std::thread::scope`], at most
//! [`available_parallelism`](std::thread::available_parallelism) of them. Partial results are
//! combined in a fixed order, so results do not depend on the number of threads.

use std::sync::{Arc, OnceLock};

use stanmath_rs_autodiff::{MultiFn, Real, try_value_and_grad};
use stanmath_rs_core::{Error, Result};

/// The partial sum of [`reduce_sum_parallel`], written once generically over the scalar type.
pub trait PartialSum<X>: Send + Sync + 'static {
    /// The sum over `slice`, which is `x[start..end]` (zero based, half open), with shared
    /// parameters `params`.
    fn eval<S: Real>(&self, slice: &[X], start: usize, end: usize, params: &[S]) -> Result<S>;
}

/// The job of [`map_rect_parallel`], written once generically over the scalar type.
pub trait MapRectJob: Send + Sync + 'static {
    /// Evaluates one job from the shared parameters `phi`, its own parameters `theta`, and its
    /// real and integer data.
    fn eval<S: Real>(&self, phi: &[S], theta: &[S], x_r: &[f64], x_i: &[i64]) -> Result<Vec<S>>;
}

/// Half open ranges of at most `grainsize` elements covering `0..n`.
fn chunks(n: usize, grainsize: usize) -> Vec<(usize, usize)> {
    (0..n)
        .step_by(grainsize)
        .map(|s| (s, (s + grainsize).min(n)))
        .collect()
}

/// `(0..n).map(job)` spread over threads, results in index order.
fn par_map<R: Send>(n: usize, job: impl Fn(usize) -> R + Sync) -> Vec<R> {
    let threads = std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .min(n);
    if threads <= 1 {
        return (0..n).map(job).collect();
    }
    let per = n.div_ceil(threads);
    let job = &job;
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|w| {
                scope.spawn(move || {
                    (w * per..((w + 1) * per).min(n))
                        .map(job)
                        .collect::<Vec<R>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
            .collect()
    })
}

/// `Σ out[i] w[i]`, the scalar whose gradient is the vector-Jacobian product of a job.
fn weighted_sum<S: Real>(out: &[S], w: &[f64]) -> S {
    out.iter().zip(w).map(|(&o, &wi)| o * wi).sum()
}

/// Sets `x_adj[k]` to NaN for every needed input `k` in `range`, the result of a partial sum or
/// job that failed in the reverse pass after succeeding in the forward pass.
fn poison(x_adj: &mut [f64], needed: &[bool], range: impl Iterator<Item = usize>) {
    for k in range {
        if needed[k] {
            x_adj[k] = f64::NAN;
        }
    }
}

struct ParallelSum<X, P> {
    p: Arc<P>,
    x: Arc<Vec<X>>,
    grainsize: usize,
}

impl<X, P> Clone for ParallelSum<X, P> {
    fn clone(&self) -> Self {
        ParallelSum {
            p: Arc::clone(&self.p),
            x: Arc::clone(&self.x),
            grainsize: self.grainsize,
        }
    }
}

impl<X: Send + Sync + 'static, P: PartialSum<X>> MultiFn for ParallelSum<X, P> {
    fn eval<S: Real>(&self, params: &[S]) -> Result<Vec<S>> {
        let ranges = chunks(self.x.len(), self.grainsize);
        if S::ORDER > 0 {
            let mut total = S::zero();
            for &(s, e) in &ranges {
                total += self.p.eval(&self.x[s..e], s, e, params)?;
            }
            return Ok(vec![total]);
        }
        let values: Vec<f64> = params.iter().map(Real::value).collect();
        let parts = par_map(ranges.len(), |c| {
            let (s, e) = ranges[c];
            self.p.eval(&self.x[s..e], s, e, &values)
        });
        let mut total = 0.0;
        for part in parts {
            total += part?;
        }
        Ok(vec![S::from_f64(total)])
    }

    fn reverse(&self, x: &[f64], _y: &[f64], y_adj: &[f64], needed: &[bool], x_adj: &mut [f64]) {
        let ranges = chunks(self.x.len(), self.grainsize);
        let grads = par_map(ranges.len(), |c| {
            let (s, e) = ranges[c];
            try_value_and_grad(|v| self.p.eval(&self.x[s..e], s, e, v), x).map(|r| r.gradient)
        });
        for g in grads {
            match g {
                Ok(g) => {
                    for (k, gk) in g.iter().enumerate() {
                        if needed[k] {
                            x_adj[k] += y_adj[0] * gk;
                        }
                    }
                }
                Err(_) => poison(x_adj, needed, 0..x.len()),
            }
        }
    }
}

/// Stan's `reduce_sum` run in parallel: `Σ partial_sum(x[start..end], start, end, params)` over a
/// partition of `x` into slices of at most `grainsize` elements.
///
/// Unlike [`reduce_sum`](crate::reduce_sum), the parameters that carry derivatives are passed
/// explicitly as `params`, because each thread evaluates the partial sum on its own tape. The
/// partition depends on `grainsize` only, so the result is also reproducible as with
/// `reduce_sum_static`. `x` is copied once so the threads can share it.
///
/// # Errors
///
/// [`Error::InvalidParameter`] when `grainsize` is zero, and the first error of a partial sum in
/// the forward pass. A partial sum that fails only in the reverse pass gives NaN gradients.
pub fn reduce_sum_parallel<X, T, P>(
    partial_sum: P,
    x: &[X],
    grainsize: usize,
    params: &[T],
) -> Result<T>
where
    X: Clone + Send + Sync + 'static,
    T: Real,
    P: PartialSum<X>,
{
    if grainsize == 0 {
        return Err(Error::parameter(
            "reduce_sum_parallel",
            "grainsize",
            0.0,
            "positive",
        ));
    }
    if x.is_empty() {
        return Ok(T::zero());
    }
    let f = ParallelSum {
        p: Arc::new(partial_sum),
        x: Arc::new(x.to_vec()),
        grainsize,
    };
    Ok(T::map_multi(params, &f)?[0])
}

struct ParallelMap<J> {
    job: Arc<J>,
    n_phi: usize,
    /// Start of each job's parameters in the flattened input, and one past the last.
    theta_starts: Arc<Vec<usize>>,
    x_r: Arc<Vec<Vec<f64>>>,
    x_i: Arc<Vec<Vec<i64>>>,
    /// Output length of each job, recorded by the `f64` forward pass.
    out_lens: Arc<OnceLock<Vec<usize>>>,
}

impl<J> Clone for ParallelMap<J> {
    fn clone(&self) -> Self {
        ParallelMap {
            job: Arc::clone(&self.job),
            n_phi: self.n_phi,
            theta_starts: Arc::clone(&self.theta_starts),
            x_r: Arc::clone(&self.x_r),
            x_i: Arc::clone(&self.x_i),
            out_lens: Arc::clone(&self.out_lens),
        }
    }
}

impl<J> ParallelMap<J> {
    fn jobs(&self) -> usize {
        self.x_r.len()
    }
    fn theta_range(&self, j: usize) -> core::ops::Range<usize> {
        self.theta_starts[j]..self.theta_starts[j + 1]
    }
}

impl<J: MapRectJob> ParallelMap<J> {
    fn forward_f64(&self, xs: &[f64]) -> Result<Vec<Vec<f64>>> {
        let phi = &xs[..self.n_phi];
        par_map(self.jobs(), |j| {
            self.job
                .eval(phi, &xs[self.theta_range(j)], &self.x_r[j], &self.x_i[j])
        })
        .into_iter()
        .collect()
    }
}

impl<J: MapRectJob> MultiFn for ParallelMap<J> {
    fn eval<S: Real>(&self, xs: &[S]) -> Result<Vec<S>> {
        if S::ORDER > 0 {
            let phi = &xs[..self.n_phi];
            let mut out = Vec::new();
            for j in 0..self.jobs() {
                out.extend(self.job.eval(
                    phi,
                    &xs[self.theta_range(j)],
                    &self.x_r[j],
                    &self.x_i[j],
                )?);
            }
            return Ok(out);
        }
        let values: Vec<f64> = xs.iter().map(Real::value).collect();
        let parts = self.forward_f64(&values)?;
        let _ = self.out_lens.set(parts.iter().map(Vec::len).collect());
        Ok(parts.into_iter().flatten().map(S::from_f64).collect())
    }

    fn reverse(&self, x: &[f64], _y: &[f64], y_adj: &[f64], needed: &[bool], x_adj: &mut [f64]) {
        let lens: Vec<usize> = match self.out_lens.get() {
            Some(l) => l.clone(),
            None => match self.forward_f64(x) {
                Ok(parts) => parts.iter().map(Vec::len).collect(),
                Err(_) => return poison(x_adj, needed, 0..x.len()),
            },
        };
        let mut out_starts = vec![0; lens.len() + 1];
        for (j, l) in lens.iter().enumerate() {
            out_starts[j + 1] = out_starts[j] + l;
        }
        let n_phi = self.n_phi;
        let grads = par_map(self.jobs(), |j| {
            let weights = &y_adj[out_starts[j]..out_starts[j + 1]];
            let mut inputs = x[..n_phi].to_vec();
            inputs.extend_from_slice(&x[self.theta_range(j)]);
            try_value_and_grad(
                |v| {
                    let (phi, theta) = v.split_at(n_phi);
                    let out = self.job.eval(phi, theta, &self.x_r[j], &self.x_i[j])?;
                    if out.len() != weights.len() {
                        return Err(Error::dimension(
                            "map_rect_parallel",
                            "a job changed its output length between passes",
                        ));
                    }
                    Ok(weighted_sum(&out, weights))
                },
                &inputs,
            )
            .map(|r| r.gradient)
        });
        for (j, g) in grads.into_iter().enumerate() {
            let theta = self.theta_range(j);
            match g {
                Ok(g) => {
                    let (g_phi, g_theta) = g.split_at(n_phi);
                    for (k, gk) in g_phi.iter().enumerate() {
                        if needed[k] {
                            x_adj[k] += gk;
                        }
                    }
                    for (k, gk) in theta.zip(g_theta) {
                        if needed[k] {
                            x_adj[k] += gk;
                        }
                    }
                }
                Err(_) => poison(x_adj, needed, (0..n_phi).chain(theta)),
            }
        }
    }
}

/// Stan's `map_rect` run in parallel: evaluates `job(phi, thetas[j], x_r[j], x_i[j])` for every
/// job `j` and concatenates the results.
///
/// Unlike [`map_rect`](crate::map_rect), the job is a [`MapRectJob`] rather than a closure, so
/// each thread can evaluate it on a tape of its own. The data are copied once so the threads can
/// share them.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when `thetas`, `x_r`, and `x_i` disagree on the number of jobs,
/// and the first error of a job in the forward pass. A job that fails only in the reverse pass
/// gives NaN gradients.
pub fn map_rect_parallel<T: Real, J: MapRectJob>(
    job: J,
    phi: &[T],
    thetas: &[Vec<T>],
    x_r: &[Vec<f64>],
    x_i: &[Vec<i64>],
) -> Result<Vec<T>> {
    let jobs = thetas.len();
    if x_r.len() != jobs || x_i.len() != jobs {
        return Err(Error::dimension(
            "map_rect_parallel",
            format!(
                "{jobs} job parameter sets, {} real data sets, {} integer data sets",
                x_r.len(),
                x_i.len()
            ),
        ));
    }
    if jobs == 0 {
        return Ok(Vec::new());
    }
    let mut inputs = phi.to_vec();
    let mut theta_starts = vec![phi.len()];
    for theta in thetas {
        inputs.extend_from_slice(theta);
        theta_starts.push(inputs.len());
    }
    let f = ParallelMap {
        job: Arc::new(job),
        n_phi: phi.len(),
        theta_starts: Arc::new(theta_starts),
        x_r: Arc::new(x_r.to_vec()),
        x_i: Arc::new(x_i.to_vec()),
        out_lens: Arc::new(OnceLock::new()),
    };
    T::map_multi(&inputs, &f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use stanmath_rs_autodiff::grad;

    /// Sum of `params[0] x^2 + exp(params[1] x)`.
    struct Quad;
    impl PartialSum<f64> for Quad {
        fn eval<S: Real>(&self, slice: &[f64], _: usize, _: usize, params: &[S]) -> Result<S> {
            Ok(slice
                .iter()
                .map(|&x| params[0] * (x * x) + (params[1] * x).exp())
                .sum())
        }
    }

    #[test]
    fn parallel_matches_sequential() {
        let x: Vec<f64> = (0..37).map(|i| f64::from(i) * 0.1).collect();
        let p = [0.3, -0.2];
        let value: f64 = reduce_sum_parallel(Quad, &x, 4, &p).unwrap();
        let direct: f64 = x.iter().map(|&xi| p[0] * xi * xi + (p[1] * xi).exp()).sum();
        assert!((value - direct).abs() < 1e-12);
        let par = grad(|v| reduce_sum_parallel(Quad, &x, 4, v).unwrap(), &p);
        let seq = grad(
            |v| crate::reduce_sum(|s: &[f64], a, b| Quad.eval(s, a, b, v), &x, 4).unwrap(),
            &p,
        );
        for (a, b) in par.iter().zip(&seq) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }
}
