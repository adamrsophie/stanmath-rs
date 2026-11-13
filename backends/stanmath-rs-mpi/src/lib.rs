//! MPI backend for Stan's `map_rect`.
//!
//! Every process runs the same program. Each one creates a [`Cluster`] and registers the same
//! jobs in the same order; then every process other than the root calls [`Cluster::listen`], which
//! serves the root until the root's cluster is dropped. The root calls [`map_rect_mpi`], which
//! splits the jobs into contiguous chunks, one per process, the root included.
//!
//! As in Stan, the data of a call site (`x_r` and `x_i`) go to the workers once, on the first call
//! with its `call_id`, and stay there; later calls with that `call_id` must pass the same data.
//! Each evaluation then sends only the parameters.
//!
//! The value is computed across the processes in `f64`. In reverse mode the vector-Jacobian
//! product is computed across the processes as well, each one recording its jobs on a tape of its
//! own. Forward-mode scalars are evaluated on the root alone, in the caller's scalar type, so
//! every mode gets exact derivatives.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, OnceLock};

use mpi::environment::Universe;
use mpi::topology::{Process, SimpleCommunicator};
use mpi::traits::*;
use stanmath_rs_autodiff::{MultiFn, Real, try_value_and_grad};
use stanmath_rs_core::{Error, Result};
use stanmath_rs_solvers::MapRectJob;

const F: &str = "map_rect_mpi";

// The command in the first entry of every header the root sends.
const STOP: u64 = 0;
const DATA: u64 = 1;
const EVAL: u64 = 2;
const VJP: u64 = 3;

/// The real and integer data of the jobs of one call site.
type Data = Vec<(Vec<f64>, Vec<i64>)>;

/// A [`MapRectJob`] behind a pointer, evaluated in `f64` or differentiated.
trait DynJob: Send + Sync {
    fn values(&self, phi: &[f64], theta: &[f64], x_r: &[f64], x_i: &[i64]) -> Result<Vec<f64>>;

    /// The gradient of `Σ w[k] out[k]` with respect to `phi` followed by `theta`.
    fn vjp(
        &self,
        phi: &[f64],
        theta: &[f64],
        x_r: &[f64],
        x_i: &[i64],
        w: &[f64],
    ) -> Result<Vec<f64>>;
}

impl<J: MapRectJob> DynJob for J {
    fn values(&self, phi: &[f64], theta: &[f64], x_r: &[f64], x_i: &[i64]) -> Result<Vec<f64>> {
        MapRectJob::eval(self, phi, theta, x_r, x_i)
    }

    fn vjp(
        &self,
        phi: &[f64],
        theta: &[f64],
        x_r: &[f64],
        x_i: &[i64],
        w: &[f64],
    ) -> Result<Vec<f64>> {
        let mut inputs = phi.to_vec();
        inputs.extend_from_slice(theta);
        try_value_and_grad(
            |v| {
                let (p, t) = v.split_at(phi.len());
                let out = MapRectJob::eval(self, p, t, x_r, x_i)?;
                if out.len() != w.len() {
                    return Err(Error::dimension(
                        F,
                        "a job changed its output length between passes",
                    ));
                }
                Ok(out.iter().zip(w).map(|(&o, &wi)| o * wi).sum())
            },
            &inputs,
        )
        .map(|r| r.gradient)
    }
}

/// The processes of an MPI program, with the jobs every process registered.
///
/// Create one per process with [`Cluster::init`]; it initializes MPI and finalizes it when
/// dropped. Dropping the root's cluster stops the workers.
pub struct Cluster {
    world: SimpleCommunicator,
    jobs: Vec<Arc<dyn DynJob>>,
    /// Call sites whose data the workers hold, with their number of jobs.
    sent: RefCell<HashMap<u64, usize>>,
    // Dropped last, so MPI is finalized after the root has stopped the workers.
    _universe: Universe,
}

/// A job registered with a [`Cluster`], for [`map_rect_mpi`].
pub struct JobHandle<J> {
    index: u64,
    job: Arc<J>,
}

impl Cluster {
    /// Initializes MPI.
    ///
    /// # Errors
    ///
    /// [`Error::Backend`] when MPI was already initialized in this process.
    pub fn init() -> Result<Self> {
        let universe = mpi::initialize().ok_or_else(|| Error::Backend {
            function: "Cluster::init",
            message: "MPI is already initialized".into(),
        })?;
        Ok(Cluster {
            world: universe.world(),
            jobs: Vec::new(),
            sent: RefCell::new(HashMap::new()),
            _universe: universe,
        })
    }

    /// This process's rank; the root has rank 0.
    pub fn rank(&self) -> usize {
        self.world.rank() as usize
    }

    /// The number of processes.
    pub fn size(&self) -> usize {
        self.world.size() as usize
    }

    /// True on the root process, the one that calls [`map_rect_mpi`].
    pub fn is_root(&self) -> bool {
        self.rank() == 0
    }

    /// Registers a job. Every process must register the same jobs in the same order.
    pub fn register<J: MapRectJob>(&mut self, job: J) -> JobHandle<J> {
        let job = Arc::new(job);
        self.jobs.push(Arc::clone(&job) as Arc<dyn DynJob>);
        JobHandle {
            index: (self.jobs.len() - 1) as u64,
            job,
        }
    }

    /// Serves the root's requests until the root's cluster is dropped. Returns at once on the
    /// root.
    pub fn listen(self) {
        if self.is_root() {
            return;
        }
        let root = self.world.process_at_rank(0);
        let mut cache: HashMap<u64, Data> = HashMap::new();
        loop {
            let (header, _) = root.receive_vec::<u64>();
            let &[cmd, job, call, k] = header.as_slice() else {
                return;
            };
            match cmd {
                DATA => {
                    let data = (0..k)
                        .map(|_| (root.receive_vec::<f64>().0, root.receive_vec::<i64>().0))
                        .collect();
                    cache.insert(call, data);
                }
                EVAL | VJP => {
                    let (phi, _) = root.receive_vec::<f64>();
                    let (thetas, _) = root.receive_vec::<f64>();
                    let (theta_lens, _) = root.receive_vec::<u64>();
                    let thetas = split(&thetas, &theta_lens);
                    let job = &*self.jobs[job as usize];
                    let data = cache.get(&call).map_or(&[][..], Vec::as_slice);
                    if cmd == EVAL {
                        match eval_chunk(job, &phi, &thetas, data) {
                            Ok(parts) => {
                                let lens: Vec<u64> = parts.iter().map(|p| p.len() as u64).collect();
                                root.send(&[0u64][..]);
                                root.send(&parts.concat()[..]);
                                root.send(&lens[..]);
                            }
                            Err(e) => send_error(&root, &e),
                        }
                    } else {
                        let (weights, _) = root.receive_vec::<f64>();
                        let (weight_lens, _) = root.receive_vec::<u64>();
                        let weights = split(&weights, &weight_lens);
                        match vjp_chunk(job, &phi, &thetas, data, &weights) {
                            Ok(g) => {
                                root.send(&[0u64][..]);
                                root.send(&g[..]);
                            }
                            Err(e) => send_error(&root, &e),
                        }
                    }
                }
                _ => return,
            }
        }
    }
}

impl Drop for Cluster {
    fn drop(&mut self) {
        if self.is_root() {
            for r in 1..self.world.size() {
                self.world.process_at_rank(r).send(&[STOP, 0, 0, 0][..]);
            }
        }
    }
}

/// `v` cut into consecutive pieces of the lengths `lens`.
fn split<'a>(v: &'a [f64], lens: &[u64]) -> Vec<&'a [f64]> {
    let mut start = 0;
    lens.iter()
        .map(|&l| {
            let piece = &v[start..start + l as usize];
            start += l as usize;
            piece
        })
        .collect()
}

fn send_error(root: &Process<'_>, e: &Error) {
    root.send(&[1u64][..]);
    root.send(e.to_string().as_bytes());
}

/// Receives a worker's reply: the payload received by `ok` on success, the worker's error
/// otherwise.
fn receive_reply<R>(p: &Process<'_>, ok: impl FnOnce(&Process<'_>) -> R) -> Result<R> {
    let (status, _) = p.receive_vec::<u64>();
    if status.first() == Some(&0) {
        Ok(ok(p))
    } else {
        let (msg, _) = p.receive_vec::<u8>();
        Err(Error::Backend {
            function: F,
            message: String::from_utf8_lossy(&msg).into_owned(),
        })
    }
}

fn eval_chunk(
    job: &dyn DynJob,
    phi: &[f64],
    thetas: &[&[f64]],
    data: &[(Vec<f64>, Vec<i64>)],
) -> Result<Vec<Vec<f64>>> {
    if data.len() != thetas.len() {
        return Err(Error::dimension(F, "a worker has no data for this call"));
    }
    thetas
        .iter()
        .zip(data)
        .map(|(theta, (x_r, x_i))| job.values(phi, theta, x_r, x_i))
        .collect()
}

/// The summed gradient in `phi`, followed by the gradient in each job's `theta`.
fn vjp_chunk(
    job: &dyn DynJob,
    phi: &[f64],
    thetas: &[&[f64]],
    data: &[(Vec<f64>, Vec<i64>)],
    weights: &[&[f64]],
) -> Result<Vec<f64>> {
    if data.len() != thetas.len() || weights.len() != thetas.len() {
        return Err(Error::dimension(F, "a worker has no data for this call"));
    }
    let mut out = vec![0.0; phi.len()];
    for ((theta, (x_r, x_i)), w) in thetas.iter().zip(data).zip(weights) {
        let g = job.vjp(phi, theta, x_r, x_i, w)?;
        for (o, gk) in out.iter_mut().zip(&g) {
            *o += gk;
        }
        out.extend_from_slice(&g[phi.len()..]);
    }
    Ok(out)
}

/// The jobs of each of `size` processes: contiguous ranges covering `0..jobs`.
fn chunks(jobs: usize, size: usize) -> Vec<Range<usize>> {
    (0..size)
        .map(|r| r * jobs / size..(r + 1) * jobs / size)
        .collect()
}

/// Stan's `map_rect` across the processes of `cluster`: evaluates
/// `job(phi, thetas[j], x_r[j], x_i[j])` for every job `j` and concatenates the results.
///
/// `call_id` names the call site. Its data are sent to the workers on its first call and reused
/// after, so every call with the same `call_id` must pass the same `x_r` and `x_i`.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when `thetas`, `x_r`, and `x_i` disagree on the number of jobs or
/// the number of jobs differs from the first call with `call_id`; [`Error::Backend`] when called on
/// a worker or when a worker's job fails in the forward pass; and the first error of the root's
/// own jobs. A job that fails only in the reverse pass gives NaN gradients.
pub fn map_rect_mpi<T: Real, J: MapRectJob>(
    cluster: &Cluster,
    job: &JobHandle<J>,
    call_id: u64,
    phi: &[T],
    thetas: &[Vec<T>],
    x_r: &[Vec<f64>],
    x_i: &[Vec<i64>],
) -> Result<Vec<T>> {
    if !cluster.is_root() {
        return Err(Error::Backend {
            function: F,
            message: "only the root process calls map_rect_mpi".into(),
        });
    }
    let jobs = thetas.len();
    if x_r.len() != jobs || x_i.len() != jobs {
        return Err(Error::dimension(
            F,
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
    let chunks = chunks(jobs, cluster.size());
    let mut sent = cluster.sent.borrow_mut();
    match sent.get(&call_id) {
        Some(&n) if n == jobs => {}
        Some(&n) => {
            return Err(Error::dimension(
                F,
                format!("call {call_id} first had {n} jobs, now {jobs}"),
            ));
        }
        None => {
            for (r, chunk) in chunks.iter().enumerate().skip(1) {
                let p = cluster.world.process_at_rank(r as i32);
                p.send(&[DATA, job.index, call_id, chunk.len() as u64][..]);
                for j in chunk.clone() {
                    p.send(&x_r[j][..]);
                    p.send(&x_i[j][..]);
                }
            }
            sent.insert(call_id, jobs);
        }
    }
    drop(sent);
    let mut inputs = phi.to_vec();
    let mut theta_starts = vec![phi.len()];
    for theta in thetas {
        inputs.extend_from_slice(theta);
        theta_starts.push(inputs.len());
    }
    let f = MpiMap {
        job: Arc::clone(&job.job),
        index: job.index,
        call_id,
        n_phi: phi.len(),
        theta_starts: Arc::new(theta_starts),
        x_r: Arc::new(x_r.to_vec()),
        x_i: Arc::new(x_i.to_vec()),
        chunks: Arc::new(chunks),
        out_lens: Arc::new(OnceLock::new()),
    };
    T::map_multi(&inputs, &f)
}

struct MpiMap<J> {
    job: Arc<J>,
    index: u64,
    call_id: u64,
    n_phi: usize,
    /// Start of each job's parameters in the flattened input, and one past the last.
    theta_starts: Arc<Vec<usize>>,
    x_r: Arc<Vec<Vec<f64>>>,
    x_i: Arc<Vec<Vec<i64>>>,
    /// The jobs of each process.
    chunks: Arc<Vec<Range<usize>>>,
    /// Output length of each job, recorded by the `f64` forward pass.
    out_lens: Arc<OnceLock<Vec<usize>>>,
}

impl<J> Clone for MpiMap<J> {
    fn clone(&self) -> Self {
        MpiMap {
            job: Arc::clone(&self.job),
            index: self.index,
            call_id: self.call_id,
            n_phi: self.n_phi,
            theta_starts: Arc::clone(&self.theta_starts),
            x_r: Arc::clone(&self.x_r),
            x_i: Arc::clone(&self.x_i),
            chunks: Arc::clone(&self.chunks),
            out_lens: Arc::clone(&self.out_lens),
        }
    }
}

impl<J: MapRectJob> MpiMap<J> {
    fn theta_range(&self, j: usize) -> Range<usize> {
        self.theta_starts[j]..self.theta_starts[j + 1]
    }

    /// Sends a command with the parameters of each worker's chunk; `extra` sends what follows.
    fn send_params(
        &self,
        world: &SimpleCommunicator,
        cmd: u64,
        x: &[f64],
        extra: impl Fn(&Process<'_>, &Range<usize>),
    ) {
        for (r, chunk) in self.chunks.iter().enumerate().skip(1) {
            let p = world.process_at_rank(r as i32);
            p.send(&[cmd, self.index, self.call_id, chunk.len() as u64][..]);
            p.send(&x[..self.n_phi]);
            p.send(&x[self.theta_starts[chunk.start]..self.theta_starts[chunk.end]]);
            let lens: Vec<u64> = chunk
                .clone()
                .map(|j| (self.theta_starts[j + 1] - self.theta_starts[j]) as u64)
                .collect();
            p.send(&lens[..]);
            extra(&p, chunk);
        }
    }

    fn local_data(&self) -> Data {
        self.chunks[0]
            .clone()
            .map(|j| (self.x_r[j].clone(), self.x_i[j].clone()))
            .collect()
    }

    fn forward_f64(&self, x: &[f64]) -> Result<Vec<Vec<f64>>> {
        let world = SimpleCommunicator::world();
        self.send_params(&world, EVAL, x, |_, _| {});
        let thetas: Vec<&[f64]> = self.chunks[0]
            .clone()
            .map(|j| &x[self.theta_range(j)])
            .collect();
        let mut parts = eval_chunk(&*self.job, &x[..self.n_phi], &thetas, &self.local_data());
        // Every reply is received, even after an error, so the next request starts clean.
        for r in 1..self.chunks.len() {
            let reply = receive_reply(&world.process_at_rank(r as i32), |p| {
                let (values, _) = p.receive_vec::<f64>();
                let (lens, _) = p.receive_vec::<u64>();
                split(&values, &lens)
                    .into_iter()
                    .map(<[f64]>::to_vec)
                    .collect::<Vec<_>>()
            });
            parts = match (parts, reply) {
                (Ok(mut a), Ok(b)) => {
                    a.extend(b);
                    Ok(a)
                }
                (Err(e), _) | (_, Err(e)) => Err(e),
            };
        }
        parts
    }
}

/// Sets `x_adj[k]` to NaN for every needed input `k` in `range`.
fn poison(x_adj: &mut [f64], needed: &[bool], range: impl Iterator<Item = usize>) {
    for k in range {
        if needed[k] {
            x_adj[k] = f64::NAN;
        }
    }
}

impl<J: MapRectJob> MultiFn for MpiMap<J> {
    fn eval<S: Real>(&self, xs: &[S]) -> Result<Vec<S>> {
        if S::ORDER > 0 {
            let phi = &xs[..self.n_phi];
            let mut out = Vec::new();
            for j in 0..self.x_r.len() {
                let theta = &xs[self.theta_range(j)];
                out.extend(self.job.eval(phi, theta, &self.x_r[j], &self.x_i[j])?);
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
        let world = SimpleCommunicator::world();
        self.send_params(&world, VJP, x, |p, chunk| {
            p.send(&y_adj[out_starts[chunk.start]..out_starts[chunk.end]]);
            let w_lens: Vec<u64> = chunk.clone().map(|j| lens[j] as u64).collect();
            p.send(&w_lens[..]);
        });
        let n_phi = self.n_phi;
        let local = &self.chunks[0];
        let thetas: Vec<&[f64]> = local.clone().map(|j| &x[self.theta_range(j)]).collect();
        let weights: Vec<&[f64]> = local
            .clone()
            .map(|j| &y_adj[out_starts[j]..out_starts[j + 1]])
            .collect();
        let mut grads = vec![vjp_chunk(
            &*self.job,
            &x[..n_phi],
            &thetas,
            &self.local_data(),
            &weights,
        )];
        for r in 1..self.chunks.len() {
            grads.push(receive_reply(&world.process_at_rank(r as i32), |p| {
                p.receive_vec::<f64>().0
            }));
        }
        for (chunk, g) in self.chunks.iter().zip(grads) {
            let theta_range = self.theta_starts[chunk.start]..self.theta_starts[chunk.end];
            match g {
                Ok(g) if g.len() == n_phi + theta_range.len() => {
                    let (g_phi, g_theta) = g.split_at(n_phi);
                    for (k, gk) in (0..n_phi).zip(g_phi).chain(theta_range.zip(g_theta)) {
                        if needed[k] {
                            x_adj[k] += gk;
                        }
                    }
                }
                _ => poison(x_adj, needed, (0..n_phi).chain(theta_range)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_cover_every_job_once() {
        for (jobs, size) in [(7, 3), (2, 4), (5, 1), (0, 2)] {
            let c = chunks(jobs, size);
            assert_eq!(c.len(), size);
            assert_eq!(c.first().map(|r| r.start), Some(0));
            assert_eq!(c.last().map(|r| r.end), Some(jobs));
            for w in c.windows(2) {
                assert_eq!(w[0].end, w[1].start);
            }
        }
    }

    #[test]
    fn split_cuts_in_order() {
        let v = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(split(&v, &[1, 0, 3]), vec![&v[..1], &v[1..1], &v[1..]]);
    }
}
