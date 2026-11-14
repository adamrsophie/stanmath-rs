//! Checks `map_rect_mpi` against `map_rect_parallel`. Run with any number of processes, for
//! example `mpirun -n 3 target/debug/examples/map_rect_mpi`; it exits nonzero on a mismatch.

use stanmath_rs_autodiff::{Real, grad};
use stanmath_rs_core::{Error, Result};
use stanmath_rs_mpi::{Cluster, map_rect_mpi};
use stanmath_rs_solvers::{MapRectJob, map_rect_parallel};

/// Outputs `phi[0] x + theta[0] exp(phi[1] x)` for each real datum, `x_i[0]` times over.
struct Job;

impl MapRectJob for Job {
    fn eval<S: Real>(&self, phi: &[S], theta: &[S], x_r: &[f64], x_i: &[i64]) -> Result<Vec<S>> {
        let reps = x_i[0] as usize;
        Ok((0..reps)
            .flat_map(|_| {
                x_r.iter()
                    .map(|&x| phi[0] * x + theta[0] * (phi[1] * x).exp())
            })
            .collect())
    }
}

/// Fails when its job parameter is negative.
struct Picky;

impl MapRectJob for Picky {
    fn eval<S: Real>(&self, _: &[S], theta: &[S], _: &[f64], _: &[i64]) -> Result<Vec<S>> {
        if theta[0] < 0.0 {
            return Err(Error::parameter(
                "Picky",
                "theta",
                theta[0].value(),
                "nonnegative",
            ));
        }
        Ok(vec![theta[0]])
    }
}

fn main() {
    let mut cluster = Cluster::init().expect("MPI init");
    let job = cluster.register(Job);
    let picky = cluster.register(Picky);
    if !cluster.is_root() {
        cluster.listen();
        return;
    }
    let jobs = 7;
    let x_r: Vec<Vec<f64>> = (0..jobs).map(|j| vec![0.1 * j as f64, 0.5, -0.3]).collect();
    let x_i: Vec<Vec<i64>> = (0..jobs).map(|j| vec![1 + (j % 2) as i64]).collect();
    let point: Vec<f64> = [0.4, -0.2]
        .into_iter()
        .chain((0..jobs).map(|j| 0.3 + 0.1 * j as f64))
        .collect();
    let mut failures = Vec::new();
    for round in 0..2 {
        let g_mpi = grad(
            |v| {
                let thetas: Vec<Vec<_>> = v[2..].iter().map(|&t| vec![t]).collect();
                let out = map_rect_mpi(&cluster, &job, 1, &v[..2], &thetas, &x_r, &x_i).unwrap();
                out.iter().map(|&o| o * o).sum()
            },
            &point,
        );
        let g_par = grad(
            |v| {
                let thetas: Vec<Vec<_>> = v[2..].iter().map(|&t| vec![t]).collect();
                let out = map_rect_parallel(Job, &v[..2], &thetas, &x_r, &x_i).unwrap();
                out.iter().map(|&o| o * o).sum()
            },
            &point,
        );
        for (k, (a, b)) in g_mpi.iter().zip(&g_par).enumerate() {
            if (a - b).abs() > 1e-12 * (1.0 + b.abs()) {
                failures.push(format!("round {round}, gradient {k}: {a} vs {b}"));
            }
        }
        let thetas: Vec<Vec<f64>> = point[2..].iter().map(|&t| vec![t]).collect();
        let v_mpi: Vec<f64> =
            map_rect_mpi(&cluster, &job, 1, &point[..2], &thetas, &x_r, &x_i).unwrap();
        let v_par: Vec<f64> = map_rect_parallel(Job, &point[..2], &thetas, &x_r, &x_i).unwrap();
        if v_mpi != v_par {
            failures.push(format!("round {round}: values {v_mpi:?} vs {v_par:?}"));
        }
    }
    let no_data = vec![vec![]; 4];
    let no_ints = vec![vec![]; 4];
    let bad: Result<Vec<f64>> = map_rect_mpi(
        &cluster,
        &picky,
        2,
        &[],
        &[vec![1.0], vec![2.0], vec![3.0], vec![-1.0]],
        &no_data,
        &no_ints,
    );
    if bad.is_ok() {
        failures.push("a failing job in the last chunk returned Ok".into());
    }
    let good: Result<Vec<f64>> = map_rect_mpi(
        &cluster,
        &picky,
        2,
        &[],
        &[vec![1.0], vec![2.0], vec![3.0], vec![4.0]],
        &no_data,
        &no_ints,
    );
    if good.as_deref().ok() != Some(&[1.0, 2.0, 3.0, 4.0][..]) {
        failures.push(format!(
            "the cluster did not recover after an error: {good:?}"
        ));
    }
    if failures.is_empty() {
        println!(
            "map_rect_mpi matches map_rect_parallel on {} processes",
            cluster.size()
        );
    } else {
        eprintln!("{}", failures.join("\n"));
        drop(cluster);
        std::process::exit(1);
    }
}
