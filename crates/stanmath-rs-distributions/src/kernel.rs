//! Fused log densities with analytic partials.
//!
//! A [`Kernel`] gives the log density of one observation and, on request, its partial derivatives
//! with respect to the real arguments. Both are written once, generically over [`Real`]: the `f64`
//! instance computes values, and the instances at forward-mode types make the partials themselves
//! differentiable. [`sum_kernel`] broadcasts the kernel over vectorized arguments and records the
//! whole sum as a single n-ary operation, so a reverse-mode log likelihood over `n` observations
//! costs one tape node with one edge per non-constant argument element, as with Stan's
//! `operands_and_partials`.

use core::marker::PhantomData;

use stanmath_rs_autodiff::{Broadcast, IntArg, NaryFn, Promoted, Real, broadcast_len};
use stanmath_rs_core::Result;

/// The log density of one observation with `M` integer and `N` real arguments.
pub(crate) trait Kernel<const M: usize, const N: usize> {
    /// Function name used in error messages.
    const NAME: &'static str;

    /// Validates one element's arguments. `None` marks an empty sequence argument; it is only
    /// passed when the broadcast length is zero, to validate the scalar arguments as Stan does.
    fn check(ints: [Option<i64>; M], x: [Option<f64>; N]) -> Result<()>;

    /// Log density of one element; writes `d/dx[j]` into `grad[j]` when `grad` is given.
    fn eval<S: Real>(ints: [i64; M], x: [S; N], grad: Option<&mut [S; N]>) -> S;
}

/// The sum of a kernel over `n` broadcast elements, as an [`NaryFn`].
///
/// Operands are laid out argument by argument: a scalar argument contributes one operand, a
/// vector argument contributes `n`.
struct KernelSum<K, const M: usize, const N: usize> {
    n: usize,
    offset: [usize; N],
    scalar: [bool; N],
    ints: Vec<[i64; M]>,
    _kernel: PhantomData<K>,
}

impl<K: Kernel<M, N>, const M: usize, const N: usize> KernelSum<K, M, N> {
    #[inline]
    fn index(&self, j: usize, i: usize) -> usize {
        self.offset[j] + if self.scalar[j] { 0 } else { i }
    }

    #[inline]
    fn gather<S: Copy>(&self, x: &[S], i: usize) -> [S; N] {
        core::array::from_fn(|j| x[self.index(j, i)])
    }
}

impl<K: Kernel<M, N>, const M: usize, const N: usize> NaryFn for KernelSum<K, M, N> {
    fn value(&self, x: &[f64]) -> f64 {
        (0..self.n)
            .map(|i| K::eval(self.ints[i], self.gather(x, i), None))
            .sum()
    }

    fn partials<S: Real>(&self, x: &[S], _y: S, needed: &[bool], out: &mut [S]) {
        let mut g = [S::zero(); N];
        for i in 0..self.n {
            K::eval(self.ints[i], self.gather(x, i), Some(&mut g));
            for (j, gj) in g.iter().enumerate() {
                let k = self.index(j, i);
                if needed[k] {
                    out[k] += *gj;
                }
            }
        }
    }

    fn value_and_partials(&self, x: &[f64], needed: &[bool], out: &mut [f64]) -> f64 {
        let mut g = [0.0; N];
        let mut total = 0.0;
        for i in 0..self.n {
            total += K::eval(self.ints[i], self.gather(x, i), Some(&mut g));
            for (j, gj) in g.iter().enumerate() {
                let k = self.index(j, i);
                if needed[k] {
                    out[k] += gj;
                }
            }
        }
        total
    }
}

/// Validates and sums kernel `K` over broadcast arguments.
pub(crate) fn sum_kernel<K, A, const M: usize, const N: usize>(
    ints: [&dyn IntArg; M],
    args: A,
) -> Result<Promoted<A, N>>
where
    K: Kernel<M, N>,
    A: Broadcast<N>,
{
    let real_lens = args.lens();
    let mut lens: Vec<Option<usize>> = real_lens.to_vec();
    lens.extend(ints.iter().map(|a| a.int_len()));
    let n = broadcast_len(K::NAME, &lens)?;
    let scalar: [bool; N] = core::array::from_fn(|j| real_lens[j].is_none());

    let mut columns: [Vec<Promoted<A, N>>; N] = core::array::from_fn(|_| Vec::new());
    let mut int_rows = Vec::with_capacity(n);
    for i in 0..n {
        let elems = args.get(i);
        let int_row: [i64; M] = core::array::from_fn(|m| ints[m].int_at(i));
        K::check(int_row.map(Some), elems.map(|e| Some(e.value())))?;
        int_rows.push(int_row);
        for j in 0..N {
            if !scalar[j] || i == 0 {
                columns[j].push(elems[j]);
            }
        }
    }
    if n == 0 {
        // Every sequence is empty; Stan still validates the scalars.
        K::check(
            core::array::from_fn(|m| ints[m].int_scalar()),
            args.scalar_values(),
        )?;
        return Ok(Promoted::<A, N>::zero());
    }
    let mut offset = [0; N];
    let mut operands = Vec::with_capacity(columns.iter().map(Vec::len).sum());
    for j in 0..N {
        offset[j] = operands.len();
        operands.append(&mut columns[j]);
    }
    let f = KernelSum::<K, M, N> {
        n,
        offset,
        scalar,
        ints: int_rows,
        _kernel: PhantomData,
    };
    Ok(Promoted::<A, N>::map_nary(&operands, &f))
}

/// Sums `term` over broadcast elements after validating each with `check`. With an empty
/// sequence argument, `check` sees `None` in its place and validates the scalars, as Stan does.
pub(crate) fn sum_terms<A, const N: usize>(
    name: &'static str,
    args: A,
    ints: &[&dyn IntArg],
    check: impl Fn(&[Option<i64>], [Option<f64>; N]) -> Result<()>,
    term: impl Fn(&[i64], [Promoted<A, N>; N]) -> Promoted<A, N>,
) -> Result<Promoted<A, N>>
where
    A: Broadcast<N>,
{
    let mut lens = args.lens().to_vec();
    lens.extend(ints.iter().map(|a| a.int_len()));
    let n = broadcast_len(name, &lens)?;
    if n == 0 {
        let scalars: Vec<Option<i64>> = ints.iter().map(|a| a.int_scalar()).collect();
        check(&scalars, args.scalar_values())?;
        return Ok(Promoted::<A, N>::zero());
    }
    let mut total = Promoted::<A, N>::zero();
    let mut int_row = vec![0i64; ints.len()];
    let mut checked = vec![None; ints.len()];
    for i in 0..n {
        for ((slot, c), a) in int_row.iter_mut().zip(&mut checked).zip(ints) {
            *slot = a.int_at(i);
            *c = Some(*slot);
        }
        let x = args.get(i);
        check(&checked, x.map(|v| Some(v.value())))?;
        total += term(&int_row, x);
    }
    Ok(total)
}
