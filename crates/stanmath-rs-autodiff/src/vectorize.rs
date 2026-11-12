//! Scalar promotion and broadcasting for vectorized functions.
//!
//! Probability functions accept any mix of scalars and slices, and any mix of `f64` data and
//! differentiable parameters, like Stan's vectorized signatures:
//!
//! ```
//! use stanmath_rs_autodiff::{Broadcast, Real, Tape};
//!
//! // A toy vectorized function: sum of (y - mu) over the broadcast length.
//! fn sum_diff<Y, M>(y: Y, mu: M) -> <(Y, M) as Broadcast<2>>::Out
//! where
//!     (Y, M): Broadcast<2>,
//! {
//!     let args = (y, mu);
//!     let n = args.broadcast_len("sum_diff").unwrap();
//!     let mut acc = <(Y, M) as Broadcast<2>>::Out::zero();
//!     for i in 0..n {
//!         let [y, mu] = args.get(i);
//!         acc += y - mu;
//!     }
//!     acc
//! }
//!
//! let data = [1.0, 2.0, 3.0];
//! let plain: f64 = sum_diff(&data[..], 1.0);
//! assert_eq!(plain, 3.0);
//!
//! let tape = Tape::new();
//! let mu = tape.var(1.0);
//! let lp = sum_diff(&data[..], mu);
//! assert_eq!(lp.gradient().wrt(mu), -3.0);
//! ```
//!
//! The machinery has three parts. [`Arg`] describes one argument: its element type and whether
//! it is a scalar (broadcast to every index) or a slice. [`Promote`] computes the common scalar
//! type of two element types: `f64` combined with `T` gives `T`. [`Lift`] converts an element to
//! the common type. [`Broadcast`] ties them together for tuples of up to five arguments.

use stanmath_rs_core::{Error, Result};

use crate::dual::Dual;
use crate::real::Real;
use crate::tape::Var;

/// The common scalar type of `Self` and `B`.
pub trait Promote<B> {
    /// The promoted type.
    type Out: Real;
}

/// Conversion of an element into the promoted scalar type `T`.
pub trait Lift<T> {
    /// Converts `self` into `T`.
    fn lift(self) -> T;
}

impl<T: Real> Promote<T> for f64 {
    type Out = T;
}
impl<'t> Promote<f64> for Var<'t> {
    type Out = Var<'t>;
}
impl<'t> Promote<Var<'t>> for Var<'t> {
    type Out = Var<'t>;
}
impl<T: Real> Promote<f64> for Dual<T> {
    type Out = Dual<T>;
}
impl<T: Real> Promote<Dual<T>> for Dual<T> {
    type Out = Dual<T>;
}

impl<T: Real> Lift<T> for f64 {
    #[inline]
    fn lift(self) -> T {
        T::from_f64(self)
    }
}
impl<'t> Lift<Var<'t>> for Var<'t> {
    #[inline]
    fn lift(self) -> Self {
        self
    }
}
impl<T: Real> Lift<Dual<T>> for Dual<T> {
    #[inline]
    fn lift(self) -> Self {
        self
    }
}

/// One argument of a vectorized function: a scalar or a sequence of elements.
pub trait Arg {
    /// Element type.
    type Elem: Copy;
    /// `None` for a scalar (broadcast), `Some(n)` for a sequence of length `n`.
    fn arg_len(&self) -> Option<usize>;
    /// Element `i`; scalars ignore `i`.
    fn arg_at(&self, i: usize) -> Self::Elem;
}

impl Arg for f64 {
    type Elem = f64;
    #[inline]
    fn arg_len(&self) -> Option<usize> {
        None
    }
    #[inline]
    fn arg_at(&self, _: usize) -> f64 {
        *self
    }
}

impl<'t> Arg for Var<'t> {
    type Elem = Var<'t>;
    #[inline]
    fn arg_len(&self) -> Option<usize> {
        None
    }
    #[inline]
    fn arg_at(&self, _: usize) -> Self {
        *self
    }
}

impl<T: Real> Arg for Dual<T> {
    type Elem = Dual<T>;
    #[inline]
    fn arg_len(&self) -> Option<usize> {
        None
    }
    #[inline]
    fn arg_at(&self, _: usize) -> Self {
        *self
    }
}

impl<E: Copy> Arg for &[E] {
    type Elem = E;
    #[inline]
    fn arg_len(&self) -> Option<usize> {
        Some(self.len())
    }
    #[inline]
    fn arg_at(&self, i: usize) -> E {
        self[i]
    }
}

impl<E: Copy> Arg for &Vec<E> {
    type Elem = E;
    #[inline]
    fn arg_len(&self) -> Option<usize> {
        Some(self.len())
    }
    #[inline]
    fn arg_at(&self, i: usize) -> E {
        self[i]
    }
}

impl<E: Copy, const N: usize> Arg for &[E; N] {
    type Elem = E;
    #[inline]
    fn arg_len(&self) -> Option<usize> {
        Some(N)
    }
    #[inline]
    fn arg_at(&self, i: usize) -> E {
        self[i]
    }
}

/// Integer arguments (counts and categories) for discrete distributions.
pub trait IntArg {
    /// `None` for a scalar (broadcast), `Some(n)` for a sequence of length `n`.
    fn int_len(&self) -> Option<usize>;
    /// Element `i`; scalars ignore `i`.
    fn int_at(&self, i: usize) -> i64;
}

impl IntArg for i64 {
    fn int_len(&self) -> Option<usize> {
        None
    }
    fn int_at(&self, _: usize) -> i64 {
        *self
    }
}

impl IntArg for &[i64] {
    fn int_len(&self) -> Option<usize> {
        Some(self.len())
    }
    fn int_at(&self, i: usize) -> i64 {
        self[i]
    }
}

impl IntArg for &Vec<i64> {
    fn int_len(&self) -> Option<usize> {
        Some(self.len())
    }
    fn int_at(&self, i: usize) -> i64 {
        self[i]
    }
}

impl<const N: usize> IntArg for &[i64; N] {
    fn int_len(&self) -> Option<usize> {
        Some(N)
    }
    fn int_at(&self, i: usize) -> i64 {
        self[i]
    }
}

/// Common length of a set of argument lengths, following Stan's rules: every sequence must have
/// the same length, scalars broadcast, and an empty sequence makes the result empty.
///
/// Returns `Ok(1)` when every argument is a scalar.
pub fn broadcast_len(function: &'static str, lens: &[Option<usize>]) -> Result<usize> {
    let mut n: Option<usize> = None;
    for &len in lens.iter().flatten() {
        match n {
            None => n = Some(len),
            Some(m) if m != len => {
                return Err(Error::dimension(
                    function,
                    format!("vector arguments have lengths {m} and {len}"),
                ));
            }
            Some(_) => {}
        }
    }
    Ok(n.unwrap_or(1))
}

/// A tuple of `N` [`Arg`]s with a common promoted scalar type.
///
/// Elements come back as an array so that generic code can destructure them:
/// `let [y, mu, sigma] = args.get(i);`.
pub trait Broadcast<const N: usize> {
    /// The common scalar type of all arguments.
    type Out: Real;
    /// Lengths of the arguments (`None` for scalars).
    fn lens(&self) -> [Option<usize>; N];
    /// Elements at index `i`, each converted to [`Broadcast::Out`].
    fn get(&self, i: usize) -> [Self::Out; N];
    /// Broadcast length; see [`broadcast_len`].
    fn broadcast_len(&self, function: &'static str) -> Result<usize> {
        broadcast_len(function, &self.lens())
    }
    /// True when no argument is a sequence.
    fn all_scalar(&self) -> bool {
        self.lens().iter().all(Option::is_none)
    }
}

/// Promoted type of two element types.
pub type Promoted2<A, B> = <A as Promote<B>>::Out;

/// Common scalar type of an argument tuple: `Promoted<(Y, Mu, Sigma), 3>`.
pub type Promoted<A, const N: usize> = <A as Broadcast<N>>::Out;

impl<A: Arg> Broadcast<1> for (A,)
where
    A::Elem: Promote<f64>,
    A::Elem: Lift<Promoted2<A::Elem, f64>>,
{
    type Out = Promoted2<A::Elem, f64>;
    fn lens(&self) -> [Option<usize>; 1] {
        [self.0.arg_len()]
    }
    #[inline]
    fn get(&self, i: usize) -> [Self::Out; 1] {
        [self.0.arg_at(i).lift()]
    }
}

// The where clauses for each arity are spelled out by hand so that the promoted type of every
// prefix is constrained explicitly.
impl<A: Arg, B: Arg> Broadcast<2> for (A, B)
where
    A::Elem: Promote<B::Elem>,
    A::Elem: Lift<Promoted2<A::Elem, B::Elem>>,
    B::Elem: Lift<Promoted2<A::Elem, B::Elem>>,
{
    type Out = Promoted2<A::Elem, B::Elem>;
    fn lens(&self) -> [Option<usize>; 2] {
        [self.0.arg_len(), self.1.arg_len()]
    }
    #[inline]
    fn get(&self, i: usize) -> [Self::Out; 2] {
        [self.0.arg_at(i).lift(), self.1.arg_at(i).lift()]
    }
}

type P3<A, B, C> = Promoted2<Promoted2<A, B>, C>;
type P4<A, B, C, D> = Promoted2<P3<A, B, C>, D>;
type P5<A, B, C, D, E> = Promoted2<P4<A, B, C, D>, E>;

impl<A: Arg, B: Arg, C: Arg> Broadcast<3> for (A, B, C)
where
    A::Elem: Promote<B::Elem>,
    Promoted2<A::Elem, B::Elem>: Promote<C::Elem>,
    A::Elem: Lift<P3<A::Elem, B::Elem, C::Elem>>,
    B::Elem: Lift<P3<A::Elem, B::Elem, C::Elem>>,
    C::Elem: Lift<P3<A::Elem, B::Elem, C::Elem>>,
{
    type Out = P3<A::Elem, B::Elem, C::Elem>;
    fn lens(&self) -> [Option<usize>; 3] {
        [self.0.arg_len(), self.1.arg_len(), self.2.arg_len()]
    }
    #[inline]
    fn get(&self, i: usize) -> [Self::Out; 3] {
        [
            self.0.arg_at(i).lift(),
            self.1.arg_at(i).lift(),
            self.2.arg_at(i).lift(),
        ]
    }
}

impl<A: Arg, B: Arg, C: Arg, D: Arg> Broadcast<4> for (A, B, C, D)
where
    A::Elem: Promote<B::Elem>,
    Promoted2<A::Elem, B::Elem>: Promote<C::Elem>,
    P3<A::Elem, B::Elem, C::Elem>: Promote<D::Elem>,
    A::Elem: Lift<P4<A::Elem, B::Elem, C::Elem, D::Elem>>,
    B::Elem: Lift<P4<A::Elem, B::Elem, C::Elem, D::Elem>>,
    C::Elem: Lift<P4<A::Elem, B::Elem, C::Elem, D::Elem>>,
    D::Elem: Lift<P4<A::Elem, B::Elem, C::Elem, D::Elem>>,
{
    type Out = P4<A::Elem, B::Elem, C::Elem, D::Elem>;
    fn lens(&self) -> [Option<usize>; 4] {
        [
            self.0.arg_len(),
            self.1.arg_len(),
            self.2.arg_len(),
            self.3.arg_len(),
        ]
    }
    #[inline]
    fn get(&self, i: usize) -> [Self::Out; 4] {
        [
            self.0.arg_at(i).lift(),
            self.1.arg_at(i).lift(),
            self.2.arg_at(i).lift(),
            self.3.arg_at(i).lift(),
        ]
    }
}

impl<A: Arg, B: Arg, C: Arg, D: Arg, E: Arg> Broadcast<5> for (A, B, C, D, E)
where
    A::Elem: Promote<B::Elem>,
    Promoted2<A::Elem, B::Elem>: Promote<C::Elem>,
    P3<A::Elem, B::Elem, C::Elem>: Promote<D::Elem>,
    P4<A::Elem, B::Elem, C::Elem, D::Elem>: Promote<E::Elem>,
    A::Elem: Lift<P5<A::Elem, B::Elem, C::Elem, D::Elem, E::Elem>>,
    B::Elem: Lift<P5<A::Elem, B::Elem, C::Elem, D::Elem, E::Elem>>,
    C::Elem: Lift<P5<A::Elem, B::Elem, C::Elem, D::Elem, E::Elem>>,
    D::Elem: Lift<P5<A::Elem, B::Elem, C::Elem, D::Elem, E::Elem>>,
    E::Elem: Lift<P5<A::Elem, B::Elem, C::Elem, D::Elem, E::Elem>>,
{
    type Out = P5<A::Elem, B::Elem, C::Elem, D::Elem, E::Elem>;
    fn lens(&self) -> [Option<usize>; 5] {
        [
            self.0.arg_len(),
            self.1.arg_len(),
            self.2.arg_len(),
            self.3.arg_len(),
            self.4.arg_len(),
        ]
    }
    #[inline]
    fn get(&self, i: usize) -> [Self::Out; 5] {
        [
            self.0.arg_at(i).lift(),
            self.1.arg_at(i).lift(),
            self.2.arg_at(i).lift(),
            self.3.arg_at(i).lift(),
            self.4.arg_at(i).lift(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tape;

    #[test]
    fn lengths_follow_stan_rules() {
        assert_eq!(broadcast_len("f", &[None, None]).unwrap(), 1);
        assert_eq!(broadcast_len("f", &[Some(3), None, Some(3)]).unwrap(), 3);
        assert_eq!(broadcast_len("f", &[Some(0), None]).unwrap(), 0);
        assert!(matches!(
            broadcast_len("f", &[Some(2), Some(3)]),
            Err(Error::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn promotion_mixes_data_and_parameters() {
        let tape = Tape::new();
        let mu = tape.var(0.5);
        let data = vec![1.0, 2.0];
        let args = (&data, mu, 2.0);
        assert_eq!(args.broadcast_len("f").unwrap(), 2);
        let [y, m, s] = args.get(1);
        assert!(y.is_constant() && !m.is_constant() && s.is_constant());
        assert_eq!((y.value(), m.value(), s.value()), (2.0, 0.5, 2.0));
    }
}
