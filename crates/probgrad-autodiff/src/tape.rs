//! Reverse-mode automatic differentiation on an explicit tape. See [`Tape`].

use core::cell::RefCell;
use core::cmp::Ordering;
use core::fmt;
use core::iter::Sum;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::real::{BinaryFn, MultiFn, NaryFn, Real, UnaryFn};

/// Index used for constants, which have no node.
const NO_NODE: u32 = u32::MAX;

type Rule = Box<dyn Fn(&[f64], &mut [f64]) + Send>;

struct CustomOp {
    first_output: u32,
    n_outputs: u32,
    /// Node index of each input, `NO_NODE` for constant inputs.
    inputs: Vec<u32>,
    rule: Rule,
}

#[derive(Default)]
struct Nodes {
    /// `edge_end[i]` is one past the last edge of node `i`; edges of node `i` start at
    /// `edge_end[i - 1]` (or 0).
    edge_end: Vec<u32>,
    parents: Vec<u32>,
    partials: Vec<f64>,
    /// Custom operations in creation order. Their output nodes have no edges.
    custom: Vec<CustomOp>,
}

/// An append-only record of operations for reverse-mode differentiation.
///
/// # Design and safety model
///
/// A [`Tape`] is an append-only list of nodes. Node `i` stores the indices of the nodes it was
/// computed from and the local partial derivatives with respect to each of them, so the reverse
/// sweep is a single backward pass over plain vectors with no pointer chasing and no `unsafe`.
/// Operations whose adjoint is cheaper to compute than to store as a dense Jacobian (matrix
/// multiplication, Cholesky factorization) register a *custom operation* instead: a closure that
/// maps output adjoints to input adjoints.
///
/// A [`Var`] is a 24 byte `Copy` handle: its value, its node index, and a shared reference to its
/// tape. The borrow ties every `Var` to its tape, so the compiler rejects a `Var` that outlives the
/// tape, and [`Tape::clear`] (which takes `&mut self`) cannot run while any `Var` is alive. A
/// `Var` without a tape is a constant; operations on constants create no nodes.
///
/// Combining variables from two different tapes panics. The check is a pointer comparison and is
/// always on, in release builds too. The closure-based drivers ([`grad`](crate::grad) and friends)
/// go further and make most cross-tape mistakes compile errors, because the closure must accept
/// variables of every tape lifetime.
///
/// Adjoints are not stored on the tape. Each sweep writes into a fresh (or caller-provided)
/// buffer, so one tape supports any number of sweeps, for example one per row of a Jacobian.
///
/// `Tape` is `Send` but not `Sync`, so `Var` is neither: each thread uses its own tape.
///
/// These guarantees are checked at compile time:
///
/// ```compile_fail
/// # use probgrad_autodiff::Tape;
/// let mut tape = Tape::new();
/// let x = tape.var(1.0);
/// tape.clear(); // error: `tape` is borrowed by `x`
/// let _ = x;
/// ```
///
/// ```compile_fail
/// # use probgrad_autodiff::Tape;
/// let x = {
///     let tape = Tape::new();
///     tape.var(1.0) // error: `tape` does not live long enough
/// };
/// ```
///
/// ```compile_fail
/// # use probgrad_autodiff::Tape;
/// let tape = Tape::new();
/// let x = tape.var(1.0);
/// std::thread::scope(|s| {
///     s.spawn(move || x * x); // error: `Var` cannot be sent to another thread
/// });
/// ```
///
/// # Example
///
/// ```
/// use probgrad_autodiff::{Real, Tape};
///
/// let tape = Tape::new();
/// let x = tape.var(2.0);
/// let y = tape.var(3.0);
/// let z = x * y + x.sin();
/// let g = z.gradient();
/// assert_eq!(g.wrt(x), 3.0 + 2.0f64.cos());
/// assert_eq!(g.wrt(y), 2.0);
/// ```
#[derive(Default)]
pub struct Tape {
    nodes: RefCell<Nodes>,
}

impl fmt::Debug for Tape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.nodes.borrow();
        f.debug_struct("Tape")
            .field("nodes", &n.edge_end.len())
            .field("edges", &n.parents.len())
            .field("custom_ops", &n.custom.len())
            .finish()
    }
}

#[cold]
#[inline(never)]
fn cross_tape_panic() -> ! {
    panic!("probgrad: operation combines variables from two different tapes")
}

#[cold]
#[inline(never)]
fn capacity_panic() -> ! {
    panic!("probgrad: tape exceeds u32::MAX nodes or edges")
}

#[inline]
fn same_tape<'t>(a: &'t Tape, b: &'t Tape) -> &'t Tape {
    if !core::ptr::eq(a, b) {
        cross_tape_panic();
    }
    a
}

#[inline]
fn to_u32(n: usize) -> u32 {
    u32::try_from(n).unwrap_or_else(|_| capacity_panic())
}

impl Tape {
    /// An empty tape.
    pub fn new() -> Self {
        Tape::default()
    }

    /// An empty tape with room for `nodes` nodes and `edges` edges before reallocating.
    pub fn with_capacity(nodes: usize, edges: usize) -> Self {
        Tape {
            nodes: RefCell::new(Nodes {
                edge_end: Vec::with_capacity(nodes),
                parents: Vec::with_capacity(edges),
                partials: Vec::with_capacity(edges),
                custom: Vec::new(),
            }),
        }
    }

    /// Creates an independent variable with value `x`.
    pub fn var(&self, x: f64) -> Var<'_> {
        Var {
            val: x,
            idx: self.push_node(&[]),
            tape: Some(self),
        }
    }

    /// Creates one independent variable per element of `xs`.
    pub fn vars(&self, xs: &[f64]) -> Vec<Var<'_>> {
        xs.iter().map(|&x| self.var(x)).collect()
    }

    /// Number of nodes recorded so far.
    pub fn len(&self) -> usize {
        self.nodes.borrow().edge_end.len()
    }

    /// True when no nodes have been recorded.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Size statistics of the recorded tape.
    pub fn stats(&self) -> TapeStats {
        let n = self.nodes.borrow();
        TapeStats {
            nodes: n.edge_end.len(),
            edges: n.parents.len(),
            custom_ops: n.custom.len(),
            bytes: n.edge_end.capacity() * 4
                + n.parents.capacity() * 4
                + n.partials.capacity() * 8
                + n.custom.capacity() * core::mem::size_of::<CustomOp>(),
        }
    }

    /// Removes all nodes while keeping the allocated memory for reuse.
    ///
    /// Taking `&mut self` guarantees that no [`Var`] on this tape is still alive.
    pub fn clear(&mut self) {
        let n = self.nodes.get_mut();
        n.edge_end.clear();
        n.parents.clear();
        n.partials.clear();
        n.custom.clear();
    }

    #[inline]
    fn push_node(&self, edges: &[(u32, f64)]) -> u32 {
        let mut n = self.nodes.borrow_mut();
        for &(parent, partial) in edges {
            n.parents.push(parent);
            n.partials.push(partial);
        }
        let end = to_u32(n.parents.len());
        n.edge_end.push(end);
        to_u32(n.edge_end.len() - 1)
    }

    /// Records a custom operation and returns its outputs.
    ///
    /// `rule(output_adjoints, input_adjoints)` must add the contribution of the output adjoints to
    /// `input_adjoints` (which arrive zeroed, one entry per input in order). It runs once per
    /// reverse sweep that reaches the outputs. Use this for operations with a structured adjoint,
    /// such as matrix products, where storing the dense Jacobian would be wasteful.
    ///
    /// Constant inputs are allowed; their adjoints are discarded. If every input is constant the
    /// outputs are constants and `rule` is dropped.
    ///
    /// # Panics
    ///
    /// Panics if an input belongs to a different tape.
    ///
    /// ```
    /// use probgrad_autodiff::Tape;
    ///
    /// let tape = Tape::new();
    /// let x = tape.var(3.0);
    /// // y = (x^2, 2x) with a hand-written adjoint.
    /// let y = tape.custom_op(&[x], &[9.0, 6.0], |out_adj, in_adj| {
    ///     in_adj[0] += out_adj[0] * 2.0 * 3.0 + out_adj[1] * 2.0;
    /// });
    /// let g = (y[0] + y[1]).gradient();
    /// assert_eq!(g.wrt(x), 8.0);
    /// ```
    pub fn custom_op<'t, F>(&'t self, inputs: &[Var<'t>], outputs: &[f64], rule: F) -> Vec<Var<'t>>
    where
        F: Fn(&[f64], &mut [f64]) + Send + 'static,
    {
        let mut any_var = false;
        let input_idx: Vec<u32> = inputs
            .iter()
            .map(|v| match v.tape {
                Some(t) => {
                    same_tape(self, t);
                    any_var = true;
                    v.idx
                }
                None => NO_NODE,
            })
            .collect();
        if !any_var || outputs.is_empty() {
            return outputs.iter().map(|&x| Var::constant(x)).collect();
        }
        let first = to_u32(self.len());
        let out: Vec<Var<'t>> = outputs
            .iter()
            .map(|&x| Var {
                val: x,
                idx: self.push_node(&[]),
                tape: Some(self),
            })
            .collect();
        self.nodes.borrow_mut().custom.push(CustomOp {
            first_output: first,
            n_outputs: to_u32(outputs.len()),
            inputs: input_idx,
            rule: Box::new(rule),
        });
        out
    }

    /// Runs a reverse sweep with the given seeds, writing adjoints of every node into `adj`.
    fn sweep(&self, seeds: &[(u32, f64)], adj: &mut Vec<f64>) {
        let n = self.nodes.borrow();
        adj.clear();
        adj.resize(n.edge_end.len(), 0.0);
        for &(i, w) in seeds {
            adj[i as usize] += w;
        }
        let mut next_custom = n.custom.len();
        let mut out_buf = Vec::new();
        let mut in_buf = Vec::new();
        for i in (0..n.edge_end.len()).rev() {
            while next_custom > 0
                && (n.custom[next_custom - 1].first_output + n.custom[next_custom - 1].n_outputs)
                    as usize
                    == i + 1
            {
                next_custom -= 1;
                let op = &n.custom[next_custom];
                let first = op.first_output as usize;
                let outs = &adj[first..first + op.n_outputs as usize];
                if outs.iter().any(|&a| a != 0.0) {
                    out_buf.clear();
                    out_buf.extend_from_slice(outs);
                    in_buf.clear();
                    in_buf.resize(op.inputs.len(), 0.0);
                    (op.rule)(&out_buf, &mut in_buf);
                    for (&k, &a) in op.inputs.iter().zip(&in_buf) {
                        if k != NO_NODE {
                            adj[k as usize] += a;
                        }
                    }
                }
            }
            let a = adj[i];
            if a == 0.0 {
                continue;
            }
            let start = if i == 0 {
                0
            } else {
                n.edge_end[i - 1] as usize
            };
            let end = n.edge_end[i] as usize;
            for e in start..end {
                adj[n.parents[e] as usize] += n.partials[e] * a;
            }
        }
    }

    /// Gradient of `y` with respect to every node on this tape.
    pub fn gradient<'t>(&'t self, y: Var<'t>) -> Gradients<'t> {
        let mut adjoints = Vec::new();
        self.gradient_into(y, &mut adjoints);
        Gradients {
            adjoints,
            tape: Some(self),
        }
    }

    /// Like [`Tape::gradient`] but reuses `adjoints` as the output buffer.
    ///
    /// After the call `adjoints[i]` is the derivative of `y` with respect to node `i`. Independent
    /// variables created first with [`Tape::vars`] occupy nodes `0..n`.
    pub fn gradient_into<'t>(&'t self, y: Var<'t>, adjoints: &mut Vec<f64>) {
        match y.tape {
            Some(t) => {
                same_tape(self, t);
                self.sweep(&[(y.idx, 1.0)], adjoints);
            }
            None => {
                adjoints.clear();
                adjoints.resize(self.len(), 0.0);
            }
        }
    }

    /// Vector-Jacobian product: the gradient of `sum(weights[i] * outputs[i])`.
    ///
    /// # Panics
    ///
    /// Panics if the slices differ in length or an output belongs to another tape.
    pub fn vjp<'t>(&'t self, outputs: &[Var<'t>], weights: &[f64]) -> Gradients<'t> {
        assert_eq!(
            outputs.len(),
            weights.len(),
            "Tape::vjp: outputs and weights differ in length"
        );
        let seeds: Vec<(u32, f64)> = outputs
            .iter()
            .zip(weights)
            .filter_map(|(v, &w)| {
                v.tape.map(|t| {
                    same_tape(self, t);
                    (v.idx, w)
                })
            })
            .collect();
        let mut adjoints = Vec::new();
        self.sweep(&seeds, &mut adjoints);
        Gradients {
            adjoints,
            tape: Some(self),
        }
    }
}

/// Size statistics of a [`Tape`], from [`Tape::stats`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TapeStats {
    /// Recorded nodes (variables and operation results).
    pub nodes: usize,
    /// Recorded edges (one stored partial derivative each).
    pub edges: usize,
    /// Recorded custom operations.
    pub custom_ops: usize,
    /// Allocated bytes of the node and edge arrays and the custom operation list (excluding the
    /// heap data owned by custom operations).
    pub bytes: usize,
}

/// Adjoints from one reverse sweep.
#[derive(Clone, Debug)]
pub struct Gradients<'t> {
    adjoints: Vec<f64>,
    tape: Option<&'t Tape>,
}

impl<'t> Gradients<'t> {
    /// Derivative with respect to `x`. Zero for constants and for variables created after the
    /// sweep.
    ///
    /// # Panics
    ///
    /// Panics if `x` belongs to a different tape.
    pub fn wrt(&self, x: Var<'t>) -> f64 {
        match (x.tape, self.tape) {
            (None, _) | (_, None) => 0.0,
            (Some(a), Some(b)) => {
                same_tape(a, b);
                self.adjoints.get(x.idx as usize).copied().unwrap_or(0.0)
            }
        }
    }

    /// Derivatives with respect to each element of `xs`.
    pub fn wrt_slice(&self, xs: &[Var<'t>]) -> Vec<f64> {
        xs.iter().map(|&x| self.wrt(x)).collect()
    }
}

/// A reverse-mode variable.
///
/// Create independent variables with [`Tape::var`]; every operation on them records a node on
/// the same tape. See [`Tape`] for the memory and safety model.
#[derive(Clone, Copy)]
pub struct Var<'t> {
    val: f64,
    idx: u32,
    tape: Option<&'t Tape>,
}

impl fmt::Debug for Var<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.tape.is_some() {
            write!(f, "Var({}, node {})", self.val, self.idx)
        } else {
            write!(f, "Var({}, constant)", self.val)
        }
    }
}

impl fmt::Display for Var<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.val, f)
    }
}

impl<'t> Var<'t> {
    /// A constant (no tape, no derivative).
    #[inline]
    pub fn constant(x: f64) -> Self {
        Var {
            val: x,
            idx: NO_NODE,
            tape: None,
        }
    }

    /// True when this value carries no derivative information.
    #[inline]
    pub fn is_constant(&self) -> bool {
        self.tape.is_none()
    }

    /// The tape this variable belongs to, or `None` for a constant.
    #[inline]
    pub fn tape(&self) -> Option<&'t Tape> {
        self.tape
    }

    /// Gradient of this variable with respect to every node on its tape.
    pub fn gradient(&self) -> Gradients<'t> {
        match self.tape {
            Some(t) => t.gradient(*self),
            None => Gradients {
                adjoints: Vec::new(),
                tape: None,
            },
        }
    }

    /// A node with value `value` and precomputed partial derivatives with respect to `operands`.
    ///
    /// This is the reverse-mode analogue of Stan's `operands_and_partials`: one node with one edge
    /// per non-constant operand.
    ///
    /// # Panics
    ///
    /// Panics if the slices differ in length or the operands live on different tapes.
    pub fn from_partials(value: f64, operands: &[Var<'t>], partials: &[f64]) -> Self {
        assert_eq!(
            operands.len(),
            partials.len(),
            "Var::from_partials: operands and partials differ in length"
        );
        let mut tape: Option<&'t Tape> = None;
        let mut edges = Vec::with_capacity(operands.len());
        for (v, &p) in operands.iter().zip(partials) {
            if let Some(t) = v.tape {
                tape = Some(match tape {
                    Some(prev) => same_tape(prev, t),
                    None => t,
                });
                if p != 0.0 {
                    edges.push((v.idx, p));
                }
            }
        }
        match tape {
            Some(t) if !edges.is_empty() => Var {
                val: value,
                idx: t.push_node(&edges),
                tape: Some(t),
            },
            _ => Var::constant(value),
        }
    }

    #[inline]
    fn unary(self, val: f64, d: f64) -> Self {
        match self.tape {
            Some(t) if d != 0.0 => Var {
                val,
                idx: t.push_node(&[(self.idx, d)]),
                tape: Some(t),
            },
            _ => Var::constant(val),
        }
    }

    #[inline]
    fn binary(self, other: Self, val: f64, da: f64, db: f64) -> Self {
        match (self.tape, other.tape) {
            (Some(ta), Some(tb)) => {
                let t = same_tape(ta, tb);
                Var {
                    val,
                    idx: t.push_node(&[(self.idx, da), (other.idx, db)]),
                    tape: Some(t),
                }
            }
            (Some(_), None) => self.unary(val, da),
            (None, Some(_)) => other.unary(val, db),
            (None, None) => Var::constant(val),
        }
    }
}

impl PartialEq for Var<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val
    }
}

impl PartialOrd for Var<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.val.partial_cmp(&other.val)
    }
}

impl PartialEq<f64> for Var<'_> {
    fn eq(&self, other: &f64) -> bool {
        self.val == *other
    }
}

impl PartialOrd<f64> for Var<'_> {
    fn partial_cmp(&self, other: &f64) -> Option<Ordering> {
        self.val.partial_cmp(other)
    }
}

impl<'t> Add for Var<'t> {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        self.binary(o, self.val + o.val, 1.0, 1.0)
    }
}

impl<'t> Sub for Var<'t> {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        self.binary(o, self.val - o.val, 1.0, -1.0)
    }
}

impl<'t> Mul for Var<'t> {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        self.binary(o, self.val * o.val, o.val, self.val)
    }
}

impl<'t> Div for Var<'t> {
    type Output = Self;
    #[inline]
    fn div(self, o: Self) -> Self {
        let q = self.val / o.val;
        self.binary(o, q, 1.0 / o.val, -q / o.val)
    }
}

impl<'t> Neg for Var<'t> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self.unary(-self.val, -1.0)
    }
}

impl<'t> Add<f64> for Var<'t> {
    type Output = Self;
    #[inline]
    fn add(self, c: f64) -> Self {
        self.unary(self.val + c, 1.0)
    }
}

impl<'t> Sub<f64> for Var<'t> {
    type Output = Self;
    #[inline]
    fn sub(self, c: f64) -> Self {
        self.unary(self.val - c, 1.0)
    }
}

impl<'t> Mul<f64> for Var<'t> {
    type Output = Self;
    #[inline]
    fn mul(self, c: f64) -> Self {
        self.unary(self.val * c, c)
    }
}

impl<'t> Div<f64> for Var<'t> {
    type Output = Self;
    #[inline]
    fn div(self, c: f64) -> Self {
        self.unary(self.val / c, 1.0 / c)
    }
}

impl<'t> Add<Var<'t>> for f64 {
    type Output = Var<'t>;
    #[inline]
    fn add(self, x: Var<'t>) -> Var<'t> {
        x + self
    }
}

impl<'t> Sub<Var<'t>> for f64 {
    type Output = Var<'t>;
    #[inline]
    fn sub(self, x: Var<'t>) -> Var<'t> {
        x.unary(self - x.val, -1.0)
    }
}

impl<'t> Mul<Var<'t>> for f64 {
    type Output = Var<'t>;
    #[inline]
    fn mul(self, x: Var<'t>) -> Var<'t> {
        x * self
    }
}

impl<'t> Div<Var<'t>> for f64 {
    type Output = Var<'t>;
    #[inline]
    fn div(self, x: Var<'t>) -> Var<'t> {
        let q = self / x.val;
        x.unary(q, -q / x.val)
    }
}

macro_rules! assign_ops {
    ($($trait:ident $method:ident $op:tt),*) => {
        $(
            impl<'t> $trait for Var<'t> {
                #[inline]
                fn $method(&mut self, o: Self) {
                    *self = *self $op o;
                }
            }

            impl<'t> $trait<f64> for Var<'t> {
                #[inline]
                fn $method(&mut self, o: f64) {
                    *self = *self $op o;
                }
            }
        )*
    };
}

assign_ops!(AddAssign add_assign +, SubAssign sub_assign -, MulAssign mul_assign *, DivAssign div_assign /);

impl<'t> Sum for Var<'t> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let xs: Vec<Self> = iter.collect();
        Real::map_nary(&xs, &crate::primitives::SumN)
    }
}

impl<'t> Real for Var<'t> {
    const ORDER: usize = 1;

    #[inline]
    fn from_f64(x: f64) -> Self {
        Var::constant(x)
    }

    #[inline]
    fn value(&self) -> f64 {
        self.val
    }

    #[inline]
    fn magnitude(&self) -> f64 {
        self.val.abs()
    }

    #[inline]
    fn is_const_zero(&self) -> bool {
        self.tape.is_none() && self.val == 0.0
    }

    #[inline]
    fn map_unary<F: UnaryFn>(self, f: &F) -> Self {
        let y = f.value(self.val);
        if self.tape.is_none() {
            return Var::constant(y);
        }
        self.unary(y, f.derivative(self.val, y))
    }

    #[inline]
    fn map_binary<F: BinaryFn>(self, other: Self, f: &F) -> Self {
        let y = f.value(self.val, other.val);
        match (self.tape, other.tape) {
            (None, None) => Var::constant(y),
            (Some(_), None) => self.unary(y, f.partial_a(self.val, other.val, y)),
            (None, Some(_)) => other.unary(y, f.partial_b(self.val, other.val, y)),
            (Some(_), Some(_)) => self.binary(
                other,
                y,
                f.partial_a(self.val, other.val, y),
                f.partial_b(self.val, other.val, y),
            ),
        }
    }

    fn map_multi<F: MultiFn>(xs: &[Self], f: &F) -> probgrad_core::Result<Vec<Self>> {
        let vals: Vec<f64> = xs.iter().map(|x| x.val).collect();
        let ys = f.eval(&vals)?;
        let Some(tape) = xs.iter().find_map(|x| x.tape) else {
            return Ok(ys.into_iter().map(Var::constant).collect());
        };
        let needed: Vec<bool> = xs.iter().map(|x| x.tape.is_some()).collect();
        let rule = f.clone();
        let y_vals = ys.clone();
        Ok(tape.custom_op(xs, &ys, move |y_adj, x_adj| {
            rule.reverse(&vals, &y_vals, y_adj, &needed, x_adj);
        }))
    }

    fn map_nary<F: NaryFn>(xs: &[Self], f: &F) -> Self {
        let vals: Vec<f64> = xs.iter().map(|x| x.val).collect();
        if xs.iter().all(|x| x.tape.is_none()) {
            return Var::constant(f.value(&vals));
        }
        let needed: Vec<bool> = xs.iter().map(|x| x.tape.is_some()).collect();
        let mut partials = vec![0.0; xs.len()];
        let y = f.value_and_partials(&vals, &needed, &mut partials);
        Var::from_partials(y, xs, &partials)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_record_nothing() {
        let tape = Tape::new();
        let c = Var::constant(2.0);
        let d = c * c + c.exp();
        assert!(d.is_constant());
        assert_eq!(tape.len(), 0);
        let x = tape.var(1.0);
        let _ = x * 0.0;
        assert_eq!(tape.len(), 1, "multiplying by a constant zero adds no node");
    }

    #[test]
    fn repeated_sweeps_start_clean() {
        let tape = Tape::new();
        let x = tape.var(1.5);
        let y = x * x;
        let z = x.exp();
        let g1 = y.gradient();
        let g2 = z.gradient();
        let g1_again = y.gradient();
        assert_eq!(g1.wrt(x), 3.0);
        assert_eq!(g2.wrt(x), 1.5f64.exp());
        assert_eq!(g1_again.wrt(x), 3.0);
    }

    #[test]
    fn clear_reuses_memory() {
        let mut tape = Tape::new();
        for k in 0..3 {
            tape.clear();
            let x = tape.var(f64::from(k));
            let y = x * x * x;
            assert_eq!(y.gradient().wrt(x), 3.0 * f64::from(k * k));
            assert_eq!(tape.len(), 3);
        }
    }

    #[test]
    #[should_panic(expected = "two different tapes")]
    fn cross_tape_panics() {
        let t1 = Tape::new();
        let t2 = Tape::new();
        let a = t1.var(1.0);
        let b = t2.var(2.0);
        let _ = a + b;
    }

    #[test]
    #[should_panic(expected = "two different tapes")]
    fn cross_tape_gradient_lookup_panics() {
        let t1 = Tape::new();
        let t2 = Tape::new();
        let a = t1.var(1.0);
        let b = t2.var(2.0);
        let _ = (a * a).gradient().wrt(b);
    }

    #[test]
    fn custom_op_matches_scalar_ops() {
        let tape = Tape::new();
        let a = tape.var(1.5);
        let b = tape.var(-0.5);
        let c = Var::constant(4.0);
        // outputs: (a*b, a + c)
        let outs = tape.custom_op(&[a, b, c], &[a.val * b.val, a.val + c.val], {
            let (av, bv) = (a.val, b.val);
            move |oa, ia| {
                ia[0] += oa[0] * bv + oa[1];
                ia[1] += oa[0] * av;
                ia[2] += oa[1];
            }
        });
        let y = outs[0] * outs[1];
        let g = y.gradient();
        let y_ref = a * b * (a + c);
        let g_ref = y_ref.gradient();
        assert_eq!(y.val, y_ref.val);
        assert!((g.wrt(a) - g_ref.wrt(a)).abs() < 1e-15);
        assert!((g.wrt(b) - g_ref.wrt(b)).abs() < 1e-15);
        assert_eq!(g.wrt(c), 0.0);
    }

    #[test]
    fn vjp_weights_outputs() {
        let tape = Tape::new();
        let x = tape.var(2.0);
        let outs = [x * x, x * 3.0, Var::constant(7.0)];
        let g = tape.vjp(&outs, &[1.0, 2.0, 5.0]);
        assert_eq!(g.wrt(x), 4.0 + 6.0);
    }

    #[test]
    fn var_is_compact() {
        assert_eq!(core::mem::size_of::<Var<'static>>(), 24);
    }
}
