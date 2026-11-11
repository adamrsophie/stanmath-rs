//! Matrix primitives: one generic algorithm and one `f64` reverse rule each.
//!
//! All matrices are column-major: element `(i, j)` of an `r` by `c` matrix is at `j * r + i`.

use probgrad_autodiff::{MultiFn, Real};
use probgrad_core::{Error, Result};

#[inline]
fn at(nrows: usize, i: usize, j: usize) -> usize {
    j * nrows + i
}

/// Absolute tolerance of the symmetry check, as in Stan Math's `check_symmetric`.
pub(crate) const SYMMETRY_TOL: f64 = 1e-8;

/// `C = A B` with `A` of size `m` by `k` and `B` of size `k` by `n`; inputs `[A, B]`.
///
/// Reverse rule: `Ā += C̄ Bᵀ`, `B̄ += Aᵀ C̄`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Multiply {
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

impl MultiFn for Multiply {
    fn eval<S: Real>(&self, x: &[S]) -> Result<Vec<S>> {
        let (m, k, n) = (self.m, self.k, self.n);
        let (a, b) = x.split_at(m * k);
        let mut c = vec![S::zero(); m * n];
        for j in 0..n {
            for l in 0..k {
                let blj = b[at(k, l, j)];
                for i in 0..m {
                    c[at(m, i, j)] += a[at(m, i, l)] * blj;
                }
            }
        }
        Ok(c)
    }

    fn reverse(&self, x: &[f64], _y: &[f64], c_adj: &[f64], needed: &[bool], x_adj: &mut [f64]) {
        let (m, k, n) = (self.m, self.k, self.n);
        let (a, b) = x.split_at(m * k);
        let (a_adj, b_adj) = x_adj.split_at_mut(m * k);
        let (a_needed, b_needed) = needed.split_at(m * k);
        if a_needed.iter().any(|&v| v) {
            for l in 0..k {
                for j in 0..n {
                    let blj = b[at(k, l, j)];
                    for i in 0..m {
                        a_adj[at(m, i, l)] += c_adj[at(m, i, j)] * blj;
                    }
                }
            }
        }
        if b_needed.iter().any(|&v| v) {
            for j in 0..n {
                for l in 0..k {
                    let mut s = 0.0;
                    for i in 0..m {
                        s += a[at(m, i, l)] * c_adj[at(m, i, j)];
                    }
                    b_adj[at(k, l, j)] += s;
                }
            }
        }
    }
}

/// Lower Cholesky factor `L` with `A = L Lᵀ`; input `A` (`n` by `n`), output `L` with zeros above
/// the diagonal.
///
/// Only the lower triangle of `A` is read; symmetry is checked on the values. Accordingly the
/// derivative flows to the lower triangle only (Murray 2016, "Differentiation of the Cholesky
/// decomposition"): with `Φ` taking the lower triangle and halving the diagonal,
/// `S = L⁻ᵀ Φ(Lᵀ L̄) L⁻¹` and `Ā = Φ(S + Sᵀ)`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Cholesky {
    pub n: usize,
}

pub(crate) fn check_symmetric(function: &'static str, a: &[f64], n: usize) -> Result<()> {
    for j in 0..n {
        for i in j + 1..n {
            let (lo, hi) = (a[at(n, i, j)], a[at(n, j, i)]);
            if !((lo - hi).abs() <= SYMMETRY_TOL) {
                return Err(Error::NotSymmetric {
                    function,
                    row: i,
                    col: j,
                });
            }
        }
    }
    Ok(())
}

impl MultiFn for Cholesky {
    fn eval<S: Real>(&self, a: &[S]) -> Result<Vec<S>> {
        const F: &str = "cholesky_decompose";
        let n = self.n;
        let values: Vec<f64> = a.iter().map(Real::value).collect();
        check_symmetric(F, &values, n)?;
        let mut l = vec![S::zero(); n * n];
        for j in 0..n {
            let mut d = a[at(n, j, j)];
            for k in 0..j {
                d -= l[at(n, j, k)].square();
            }
            if !(d.value() > 0.0) || !d.is_finite() {
                return Err(Error::NotPositiveDefinite { function: F });
            }
            let ljj = d.sqrt();
            l[at(n, j, j)] = ljj;
            for i in j + 1..n {
                let mut s = a[at(n, i, j)];
                for k in 0..j {
                    s -= l[at(n, i, k)] * l[at(n, j, k)];
                }
                l[at(n, i, j)] = s / ljj;
            }
        }
        Ok(l)
    }

    fn reverse(&self, _a: &[f64], l: &[f64], l_adj: &[f64], _needed: &[bool], a_adj: &mut [f64]) {
        let n = self.n;
        // P = Φ(Lᵀ L̄), using only the lower triangle of L̄ (L is constant above the diagonal).
        let mut p = vec![0.0; n * n];
        for q in 0..n {
            for r in q..n {
                // (Lᵀ L̄)[r, q] = sum_s L[s, r] L̄[s, q] over s >= max(r, q) = r
                let mut g = 0.0;
                for s in r..n {
                    g += l[at(n, s, r)] * l_adj[at(n, s, q)];
                }
                p[at(n, r, q)] = if r == q { 0.5 * g } else { g };
            }
        }
        // W = P L⁻¹: each row w of W solves Lᵀ wᵀ = pᵀ.
        let mut w = vec![0.0; n * n];
        let mut row = vec![0.0; n];
        for r in 0..n {
            for c in 0..n {
                row[c] = p[at(n, r, c)];
            }
            solve_lower_transpose(l, n, &mut row);
            for c in 0..n {
                w[at(n, r, c)] = row[c];
            }
        }
        // S = L⁻ᵀ W, column by column.
        for c in 0..n {
            let col = &mut w[c * n..(c + 1) * n];
            solve_lower_transpose(l, n, col);
        }
        let s = w;
        for j in 0..n {
            a_adj[at(n, j, j)] += s[at(n, j, j)];
            for i in j + 1..n {
                a_adj[at(n, i, j)] += s[at(n, i, j)] + s[at(n, j, i)];
            }
        }
    }
}

/// Solves `L z = r` in place for lower triangular `L` (forward substitution).
fn solve_lower(l: &[f64], n: usize, r: &mut [f64]) {
    for i in 0..n {
        let mut s = r[i];
        for k in 0..i {
            s -= l[at(n, i, k)] * r[k];
        }
        r[i] = s / l[at(n, i, i)];
    }
}

/// Solves `Lᵀ z = r` in place for lower triangular `L` (backward substitution).
fn solve_lower_transpose(l: &[f64], n: usize, r: &mut [f64]) {
    for i in (0..n).rev() {
        let mut s = r[i];
        for k in i + 1..n {
            s -= l[at(n, k, i)] * r[k];
        }
        r[i] = s / l[at(n, i, i)];
    }
}

fn check_diagonal<S: Real>(function: &'static str, l: &[S], n: usize) -> Result<()> {
    for i in 0..n {
        let d = l[at(n, i, i)].value();
        if d == 0.0 || !d.is_finite() {
            return Err(Error::parameter(
                function,
                "L",
                d,
                "a nonzero finite diagonal",
            ));
        }
    }
    Ok(())
}

/// `X = L⁻¹ B` (`transpose = false`) or `X = L⁻ᵀ B` (`transpose = true`) for lower triangular `L`
/// (`n` by `n`) and `B` (`n` by `m`); inputs `[L, B]`. Only the lower triangle of `L` is read.
///
/// Reverse rules: for `L X = B`, `B̄ = L⁻ᵀ X̄` and `L̄ = -tril(B̄ Xᵀ)`; for `Lᵀ X = B`,
/// `B̄ = L⁻¹ X̄` and `L̄ = -tril(X B̄ᵀ)`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TriangularSolve {
    pub n: usize,
    pub m: usize,
    pub transpose: bool,
}

impl MultiFn for TriangularSolve {
    fn eval<S: Real>(&self, x: &[S]) -> Result<Vec<S>> {
        let (n, m) = (self.n, self.m);
        let (l, b) = x.split_at(n * n);
        check_diagonal("mdivide_left_tri_low", l, n)?;
        let mut out = b.to_vec();
        for c in 0..m {
            let col = &mut out[c * n..(c + 1) * n];
            if self.transpose {
                for i in (0..n).rev() {
                    let mut s = col[i];
                    for k in i + 1..n {
                        s -= l[at(n, k, i)] * col[k];
                    }
                    col[i] = s / l[at(n, i, i)];
                }
            } else {
                for i in 0..n {
                    let mut s = col[i];
                    for k in 0..i {
                        s -= l[at(n, i, k)] * col[k];
                    }
                    col[i] = s / l[at(n, i, i)];
                }
            }
        }
        Ok(out)
    }

    fn reverse(&self, x: &[f64], xs: &[f64], x_bar: &[f64], needed: &[bool], adj: &mut [f64]) {
        let (n, m) = (self.n, self.m);
        let l = &x[..n * n];
        // B̄ = L⁻ᵀ X̄ or L⁻¹ X̄
        let mut b_bar = x_bar.to_vec();
        for c in 0..m {
            let col = &mut b_bar[c * n..(c + 1) * n];
            if self.transpose {
                solve_lower(l, n, col);
            } else {
                solve_lower_transpose(l, n, col);
            }
        }
        let (l_adj, b_adj) = adj.split_at_mut(n * n);
        let (l_needed, b_needed) = needed.split_at(n * n);
        if b_needed.iter().any(|&v| v) {
            for (a, v) in b_adj.iter_mut().zip(&b_bar) {
                *a += v;
            }
        }
        if l_needed.iter().any(|&v| v) {
            for j in 0..n {
                for i in j..n {
                    let mut s = 0.0;
                    for c in 0..m {
                        s += if self.transpose {
                            xs[at(n, i, c)] * b_bar[at(n, j, c)]
                        } else {
                            b_bar[at(n, i, c)] * xs[at(n, j, c)]
                        };
                    }
                    l_adj[at(n, i, j)] -= s;
                }
            }
        }
    }
}
