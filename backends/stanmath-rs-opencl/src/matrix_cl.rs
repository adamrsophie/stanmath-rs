//! Operations on matrices that stay in device memory: Stan's primitive `matrix_cl<double>`
//! functions. Each takes and returns [`MatrixCl`], so a chain of them never copies to the host.

use stanmath_rs_core::{Error, Result};

use crate::device::{Arg, DeviceBuffer, launch};
use crate::elementwise::{binary_source, source};
use crate::{Binary, MatrixCl, Unary};

const STRUCTURE: &str = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void transpose(__global const double* a, __global double* b, const ulong r,
                        const ulong c) {
  const size_t i = get_global_id(0), j = get_global_id(1);
  b[j + i * c] = a[i + j * r];
}
// b = the r2 by c2 block of the r by c matrix a starting at row i0, column j0.
__kernel void block(__global const double* a, __global double* b, const ulong r,
                    const ulong r2, const ulong i0, const ulong j0) {
  const size_t i = get_global_id(0), j = get_global_id(1);
  b[i + j * r2] = a[(i0 + i) + (j0 + j) * r];
}
__kernel void diagonal(__global const double* a, __global double* d, const ulong r) {
  const size_t i = get_global_id(0);
  d[i] = a[i + i * r];
}
__kernel void diag_matrix(__global const double* d, __global double* a, const ulong n) {
  const size_t i = get_global_id(0), j = get_global_id(1);
  a[i + j * n] = i == j ? d[i] : 0.0;
}
__kernel void copy(__global const double* a, __global double* b) {
  const size_t k = get_global_id(0);
  b[k] = a[k];
}
__kernel void add_diag(__global double* a, const ulong r, const double d) {
  const size_t i = get_global_id(0);
  a[i + i * r] += d;
}
// c = [a; b] for a of ra rows and b of rb rows, both with the same columns.
__kernel void append_row(__global const double* a, __global const double* b,
                         __global double* c, const ulong ra, const ulong rb) {
  const size_t i = get_global_id(0), j = get_global_id(1);
  c[i + j * (ra + rb)] = i < ra ? a[i + j * ra] : b[(i - ra) + j * rb];
}
// c = [a, b]: column major, so b's entries follow a's.
__kernel void append_col(__global const double* a, __global const double* b,
                         __global double* c, const ulong na) {
  const size_t k = get_global_id(0);
  c[k] = k < na ? a[k] : b[k - na];
}
__kernel void column_sums(__global const double* a, __global double* s, const ulong r) {
  const size_t j = get_global_id(0);
  double t = 0.0;
  for (size_t i = 0; i < r; i++) t += a[i + j * r];
  s[j] = t;
}
__kernel void zero_upper(__global double* a, const ulong n) {
  const size_t i = get_global_id(0), j = get_global_id(1);
  if (j > i) a[i + j * n] = 0.0;
}
";

fn run(function: &'static str, kernel: &str, args: &[Arg<'_>], global: &[usize]) -> Result<()> {
    launch(
        function,
        "structure",
        kernel,
        || STRUCTURE.into(),
        args,
        global,
    )
}

fn new(rows: usize, cols: usize) -> Result<MatrixCl> {
    Ok(MatrixCl {
        rows,
        cols,
        buffer: DeviceBuffer::new(rows * cols)?,
    })
}

fn copy(function: &'static str, a: &MatrixCl) -> Result<MatrixCl> {
    let b = new(a.rows, a.cols)?;
    run(
        function,
        "copy",
        &[Arg::Buffer(&a.buffer), Arg::Buffer(&b.buffer)],
        &[a.rows * a.cols],
    )?;
    Ok(b)
}

impl MatrixCl {
    /// `aᵀ`. Stan's `transpose`.
    ///
    /// # Errors
    ///
    /// [`Error::Backend`] when a kernel fails.
    pub fn transpose(&self) -> Result<MatrixCl> {
        let b = new(self.cols, self.rows)?;
        run(
            "transpose",
            "transpose",
            &[
                Arg::Buffer(&self.buffer),
                Arg::Buffer(&b.buffer),
                Arg::Size(self.rows),
                Arg::Size(self.cols),
            ],
            &[self.rows, self.cols],
        )?;
        Ok(b)
    }

    /// The `rows` by `cols` block starting at zero-based row `i` and column `j`. Stan's `block`
    /// (one based there); `col`, `row`, `head`, `tail`, and `segment` are blocks too.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] when the block leaves the matrix; [`Error::Backend`] when a
    /// kernel fails.
    pub fn block(&self, i: usize, j: usize, rows: usize, cols: usize) -> Result<MatrixCl> {
        if i + rows > self.rows || j + cols > self.cols {
            return Err(Error::dimension(
                "block",
                format!(
                    "{rows}x{cols} block at ({i}, {j}) of a {}x{} matrix",
                    self.rows, self.cols
                ),
            ));
        }
        let b = new(rows, cols)?;
        run(
            "block",
            "block",
            &[
                Arg::Buffer(&self.buffer),
                Arg::Buffer(&b.buffer),
                Arg::Size(self.rows),
                Arg::Size(rows),
                Arg::Size(i),
                Arg::Size(j),
            ],
            &[rows, cols],
        )?;
        Ok(b)
    }

    /// The main diagonal as a column vector. Stan's `diagonal`.
    ///
    /// # Errors
    ///
    /// [`Error::Backend`] when a kernel fails.
    pub fn diagonal(&self) -> Result<MatrixCl> {
        let n = self.rows.min(self.cols);
        let d = new(n, 1)?;
        run(
            "diagonal",
            "diagonal",
            &[
                Arg::Buffer(&self.buffer),
                Arg::Buffer(&d.buffer),
                Arg::Size(self.rows),
            ],
            &[n],
        )?;
        Ok(d)
    }

    /// The square matrix with the entries of this vector on its diagonal. Stan's `diag_matrix`.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] unless this is a vector; [`Error::Backend`] when a kernel
    /// fails.
    pub fn diag_matrix(&self) -> Result<MatrixCl> {
        if self.rows != 1 && self.cols != 1 {
            return Err(Error::dimension("diag_matrix", "expected a vector"));
        }
        let n = self.rows * self.cols;
        let a = new(n, n)?;
        run(
            "diag_matrix",
            "diag_matrix",
            &[
                Arg::Buffer(&self.buffer),
                Arg::Buffer(&a.buffer),
                Arg::Size(n),
            ],
            &[n, n],
        )?;
        Ok(a)
    }

    /// This matrix with `d` added to its diagonal. Stan's `add_diag`.
    ///
    /// # Errors
    ///
    /// [`Error::Backend`] when a kernel fails.
    pub fn add_diag(&self, d: f64) -> Result<MatrixCl> {
        let a = copy("add_diag", self)?;
        run(
            "add_diag",
            "add_diag",
            &[Arg::Buffer(&a.buffer), Arg::Size(self.rows), Arg::Real(d)],
            &[self.rows.min(self.cols)],
        )?;
        Ok(a)
    }

    /// `[self; b]`, stacking rows. Stan's `append_row`.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] when the column counts differ; [`Error::Backend`] when a
    /// kernel fails.
    pub fn append_row(&self, b: &MatrixCl) -> Result<MatrixCl> {
        if self.cols != b.cols {
            return Err(Error::dimension(
                "append_row",
                format!("{} and {} columns", self.cols, b.cols),
            ));
        }
        let c = new(self.rows + b.rows, self.cols)?;
        run(
            "append_row",
            "append_row",
            &[
                Arg::Buffer(&self.buffer),
                Arg::Buffer(&b.buffer),
                Arg::Buffer(&c.buffer),
                Arg::Size(self.rows),
                Arg::Size(b.rows),
            ],
            &[self.rows + b.rows, self.cols],
        )?;
        Ok(c)
    }

    /// `[self, b]`, joining columns. Stan's `append_col`.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] when the row counts differ; [`Error::Backend`] when a kernel
    /// fails.
    pub fn append_col(&self, b: &MatrixCl) -> Result<MatrixCl> {
        if self.rows != b.rows {
            return Err(Error::dimension(
                "append_col",
                format!("{} and {} rows", self.rows, b.rows),
            ));
        }
        let c = new(self.rows, self.cols + b.cols)?;
        let na = self.rows * self.cols;
        run(
            "append_col",
            "append_col",
            &[
                Arg::Buffer(&self.buffer),
                Arg::Buffer(&b.buffer),
                Arg::Buffer(&c.buffer),
                Arg::Size(na),
            ],
            &[na + b.rows * b.cols],
        )?;
        Ok(c)
    }

    /// The sum of the entries, reduced per column on the device. Stan's `sum`; `mean` is this
    /// over the number of entries.
    ///
    /// # Errors
    ///
    /// [`Error::Backend`] when a kernel fails.
    pub fn sum(&self) -> Result<f64> {
        let s = DeviceBuffer::new(self.cols)?;
        run(
            "sum",
            "column_sums",
            &[
                Arg::Buffer(&self.buffer),
                Arg::Buffer(&s),
                Arg::Size(self.rows),
            ],
            &[self.cols],
        )?;
        Ok(s.to_vec()?.iter().sum())
    }

    /// `op` applied to every entry. Stan's elementwise functions on `matrix_cl`.
    ///
    /// # Errors
    ///
    /// [`Error::Backend`] when a kernel fails.
    pub fn map(&self, op: Unary) -> Result<MatrixCl> {
        let b = new(self.rows, self.cols)?;
        launch(
            op.name(),
            op.name(),
            "value",
            || source(op),
            &[Arg::Buffer(&self.buffer), Arg::Buffer(&b.buffer)],
            &[self.rows * self.cols],
        )?;
        Ok(b)
    }

    /// `op(self[i, j], b[i, j])` for every entry: `add`, `subtract`, `elt_multiply`, and the
    /// rest.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] when the shapes differ; [`Error::Backend`] when a kernel
    /// fails.
    pub fn zip(&self, op: Binary, b: &MatrixCl) -> Result<MatrixCl> {
        if (self.rows, self.cols) != (b.rows, b.cols) {
            return Err(Error::dimension(
                op.name(),
                format!("{}x{} and {}x{}", self.rows, self.cols, b.rows, b.cols),
            ));
        }
        let c = new(self.rows, self.cols)?;
        launch(
            op.name(),
            op.name(),
            "value",
            || binary_source(op),
            &[
                Arg::Buffer(&self.buffer),
                Arg::Buffer(&b.buffer),
                Arg::Buffer(&c.buffer),
            ],
            &[self.rows * self.cols],
        )?;
        Ok(c)
    }

    /// The lower Cholesky factor, computed without leaving the device. Stan's
    /// `cholesky_decompose` on `matrix_cl`. Symmetry is not checked, as only the lower triangle
    /// is read.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] unless square, [`Error::NotPositiveDefinite`], and
    /// [`Error::Backend`] when a kernel fails.
    pub fn cholesky_decompose(&self) -> Result<MatrixCl> {
        use crate::cholesky::{FUNCTION, factor_in_place};
        let n = self.rows;
        if self.cols != n {
            return Err(Error::dimension(FUNCTION, "expected a square matrix"));
        }
        let l = copy(FUNCTION, self)?;
        factor_in_place(&l.buffer, n)?;
        run(
            FUNCTION,
            "zero_upper",
            &[Arg::Buffer(&l.buffer), Arg::Size(n)],
            &[n, n],
        )?;
        if l.diagonal()?
            .buffer
            .to_vec()?
            .iter()
            .any(|&d| d.is_nan() || d <= 0.0)
        {
            return Err(Error::NotPositiveDefinite { function: FUNCTION });
        }
        Ok(l)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Binary, Unary, from_matrix_cl, to_matrix_cl};
    use stanmath_rs_core::Matrix;

    fn m(r: usize, c: usize, start: f64) -> Matrix<f64> {
        Matrix::from_column_major(r, c, (0..r * c).map(|k| start + k as f64).collect())
    }

    fn host(a: &crate::MatrixCl) -> Matrix<f64> {
        from_matrix_cl(a).unwrap()
    }

    #[test]
    fn structural_operations_match_the_host() {
        let a = m(3, 4, 1.0);
        let ac = to_matrix_cl(&a).unwrap();
        assert_eq!(host(&ac.transpose().unwrap()), a.transpose());
        let b = host(&ac.block(1, 2, 2, 2).unwrap());
        assert_eq!(b.as_slice(), &[8.0, 9.0, 11.0, 12.0]);
        assert!(ac.block(2, 0, 2, 1).is_err());
        assert_eq!(host(&ac.diagonal().unwrap()).as_slice(), &[1.0, 5.0, 9.0]);
        let d = host(&ac.diagonal().unwrap().diag_matrix().unwrap());
        assert_eq!(d.as_slice(), &[1.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 9.0]);
        let ad = host(&ac.add_diag(10.0).unwrap());
        assert_eq!(ad.as_slice()[..5], [11.0, 2.0, 3.0, 4.0, 15.0]);
        assert_eq!(host(&ac).as_slice(), a.as_slice(), "add_diag copies");
        let below = to_matrix_cl(&m(2, 4, 20.0)).unwrap();
        let r = host(&ac.append_row(&below).unwrap());
        assert_eq!((r.nrows(), r.ncols()), (5, 4));
        assert_eq!(&r.as_slice()[..5], &[1.0, 2.0, 3.0, 20.0, 21.0]);
        let right = to_matrix_cl(&m(3, 1, 50.0)).unwrap();
        let c = host(&ac.append_col(&right).unwrap());
        assert_eq!(&c.as_slice()[12..], &[50.0, 51.0, 52.0]);
        assert!(ac.append_col(&below).is_err());
        assert_eq!(ac.sum().unwrap(), 78.0);
    }

    #[test]
    fn elementwise_and_cholesky_stay_on_the_device() {
        let a = m(2, 3, 0.5);
        let ac = to_matrix_cl(&a).unwrap();
        let e = host(&ac.map(Unary::Exp).unwrap());
        for (x, y) in a.as_slice().iter().zip(e.as_slice()) {
            assert!((x.exp() - y).abs() <= 1e-14 * y);
        }
        let s = host(&ac.zip(Binary::Add, &ac).unwrap());
        assert_eq!(s.as_slice(), &[1.0, 3.0, 5.0, 7.0, 9.0, 11.0]);
        assert!(ac.zip(Binary::Add, &ac.transpose().unwrap()).is_err());
        let spd = Matrix::from_column_major(2, 2, vec![4.0, 2.0, 2.0, 3.0]);
        let l = host(&to_matrix_cl(&spd).unwrap().cholesky_decompose().unwrap());
        let want = [2.0, 1.0, 0.0, 2f64.sqrt()];
        for (x, y) in l.as_slice().iter().zip(want) {
            assert!((x - y).abs() <= 1e-14);
        }
        let bad = Matrix::from_column_major(2, 2, vec![1.0, 2.0, 2.0, 1.0]);
        assert!(to_matrix_cl(&bad).unwrap().cholesky_decompose().is_err());
    }
}
