//! OpenCL C special functions that the device library lacks, shared by every program.

/// The fp64 pragma, then digamma, trigamma, tetragamma, and inv_Phi.
pub(crate) const PRELUDE: &str = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// Polygamma functions: reflection below 0.5, the recurrence up to a bound, then the asymptotic
// series.
double digamma(double x) {
  if (x < 0.5) return digamma(1.0 - x) - M_PI / tan(M_PI * x);
  double r = 0.0;
  while (x < 10.0) { r -= 1.0 / x; x += 1.0; }
  const double f = 1.0 / (x * x);
  return r + log(x) - 0.5 / x
         - f * (1.0 / 12.0 - f * (1.0 / 120.0 - f * (1.0 / 252.0 - f * (1.0 / 240.0
         - f * (1.0 / 132.0)))));
}
double trigamma(double x) {
  if (x < 0.5) {
    const double s = sin(M_PI * x);
    return M_PI * M_PI / (s * s) - trigamma(1.0 - x);
  }
  double r = 0.0;
  while (x < 20.0) { r += 1.0 / (x * x); x += 1.0; }
  const double f = 1.0 / (x * x);
  return r + 1.0 / x + 0.5 * f
         + f / x * (1.0 / 6.0 - f * (1.0 / 30.0 - f * (1.0 / 42.0 - f * (1.0 / 30.0))));
}
double tetragamma(double x) {
  if (x < 0.5) {
    const double s = sin(M_PI * x);
    return tetragamma(1.0 - x) - 2.0 * M_PI * M_PI * M_PI * cos(M_PI * x) / (s * s * s);
  }
  double r = 0.0;
  while (x < 20.0) { r -= 2.0 / (x * x * x); x += 1.0; }
  const double f = 1.0 / (x * x);
  return r - f - f / x - f * f * (0.5 - f * (1.0 / 6.0 - f * (1.0 / 6.0 - f * (0.3
         - f * (5.0 / 6.0)))));
}
// Inverse of the standard normal CDF: a rational first guess, then two Halley steps.
double inv_Phi(double p) {
  if (p <= 0.0) return -INFINITY;
  if (p >= 1.0) return INFINITY;
  double x;
  if (p < 0.02425 || p > 0.97575) {
    const double q = sqrt(-2.0 * log(p < 0.5 ? p : 1.0 - p));
    x = (((((-7.784894002430293e-03 * q - 3.223964580411365e-01) * q - 2.400758277161838e+00) * q
          - 2.549732539343734e+00) * q + 4.374664141464968e+00) * q + 2.938163982698783e+00)
        / ((((7.784695709041462e-03 * q + 3.224671290700398e-01) * q + 2.445134137142996e+00) * q
          + 3.754408661907416e+00) * q + 1.0);
    if (p > 0.5) x = -x;
  } else {
    const double q = p - 0.5;
    const double r = q * q;
    x = (((((-3.969683028665376e+01 * r + 2.209460984245205e+02) * r - 2.759285104469687e+02) * r
          + 1.383577518672690e+02) * r - 3.066479806614716e+01) * r + 2.506628277459239e+00) * q
        / (((((-5.447609879822406e+01 * r + 1.615858368580409e+02) * r - 1.556989798598866e+02) * r
          + 6.680131188771972e+01) * r - 1.328068155288572e+01) * r + 1.0);
  }
  for (int k = 0; k < 2; k++) {
    const double e = 0.5 * erfc(-x * M_SQRT1_2) - p;
    const double u = e * 2.5066282746310002 * exp(0.5 * x * x);
    x -= u / (1.0 + 0.5 * x * u);
  }
  return x;
}
";
