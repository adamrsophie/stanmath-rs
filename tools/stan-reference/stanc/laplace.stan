// Compiled by stanc in the probe workflow to show the C++ calls of the Laplace marginals.
functions {
  matrix k_f(array[] vector x, real alpha, real rho) {
    return add_diag(gp_exp_quad_cov(x, alpha, rho), 1e-8);
  }
}
data {
  int n;
  array[n] int y;
  array[n] int idx;
  array[n] vector[1] x;
}
parameters {
  real<lower=0> alpha;
  real<lower=0> rho;
  real<lower=0> eta;
}
model {
  target += laplace_marginal_poisson_log_lupmf(y | idx, rep_vector(0, n), k_f, (x, alpha, rho));
  target += laplace_marginal_poisson_log_lpmf(y | idx, rep_vector(0, n), k_f, (x, alpha, rho));
  target += laplace_marginal_neg_binomial_2_log_lupmf(y | idx, eta, rep_vector(0, n), k_f, (x, alpha, rho));
  target += laplace_marginal_bernoulli_logit_lupmf(y | idx, rep_vector(0, n), k_f, (x, alpha, rho));
}
