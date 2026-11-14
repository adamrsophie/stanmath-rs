#!/usr/bin/env bash
# Compiles one small Stan model per candidate Laplace call and reports which ones stanc accepts,
# with the C++ that stanc generates for each accepted call.
set -u
header='functions {
  matrix k_f(array[] vector x, real alpha, real rho) {
    return add_diag(gp_exp_quad_cov(x, alpha, rho), 1e-8);
  }
  real ll_f(vector theta, array[] int y, array[] int idx) {
    return poisson_log_lpmf(y | theta[idx]);
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
model {'
m='rep_vector(0, n)'
k='k_f, (x, alpha, rho)'
o='generate_laplace_options(n)'
calls=(
  "laplace_marginal_neg_binomial_2_log_lupmf(y | idx, rep_vector(eta, n), $m, 1, $k)"
  "laplace_marginal_neg_binomial_2_log_lupmf(y | idx, $m, eta, 1, $k)"
  "laplace_marginal_neg_binomial_2_log_lupmf(y | idx, $m, rep_vector(eta, n), 1, $k)"
  "laplace_marginal_tol_neg_binomial_2_log_lupmf(y | idx, rep_vector(eta, n), $m, 1, $k, $o)"
)
i=0
for c in "${calls[@]}"; do
  i=$((i + 1))
  printf '%s\n  target += %s;\n}\n' "$header" "$c" > "v$i.stan"
  if ./stanc --o "v$i.hpp" "v$i.stan" > "v$i.err" 2>&1; then
    echo "OK   $i: $c"
    grep -n "laplace_marginal" "v$i.hpp" | grep -v "^.*//" | head -8
  else
    echo "FAIL $i: $c"
    grep -vE "^\s*[0-9]+:|^\s*-+$|^\s*\^$|^$" "v$i.err" | head -4
  fi
done
