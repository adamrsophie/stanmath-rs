//! Property-based tests: identities that must hold for random inputs.

use proptest::prelude::*;
use stanmath_rs_autodiff::functions::*;
use stanmath_rs_autodiff::*;

fn rel_close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
}

fn composite<T: Real>(x: &[T]) -> T {
    let a = x[0].sin() * x[1].exp() + log1p_exp(x[2] * x[0]);
    let b = (x[1].square() + 1.0).ln() / (x[2].tanh() + 2.0);
    a * b + inv_logit(x[0] - x[1]) + log_sum_exp(x[2], x[0])
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn reverse_equals_forward(x0 in -3.0..3.0f64, x1 in -3.0..3.0f64, x2 in -3.0..3.0f64) {
        let x = [x0, x1, x2];
        let g = grad(|v| composite(v), &x);
        let (_, j) = jacobian_forward(|v| vec![composite(v)], &x);
        for k in 0..3 {
            prop_assert!(rel_close(g[k], j[(0, k)], 1e-13), "{} vs {}", g[k], j[(0, k)]);
        }
    }

    #[test]
    fn hessian_modes_agree_and_are_symmetric(x0 in -2.0..2.0f64, x1 in -2.0..2.0f64, x2 in -2.0..2.0f64) {
        let x = [x0, x1, x2];
        let h = hessian(|v| composite(v), &x);
        let hf = hessian_forward(composite, &x);
        for i in 0..3 {
            for k in 0..3 {
                prop_assert!(rel_close(h[(i, k)], hf[(i, k)], 1e-12));
                prop_assert!(rel_close(h[(i, k)], h[(k, i)], 1e-12));
            }
        }
    }

    #[test]
    fn inverse_function_identities(x in -30.0..30.0f64, y in -15.0..15.0f64) {
        // d/dx f(g(x)) = 1 where f and g are inverses. logit(inv_logit(x)) is only well
        // conditioned for moderate |x|: beyond that, rounding p = inv_logit(x) to a double
        // already destroys the digits of 1 - p, whatever the differentiation method.
        prop_assert!(rel_close(derivative(|t| logit(inv_logit(t)), y), 1.0, 1e-6));
        prop_assert!(rel_close(derivative(|t| log(exp(t)), x), 1.0, 1e-14));
        prop_assert!(rel_close(derivative(|t| asinh(sinh(t * 0.1)), x), 0.1, 1e-12));
    }

    #[test]
    fn log_sum_exp_gradient_is_softmax(a in -50.0..50.0f64, b in -50.0..50.0f64, c in -50.0..50.0f64) {
        let x = [a, b, c];
        let g = grad(|v| log_sum_exp_slice(v), &x);
        let s = softmax(&x);
        let total: f64 = g.iter().sum();
        prop_assert!(rel_close(total, 1.0, 1e-14));
        for k in 0..3 {
            prop_assert!(rel_close(g[k], s[k], 1e-14));
        }
    }

    #[test]
    fn tape_reuse_is_deterministic(x0 in -2.0..2.0f64, x1 in -2.0..2.0f64, x2 in -2.0..2.0f64) {
        let x = [x0, x1, x2];
        let mut ws = GradientWorkspace::new();
        let mut g1 = [0.0; 3];
        let mut g2 = [0.0; 3];
        let v1 = ws.value_and_grad_into(|v| composite(v), &x, &mut g1);
        let v2 = ws.value_and_grad_into(|v| composite(v), &x, &mut g2);
        prop_assert_eq!(v1, v2);
        prop_assert_eq!(g1, g2);
    }
}
