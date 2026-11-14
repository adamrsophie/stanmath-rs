//! Reverse-mode gradients, on an explicit tape and through the closure drivers.
//!
//! Run with `cargo run --example reverse_mode`.

use stanmath_rs::prelude::*;

fn main() {
    // Explicit tape: build the expression, then sweep backward once.
    let tape = Tape::new();
    let x = tape.var(1.5);
    let y = tape.var(-0.5);
    let z = (x * y).sin() + x.exp() * y.square();
    let g = z.gradient();
    println!("z = {:.12}", z.value());
    println!("dz/dx = {:.12}, dz/dy = {:.12}", g.wrt(x), g.wrt(y));

    // Closure driver: the tape is created and dropped for you.
    let r = value_and_grad(
        |p| (p[0] * p[1]).sin() + p[0].exp() * p[1].square(),
        &[1.5, -0.5],
    );
    assert_eq!(r.gradient, vec![g.wrt(x), g.wrt(y)]);
    println!("value_and_grad: {r:?}");

    // Repeated evaluation without reallocating the tape, as a sampler would do.
    let mut workspace = GradientWorkspace::new();
    let mut gradient = [0.0; 2];
    for step in 0..3 {
        let position = [1.5 + 0.1 * f64::from(step), -0.5];
        let value = workspace.value_and_grad_into(
            |p| (p[0] * p[1]).sin() + p[0].exp() * p[1].square(),
            &position,
            &mut gradient,
        );
        println!("step {step}: value {value:.6}, gradient {gradient:?}");
    }

    // Vector-Jacobian product: one sweep for a weighted sum of outputs.
    let (values, wj) = vjp(
        |p| vec![p[0] * p[1], p[0] + p[1]],
        &[2.0, 3.0],
        &[1.0, 10.0],
    );
    println!("outputs {values:?}, w^T J = {wj:?}");
}
