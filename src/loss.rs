//! Loss functions for training the networks.
use traits::{LossFunction, DifferentiableLossFunction};

/// The square error is defined as `e = (y - t)^2`, with derivative
/// `de/dy = 2 * (y - t)`.
pub struct SquaredError;

impl LossFunction for SquaredError {
    fn loss1(self: &SquaredError, pred: f32, target: f32) -> f32 {
        (pred - target) * (pred - target)
    }
}

impl DifferentiableLossFunction for SquaredError {
    fn deriv1(self: &SquaredError, pred: f32, target: f32) -> f32 {
        2.0 * (pred - target)
    }
}
