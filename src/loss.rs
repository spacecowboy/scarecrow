use traits::{LossFunction, DifferentiableLossFunction};

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
