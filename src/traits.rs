//! The traits that make up neural network.
use std::collections::LinkedList;

/// A single layer in a neural network.
pub trait Layer {
    /// Expected number of inputs.
    fn input_count(&self) -> usize;
    /// Expected number of outputs.
    fn output_count(&self) -> usize;

    /// Output of the layer.
    fn output(&self, &[f32]) -> Vec<f32>;

    /// Propagates the delta signal through this layer. Multiplies the
    /// signal with the derivative of the layer with respect to its
    /// inputs. Returns a vector of shape (inputs,) where the neurons'
    /// contributions to the delta signal have been summed up and
    /// where the delta signal has shape (neurons,). This function
    /// takes the output of the layer and is suitable when the
    /// derivative of the layer is easily expressed in terms of
    /// itself.
    #[allow(unused_variables)]
    fn delta_from_outputs(&self, delta: &[f32], outputs: &[f32]) -> Option<Vec<f32>> {
        None
    }

    /// Propagates the delta signal through this layer. Multiplies the
    /// signal with the derivative of the layer with respect to its
    /// inputs. Returns a vector of shape (inputs,) where the neurons'
    /// contributions to the delta signal have been summed up and
    /// where the delta signal has shape (neurons,). This function
    /// takes the input to the layer and is suitable when the
    /// derivative of the layer is easily expressed in terms of its
    /// inputs.
    #[allow(unused_variables)]
    fn delta_from_inputs(&self, delta: &[f32], inputs: &[f32]) -> Option<Vec<f32>> {
        None
    }

    /// Derivative of the layer with respect to its inputs. Used for
    /// chain differentiation. Will panic if the layer doesn't
    /// implement a suitable delta function.
    fn delta(&self, delta: &[f32], inputs: &[f32], outputs: &[f32]) -> Vec<f32> {
        self.delta_from_outputs(delta, outputs).or(self.delta_from_inputs(delta, inputs)).unwrap()
    }

    /// Derivative of the layer with respect to its weights. The input
    /// argument is the input to the layer. Returns None if not
    /// implemented for this layer.
    fn derivw(&self, &[f32]) -> Option<Vec<f32>> {
        None
    }
}

/// A layer containing weights which can be trained.
pub trait WeightedLayer: Layer {
    fn weight_count(&self) -> usize;
    fn neuron_count(&self) -> usize;
    fn weights_mut(&mut self) -> Option<&mut Vec<f32>>;
    fn bias_mut(&mut self) -> Option<&mut Vec<f32>>;
    fn update(&mut self, weight_updates: &[f32], bias_updates: &[f32]) {
        if let Some(weights) = self.weights_mut() {
            for (w, dw) in weights.iter_mut().zip(weight_updates) {
                *w += *dw;
            }
        }
        if let Some(biases) = self.bias_mut() {
            for (b, db) in biases.iter_mut().zip(bias_updates) {
                *b += *db;
            }
        }
    }
}

/// A loss function - also known as an error function.
pub trait LossFunction {
    /// Loss, or error, for a single prediction vs target.
    fn loss1(&self, f32, f32) -> f32;

    /// The loss, or error, of the predictions vs the targets.
    fn loss(&self, preds: &[f32], targets: &[f32]) -> Vec<f32> {
        let mut loss = Vec::new();
        for (p, t) in preds.iter().zip(targets) {
            loss.push(self.loss1(*p, *t));
        }
        loss
    }
}

/// A loss function which can be differentiated.
pub trait DifferentiableLossFunction: LossFunction {
    /// The derivative of a single loss value with respect to the
    /// prediction.
    fn deriv1(&self, f32, f32) -> f32;

    /// The derivative of the loss with respect to the predictions.
    fn deriv(&self, preds: &[f32], targets: &[f32]) -> Vec<f32> {
        let mut derivs = Vec::new();
        for (p, t) in preds.iter().zip(targets) {
            derivs.push(self.deriv1(*p, *t));
        }
        derivs
    }
}

/// A training algorithm for a neural network.
pub trait SupervisedTrainer {
    fn train(&self, layers: &mut LinkedList<Box<WeightedLayer>>, inputs: &[f32], targets: &[f32]);
}
