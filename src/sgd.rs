//! Implementation of stochastic gradient descent.
use loss::*;
use utils::*;
use layers::{LayerUpdates, LayerOut};
use traits::{WeightedLayer, DifferentiableLossFunction, SupervisedTrainer};

use std::collections::LinkedList;

/// Stochastic gradient descent trainer.
pub struct SGDTrainer {
    /// The learning rate
    pub rate: f32,
    /// The number of iterations to train
    pub epochs: usize,
    /// The loss function to use
    pub loss: Box<DifferentiableLossFunction>,
}

impl SGDTrainer {
    pub fn new(epochs: usize, rate: f32) -> SGDTrainer {
        SGDTrainer {
            rate: rate,
            epochs: epochs,
            loss: Box::new(SquaredError),
        }
    }

    fn weight_step(&self, layer: &Box<WeightedLayer>, inputs: &[f32], delta: &[f32]) -> Vec<f32> {
        let mut step = vec!(0.0; layer.weight_count());
        if let Some(derivs) = layer.derivw(inputs) {
            assert_eq!(derivs.len(), step.len());
            assert_eq!(delta.len(), layer.neuron_count());
            // Iterate per neuron and the contributions from later
            // layers.
            for (i, w) in step.iter_mut().enumerate() {
                // Neuron index
                let ni = i / layer.input_count();
                *w -= self.rate * delta[ni] * derivs[i];
            }
        }
        step
    }

    fn bias_step(&self, layer: &Box<WeightedLayer>, delta: &[f32]) -> Vec<f32> {
        let mut step = vec!(0.0; layer.neuron_count());
        // Iterate per neuron bias and contributions from later layers
        for (b, ud) in step.iter_mut().zip(delta) {
            *b -= self.rate * ud;
        }
        step
    }
}

impl SupervisedTrainer for SGDTrainer {
    fn train(&self, layers: &mut LinkedList<Box<WeightedLayer>>, inputs: &[f32], targets: &[f32]) {
        let input_count = layers.front().map(|l| l.input_count()).unwrap_or(0);
        let output_count = layers.back().map(|l| l.output_count()).unwrap_or(0);

        for _ in 0..self.epochs {
            let mut updates: LinkedList<LayerUpdates> = LinkedList::new();
            for l in layers.iter() {
                let ws = vec![0.0; l.weight_count()];
                let bs = vec![0.0; l.neuron_count()];
                updates.push_back(LayerUpdates { ws: ws, bs: bs });
            }

            for (x, t) in inputs.chunks(input_count).zip(targets.chunks(output_count)) {
                // Forward pass
                let mut outputs: LinkedList<LayerOut> = LinkedList::new();
                for l in layers.iter() {
                    let inputs = outputs.back().map_or(x.to_vec(), |o| o.output.clone());
                    let out = l.output(&inputs);
                    outputs.push_back(LayerOut {
                        inputs: inputs,
                        output: out,
                    });
                }

                // Calculate error differential
                let mut delta_signal;
                {
                    let y = outputs.back().map(|o| &o.output).unwrap();
                    delta_signal = self.loss.deriv(y, t);
                }

                // backward pass
                for ((l, lo), lu) in layers.iter_mut()
                    .rev()
                    .zip(outputs.iter().rev())
                    .zip(updates.iter_mut().rev()) {
                    let ws = self.weight_step(&l, &lo.inputs, &delta_signal);
                    add_mut(&mut lu.ws, &ws);

                    let bs = self.bias_step(&l, &delta_signal);
                    add_mut(&mut lu.bs, &bs);

                    delta_signal = l.delta(&delta_signal, &lo.inputs, &lo.output);
                }
            }

            // update batch
            for (l, lu) in layers.iter_mut().zip(updates.iter()) {
                l.update(&lu.ws, &lu.bs);
            }
        }
    }
}
