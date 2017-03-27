//! Implementation of different kinds of layers.
use super::traits::{Layer, WeightedLayer};
use super::utils::{dot, normal_vector};

pub struct LayerOut {
    pub inputs: Vec<f32>,
    pub output: Vec<f32>,
}

pub struct LayerUpdates {
    pub ws: Vec<f32>,
    pub bs: Vec<f32>,
}

pub struct DenseLayer {
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    /// (inputs per neuron, number of neurons)
    pub shape: (usize, usize),
}

impl DenseLayer {
    pub fn uniform(val: f32, inputs: usize, neurons: usize) -> DenseLayer {
        DenseLayer {
            weights: vec!(val; inputs * neurons),
            bias: vec!(val; neurons),
            shape: (inputs, neurons),
        }
    }

    pub fn random(inputs: usize, neurons: usize) -> DenseLayer {
        DenseLayer {
            weights: normal_vector(inputs * neurons),
            bias: normal_vector(neurons),
            shape: (inputs, neurons),
        }
    }
}

impl Layer for DenseLayer {
    fn input_count(self: &DenseLayer) -> usize {
        self.shape.0
    }

    fn output_count(self: &DenseLayer) -> usize {
        self.shape.1
    }

    /// Output of this layer is a vector of weight and input dot products.
    fn output(self: &DenseLayer, inputs: &[f32]) -> Vec<f32> {
        assert_eq!(self.shape.0, inputs.len());
        let neuron_weights = self.weights.chunks(self.shape.0);
        let mut out: Vec<f32> = Vec::new();
        for (i, w) in neuron_weights.enumerate() {
            out.push(dot(w, inputs) + self.bias[i]);
        }

        out
    }

    fn delta_from_inputs(self: &DenseLayer, delta: &[f32], inputs: &[f32]) -> Option<Vec<f32>> {
        assert_eq!(self.shape.0, inputs.len());
        assert_eq!(self.shape.1, delta.len());
        let mut result: Vec<f32> = vec!(0.0; self.shape.0);

        let neuron_weights = self.weights.chunks(self.shape.0);

        for (d, nw) in delta.iter().zip(neuron_weights) {
            for (i, w) in nw.iter().enumerate() {
                result[i] += d * w;
            }
        }

        Some(result)
    }

    /// Vector of derivatives with respect to the weights for each
    /// neuron. Returns a vector with the same logical dimensions as
    /// the layer's shape.
    fn derivw(self: &DenseLayer, inputs: &[f32]) -> Option<Vec<f32>> {
        assert_eq!(self.shape.0, inputs.len());
        let mut derivs: Vec<f32> = Vec::new();
        derivs.reserve(self.shape.0 * self.shape.1);

        for _ in 0..self.shape.1 {
            for i in inputs {
                derivs.push(*i);
            }
        }
        Some(derivs)
    }
}

impl WeightedLayer for DenseLayer {
    fn weight_count(self: &DenseLayer) -> usize {
        self.weights.len()
    }

    fn neuron_count(self: &DenseLayer) -> usize {
        self.output_count()
    }
    fn weights_mut(self: &mut DenseLayer) -> Option<&mut Vec<f32>> {
        Some(&mut self.weights)
    }

    fn bias_mut(self: &mut DenseLayer) -> Option<&mut Vec<f32>> {
        Some(&mut self.bias)
    }
}

pub struct HyperbolicLayer {
    pub size: usize,
}

impl WeightedLayer for HyperbolicLayer {
    fn weight_count(&self) -> usize {
        0
    }
    fn neuron_count(&self) -> usize {
        0
    }
    fn weights_mut(self: &mut HyperbolicLayer) -> Option<&mut Vec<f32>> {
        None
    }

    fn bias_mut(self: &mut HyperbolicLayer) -> Option<&mut Vec<f32>> {
        None
    }
}

impl Layer for HyperbolicLayer {
    fn input_count(self: &HyperbolicLayer) -> usize {
        self.size
    }

    fn output_count(self: &HyperbolicLayer) -> usize {
        self.size
    }

    fn output(self: &HyperbolicLayer, inputs: &[f32]) -> Vec<f32> {
        let mut out: Vec<f32> = Vec::new();
        for x in inputs {
            out.push(x.tanh());
        }
        out
    }

    /// y = tanh(x) and dy / dx = 1 - y^2
    fn delta_from_outputs(self: &HyperbolicLayer,
                          delta: &[f32],
                          outputs: &[f32])
                          -> Option<Vec<f32>> {
        assert_eq!(self.size, outputs.len());
        assert_eq!(self.size, delta.len());
        let mut derivs: Vec<f32> = vec![0.0; self.size];
        for ((d, y), yd) in delta.iter().zip(outputs).zip(derivs.iter_mut()) {
            *yd = d * (1.0 - y * y);
        }
        Some(derivs)
    }
}

pub struct SigmoidLayer {
    pub size: usize,
}

impl Layer for SigmoidLayer {
    fn input_count(self: &SigmoidLayer) -> usize {
        self.size
    }

    fn output_count(self: &SigmoidLayer) -> usize {
        self.size
    }

    fn output(self: &SigmoidLayer, inputs: &[f32]) -> Vec<f32> {
        let mut out: Vec<f32> = Vec::new();
        for x in inputs {
            out.push(1.0 / (1.0 + (-x).exp()));
        }
        out
    }

    /// dy / dx = y ( 1 - y )
    fn delta_from_outputs(self: &SigmoidLayer, delta: &[f32], outputs: &[f32]) -> Option<Vec<f32>> {
        assert_eq!(self.size, outputs.len());
        assert_eq!(self.size, delta.len());
        let mut derivs: Vec<f32> = vec![0.0; self.size];
        for ((d, y), yd) in delta.iter().zip(outputs).zip(derivs.iter_mut()) {
            *yd = d * (y * (1.0 - y));
        }
        Some(derivs)
    }
}

impl WeightedLayer for SigmoidLayer {
    fn weight_count(&self) -> usize {
        0
    }
    fn neuron_count(&self) -> usize {
        0
    }
    fn weights_mut(self: &mut SigmoidLayer) -> Option<&mut Vec<f32>> {
        None
    }

    fn bias_mut(self: &mut SigmoidLayer) -> Option<&mut Vec<f32>> {
        None
    }
}

pub struct RectifiedLayer {
    pub size: usize,
}

impl WeightedLayer for RectifiedLayer {
    fn weight_count(&self) -> usize {
        0
    }
    fn neuron_count(&self) -> usize {
        0
    }
    fn weights_mut(self: &mut RectifiedLayer) -> Option<&mut Vec<f32>> {
        None
    }

    fn bias_mut(self: &mut RectifiedLayer) -> Option<&mut Vec<f32>> {
        None
    }
}

impl Layer for RectifiedLayer {
    fn input_count(self: &RectifiedLayer) -> usize {
        self.size
    }

    fn output_count(self: &RectifiedLayer) -> usize {
        self.size
    }

    fn output(self: &RectifiedLayer, inputs: &[f32]) -> Vec<f32> {
        let mut out: Vec<f32> = Vec::new();
        for x in inputs {
            out.push(if *x < 0.0 { 0.0 } else { *x });
        }
        out
    }

    /// dy / dx = sigmoid function
    fn delta_from_inputs(self: &RectifiedLayer, delta: &[f32], inputs: &[f32]) -> Option<Vec<f32>> {
        assert_eq!(self.size, inputs.len());
        assert_eq!(self.size, delta.len());
        let mut derivs: Vec<f32> = Vec::new();
        for (d, x) in delta.iter().zip(inputs) {
            derivs.push(d * (1.0 / (1.0 + (-x).exp())));
        }
        Some(derivs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use traits::Layer;

    #[test]
    fn dense_output() {
        // Input shape is two, layer contains three neurons, output
        // will thus be three
        let w = vec![0.5, 2.0, -1.0, 0.5, 2.0, 3.0];
        let b = vec![0.1, 0.2, 0.3];

        let l = DenseLayer {
            weights: w,
            bias: b,
            shape: (2, 3),
        };

        assert_eq!(l.output(&vec![1.0, -1.0]), vec![-1.4, -1.3, -0.7]);
    }

    #[test]
    fn dense_delta_from_inputs() {
        let w = vec![0.5, 2.0, -1.0, 0.5, 2.0, 3.0];
        let b = vec![0.1, 0.2, 0.3];
        let l = DenseLayer {
            weights: w,
            bias: b,
            shape: (2, 3),
        };

        let x = vec![1.0, 2.0];
        assert_eq!(l.delta_from_inputs(&vec![1.0, 1.0, 1.0], &x),
                   Some(vec![1.5, 5.5]));
    }

    #[test]
    fn dense_derivw() {
        let w = vec![0.5, 2.0, -1.0, 0.5, 2.0, 3.0];
        let b = vec![0.1, 0.2, 0.3];
        let l = DenseLayer {
            weights: w,
            bias: b,
            shape: (2, 3),
        };

        let x = vec![1.0, 2.0];
        assert_eq!(l.derivw(&x), Some(vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]));
    }

    #[test]
    fn hyperbolic_output() {
        let l = HyperbolicLayer { size: 5 };
        let expected = vec![-1.0, -0.7615942, 0.0, 0.7615942, 1.0];

        assert_eq!(l.output(&vec![-999999.0, -1.0, 0.0, 1.0, 999999.0]),
                   expected);
    }

    #[test]
    fn hyperbolic_derivo() {
        let l = HyperbolicLayer { size: 3 };
        let expected = vec![1.0, 0.0, -3.0];

        assert_eq!(l.delta_from_outputs(&vec![1.0, 1.0, 1.0], &vec![0.0, 1.0, 2.0]),
                   Some(expected));
    }

    #[test]
    fn sigmoid_output() {
        let l = SigmoidLayer { size: 5 };
        let expected = vec![0.0, 0.26894143, 0.5, 0.7310586, 1.0];

        assert_eq!(l.output(&vec![-999999.0, -1.0, 0.0, 1.0, 999999.0]),
                   expected);
    }

    #[test]
    fn sigmoid_delta_from_outputs() {
        let l = SigmoidLayer { size: 3 };
        let expected = vec![0.0, 0.25, 0.0];

        assert_eq!(l.delta_from_outputs(&vec![1.0, 1.0, 1.0], &vec![0.0, 0.5, 1.0]),
                   Some(expected));
    }

    #[test]
    fn rectified_output() {
        let l = RectifiedLayer { size: 5 };
        let expected = vec![0.0, 0.0, 0.0, 1.0, 999.0];

        assert_eq!(l.output(&vec![-999999.0, -1.0, 0.0, 1.0, 999.0]), expected);
    }

    #[test]
    fn rectified_delta_from_inputs() {
        let l = RectifiedLayer { size: 5 };
        let expected = vec![0.0, 0.26894143, 0.5, 0.7310586, 1.0];

        assert_eq!(l.delta_from_inputs(&vec![1.0, 1.0, 1.0, 1.0, 1.0],
                                       &vec![-999999.0, -1.0, 0.0, 1.0, 999.0]),
                   Some(expected));
    }
}
