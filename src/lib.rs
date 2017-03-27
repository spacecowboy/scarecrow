//! `scarecrow` is a basic and simple implementation of an artificial
//! neural network. It demonstrates the basics behind machine
//! learning.
//!
//! ## Example - XOR
//!
//! This demonstrates the capability of the neural network to learn a
//! non-linear function. First define your inputs and targets:
//!
//! ```
//! // Two binary input values, 4 possible combinations
//! let inputs = vec![0.0, 0.0,
//!                   0.0, 1.0,
//!                   1.0, 0.0,
//!                   1.0, 1.0];
//! // Four binary output targets, one for each possible input value
//! let targets = vec![0.0,
//!                    1.0,
//!                    1.0,
//!                    0.0];
//! ```
//!
//! Then, we construct a neural network by adding a number of layers
//! to a list:
//!
//! ```
//! let mut layers: LinkedList<Box<WeightedLayer>> = LinkedList::new();
//! // We start by a hidden "dense" layer of 6 neurons which should
//! // accept 2 input values.
//! layers.push_back(Box::new(DenseLayer::random(2, 6)));
//! // We attach hyperbolic activation functions to the dense layer
//! layers.push_back(Box::new(HyperbolicLayer { size: 6 }));
//! // We follow this with a final "dense" layer with a single neuron,
//! // expecting 6 inputs from the preceeding layer.
//! layers.push_back(Box::new(DenseLayer::random(6, 1)));
//! // This will be output neuron so we attach a sigmoid activation function
//! // to get an output between 0 and 1.
//! layers.push_back(Box::new(SigmoidLayer { size: 1 }));
//! ```
//!
//! Since this is before training, we should expect a completely
//! random output from the network. This can be seen by feeding the
//! inputs through the network:
//!
//! ```
//! for (x, t) in inputs.chunks(2).zip(targets.chunks(1)) {
//!     let mut o = x.to_vec();
//!     for l in layers.iter() {
//!         o = l.output(&o);
//!     }
//!     println!("X: {:?}, Y: {:?}, T: {:?}", x, o, t);
//! }
//! ```
//!
//! Example output:
//!
//!
//!
//!
//!
//!


extern crate rand;

pub mod traits;
pub mod layers;
pub mod utils;
pub mod sgd;
pub mod loss;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
