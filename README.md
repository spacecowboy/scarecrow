# scarecrow

[![Build Status](https://travis-ci.org/spacecowboy/scarecrow.svg?branch=master)](https://travis-ci.org/spacecowboy/scarecrow)

`scarecrow` is a basic and simple implementation of an artificial
neural network.

## Example

This demonstrates the capability of the neural network to learn a
non-linear function, namely [XOR](https://en.wikipedia.org/wiki/Exclusive_or).
It trains on a truth-table using
[gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent).

First we define inputs `X` and targets `T`:

```rust
// Two binary input values, 4 possible combinations
let inputs = vec![0.0, 0.0,
                  0.0, 1.0,
                  1.0, 0.0,
                  1.0, 1.0];
// Four binary output targets, one for each possible input value
let targets = vec![0.0,
                   1.0,
                   1.0,
                   0.0];
```

Then, we construct a neural network by adding a number of layers
to a list:

```rust
let mut layers: LinkedList<Box<WeightedLayer>> = LinkedList::new();
// We start by a hidden "dense" layer of 6 neurons which should
// accept 2 input values.
layers.push_back(Box::new(DenseLayer::random(2, 6)));
// We attach hyperbolic activation functions to the dense layer
layers.push_back(Box::new(HyperbolicLayer { size: 6 }));
// We follow this with a final "dense" layer with a single neuron,
// expecting 6 inputs from the preceeding layer.
layers.push_back(Box::new(DenseLayer::random(6, 1)));
// This will be output neuron so we attach a sigmoid activation function
// to get an output between 0 and 1.
layers.push_back(Box::new(SigmoidLayer { size: 1 }));
```

Since this is before training, we should expect a completely
random output from the network. This can be seen by feeding the
inputs through the network:

```rust
for (x, t) in inputs.chunks(2).zip(targets.chunks(1)) {
    let mut o = x.to_vec();
    for l in layers.iter() {
        o = l.output(&o);
    }
    println!("X: {:?}, Y: {:?}, T: {:?}", x, o, t);
}
```

Example of network output `Y`:

```text
X: [0, 0], Y: [0.4244223], T: [0]
X: [0, 1], Y: [0.049231697], T: [1]
X: [1, 0], Y: [0.12347225], T: [1]
X: [1, 1], Y: [0.02869209], T: [0]
```

To train the network, first create a suitable trainer and then
call its train method:

```rust
// A trainer which uses stochastic gradient descent. Run for
// 1000 iterations with a learning rate of 0.1.
let trainer = SGDTrainer::new(1000, 0.1);
// Train the network on the given inputs and targets
trainer.train(&mut layers, &inputs, &targets);
```

Now calculate the output for the trained network:

```rust
for (x, t) in inputs.chunks(2).zip(targets.chunks(1)) {
    let mut o = x.to_vec();
    for l in layers.iter() {
        o = l.output(&o);
    }
    println!("X: {:?}, Y: {:?}, T: {:?}", x, o, t);
}
```

Final result, note that network output `Y` is quite close to
targets `T`:

```text
X: [0, 0], Y: [0.03515992], T: [0]
X: [0, 1], Y: [0.96479124], T: [1]
X: [1, 0], Y: [0.96392107], T: [1]
X: [1, 1], Y: [0.03710678], T: [0]
```
