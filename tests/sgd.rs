extern crate scarecrow;

use scarecrow::traits::*;
use scarecrow::layers::*;
use scarecrow::sgd::*;

use std::collections::LinkedList;

#[test]
fn train_xor() {
    // Input shape is two
    let inputs = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let targets = vec![0.0, 1.0, 1.0, 0.0];

    // Store layers in a list
    let mut layers: LinkedList<Box<WeightedLayer>> = LinkedList::new();

    layers.push_back(Box::new(DenseLayer::random(2, 6)));
    layers.push_back(Box::new(HyperbolicLayer { size: 6 }));
    layers.push_back(Box::new(DenseLayer::random(6, 1)));
    layers.push_back(Box::new(SigmoidLayer { size: 1 }));

    // Calculate initial output
    for (x, t) in inputs.chunks(2).zip(targets.chunks(1)) {
        let mut o = x.to_vec();

        for l in layers.iter() {
            o = l.output(&o);
        }

        println!("X: {:?}, Y: {:?}, T: {:?}", x, o, t);
        assert_eq!(o.len(), 1);
        // assert!(o[0] - 0.502 < 0.01);
    }

    let trainer = SGDTrainer::new(1000, 0.1);

    trainer.train(&mut layers, &inputs, &targets);

    // Calculate final output
    for (x, t) in inputs.chunks(2).zip(targets.chunks(1)) {
        let mut o = x.to_vec();

        for l in layers.iter() {
            o = l.output(&o);
        }

        println!("X: {:?}, Y: {:?}, T: {:?}", x, o, t);
        assert_eq!(o.len(), 1);
        assert!(trainer.loss.loss1(o[0], t[0]) < 0.01);
    }
}
