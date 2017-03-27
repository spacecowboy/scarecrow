//use rand::Rng;
use rand;
use rand::distributions::{Normal, IndependentSample};

pub fn normal_vector(size: usize) -> Vec<f32> {
    let normal = Normal::new(0.0, 1.0);
    let mut rng = rand::thread_rng();

    let mut result: Vec<f32> = vec![0.0; size];
    for x in result.iter_mut() {
        *x = normal.ind_sample(&mut rng) as f32;
    }
    result
}

/// Sum up a vector.
pub fn sum(v: &[f32]) -> f32 {
    v.iter().fold(0.0, |sum, val| sum + val)
}

/// Perform an element-wise product and sum of two vectors. The two vectors
/// must be of equal length.
pub fn dot(x: &[f32], y: &[f32]) -> f32 {
    assert_eq!(x.len(), y.len());
    let mut result = 0.0;
    for (a, b) in x.iter().zip(y) {
        result += a * b;
    }
    result
}

/// Element-wise addition of two vectors. They must be of equal length.
pub fn add(x: &[f32], y: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), y.len());
    let mut x = x.to_vec();
    for (a, b) in x.iter_mut().zip(y) {
        *a += *b;
    }
    x
}

/// Addition of vector and scalar.
pub fn add_scalar(x: &[f32], y: f32) -> Vec<f32> {
    let mut x = x.to_vec();
    for a in x.iter_mut() {
        *a += y;
    }
    x
}

/// Element-wise addition of two vectors in-place. They must be of
/// equal length.
pub fn add_mut(x: &mut [f32], y: &[f32]) {
    assert_eq!(x.len(), y.len());
    for (a, b) in x.iter_mut().zip(y) {
        *a += *b;
    }
}

/// Element-wise product of two vectors. They must be of equal length.
pub fn product(x: &[f32], y: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), y.len());
    let mut x = x.to_vec();
    for (a, b) in x.iter_mut().zip(y) {
        *a *= *b;
    }
    x
}

/// Element-wise product of two vectors. They must be of equal length.
pub fn product_mut(x: &mut [f32], y: &[f32]) {
    assert_eq!(x.len(), y.len());
    for (a, b) in x.iter_mut().zip(y) {
        *a *= *b;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_test() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        assert!(dot(&a, &b) - 32.0 < 0.00001);
    }

    #[test]
    fn sum_test() {
        let a = vec![1.0, 2.0, 3.0];

        assert!((sum(&a) - 6.0).abs() < 0.00001);
    }

    #[test]
    fn add_test() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        assert_eq!(add(&a, &b), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn product_test() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        assert_eq!(product(&a, &b), vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn normal_vector_test() {
        assert_eq!(normal_vector(9).len(), 9);
    }
}
