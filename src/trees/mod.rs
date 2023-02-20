pub mod decision_trees;
use ndarray::{Array1, Array2};
use std::collections::HashSet;

pub fn accuracy<T>(preds: Vec<T>, ground_truth: Vec<T>) -> f64
where
    T: Eq + PartialEq,
{
    let correct = preds
        .iter()
        .zip(&ground_truth)
        .filter(|&(p, gt)| p == gt)
        .count();
    correct as f64 / preds.len() as f64
}

pub enum Unique<'a> {
    Vec(&'a Vec<f64>),
    Ndarray(&'a Array1<f64>),
}

pub trait ModelInterface {
    fn fit(&mut self, X: &Array2<f64>, y: &Array1<f64>);
    fn predict(&mut self, X: Array2<f64>) -> Vec<u64>;
    fn print_tree(&self);
}

pub fn get_unique_values(unique_container: &Unique) -> Vec<f64> {
    let vec_string: Vec<String> = match unique_container {
        Unique::Vec(vec) => vec.iter().cloned().map(|v| v.to_string()).collect(),
        Unique::Ndarray(vec_arr) => vec_arr
            .to_vec()
            .iter()
            .cloned()
            .map(|v| v.to_string())
            .collect(),
    };

    let mut unique_values = HashSet::new();
    for val in vec_string {
        if !unique_values.contains(&val) {
            unique_values.insert(val);
        }
    }

    let conv_vec = unique_values
        .into_iter()
        .map(|v| v.parse::<f64>().unwrap())
        .collect();
    conv_vec
}
