pub mod decision_trees;
pub mod random_forest;
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

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

pub fn majority_vote_label(unique_container: &Unique) -> Option<u64> {
    let y: Vec<u64> = match unique_container {
        Unique::Vec(vec) => vec.iter().map(|x| *x as u64).collect(),
        Unique::Ndarray(array) => array
            .to_vec()
            .par_iter()
            .map(|x| *x as u64)
            .collect::<Vec<u64>>(),
    };

    let freq_counts: HashMap<_, _> =
        y.iter()
            .map(|x| *x as u64)
            .fold(HashMap::new(), |mut map, c| {
                *map.entry(c).or_insert(0) += 1;
                map
            });
    let most_common_label = freq_counts
        .par_iter()
        .max_by_key(|(_, v)| **v as u32)
        .map(|(k, _)| *k as u64);

    most_common_label
}

pub fn get_unique_values(unique_container: &Unique) -> Vec<f64> {
    let vec_string: Vec<String> = match unique_container {
        Unique::Vec(vec) => vec.iter().cloned().map(|v| v.to_string()).collect(),
        Unique::Ndarray(vec_arr) => vec_arr
            .to_vec()
            .par_iter()
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
        .par_iter()
        .map(|v| v.parse::<f64>().unwrap())
        .collect();

    conv_vec
}
