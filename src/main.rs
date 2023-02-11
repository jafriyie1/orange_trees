// mod trees;

use ndarray::{Array1, Array2};
use std::collections::{HashMap, HashSet};

#[allow(dead_code)]
pub struct Node {
    feature: Option<f64>,
    threshold: Option<f64>,
    left: Option<f64>,
    right: Option<f64>,
    value: Option<u64>,
}

impl Node {
    pub fn new(
        feature: Option<f64>,
        threshold: Option<f64>,
        left: Option<f64>,
        right: Option<f64>,
        value: Option<u64>,
    ) -> Self {
        Self {
            feature,
            threshold,
            left,
            right,
            value,
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.value.is_some()
    }
}

pub struct DecisionTree {
    max_depth: u32,
    min_samples_split: u32,
    root: Option<Box<Node>>,
    n_samples: Option<u32>,
    n_features: Option<u32>,
    n_class_labels: Option<u32>,
}

impl DecisionTree {
    pub fn new(max_depth: u32, min_samples_split: u32) -> Self {
        Self {
            max_depth,
            min_samples_split,
            root: None,
            n_samples: None,
            n_features: None,
            n_class_labels: None,
        }
    }

    fn is_finished(&self, depth: u32) -> bool {
        if (depth >= self.max_depth
            || self.n_class_labels.unwrap() == 1 as u32
            || self.n_samples.unwrap() < self.min_samples_split)
        {
            true
        } else {
            false
        }
    }

    fn entropy(self, y: Array1<f64>) {}

    fn create_split(self, X: Array2<f64>, threshold: f64) {}

    fn information_gain(self, X: Array2<f64>, y: Array1<f64>, threshold: f64) {}
    fn best_split(self, X: Array2<f64>, y: Array1<f64>, features: f64) {}
    fn best_tree(self, X: Array2<f64>, y: Array1<f64>, features: f64) {}

    fn build_tree(&mut self, X: Array2<f64>, y: Array1<u64>, depth: u32) -> Option<Box<Node>> {
        let shape = &X.shape();
        self.n_samples = Some(shape[0] as u32);
        self.n_features = Some(shape[1] as u32);

        // get unique values from array
        let unique_values: HashSet<_> = y.iter().cloned().collect();
        self.n_class_labels = Some(unique_values.len() as u32);

        // stopping criteria
        if self.is_finished(depth) {
            let freq_counts: HashMap<_, _> =
                y.iter().cloned().fold(HashMap::new(), |mut map, c| {
                    *map.entry(c).or_insert(0) += 1;
                    map
                });
            let most_common_label = freq_counts
                .into_iter()
                .max_by_key(|(_, v)| *v as u32)
                .map(|(k, _)| k);

            return Some(Box::new(Node::new(
                None,
                None,
                None,
                None,
                most_common_label,
            )));
        } else {
            None
        }
    }

    fn traverse_tree(self, X: Array2<f64>, y: Array1<u64>, features: f64) {}

    fn fit(&mut self, X: Array2<f64>, y: Array1<u64>) {
        let mut depth = 0;
        self.root = self.build_tree(X, y, depth);
    }

    fn predict(self, X: Array2<f64>) {}
}

fn main() {
    println!("Hello, world!");
}
