// mod trees;

use ndarray;
use ndarray::{Array, Array1, Array2};
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
        if depth >= self.max_depth
            || self.n_class_labels.unwrap() == 1 as u32
            || self.n_samples.unwrap() < self.min_samples_split
        {
            true
        } else {
            false
        }
    }

    fn entropy(&self, y: &Array1<f64>) -> f64 {
        let y_length = y.len() as u64;
        let unique_values: HashSet<_> = y.iter().cloned().map(|v| v as u64).collect();
        let proportions: Vec<f64> = unique_values
            .into_iter()
            .map(|val| (val as f64 / y_length as f64) as f64)
            .collect();

        // need to filer on negative values else log won't work
        // let entropy = propotions.into_iter()._mapsds  (|p| p * p.log2()).sum();
        let entropy: f64 = proportions
            .into_iter()
            .filter_map(|p| if p > 0.0 { Some(p * p.log2()) } else { None })
            .sum();
        let entropy = entropy * -1.0;
        entropy
    }

    fn information_gain(
        &mut self,
        selected_feat: Array1<f64>,
        y: Array1<f64>,
        threshold: f64,
    ) -> f64 {
        // formula for information gain
        let parent_entropy = self.entropy(&y);
        let (right_split, left_split) = self.create_split(&selected_feat, &threshold);

        let n = &y.len();
        let n_left = left_split.len();
        let n_right = right_split.len();

        if n_left == 0 || n_right == 0 {
            return 0.0;
        }

        let y_idx_right = y.select(ndarray::Axis(0), &right_split);
        let y_idx_left = y.select(ndarray::Axis(0), &left_split);

        let child_loss = (n_left as f64 / *n as f64) * self.entropy(&y_idx_left)
            + (n_right as f64 / *n as f64) * self.entropy(&y_idx_right);
        parent_entropy - child_loss
    }

    fn create_split<'a>(
        &mut self,
        selected_feat: &'a Array1<f64>,
        threshold: &'a f64,
    ) -> (Vec<usize>, Vec<usize>) {
        let left_idx: Vec<_> = selected_feat
            .indexed_iter()
            .filter(|&(_, x)| x <= threshold)
            .map(|(a, _)| (a))
            .collect();
        let right_idx: Vec<_> = selected_feat
            .indexed_iter()
            .filter(|&(_, x)| x > threshold)
            .map(|(a, _)| (a))
            .collect();

        (right_idx, left_idx)
    }

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
