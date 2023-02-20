use ndarray;
use ndarray::{Array1, Array2, Axis};
use rand;
use rand::Rng;
use std::collections::HashMap;

use super::{get_unique_values, ModelInterface, Unique};

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Node {
    feature: Option<f64>,
    threshold: Option<f64>,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
    value: Option<u64>,
}

impl Node {
    pub fn new(
        feature: Option<f64>,
        threshold: Option<f64>,
        left: Option<Box<Node>>,
        right: Option<Box<Node>>,
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
        let unique_values = get_unique_values(&Unique::Ndarray(&y));
        let mut freq_counts = Vec::with_capacity(unique_values.len());

        for label in &unique_values {
            let count = y
                .iter()
                .cloned()
                .filter(|elem| (*elem as f64) == *label)
                .count();
            freq_counts.push(count);
        }

        let proportions: Vec<f64> = freq_counts
            .into_iter()
            .map(|val| (val as f64 / y_length as f64) as f64)
            .collect();

        let entropy: f64 = proportions
            .into_iter()
            .filter_map(|p| if p > 0.0 { Some(p * p.log2()) } else { None })
            .sum();
        let entropy = entropy * -1.0;
        entropy
    }

    fn information_gain(
        &self,
        selected_feat: &Array1<f64>,
        y: &Array1<f64>,
        threshold: f64,
    ) -> f64 {
        // formula for information gain
        let parent_entropy = self.entropy(y);
        let (left_split, right_split) = self.create_split(selected_feat, &threshold);
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
        &self,
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

        (left_idx, right_idx)
    }

    fn best_split(&self, X: &Array2<f64>, y: &Array1<f64>, features: Vec<f64>) -> (f64, f64) {
        let mut split: HashMap<&str, Option<f64>> = HashMap::new();
        split.insert("score", Some(-1.0));
        split.insert("feat", None);
        split.insert("thresh", None);

        for feat in features {
            let x_feat = X.select(Axis(1), &[feat as usize]).remove_axis(Axis(1));
            let thresholds = get_unique_values(&Unique::Ndarray(&x_feat));

            for thresh in thresholds {
                let score = self.information_gain(&x_feat, &y, thresh);
                if score > split["score"].unwrap() {
                    let score_entry = split.entry("score").or_insert(Some(score));
                    *score_entry = Some(score);

                    let feat_entry = split.entry("feat").or_insert(Some(feat));
                    *feat_entry = Some(feat);

                    let thresh_entry = split.entry("thresh").or_insert(Some(thresh));
                    *thresh_entry = Some(thresh);
                }
            }
        }

        (split["feat"].unwrap(), split["thresh"].unwrap())
    }

    fn build_tree(&mut self, X: &Array2<f64>, y: &Array1<f64>, depth: u32) -> Option<Box<Node>> {
        let shape = &X.shape();
        self.n_samples = Some(shape[0] as u32);
        self.n_features = Some(shape[1] as u32);

        // get unique values from array
        let unique_values = get_unique_values(&Unique::Ndarray(&y));
        self.n_class_labels = Some(unique_values.len() as u32);

        // stopping criteria
        if self.is_finished(depth) {
            let freq_counts: HashMap<_, _> =
                y.iter()
                    .map(|x| *x as u64)
                    .fold(HashMap::new(), |mut map, c| {
                        *map.entry(c).or_insert(0) += 1;
                        map
                    });
            let most_common_label = freq_counts
                .into_iter()
                .max_by_key(|(_, v)| *v as u32)
                .map(|(k, _)| k as u64);

            return Some(Box::new(Node::new(
                None,
                None,
                None,
                None,
                most_common_label,
            )));
        }

        let mut rng = rand::thread_rng();
        let features = (0..self.n_features.unwrap())
            .map(|_| rng.gen_range(0..self.n_features.unwrap()))
            .map(|x| x as f64)
            .collect();

        let (best_feat, best_threshold) = self.best_split(X, y, features);
        let x_feature = X.select(Axis(1), &[best_feat as usize]);
        let x_feature = x_feature.remove_axis(Axis(1));
        let (left_idx, right_idx) = self.create_split(&x_feature, &best_threshold);

        let left_x = X.select(Axis(0), &left_idx);
        let left_y = &y.select(Axis(0), &left_idx);
        let right_x = X.select(Axis(0), &right_idx);
        let right_y = &y.select(Axis(0), &right_idx);

        let left_child = self.build_tree(&left_x, left_y, depth + 1);
        let right_child = self.build_tree(&right_x, right_y, depth + 1);

        Some(Box::new(Node::new(
            Some(best_feat),
            Some(best_threshold),
            left_child,
            right_child,
            None,
        )))
    }

    fn traverse_tree(&self, x: Array1<f64>, node: &Option<Box<Node>>) -> u64 {
        if let Some(node) = node {
            if node.is_leaf() {
                return node.value.unwrap();
            }
            let x_feat = x.select(Axis(0), &[node.feature.unwrap() as usize]);
            if x_feat[[0]] <= node.threshold.unwrap() {
                return self.traverse_tree(x, &node.left);
            }

            return self.traverse_tree(x, &node.right);
        }
        panic!("The Decision Tree is empty");
    }
}

impl ModelInterface for DecisionTree {
    fn fit(&mut self, X: &Array2<f64>, y: &Array1<f64>) {
        let depth = 0;
        self.root = self.build_tree(X, y, depth);
    }

    fn predict(&mut self, X: Array2<f64>) -> Vec<u64> {
        let shape = &X.shape();
        let n_samples = shape[0] as usize;
        let mut preds = Vec::with_capacity(n_samples);

        for row in X.axis_iter(Axis(0)) {
            let y_pred = self.traverse_tree(row.to_owned(), &self.root);
            preds.push(y_pred);
        }

        preds
    }

    fn print_tree(&self) {
        println!(
            "feature: {:?}--threshold: {:?}",
            &self.root.as_ref().unwrap().feature,
            &self.root.as_ref().unwrap().threshold
        );

        let rec = |n: &Option<Box<Node>>| match n {
            Some(n) => {
                if n.value.is_some() {
                    println!("class label: {:?}", n.value);
                } else {
                    println!("feature: {:?}--threshold: {:?}", n.feature, n.threshold)
                }
            }
            None => println!("Empty"),
        };

        rec(&self.root.as_ref().unwrap().left);
        rec(&self.root.as_ref().unwrap().right);
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;
    use rand::thread_rng;

    use crate::trees::accuracy;

    use super::*;

    #[test]
    fn test_model() {
        let mut shuffle_rng = thread_rng();
        let (train, valid) = linfa_datasets::winequality()
            .shuffle(&mut shuffle_rng)
            .map_targets(|x| *x > 3)
            .map_targets(|x| *x as u8)
            .split_with_ratio(0.90);

        let train_y: Vec<f64> = train.targets.iter().map(|x| *x as f64).collect();
        let train_y = arr1(&train_y);
        let train_x = train.records;

        let valid_y: Vec<f64> = valid.targets.iter().map(|x| *x as f64).collect();
        let valid_y = arr1(&valid_y);
        let valid_x = valid.records;
        // Get the features and labels for the testing set

        let mut model = DecisionTree::new(10, 2);
        model.fit(&train_x, &train_y);
        let preds = model.predict(valid_x);
        let acc = accuracy(preds, valid_y.to_vec().iter().map(|x| *x as u64).collect());
        let train_preds = model.predict(train_x);

        assert_eq!(train_preds.len(), train_y.len());
        let train_acc = accuracy(
            train_preds,
            train_y.to_vec().iter().map(|x| *x as u64).collect(),
        );

        assert!(acc >= 0.90);
        assert!(train_acc >= 0.90);
    }
}
