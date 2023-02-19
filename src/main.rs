// mod trees;

use linfa_datasets;
use ndarray;
use ndarray::{arr1, Array1, Array2, Axis};
use rand;
use rand::Rng;
use std::collections::{HashMap, HashSet};

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
            root: Some(Box::new(Node::new(None, None, None, None, None))),
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
        &self,
        selected_feat: &Array1<f64>,
        y: &Array1<f64>,
        threshold: f64,
    ) -> f64 {
        // formula for information gain
        let parent_entropy = self.entropy(y);
        let (right_split, left_split) = self.create_split(selected_feat, &threshold);

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

        (right_idx, left_idx)
    }

    fn best_split(&self, X: &Array2<f64>, y: &Array1<f64>, features: Vec<f64>) -> (f64, f64) {
        println!("{:?}", X.shape());
        let mut split: HashMap<&str, Option<f64>> = HashMap::new();
        split.insert("score", Some(-1.0));
        split.insert("feat", None);
        split.insert("thres", None);

        for feat in features {
            println!("{:?}", &feat);

            let temp = X.select(Axis(1), &[feat as usize])          
            let x_feat = X.select(Axis(1), &[feat as usize]).remove_axis(Axis(0));
            // println!("{:?}", &x_feat);
            let thresholds: HashSet<_> = x_feat.iter().cloned().map(|v| v as u64).collect();

            for thresh in thresholds {
                let score = self.information_gain(&x_feat, &y, thresh as f64);

                // println!("{:?}", &split);
                if score > split["score"].unwrap() {
                    split.entry("score").or_insert(Some(score));
                    split.entry("feat").or_insert(Some(feat));
                    split.entry("thresh").or_insert(Some(thresh as f64));
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
        let unique_values: HashSet<_> = y.iter().map(|x| *x as u64).collect();
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
        let x_feature = x_feature.remove_axis(Axis(0));
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
            let x_feat = x.select(Axis(1), &[node.feature.unwrap() as usize]);
            if x_feat[[0]] < node.threshold.unwrap() {
                return self.traverse_tree(x, &node.left);
            }

            return self.traverse_tree(x, &node.right);
        }
        panic!("The Decision Tree is empty");
    }

    fn fit(&mut self, X: Array2<f64>, y: Array1<f64>) {
        let depth = 0;
        self.root = self.build_tree(&X, &y, depth);
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
}

fn main() {
    // let mut df = LazyCsvReader::new("datasets/breast_cancer.csv")
    //     .finish()
    //     .unwrap();
    // println!("{:?}", df.head(Some(5)));

    // df = df.drop_columns(["id"]);
    // df = df
    //     .map("diagnosis", |value: &str| match value {
    //         "M" => Ok(1),
    //         "B" => Ok(0),
    //         _ => Err(_),
    //     })
    //     .unwrap();
    // let num_rows = df.height();

    // let train_mask = vec![true; (num_rows as f64 * 0.85) as usize];
    // let train_df = df.slice(&train_mask);
    let (train, valid) = linfa_datasets::winequality().split_with_ratio(0.8);
    let train_y: Vec<f64> = train.targets.iter().map(|x| *x as f64).collect();
    let train_y = arr1(&train_y);
    let train_x = train.records;

    let valid_y: Vec<f64> = valid.targets.iter().map(|x| *x as f64).collect();
    let valid_y = arr1(&valid_y);
    let valid_x = valid.records;
    // Get the features and labels for the testing set
    println!("{:?}", train_y);
    println!("Hello, world!");

    let mut model = DecisionTree::new(10, 4);
    model.fit(train_x, train_y);
}
