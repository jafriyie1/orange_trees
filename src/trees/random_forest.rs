use ndarray::Axis;
use rand::Rng;
use rayon::prelude::*;
use std::collections::HashSet;

use super::{decision_trees::DecisionTree, majority_vote_label, ModelInterface, Unique};

pub enum FeatureSampler {
    LogTwo(f64),
    Sqrt(f64),
    All(f64),
}

pub fn get_n_features(n_features: FeatureSampler) -> u64 {
    let out_n_features = match n_features {
        FeatureSampler::LogTwo(n_feat) => (n_feat.log2()) as u64,
        FeatureSampler::Sqrt(n_feat) => (n_feat.sqrt()) as u64,
        FeatureSampler::All(n_feat) => n_feat as u64,
    };

    out_n_features
}

pub struct RandomForest {
    n_features: usize,
    n_trees: usize,
    min_leaf: u64,
    min_sample_size: u64,
    max_tree_depth: u64,
    decision_trees: Vec<DecisionTree>,
}

impl RandomForest {
    pub fn new(
        n_features: usize,
        n_trees: usize,
        min_leaf: u64,
        min_sample_size: u64,
        max_tree_depth: u64,
    ) -> Self {
        Self {
            n_features,
            n_trees,
            min_leaf,
            min_sample_size,
            max_tree_depth,
            decision_trees: Vec::<DecisionTree>::with_capacity(n_trees),
        }
    }
}

impl ModelInterface for RandomForest {
    fn fit(&mut self, X: &ndarray::Array2<f64>, y: &ndarray::Array1<f64>) {
        let feature_shape = X.shape();
        let total_features = feature_shape[1];

        for _ in 0..self.n_trees {
            let mut rand_gen = rand::thread_rng();
            let rand_feats: Vec<usize> = (0..self.n_features)
                .map(|_| rand_gen.gen_range(0..total_features))
                .collect::<HashSet<usize>>()
                .into_iter()
                .collect::<Vec<usize>>();

            let sampled_x = X.select(Axis(1), &rand_feats);
            let mut small_tree =
                DecisionTree::new(self.max_tree_depth as u32, self.min_sample_size as u32);
            small_tree.fit(&sampled_x, y);
            self.decision_trees.push(small_tree);
        }
    }

    fn predict(&mut self, X: ndarray::Array2<f64>) -> Vec<u64> {
        let shape = &X.shape();
        let n_samples = shape[0] as usize;
        let mut preds = Vec::with_capacity(n_samples);

        for row in X.axis_iter(Axis(0)) {
            let single_preds: Vec<f64> = self
                .decision_trees
                .par_iter_mut()
                .map(|tree| {
                    let preds = tree.single_prediction(row.to_owned());
                    preds as f64
                })
                .collect();

            let unique_preds = Unique::Vec(&single_preds);
            let label = majority_vote_label(&unique_preds);
            preds.push(label.unwrap());
        }

        preds
    }

    fn print_tree(&self) {
        todo!()
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
        let n_features = train_x.shape()[1];
        let use_n_features = get_n_features(FeatureSampler::All(n_features as f64));

        let valid_y: Vec<f64> = valid.targets.iter().map(|x| *x as f64).collect();
        let valid_y = arr1(&valid_y);
        let valid_x = valid.records;
        // Get the features and labels for the testing set

        let mut model = RandomForest::new(use_n_features as usize, 100, 5, 2, 5);
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
