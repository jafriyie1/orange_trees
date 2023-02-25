use ndarray::{arr1, arr2, s, stack, Array, Array1, Array2, Axis};
use rand::Rng;
use rayon::prelude::*;

use super::{decision_trees::DecisionTree, majority_vote_label, ModelInterface, Unique};

pub struct CrossEntropy;

impl CrossEntropy {
    fn clipper(&self, v: &f64) -> f64 {
        if v > &1e15 {
            1e15 as f64
        } else if v < &1e-15 {
            1e-15 as f64
        } else {
            *v
        }
    }

    pub fn loss(&self, y: &Array1<f64>, p: &Array1<f64>) -> Array1<f64> {
        let clip_p = p.mapv(|p| self.clipper(&p));
        let log_p = clip_p.map(|p| p.ln());
        let one_log_p = clip_p.map(|p| (1.0 - p).ln());

        y * log_p - (y - 1.0) * one_log_p
    }

    pub fn gradient(&self, y: &Array1<f64>, p: &Array1<f64>) -> Array1<f64> {
        // avoid dividing by zero
        let clip_p = p.mapv(|p| self.clipper(&p));

        -1.0 * ((y / clip_p) + (1.0 - y) / (1.0 - p))
    }
}

pub struct GradientBoostingTrees {
    loss: CrossEntropy,
    n_trees: usize,
    min_sample_size: u64,
    max_tree_depth: u64,
    learning_rate: f64,
    decision_trees: Vec<DecisionTree>,
}

impl GradientBoostingTrees {
    pub fn new(
        n_trees: usize,
        min_sample_size: u64,
        max_tree_depth: u64,
        learning_rate: f64,
    ) -> Self {
        Self {
            loss: CrossEntropy,
            n_trees,
            min_sample_size,
            max_tree_depth,
            learning_rate,
            decision_trees: Vec::<DecisionTree>::with_capacity(n_trees),
        }
    }
}

impl ModelInterface for GradientBoostingTrees {
    fn fit(&mut self, X: &Array2<f64>, y: &Array1<f64>) {
        let init_common_label = majority_vote_label(&Unique::Ndarray(&y));
        let mut y_pred = Array1::from_elem(y.len(), init_common_label.unwrap() as f64);

        for _ in 0..self.n_trees {
            let gradient = self.loss.gradient(&y, &y_pred);
            let mut decision_tree =
                DecisionTree::new(self.max_tree_depth as u32, self.min_sample_size as u32);
            decision_tree.fit(X, &gradient);
            let mut init_update = decision_tree.predict(&X);
            let mut init_update: Vec<f64> = init_update.into_iter().map(|x| x as f64).collect();
            let update = arr1(&init_update);
            self.decision_trees.push(decision_tree);
            y_pred = y_pred - (update * self.learning_rate);
        }
    }

    fn predict(&mut self, X: &Array2<f64>) -> Vec<u64> {
        let shape = &X.shape();
        let n_samples = shape[0] as usize;
        let mut out_preds = Vec::with_capacity(n_samples);
        // placeholder value
        let dummy_class = -10.0;
        let mut y_pred = Array1::from_elem(n_samples, dummy_class);

        let mut pred_array = Vec::with_capacity(self.n_trees);
        for tree in &mut self.decision_trees {
            let init_update = tree.predict(X);
            let init_update: Vec<f64> = init_update.into_iter().map(|x| x as f64).collect();

            let update = arr1(&init_update);
            let update = update * self.learning_rate;

            let if_eq_dummy = y_pred.iter().all(|&x| x == dummy_class);
            y_pred = if if_eq_dummy {
                update * -1.0
            } else {
                let temp_y = y_pred - update;
                temp_y
            };
            let new_copy = y_pred.clone();
            let new_y_pred = new_copy.view();
            pred_array.push(new_y_pred.to_owned());
        }

        let mut logits = Array2::zeros((self.n_trees, n_samples));

        for (i, arr1) in pred_array.iter().enumerate() {
            logits.slice_mut(s![i, ..]).assign(arr1);
        }
        let logits_transposed = logits.t();
        // let logits = ndarray::stack(Axis(1), &[array2.view()]).unwrap();
        let exp_logits = logits_transposed.mapv(|x| x.exp());
        // softmax
        let numerator = y_pred.mapv(|x| x.exp());
        let denominator = exp_logits.sum_axis(Axis(1));
        let expand_denominator = denominator.insert_axis(Axis(1));
        let softmax = numerator / expand_denominator;

        for row in softmax.axis_iter(Axis(1)) {
            let idx = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(index, _)| index)
                .unwrap() as u64;
            out_preds.push(idx);
        }

        out_preds
    }

    fn print_tree(&self) {
        todo!()
    }
}
