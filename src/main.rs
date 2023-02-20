mod trees;
use linfa_datasets;
use ndarray;
use ndarray::arr1;
use rand::{self, thread_rng};

use trees::{accuracy, decision_trees::DecisionTree};

use crate::trees::ModelInterface;

fn main() {
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

    println!("The validation accuracy is: {:?}", acc);

    let train_preds = model.predict(train_x);
    let train_acc = accuracy(
        train_preds,
        train_y.to_vec().iter().map(|x| *x as u64).collect(),
    );

    println!("The train accuracy is: {:?}", train_acc);
    model.print_tree();
}
