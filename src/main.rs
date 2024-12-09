use csv::ReaderBuilder;
use ndarray::{ Array2, s};
use ndarray_csv::Array2Reader;
use std::error::Error;
use std::fs::File;
mod math;
mod network;
mod matrix;
use matrix::Matrix;
use network::Network;
mod propogations;

fn read_csv(path_to_file: &str) -> Result<Array2<u64>, Box<dyn Error>> {
    let file = File::open(path_to_file)?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
    Ok(reader.deserialize_array2((42000, 785))?)
}

impl Network {
fn gradient_descent(
    &mut self,
    inputs: &Matrix,
    targets: &Matrix,
    learning_rate: f64,
    epochs: usize,
) {
    for epoch in 0..epochs {
        // Perform backpropagation for each epoch
        self.backprop(inputs, targets, learning_rate);

        // Optionally, compute and print the loss to track progress
        let outputs = self.forward_prop(inputs.clone());
        let loss: f64 = targets
            .subtract(&outputs)
            .data
            .iter()
            .map(|x| x * x)
            .sum::<f64>() / targets.data.len() as f64;
        if epoch % 100 == 0 {
            println!("Epoch {}: Loss = {:?}", epoch + 1, loss);
        }

    }
}
}

fn main() {
    // let data = read_csv("train.csv");
    match read_csv("train.csv") {
        Ok(data) => {
            // If successful, print the shape of the array
            
            let (m, n) = data.dim();
            println!("Array shape: {:?}, {:?}", m, n);
                // Split the data into Y (first column) and X (remaining columns)
    let y_train = data.column(0).to_owned(); // First column as labels (1D array)
    let x_train = data.slice(s![.., 1..]).to_owned(); // Remaining columns as features

    // Print sizes
    println!("Shape of Y (labels): {:?}", y_train.dim());
    println!("Shape of X (features): {:?}", x_train.dim());

        }
        Err(e) => {
            // Handle errors gracefully
            eprintln!("Error reading CSV: {}", e);
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
#[test]
fn test_train_function() {
    let mut network = Network::new(vec![4, 3, 2]);

    // Create example inputs and targets
    let inputs = Matrix {
        rows: 4,
        columns: 1,
        data: vec![0.5, 0.2, 0.8, 0.1],
    };

    let targets = Matrix {
        rows: 2,
        columns: 1,
        data: vec![0.0, 1.0],
    };

    // Train the network
    network.gradient_descent(&inputs, &targets,  0.01,  10000);

    // Check if the loss decreases after training
    let loss_after: f64 = targets
        .subtract(&network.forward_prop(inputs.clone()))
        .data
        .iter()
        .map(|x| x * x)
        .sum::<f64>() / targets.data.len() as f64;

    assert!(loss_after < 1e-2, "Loss did not decrease sufficiently: {:.6}", loss_after);
}}
