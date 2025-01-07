use csv::ReaderBuilder;
use ndarray::{ Array2, s};
use std::error::Error;
use std::fs::File;
mod math;
mod network;
mod matrix;
use matrix::Matrix;
use network::Network;
mod propogations;
mod model_saving;

fn read_csv(path_to_file: &str) -> Result<(Array2<f64>, Vec<u64>), Box<dyn Error>> {
    use csv::StringRecord;
    use ndarray::Array2;

    let file = File::open(path_to_file)?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);

    // Read the header to determine dimensions dynamically
    let headers = reader.headers()?.len();
    let mut raw_data = Vec::new();

    for result in reader.records() {
        let record: StringRecord = result?;
        let row: Vec<f64> = record
            .iter()
            .map(|value| value.parse::<f64>().unwrap_or(0.0)) // Convert to f64 or use 0.0 for null/empty
            .collect();
        raw_data.push(row);
    }

    // Convert the vector of rows into an Array2<f64>
    let num_rows = raw_data.len();
    let num_cols = headers;
    let flat_data: Vec<f64> = raw_data.into_iter().flatten().collect();
    let array_data = Array2::from_shape_vec((num_rows, num_cols), flat_data)?;

    // Split the data into labels (first column) and features (remaining columns)
    let y_train = array_data.column(0).to_owned().map(|&x| x as u64).to_vec(); // Labels as a vector
    let x_train = array_data.slice(s![.., 1..]).map(|&x| x as f64); // Features converted to f64

    Ok((x_train, y_train))
}

impl Network {
fn gradient_descent(
    &mut self,
    inputs: &Matrix,
    targets: &Matrix,
    epochs: usize,
) {

    for epoch in 0..epochs {
        let initial_rate = 0.1;
let decay_rate = 0.0001;
let learning_rate = initial_rate / (1.0 + decay_rate * epoch as f64); // Inverse learning rate

        // Perform backpropagation for each epoch
        self.backprop(inputs, targets, learning_rate);

        // Optionally, compute and print the loss to track progress
        let outputs = self.forward_prop(inputs.clone());
        if epoch % 2 == 0 {
                    // Calculate the loss
        let loss = self.categorical_cross_entropy(targets, &outputs);
        println!("Epoch {}: Loss = {:.6}", epoch, loss);
        
        }

    }
}
}
fn read_first_n_samples(path_to_file: &str, n: usize) -> Result<(Array2<f64>, Vec<u64>), Box<dyn Error>> {
    use csv::StringRecord;
    use ndarray::{s, Array2};
    use std::fs::File;

    let file = File::open(path_to_file)?;
    let mut reader = csv::ReaderBuilder::new().has_headers(true).from_reader(file);

    // Read all rows but limit to `n` samples
    let mut raw_data = Vec::new();
    for (i, result) in reader.records().enumerate() {
        if i >= n {
            break;
        }
        let record: StringRecord = result?;
        let row: Vec<f64> = record
            .iter()
            .map(|value| value.parse::<f64>().unwrap_or(0.0)) // Convert to f64 or use 0.0 for null/empty
            .collect();
        raw_data.push(row);
    }

    // Convert the vector of rows into an Array2<f64>
    let num_rows = raw_data.len();
    let num_cols = raw_data[0].len();
    let flat_data: Vec<f64> = raw_data.into_iter().flatten().collect();
    let array_data = Array2::from_shape_vec((num_rows, num_cols), flat_data)?;

    // Split the data into labels (first column) and features (remaining columns)
    let y_train: Vec<u64> = array_data.column(0).to_owned().iter().map(|&x| x as u64).collect();
    let x_train = array_data.slice(s![.., 1..]).to_owned();

    Ok((x_train, y_train))
}


fn main() {
    //For now use this for sake of overfitting, once model is able to learn, replace this with read_csv()
    match read_first_n_samples("fashion-mnist_test_first_2000.csv", 5) {
        Ok((x_train, y_train)) => {
            println!("Data successfully loaded.");
            println!("Shape of X (features): {:?}", x_train.dim());
            println!("Shape of Y (labels): {:?}", y_train.len());
            
            // Normalize and transpose features
            let mean = x_train.mean().unwrap();
            let std = x_train.std(0.0);
            let x_train_normalized = x_train.mapv(|x| (x - mean) / std).reversed_axes(); // Now (784, 42000)
            
            // Convert the feature matrix to your `Matrix` structure
            let input_matrix = Matrix {
                rows: x_train_normalized.nrows(),  
                columns: x_train_normalized.ncols(), 
                data: x_train_normalized.iter().cloned().collect(),
            };

            // Convert the labels to a one-hot encoded `Matrix`
            let num_classes = 10; // MNIST has 10 classes (digits 0-9)
            let target_matrix = Matrix {
                rows: num_classes,
                columns: y_train.len(),
                data: y_train
                    .iter()
                    .flat_map(|&label| {
                        (0..num_classes).map(move |class| if class == label.try_into().unwrap() { 1.0 } else { 0.0 })
                    })
                    .collect(),
            };

            println!(
                "Input matrix shape: {}x{}",
                input_matrix.rows, input_matrix.columns
            );
            println!(
                "Target matrix shape: {}x{}",
                target_matrix.rows, target_matrix.columns
            );

            // Initialize the network
            let mut network = Network::new(vec![784, 70, 10]); // Example architecture

            // Train the network
            network.gradient_descent(&input_matrix, &target_matrix, 3);

            // Save the model
            network.save_model("mnist_model.bin").expect("Failed to save model");

            // Load the model
            let mut loaded_network = Network::load_model("mnist_model.bin").expect("Failed to load model");

            // Test accuracy
            let test_inputs  = &input_matrix;
            let test_labels = &target_matrix;

            let accuracy = loaded_network.test_accuracy(test_inputs, test_labels);
            println!("Test Accuracy: {:.5}%", accuracy * 100.0);
            
        }
        Err(e) => {
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
    network.gradient_descent(&inputs, &targets,   10000);

    // Check if the loss decreases after training
    let loss_after: f64 = targets
        .subtract(&network.forward_prop(inputs.clone()))
        .data
        .iter()
        .map(|x| x * x)
        .sum::<f64>() / targets.data.len() as f64;

    assert!(loss_after < 1e-2, "Loss did not decrease sufficiently: {:.6}", loss_after);
}}
