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
mod model_saving;

fn read_csv(path_to_file: &str) -> Result<(Array2<f64>, Vec<u64>), Box<dyn Error>> {
    let file = File::open(path_to_file)?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);

    // Read the entire CSV into a 2D array
    let raw_data: Array2<u64> = reader.deserialize_array2((2000, 785))?;
    
    // Split the data into labels (first column) and features (remaining columns)
    let y_train = raw_data.column(0).to_owned().to_vec(); // Labels as a vector
    let x_train = raw_data.slice(s![.., 1..]).map(|&x| x as f64); // Features converted to f64

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
        let initial_learning_rate = 0.05;
        let decay_rate = 0.05;
        let learning_rate = initial_learning_rate / (1.0 + epoch as f64 * decay_rate); // Decay rate < 1.0
        // Perform backpropagation for each epoch
        self.backprop(inputs, targets, learning_rate);

        // Optionally, compute and print the loss to track progress
        let outputs = self.forward_prop(inputs.clone());
        if epoch % 5 == 0 {
                    // Calculate the loss
        let loss = self.categorical_cross_entropy(targets, &outputs);
        println!("Epoch {}: Loss = {:.6}", epoch, loss);
        
        }

    }
}
}

fn main() {
    match read_csv("output_first_2000.csv") {
        Ok((x_train, y_train)) => {
            println!("Data successfully loaded.");
            println!("Shape of X (features): {:?}", x_train.dim());
            println!("Shape of Y (labels): {:?}", y_train.len());

            // Normalize and transpose features
            let x_train_normalized = x_train.mapv(|x| x / 255.0).reversed_axes(); // Now (784, 42000)

            // Convert the feature matrix to your `Matrix` structure
            let input_matrix = Matrix {
                rows: x_train_normalized.nrows(),  // 784
                columns: x_train_normalized.ncols(), // 42000
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
            let mut network = Network::new(vec![784, 128, 10]); // Example architecture

            // Train the network
            network.gradient_descent(&input_matrix, &target_matrix, 10);

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
