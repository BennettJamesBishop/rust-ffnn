mod math;
mod network;
mod matrix;
use matrix::Matrix;
use network::Network;
mod propogations;
mod helpers;
use crate::helpers::read_first_n_samples; //Add read_csv once necessary

fn main() {
    //For now use read_first_n_samples() for small training, once model is able to learn, replace this line with:
    //match read_csv("fashion-mnist_test_first_2000.csv") {
    match read_first_n_samples("fashion-mnist_test_first_2000.csv", 100) {
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
            network.train(&input_matrix, &target_matrix, 3);

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
    network.train(&inputs, &targets,   10000);

    // Check if the loss decreases after training
    let loss_after: f64 = targets
        .subtract(&network.forward_prop(inputs.clone()))
        .data
        .iter()
        .map(|x| x * x)
        .sum::<f64>() / targets.data.len() as f64;

    assert!(loss_after < 1e-2, "Loss did not decrease sufficiently: {:.6}", loss_after);
}}
