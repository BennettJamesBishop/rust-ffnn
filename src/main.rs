mod math;
mod network;
mod matrix;
use matrix::Matrix;
use network::Network;
mod propogations;
mod helpers;
use crate::helpers::read_first_n_samples; //Add read_csv once necessary
fn balance_dataset(inputs: &Matrix, targets: &Matrix) -> (Matrix, Matrix) {
    let mut balanced_inputs = Vec::new();
    let mut balanced_targets = Vec::new();

    // Count occurrences of each class
    let class_counts = targets.data.chunks(targets.rows).fold(vec![0; targets.rows], |mut counts, row| {
        for (i, &val) in row.iter().enumerate() {
            if val == 1.0 {
                counts[i] += 1;
            }
        }
        counts
    });

    // Find the maximum class count
    let max_count = *class_counts.iter().max().unwrap();

    // Oversample underrepresented classes
    for (class_idx, &count) in class_counts.iter().enumerate() {
        // Find all samples belonging to this class
        let class_samples: Vec<usize> = targets
            .data
            .chunks(targets.rows)
            .enumerate()
            .filter(|(_, row)| row[class_idx] == 1.0)
            .map(|(i, _)| i)
            .collect();

        // Oversample to reach max_count
        let mut oversampled_indices = class_samples.clone();
        while oversampled_indices.len() < max_count {
            oversampled_indices.push(class_samples[rand::random::<usize>() % class_samples.len()]);
        }

        // Add samples to the balanced dataset
        for idx in oversampled_indices {
            balanced_inputs.extend_from_slice(&inputs.data[idx * inputs.rows..(idx + 1) * inputs.rows]);
            balanced_targets.extend_from_slice(&targets.data[idx * targets.rows..(idx + 1) * targets.rows]);
        }
    }

    // Return the balanced dataset
    (
        Matrix {
            rows: inputs.rows,
            columns: balanced_inputs.len() / inputs.rows,
            data: balanced_inputs,
        },
        Matrix {
            rows: targets.rows,
            columns: balanced_targets.len() / targets.rows,
            data: balanced_targets,
        },
    )
}

fn main() {
    //Once model is able to learn, replace read_first_n_samples() with read_csv()
    match read_first_n_samples("fashion-mnist_test_first_2000.csv", 2000) {
        Ok((x_train, y_train)) => {
            println!("Data successfully loaded.");
            println!("Shape of X (features): {:?}", x_train.dim());
            println!("Shape of Y (labels): {:?}", y_train.len());
            //println!("Original Class distribution: {:?}", y_train.iter());
            
            // Normalize and transpose features
            let mean = x_train.mean().unwrap();
            let std = x_train.std(0.0);
            let x_train_normalized = x_train.mapv(|x| (x - mean) / std).reversed_axes(); // Now (784, 42000)
            
            // Convert input data to `Matrix` structure
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

            let (new_input_matrix, new_target_matrix) = balance_dataset(&input_matrix, &target_matrix);

            println!(
                "Input matrix shape: {}x{}",
                new_input_matrix.rows, new_input_matrix.columns
            );
            println!(
                "Target matrix shape: {}x{}",
                new_target_matrix.rows, new_target_matrix.columns
            );

            // Initialize the network
            let mut network = Network::new(vec![784, 256, 128, 10]); // Example architecture

            // Train the network
            network.train_adam(&input_matrix, &target_matrix, 500, 0.9, 0.999, 1.0e-8 );

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

