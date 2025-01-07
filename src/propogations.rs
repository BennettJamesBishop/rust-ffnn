use crate::matrix::Matrix;
use crate::network::Network;

//This mod contains the forward propogation and back propogation functions

impl Network {
    pub fn forward_prop(&mut self, inputs: Matrix) -> Matrix {
        // Validate input dimensions
        assert!(
            self.layers[0] == inputs.rows,
            "Invalid number of inputs: expected {}, got {}",
            self.layers[0],
            inputs.rows
        );

        // Initialize activations with the input data
        let mut current = inputs;
        self.data = vec![current.clone()]; // Store input as the first "activation"
        // Propagate through each layer
        for i in 0..self.weights.len() {
            println!("Layer {} activations before forward propogation: {:?}", i, &current.data[..2]);
            // Weighted sum: Z = W * A + b
            current = self.weights[i]
                .dot_product(&current) // Matrix multiplication: W * A
                .add(&self.biases[i]); // Add biases: + b
    
            if i < self.weights.len() - 1 {
                // Hidden layers: Apply ReLU
                current.leaky_relu(0.001);
            } else {
                // Output layer: Apply softmax
                current = current.softmax();
            }
            println!("Layer {} activations after forward propogation: {:?}", i, &current.data[..2]);
            // Store the activations
            self.data.push(current.clone());
        }
        current
    }
    
    pub fn backprop(
        &mut self,
        predictions: &Matrix,
        targets: &Matrix,
        learning_rate: f64,
    ) {     
        // Initialize gradients
        let mut d_weights = vec![];
        let mut d_biases = vec![];
    
        // Compute the error at the output layer
        let mut error = predictions.subtract(targets); // Error: Output - Target
        
        // Backward propagate through layers
        for i in (0..self.weights.len()).rev() {
            // Derivative of activation function
            let activation_derivative = if i == self.weights.len() - 1 {   
                // Output layer: softmax derivative
                error.clone() // dA = Output - Target (softmax + cross-entropy simplified)
            } else {
                // Hidden layers: ReLU derivative
                self.data[i + 1].leaky_relu_derivative(0.001)
            };
    
            // Gradient of weights and biases
            let delta = error.multiply(&activation_derivative);
            let d_weight = delta.dot_product(&self.data[i].transpose());
            let d_bias = delta.average_columns(); // Average bias gradients
            
            // Store gradients
            d_weights.insert(0, d_weight);
            d_biases.insert(0, d_bias);
            
            // Compute error for the next layer
            if i > 0 {
                error = self.weights[i].transpose().dot_product(&delta);
            }
        }
    
        // Update weights and biases using gradients
        for i in 0..self.weights.len() {
           // println!("Before update, weights [layer {}]: {:?}", i, &self.weights[i].data[..5]);
            // println!("Before update, biases [layer {}]: {:?}", i, &self.weights[i].data[..5]);
            self.weights[i] = self.weights[i].subtract(&d_weights[i].scale(learning_rate));
            self.biases[i] = self.biases[i].subtract(&d_biases[i].scale(learning_rate));
            // println!("After update, weights [layer {}]: {:?}", i, &self.weights[i].data[..4]);
        }
    }
    
    pub fn train(
        &mut self,
        inputs: &Matrix,
        targets: &Matrix,
        epochs: usize,
    ) {
    
        for epoch in 0..epochs {
            let initial_rate = 0.0001;
            let decay_rate = 0.000001;
            let learning_rate = initial_rate / (1.0 + decay_rate * epoch as f64); // Inverse learning rate
    
            let predictions = self.forward_prop(inputs.clone());
            // Perform backpropagation for each epoch
            self.backprop(&predictions, targets, learning_rate);
    
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_prop_mnist() {
        // Network suitable for MNIST: input -> hidden -> output
        let mut network = Network::new(vec![784, 128, 10]);
    
        // Create a mock input matrix representing a single MNIST image
        // Input size: 784 rows (flattened 28x28 image), 1 column (single image)
        let inputs = Matrix {
            rows: 784,
            columns: 1,
            data: vec![0.8; 784], // Example data: all pixels set to 0.5
        };
    
        // Perform forward propagation
        let output = network.forward_prop(inputs);
    
        // Check output dimensions
        assert_eq!(output.rows, 10); // Output should have 10 rows (one for each digit class)
        assert_eq!(output.columns, 1); // Single column for the output vector
    
        // Ensure that the outputs are valid probabilities (e.g., between 0 and 1 if using sigmoid in the output layer)
        for &value in &output.data {
            assert!(value >= 0.0 && value <= 1.0, "Output value out of range: {}", value);
        }

        let sum: f64 = output.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Sum of probabilities is not 1: {}", sum);

    }

    #[test]
    fn test_backprop_function() {
        // Create a simple network
        let mut network = Network::new(vec![2,3,2]);

        // Define inputs and targets
        let inputs = Matrix {
            rows: 2,
            columns: 1,
            data: vec![1.0, -1.0], // Example input
        };
        let targets = Matrix {
            rows: 2,
            columns: 1,
            data: vec![0.0, 1.0], // Example target (one-hot encoded)
        };

        // Perform forward propagation
        let predictions = network.forward_prop(inputs.clone());

        // Perform backpropagation
        let original_weights = network.weights.clone();
        network.backprop(&predictions, &targets, 0.01);

        // Check that weights have been updated
        for (i, (orig, updated)) in original_weights.iter().zip(&network.weights).enumerate() {
            assert!(
                orig.data != updated.data,
                "Weights for layer {} were not updated during backpropagation.",
                i
            );
        }
    }

    #[test]
    fn test_train_function() {
        // Step 1: Create a simple network
        let mut network = Network::new(vec![2, 3, 2]);

        // Step 2: Define inputs and targets
        let inputs = Matrix {
            rows: 2,
            columns: 4, // Batch of 4 examples
            data: vec![1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.1, -0.1], // Example inputs
        };
        let targets = Matrix {
            rows: 2,
            columns: 4,
            data: vec![
                0.0, 1.0, // One-hot for example 1
                1.0, 0.0, // One-hot for example 2
                0.0, 1.0, // One-hot for example 3
                1.0, 0.0, // One-hot for example 4
            ],
        };

        // Step 3: Record initial weights
        let initial_weights: Vec<Matrix> = network.weights.clone();

        // Step 4: Perform initial forward pass and loss calculation
        let initial_predictions = network.forward_prop(inputs.clone());
        let initial_loss = network.categorical_cross_entropy(&targets, &initial_predictions);
        println!("Initial loss: {:.6}", initial_loss);

        // Step 5: Train the network
        network.train(&inputs, &targets, 20);

        // Step 6: Perform forward pass after training and calculate loss
        let predictions_after_training = network.forward_prop(inputs.clone());
        let loss_after_training = network.categorical_cross_entropy(&targets, &predictions_after_training);
        println!("Loss after training: {:.6}", loss_after_training);

        // Step 7: Ensure loss has decreased
        assert!(
            loss_after_training < initial_loss,
            "Loss did not decrease after training: initial = {:.6}, final = {:.6}",
            initial_loss,
            loss_after_training
        );

        // Step 8: Check that weights have been updated
        for (i, (initial, updated)) in initial_weights.iter().zip(&network.weights).enumerate() {
            assert!(
                initial.data != updated.data,
                "Weights for layer {} were not updated during training.",
                i
            );
        }
        }

}