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
            // Weighted sum: Z = W * A + b
            current = self.weights[i]
                .dot_product(&current) // Matrix multiplication: W * A
                .add(&self.biases[i]); // Add biases: + b
    
            if i < self.weights.len() - 1 {
                // Hidden layers: Apply ReLU
                current.leaky_relu(0.01);
            } else {
                // Output layer: Apply softmax
                current = current.softmax();
            }
    
            // Store the activations
            self.data.push(current.clone());
        }
    
        // The final activation is the output of the network
        current
    }
    
    pub fn backprop(
        &mut self,
        inputs: &Matrix,
        targets: &Matrix,
        learning_rate: f64,
    ) {
        // Validate input dimensions
        assert_eq!(self.layers[0], inputs.rows, "Mismatch in input size.");
        assert_eq!(self.layers.last().unwrap(), &targets.rows, "Mismatch in output size.");
    
        // Forward propagate to compute activations
        let outputs = self.forward_prop(inputs.clone());
    
        // Initialize gradients
        let mut d_weights = vec![];
        let mut d_biases = vec![];
    
        // Compute the error at the output layer
        let mut error = outputs.subtract(targets); // Error: Output - Target
    
        // Backward propagate through layers
        for i in (0..self.weights.len()).rev() {
            // Derivative of activation function
            let activation_derivative = if i == self.weights.len() - 1 {   
                // Output layer: softmax derivative
                outputs.subtract(targets) // dA = Output - Target (softmax + cross-entropy simplified)
            } else {
                // Hidden layers: ReLU derivative
                self.data[i + 1].leaky_relu_derivative(0.01)
            };
    
            // Gradient of weights and biases
            let delta = error.multiply(&activation_derivative);
            let d_weight = delta.dot_product(&self.data[i].transpose());
            let d_bias = delta.sum_columns(); // Aggregate bias gradients
    
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
            self.weights[i] = self.weights[i].subtract(&d_weights[i].scale(learning_rate));
            self.biases[i] = self.biases[i].subtract(&d_biases[i].scale(learning_rate));
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
    fn test_backprop_mnist() {
        // Network suitable for MNIST: input -> hidden -> output
        let mut network = Network::new(vec![784, 128, 10]);
        
        // Create mock input and target matrices
        let inputs = Matrix {
            rows: 784,
            columns: 1,
            data: vec![0.5; 784], // Example input: all pixels set to 0.8
        };
        let mut targets = Matrix {
            rows: 10,
            columns: 1,
            data: vec![0.0; 10], // Example target: one-hot encoded
        };

        // Set one of the target values to 1 (e.g., target class 3)
        targets.data[3] = 1.0;

        // Store initial weights and biases for comparison
        let initial_weights = network.weights.clone();
        let initial_biases = network.biases.clone();

        // Perform backpropagation
        network.backprop(&inputs, &targets, 0.001);

        // Ensure weights and biases have been updated
        for (initial, updated) in initial_weights.iter().zip(&network.weights) {
            assert!(
                !initial.data.iter().zip(&updated.data).all(|(a, b)| (a - b).abs() < 1e-6),
                "Weights did not change after backpropagation"
            );
        }

        for (initial, updated) in initial_biases.iter().zip(&network.biases) {
            assert!(
                !initial.data.iter().zip(&updated.data).all(|(a, b)| (a - b).abs() < 1e-6),
                "Biases did not change after backpropagation"
            );
        }

    }
    
}