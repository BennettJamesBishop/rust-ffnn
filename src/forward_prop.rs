use crate::matrix::Matrix;
use crate::neural_net::Network;
use rand::Rng;
//This mod contains functions for forward prop, including relu, softmax, random matrix initializer, and forward prop itself

impl Matrix {
    pub fn relu(&mut self) {
        // Apply ReLU activation in-place
        for i in 0..self.data.len() {
            if self.data[i] < 0.0 {
                self.data[i] = 0.0;
            }
        }
    }

        /// Apply the softmax function to the entire flattened `data` vector
    pub fn softmax(&self) -> Matrix {
        // Find the maximum value in the data for numerical stability
        let max_val = self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Subtract max_val from each element, exponentiate, and store results
        let exp_values: Vec<f64> = self.data.iter().map(|&x| (x - max_val).exp()).collect();

        // Compute the sum of all exponentials
        let sum_exp: f64 = exp_values.iter().sum();

        // Normalize each value by dividing by the sum of exponentials
        let softmax_data: Vec<f64> = exp_values.iter().map(|&x| x / sum_exp).collect();

        // Return a new matrix with the softmax values
        Matrix {
            rows: self.rows,
            columns: self.columns,
            data: softmax_data,
        }}

    pub fn random_matrix(rows: usize, columns: usize) -> Matrix {
        let mut buffer = Vec::<f64>::with_capacity(rows * columns);

        for _ in 0..rows*columns {
            let num = rand::thread_rng().gen_range(0.0..1.0);
            buffer.push(num);
        }

        Matrix {
            rows: rows,
            columns: columns,
            data: buffer
        }
    }

}

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
        for i in 0..self.weights.len()  {
            // Weighted sum: Z = W * A + b
            current = self.weights[i]
                .dot_product(&current) // Matrix multiplication: W * A
                .add(&self.biases[i]); // Add biases: + b

            if i < self.weights.len() - 1 {
                // Hidden layers: Apply ReLU
                current.relu();
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_on_data() {
        let matrix = Matrix {
            rows: 3,
            columns: 3,
            data: vec![2500.0, 2300.0, 1000.0, 2500.0, 2300.0, 1500.0, 2500.0, 2300.0, 1500.0],
        };

        // Apply softmax to the flattened data vector
        let result = matrix.softmax();

        // Expected results after applying softmax
        let expected = vec![0.3333,0.0,0.0,0.3333,0.0,0.0,0.3333,0.0,0.0];

        // Check that each value is approximately correct
        for (result, &expected_value) in result.data.iter().zip(expected.iter()) {
            assert!(
                (result - expected_value).abs() < 1e-3,
                "Expected {}, got {}",
                expected_value,
                result
            );
        }

        // Validate softmax property: The sum of all probabilities in the flattened vector should be 1
        let total_sum: f64 = result.data.iter().sum();
        assert!(
            (total_sum - 1.0).abs() < 1e-5,
            "Sum of probabilities is not 1: {}",
            total_sum
        );
    }

    #[test]
    fn test_random_matrix() {
        let rows = 3;
        let columns = 4;
        let matrix = Matrix::random_matrix(rows, columns);

        // Check the dimensions
        assert_eq!(matrix.rows, rows);
        assert_eq!(matrix.columns, columns);

        // Check the data length
        assert_eq!(matrix.data.len(), rows * columns);

        // Check that all elements are within the range [0.0, 1.0)
        for value in matrix.data {
            assert!(value >= 0.0 && value < 1.0);
        }
    }

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
}