
use crate::lin_alg::Matrix;

pub struct Network {
    layers: Vec<usize>, //number of neurons in each layer eg: [16, 12, 10]
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    data: Vec<Matrix>,
}

impl Network {

    pub fn new(layers: Vec<usize>) -> Self {
        // Initialize weights and biases
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..layers.len() - 1 {
            // Randomly initialize weights as matrices with dimensions (current_layer x next_layer)
            weights.push(Matrix::random(layers[i+1], layers[i]));
            
            // Randomly initialize biases as column matrices (1 x next_layer)
            biases.push(Matrix::random(layers[i + 1], 1));
        }

        // Create an empty data vector
        let data = Vec::new();

        Network {
            layers,
            weights,
            biases,
            data: data,
        }
    }

    pub fn feed_forward(&mut self, inputs: Matrix) -> Matrix {
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
    fn test_feed_forward_mnist() {
        // Define a network suitable for MNIST: input -> hidden -> output
        let mut network = Network::new(vec![784, 128, 10]);
    
        // Create a mock input matrix representing a single MNIST image
        // Input size: 784 rows (flattened 28x28 image), 1 column (single image)
        let inputs = Matrix {
            rows: 784,
            columns: 1,
            data: vec![0.8; 784], // Example data: all pixels set to 0.5
        };
    
        // Perform forward propagation
        let output = network.feed_forward(inputs);
    
        // Check output dimensions
        assert_eq!(output.rows, 10); // Output should have 10 rows (one for each digit class)
        assert_eq!(output.columns, 1); // Single column for the output vector
    
        // Optional: Print the output for debugging
        println!("Final output activations: {:?}", output.data);
    
        // Ensure that the outputs are valid probabilities (e.g., between 0 and 1 if using sigmoid in the output layer)
        for &value in &output.data {
            assert!(value >= 0.0 && value <= 1.0, "Output value out of range: {}", value);
        }
    }
    
}
