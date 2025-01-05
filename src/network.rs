use crate::matrix::Matrix;

//This module defines the Network struct and contains the function to initialize one

pub struct Network {
    pub layers: Vec<usize>, //number of neurons in each layer eg: [16, 12, 10]
    pub weights: Vec<Matrix>,
    pub biases: Vec<Matrix>,
    pub data: Vec<Matrix>,
}

impl Network {

    pub fn new(layers: Vec<usize>) -> Self {
        // Initialize weights and biases
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..layers.len() - 1 {
            // Randomly initialize weights as matrices with dimensions (current_layer x next_layer)
            weights.push(Matrix::he_initialization(layers[i+1], layers[i]));
            
            // Randomly initialize biases as column matrices (1 x next_layer)
            biases.push(Matrix::zeros(layers[i + 1], 1)); 
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


}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_initialization() {
        // Define the network architecture (number of neurons in each layer)
        let layers = vec![4, 3, 2];

        // Create a new network
        let network = Network::new(layers.clone());

        // Check the number of layers matches the input
        assert_eq!(network.layers, layers, "Layers are not correctly initialized");

        // Check the number of weight matrices (one for each connection between layers)
        assert_eq!(
            network.weights.len(),
            layers.len() - 1,
            "Incorrect number of weight matrices"
        );

        // Check the dimensions of each weight matrix
        for i in 0..network.weights.len() {
            assert_eq!(
                network.weights[i].rows,
                layers[i + 1],
                "Incorrect number of rows in weight matrix {}",
                i
            );
            assert_eq!(
                network.weights[i].columns,
                layers[i],
                "Incorrect number of columns in weight matrix {}",
                i
            );
        }

        // Check the number of bias matrices (one for each layer except input layer)
        assert_eq!(
            network.biases.len(),
            layers.len() - 1,
            "Incorrect number of bias matrices"
        );

        // Check the dimensions of each bias matrix
        for i in 0..network.biases.len() {
            assert_eq!(
                network.biases[i].rows,
                layers[i + 1],
                "Incorrect number of rows in bias matrix {}",
                i
            );
            assert_eq!(
                network.biases[i].columns,
                1,
                "Bias matrix {} should have 1 column",
                i
            );
        }

        // Check the data vector is empty upon initialization
        assert!(
            network.data.is_empty(),
            "Data vector should be empty upon initialization"
        );
    }
  
    
}
