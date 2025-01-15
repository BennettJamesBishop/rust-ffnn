use crate::matrix::Matrix;

//This module defines the Network struct and contains the function to initialize one
pub struct Network {
    pub layers: Vec<usize>,
    pub weights: Vec<Matrix>,
    pub biases: Vec<Matrix>,
    pub data: Vec<Matrix>,
    pub m_weights: Vec<Matrix>, // Momentum for weights
    pub v_weights: Vec<Matrix>, // RMSProp for weights
    pub m_biases: Vec<Matrix>,  // Momentum for biases
    pub v_biases: Vec<Matrix>,  // RMSProp for biases
}


impl Network {

    pub fn new(layers: Vec<usize>) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut m_weights = Vec::new();
        let mut v_weights = Vec::new();
        let mut m_biases = Vec::new();
        let mut v_biases = Vec::new();

        for i in 0..layers.len() - 1 {
            weights.push(Matrix::he_initialization(layers[i + 1], layers[i]));
            biases.push(Matrix::zeros(layers[i + 1], 1));
            m_weights.push(Matrix::zeros(layers[i + 1], layers[i]));
            v_weights.push(Matrix::zeros(layers[i + 1], layers[i]));
            m_biases.push(Matrix::zeros(layers[i + 1], 1));
            v_biases.push(Matrix::zeros(layers[i + 1], 1));
        }

        Network {
            layers,
            weights,
            biases,
            data: Vec::new(),
            m_weights,
            v_weights,
            m_biases,
            v_biases,
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
