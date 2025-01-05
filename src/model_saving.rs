use crate::matrix::Matrix;
use crate::network::Network;
use std::io::{Read, Write};

impl Network {

    pub fn categorical_cross_entropy(&self, targets: &Matrix, predictions: &Matrix) -> f64 {
        assert_eq!(
            targets.rows, predictions.rows,
            "Targets and predictions must have the same number of rows (classes)"
        );
        assert_eq!(
            targets.columns, predictions.columns,
            "Targets and predictions must have the same number of columns (samples)"
        );

        let epsilon = 1e-10; // Prevent log(0)
        let mut loss = 0.0;

        for col in 0..targets.columns {
            for row in 0..targets.rows {
                let t = targets.data[row * targets.columns + col]; // Target value
                let p = predictions.data[row * predictions.columns + col].max(epsilon); // Predicted probability
                loss += t * p.ln();
            }
        }

        -loss / targets.columns as f64 // Normalize by the number of samples
    }

    pub fn save_model(&self, file_path: &str) -> Result<(), std::io::Error> {
        let mut file = std::fs::File::create(file_path)?;

        // Write layers
        let layer_count = self.layers.len() as u64;
        file.write_all(&layer_count.to_le_bytes())?;
        for &layer in &self.layers {
            file.write_all(&(layer as u64).to_le_bytes())?;
        }

        // Write weights
        let weight_count = self.weights.len() as u64;
        file.write_all(&weight_count.to_le_bytes())?;
        for matrix in &self.weights {
            matrix.write_to_file(&mut file)?;
        }

        // Write biases
        let bias_count = self.biases.len() as u64;
        file.write_all(&bias_count.to_le_bytes())?;
        for matrix in &self.biases {
            matrix.write_to_file(&mut file)?;
        }

        Ok(())
    }

    pub fn load_model(file_path: &str) -> Result<Self, std::io::Error> {
        let mut file = std::fs::File::open(file_path)?;

        // Read layers
        let mut buffer = [0u8; 8];
        file.read_exact(&mut buffer)?;
        let layer_count = u64::from_le_bytes(buffer) as usize;

        let mut layers = Vec::with_capacity(layer_count);
        for _ in 0..layer_count {
            file.read_exact(&mut buffer)?;
            layers.push(u64::from_le_bytes(buffer) as usize);
        }

        // Read weights
        file.read_exact(&mut buffer)?;
        let weight_count = u64::from_le_bytes(buffer) as usize;

        let mut weights = Vec::with_capacity(weight_count);
        for _ in 0..weight_count {
            weights.push(Matrix::read_from_file(&mut file)?);
        }

        // Read biases
        file.read_exact(&mut buffer)?;
        let bias_count = u64::from_le_bytes(buffer) as usize;

        let mut biases = Vec::with_capacity(bias_count);
        for _ in 0..bias_count {
            biases.push(Matrix::read_from_file(&mut file)?);
        }

        Ok(Network {
            layers,
            weights,
            biases,
            data: Vec::new(),
        })
    }

pub fn test_accuracy(&mut self, inputs: &Matrix, targets: &Matrix) -> f64 {
    assert_eq!(
        inputs.columns, targets.columns,
        "Number of samples in inputs and targets must match"
    );

    // Perform forward propagation for all inputs at once
    let outputs = self.forward_prop(inputs.clone());

    // Debug softmax outputs
    let sum_probs: Vec<f64> = outputs
        .data
        .chunks(outputs.rows)
        .map(|column| column.iter().sum())
        .collect();
    println!("Sum of probabilities for each sample: {:?}", &sum_probs[..10]);

    // Predicted classes
    let predicted_classes: Vec<usize> = outputs
        .data
        .chunks(outputs.rows)
        .map(|column| {
            column
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index)
                .unwrap()
        })
        .collect();
    println!("Predicted classes: {:?}", &predicted_classes[..10]);

    // True classes
    let true_classes: Vec<usize> = targets
        .data
        .chunks(targets.rows)
        .map(|column| {
            column
                .iter()
                .enumerate()
                .find(|(_, &value)| value == 1.0)
                .map(|(index, _)| index)
                .unwrap()
        })
        .collect();
    println!("True classes: {:?}", &true_classes[..10]);

    // Calculate accuracy
    let correct = predicted_classes
        .iter()
        .zip(true_classes.iter())
        .filter(|(&predicted, &true_class)| predicted == true_class)
        .count();

    correct as f64 / inputs.columns as f64
}

}

impl Matrix {
    pub fn write_to_file(&self, file: &mut std::fs::File) -> Result<(), std::io::Error> {
        file.write_all(&(self.rows as u64).to_le_bytes())?;
        file.write_all(&(self.columns as u64).to_le_bytes())?;
        for &value in &self.data {
            file.write_all(&value.to_le_bytes())?;
        }
        Ok(())
    }

    pub fn read_from_file(file: &mut std::fs::File) -> Result<Self, std::io::Error> {
        let mut buffer = [0u8; 8];

        // Read dimensions
        file.read_exact(&mut buffer)?;
        let rows = u64::from_le_bytes(buffer) as usize;

        file.read_exact(&mut buffer)?;
        let columns = u64::from_le_bytes(buffer) as usize;

        // Read data
        let mut data = Vec::with_capacity(rows * columns);
        for _ in 0..rows * columns {
            let mut value_buffer = [0u8; 8];
            file.read_exact(&mut value_buffer)?;
            data.push(f64::from_le_bytes(value_buffer));
        }

        Ok(Matrix { rows, columns, data })
    }


    pub fn argmax(&self) -> usize {
        self.data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }
}
