use csv::ReaderBuilder;
use std::error::Error;
use std::fs::File;
use crate::matrix::Matrix;
use crate::network::Network;
use std::io::{Read, Write};
use ndarray::{ Array2, s};

// MATRIX HELPER FUNCTS: write_to_file, read_from_file, 
impl Matrix {
    //Writes matrix to csv file
    pub fn write_to_file(&self, file: &mut std::fs::File) -> Result<(), std::io::Error> {
        file.write_all(&(self.rows as u64).to_le_bytes())?;
        file.write_all(&(self.columns as u64).to_le_bytes())?;
        for &value in &self.data {
            file.write_all(&value.to_le_bytes())?;
        }
        Ok(())
    }

    //Reads matrix from csv file
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
}


// NETWORK HELPER FUNCTS: save_model, load_model, categorical_cross_entropy, test_accuracy
impl Network {
    // Saves model to bin file
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

        
        //Loads model from bin file
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


    // Categorical cross-entropy loss, used for printing, not for learning
    pub fn categorical_cross_entropy(&self, targets: &Matrix, predictions: &Matrix) -> f64 {
        // Ensure dimensions match: targets and predictions should have the same size
        assert_eq!(targets.rows, predictions.rows);
        assert_eq!(targets.columns, predictions.columns);

        let mut total_loss = 0.0;

        for i in 0..targets.rows {
            for j in 0..targets.columns {
                // Avoid log(0) by adding a small epsilon
                let predicted = predictions.data[i * predictions.columns + j].max(1e-12);
                let target = targets.data[i * targets.columns + j];
                total_loss += target * predicted.ln(); // Cross-entropy contribution
            }
        }

        // Return the average loss over all examples
        -total_loss / targets.columns as f64
    }
        

    // Accuracy used for printing, not for calculations
    pub fn test_accuracy(&mut self, inputs: &Matrix, targets: &Matrix) -> f64 {
        assert_eq!(
            inputs.columns, targets.columns,
            "Number of samples in inputs and targets must match"
        );

        // Perform forward propagation for all inputs at once
        let outputs = self.forward_prop(inputs.clone());
        println!("Output probabailities: {:?}", outputs.data[..10].to_vec());
        // Debug softmax outputs
        let sum_probs: Vec<f64> = outputs
            .data
            .chunks(outputs.rows)
            .map(|column| column.iter().sum())
            .collect();
        println!("Sum of probabilities for each sample: {:?}", &sum_probs[..2]);

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
        println!("Predicted classes: {:?}", &predicted_classes[..4]);

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
        println!("True classes: {:?}", &true_classes[..4]);

        // Calculate accuracy
        let correct = predicted_classes
            .iter()
            .zip(true_classes.iter())
            .filter(|(&predicted, &true_class)| predicted == true_class)
            .count();

        correct as f64 / inputs.columns as f64
    }
}


//OTHER HELPER FUNCTS: read_first_n_samples, read_csv

// Reads first n samples from dataset
pub fn read_first_n_samples(path_to_file: &str, n: usize) -> Result<(Array2<f64>, Vec<u64>), Box<dyn Error>> {
    use csv::StringRecord;
    use ndarray::{s, Array2};
    use std::fs::File;

    let file = File::open(path_to_file)?;
    let mut reader = csv::ReaderBuilder::new().has_headers(true).from_reader(file);

    // Read all rows but limit to `n` samples
    let mut raw_data = Vec::new();
    for (i, result) in reader.records().enumerate() {
        if i >= n {
            break;
        }
        let record: StringRecord = result?;
        let row: Vec<f64> = record
            .iter()
            .map(|value| value.parse::<f64>().unwrap_or(0.0)) // Convert to f64 or use 0.0 for null/empty
            .collect();
        raw_data.push(row);
    }

    // Convert the vector of rows into an Array2<f64>
    let num_rows = raw_data.len();
    let num_cols = raw_data[0].len();
    let flat_data: Vec<f64> = raw_data.into_iter().flatten().collect();
    let array_data = Array2::from_shape_vec((num_rows, num_cols), flat_data)?;

    // Split the data into labels (first column) and features (remaining columns)
    let y_train: Vec<u64> = array_data.column(0).to_owned().iter().map(|&x| x as u64).collect();
    let x_train = array_data.slice(s![.., 1..]).to_owned();

    Ok((x_train, y_train))
}


// Reads CSV file into an array
pub fn _read_csv(path_to_file: &str) -> Result<(Array2<f64>, Vec<u64>), Box<dyn Error>> {
    use csv::StringRecord;
    use ndarray::Array2;

    let file = File::open(path_to_file)?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);

    // Read the header to determine dimensions dynamically
    let headers = reader.headers()?.len();
    let mut raw_data = Vec::new();

    for result in reader.records() {
        let record: StringRecord = result?;
        let row: Vec<f64> = record
            .iter()
            .map(|value| value.parse::<f64>().unwrap_or(0.0)) // Convert to f64 or use 0.0 for null/empty
            .collect();
        raw_data.push(row);
    }

    // Convert the vector of rows into an Array2<f64>
    let num_rows = raw_data.len();
    let num_cols = headers;
    let flat_data: Vec<f64> = raw_data.into_iter().flatten().collect();
    let array_data = Array2::from_shape_vec((num_rows, num_cols), flat_data)?;

    // Split the data into labels (first column) and features (remaining columns)
    let y_train = array_data.column(0).to_owned().map(|&x| x as u64).to_vec(); // Labels as a vector
    let x_train = array_data.slice(s![.., 1..]).map(|&x| x as f64); // Features converted to f64

    Ok((x_train, y_train))
}


