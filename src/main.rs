use csv::ReaderBuilder;
use ndarray::{ Array2, s};
use ndarray_csv::Array2Reader;
use std::error::Error;
use std::fs::File;
mod math;
mod network;
mod matrix;
mod propogations;

fn read_csv(path_to_file: &str) -> Result<Array2<u64>, Box<dyn Error>> {
    let file = File::open(path_to_file)?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
    Ok(reader.deserialize_array2((42000, 785))?)
}




fn main() {
    // let data = read_csv("train.csv");
    match read_csv("train.csv") {
        Ok(data) => {
            // If successful, print the shape of the array
            
            let (m, n) = data.dim();
            println!("Array shape: {:?}, {:?}", m, n);
                // Split the data into Y (first column) and X (remaining columns)
    let y_train = data.column(0).to_owned(); // First column as labels (1D array)
    let x_train = data.slice(s![.., 1..]).to_owned(); // Remaining columns as features

    // Print sizes
    println!("Shape of Y (labels): {:?}", y_train.dim());
    println!("Shape of X (features): {:?}", x_train.dim());

        }
        Err(e) => {
            // Handle errors gracefully
            eprintln!("Error reading CSV: {}", e);
        }
    }
}

