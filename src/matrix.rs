use rand::Rng;
//This module defines the Matrix struct and contains the function to initialize one
#[derive(Clone)]
#[derive(Debug)]
pub struct Matrix {
    pub rows: usize,
    pub columns: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn he_initialization(rows: usize, columns: usize) -> Matrix {
        let scale = (2.0 / columns as f64).sqrt(); // Scale by incoming connections
        let data = (0..rows * columns)
            .map(|_| rand::thread_rng().gen_range(-scale..scale))
            .collect();
        Matrix { rows, columns, data }
    }
    
    pub fn zeros(rows: usize, columns: usize) -> Matrix {
        Matrix {
            rows,
            columns,
            data: vec![0.0; rows * columns],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*; // Import the necessary structs and functions

    #[test]
    fn test_zeros_function() {
        let rows = 3;
        let columns = 4;
        let matrix = Matrix::zeros(rows, columns);

        // Check dimensions
        assert_eq!(matrix.rows, rows);
        assert_eq!(matrix.columns, columns);

        // Check that all values are zero
        assert!(matrix.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_he_initialization_function() {
        let rows = 3;
        let columns = 4;
        let matrix = Matrix::he_initialization(rows, columns);

        // Check dimensions
        assert_eq!(matrix.rows, rows);
        assert_eq!(matrix.columns, columns);

        // Check that the values fall within the expected range
        let scale = (2.0 / columns as f64).sqrt();
        assert!(matrix.data.iter().all(|&x| x >= -scale && x <= scale));
    }
}

