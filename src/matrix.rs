use rand::Rng;
//This module defines the Matrix struct and contains the function to initialize one
#[derive(Clone)]
pub struct Matrix {
    pub rows: usize,
    pub columns: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn new(rows: usize, columns: usize) -> Matrix {
        let mut buffer = Vec::<f64>::with_capacity(rows * columns);

        for _ in 0..rows * columns {
            let num = rand::thread_rng().gen_range(0.0..1.0);
            buffer.push(num);
        }

        Matrix {
            rows: rows,
            columns: columns,
            data: buffer,
        }
    }

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
use super::*;
    #[test]
    fn test_matrix_initializer() {
        let rows = 3;
        let columns = 4;
        let matrix = Matrix::new(rows, columns);

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
}
