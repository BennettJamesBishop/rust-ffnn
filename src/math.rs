use crate::matrix::Matrix;

//This mod contains linear algebra equations for the Matrix (add, subtract, elemt-wise multiply, dot product)

impl Matrix {
    pub fn add(&self, matrix2: &Matrix) -> Matrix {
    if self.rows != matrix2.rows {
        panic!(
            "Cannot add matrices: row mismatch. Matrix 1 has {} rows, Matrix 2 has {} rows.",
            self.rows, matrix2.rows
        );
    }

    let mut buffer = Vec::<f64>::with_capacity(self.rows * self.columns);

    if self.columns == matrix2.columns {
        // Standard element-wise addition
        for i in 0..self.data.len() {
            buffer.push(self.data[i] + matrix2.data[i]);
        }
    } else if matrix2.columns == 1 {
        // Broadcasting: Add bias to each column of the matrix
        for row in 0..self.rows {
            for col in 0..self.columns {
                buffer.push(self.data[row * self.columns + col] + matrix2.data[row]);
            }
        }
    } else {
        panic!(
            "Cannot add matrices: column mismatch. Matrix 1 has {} columns, Matrix 2 has {} columns.",
            self.columns, matrix2.columns
        );
    }

    Matrix {
        rows: self.rows,
        columns: self.columns,
        data: buffer
    }
    }

    /// Sum the columns of the matrix (reduce to a single column vector)
    pub fn sum_columns(&self) -> Matrix {
        let mut result = vec![0.0; self.rows];

        for row in 0..self.rows {
            for col in 0..self.columns {
                result[row] += self.data[row * self.columns + col];
            }
        }

        Matrix {
            rows: self.rows,
            columns: 1,
            data: result,
        }
    }

    /// Average the columns of the matrix (reduce to a single column vector)
    pub fn average_columns(&self) -> Matrix {
        let sum_matrix = self.sum_columns(); // Sum the columns
        let batch_size = self.columns as f64;

        Matrix {
            rows: sum_matrix.rows,
            columns: 1,
            data: sum_matrix.data.iter().map(|&val| val / batch_size).collect(),
        }
    }

pub fn subtract(&self, matrix2: &Matrix) -> Matrix {
    // Ensure both matrices have the same dimensions
    assert_eq!(self.rows, matrix2.rows, "Row count mismatch in subtract()");
    assert_eq!(self.columns, matrix2.columns, "Column count mismatch in subtract()");

    // Perform element-wise subtraction
    let data: Vec<f64> = self
        .data
        .iter()
        .zip(&matrix2.data)
        .map(|(a, b)| a - b)
        .collect();

    // Return the result as a new Matrix
    Matrix {
        rows: self.rows,
        columns: self.columns,
        data,
    }
    }

    pub fn multiply(&self, matrix2: &Matrix) -> Matrix {
        if self.rows != matrix2.rows  {
            panic!("Cant multiply, matrices have different number of rows");
        } else if self.columns != matrix2.columns  {
            panic!("Cant multiply, matrices have different columns");
        }
        
        let mut buffer = Vec::<f64>::with_capacity(self.rows * self.columns);
        for i in 0..self.data.len() {
                let result = self.data[i] * matrix2.data[i];
                buffer.push(result);
            }
        Matrix {
            rows: self.rows,
            columns: self.columns,
            data: buffer
        }
    }

    pub fn scale(&self, scalar: f64) -> Matrix {
        let data: Vec<f64> = self.data.iter().map(|&value| value * scalar).collect();

        Matrix {
            rows: self.rows,
            columns: self.columns,
            data,
        }
    }

    pub fn dot_product(&self, matrix2: &Matrix) -> Matrix {
        if self.columns != matrix2.rows  {
            panic!("Cant do dot product, matrices have different wrong dimensions");
        }

        let mut product = vec![0.0; self.rows * matrix2.columns];

        for row in 0..self.rows {
            for col in 0..matrix2.columns {
                let mut sum = 0.0;
                for shared_dim in 0..self.columns {
                    sum += self.data[row * self.columns + shared_dim] * matrix2.data[shared_dim * matrix2.columns + col]
                }
                product[row * matrix2.columns + col] = sum;
            }
        }

        Matrix {
            rows: self.rows,
            columns: matrix2.columns,
            data: product
        }
    }                  

    pub fn transpose(&self) -> Matrix {
        // Initialize a vector to hold the transposed data
        let mut transposed_data = vec![0.0; self.rows * self.columns];
        
        // Fill in the transposed data
        for i in 0..self.rows {
            for j in 0..self.columns {
                // Transpose index calculation
                transposed_data[j * self.rows + i] = self.data[i * self.columns + j];
            }
        }

        // Create the new transposed matrix
        Matrix {
            rows: self.columns,
            columns: self.rows,
            data: transposed_data,
        }
    }

    pub fn leaky_relu(&mut self, alpha: f64) {
        self.data.iter_mut().for_each(|x| {
            *x = if *x > 0.0 { *x } else { alpha * *x };
        });
    }

    pub fn leaky_relu_derivative(&self, alpha: f64) -> Matrix {
        let mut derivative = self.clone();
        derivative.data.iter_mut().for_each(|x| {
            *x = if *x > 0.0 { 1.0 } else { alpha };
        });
        derivative
    }

    pub fn softmax(&self) -> Matrix {
        let mut result = self.clone();
        for col in 0..self.columns {
            let column_start = col * self.rows;
            let column_end = column_start + self.rows;
            let column = &self.data[column_start..column_end];
    
            let max_val = column.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_column: Vec<f64> = column.iter().map(|x| (x - max_val).exp()).collect();
    
            let sum_exp: f64 = exp_column.iter().sum();
    
            for (i, value) in exp_column.iter().enumerate() {
                result.data[column_start + i] = value / sum_exp;
            }
        }
        result
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_add_matrices() {
        let matrix1 = Matrix {
            rows: 2,
            columns: 2,
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let matrix2 = Matrix {
            rows: 2,
            columns: 2,
            data: vec![5.0, 6.0, 7.0, 8.0],
        };

        let result = matrix1.add(&matrix2);

        // Check dimensions
        assert_eq!(result.rows, 2);
        assert_eq!(result.columns, 2);

        // Check data
        assert_eq!(result.data, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_add_matrices_different_columns() {
        let matrix1 = Matrix {
            rows: 3,
            columns: 2,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        
        let matrix2 = Matrix {
            rows: 3,
            columns: 1,
            data: vec![0.5, 1.0, 1.5],
        };
        
        let result = matrix1.add(&matrix2);
        println!("{:?}", result.data); // [1.5, 2.5, 4.0, 5.0, 6.5, 7.5]
        // Check dimensions
        assert_eq!(result.rows, 3);
        assert_eq!(result.columns, 2);

        // Check data
        assert_eq!(result.data, vec![1.5, 2.5, 4.0, 5.0, 6.5, 7.5]);
    }        
    
    #[test]
    #[should_panic(expected = "Cannot add matrices: row mismatch. Matrix 1 has 2 rows, Matrix 2 has 3 rows.")]
    fn test_add_matrices_different_rows() {
        let matrix1 = Matrix {
            rows: 2,
            columns: 2,
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let matrix2 = Matrix {
            rows: 3,
            columns: 2,
            data: vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        };

        matrix1.add(&matrix2); // Should panic
    }

    #[test]
    fn test_subtract_matrices() {
        let matrix1 = Matrix {
            rows: 2,
            columns: 2,
            data: vec![10.0, 20.0, 30.0, 40.0],
        };
        let matrix2 = Matrix {
            rows: 2,
            columns: 2,
            data: vec![5.0, 6.0, 7.0, 8.0],
        };

        let result = matrix1.subtract(&matrix2);

        // Check dimensions
        assert_eq!(result.rows, 2);
        assert_eq!(result.columns, 2);

        // Check data
        assert_eq!(result.data, vec![5.0, 14.0, 23.0, 32.0]);
    }

    #[test]
    fn test_multiply_matrices() {
        let matrix1 = Matrix {
            rows: 2,
            columns: 2,
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let matrix2 = Matrix {
            rows: 2,
            columns: 2,
            data: vec![5.0, 6.0, 7.0, 8.0],
        };

        let result = matrix1.multiply(&matrix2);

        // Check dimensions
        assert_eq!(result.rows, 2);
        assert_eq!(result.columns, 2);

        // Check data
        assert_eq!(result.data, vec![5.0, 12.0, 21.0, 32.0]);
    }

    #[test]
    #[should_panic(expected = "Cant multiply, matrices have different columns")]
    fn test_multiply_matrices_different_columns() {
        let matrix1 = Matrix {
            rows: 2,
            columns: 2,
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let matrix2 = Matrix {
            rows: 2,
            columns: 3,
            data: vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        };

        matrix1.multiply(&matrix2); // Should panic
    }

    #[test]
    fn test_scale() {
        let matrix = Matrix {
            rows: 2,
            columns: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };

        let scaled_matrix = matrix.scale(2.0);

        let expected = Matrix {
            rows: 2,
            columns: 3,
            data: vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        };

        assert_eq!(scaled_matrix.rows, expected.rows);
        assert_eq!(scaled_matrix.columns, expected.columns);
        assert_eq!(scaled_matrix.data, expected.data);
    }

    #[test]
    fn test_dot_product_matrices() {
        let matrix1 = Matrix {
            rows: 2,
            columns: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let matrix2 = Matrix {
            rows: 3,
            columns: 2,
            data: vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        };

        let result = matrix1.dot_product(&matrix2);

        // Check dimensions
        assert_eq!(result.rows, 2);
        assert_eq!(result.columns, 2);

        // Check data
        assert_eq!(result.data, vec![58.0, 64.0, 139.0, 154.0]); // Matrix multiplication result
    }

    #[test]
    #[should_panic(expected = "Cant do dot product, matrices have different wrong dimensions")]
    fn test_dot_product_matrices_invalid_dimensions() {
        let matrix1 = Matrix {
            rows: 2,
            columns: 2,
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let matrix2 = Matrix {
            rows: 3,
            columns: 2,
            data: vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        };

        matrix1.dot_product(&matrix2); // Should panic
    }

    #[test]
    fn test_matrix_transpose() {
        let original_matrix = Matrix {
            rows: 2,
            columns: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };

        let expected_transposed = Matrix {
            rows: 3,
            columns: 2,
            data: vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        };

        let result = original_matrix.transpose();

        // Assert rows and columns
        assert_eq!(result.rows, expected_transposed.rows);
        assert_eq!(result.columns, expected_transposed.columns);

        // Assert data
        assert_eq!(result.data, expected_transposed.data);
    }

    #[test]
    fn test_single_row_matrix_transpose() {
        let single_row = Matrix {
            rows: 1,
            columns: 3,
            data: vec![1.0, 2.0, 3.0],
        };

        let expected_transposed = Matrix {
            rows: 3,
            columns: 1,
            data: vec![1.0, 2.0, 3.0],
        };

        let result = single_row.transpose();

        // Assert rows and columns
        assert_eq!(result.rows, expected_transposed.rows);
        assert_eq!(result.columns, expected_transposed.columns);

        // Assert data
        assert_eq!(result.data, expected_transposed.data);
    }

    #[test]
    fn test_single_column_matrix_transpose() {
        let single_column = Matrix {
            rows: 3,
            columns: 1,
            data: vec![1.0, 2.0, 3.0],
        };

        let expected_transposed = Matrix {
            rows: 1,
            columns: 3,
            data: vec![1.0, 2.0, 3.0],
        };

        let result = single_column.transpose();

        // Assert rows and columns
        assert_eq!(result.rows, expected_transposed.rows);
        assert_eq!(result.columns, expected_transposed.columns);

        // Assert data
        assert_eq!(result.data, expected_transposed.data);
    }

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
}
    

