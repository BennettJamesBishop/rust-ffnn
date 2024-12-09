
use rand::Rng;
pub struct Matrix {
    pub rows: usize,
    pub columns: usize,
    pub data: Vec<f64>
}

impl Matrix {

    pub fn random(rows: usize, columns: usize) -> Matrix {
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

    pub fn add(&self, matrix2: &Matrix) -> Matrix {
        if self.rows != matrix2.rows  {
            panic!("Cant add, matrices have different number of rows");
        } else if self.columns != matrix2.columns  {
            panic!("Cant add, matrices have different columns");
        }

        let mut buffer = Vec::<f64>::with_capacity(self.rows * self.columns);
        for i in 0..self.data.len() {
                let result = self.data[i] + matrix2.data[i];
                buffer.push(result)
        }

        Matrix {
            rows: self.rows,
            columns: self.columns,
            data: buffer
        }

        }

    pub fn subtract(&self, matrix2: &Matrix) -> Matrix {
            if self.rows != matrix2.rows  {
                panic!("Cant subtract, matrices have different number of rows");
            } else if self.columns != matrix2.columns  {
                panic!("Cant subtract, matrices have different columns");
            }
    
            let mut buffer = Vec::<f64>::with_capacity(self.rows * self.columns);
            for i in 0..self.data.len() {
                    let result = self.data[i] - matrix2.data[i];
                    buffer.push(result);
            }

            Matrix {
                rows: self.rows,
                columns: self.columns,
                data: buffer
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

            // [0, 1, 3, 2, 4, 4]   [3, 2, 3, 5, 3, 1]

            // [0,1,3]               [3, 2]
            // [2,4,4]               [3, 5]
            //                       [3, 1]

            //                       [12, 8]
            //                       [30, 28]

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
    
    }

    #[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_matrix() {
        let rows = 3;
        let columns = 4;
        let matrix = Matrix::random(rows, columns);

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
    #[should_panic(expected = "Cant add, matrices have different number of rows")]
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
    #[should_panic(expected = "Cant subtract, matrices have different columns")]
    fn test_subtract_matrices_different_columns() {
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

        matrix1.subtract(&matrix2); // Should panic
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
}
