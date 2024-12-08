
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