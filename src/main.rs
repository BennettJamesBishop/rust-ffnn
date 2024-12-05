use csv::ReaderBuilder;
use ndarray::{Array1, Array2, Array, Axis, s};
use ndarray_csv::Array2Reader;
use std::error::Error;
use std::fs::File;
use rand::Rng;


fn read_csv(path_to_file: &str) -> Result<Array2<u64>, Box<dyn Error>> {
    let file = File::open(path_to_file)?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
    Ok(reader.deserialize_array2((42000, 785))?)
}


fn init_params() -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    // Create random number generator
    let mut rng = rand::thread_rng();

    // Initialize W1 (10x784 matrix) and b1 (10x1 matrix)
    let w1 = Array::from_shape_fn((10, 784), |_| rng.gen::<f64>() - 0.5);
    let b1 = Array::from_shape_fn((10, 1), |_| rng.gen::<f64>() - 0.5);

    // Initialize W2 (10x10 matrix) and b2 (10x1 matrix)
    let w2 = Array::from_shape_fn((10, 10), |_| rng.gen::<f64>() - 0.5);
    let b2 = Array::from_shape_fn((10, 1), |_| rng.gen::<f64>() - 0.5);

    // Return the initialized parameters
    (w1, b1, w2, b2)
}


fn reLU(z: Array2<f64>) -> Array2<f64> {
    z.mapv(|x| x.max(0.0)) // Apply max(0, x) element-wise
}

fn softmax(z: Array2<f64>) -> Array2<f64> {
    // Compute the exponentials of the stabilized values
    let exp_z = z.mapv(f64::exp);
    // Compute the sum of exponentials for each row
    let sum_exp_z = exp_z.sum_axis(Axis(1)).insert_axis(Axis(1));
    // Divide each exponential by the sum of exponentials for its row
    &exp_z / &sum_exp_z
}

fn forward_prop(
    w1: &Array2<f64>, 
    b1: &Array2<f64>, 
    w2: &Array2<f64>, 
    b2: &Array2<f64>, 
    X: &Array2<f64>
) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) { 
    // Ensure b1 is broadcast to the shape of w1.dot(X)
    let Z1 = w1.dot(&X) + b1.broadcast(w1.dot(&X).dim()).unwrap();
    let A1 = reLU(Z1.clone()); 
    let Z2 = w2.dot(&A1) + b2.broadcast(w2.dot(&A1).dim()).unwrap();
    let A2 = softmax(Z2.clone()); 

    (Z1, A1, Z2, A2)
}

fn reLU_deriv(z: Array2<f64>) -> (Array2<f64>) {
    z.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
}

fn one_hot(y: Array1<usize>) -> Array2<f64> {

    let num_classes = 10;

    // Create a 2D array of zeros with shape (num_samples, num_classes)
    let mut one_hot_y = Array2::<f64>::zeros((y.len(), num_classes));

    // Set the appropriate indices to 1
    for (i, &class) in y.iter().enumerate() {
        one_hot_y[[i, class]] = 1.0;
    }

    // Transpose the matrix
    one_hot_y.t().to_owned()
}


fn back_prop(
    z1: Array2<f64>,
    a1: Array2<f64>,
    a2: Array2<f64>,
    w1: Array2<f64>,
    w2: Array2<f64>,
    x: Array2<f64>,
    y: Array1<usize>,
    m: usize, // Number of samples
) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    let one_hot_y = one_hot(y);
    let dz2 = &a2 - &one_hot_y;
    let dw2 = (1.0 / m as f64) * dz2.dot(&a1.t());
    let db2 = Array2::from_elem((w2.nrows(), 1), (1.0 / m as f64) * dz2.sum_axis(Axis(0)).sum());
    let dz1 = w2.t().dot(&dz2) * reLU_deriv(z1);
    let dw1 = (1.0 / m as f64) * dz1.dot(&x.t());
    let db1 = Array2::from_elem((w1.nrows(), 1), (1.0 / m as f64) * dz1.sum_axis(Axis(0)).sum());
    (dw1, db1, dw2, db2)
}

fn update_params(w1: Array2<f64>, b1: Array2<f64>, w2: Array2<f64>, b2: Array2<f64>, dw1: Array2<f64>, db1: Array2<f64>, dw2: Array2<f64>, db2: Array2<f64>, alpha: f64) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    let w1 = w1 - alpha * dw1;
    let b1 = b1 - alpha * db1;
    let w2 = w2 - alpha * dw2;  
    let b2 = b2 - alpha * db2;    
    (w1, b1, w2, b2)
}
    
fn gradient_descent(
    x: Array2<f64>,
    y: Array1<usize>,
    alpha: f64,
    iterations: usize,
) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    // Initialize parameters
    let (mut w1, mut b1, mut w2, mut b2) = init_params();

    for i in 0..iterations {
        // Perform forward propagation
        let (z1, a1, z2, a2) = forward_prop(&w1, &b1, &w2, &b2, &x);

        // Perform backward propagation
        let (dw1, db1, dw2, db2) = back_prop(z1, a1, a2, w1.clone(), w2.clone(), x.clone(), y.clone(), x.nrows());

        // Update parameters
        let (new_w1, new_b1, new_w2, new_b2) = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha);
        w1 = new_w1;
        b1 = new_b1;
        w2 = new_w2;
        b2 = new_b2;

        // Print progress every 10 iterations
        // if i % 10 == 0 {
        //     println!("Iteration: {}", i);
        //     let predictions = get_predictions(&a2);
        //     let accuracy = get_accuracy(&predictions, &y);
        //     println!("Accuracy: {:.2}%", accuracy * 100.0);
        // }
    }

    (w1, b1, w2, b2)
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

            let  (w1, b1, w2, b2) = init_params();

           (W1, b1, W2, b2) = gradient_descent(x_train, y_train, 0.10, 500)

        }
        Err(e) => {
            // Handle errors gracefully
            eprintln!("Error reading CSV: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_relu() {
        // Input: 2D array
        let input = array![
            [1.0, -2.0, 3.0],
            [-1.0, 0.0, 2.0]
        ];

        // Expected output after applying ReLU
        let expected = array![
            [1.0, 0.0, 3.0],
            [0.0, 0.0, 2.0]
        ];

        // Compute ReLU and assert equality
        let result = reLU(input);
        assert_eq!(result, expected, "ReLU output is incorrect");
    }

    #[test]
    fn test_softmax() {
        // Input: 2D array
        let input = array![
            [1.0, 2.0, 3.0],
            [1.0, 0.0, -1.0]
        ];

        // Expected output: softmax probabilities
        let expected = array![
            [
                f64::exp(1.0) / (f64::exp(1.0) + f64::exp(2.0) + f64::exp(3.0)),
                f64::exp(2.0) / (f64::exp(1.0) + f64::exp(2.0) + f64::exp(3.0)),
                f64::exp(3.0) / (f64::exp(1.0) + f64::exp(2.0) + f64::exp(3.0)),
            ],
            [
                f64::exp(1.0) / (f64::exp(1.0) + f64::exp(0.0) + f64::exp(-1.0)),
                f64::exp(0.0) / (f64::exp(1.0) + f64::exp(0.0) + f64::exp(-1.0)),
                f64::exp(-1.0) / (f64::exp(1.0) + f64::exp(0.0) + f64::exp(-1.0)),
            ]
        ];

        // Compute softmax
        let result = softmax(input);

        // Assert equality
        result.indexed_iter().for_each(|((i, j), &value)| {
            assert!(
                (value - expected[[i, j]]).abs() < 1e-6,
                "Softmax output mismatch at ({}, {}): got {}, expected {}",
                i,
                j,
                value,
                expected[[i, j]]
            );
        });
    }

    #[test]
    fn test_forward_prop() {
        // Define mock weights, biases, and input
        let w1 = array![[0.2, 0.8], [0.5, 0.4]];
        let b1 = array![[0.1], [0.2]];
        let w2 = array![[0.6, 0.3], [0.9, 0.7]];
        let b2 = array![[0.5], [0.1]];
        let X = array![[1.0, 0.0], [0.0, 1.0]];

        // Perform forward propagation
        let (z1, a1, z2, a2) = forward_prop(w1, b1, w2, b2, X);

        // Verify shapes of intermediate outputs
        assert_eq!(z1.shape(), &[2, 2], "Shape of Z1 is incorrect");
        assert_eq!(a1.shape(), &[2, 2], "Shape of A1 is incorrect");
        assert_eq!(z2.shape(), &[2, 2], "Shape of Z2 is incorrect");
        assert_eq!(a2.shape(), &[2, 2], "Shape of A2 is incorrect");

        let expected_z1 = array![[0.3, 0.9], [0.7, 0.6]];
        z1.indexed_iter().for_each(|((i, j), &value)| {
            assert!(
                (value - expected_z1[[i, j]]).abs() < 1e-6,
                "Z1 mismatch at ({}, {}): got {}, expected {}",
                i,
                j,
                value,
                expected_z1[[i, j]]
            );
        });
    
        let expected_a1 = array![[0.3, 0.9], [0.7, 0.6]]; // ReLU does not change positive values
        a1.indexed_iter().for_each(|((i, j), &value)| {
            assert!(
                (value - expected_a1[[i, j]]).abs() < 1e-6,
                "A1 mismatch at ({}, {}): got {}, expected {}",
                i,
                j,
                value,
                expected_a1[[i, j]]
            );
        });
    
        let expected_z2 = array![[0.89, 1.22], [0.86, 1.33]];
        z2.indexed_iter().for_each(|((i, j), &value)| {
            assert!(
                (value - expected_z2[[i, j]]).abs() < 1e-6,
                "Z2 mismatch at ({}, {}): got {}, expected {}",
                i,
                j,
                value,
                expected_z2[[i, j]]
            );
        });
    
        let expected_a2 = array![
            [0.41824062, 0.58175938],
            [0.38461624, 0.61538376]
        ];
        a2.indexed_iter().for_each(|((i, j), &value)| {
            assert!(
                (value - expected_a2[[i, j]]).abs() < 1e-6,
                "A2 mismatch at ({}, {}): got {}, expected {}",
                i,
                j,
                value,
                expected_a2[[i, j]]
            );
        });
    }

    #[test]
    fn test_relu_deriv() {
        // Input: 2D array
        let input = array![
            [1.0, -2.0, 3.0],
            [-1.0, 0.0, 2.0]
        ];

        // Expected output after applying ReLU
        let expected = array![
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0]
        ];

        // Compute ReLU and assert equality
        let result = reLU_deriv(input);
        assert_eq!(result, expected, "ReLU derivative output is incorrect");
    }

    #[test]
    fn test_one_hot_single_label() {
        // Test with a single label
        let y = array![3];
        let expected = array![
            [0.0],
            [0.0],
            [0.0],
            [1.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0]
        ];
        let result = one_hot(y);
        assert_eq!(result, expected, "One-hot encoding for a single label is incorrect");
    }

    #[test]
    fn test_one_hot_multiple_labels() {
        // Test with multiple labels
        let y = array![0, 1, 9, 3, 7];
        let expected = array![
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ];
        let result = one_hot(y);
        assert_eq!(result, expected, "One-hot encoding for multiple labels is incorrect");
    }

    #[test]
    fn test_one_hot_all_classes() {
        // Test with all class labels from 0 to 9
        let y = array![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let expected = Array2::<f64>::eye(10);
        let result = one_hot(y);
        assert_eq!(result, expected, "One-hot encoding for all class labels is incorrect");
    }

    #[test]
    fn test_one_hot_empty_labels() {
        // Test with an empty label array
        let y = array![];
        let expected: Array2<f64> = Array2::zeros((10, 0));
        let result = one_hot(y);
        assert_eq!(result, expected, "One-hot encoding for empty labels is incorrect");
    }

    #[test]
     fn test_back_prop() {
            // Mock inputs
            let z1 = array![[0.5, -0.2], [0.8, 1.0]];
            let a1 = array![[0.6, 0.0], [0.7, 0.9]];
            let a2 = array![[0.8, 0.4], [0.2, 0.6]];
            let w1 = array![[0.1, 0.2], [0.3, 0.4]];
            let w2 = array![[0.5, 0.6], [0.7, 0.8]];
            let x = array![[1.0, 0.5], [0.3, 0.7]];
            let y = Array1::from(vec![0, 1]);
            let m = 2;
    
            // Expected outputs (manually calculated or mocked)
            let expected_dw1 = array![[ 0.02, 0.01 ],
            [ 0.02, 0.01 ]];
            let expected_db1 = array![[ 0.04 ],
            [ 0.0 ]];
            let expected_dw2 = array![[ 0.01,  0.18 ],
            [-0.01, -0.18 ]];
            let expected_db2 = array![[ 0.0 ],
            [ 0.0 ]];
    
      
            // Call the back_prop function
            let (dw1, db1, dw2, db2) = back_prop(z1, a1, a2, w1, w2, x, y, m);
    
            // Tolerance for comparison
            let tol = 1e-6;
    
            fn arrays_close(a: &Array2<f64>, b: &Array2<f64>, tol: f64) -> bool {
                if a.dim() != b.dim() {
                    return false; // Dimensions must match
                }
            
                a.iter()
                    .zip(b.iter())
                    .all(|(a_elem, b_elem)| (a_elem - b_elem).abs() <= tol)
            }
            
            // Assertions
            assert!(
                arrays_close(&dw1, &expected_dw1, tol),
                "dw1 mismatch: expected {:?}, got {:?}",
                expected_dw1,
                dw1
            );
            assert!(
                arrays_close(&db1, &expected_db1, tol),
                "db1 mismatch: expected {:?}, got {:?}",
                expected_db1,
                db1
            );
            assert!(
                arrays_close(&dw2, &expected_dw2, tol),
                "dw2 mismatch: expected {:?}, got {:?}",
                expected_dw2,
                dw2
            );
            assert!(
                arrays_close(&db2, &expected_db2, tol),
                "db2 mismatch: expected {:?}, got {:?}",
                expected_db2,
                db2
            );
        }
    
}