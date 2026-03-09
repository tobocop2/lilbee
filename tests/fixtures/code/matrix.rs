/// Transpose a 2-D matrix, swapping rows and columns.
pub fn transpose(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let rows = matrix.len();
    if rows == 0 {
        return vec![];
    }
    let cols = matrix[0].len();
    let mut result = vec![vec![0.0; rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            result[j][i] = matrix[i][j];
        }
    }
    result
}

/// Compute the determinant of a square matrix via cofactor expansion.
pub fn determinant(matrix: &Vec<Vec<f64>>) -> f64 {
    let n = matrix.len();
    if n == 1 {
        return matrix[0][0];
    }
    if n == 2 {
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    }
    let mut det = 0.0;
    for col in 0..n {
        let minor: Vec<Vec<f64>> = (1..n)
            .map(|r| {
                matrix[r]
                    .iter()
                    .enumerate()
                    .filter(|&(c, _)| c != col)
                    .map(|(_, &v)| v)
                    .collect()
            })
            .collect();
        let sign = if col % 2 == 0 { 1.0 } else { -1.0 };
        det += sign * matrix[0][col] * determinant(&minor);
    }
    det
}
