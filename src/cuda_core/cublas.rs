

use std::ptr;

use super::{cuda::*, cublas_ffi::*};
use ::cuvector::*;
use ::cumatrix::*;
use ::meta::result::*;
#[cfg(not(feature = "disable_checks"))]
use ::meta::assert::*;



pub struct Cublas {
    handle: *mut StructCublasContext,
}
impl Drop for Cublas {
    fn drop(&mut self) {
        cublas_destroy(self.handle)
    }
}
impl Cublas {

    /// Returns a new instance of Cublas.
    pub fn new() -> CumathResult<Cublas> {
        let mut handle = ptr::null_mut();
        match cublas_create(&mut handle) {
            Some(err) => Err(CumathError::new(err)),
            None => Ok(Cublas { handle }),
        }
    }

    /// Sets the cuda stream used by this instance of Cublas.
    /// Uses cuda's default stream by default.
    pub fn set_stream(&mut self, stream: &CudaStream) {
        cublas_set_stream(self.handle, stream.stream)
    }

    /// Returns the smallest i where abs(vector[i-1]) = max(abs(value) for value in vector)
    pub fn amax_idx(&self, vector: &CuVectorDeref<f32>) -> i32 {
        let mut output = 0;
        cublas_isamax(self.handle, vector.len() as i32, vector.as_ptr(), 1, &mut output);
        output
    }

    /// Returns the smallest i where abs(vector[i-1]) = min(abs(value) for value in vector)
    pub fn amin_idx(&self, vector: &CuVectorDeref<f32>) -> i32 {
        let mut output = 0;
        cublas_isamin(self.handle, vector.len() as i32, vector.as_ptr(), 1, &mut output);
        output
    }

    /// Returns sum(abs(value) for value in vector)
    pub fn asum(&self, vector: &CuVectorDeref<f32>) -> f32 {
        let mut output = 0.0;
        cublas_sasum(self.handle, vector.len() as i32, vector.as_ptr(), 1, &mut output);
        output
    }

    /// y[i] = y[i] + alpha*x[i]
    pub fn axpy(&self, alpha: f32, x: &CuVectorDeref<f32>, y: &mut CuVectorDeref<f32>) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(x.len(), "x.len()", y.len(), "y.len()");
        }
        cublas_saxpy(self.handle, x.len() as i32, &alpha, x.as_ptr(), 1, y.as_mut_ptr(), 1)
    }

    /// y[i] = y[i] * alpha
    pub fn scal(&self, vector: &mut CuVectorDeref<f32>, alpha: f32) {
        cublas_sscal(self.handle, vector.len() as i32, &alpha, vector.as_mut_ptr(), 1);
    }

    /// output = matrix_mult(left_op, right_op)
    pub fn mult_m_m(&self, left_op: &CuMatrixDeref<f32>, right_op: &CuMatrixDeref<f32>, output: &mut CuMatrixDeref<f32>) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(left_op.cols(), "left_op.cols()", right_op.rows(), "right_op.rows()");
            assert_eq_usize(left_op.rows(), "left_op.rows()", output.rows(), "output.rows()");
            assert_eq_usize(right_op.cols(), "right_op.cols()", output.cols(), "output.cols()");
        }
        cublas_sgemm(self.handle,
                     CublasOperation::None, CublasOperation::None,
                     left_op.rows() as i32, right_op.cols() as i32, left_op.cols() as i32, &1.0,
                     left_op.as_ptr(), left_op.rows() as i32,
                     right_op.as_ptr(), right_op.rows() as i32,
                     &0.0, output.as_mut_ptr(), output.rows() as i32)
    }

    /// output = matrix_mult(left_op as RowMatrix, right_op)
    pub fn mult_row_m(&self, left_op: &CuVectorDeref<f32>, right_op: &CuMatrixDeref<f32>, output: &mut CuVectorDeref<f32>) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(left_op.len(), "left_op.len()", right_op.rows(), "right_op.rows()");
            assert_eq_usize(right_op.cols(), "right_op.cols()", output.len(), "output.len()");
        }
        cublas_sgemv(self.handle,
                     CublasOperation::Transpose,
                     right_op.rows() as i32, right_op.cols() as i32, &1.0,
                     right_op.as_ptr(), right_op.rows() as i32,
                     left_op.as_ptr(), 1,
                     &0.0, output.as_mut_ptr(), 1)
    }

    /// output = matrix_mult(left_op, right_op as ColMatrix)
    pub fn mult_m_col(&self, left_op: &CuMatrixDeref<f32>, right_op: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(left_op.cols(), "left_op.cols()", right_op.len(), "right_op.len()");
            assert_eq_usize(left_op.rows(), "left_op.rows()", output.len(), "output.len()");
        }
        cublas_sgemv(self.handle,
                     CublasOperation::None,
                     left_op.rows() as i32, left_op.cols() as i32, &1.0,
                     left_op.as_ptr(), left_op.rows() as i32,
                     right_op.as_ptr(), 1,
                     &0.0, output.as_mut_ptr(), 1)
    }

    /// output = matrix_mult(left_op as RowMatrix, right_op as ColMatrix)
    pub fn mult_col_row(&self, left_op: &CuVectorDeref<f32>, right_op: &CuVectorDeref<f32>, output: &mut CuMatrixDeref<f32>) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(left_op.len(), "left_op.len()", output.rows(), "output.rows()");
            assert_eq_usize(right_op.len(), "right_op.len()", output.cols(), "output.cols()");
        }
        cublas_sgemm(self.handle,
                     CublasOperation::None, CublasOperation::None,
                     left_op.len() as i32, right_op.len() as i32, 1, &1.0,
                     left_op.as_ptr(), left_op.len() as i32,
                     right_op.as_ptr(), 1,
                     &0.0, output.as_mut_ptr(), output.rows() as i32)
    }

    /// output = out_scl * output + in_scl * matrix_mult(left_op, right_op)
    pub fn mult_col_row_rescaled(&self, left_op: &CuVectorDeref<f32>, right_op: &CuVectorDeref<f32>, output: &mut CuMatrixDeref<f32>, in_scl: f32, out_scl: f32) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(left_op.len(), "left_op.len()", output.rows(), "output.rows()");
            assert_eq_usize(right_op.len(), "right_op.len()", output.cols(), "output.cols()");
        }
        cublas_sgemm(self.handle,
                     CublasOperation::None, CublasOperation::None,
                     left_op.len() as i32, right_op.len() as i32, 1, &in_scl,
                     left_op.as_ptr(), left_op.len() as i32,
                     right_op.as_ptr(), 1,
                     &out_scl, output.as_mut_ptr(), output.rows() as i32)
    }

}


#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn amax_idx() {
        let input_data = vec![-1.0, 2.0, 1.0, -2.0, 7.0, 5.5, -3.7, 1.1, 0.7];

        let cublas = Cublas::new().unwrap();
        let vector = CuVector::<f32>::from_host_data(input_data.as_slice());
        let idx = cublas.amax_idx(&vector);

        assert_eq!(4, idx -1);
    }

    #[test]
    fn amin_idx() {
        let input_data = vec![-1.0, 2.0, 1.0, -2.0, 7.0, 5.5, -3.7, 1.1, -0.7];

        let cublas = Cublas::new().unwrap();
        let vector = CuVector::<f32>::from_host_data(input_data.as_slice());
        let idx = cublas.amin_idx(&vector);

        assert_eq!(8, idx -1);
    }

    #[test]
    fn asum() {
        let input_data = vec![-1.0, 2.0, 1.0, -2.0, 7.0, 5.5, -3.7, 1.1, 0.7];

        let cublas = Cublas::new().unwrap();
        let vector = CuVector::<f32>::from_host_data(input_data.as_slice());
        let asum = cublas.asum(&vector);

        assert_eq!(24.0, asum);
    }

    #[test]
    fn axpy() {
        let x_data = vec![-1.0, 1.5, 0.2, -2.0];
        let y_data = vec![1.0, 0.0, -0.15, 5.0];

        let cublas = Cublas::new().unwrap();
        let x = CuVector::<f32>::from_host_data(x_data.as_slice());
        let mut y = CuVector::<f32>::from_host_data(y_data.as_slice());
        cublas.axpy(2.0, &x, &mut y);

        x.dev_assert_equals(&[-1.0, 1.5, 0.2, -2.0]);
        y.dev_assert_equals(&[-1.0, 3.0, 0.25, 1.0]);
    }

    #[test]
    fn scal() {
        let x_data = vec![-1.0, 1.5, 0.2, -2.0];

        let cublas = Cublas::new().unwrap();
        let mut x = CuVector::<f32>::from_host_data(x_data.as_slice());
        cublas.scal(&mut x, 2.0);

        x.dev_assert_equals(&[-2.0, 3.0, 0.4, -4.0]);
    }

    #[test]
    fn mult_m_m() {
        let input_data = vec![-1.0, 2.0, 1.0, -2.0, 7.0, 5.5];

        let cublas = Cublas::new().unwrap();
        let matrix1 = CuMatrix::<f32>::from_host_data(2, 3, input_data.as_slice());
        let matrix2 = CuMatrix::<f32>::from_host_data(3, 2, input_data.as_slice());
        let mut output = CuMatrix::<f32>::zero(2, 2);

        cublas.mult_m_m(&matrix1, &matrix2, &mut output);

        let mut output_buffer = vec![0.0; 4];
        output.clone_to_host(output_buffer.as_mut_slice());

        assert_equals_float(output_buffer[0], 10.0);
        assert_equals_float(output_buffer[1], -0.5);
        assert_equals_float(output_buffer[2], 47.5);
        assert_equals_float(output_buffer[3], 12.25);
    }

    #[test]
    fn mult_m_col() {
        let col_vector_data = vec![1.0, -2.0, 3.0];
        let matrix_data = vec![-1.5, 2.0, 1.5, -0.5, 1.0, 3.5];

        let cublas = Cublas::new().unwrap();
        let col_vector = CuVector::<f32>::from_host_data(col_vector_data.as_slice());
        let matrix = CuMatrix::<f32>::from_host_data(2, 3, matrix_data.as_slice());
        let mut output = CuVector::<f32>::zero(2);

        cublas.mult_m_col(&matrix, &col_vector, &mut output);

        let mut output_buffer = vec![0.0; 2];
        output.clone_to_host(output_buffer.as_mut_slice());

        assert_equals_float(output_buffer[0], -1.5);
        assert_equals_float(output_buffer[1], 13.5);
    }

    #[test]
    fn mult_col_row() {
        let col_vector_data = vec![2.2, -3.2, 1.1];
        let row_vector_data = vec![-1.0, 2.0];

        let cublas = Cublas::new().unwrap();
        let col_vector = CuVector::<f32>::from_host_data(col_vector_data.as_slice());
        let row_vector = CuVector::<f32>::from_host_data(row_vector_data.as_slice());
        let mut output = CuMatrix::<f32>::zero(3, 2);

        cublas.mult_col_row(&col_vector, &row_vector, &mut output);

        let mut output_buffer = vec![0.0; 6];
        output.clone_to_host(output_buffer.as_mut_slice());

        assert_equals_float(output_buffer[0], -2.2);
        assert_equals_float(output_buffer[1], 3.2);
        assert_equals_float(output_buffer[2], -1.1);
        assert_equals_float(output_buffer[3], 4.4);
        assert_equals_float(output_buffer[4], -6.4);
        assert_equals_float(output_buffer[5], 2.2);
    }

    #[test]
    fn mult_v_m() {
        let vector_data = vec![2.2, -3.2, 1.1];
        let matrix_data = vec![-1.0, 2.0, 1.0, -2.0, 7.0, 5.5];

        let cublas = Cublas::new().unwrap();
        let vector = CuVector::<f32>::from_host_data(vector_data.as_slice());
        let matrix = CuMatrix::<f32>::from_host_data(3, 2, matrix_data.as_slice());
        let mut output = CuVector::<f32>::zero(2);

        cublas.mult_row_m(&vector, &matrix, &mut output);

        let mut output_buffer = vec![0.0; 2];
        output.clone_to_host(output_buffer.as_mut_slice());

        assert_equals_float(output_buffer[0], -7.5);
        assert_equals_float(output_buffer[1], -20.75);
    }

    /*
    #[test]
    fn benchmark_stream() {
        use std::time::Instant;

        let input_data = vec![-1.0; 900];

        let mut cublas0 = Cublas::new();
        let mut cublas1 = Cublas::new();

        let stream0 = CudaStream::new();
        let stream1 = CudaStream::new();

        let matrix1 = CuMatrix::from_data(30, 30, input_data.as_slice());
        let matrix2 = CuMatrix::from_data(30, 30, input_data.as_slice());
        let mut output0 = CuMatrix::new(30, 30, 0.0);
        let mut output1 = CuMatrix::new(30, 30, 0.0);


        let t0 = Instant::now();
        cublas0.set_stream(&stream0);
        for _ in 0..100000 {
            cublas0.mult_m_m(&matrix1, &matrix2, &mut output0);
        }
        stream0.synchronize();
        let dt = t0.elapsed();
        println!("1 finished in {}.{}", dt.as_secs(), dt.subsec_nanos());

        let t0 = Instant::now();
        for _ in 0..50000 {
            cublas0.set_stream(&stream0);
            cublas0.mult_m_m(&matrix1, &matrix2, &mut output0);
            cublas0.set_stream(&stream1);
            cublas0.mult_m_m(&matrix1, &matrix2, &mut output1);
        }
        stream0.synchronize();
        stream1.synchronize();
        let dt = t0.elapsed();
        println!("2 finished in {}.{}", dt.as_secs(), dt.subsec_nanos());

        let t0 = Instant::now();
        cublas0.set_stream(&stream0);
        cublas1.set_stream(&stream1);
        for _ in 0..50000 {
            cublas0.mult_m_m(&matrix1, &matrix2, &mut output0);
            cublas1.mult_m_m(&matrix1, &matrix2, &mut output1);
        }
        stream0.synchronize();
        stream1.synchronize();
        let dt = t0.elapsed();
        println!("3 finished in {}.{}", dt.as_secs(), dt.subsec_nanos());
    }
    */

}