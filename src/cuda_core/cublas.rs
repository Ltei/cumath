

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


    // Level 1

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

    /// Returns the dot product of x and y
    pub fn dot(&self, x: &CuVectorDeref<f32>, y: &CuVectorDeref<f32>) -> f32 {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(x.len(), "x.len()", y.len(), "y.len()");
        }
        let mut result = 0.0;
        cublas_sdot(self.handle, x.len() as i32, x.as_ptr(), 1, y.as_ptr(), 1, &mut result);
        result
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

    /// Swap vector1 and vector2 elements in memory
    pub fn swap(&self, vector1: &mut CuVectorDeref<f32>, vector2: &mut CuVectorDeref<f32>) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(vector1.len(), "x.len()", vector2.len(), "y.len()");
        }
        cublas_sswap(self.handle, vector1.len() as i32, vector1.as_mut_ptr(), 1, vector2.as_mut_ptr(), 1);
    }


    // Level 2

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


    // Level 3

    /// output = matrix_mult( transa(a), transb(b) )
    /// Isn't public because checking dimensions would be too costy, so use mult_m_m, mult_mt_m.. instead
    fn gemm(&self, alpha: f32, beta: f32, a: &CuMatrixDeref<f32>, transa: CublasOperation, b: &CuMatrixDeref<f32>, transb: CublasOperation, output: &mut CuMatrixDeref<f32>) {
        cublas_sgemm(self.handle,
                     transa, transb,
                     a.rows() as i32, b.cols() as i32, a.cols() as i32, &alpha,
                     a.as_ptr(), a.rows() as i32,
                     b.as_ptr(), b.rows() as i32,
                     &beta, output.as_mut_ptr(), output.rows() as i32)
    }
    //pub fn mult_mt_m(&self, a: &CuVectorDeref<f32>, b: &CuVectorDeref<f32>, output: &mut CuMatrixDeref<f32>) TODO
    //pub fn mult_m_mt(&self, a: &CuVectorDeref<f32>, b: &CuVectorDeref<f32>, output: &mut CuMatrixDeref<f32>) TODO
    //pub fn mult_mt_mt(&self, a: &CuVectorDeref<f32>, b: &CuVectorDeref<f32>, output: &mut CuMatrixDeref<f32>) TODO

    /// output = matrix_mult(left_op, right_op)
    pub fn mult_m_m(&self, left_op: &CuMatrixDeref<f32>, right_op: &CuMatrixDeref<f32>, output: &mut CuMatrixDeref<f32>) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(left_op.cols(), "left_op.cols()", right_op.rows(), "right_op.rows()");
            assert_eq_usize(left_op.rows(), "left_op.rows()", output.rows(), "output.rows()");
            assert_eq_usize(right_op.cols(), "right_op.cols()", output.cols(), "output.cols()");
        }
        self.gemm(1.0, 0.0, left_op, CublasOperation::None, right_op, CublasOperation::None, output)
        /*cublas_sgemm(self.handle,
                     CublasOperation::None, CublasOperation::None,
                     left_op.rows() as i32, right_op.cols() as i32, left_op.cols() as i32, &1.0,
                     left_op.as_ptr(), left_op.rows() as i32,
                     right_op.as_ptr(), right_op.rows() as i32,
                     &0.0, output.as_mut_ptr(), output.rows() as i32)*/
    }

}