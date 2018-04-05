

use std::ptr;
use super::*;



pub struct Cublas {
    handle: *mut StructCublasContext,
}
impl Drop for Cublas {
    fn drop(&mut self) {
        unsafe { cublasDestroy_v2(self.handle) }.assert_success();
    }
}
impl Cublas {
    pub fn new() -> Cublas {
        let mut handle = ptr::null_mut();
        unsafe { cublasCreate_v2(&mut handle) }.assert_success();
        Cublas { handle }
    }

    pub fn mult_m_m(&self, left_op: &CuMatrixOp, right_op: &CuMatrixOp, output: &mut CuMatrixOpMut) {
        assert_eq_usize(left_op.cols(), "left_op.cols()", right_op.rows(), "right_op.rows()");
        assert_eq_usize(left_op.rows(), "left_op.rows()", output.rows(), "output.rows()");
        assert_eq_usize(right_op.cols(), "right_op.cols()", output.cols(), "output.cols()");
        unsafe {
            cublasSgemm_v2(self.handle,
                           CublasOperation::None, CublasOperation::None,
                           left_op.rows() as i32, right_op.cols() as i32, left_op.cols() as i32, &1.0,
                           left_op.ptr(), left_op.rows() as i32,
                           right_op.ptr(), right_op.rows() as i32,
                           &0.0, output.ptr_mut(), output.rows() as i32)
        }.assert_success();
    }
    pub fn mult_row_m(&self, left_op: &CuVectorOp, right_op: &CuMatrixOp, output: &mut CuVectorOpMut) {
        assert_eq_usize(left_op.len(), "left_op.len()", right_op.rows(), "right_op.rows()");
        assert_eq_usize(right_op.cols(), "right_op.cols()", output.len(), "output.len()");
        unsafe {
            cublasSgemv_v2(self.handle,
                           CublasOperation::Transpose,
                           right_op.rows() as i32, right_op.cols() as i32, &1.0,
                           right_op.ptr(), right_op.rows() as i32,
                           left_op.ptr(), 1,
                           &0.0, output.ptr_mut(), 1)
        }.assert_success();
    }
    pub fn mult_m_col(&self, left_op: &CuMatrixOp, right_op: &CuVectorOp, output: &mut CuVectorOpMut) {
        assert_eq_usize(left_op.cols(), "left_op.cols()", right_op.len(), "right_op.len()");
        assert_eq_usize(left_op.rows(), "left_op.rows()", output.len(), "output.len()");
        unsafe {
            cublasSgemv_v2(self.handle,
                           CublasOperation::None,
                           left_op.rows() as i32, left_op.cols() as i32, &1.0,
                           left_op.ptr(), left_op.rows() as i32,
                           right_op.ptr(), 1,
                           &0.0, output.ptr_mut(), 1)
        }.assert_success();
    }
    pub fn mult_col_row(&self, left_op: &CuVectorOp, right_op: &CuVectorOp, output: &mut CuMatrixOpMut) {
        assert_eq_usize(left_op.len(), "left_op.len()", output.rows(), "output.rows()");
        assert_eq_usize(right_op.len(), "right_op.len()", output.cols(), "output.cols()");
        unsafe {
            cublasSgemm_v2(self.handle,
                           CublasOperation::None, CublasOperation::None,
                           left_op.len() as i32, right_op.len() as i32, 1, &1.0,
                           left_op.ptr(), left_op.len() as i32,
                           right_op.ptr(), 1,
                           &0.0, output.ptr_mut(), output.rows() as i32)
        }.assert_success();
    }

    /** output = out_scl * output + in_scl * left_op * right_op */
    pub fn mult_col_row_(&self, left_op: &CuVectorOp, right_op: &CuVectorOp, output: &mut CuMatrixOpMut, in_scl: f32, out_scl: f32) {
        assert_eq_usize(left_op.len(), "left_op.len()", output.rows(), "output.rows()");
        assert_eq_usize(right_op.len(), "right_op.len()", output.cols(), "output.cols()");
        unsafe {
            cublasSgemm_v2(self.handle,
                           CublasOperation::None, CublasOperation::None,
                           left_op.len() as i32, right_op.len() as i32, 1, &in_scl,
                           left_op.ptr(), left_op.len() as i32,
                           right_op.ptr(), 1,
                           &out_scl, output.ptr_mut(), output.rows() as i32)
        }.assert_success();
    }

    /*pub fn asum_m(&self, matrix: &CuMatrix) -> f32 {
        let mut output = 0.0;
        unsafe { cublasSasum_v2(self.handle, matrix.len as i32, matrix.data, 1, &mut output) }.assert_success();
        output
    }*/
    pub fn asum_v(&self, vector: &CuVectorOp) -> f32 {
        let mut output = 0.0;
        unsafe { cublasSasum_v2(self.handle, vector.len() as i32, vector.ptr(), 1, &mut output) }.assert_success();
        output
    }

    pub fn axpy_m(&self, alpha: f32, x: &CuMatrixOp, y: &mut CuMatrixOpMut) {
        unsafe {
            cublasSaxpy_v2(self.handle, x.len() as i32, &alpha, x.ptr(), 1, y.ptr_mut(), 1)
        }.assert_success()
    }
    pub fn axpy_v(&self, alpha: f32, x: &CuVectorOp, y: &mut CuVectorOpMut) {
        unsafe {
            cublasSaxpy_v2(self.handle, x.len() as i32, &alpha, x.ptr(), 1, y.ptr_mut(), 1)
        }.assert_success()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn assert_equals_float(a: f32, b: f32) {
        let d = a-b;
        if d < -0.000001 || d > 0.000001 {
            panic!("{} != {}", a, b);
        }
    }

    #[test]
    fn cublas_abs_sum() {
        let input_data = vec![-1.0, 2.0, 1.0, -2.0, 7.0, 5.5, -3.7, 1.1, 0.7];

        let cublas = Cublas::new();
        let vector = CuVector::from_data(input_data.as_slice());
        let asum = cublas.asum_v(&vector);

        assert_eq!(24.0, asum);
    }

    #[test]
    fn cublas_mult_m_m() {
        let input_data = vec![-1.0, 2.0, 1.0, -2.0, 7.0, 5.5];

        let cublas = Cublas::new();
        let matrix1 = CuMatrix::from_data(2, 3, input_data.as_slice());
        let matrix2 = CuMatrix::from_data(3, 2, input_data.as_slice());
        let mut output = CuMatrix::new(2, 2, 0.0);

        cublas.mult_m_m(&matrix1, &matrix2, &mut output);

        let mut output_buffer = vec![0.0; 4];
        output.clone_to_host(output_buffer.as_mut_slice());

        assert_equals_float(output_buffer[0], 10.0);
        assert_equals_float(output_buffer[1], -0.5);
        assert_equals_float(output_buffer[2], 47.5);
        assert_equals_float(output_buffer[3], 12.25);
    }

    #[test]
    fn cublas_mult_m_col() {
        let col_vector_data = vec![1.0, -2.0, 3.0];
        let matrix_data = vec![-1.5, 2.0, 1.5, -0.5, 1.0, 3.5];

        let cublas = Cublas::new();
        let col_vector = CuVector::from_data(col_vector_data.as_slice());
        let matrix = CuMatrix::from_data(2, 3, matrix_data.as_slice());
        let mut output = CuVector::new(2, 0.0);

        cublas.mult_m_col(&matrix, &col_vector, &mut output);

        let mut output_buffer = vec![0.0; 2];
        output.clone_to_host(output_buffer.as_mut_slice());

        assert_equals_float(output_buffer[0], -1.5);
        assert_equals_float(output_buffer[1], 13.5);
    }

    #[test]
    fn cublas_mult_v_m() {
        let vector_data = vec![2.2, -3.2, 1.1];
        let matrix_data = vec![-1.0, 2.0, 1.0, -2.0, 7.0, 5.5];

        let cublas = Cublas::new();
        let vector = CuVector::from_data(vector_data.as_slice());
        let matrix = CuMatrix::from_data(3, 2, matrix_data.as_slice());
        let mut output = CuVector::new(2, 0.0);

        cublas.mult_row_m(&vector, &matrix, &mut output);

        let mut output_buffer = vec![0.0; 2];
        output.clone_to_host(output_buffer.as_mut_slice());

        assert_equals_float(output_buffer[0], -7.5);
        assert_equals_float(output_buffer[1], -20.75);
    }

    #[test]
    fn cublas_mult_col_row() {
        let col_vector_data = vec![2.2, -3.2, 1.1];
        let row_vector_data = vec![-1.0, 2.0];

        let cublas = Cublas::new();
        let col_vector = CuVector::from_data(col_vector_data.as_slice());
        let row_vector = CuVector::from_data(row_vector_data.as_slice());
        let mut output = CuMatrix::new(3, 2, 0.0);

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
}