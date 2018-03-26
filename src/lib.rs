
extern crate libc;


mod ffi;
mod assert;

use ffi::cuda_ffi::*;
use ffi::cublas_ffi::*;
use ffi::curand_ffi::*;

use assert::*;

pub mod cuvector;
pub mod cumatrix;
pub use cuvector::*;
pub use cumatrix::*;


pub struct Cuda { }
impl Cuda {
    pub fn synchronize() {
        unsafe { cudaDeviceSynchronize() }.assert_success()
    }
}

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
        let mut handle = std::ptr::null_mut();
        unsafe { cublasCreate_v2(&mut handle) }.assert_success();
        Cublas { handle }
    }

    pub fn mult_m_m(&self, left_op: &CuMatrix, right_op: &CuMatrix, output: &mut CuMatrix) {
        assert_eq_usize(left_op.cols, "left_op.cols()", right_op.rows, "right_op.rows()");
        assert_eq_usize(left_op.rows, "left_op.rows()", output.rows, "output.rows()");
        assert_eq_usize(right_op.cols, "right_op.cols()", output.cols, "output.cols()");
        unsafe {
            cublasSgemm_v2(self.handle,
                           CublasOperation::None, CublasOperation::None,
                           left_op.rows as i32, right_op.cols as i32, left_op.cols as i32, &1.0,
                           left_op.data, left_op.rows as i32,
                           right_op.data, right_op.rows as i32,
                           &0.0, output.data, output.rows as i32)
        }.assert_success();
    }
    pub fn mult_row_m(&self, left_op: &CuVector, right_op: &CuMatrix, output: &mut CuVector) {
        assert_eq_usize(left_op.len, "left_op.len()", right_op.rows, "right_op.rows()");
        assert_eq_usize(right_op.cols, "left_op.cols()", output.len, "output.len()");
        unsafe {
            cublasSgemv_v2(self.handle,
                           CublasOperation::Transpose,
                           (right_op.rows as i32), (right_op.cols as i32), &1.0,
                           right_op.data, (right_op.rows as i32),
                           left_op.data, 1,
                           &0.0, output.data, 1)
        }.assert_success();
    }
    pub fn mult_m_col(&self, left_op: &CuMatrix, right_op: &CuVector, output: &mut CuVector) {
        assert_eq_usize(left_op.cols, "left_op.cols()", right_op.len, "right_op.len()");
        assert_eq_usize(left_op.rows, "left_op.rows()", output.len, "output.len()");
        unsafe {
            cublasSgemv_v2(self.handle,
                           CublasOperation::None,
                           (left_op.rows as i32), (left_op.cols as i32), &1.0,
                           left_op.data, (left_op.rows as i32),
                           right_op.data, 1,
                           &0.0, output.data, 1)
        }.assert_success();
    }
    pub fn mult_col_row(&self, left_op: &CuVector, right_op: &CuVector, output: &mut CuMatrix) {
        assert_eq_usize(left_op.len, "left_op.len()", output.rows, "output.rows()");
        assert_eq_usize(right_op.len, "right_op.len()", output.cols, "output.cols()");
        unsafe {
            cublasSgemm_v2(self.handle,
                           CublasOperation::None, CublasOperation::None,
                           (left_op.len as i32), (right_op.len as i32), 1, &1.0,
                           left_op.data, (left_op.len as i32),
                           right_op.data, 1,
                           &0.0, output.data, (output.rows as i32))
        }.assert_success();
    }

    /** output = out_scl * output + in_scl * left_op * right_op */
    pub fn mult_col_row_(&self, left_op: &CuVector, right_op: &CuVector, output: &mut CuMatrix, in_scl: f32, out_scl: f32) {
        assert_eq_usize(left_op.len, "left_op.len()", output.rows, "output.rows()");
        assert_eq_usize(right_op.len, "right_op.len()", output.cols, "output.cols()");
        unsafe {
            cublasSgemm_v2(self.handle,
                           CublasOperation::None, CublasOperation::None,
                           (left_op.len as i32), (right_op.len as i32), 1, &in_scl,
                           left_op.data, (left_op.len as i32),
                           right_op.data, 1,
                           &out_scl, output.data, (output.rows as i32))
        }.assert_success();
    }

    pub fn abs_sum_m(&self, matrix: &CuMatrix) -> f32 {
        let mut output = 0.0;
        unsafe { cublasSasum_v2(self.handle, matrix.len as i32, matrix.data, 1, &mut output) }.assert_success();
        output
    }
    pub fn abs_sum_v(&self, vector: &CuVector) -> f32 {
        let mut output = 0.0;
        unsafe { cublasSasum_v2(self.handle, vector.len as i32, vector.data, 1, &mut output) }.assert_success();
        output
    }
}

pub struct CurandGenerator {
    handle: *mut StructCurandGenerator,
}
impl Drop for CurandGenerator {
    fn drop(&mut self) {
        unsafe { curandDestroyGenerator(self.handle) }.assert_success();
    }
}
impl CurandGenerator {
    pub fn new() -> CurandGenerator {
        let mut handle = std::ptr::null_mut();
        unsafe { curandCreateGenerator(&mut handle, CurandRngType::PseudoDefault) }.assert_success();
        CurandGenerator { handle }
    }

    pub fn generate_uniform_v(&mut self, output: &mut CuVector) {
        unsafe { curandGenerateUniform(self.handle, output.data, output.len) }.assert_success();
    }
    pub fn generate_uniform_m(&mut self, output: &mut CuMatrix) {
        unsafe { curandGenerateUniform(self.handle, output.data, output.len) }.assert_success();
    }
    pub fn generate_uniform_range_m(&mut self, output: &mut CuMatrix, min: f32, max: f32) {
        assert!(min < max);
        unsafe {
            curandGenerateUniform(self.handle, output.data, output.len).assert_success();
            output.add_scl_self(-0.5);
            output.scale_self(max-min);
        }
    }
}


#[cfg(test)]
mod tests {
    fn assert_equals_float(a: f32, b: f32) {
        let d = a-b;
        if d < -0.000001 || d > 0.000001 {
            panic!("{} != {}", a, b);
        }
    }

    #[test]
    fn cublas_abs_sum() {
        let input_data = vec![-1.0, 2.0, 1.0, -2.0, 7.0, 5.5, -3.7, 1.1, 0.7];

        let cublas = super::Cublas::new();
        let matrix = super::CuMatrix::from_data(3, 3, input_data.as_slice());

        assert_eq!(24.0, cublas.abs_sum_m(&matrix));
    }
    #[test]
    fn cublas_mult_m_m() {
        let input_data = vec![-1.0, 2.0, 1.0, -2.0, 7.0, 5.5];

        let cublas = super::Cublas::new();
        let matrix1 = super::CuMatrix::from_data(2, 3, input_data.as_slice());
        let matrix2 = super::CuMatrix::from_data(3, 2, input_data.as_slice());
        let mut output = super::CuMatrix::new(2, 2, 0.0);

        cublas.mult_m_m(&matrix1, &matrix2, &mut output);

        let mut output_buffer = vec![0.0; 9];
        output.copy_to_host(output_buffer.as_mut_slice());

        assert_equals_float(output_buffer[0], 10.0);
        assert_equals_float(output_buffer[1], -0.5);
        assert_equals_float(output_buffer[2], 47.5);
        assert_equals_float(output_buffer[3], 12.25);
    }
    #[test]
    fn cublas_mult_m_col() {
        let col_vector_data = vec![1.0, -2.0, 3.0];
        let matrix_data = vec![-1.5, 2.0, 1.5, -0.5, 1.0, 3.5];

        let cublas = super::Cublas::new();
        let col_vector = super::CuVector::from_data(col_vector_data.as_slice());
        let matrix = super::CuMatrix::from_data(2, 3, matrix_data.as_slice());
        let mut output = super::CuVector::new(2, 0.0);

        cublas.mult_m_col(&matrix, &col_vector, &mut output);

        let mut output_buffer = vec![0.0; 2];
        output.copy_to_host(output_buffer.as_mut_slice());

        assert_equals_float(output_buffer[0], -1.5);
        assert_equals_float(output_buffer[1], 13.5);
    }
    #[test]
    fn cublas_mult_v_m() {
        let vector_data = vec![2.2, -3.2, 1.1];
        let matrix_data = vec![-1.0, 2.0, 1.0, -2.0, 7.0, 5.5];

        let cublas = super::Cublas::new();
        let vector = super::CuVector::from_data(vector_data.as_slice());
        let matrix = super::CuMatrix::from_data(3, 2, matrix_data.as_slice());
        let mut output = super::CuVector::new(2, 0.0);

        cublas.mult_row_m(&vector, &matrix, &mut output);

        let mut output_buffer = vec![0.0; 2];
        output.copy_to_host(output_buffer.as_mut_slice());

        assert_equals_float(output_buffer[0], -7.5);
        assert_equals_float(output_buffer[1], -20.75);
    }
    #[test]
    fn cublas_mult_col_row() {
        let col_vector_data = vec![2.2, -3.2, 1.1];
        let row_vector_data = vec![-1.0, 2.0];

        let cublas = super::Cublas::new();
        let col_vector = super::CuVector::from_data(col_vector_data.as_slice());
        let row_vector = super::CuVector::from_data(row_vector_data.as_slice());
        let mut output = super::CuMatrix::new(3, 2, 0.0);

        cublas.mult_col_row(&col_vector, &row_vector, &mut output);

        let mut output_buffer = vec![0.0; 6];
        output.copy_to_host(output_buffer.as_mut_slice());

        assert_equals_float(output_buffer[0], -2.2);
        assert_equals_float(output_buffer[1], 3.2);
        assert_equals_float(output_buffer[2], -1.1);
        assert_equals_float(output_buffer[3], 4.4);
        assert_equals_float(output_buffer[4], -6.4);
        assert_equals_float(output_buffer[5], 2.2);
    }

    #[test]
    fn curand_generate_uniform() {
        let mut generator = super::CurandGenerator::new();

        let mut vector = super::CuVector::new(10, 0.0);
        generator.generate_uniform_v(&mut vector);

        let mut buffer = vec![0.0; 10];
        vector.copy_to_host(&mut buffer);

        buffer.iter().for_each(|x| {
            assert!(x >= &0.0 && x <= &1.0);
        });
    }
}

