
extern crate libc;


mod ffi;
mod assert;

use ffi::cuda_ffi::*;
use ffi::cublas_ffi::*;
use ffi::curand_ffi::*;

use assert::*;

mod tags;
use tags::*;

mod vector;
pub use vector::*;
mod matrix;
pub use matrix::*;
mod cublas;
pub use cublas::*;

pub struct Cuda { }
impl Cuda {
    pub fn synchronize() {
        cuda_synchronize();
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

    pub fn generate_uniform_v(&mut self, output: &mut CuVectorOpMut) {
        unsafe { curandGenerateUniform(self.handle, output.ptr_mut(), output.len()) }.assert_success();
    }
    pub fn generate_uniform_range_v(&mut self, output: &mut CuVectorOpMut, min: f32, max: f32) {
        assert!(min <= max);
        unsafe {
            curandGenerateUniform(self.handle, output.ptr_mut(), output.len()).assert_success();
            CuVector::aypb(max-min, min, output);
        }
    }

    pub fn generate_uniform_m<CuMatrixOpMutPackedT: CuMatrixOpMut + CuPacked>(&mut self, output: &mut CuMatrixOpMutPackedT) {
        unsafe { curandGenerateUniform(self.handle, output.ptr_mut(), output.len()) }.assert_success();
    }
    pub fn generate_uniform_range_m<CuMatrixOpMutPackedT: CuMatrixOpMut + CuPacked>(&mut self, output: &mut CuMatrixOpMutPackedT, min: f32, max: f32) {
        assert!(min <= max);
        unsafe {
            curandGenerateUniform(self.handle, output.ptr_mut(), output.len()).assert_success();
            CuMatrix::aypb(max-min, min, output);
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    /*fn assert_equals_float(a: f32, b: f32) {
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

        assert_eq!(24.0, cublas.asum_v(&vector));
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
    }*/

    #[test]
    fn curand_generate_uniform_v() {
        let mut generator = CurandGenerator::new();

        let mut vector = CuVector::new(10, 0.0);
        generator.generate_uniform_v(&mut vector);

        let mut buffer = vec![0.0; 10];
        vector.clone_to_host(&mut buffer);

        buffer.iter().for_each(|x| {
            assert!(x >= &0.0 && x <= &1.0);
        });
    }

    #[test]
    fn curand_generate_uniform_range_v() {
        let min = -5.0;
        let max = 15.0;

        let mut generator = CurandGenerator::new();

        let mut vector = CuVector::new(10, 0.0);
        generator.generate_uniform_range_v(&mut vector, min, max);

        let mut buffer = vec![0.0; 10];
        vector.clone_to_host(&mut buffer);

        buffer.iter().for_each(|x| {
            assert!(*x >= min && *x <= max);
        });
    }
}