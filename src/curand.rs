
use std::ptr;

use super::*;
use ffi::curand_ffi::*;



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
        let mut handle = ptr::null_mut();
        unsafe { curandCreateGenerator(&mut handle, CurandRngType::PseudoDefault) }.assert_success();
        CurandGenerator { handle }
    }

    pub fn generate_uniform_v(&mut self, output: &mut CuVectorOpMut) {
        unsafe { curandGenerateUniform(self.handle, output.as_mut_ptr(), output.len()) }.assert_success();
    }
    pub fn generate_uniform_range_v(&mut self, output: &mut CuVectorOpMut, min: f32, max: f32) {
        assert!(min <= max);
        unsafe {
            curandGenerateUniform(self.handle, output.as_mut_ptr(), output.len()).assert_success();
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