
use std::ptr;

use ffi::curand_ffi::*;
use ::cudata::*;



pub struct CurandGenerator {
    handle: *mut StructCurandGenerator,
}

impl Drop for CurandGenerator {
    fn drop(&mut self) {
        curand_destroy_generator(self.handle);
    }
}

impl CurandGenerator {
    pub fn new() -> CurandGenerator {
        let mut handle = ptr::null_mut();
        curand_create_generator(&mut handle, CurandRngType::PseudoDefault);
        CurandGenerator { handle }
    }

    pub fn generate_uniform(&mut self, output: &mut CuPackedDataMut) {
        curand_generate_uniform(self.handle, output.as_mut_ptr(), output.len());
    }

    pub fn generate_uniform_range(&mut self, output: &mut CuPackedDataMut, min: f32, max: f32) {
        use ffi::vectorkernel_ffi::VectorKernel_aYpb;
        assert!(min <= max);
        unsafe {
            curand_generate_uniform(self.handle, output.as_mut_ptr(), output.len());
            VectorKernel_aYpb(max-min, min, output.as_mut_ptr(), output.len() as i32);
        }
    }
}


#[cfg(test)]
mod tests {
    use super::CurandGenerator;
    use cuvector::{CuVector, CuVectorOp};

    #[test]
    fn curand_generate_uniform() {
        let mut generator = CurandGenerator::new();

        let mut vector = CuVector::new(10, 0.0);
        generator.generate_uniform(&mut vector);

        let mut buffer = vec![0.0; 10];
        vector.clone_to_host(&mut buffer);

        buffer.iter().for_each(|x| {
            assert!(x >= &0.0 && x <= &1.0);
        });
    }

    #[test]
    fn curand_generate_uniform_range() {
        let min = -5.0;
        let max = 15.0;

        let mut generator = CurandGenerator::new();

        let mut vector = CuVector::new(10, 0.0);
        generator.generate_uniform_range(&mut vector, min, max);

        let mut buffer = vec![0.0; 10];
        vector.clone_to_host(&mut buffer);

        buffer.iter().for_each(|x| {
            assert!(*x >= min && *x <= max);
        });
    }
}