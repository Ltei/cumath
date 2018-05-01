
use std::ptr;

use super::{cuda::*, curand_ffi::*};
use ::cudata::*;
use ::meta::result::*;

pub use super::curand_ffi::CurandRngType;



pub struct CurandGenerator {
    handle: *mut StructCurandGenerator,
}

impl Drop for CurandGenerator {
    fn drop(&mut self) {
        curand_destroy_generator(self.handle);
    }
}

impl CurandGenerator {
    pub fn new(rng_type: CurandRngType) -> CumathResult<CurandGenerator> {
        let mut handle = ptr::null_mut();
        match curand_create_generator(&mut handle, rng_type) {
            Some(err) => Err(CumathError::new(err)),
            None => Ok(CurandGenerator { handle }),
        }
    }

    pub fn set_stream(&mut self, stream: &CudaStream) {
        curand_set_stream(self.handle, stream.stream);
    }

    pub fn generate_uniform(&mut self, output: &mut CuPackedDataMut) {
        curand_generate_uniform(self.handle, output.as_mut_ptr(), output.len());
    }

    pub fn generate_uniform_range(&mut self, output: &mut CuPackedDataMut, min: f32, max: f32, stream: &CudaStream) {
        use cuvector::ffi::VectorKernel_aYpb;
        #[cfg(not(feature = "disable_checks"))] {
            assert!(min <= max, "min > max");
        }
            curand_generate_uniform(self.handle, output.as_mut_ptr(), output.len());
        unsafe {
            VectorKernel_aYpb(max-min, min, output.as_mut_ptr(), output.len() as i32, stream.stream);
        }
    }

    pub fn generate_normal(&mut self, output: &mut CuPackedDataMut, mean: f32, stddev: f32) {
        #[cfg(not(feature = "disable_checks"))] {
            assert!(stddev >= 0.0, "stddev < 0.0");
        }
        curand_generate_normal(self.handle, output.as_mut_ptr(), output.len(), mean, stddev);
    }

    pub fn generate_lognormal(&mut self, output: &mut CuPackedDataMut, mean: f32, stddev: f32) {
        #[cfg(not(feature = "disable_checks"))] {
            assert!(stddev >= 0.0, "stddev < 0.0");
        }
        curand_generate_lognormal(self.handle, output.as_mut_ptr(), output.len(), mean, stddev);
    }

    pub fn generate_poisson(&mut self, output: &mut CuPackedDataMut, lambda: f32) {
        #[cfg(not(feature = "disable_checks"))] {
            assert!(lambda >= 0.0, "lambda < 0.0");
        }
        curand_generate_poisson(self.handle, output.as_mut_ptr(), output.len(), lambda);
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use cuvector::{CuVector, CuVectorOp};

    #[test]
    fn curand_generate_uniform() {
        let mut generator = CurandGenerator::new(CurandRngType::PseudoDefault).unwrap();

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

        let mut generator = CurandGenerator::new(CurandRngType::PseudoDefault).unwrap();

        let mut vector = CuVector::new(10, 0.0);
        generator.generate_uniform_range(&mut vector, min, max, &super::DEFAULT_STREAM);

        let mut buffer = vec![0.0; 10];
        vector.clone_to_host(&mut buffer);

        buffer.iter().for_each(|x| {
            assert!(*x >= min && *x <= max);
        });
    }

}