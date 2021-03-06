
use std::ptr;

use super::{cuda::*, curand_ffi::*};
use cuvector::CuVectorDeref;
use meta::result::*;

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

    pub fn generate_uniform(&mut self, output: &mut CuVectorDeref<f32>) {
        curand_generate_uniform(self.handle, output.as_mut_ptr(), output.len());
    }

    pub fn generate_uniform_range(&mut self, output: &mut CuVectorDeref<f32>, min: f32, max: f32, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert!(min <= max, "min > max");
        }
            curand_generate_uniform(self.handle, output.as_mut_ptr(), output.len());
        unsafe {
            ::kernel::VectorPacked_aypb_f32(max-min, output.as_mut_ptr(), min, output.len() as i32, stream.stream);
        }
    }

    pub fn generate_normal(&mut self, output: &mut CuVectorDeref<f32>, mean: f32, stddev: f32) {
        #[cfg(not(feature = "disable_checks"))] {
            assert!(stddev >= 0.0, "stddev < 0.0");
        }
        curand_generate_normal(self.handle, output.as_mut_ptr(), output.len(), mean, stddev);
    }

    pub fn generate_lognormal(&mut self, output: &mut CuVectorDeref<f32>, mean: f32, stddev: f32) {
        #[cfg(not(feature = "disable_checks"))] {
            assert!(stddev >= 0.0, "stddev < 0.0");
        }
        curand_generate_lognormal(self.handle, output.as_mut_ptr(), output.len(), mean, stddev);
    }

    pub fn generate_poisson(&mut self, output: &mut CuVectorDeref<f32>, lambda: f32) {
        #[cfg(not(feature = "disable_checks"))] {
            assert!(lambda >= 0.0, "lambda < 0.0");
        }
        curand_generate_poisson(self.handle, output.as_mut_ptr(), output.len(), lambda);
    }
}