
use std::ptr;

use ffi::curand_ffi::*;
use ::cudata::*;
use cuda::*;

pub use ffi::curand_ffi::CurandRngType;



pub struct CurandGenerator {
    handle: *mut StructCurandGenerator,
}

impl Drop for CurandGenerator {
    fn drop(&mut self) {
        curand_destroy_generator(self.handle);
    }
}

impl CurandGenerator {
    pub fn new(rng_type: CurandRngType) -> CurandGenerator {
        let mut handle = ptr::null_mut();
        curand_create_generator(&mut handle, rng_type);
        CurandGenerator { handle }
    }

    pub fn set_stream(&mut self, stream: &CudaStream) {
        curand_set_stream(self.handle, stream.stream);
    }

    pub fn generate_uniform(&mut self, output: &mut CuPackedDataMut) {
        curand_generate_uniform(self.handle, output.as_mut_ptr(), output.len());
    }

    pub fn generate_uniform_range(&mut self, output: &mut CuPackedDataMut, min: f32, max: f32, stream: &CudaStream) {
        use ffi::vectorkernel_ffi::VectorKernel_aYpb;
        assert!(min <= max);
        unsafe {
            curand_generate_uniform(self.handle, output.as_mut_ptr(), output.len());
            VectorKernel_aYpb(max-min, min, output.as_mut_ptr(), output.len() as i32, stream.stream);
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use cuvector::{CuVector, CuVectorOp};

    #[test]
    fn curand_generate_uniform() {
        let mut generator = CurandGenerator::new(CurandRngType::PseudoDefault);

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

        let mut generator = CurandGenerator::new(CurandRngType::PseudoDefault);

        let mut vector = CuVector::new(10, 0.0);
        generator.generate_uniform_range(&mut vector, min, max, &super::DEFAULT_STREAM);

        let mut buffer = vec![0.0; 10];
        vector.clone_to_host(&mut buffer);

        buffer.iter().for_each(|x| {
            assert!(*x >= min && *x <= max);
        });
    }


    /*
    #[test]
    fn benchmark_stream() {
        use std::time::Instant;

        let mut generator1 = CurandGenerator::new(CurandRngType::PseudoDefault);
        let mut generator2 = CurandGenerator::new(CurandRngType::PseudoDefault);

        let mut vector1 = CuVector::new(10000, 1.0);
        let mut vector2 = CuVector::new(10000, 1.0);

        let stream1 = CudaStream::new();
        let stream2 = CudaStream::new();

        let t0 = Instant::now();
        generator1.set_stream(&stream1);
        for _ in 0..1000000 {
            generator1.generate_uniform(&mut vector1);
        }
        stream1.synchronize();
        let dt = t0.elapsed();
        println!("1 finished in {}.{}", dt.as_secs(), dt.subsec_nanos());

        /*let t0 = Instant::now();
        for _ in 0..500000 {
            generator1.set_stream(&stream1);
            generator1.generate_uniform(&mut vector1);
            generator1.set_stream(&stream2);
            generator1.generate_uniform(&mut vector2);
        }
        stream1.synchronize();
        stream2.synchronize();
        let dt = t0.elapsed();
        println!("2 finished in {}.{}", dt.as_secs(), dt.subsec_nanos());*/

        let t0 = Instant::now();
        generator1.set_stream(&stream1);
        generator2.set_stream(&stream2);
        for _ in 0..500000 {
            generator1.generate_uniform(&mut vector1);
            generator2.generate_uniform(&mut vector2);
        }
        stream1.synchronize();
        stream2.synchronize();
        let dt = t0.elapsed();
        println!("3 finished in {}.{}", dt.as_secs(), dt.subsec_nanos());

        let t0 = Instant::now();
        generator1.set_stream(&stream1);
        for _ in 0..1000000 {
            generator1.generate_uniform(&mut vector1);
        }
        stream1.synchronize();
        let dt = t0.elapsed();
        println!("1 finished in {}.{}", dt.as_secs(), dt.subsec_nanos());

        /*let t0 = Instant::now();
        for _ in 0..500000 {
            generator1.set_stream(&stream1);
            generator1.generate_uniform(&mut vector1);
            generator1.set_stream(&stream2);
            generator1.generate_uniform(&mut vector2);
        }
        stream1.synchronize();
        stream2.synchronize();
        let dt = t0.elapsed();
        println!("2 finished in {}.{}", dt.as_secs(), dt.subsec_nanos());*/

        let t0 = Instant::now();
        generator1.set_stream(&stream1);
        generator2.set_stream(&stream2);
        for _ in 0..500000 {
            generator1.generate_uniform(&mut vector1);
            generator2.generate_uniform(&mut vector2);
        }
        stream1.synchronize();
        stream2.synchronize();
        let dt = t0.elapsed();
        println!("3 finished in {}.{}", dt.as_secs(), dt.subsec_nanos());

        let t0 = Instant::now();
        generator1.set_stream(&stream1);
        for _ in 0..1800000 {
            generator1.generate_uniform(&mut vector1);
        }
        stream1.synchronize();
        let dt = t0.elapsed();
        println!("12 finished in {}.{}", dt.as_secs(), dt.subsec_nanos());

        /*let t0 = Instant::now();
        for _ in 0..500000 {
            generator1.set_stream(&stream1);
            generator1.generate_uniform(&mut vector1);
            generator1.set_stream(&stream2);
            generator1.generate_uniform(&mut vector2);
        }
        stream1.synchronize();
        stream2.synchronize();
        let dt = t0.elapsed();
        println!("2 finished in {}.{}", dt.as_secs(), dt.subsec_nanos());*/

        let mut vector3 = CuVector::new(10000, 1.0);
        let mut generator3 = CurandGenerator::new(CurandRngType::PseudoDefault);
        let stream3 = CudaStream::new();

        let t0 = Instant::now();
        generator1.set_stream(&stream1);
        generator2.set_stream(&stream2);
        generator3.set_stream(&stream3);
        for _ in 0..600000 {
            generator1.generate_uniform(&mut vector1);
            generator2.generate_uniform(&mut vector2);
            generator3.generate_uniform(&mut vector3);
        }
        stream1.synchronize();
        stream2.synchronize();
        let dt = t0.elapsed();
        println!("32 finished in {}.{}", dt.as_secs(), dt.subsec_nanos());
    }
    */


}