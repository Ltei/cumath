
#![allow(dead_code)]

use super::cuda_ffi::*;


pub enum StructCurandGenerator {}


#[derive(PartialEq, Debug, Clone, Copy)]
#[repr(u32)]
pub enum CurandStatus {
    Success = 0,
    VersionMismatch = 100,
    NotInitialized = 101,
    AllocationFailed = 102,
    TypeError = 103,
    OutOfRand = 104,
    LengthNotMultiple = 105,
    DoublePrecisionRequired = 106,
    LaunchFailure = 201,
    PreexistingFailure = 202,
    InitializationFailed = 203,
    ArchMismatch = 204,
    InternalError = 999,
}
impl CurandStatus {
    fn assert_success(&self) {
        assert_eq!(self, &CurandStatus::Success)
    }
    fn get_error_str(&self) -> Option<&'static str> {
        match *self {
            CurandStatus::Success => None,
            CurandStatus::VersionMismatch => Some("VersionMismatch"),
            CurandStatus::NotInitialized => Some("NotInitialized"),
            CurandStatus::AllocationFailed => Some("AllocationFailed"),
            CurandStatus::TypeError => Some("TypeError"),
            CurandStatus::OutOfRand => Some("OutOfRand"),
            CurandStatus::LengthNotMultiple => Some("LengthNotMultiple"),
            CurandStatus::DoublePrecisionRequired => Some("DoublePrecisionRequired"),
            CurandStatus::LaunchFailure => Some("LaunchFailure"),
            CurandStatus::PreexistingFailure => Some("PreexistingFailure"),
            CurandStatus::InitializationFailed => Some("InitializationFailed"),
            CurandStatus::ArchMismatch => Some("ArchMismatch"),
            CurandStatus::InternalError => Some("InternalError"),
        }
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum CurandRngType {
    Test = 0,
    PseudoDefault = 100,
    PseudoXORWOW = 101,
    PseudoMRG32K3A = 121,
    PseudoMTGP32 = 141,
    PseudoMT19937 = 142,
    PseudoPHILOX4_32 = 161,
    QuasiDefault = 200,
    QuasiSOBOL32 = 201,
    QuasiScrambledSOBOL32 = 202,
    QuasiSOBOL64 = 203,
    QuasiScrambledSOBOL64 = 204,
}


extern {
    fn curandCreateGenerator(generator: *mut*mut StructCurandGenerator, rng_type: CurandRngType) -> CurandStatus;
    fn curandDestroyGenerator(generator: *mut StructCurandGenerator) -> CurandStatus;

    fn curandGenerateUniform(generator: *mut StructCurandGenerator, outputPtr: *mut f32, num: usize) -> CurandStatus;
    fn curandGenerateNormal(generator: *mut StructCurandGenerator, outputPtr: *mut f32, num: usize, mean: f32, stddev: f32) -> CurandStatus;
    fn curandGenerateLogNormal(generator: *mut StructCurandGenerator, outputPtr: *mut f32, num: usize, mean: f32, stddev: f32) -> CurandStatus;
    fn curandGeneratePoisson(generator: *mut StructCurandGenerator, outputPtr: *mut f32, num: usize, lambda: f32) -> CurandStatus;

    fn curandSetStream(generator: *mut StructCurandGenerator, stream: cudaStream_t) -> CurandStatus;
}


#[inline]
pub fn curand_create_generator(generator: *mut*mut StructCurandGenerator, rng_type: CurandRngType) -> Option<&'static str> {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { curandCreateGenerator(generator, rng_type) }.get_error_str()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { curandCreateGenerator(generator, rng_type) };
        None
    }
}

#[inline]
pub fn curand_destroy_generator(generator: *mut StructCurandGenerator) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { curandDestroyGenerator(generator) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { curandDestroyGenerator(generator) };
    }
}

#[inline]
pub fn curand_generate_uniform(generator: *mut StructCurandGenerator, output_ptr: *mut f32, num: usize) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { curandGenerateUniform(generator, output_ptr, num) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { curandGenerateUniform(generator, output_ptr, num) };
    }
}

#[inline]
pub fn curand_generate_normal(generator: *mut StructCurandGenerator, output_ptr: *mut f32, num: usize, mean: f32, stddev: f32) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { curandGenerateNormal(generator, output_ptr, num, mean, stddev) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { curandGenerateNormal(generator, output_ptr, num, mean, stddev) };
    }
}

#[inline]
pub fn curand_generate_lognormal(generator: *mut StructCurandGenerator, output_ptr: *mut f32, num: usize, mean: f32, stddev: f32) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { curandGenerateLogNormal(generator, output_ptr, num, mean, stddev) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { curandGenerateLogNormal(generator, output_ptr, num, mean, stddev) };
    }
}

#[inline]
pub fn curand_generate_poisson(generator: *mut StructCurandGenerator, output_ptr: *mut f32, num: usize, lambda: f32) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { curandGeneratePoisson(generator, output_ptr, num, lambda) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { curandGeneratePoisson(generator, output_ptr, num, lambda) };
    }
}

#[inline]
pub fn curand_set_stream(generator: *mut StructCurandGenerator, stream: cudaStream_t) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { curandSetStream(generator, stream) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { curandSetStream(generator, stream) };
    }
}



