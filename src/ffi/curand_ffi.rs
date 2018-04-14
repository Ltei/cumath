
#![allow(dead_code)]



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
    pub fn assert_success(&self) {
        assert_eq!(self, &CurandStatus::Success)
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
}


pub fn curand_create_generator(generator: *mut*mut StructCurandGenerator, rng_type: CurandRngType) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { curandCreateGenerator(generator, rng_type) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { curandCreateGenerator(generator, rng_type) };
    }
}

pub fn curand_destroy_generator(generator: *mut StructCurandGenerator) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { curandDestroyGenerator(generator) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { curandDestroyGenerator(generator) };
    }
}

pub fn curand_generate_uniform(generator: *mut StructCurandGenerator, output_ptr: *mut f32, num: usize) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { curandGenerateUniform(generator, output_ptr, num) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { curandGenerateUniform(generator, output_ptr, num) };
    }
}