#![allow(dead_code)]

use super::cuda_ffi::*;



pub enum StructCublasContext {}


#[derive(PartialEq, Debug, Clone, Copy)]
#[repr(u32)]
pub enum CublasStatus {
    Success = 0,
    NotInitialized = 1,
    AllocFailed = 3,
    InvalidValue = 7,
    ArchMismatch = 8,
    MappingError = 11,
    ExecutionFailed = 13,
    InternalError = 14,
    NotSupported = 15,
    LicenseError = 16,
}
impl CublasStatus {
    fn assert_success(&self) {
        assert_eq!(self, &CublasStatus::Success);
    }
    fn get_error_str(&self) -> Option<&'static str> {
        match *self {
            CublasStatus::Success => None,
            CublasStatus::NotInitialized => Some("NotInitialized"),
            CublasStatus::AllocFailed => Some("AllocFailed"),
            CublasStatus::InvalidValue => Some("InvalidValue"),
            CublasStatus::ArchMismatch => Some("ArchMismatch"),
            CublasStatus::MappingError => Some("MappingError"),
            CublasStatus::ExecutionFailed => Some("ExecutionFailed"),
            CublasStatus::InternalError => Some("InternalError"),
            CublasStatus::NotSupported => Some("NotSupported"),
            CublasStatus::LicenseError => Some("LicenseError"),
        }
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]
#[repr(u32)]
pub enum CublasOperation {
    None = 0,
    Transpose = 1,
    ConjugateTranspose = 2,
}


extern {
    fn cublasCreate_v2(handle: *mut*mut StructCublasContext) -> CublasStatus;

    fn cublasDestroy_v2(handle: *mut StructCublasContext) -> CublasStatus;

    fn cublasSasum_v2(handle: *mut StructCublasContext,
                          n: i32,
                          x: *const f32,
                          incx: i32,
                          result: *mut f32) -> CublasStatus;

    fn cublasSaxpy_v2(handle: *mut StructCublasContext,
                          n: i32,
                          alpha: *const f32,
                          x: *const f32,
                          incx: i32,
                          y: *mut f32,
                          incy: i32) -> CublasStatus;

    fn cublasSgemv_v2(handle: *mut StructCublasContext,
                          trans: CublasOperation,
                          m: i32, n: i32,
                          alpha: *const f32,
                          A: *const f32, lda: i32,
                          x: *const f32, incx: i32,
                          beta: *const f32,
                          y: *mut f32, incy: i32) -> CublasStatus;

    fn cublasSgemm_v2(handle: *mut StructCublasContext,
                          transa: CublasOperation, transb: CublasOperation,
                          m: i32, n: i32, k: i32,
                          alpha: *const f32,
                          A: *const f32, lda: i32,
                          B: *const f32, ldb: i32,
                          beta: *const f32,
                          C: *mut f32, ldc: i32) -> CublasStatus;

    fn cublasSetStream_v2(handle: *mut StructCublasContext, stream: *mut Struct_cudaStream_t) -> CublasStatus;
}


#[inline]
pub fn cublas_create(handle: *mut*mut StructCublasContext) -> Option<&'static str> {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cublasCreate_v2(handle) }.get_error_str()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cublasCreate_v2(handle) };
        None
    }
}

#[inline]
pub fn cublas_destroy(handle: *mut StructCublasContext) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cublasDestroy_v2(handle) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cublasDestroy_v2(handle) };
    }
}

#[inline]
pub fn cublas_sasum(handle: *mut StructCublasContext,
                    n: i32,
                    x: *const f32, incx: i32,
                    result: *mut f32) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cublasSasum_v2(handle, n, x, incx, result) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cublasSasum_v2(handle, n, x, incx, result) };
    }
}

#[inline]
pub fn cublas_saxpy(handle: *mut StructCublasContext,
                    n: i32, alpha: *const f32,
                    x: *const f32, incx: i32,
                    y: *mut f32, incy: i32) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cublasSaxpy_v2(handle, n, alpha, x, incx, y, incy) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cublasSaxpy_v2(handle, n, alpha, x, incx, y, incy) };
    }
}

#[inline]
pub fn cublas_sgemv(handle: *mut StructCublasContext,
                    trans: CublasOperation,
                    m: i32, n: i32,
                    alpha: *const f32,
                    a: *const f32, lda: i32,
                    x: *const f32, incx: i32,
                    beta: *const f32,
                    y: *mut f32, incy: i32) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cublasSgemv_v2(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cublasSgemv_v2(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy) };
    }
}

#[inline]
pub fn cublas_sgemm(handle: *mut StructCublasContext,
                    transa: CublasOperation, transb: CublasOperation,
                    m: i32, n: i32, k: i32,
                    alpha: *const f32,
                    a: *const f32, lda: i32,
                    b: *const f32, ldb: i32,
                    beta: *const f32,
                    c: *mut f32, ldc: i32) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cublasSgemm_v2(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cublasSgemm_v2(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) };
    }
}

#[inline]
pub fn cublas_set_stream(handle: *mut StructCublasContext, stream: *mut Struct_cudaStream_t) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cublasSetStream_v2(handle, stream) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cublasSetStream_v2(handle, stream) };
    }
}