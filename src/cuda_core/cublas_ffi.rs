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


    // Helper

    fn cublasCreate_v2(handle: *mut*mut StructCublasContext) -> CublasStatus;

    fn cublasDestroy_v2(handle: *mut StructCublasContext) -> CublasStatus;

    fn cublasSetStream_v2(
        handle: *mut StructCublasContext,
        stream: *mut Struct_cudaStream_t
    ) -> CublasStatus;


    // Level 1

    fn cublasIsamax_v2(
        handle: *mut StructCublasContext,
        n: i32,
        x: *const f32,
        incx: i32,
        result: *mut i32,
    ) -> CublasStatus;

    fn cublasIsamin_v2(
        handle: *mut StructCublasContext,
        n: i32,
        x: *const f32,
        incx: i32,
        result: *mut i32,
    ) -> CublasStatus;

    fn cublasSasum_v2(
        handle: *mut StructCublasContext,
        n: i32,
        x: *const f32,
        incx: i32,
        result: *mut f32
    ) -> CublasStatus;

    fn cublasSaxpy_v2(
        handle: *mut StructCublasContext,
        n: i32,
        alpha: *const f32,
        x: *const f32,
        incx: i32,
        y: *mut f32,
        incy: i32
    ) -> CublasStatus;

    fn cublasSdot_v2(
        handle: *mut StructCublasContext,
        n: i32,
        x: *const f32,
        incx: i32,
        y: *const f32,
        incy: i32,
        result: *mut f32,
    ) -> CublasStatus;

    fn cublasSscal_v2(
        handle: *mut StructCublasContext,
        n: i32,
        alpha: *const f32,
        x: *mut  f32,
        incx: i32,
    ) -> CublasStatus;

    fn cublasSswap_v2(
        handle: *mut StructCublasContext,
        n: i32,
        x: *const f32,
        incx: i32,
        y: *const f32,
        incy: i32
    ) -> CublasStatus;


    // Level 2

    fn cublasSgemv_v2(
        handle: *mut StructCublasContext,
        trans: CublasOperation,
        m: i32, n: i32,
        alpha: *const f32,
        A: *const f32, lda: i32,
        x: *const f32, incx: i32,
        beta: *const f32,
        y: *mut f32, incy: i32
    ) -> CublasStatus;


    // Level 3

    fn cublasSgemm_v2(
        handle: *mut StructCublasContext,
        transa: CublasOperation, transb: CublasOperation,
        m: i32, n: i32, k: i32,
        alpha: *const f32,
        A: *const f32, lda: i32,
        B: *const f32, ldb: i32,
        beta: *const f32,
        C: *mut f32, ldc: i32
    ) -> CublasStatus;

    fn cublasSgemmBatched_v2(
        handle: *mut StructCublasContext,
        transa: CublasOperation, transb: CublasOperation,
        m: i32, n: i32, k: i32,
        alpha: *const f32,
        Aarray: *const*const f32, lda: i32,
        Barray: *const*const f32, ldb: i32,
        beta: *const f32,
        Carray: *mut*mut f32, ldc: i32
    ) -> CublasStatus;


}


// Helper

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
pub fn cublas_set_stream(handle: *mut StructCublasContext, stream: *mut Struct_cudaStream_t) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cublasSetStream_v2(handle, stream) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cublasSetStream_v2(handle, stream) };
    }
}


// Level 1

#[inline]
pub fn cublas_isamax(handle: *mut StructCublasContext, n: i32, x: *const f32, incx: i32, result: *mut i32) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cublasIsamax_v2(handle, n, x, incx, result) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cublasIsamax_v2(handle, n, x, incx, result) };
    }
}

#[inline]
pub fn cublas_isamin(handle: *mut StructCublasContext, n: i32, x: *const f32, incx: i32, result: *mut i32) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cublasIsamin_v2(handle, n, x, incx, result) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cublasIsamin_v2(handle, n, x, incx, result) };
    }
}

#[inline]
pub fn cublas_sasum(handle: *mut StructCublasContext, n: i32, x: *const f32, incx: i32, result: *mut f32) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cublasSasum_v2(handle, n, x, incx, result) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cublasSasum_v2(handle, n, x, incx, result) };
    }
}

#[inline]
pub fn cublas_saxpy(handle: *mut StructCublasContext, n: i32, alpha: *const f32, x: *const f32, incx: i32, y: *mut f32, incy: i32) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cublasSaxpy_v2(handle, n, alpha, x, incx, y, incy) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cublasSaxpy_v2(handle, n, alpha, x, incx, y, incy) };
    }
}

#[inline]
pub fn cublas_sdot(handle: *mut StructCublasContext, n: i32, x: *const f32, incx: i32, y: *const f32, incy: i32, result: *mut f32) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cublasSdot_v2(handle, n, x, incx, y, incy, result) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cublasSdot_v2(handle, n, x, incx, y, incy, result) };
    }
}

#[inline]
pub fn cublas_sscal(handle: *mut StructCublasContext, n: i32, alpha: *const f32, x: *mut f32, incx: i32) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cublasSscal_v2(handle, n, alpha, x, incx) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cublasSscal_v2(handle, n, alpha, x, incx) };
    }
}

#[inline]
pub fn cublas_sswap(handle: *mut StructCublasContext, n: i32, x: *const f32, incx: i32, y: *const f32, incy: i32) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cublasSswap_v2(handle, n, x, incx, y, incy) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cublasSswap_v2(handle, n, x, incx, y, incy) };
    }
}


// Level 2

#[inline]
pub fn cublas_sgemv(handle: *mut StructCublasContext, trans: CublasOperation, m: i32, n: i32, alpha: *const f32, a: *const f32, lda: i32, x: *const f32, incx: i32, beta: *const f32, y: *mut f32, incy: i32) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cublasSgemv_v2(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cublasSgemv_v2(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy) };
    }
}


// Level 3

#[inline]
pub fn cublas_sgemm(handle: *mut StructCublasContext, transa: CublasOperation, transb: CublasOperation, m: i32, n: i32, k: i32, alpha: *const f32, a: *const f32, lda: i32, b: *const f32, ldb: i32, beta: *const f32, c: *mut f32, ldc: i32) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cublasSgemm_v2(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cublasSgemm_v2(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) };
    }
}

#[inline]
pub fn cublas_sgemm_batched(handle: *mut StructCublasContext, transa: CublasOperation, transb: CublasOperation, m: i32, n: i32, k: i32, alpha: *const f32, a_array: *const*const f32, lda: i32, b_array: *const*const f32, ldb: i32, beta: *const f32, c_array: *mut*mut f32, ldc: i32) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cublasSgemmBatched_v2(handle, transa, transb, m, n, k, alpha, a_array, lda, b_array, ldb, beta, c_array, ldc) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cublasSgemmBatched_v2(handle, transa, transb, m, n, k, alpha, a_array, lda, b_array, ldb, beta, c_array, ldc) };
    }
}