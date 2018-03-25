#![allow(dead_code)]



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
    pub fn assert_success(&self) {
        assert_eq!(self, &CublasStatus::Success);
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
    pub fn cublasCreate_v2(handle: *mut*mut StructCublasContext) -> CublasStatus;

    pub fn cublasDestroy_v2(handle: *mut StructCublasContext) -> CublasStatus;

    pub fn cublasSasum_v2(handle: *mut StructCublasContext,
                          n: i32,
                          x: *const f32,
                          incx: i32,
                          result: *mut f32) -> CublasStatus;

    pub fn cublasSaxpy_v2(handle: *mut StructCublasContext,
                          n: i32,
                          alpha: *const f32,
                          x: *const f32,
                          incx: i32,
                          y: *mut f32,
                          incy: i32) -> CublasStatus;

    pub fn cublasSgemv_v2(handle: *mut StructCublasContext,
                          trans: CublasOperation,
                          m: i32, n: i32,
                          alpha: *const f32,
                          A: *const f32, lda: i32,
                          x: *const f32, incx: i32,
                          beta: *const f32,
                          y: *mut f32, incy: i32) -> CublasStatus;

    pub fn cublasSgemm_v2(handle: *mut StructCublasContext,
                          transa: CublasOperation, transb: CublasOperation,
                          m: i32, n: i32, k: i32,
                          alpha: *const f32,
                          A: *const f32, lda: i32,
                          B: *const f32, ldb: i32,
                          beta: *const f32,
                          C: *mut f32, ldc: i32) -> CublasStatus;
}