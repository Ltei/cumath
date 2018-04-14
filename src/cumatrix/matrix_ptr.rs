

/// A pointer over a matrix : It won't free the inner GPU-pointer when it goes out of scope
pub struct CuMatrixPtr {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) len: usize,
    pub(crate) ptr: *mut f32,
}
impl_CuPackedData!(CuMatrixPtr);
impl_CuPackedDataMut!(CuMatrixPtr);
impl_CuMatrixOp_packed!(CuMatrixPtr);
impl_CuMatrixOpMut_packed!(CuMatrixPtr);