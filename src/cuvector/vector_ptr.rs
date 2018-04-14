

/// A pointer over a vector : It won't free the inner GPU-pointer when it goes out of scope
pub struct CuVectorPtr {
    pub(crate) len: usize,
    pub(crate) ptr: *mut f32,
}
impl_CuPackedData!(CuVectorPtr);
impl_CuPackedDataMut!(CuVectorPtr);
impl_CuVectorOp!(CuVectorPtr);
impl_CuVectorOpMut!(CuVectorPtr);