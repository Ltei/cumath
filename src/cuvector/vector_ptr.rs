


/// A pointer over a vector :
/// - It won't free the inner GPU-pointer when it goes out of scope
/// - It won't check if the underlying memory is still allocated when used
/// -> Use at your own risk
pub struct CuVectorPtr {
    pub(crate) len: usize,
    pub(crate) ptr: *mut f32,
}
impl_CuPackedDataMut!(CuVectorPtr);
impl_CuVectorOpMut!(CuVectorPtr);