


/// A pointer over a vector :
/// - It won't free the inner GPU-pointer when it goes out of scope
/// - It won't check if the underlying memory is still allocated when used
/// -> Use at your own risk
pub struct CuVectorPtr {
    pub(crate) deref: CuVectorPtrDeref,
}
impl CuVectorPtr {

    /// [inline]
    /// Returns the length of the underlying vector (even if the pointed memory has been freed)
    #[inline]
    pub fn len(&self) -> usize {
        self.deref.len
    }

    /// [inline]
    /// Returns an immutable reference to the underlying vector
    #[inline]
    pub unsafe fn deref(&self) -> &CuVectorPtrDeref {
        &self.deref
    }

    /// [inline]
    /// Returns an mutable reference to the underlying vector
    #[inline]
    pub unsafe fn deref_mut(&mut self) -> &mut CuVectorPtrDeref {
        &mut self.deref
    }

}

pub struct CuVectorPtrDeref {
    pub(crate) len: usize,
    pub(crate) ptr: *mut f32,
}
impl_CuPackedDataMut!(CuVectorPtrDeref);
impl_CuVectorOpMut!(CuVectorPtrDeref);