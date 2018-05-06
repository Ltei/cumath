
use super::*;
use kernel::*;
use cuda_core::{cuda_ffi::cudaMemcpyKind};
use CuDataType;


/// A pointer over a continuous matrix : It won't free the inner GPU-pointer when it goes out of scope
pub struct CuMatrixPtr<T: CuDataType> {
    pub(crate) deref: CuMatrixPtrDeref<T>,
}
impl<T: CuDataType> CuMatrixPtr<T> {

    /// [inline]
    /// Returns the number of rows of the underlying matrix (even if the pointed memory has been freed)
    #[inline]
    pub fn rows(&self) -> usize {
        self.deref.rows
    }

    /// [inline]
    /// Returns the number of columns of the underlying matrix (even if the pointed memory has been freed)
    #[inline]
    pub fn cols(&self) -> usize {
        self.deref.cols
    }

    /// [inline]
    /// Returns the length of the underlying matrix (even if the pointed memory has been freed)
    #[inline]
    pub fn len(&self) -> usize {
        self.deref.len
    }

    /// [inline]
    /// Returns an immutable reference to the underlying matrix
    #[inline]
    pub unsafe fn deref(&self) -> &CuMatrixPtrDeref<T> {
        &self.deref
    }

    /// [inline]
    /// Returns an mutable reference to the underlying matrix
    #[inline]
    pub unsafe fn deref_mut(&mut self) -> &mut CuMatrixPtrDeref<T> {
        &mut self.deref
    }

}


pub struct CuMatrixPtrDeref<T: CuDataType> {
    pub(crate) ptr: *mut T,
    pub(crate) len: usize,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

impl_mutable_packed_matrix_holder!(CuMatrixPtrDeref);