
use super::*;
use CuDataType;


/// A pointer over a matrix : It won't free the inner GPU-pointer when it goes out of scope
#[derive(Debug)]
pub struct CuMatrixPtr<T: CuDataType> {
    pub(crate) deref: CuMatrixDeref<T>,
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
    pub unsafe fn deref(&self) -> &CuMatrixDeref<T> {
        &self.deref
    }

}


/// A pointer over a mutable matrix : It won't free the inner GPU-pointer when it goes out of scope
#[derive(Debug)]
pub struct CuMatrixMutPtr<T: CuDataType> {
    pub(crate) deref: CuMatrixDeref<T>,
}
impl<T: CuDataType> CuMatrixMutPtr<T> {

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
    pub unsafe fn deref(&self) -> &CuMatrixDeref<T> {
        &self.deref
    }

    /// [inline]
    /// Returns an mutable reference to the underlying matrix
    #[inline]
    pub unsafe fn deref_mut(&mut self) -> &mut CuMatrixDeref<T> {
        &mut self.deref
    }

}
