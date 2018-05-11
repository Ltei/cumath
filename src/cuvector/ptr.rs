
use super::*;
use CuDataType;



/// An immutable pointer over a vector :
/// - It won't free the inner GPU-pointer when it goes out of scope
/// - It won't check if the underlying memory is still allocated when used
/// -> Use at your own risk
#[derive(Debug)]
pub struct CuVectorPtr<T: CuDataType> {
    pub(crate) deref: CuVectorDeref<T>,
}

impl<T: CuDataType> CuVectorPtr<T> {

    /// [inline]
    /// Returns the length of the underlying vector (even if the pointed memory has been freed)
    #[inline]
    pub fn len(&self) -> usize { self.deref.len as usize }

    /// [inline]
    /// Returns an immutable reference to the underlying vector
    #[inline]
    pub unsafe fn deref(&self) -> &CuVectorDeref<T> {
        &self.deref
    }

}




/// A mutable pointer over a vector :
/// - It won't free the inner GPU-pointer when it goes out of scope
/// - It won't check if the underlying memory is still allocated when used
/// -> Use at your own risk
#[derive(Debug)]
pub struct CuVectorMutPtr<T: CuDataType> {
    pub(crate) deref: CuVectorDeref<T>,
}

impl<T: CuDataType> CuVectorMutPtr<T> {

    /// [inline]
    /// Returns the length of the underlying vector (even if the pointed memory has been freed)
    #[inline]
    pub fn len(&self) -> usize { self.deref.len as usize }

    /// [inline]
    /// Returns an immutable reference to the underlying vector
    #[inline]
    pub unsafe fn deref(&self) -> &CuVectorDeref<T> {
        &self.deref
    }

    /// [inline]
    /// Returns an mutable reference to the underlying vector
    #[inline]
    pub unsafe fn deref_mut(&mut self) -> &mut CuVectorDeref<T> {
        &mut self.deref
    }

}