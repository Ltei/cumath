
use super::*;
use std::ops::{Deref, DerefMut};



/// A vector slice.
/// Holds a pointer to possibly non-continuous GPU memory.
#[derive(Debug)]
pub struct CuMatrixFragment<'a, T: CuDataType + 'a> {
    pub(crate) _parent: PhantomData<&'a CuMatrixDeref<T>>,
    pub(crate) deref: CuMatrixDeref<T>,
}

impl<'a, T: CuDataType + 'a> Deref for CuMatrixFragment<'a, T> {
    type Target = CuMatrixDeref<T>;
    fn deref(&self) -> &CuMatrixDeref<T> { &self.deref }
}


/// A mutable vector slice.
/// Holds a pointer to possibly non-continuous GPU memory.
#[derive(Debug)]
pub struct CuMatrixFragmentMut<'a, T:CuDataType + 'a> {
    pub(crate) parent: PhantomData<&'a CuMatrixDeref<T>>,
    pub(crate) deref: CuMatrixDeref<T>,
}

impl<'a, T: CuDataType + 'a> Deref for CuMatrixFragmentMut<'a, T> {
    type Target = CuMatrixDeref<T>;
    fn deref(&self) -> &CuMatrixDeref<T> { &self.deref }
}
impl<'a, T: CuDataType + 'a> DerefMut for CuMatrixFragmentMut<'a, T> {
    fn deref_mut(&mut self) -> &mut CuMatrixDeref<T> { &mut self.deref }
}