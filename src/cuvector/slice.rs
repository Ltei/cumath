

use super::*;
use std::{ops::{Deref, DerefMut}};


/// A vector slice.
/// Holds a pointer to continuous GPU memory.
#[derive(Debug)]
pub struct CuVectorSlice<'a, T: CuDataType + 'a> {
    pub(crate) _parent: PhantomData<&'a CuVectorDeref<T>>,
    pub(crate) deref: CuVectorDeref<T>,
}

impl<'a, T: CuDataType + 'a> Deref for CuVectorSlice<'a, T> {
    type Target = CuVectorDeref<T>;
    fn deref(&self) -> &CuVectorDeref<T> { &self.deref }
}


/// A mutable vector slice.
/// Holds a pointer to continuous GPU memory.
#[derive(Debug)]
pub struct CuVectorSliceMut<'a, T: CuDataType + 'a> {
    pub(crate) _parent: PhantomData<&'a CuVectorDeref<T>>,
    pub(crate) deref: CuVectorDeref<T>,
}

impl<'a, T: CuDataType + 'a> Deref for CuVectorSliceMut<'a, T> {
    type Target = CuVectorDeref<T>;
    fn deref(&self) -> &CuVectorDeref<T> { &self.deref }
}
impl<'a, T: CuDataType + 'a> DerefMut for CuVectorSliceMut<'a, T> {
    fn deref_mut(&mut self) -> &mut CuVectorDeref<T> { &mut self.deref }
}