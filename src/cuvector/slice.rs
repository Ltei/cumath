

use super::*;


/// A vector slice.
/// Holds a pointer to continuous GPU memory.
pub struct CuVectorSlice<'a, T: CuDataType + 'a> {
    pub(crate) _parent: PhantomData<&'a CuVectorOp<T>>,
    pub(crate) len: usize,
    pub(crate) ptr: *const T,
}
impl_immutable_vector_holder!(CuVectorSlice, 'a);


/// A mutable vector slice.
/// Holds a pointer to continuous GPU memory.
pub struct CuVectorSliceMut<'a, T: CuDataType + 'a> {
    pub(crate) _parent: PhantomData<&'a CuVectorOp<T>>, // TODO Should be CuVectorOpMut<T>
    pub(crate) len: usize,
    pub(crate) ptr: *mut T,
}
impl_mutable_vector_holder!(CuVectorSliceMut, 'a);