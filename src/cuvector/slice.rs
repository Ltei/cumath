

use super::*;


/// A vector slice.
/// Holds a pointer to continuous GPU memory.
pub struct CuVectorSlice<'a> {
    pub(crate) _parent: PhantomData<&'a CuVectorOp>,
    pub(crate) len: usize,
    pub(crate) ptr: *const f32,
}
impl_CuPackedData!(CuVectorSlice<'a>, 'a);
impl_CuVectorOp!(CuVectorSlice<'a>, 'a);


/// A mutable vector slice.
/// Holds a pointer to continuous GPU memory.
pub struct CuVectorSliceMut<'a> {
    pub(crate) _parent: PhantomData<&'a CuVectorOpMut>,
    pub(crate) len: usize,
    pub(crate) ptr: *mut f32,
}
impl_CuPackedDataMut!(CuVectorSliceMut<'a>, 'a);
impl_CuVectorOpMut!(CuVectorSliceMut<'a>, 'a);