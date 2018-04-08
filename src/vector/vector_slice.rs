

use super::*;



// CuVectorSlice
pub struct CuVectorSlice<'a, T> where T: CuVectorOp + 'a {
    pub(super) parent: PhantomData<&'a T>,
    pub(super) len: usize,
    pub(super) ptr: *const f32,
}
impl<'a, T> CuVectorOp for CuVectorSlice<'a, T> where T: CuVectorOp + 'a  {
    fn len(&self) -> usize { self.len }
    fn ptr(&self) -> *const f32 { self.ptr }
}

// CuVectorSliceMut
pub struct CuVectorSliceMut<'a, T> where T: CuVectorOp + 'a {
    pub(super) parent: PhantomData<&'a T>,
    pub(super) len: usize,
    pub(super) ptr: *mut f32,
}
impl<'a, T> CuVectorOp for CuVectorSliceMut<'a, T> where T: CuVectorOp + 'a  {
    fn len(&self) -> usize { self.len }
    fn ptr(&self) -> *const f32 { self.ptr }
}
impl<'a, T> CuVectorOpMut for CuVectorSliceMut<'a, T> where T: CuVectorOp + 'a  {
    fn ptr_mut(&mut self) -> *mut f32 { self.ptr }
}

