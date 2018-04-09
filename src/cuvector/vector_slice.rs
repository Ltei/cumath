

use super::*;



/// A vector slice.
/// Holds a pointer to continuous GPU memory.
pub struct CuVectorSlice<'a> {
    pub(crate) parent: PhantomData<&'a CuVectorOp>,
    pub(crate) len: usize,
    pub(crate) ptr: *const f32,
}
impl<'a> CuVectorOp for CuVectorSlice<'a>  {
    fn len(&self) -> usize { self.len }
    fn as_ptr(&self) -> *const f32 { self.ptr }
}

/// A mutable vector slice.
/// Holds a pointer to continuous GPU memory.
pub struct CuVectorSliceMut<'a> {
    pub(crate) parent: PhantomData<&'a CuVectorOpMut>,
    pub(crate) len: usize,
    pub(crate) ptr: *mut f32,
}
impl<'a> CuVectorOp for CuVectorSliceMut<'a>   {
    fn len(&self) -> usize { self.len }
    fn as_ptr(&self) -> *const f32 { self.ptr }
}
impl<'a> CuVectorOpMut for CuVectorSliceMut<'a>  {
    fn as_mut_ptr(&mut self) -> *mut f32 { self.ptr }
}


#[cfg(test)]
mod tests {

    use cuvector::CuVectorOp;

    #[test]
    fn test() {
        let vector = super::CuVector::new(10, 0.0);
        {
            let _slice1 = vector.slice(0, 2);
            let _slice2 = vector.slice(0, 2);
        }
        /*{
            let slice1 = vector.slice_mut(0, 2);
            let slice2 = vector.slice_mut(0, 2);
        }*/

    }

}