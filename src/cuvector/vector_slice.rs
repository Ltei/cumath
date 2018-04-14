

use super::*;


/// A vector slice.
/// Holds a pointer to continuous GPU memory.
pub struct CuVectorSlice<'a> {
    pub(crate) parent: PhantomData<&'a CuVectorOp>,
    pub(crate) len: usize,
    pub(crate) ptr: *const f32,
}
impl_CuPackedData!(CuVectorSlice<'a>, 'a);
impl_CuVectorOp!(CuVectorSlice<'a>, 'a);


/// A mutable vector slice.
/// Holds a pointer to continuous GPU memory.
pub struct CuVectorSliceMut<'a> {
    pub(crate) parent: PhantomData<&'a CuVectorOpMut>,
    pub(crate) len: usize,
    pub(crate) ptr: *mut f32,
}
impl_CuPackedData!(CuVectorSliceMut<'a>, 'a);
impl_CuPackedDataMut!(CuVectorSliceMut<'a>, 'a);
impl_CuVectorOp!(CuVectorSliceMut<'a>, 'a);
impl_CuVectorOpMut!(CuVectorSliceMut<'a>, 'a);



#[cfg(test)]
mod tests {

    use cuvector::CuVectorOp;

    #[test]
    fn test() {
        let vector = super::CuVector::new(10, 0.0);
        {
            let _slice1 = vector.slice(0, 2);
            let _slice2 = vector.slice(0, 2);
            let _len = _slice2.len();
        }
        /*{
            let slice1 = vector.slice_mut(0, 2);
            let slice2 = vector.slice_mut(0, 2);
        }*/

    }

}