
use super::*;
use kernel::*;



/// A vector slice.
/// Holds a pointer to possibly non-continuous GPU memory.
pub struct CuMatrixFragment<'a, T: CuDataType + 'a> {
    pub(crate) _parent: PhantomData<&'a CuMatrixOp<T>>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) leading_dimension: usize,
    pub(crate) ptr: *const T,
}
impl_immutable_fragmented_matrix_holder!(CuMatrixFragment, 'a);


/// A mutable vector slice.
/// Holds a pointer to possibly non-continuous GPU memory.
pub struct CuMatrixFragmentMut<'a, T:CuDataType + 'a> {
    pub(crate) parent: PhantomData<&'a CuMatrixOp<T>>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) leading_dimension: usize,
    pub(crate) ptr: *mut T,
}
impl_mutable_fragmented_matrix_holder!(CuMatrixFragmentMut, 'a);