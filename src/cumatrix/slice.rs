
use super::*;
use kernel::*;



/// A vector slice.
/// Holds a pointer to continuous GPU memory.
pub struct CuMatrixSlice<'a, T: CuDataType + 'a> {
    pub(crate) _parent: PhantomData<&'a CuMatrixOp<T>>,
    pub(crate) ptr: *const T,
    pub(crate) len: usize,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}
impl_immutable_packed_matrix_holder!(CuMatrixSlice, 'a);

impl<'a, T: CuDataType + 'a> CuMatrixSlice<'a, T> {
    pub fn slice_col<'b>(&'b self, col_offset: usize, nb_cols: usize) -> CuMatrixSlice<'b, T> {
        #[cfg(not(feature = "disable_checks"))] {
            assert_infeq_usize(col_offset + nb_cols, "col_offset+nb_cols", self.cols(), "self.cols()");
        }

        CuMatrixSlice {
            _parent: PhantomData,
            ptr: unsafe { self.ptr.offset((col_offset*self.leading_dimension()) as isize) },
            len: self.leading_dimension()*nb_cols,
            rows: self.leading_dimension(),
            cols: nb_cols,
        }
    }
}


/// A mutable vector slice.
/// Holds a pointer to continuous GPU memory.
pub struct CuMatrixSliceMut<'a, T: CuDataType + 'a> {
    pub(crate) _parent: PhantomData<&'a CuMatrixOp<T>>,
    pub(crate) ptr: *mut T,
    pub(crate) len: usize,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}
impl_mutable_packed_matrix_holder!(CuMatrixSliceMut, 'a);

impl<'a, T: CuDataType + 'a> CuMatrixSliceMut<'a, T> {
    pub fn slice_col<'b>(&'b self, col_offset: usize, nb_cols: usize) -> CuMatrixSlice<'b, T> {
        #[cfg(not(feature = "disable_checks"))] {
            assert_infeq_usize(col_offset + nb_cols, "col_offset+nb_cols", self.cols(), "self.cols()");
        }
        CuMatrixSlice {
            _parent: PhantomData,
            rows: self.leading_dimension(),
            cols: nb_cols,
            len: self.leading_dimension()*nb_cols,
            ptr: unsafe { self.ptr.offset((col_offset*self.leading_dimension()) as isize) },
        }
    }
    pub fn slice_col_mut<'b>(&'b mut self, col_offset: usize, nb_cols: usize) -> CuMatrixSliceMut<'b, T> {
        #[cfg(not(feature = "disable_checks"))] {
            assert_infeq_usize(col_offset + nb_cols, "col_offset+nb_cols", self.cols(), "self.cols()");
        }
        CuMatrixSliceMut {
            _parent: PhantomData,
            rows: self.leading_dimension(),
            cols: nb_cols,
            len: self.leading_dimension()*nb_cols,
            ptr: unsafe { self.ptr.offset((col_offset*self.leading_dimension()) as isize) },
        }
    }
}