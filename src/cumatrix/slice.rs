
use super::*;



/// A vector slice.
/// Holds a pointer to continuous GPU memory.
pub struct CuMatrixSlice<'a> {
    pub(crate) parent: PhantomData<&'a CuMatrixOp>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) len: usize,
    pub(crate) ptr: *const f32,
}
impl_CuPackedData!(CuMatrixSlice<'a>, 'a);
impl_CuMatrixOp_packed!(CuMatrixSlice<'a>, 'a);

impl<'a> CuMatrixSlice<'a> {
    pub fn slice_col<'b>(&'b self, col_offset: usize, nb_cols: usize) -> CuMatrixSlice<'b> {
        #[cfg(not(feature = "disable_checks"))] {
            assert_infeq_usize(col_offset + nb_cols, "col_offset+nb_cols", self.cols(), "self.cols()");
        }

        CuMatrixSlice {
            parent: PhantomData,
            rows: self.leading_dimension(),
            cols: nb_cols,
            len: self.leading_dimension()*nb_cols,
            ptr: unsafe { self.ptr.offset((col_offset*self.leading_dimension()) as isize) },
        }
    }
}


/// A mutable vector slice.
/// Holds a pointer to continuous GPU memory.
pub struct CuMatrixSliceMut<'a> {
    pub(crate) _parent: PhantomData<&'a CuMatrixOpMut>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) len: usize,
    pub(crate) ptr: *mut f32,
}
impl_CuPackedDataMut!(CuMatrixSliceMut<'a>, 'a);
impl_CuMatrixOpMut_packed!(CuMatrixSliceMut<'a>, 'a);

impl <'a> CuMatrixSliceMut<'a> {
    pub fn slice_col<'b>(&'b self, col_offset: usize, nb_cols: usize) -> CuMatrixSlice<'b> {
        #[cfg(not(feature = "disable_checks"))] {
            assert_infeq_usize(col_offset + nb_cols, "col_offset+nb_cols", self.cols(), "self.cols()");
        }
        CuMatrixSlice {
            parent: PhantomData,
            rows: self.leading_dimension(),
            cols: nb_cols,
            len: self.leading_dimension()*nb_cols,
            ptr: unsafe { self.ptr.offset((col_offset*self.leading_dimension()) as isize) },
        }
    }
    pub fn slice_col_mut<'b>(&'b mut self, col_offset: usize, nb_cols: usize) -> CuMatrixSliceMut<'b> {
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