
use super::*;
use std::ops::{Deref, DerefMut};



/// A vector slice.
/// Holds a pointer to continuous GPU memory.
#[derive(Debug)]
pub struct CuMatrixSlice<'a, T: CuDataType + 'a> {
    pub(crate) _parent: PhantomData<&'a CuMatrixDeref<T>>,
    pub(crate) deref: CuMatrixDeref<T>,
}

impl<'a, T: CuDataType + 'a> Deref for CuMatrixSlice<'a, T> {
    type Target = CuMatrixDeref<T>;
    fn deref(&self) -> &CuMatrixDeref<T> { &self.deref }
}

impl<'a, T: CuDataType + 'a> CuMatrixSlice<'a, T> {

    /// Returns a vector slice containing this matrix datas
    pub fn as_vector(&self) -> ::CuVectorSlice<T> {
        ::CuVectorSlice {
            _parent: PhantomData,
            deref: ::CuVectorDeref {
                ptr: self.ptr,
                len: self.len,
            }
        }
    }

    pub fn slice_col<'b>(&'b self, col_offset: usize, nb_cols: usize) -> CuMatrixSlice<'b, T> {
        #[cfg(not(feature = "disable_checks"))] {
            assert_infeq_usize(col_offset + nb_cols, "col_offset+nb_cols", self.cols(), "self.cols()");
        }
        CuMatrixSlice {
            _parent: PhantomData,
            deref: CuMatrixDeref {
                ptr: unsafe { self.ptr.offset((col_offset * self.leading_dimension()) as isize) },
                len: self.deref.leading_dimension * nb_cols,
                rows: self.deref.leading_dimension,
                cols: nb_cols,
                leading_dimension: self.deref.leading_dimension
            }
        }
    }

}


/// A mutable vector slice.
/// Holds a pointer to continuous GPU memory.
#[derive(Debug)]
pub struct CuMatrixSliceMut<'a, T: CuDataType + 'a> {
    pub(crate) _parent: PhantomData<&'a CuMatrixDeref<T>>,
    pub(crate) deref: CuMatrixDeref<T>,
}

impl<'a, T: CuDataType + 'a> Deref for CuMatrixSliceMut<'a, T> {
    type Target = CuMatrixDeref<T>;
    fn deref(&self) -> &CuMatrixDeref<T> { &self.deref }
}
impl<'a, T: CuDataType + 'a> DerefMut for CuMatrixSliceMut<'a, T> {
    fn deref_mut(&mut self) -> &mut CuMatrixDeref<T> { &mut self.deref }
}

impl<'a, T: CuDataType + 'a> CuMatrixSliceMut<'a, T> {

    /// Returns a vector slice containing this matrix datas
    pub fn as_vector(&self) -> ::CuVectorSlice<T> {
        ::CuVectorSlice {
            _parent: PhantomData,
            deref: ::CuVectorDeref {
                ptr: self.ptr,
                len: self.len,
            }
        }
    }

    /// Returns a mutable vector slice containing this matrix datas
    pub fn as_mut_vector(&mut self) -> ::CuVectorSliceMut<T> {
        ::CuVectorSliceMut {
            _parent: PhantomData,
            deref: ::CuVectorDeref {
                ptr: self.as_mut_ptr(),
                len: self.len(),
            }
        }
    }

    pub fn slice_col<'b>(&'b self, col_offset: usize, nb_cols: usize) -> CuMatrixSlice<'b, T> {
        #[cfg(not(feature = "disable_checks"))] {
            assert_infeq_usize(col_offset + nb_cols, "col_offset+nb_cols", self.cols(), "self.cols()");
        }
        CuMatrixSlice {
            _parent: PhantomData,
            deref: CuMatrixDeref {
                ptr: unsafe { self.ptr.offset((col_offset * self.deref.leading_dimension) as isize) },
                len: self.deref.leading_dimension * nb_cols,
                rows: self.deref.leading_dimension,
                cols: nb_cols,
                leading_dimension: self.deref.leading_dimension,
            }
        }
    }
    pub fn slice_col_mut<'b>(&'b mut self, col_offset: usize, nb_cols: usize) -> CuMatrixSliceMut<'b, T> {
        #[cfg(not(feature = "disable_checks"))] {
            assert_infeq_usize(col_offset + nb_cols, "col_offset+nb_cols", self.cols(), "self.cols()");
        }
        CuMatrixSliceMut {
            _parent: PhantomData,
            deref: CuMatrixDeref {
                ptr: unsafe { self.ptr.offset((col_offset * self.deref.leading_dimension) as isize) },
                len: self.deref.leading_dimension * nb_cols,
                rows: self.deref.leading_dimension,
                cols: nb_cols,
                leading_dimension: self.deref.leading_dimension,
            }
        }
    }

}