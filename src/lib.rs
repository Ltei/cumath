
#[macro_use]
mod macros;
pub use macros::*;

mod ffi;
mod meta;

mod cuda;
pub use cuda::*;

mod cublas;
pub use cublas::*;

mod curand;
pub use curand::*;

mod cuvector;
pub use cuvector::*;

mod cumatrix;
pub use cumatrix::*;



/// Holds generic utilities that applies to vectors and matrices
pub mod cudata {

    use super::*;
    use std::marker::PhantomData;

    /// Continuous GPU-memory trait
    pub trait CuPackedData {
        fn len(&self) -> usize;
        fn as_ptr(&self) -> *const f32;
    }

    /// Mutable continuous GPU-memory trait
    pub trait CuPackedDataMut: CuPackedData {
        fn as_mut_ptr(&mut self) -> *mut f32;
    }

    /// An iterator over GPU data, returning vector and matrix slices.
    pub struct CuDataSliceIter<'a> {
        pub(super) parent: PhantomData<&'a CuPackedData>,
        pub(super) len: usize,
        pub(super) ptr: *const f32,
    }
    impl<'a> CuDataSliceIter<'a> {

        #[inline]
        pub fn len(&self) -> usize { self.len }

        /// Returns Some if the iterator's remaining length is more than 'len'
        /// Returns None otherwise
        pub fn next_vector<'b, 'c>(&'c mut self, len: usize) -> Option<CuVectorSlice<'b>> where 'a: 'b, 'b: 'c {
            match len <= self.len {
                true => {
                    let ptr = self.ptr;
                    self.ptr = unsafe { self.ptr.offset(len as isize) };
                    self.len -= len;
                    Some(CuVectorSlice { _parent: PhantomData, len, ptr })
                }
                false => None
            }
        }

        /// Returns Some if the iterator's remaining length is more than 'rows'*'cols'
        /// Returns None otherwise
        pub fn next_matrix<'b, 'c>(&'c mut self, rows: usize, cols: usize) -> Option<CuMatrixSlice<'b>> where 'a: 'b, 'b: 'c {
            let len = rows * cols;
            match len <= self.len {
                true => {
                    let ptr = self.ptr;
                    self.ptr = unsafe { self.ptr.offset(len as isize) };
                    self.len -= len;
                    Some(CuMatrixSlice { parent: PhantomData, rows, cols, len, ptr })
                }
                false => None
            }
        }

        /// Skip the next 'len' values
        pub fn skip(&mut self, len: usize) {
            if len <= self.len {
                self.ptr = unsafe { self.ptr.offset(len as isize) };
                self.len -= len;
            } else {
                self.len = 0;
            }
        }

    }

    /// An iterator over mutable GPU data, returning vector and matrix slices.
    pub struct CuDataSliceMutIter<'a> {
        pub(super) parent: PhantomData<&'a CuPackedData>,
        pub(super) len: usize,
        pub(super) ptr: *mut f32,
    }
    impl<'a> CuDataSliceMutIter<'a> {

        #[inline]
        pub fn len(&self) -> usize { self.len }

        /// Returns Some if the iterator's remaining length is more than 'len'
        /// Returns None otherwise
        pub fn next_vector<'b, 'c>(&'c mut self, len: usize) -> Option<CuVectorSliceMut<'b>> where 'a: 'b, 'b: 'c {
            match len <= self.len {
                true => {
                    let ptr = self.ptr;
                    self.ptr = unsafe { self.ptr.offset(len as isize) };
                    self.len -= len;
                    Some(CuVectorSliceMut { _parent: PhantomData, len, ptr })
                }
                false => None
            }
        }

        /// Returns Some if the iterator's remaining length is more than 'rows'*'cols'
        /// Returns None otherwise
        pub fn next_matrix<'b, 'c>(&'c mut self, rows: usize, cols: usize) -> Option<CuMatrixSliceMut<'b>> where 'a: 'b, 'b: 'c {
            let len = rows * cols;
            match len <= self.len {
                true => {
                    let ptr = self.ptr;
                    self.ptr = unsafe { self.ptr.offset(len as isize) };
                    self.len -= len;
                    Some(CuMatrixSliceMut { parent: PhantomData, rows, cols, len, ptr })
                }
                false => None
            }
        }

        /// Skip the next 'len' values
        pub fn skip(&mut self, len: usize) {
            if len <= self.len {
                self.ptr = unsafe { self.ptr.offset(len as isize) };
                self.len -= len;
            } else {
                self.len = 0;
            }
        }

    }

    /// Returns an iterator over GPU data, returning vector and matrix slices.
    pub fn slice_iter<'a> (data: &'a CuPackedData) -> CuDataSliceIter<'a> {
        CuDataSliceIter {
            parent: PhantomData,
            len: data.len(),
            ptr: data.as_ptr(),
        }
    }

    /// Returns an iterator over mutable GPU data, returning vector and matrix slices.
    pub fn slice_mut_iter<'a> (data: &'a mut CuPackedDataMut) -> CuDataSliceMutIter<'a> {
        CuDataSliceMutIter {
            parent: PhantomData,
            len: data.len(),
            ptr: data.as_mut_ptr(),
        }
    }

    /// Returns an immutable matrix slice.
    pub fn matrix_slice<'a>(data: &'a CuPackedData, offset: usize, nb_rows: usize, nb_cols: usize) -> CuMatrixSlice<'a> {
        assert!(offset + nb_rows*nb_cols < data.len());
        CuMatrixSlice {
            parent: PhantomData,
            rows: nb_rows,
            cols: nb_cols,
            len: nb_rows * nb_cols,
            ptr: unsafe { data.as_ptr().offset(offset as isize) },
        }
    }
    /// Returns a mutable matrix slice.
    pub fn matrix_slice_mut<'a>(data: &'a mut CuPackedDataMut, offset: usize, nb_rows: usize, nb_cols: usize) -> CuMatrixSliceMut<'a> {
        assert!(offset + nb_rows*nb_cols < data.len());
        CuMatrixSliceMut {
            parent: PhantomData,
            rows: nb_rows,
            cols: nb_cols,
            len: nb_rows * nb_cols,
            ptr: unsafe { data.as_mut_ptr().offset(offset as isize) },
        }
    }

    /// Returns an immutable vector slice.
    pub fn vector_slice<'a>(data: &'a CuPackedData, offset: usize, len: usize) -> CuVectorSlice<'a> {
        assert!(offset + len < data.len());
        CuVectorSlice {
            _parent: PhantomData,
            len,
            ptr: unsafe { data.as_ptr().offset(offset as isize) },
        }
    }

    /// Returns a mutable vector slice.
    pub fn vector_slice_mut<'a>(data: &'a mut CuPackedDataMut, offset: usize, len: usize) -> CuVectorSliceMut<'a> {
        assert!(offset + len < data.len());
        CuVectorSliceMut {
            _parent: PhantomData,
            len,
            ptr: unsafe { data.as_mut_ptr().offset(offset as isize) },
        }
    }

}
