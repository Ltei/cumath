
use super::*;
use std::{marker::PhantomData};


pub(crate) struct _Dummy {}
pub struct CuDataIter<'a> {
    _marker: PhantomData<&'a _Dummy>,
    len: usize,
    ptr: *const f32,
}
impl<'a> CuDataIter<'a> {

    pub fn next_vector<'b, 'c>(&'c mut self, len: usize) -> Option<CuVectorSlice<'b>> where 'a: 'b, 'b: 'c {
        match len <= self.len {
            true => {
                let ptr = self.ptr;
                self.ptr = unsafe { self.ptr.offset(len as isize) };
                self.len -= len;
                Some(CuVectorSlice { parent: PhantomData, len, ptr })
            }
            false => None
        }
    }

}
pub struct CuDataIterMut<'a> {
    _marker: PhantomData<&'a _Dummy>,
    len: usize,
    ptr: *mut f32,
}
impl<'a> CuDataIterMut<'a> {

    pub fn next_vector<'b, 'c>(&'c mut self, len: usize) -> Option<CuVectorSliceMut<'b>> where 'a: 'b, 'b: 'c {
        match len <= self.len {
            true => {
                let ptr = self.ptr;
                self.ptr = unsafe { self.ptr.offset(len as isize) };
                self.len -= len;
                Some(CuVectorSliceMut { parent: PhantomData, len, ptr })
            }
            false => None
        }
    }

}
pub struct CuData {}
impl CuData {

    pub fn vector_to_matrix(mut vector: CuVector, rows: usize, cols: usize) -> CuMatrix {
        assert_eq!(rows*cols, vector.len());
        CuMatrix { rows, cols, len: vector.len(), ptr: vector.as_mut_ptr() }
    }
    pub fn matrix_to_vector(mut matrix: CuMatrix) -> CuVector {
        CuVector { len: matrix.len(), ptr: matrix.ptr_mut() }
    }

    pub fn vector_iter<'a>(vector: &'a CuVectorOp) -> CuDataIter<'a> {
        CuDataIter {
            _marker: PhantomData,
            len: vector.len(),
            ptr: vector.as_ptr(),
        }
    }
    pub fn vector_iter_mut<'a>(vector: &'a mut CuVectorOpMut) -> CuDataIterMut<'a> {
        CuDataIterMut {
            _marker: PhantomData,
            len: vector.len(),
            ptr: vector.as_mut_ptr(),
        }
    }

}