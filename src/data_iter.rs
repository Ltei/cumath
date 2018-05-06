
use super::*;
use std::marker::PhantomData;



pub struct CuDataIter<'a, T: CuDataType + 'a> {
    pub(crate) _parent: PhantomData<&'a CuVectorOp<T>>,
    pub(crate) len: usize,
    pub(crate) ptr: *const T,
}
impl<'a, T: CuDataType + 'a> CuDataIter<'a, T> {
    pub fn new(data: &CuVectorOp<T>) -> CuDataIter<'a, T> {
        CuDataIter {
            _parent: PhantomData,
            len: data.len(),
            ptr: data.as_ptr(),
        }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn skip(&mut self, len: usize) {
        if len > self.len { panic!() }
        self.len -= len;
        self.ptr = unsafe { self.ptr.offset(len as isize) }
    }
    pub fn next_vector<'b, 'c>(&'c mut self, len: usize) -> Option<CuVectorSlice<'b, T>> where 'a: 'b, 'b: 'c {
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
    pub fn next_matrix<'b, 'c>(&'c mut self, rows: usize, cols: usize) -> Option<CuMatrixSlice<'b, T>> where 'a: 'b, 'b: 'c {
        let len = rows * cols;
        match len <= self.len {
            true => {
                let ptr = self.ptr;
                self.ptr = unsafe { self.ptr.offset(len as isize) };
                self.len -= len;
                Some(CuMatrixSlice { _parent: PhantomData, ptr, len, rows, cols })
            }
            false => None
        }
    }
}

pub struct CuDataMutIter<'a, T: CuDataType + 'a> {
    pub(crate) _parent: PhantomData<&'a CuVectorOpMut<T>>,
    pub(crate) len: usize,
    pub(crate) ptr: *mut T,
}
impl<'a, T: CuDataType + 'a> CuDataMutIter<'a, T> {
    pub fn new(data: &mut CuVectorOpMut<T>) -> CuDataMutIter<'a, T> {
        CuDataMutIter {
            _parent: PhantomData,
            len: data.len(),
            ptr: data.as_mut_ptr(),
        }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn skip(&mut self, len: usize) {
        if len > self.len { panic!() }
        self.len -= len;
        self.ptr = unsafe { self.ptr.offset(len as isize) }
    }
    pub fn next_vector<'b, 'c>(&'c mut self, len: usize) -> Option<CuVectorSliceMut<'b, T>> where 'a: 'b, 'b: 'c {
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
    pub fn next_matrix<'b, 'c>(&'c mut self, rows: usize, cols: usize) -> Option<CuMatrixSliceMut<'b, T>> where 'a: 'b, 'b: 'c {
        let len = rows * cols;
        match len <= self.len {
            true => {
                let ptr = self.ptr;
                self.ptr = unsafe { self.ptr.offset(len as isize) };
                self.len -= len;
                Some(CuMatrixSliceMut { _parent: PhantomData, ptr, len, rows, cols })
            }
            false => None
        }
    }
}