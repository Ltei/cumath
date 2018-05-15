
use super::*;


/// An iterator over a vector, returning vector slices.
pub struct CuVectorSliceIter<'a, T: CuDataType + 'a> {
    pub(crate) _parent: PhantomData<&'a CuVectorDeref<T>>,
    pub(crate) len: usize,
    pub(crate) ptr: *mut T,
}
impl<'a, T: CuDataType + 'a> CuVectorSliceIter<'a, T> {

    #[inline]
    pub fn len(&self) -> usize { self.len as usize }

    pub fn next<'b, 'c>(&'c mut self, len: usize) -> Option<CuVectorSlice<'b, T>> where 'a: 'b, 'b: 'c {
        match len <= self.len {
            true => {
                let ptr = self.ptr;
                self.ptr = unsafe { self.ptr.offset(len as isize) };
                self.len -= len;
                Some(CuVectorSlice {
                    _parent: PhantomData,
                    deref: CuVectorDeref {
                        len,
                        ptr
                    }
                })
            }
            false => None
        }
    }
    pub fn last<'b, 'c>(&'c mut self, len: usize) -> Option<CuVectorSlice<'b, T>> where 'a: 'b, 'b: 'c {
        match len <= self.len {
            true => {
                self.len -= len;
                Some(CuVectorSlice {
                    _parent: PhantomData,
                    deref: CuVectorDeref {
                        len,
                        ptr: unsafe { self.ptr.offset(self.len as isize) },
                    }
                })
            }
            false => None
        }
    }
    pub fn next_matrix<'b, 'c>(&'c mut self, rows: usize, cols: usize) -> Option<::CuMatrixSlice<'b, T>> where 'a: 'b, 'b: 'c {
        let len = rows * cols;
        match len <= self.len {
            true => {
                let ptr = self.ptr;
                self.ptr = unsafe { self.ptr.offset(len as isize) };
                self.len -= len;
                Some(::CuMatrixSlice {
                    _parent: PhantomData,
                    deref: ::CuMatrixDeref {
                        ptr,
                        len,
                        rows,
                        cols,
                        leading_dimension: rows,
                    }
                })
            }
            false => None
        }
    }
    pub fn skip(&mut self, len: usize) {
        if len <= self.len {
            self.ptr = unsafe { self.ptr.offset(len as isize) };
            self.len -= len;
        } else {
            self.len = 0;
        }
    }

}




/// An iterator over a mutable vector, returning mutable vector slices.
pub struct CuVectorSliceIterMut<'a, T: CuDataType + 'a> {
    pub(crate) _parent: PhantomData<&'a CuVectorDeref<T>>,
    pub(crate) len: usize,
    pub(crate) ptr: *mut T,
}
impl<'a, T: CuDataType + 'a> CuVectorSliceIterMut<'a, T> {

    #[inline]
    pub fn len(&self) -> usize { self.len as usize }

    pub fn next<'b, 'c>(&'c mut self, len: usize) -> Option<CuVectorSliceMut<'b, T>> where 'a: 'b, 'b: 'c {
        match len <= self.len {
            true => {
                let ptr = self.ptr;
                self.ptr = unsafe { self.ptr.offset(len as isize) };
                self.len -= len;
                Some(CuVectorSliceMut {
                    _parent: PhantomData,
                    deref: CuVectorDeref {
                        len,
                        ptr,
                    }
                })
            }
            false => None
        }
    }
    pub fn last<'b, 'c>(&'c mut self, len: usize) -> Option<CuVectorSliceMut<'b, T>> where 'a: 'b, 'b: 'c {
        match len <= self.len {
            true => {
                self.len -= len;
                Some(CuVectorSliceMut {
                    _parent: PhantomData,
                    deref: CuVectorDeref {
                        len,
                        ptr: unsafe { self.ptr.offset(self.len as isize) },
                    }
                })
            }
            false => None
        }
    }
    pub fn next_matrix<'b, 'c>(&'c mut self, rows: usize, cols: usize) -> Option<::CuMatrixSliceMut<'b, T>> where 'a: 'b, 'b: 'c {
        let len = rows * cols;
        match len <= self.len {
            true => {
                let ptr = self.ptr;
                self.ptr = unsafe { self.ptr.offset(len as isize) };
                self.len -= len;
                Some(::CuMatrixSliceMut {
                    _parent: PhantomData,
                    deref: ::CuMatrixDeref {
                        ptr,
                        len,
                        rows,
                        cols,
                        leading_dimension: rows,
                    }
                })
            }
            false => None
        }
    }
    pub fn skip(&mut self, len: usize) {
        if len <= self.len {
            self.ptr = unsafe { self.ptr.offset(len as isize) };
            self.len -= len;
        } else {
            self.len = 0;
        }
    }

}




#[cfg(test)]
mod tests {

    use cuvector::*;

    #[test]
    fn last() {

        let vector = CuVector::<i32>::from_host_data(&[1, -5, 1, 340, 2]);
        let mut iter = vector.slice_iter();

        iter.last(3).unwrap().dev_assert_equals(&[1, 340, 2]);

    }

}