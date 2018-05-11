

use super::*;
use std::ptr;


pub struct CuMatrixView<T: CuDataType> {
    pub(crate) offset: usize,
    pub(crate) deref: CuMatrixDeref<T>,
}

impl<T: CuDataType> CuMatrixView<T> {

    pub fn new(offset: usize, rows: usize, cols: usize, leading_dimension: usize) -> CuMatrixView<T> {
        CuMatrixView {
            offset,
            deref: CuMatrixDeref {
                ptr: ptr::null_mut(),
                len: rows*cols,
                rows, cols, leading_dimension,
            }
        }
    }

    /// [inline]
    /// Returns the vector's length.
    #[inline]
    pub fn len(&self) -> usize { self.deref.len }

    pub fn borrow(&mut self, vector: &::CuVectorDeref<T>) -> &CuMatrixDeref<T> {
        assert_infeq_usize(self.offset+self.deref.leading_dimension*self.deref.cols,
                           "self.offset+self.deref.leading_dimension*self.deref.cols",
                           vector.len(), "vector.len()");
        self.deref.ptr = unsafe { vector.ptr.offset(self.offset as isize) };
        &self.deref
    }

    pub fn borrow_mut(&mut self, vector: &mut ::CuVectorDeref<T>) -> &mut CuMatrixDeref<T> {
        assert_infeq_usize(self.offset+self.deref.leading_dimension*self.deref.cols,
                           "self.offset+self.deref.leading_dimension*self.deref.cols",
                           vector.len(), "vector.len()");
        self.deref.ptr = unsafe { vector.ptr.offset(self.offset as isize) };
        &mut self.deref
    }

}


#[cfg(test)]
mod tests {

    use cumatrix::*;

    #[test]
    fn borrow() {
        let vector = ::CuVector::from_host_data((0..15).collect::<Vec<_>>().as_slice());
        let mut view = CuMatrixView::new(3, 3, 3, 3);
        view.borrow(&vector).dev_assert_equals(&[3, 4, 5, 6, 7, 8, 9, 10, 11])
    }

}