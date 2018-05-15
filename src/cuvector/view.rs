

use super::*;
use std::ptr;



pub struct CuVectorView<T: CuDataType> {
    pub(crate) offset: usize,
    pub(crate) deref: CuVectorDeref<T>,
}

impl<T: CuDataType> CuVectorView<T> {

    pub fn new(offset: usize, len: usize) -> CuVectorView<T> {
        CuVectorView {
            offset,
            deref: CuVectorDeref {
                ptr: ptr::null_mut(),
                len
            }
        }
    }

    /// [inline]
    /// Returns the vector's length.
    #[inline]
    pub fn len(&self) -> usize { self.deref.len }

    pub fn borrow(&mut self, vector: &CuVectorDeref<T>) -> &CuVectorDeref<T> {
        #[cfg(not(feature = "disable_checks"))] {
            assert_infeq_usize(self.offset+self.deref.len, "self.offset+self.deref.len", vector.len(), "vector.len()");
        }
        self.deref.ptr = unsafe { vector.ptr.offset(self.offset as isize) };
        &self.deref
    }

    pub fn borrow_mut(&mut self, vector: &mut CuVectorDeref<T>) -> &mut CuVectorDeref<T> {
        #[cfg(not(feature = "disable_checks"))] {
            assert_infeq_usize(self.offset + self.deref.len, "self.offset+self.deref.len", vector.len(), "vector.len()");
        }
        self.deref.ptr = unsafe { vector.ptr.offset(self.offset as isize) };
        &mut self.deref
    }

}