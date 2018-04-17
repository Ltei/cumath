
use std::{marker::PhantomData, mem::size_of};

use ffi::{cuda_ffi::*, vectorkernel_ffi::*};
use meta::codec::*;

#[cfg(not(feature = "disable_checks"))]
use meta::assert::*;




mod vector;
pub use self::vector::*;
mod vector_slice;
pub use self::vector_slice::*;
mod vector_ptr;
pub use self::vector_ptr::*;
mod vector_slice_iter;
pub use self::vector_slice_iter::*;


/// Immutable vector operator trait.
pub trait CuVectorOp {

    /// [inline]
    /// Returns the vector's length.
    #[inline]
    fn len(&self) -> usize;

    /// [inline]
    /// Returns a pointer on the vector's data.
    #[inline]
    fn as_ptr(&self) -> *const f32;

    /*/// Returns a new GPU-allocated copy of 'self'.
    fn clone(&self) -> CuVector {
        let mut output = {
            let len = self.len();
            let mut data = ptr::null_mut();
            cuda_malloc(&mut data, len*size_of::<f32>());
            CuVector { len, ptr: (data as *mut f32) }
        };
        output.clone_from_device(self);
        output
    }*/

    /// Creates a vector slice starting from self.ptr()+offset, to self.ptr()+offset+len.
    fn slice<'a>(&'a self, offset: usize, len: usize) -> CuVectorSlice<'a> {
        #[cfg(not(feature = "disable_checks"))] {
            assert_infeq_usize(offset+len, "offset+len", self.len(), "self.len");
        }
        CuVectorSlice {
            _parent: PhantomData,
            len,
            ptr: unsafe { self.as_ptr().offset(offset as isize) },
        }
    }

    /// Returns an iterator over a vector, returning vector slices.
    fn slice_iter<'a>(&'a self) -> CuVectorSliceIter<'a> {
        CuVectorSliceIter {
            parent: PhantomData,
            len: self.len(),
            ptr: self.as_ptr(),
        }
    }

    /// Clone this vector's data to host memory.
    fn clone_to_host(&self, data: &mut [f32]) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(self.len(), "self.len()", data.len(), "data.len()");
        }
        cuda_memcpy(data.as_mut_ptr(), self.as_ptr(), data.len()*size_of::<f32>(), CudaMemcpyKind::DeviceToHost);
    }

    #[allow(dead_code)]
    fn dev_print(&self, msg: &str) {
        let mut buffer = vec![0.0; self.len()];
        self.clone_to_host(&mut buffer);
        print!("{}   ", msg);
        for i in 0..self.len() {
            print!("{:.5}, ", buffer[i])
        }
        println!()
    }
    #[allow(dead_code)]
    fn dev_assert_equals(&self, data: &[f32]) where Self: Sized {
        if self.len() != data.len() { panic!(); }
        let mut buffer = vec![0.0; self.len()];
        self.clone_to_host(&mut buffer);
        for i in 0..data.len() {
            let delta = data[i]-buffer[i];
            if delta < -0.00001 || delta > 0.00001 { panic!("At index {} : {:.8} != {:.8}", i, data[i], buffer[i]); }
        }
    }

}

/// Mutable vector operator trait.
pub trait CuVectorOpMut: CuVectorOp {

    /// [inline]
    /// Returns a mutable pointer on the vector's data
    #[inline]
    fn as_mut_ptr(&mut self) -> *mut f32;

    /// Returns a mutable vector slice pointing to this vector's data :
    /// self[offset..offset+len]
    fn slice_mut<'a>(&'a mut self, offset: usize, len: usize) -> CuVectorSliceMut<'a> {
        #[cfg(not(feature = "disable_checks"))] {
            assert_infeq_usize(offset + len, "offset+len", self.len(), "self.len");
        }
        CuVectorSliceMut {
            _parent: PhantomData,
            len,
            ptr: unsafe { self.as_mut_ptr().offset(offset as isize) },
        }
    }

    /// Returns two mutable vector slices pointing to this vector's data :
    /// The first one will be self[offset0..offset0+len0]
    /// The second one will be self[offset0+len0+offset1..offset0+len0+offset1+len1]
    fn slice_mut2<'a>(&'a mut self, offset0: usize, len0: usize, mut offset1: usize, len1: usize) -> (CuVectorSliceMut<'a>, CuVectorSliceMut<'a>) {
        offset1 += offset0+len0;
        #[cfg(not(feature = "disable_checks"))] {
            assert!(offset1 + len1 <= self.len());
        }

        (CuVectorSliceMut {
            _parent: PhantomData, len: len0,
            ptr: unsafe { self.as_mut_ptr().offset(offset0 as isize) },
        },
        CuVectorSliceMut {
            _parent: PhantomData, len: len1,
            ptr: unsafe { self.as_mut_ptr().offset(offset1 as isize) },
        })
    }

    /// Returns an iterator over a mutable vector, returning mutable vector slices.
    fn slice_mut_iter<'a>(&'a mut self) -> CuVectorSliceIterMut<'a> {
        CuVectorSliceIterMut {
            parent: PhantomData,
            len: self.len(),
            ptr: self.as_mut_ptr(),
        }
    }

    unsafe fn force_ownership(&mut self) -> CuVectorPtr {
        CuVectorPtr {
            len: self.len(),
            ptr: self.as_mut_ptr(),
        }
    }

    /// Copy host memory into this vector.
    fn clone_from_host(&mut self, data: &[f32]) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(self.len(), "self.len()", data.len(), "data.len()");
        }
        cuda_memcpy(self.as_mut_ptr(), data.as_ptr(), data.len()*size_of::<f32>(), CudaMemcpyKind::HostToDevice);
    }

    /// Clone device memory into this vector.
    fn clone_from_device(&mut self, source: &CuVectorOp) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(self.len(), "self.len()", source.len(), "data.len()");
        }
        cuda_memcpy(self.as_mut_ptr(), source.as_ptr(), self.len()*size_of::<f32>(), CudaMemcpyKind::DeviceToDevice);
    }

    /// Initializes the vector with value.
    fn init(&mut self, value: f32) {
        unsafe { VectorKernel_init(self.as_mut_ptr(), self.len() as i32, value) }
    }

    /// Add value to each element of the vector.
    fn add_value_self(&mut self, value: f32) {
        unsafe { VectorKernel_addValue(self.as_ptr(), self.as_mut_ptr(), self.len() as i32, value) }
    }

    /// Scale each element of the vector by value.
    fn scale_self(&mut self, value: f32) {
        unsafe { VectorKernel_scl(self.as_ptr(), self.as_mut_ptr(), self.len() as i32, value) }
    }

    /// Add an other vector to this one.
    fn add_self(&mut self, right_op: &CuVectorOp) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(self.len(), "self.len()", right_op.len(), "right_op.len()");
        }
        unsafe { VectorKernel_add(self.as_ptr(), right_op.as_ptr(), self.as_mut_ptr(), self.len() as i32) }
    }

    /// Substract an other vector to this one.
    fn sub_self(&mut self, right_op: &CuVectorOp) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(self.len(), "self.len()", right_op.len(), "right_op.len()");
        }
        unsafe { VectorKernel_sub(self.as_ptr(), right_op.as_ptr(), self.as_mut_ptr(), self.len() as i32) }
    }

    /// Multiply each element of the vector by the corresponding element in the parameter vector.
    fn pmult_self(&mut self, right_op: &CuVectorOp) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(self.len(), "self.len()", right_op.len(), "right_op.len()");
        }
        unsafe { VectorKernel_pmult(self.as_ptr(), right_op.as_ptr(), self.as_mut_ptr(), self.len() as i32) }
    }

    /// Multiply each element of the vector by itself.
    fn psquare_self(&mut self) {
        unsafe { VectorKernel_psquare(self.as_ptr(),self.as_mut_ptr(), self.len() as i32) }
    }

    /// Apply the sigmoid function to each element of the vector.
    fn sigmoid_self(&mut self) {
        unsafe { VectorKernel_sigmoid(self.as_ptr(), self.as_mut_ptr(), self.len() as i32) }
    }

}

impl<T: CuVectorOp> Codec for T {
    type OutputType = CuVector;

    fn encode(&self) -> String {
        let mut host_data = vec![0.0; self.len()];
        self.clone_to_host(host_data.as_mut_slice());

        host_data.iter().map(|x| {
            format!("{} ", x)
        }).collect::<String>()
    }
    fn decode(data: &str) -> CuVector {
        CuVector::from_data(
            data.split_whitespace().map(|x| {
                x.parse::<f32>().unwrap_or_else(|err| { panic!("{}", err) })
            }).collect::<Vec<f32>>().as_slice()
        )
    }
}




// TESTS

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn codec() {
        let data = [1.2, -2.2656146, 7.12, 2.0, 4.5];
        CuVector::decode(&CuVector::from_data(&data).encode()).dev_assert_equals(&data);
    }

    #[test]
    fn test() {
        let mut vector = CuVector::new(2, 0.0);
        vector.add_value_self(-1.2);
        vector.dev_print("Hello");
    }

}