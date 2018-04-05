
use std::{marker::PhantomData, mem::size_of};

use ffi::{cuda_ffi::*, vectorkernel_ffi::*};
use assert::*;

mod vector;
pub use self::vector::*;
mod vector_slice;
pub use self::vector_slice::*;



pub trait CuVectorOp {

    #[inline]
    fn len(&self) -> usize;
    #[inline]
    fn ptr(&self) -> *const f32;

    fn slice<'a>(&'a self, offset: usize, len: usize) -> CuVectorSlice<'a, Self> where Self: Sized {
        CuVectorSlice {
            parent: PhantomData,
            len,
            ptr: unsafe { self.ptr().offset(offset as isize) },
        }
    }

    fn clone_to_host(&self, data: &mut [f32]) {
        assert_eq_usize(self.len(), "self.len()", data.len(), "data.len()");
        cuda_memcpy(data.as_mut_ptr(), self.ptr(), data.len()*size_of::<f32>(), CudaMemcpyKind::DeviceToHost);
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

}

pub trait CuVectorOpMut: CuVectorOp {

    #[inline]
    fn ptr_mut(&mut self) -> *mut f32;

    fn slice_mut<'a>(&'a mut self, offset: usize, len: usize) -> CuVectorSliceMut<'a, Self> where Self: Sized {
        CuVectorSliceMut {
            parent: PhantomData,
            len,
            ptr: unsafe { self.ptr_mut().offset(offset as isize) },
        }
    }


    fn clone_from_host(&mut self, data: &[f32]) {
        assert_eq_usize(self.len(), "self.len()", data.len(), "data.len()");
        cuda_memcpy(self.ptr_mut(), data.as_ptr(), data.len()*size_of::<f32>(), CudaMemcpyKind::HostToDevice);
    }
    fn clone_from_device(&mut self, source: &CuVectorOp) {
        assert_eq_usize(self.len(), "self.len()", source.len(), "data.len()");
        cuda_memcpy(self.ptr_mut(), source.ptr(), self.len()*size_of::<f32>(), CudaMemcpyKind::DeviceToDevice);
    }


    fn init(&mut self, value: f32) {
        unsafe { VectorKernel_init(self.ptr_mut(), self.len() as i32, value) }
    }

    fn add_value_self(&mut self, value: f32) {
        unsafe { VectorKernel_addValue(self.ptr(), self.ptr_mut(), self.len() as i32, value) }
    }
    fn scale_self(&mut self, value: f32) {
        unsafe { VectorKernel_scl(self.ptr(), self.ptr_mut(), self.len() as i32, value) }
    }
    fn add_self(&mut self, right_op: &CuVectorOp) {
        assert_eq_usize(self.len(), "self.len()", right_op.len(), "right_op.len()");
        unsafe { VectorKernel_add(self.ptr(), right_op.ptr(), self.ptr_mut(), self.len() as i32) }
    }
    fn pmult_self(&mut self, right_op: &CuVectorOp) {
        assert_eq_usize(self.len(), "self.len()", right_op.len(), "right_op.len()");
        unsafe { VectorKernel_pmult(self.ptr(), right_op.ptr(), self.ptr_mut(), self.len() as i32) }
    }
    fn sigmoid_self(&mut self) {
        unsafe { VectorKernel_sigmoid(self.ptr(), self.ptr_mut(), self.len() as i32) }
    }

}