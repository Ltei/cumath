

use std::{marker::PhantomData, mem::size_of};
use assert::*;
use ffi::cuda_ffi::*;

use ffi::matrixkernel_ffi::*;



mod matrix;
pub use self::matrix::*;
mod matrix_slice;
pub use self::matrix_slice::*;
mod sub_matrix;
pub use self::sub_matrix::*;




pub trait CuMatrixOp {

    #[inline]
    fn rows(&self) -> usize;
    #[inline]
    fn cols(&self) -> usize;
    #[inline]
    fn len(&self) -> usize;
    #[inline]
    fn leading_dimension(&self) -> usize;
    #[inline]
    fn ptr(&self) -> *const f32;

    fn slice<'a>(&'a self, row_offset: usize, col_offset: usize, nb_rows: usize, nb_cols: usize) -> CuSubMatrix<'a, Self> where Self: Sized {
        CuSubMatrix {
            parent: PhantomData,
            rows: nb_rows,
            cols: nb_cols,
            leading_dimension: self.leading_dimension(),
            ptr: unsafe { self.ptr().offset((row_offset + col_offset*self.leading_dimension()) as isize) },
        }
    }

    fn clone_to_host(&self, data: &mut [f32]) {
        assert_eq_usize(self.len(), "self.len()", data.len(), "data.len()");
        cuda_memcpy2d(data.as_mut_ptr(),
                     self.rows() * size_of::<f32>(),
                     self.ptr(),
                     self.leading_dimension() * size_of::<f32>(),
                     self.rows() * size_of::<f32>(),
                     self.cols(),
                     CudaMemcpyKind::DeviceToHost);
    }

    #[allow(dead_code)]
    fn dev_print(&self, msg: &str) {
        let mut buffer = vec![0.0; self.len()];
        self.clone_to_host(&mut buffer);
        println!("{}", msg);
        for row in 0..self.rows() {
            for col in 0..self.cols() {
                print!("{:.5}, ", buffer[row+col*self.rows()])
            }
            println!()
        }
    }
    fn assert_equals_float(a: f32, b: f32) where Self: Sized {
        let d = a-b;
        if d < -0.000001 || d > 0.000001 {
            panic!("{} != {}", a, b);
        }
    }
    #[allow(dead_code)]
    fn dev_equals(&self, data: &[f32]) where Self: Sized {
        assert_eq_usize(self.len(), "self.len()", data.len(), "data.len()");
        let mut buffer = vec![0.0; self.len()];
        self.clone_to_host(&mut buffer);
        let mut iter = buffer.iter();
        data.iter().for_each(|x| { Self::assert_equals_float(*x, *iter.next().unwrap()) });
    }

}


pub trait CuMatrixOpMut: CuMatrixOp  {

    #[inline]
    fn ptr_mut(&mut self) -> *mut f32;

    fn slice_mut<'a>(&'a mut self, row_offset: usize, col_offset: usize, nb_rows: usize, nb_cols: usize) -> CuSubMatrixMut<'a, Self> where Self: Sized {
        CuSubMatrixMut {
            parent: PhantomData,
            rows: nb_rows,
            cols: nb_cols,
            leading_dimension: self.leading_dimension(),
            ptr: unsafe { self.ptr_mut().offset((row_offset + col_offset*self.leading_dimension()) as isize) },
        }
    }

    fn clone_from_host(&mut self, data: &[f32]) {
        assert_eq_usize(self.len(), "self.len()", data.len(), "data.len()");
        cuda_memcpy2d(self.ptr_mut(),
                      self.leading_dimension() * size_of::<f32>(),
                      data.as_ptr(),
                      self.rows() * size_of::<f32>(),
                      self.rows() * size_of::<f32>(),
                      self.cols(),
                      CudaMemcpyKind::HostToDevice);
    }
    fn clone_from_device(&mut self, data: &CuMatrixOp) {
        assert_eq_usize(self.len(), "self.len()", data.len(), "data.len()");
        cuda_memcpy2d(self.ptr_mut(),
                      self.leading_dimension() * size_of::<f32>(),
                      data.ptr(),
                      data.leading_dimension() * size_of::<f32>(),
                      self.rows() * size_of::<f32>(),
                      self.cols(),
                      CudaMemcpyKind::DeviceToDevice);
    }

    fn init(&mut self, value: f32) {
        unsafe {
            MatrixKernel_init(self.ptr_mut(), self.leading_dimension() as i32,
                              self.rows() as i32, self.cols() as i32, value);
        }
    }

    fn add_value_self(&mut self, value: f32) {
        unsafe {
            MatrixKernel_addValue(self.ptr(), self.leading_dimension() as i32,
                                  self.ptr_mut(), self.leading_dimension() as i32,
                                  self.rows() as i32, self.cols() as i32, value);
        }
    }
    fn scale_self(&mut self, value: f32) {
        unsafe {
            MatrixKernel_scale(self.ptr(), self.leading_dimension() as i32,
                               self.ptr_mut(), self.leading_dimension() as i32,
                               self.rows() as i32, self.cols() as i32, value);
        }
    }
    fn add_self(&mut self, to_add: &CuMatrixOp) {
        unsafe {
            MatrixKernel_add(self.ptr(), self.leading_dimension() as i32,
                             to_add.ptr(), to_add.leading_dimension() as i32,
                             self.ptr_mut(), self.leading_dimension() as i32,
                             self.rows() as i32, self.cols() as i32)
        }
    }

}