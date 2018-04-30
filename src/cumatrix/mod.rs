
#[macro_use]
mod macros;
mod ffi;
use self::ffi::*;

use std::{self, marker::PhantomData, mem::size_of, fmt};
use cuda_core::{cuda::*, cuda_ffi::*};
#[cfg(not(feature = "disable_checks"))]
use meta::assert::*;



mod matrix;
pub use self::matrix::*;
mod slice;
pub use self::slice::*;
mod fragment;
pub use self::fragment::*;
mod ptr;
pub use self::ptr::*;
mod math;
pub use self::math::*;


/// Immutable matrix operator trait.
pub trait CuMatrixOp: fmt::Debug {

    /// [inline]
    /// Returns the number of rows in the matrix.
    #[inline]
    fn rows(&self) -> usize;

    /// [inline]
    /// Returns the number of columns in the matrix.
    #[inline]
    fn cols(&self) -> usize;

    /// [inline]
    /// Returns the number of elements in the matrix.
    #[inline]
    fn len(&self) -> usize;

    /// [inline]
    /// Returns the leading dimension of the matrix.
    #[inline]
    fn leading_dimension(&self) -> usize;

    /// [inline]
    /// Returns a pointer on the matrix's data.
    #[inline]
    fn as_ptr(&self) -> *const f32;

    /// Returns a new copy of 'self'.
    fn clone(&self) -> CuMatrix where Self: Sized {
        let mut output = {
            let len = self.len();
            let mut data = std::ptr::null_mut();
            cuda_malloc(&mut data, len*size_of::<f32>());
            CuMatrix { rows: self.rows(), cols: self.cols(), len, ptr: (data as *mut f32) }
        };
        output.clone_from_device(self);
        output
    }

    /// Returns an immutable sub-matrix.
    fn slice<'a>(&'a self, row_offset: usize, col_offset: usize, nb_rows: usize, nb_cols: usize) -> CuMatrixFragment<'a> {
        #[cfg(not(feature = "disable_checks"))] {
            assert_infeq_usize(row_offset + nb_rows, "row_offset+nb_rows", self.rows(), "self.rows()");
            assert_infeq_usize(col_offset + nb_cols, "col_offset+nb_cols", self.cols(), "self.cols()");
        }
        CuMatrixFragment {
            _parent: PhantomData,
            rows: nb_rows,
            cols: nb_cols,
            leading_dimension: self.leading_dimension(),
            ptr: unsafe { self.as_ptr().offset((row_offset + col_offset*self.leading_dimension()) as isize) },
        }
    }

    /// Clone this matrix's data to host memory.
    fn clone_to_host(&self, data: &mut [f32]) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(self.len(), "self.len()", data.len(), "data.len()");
        }
        cuda_memcpy2d(data.as_mut_ptr(),
                      self.rows() * size_of::<f32>(),
                      self.as_ptr(),
                      self.leading_dimension() * size_of::<f32>(),
                      self.rows() * size_of::<f32>(),
                      self.cols(),
                      cudaMemcpyKind::DeviceToHost);
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

/// Mutable matrix operator trait.
pub trait CuMatrixOpMut: CuMatrixOp  {

    /// [inline]
    /// Returns a mutable pointer on the matrix's data
    #[inline]
    fn as_mut_ptr(&mut self) -> *mut f32;

    /// [inline]
    /// Supertrait upcasting
    #[inline]
    fn as_immutable(&self) -> &CuMatrixOp;

    /// Returns a mutable sub-matrix.
    fn slice_mut<'a>(&'a mut self, row_offset: usize, col_offset: usize, nb_rows: usize, nb_cols: usize) -> CuMatrixFragmentMut<'a> {
        #[cfg(not(feature = "disable_checks"))] {
            assert_infeq_usize(row_offset + nb_rows, "row_offset+nb_rows", self.rows(), "self.rows()");
            assert_infeq_usize(col_offset + nb_cols, "col_offset+nb_cols", self.cols(), "self.cols()");
        }
        CuMatrixFragmentMut {
            parent: PhantomData,
            rows: nb_rows,
            cols: nb_cols,
            leading_dimension: self.leading_dimension(),
            ptr: unsafe { self.as_mut_ptr().offset((row_offset + col_offset*self.leading_dimension()) as isize) },
        }
    }

    fn force_ownership(&mut self) -> CuMatrixPtr {
        CuMatrixPtr {
            deref: CuMatrixPtrDeref {
                rows: self.rows(),
                cols: self.cols(),
                len: self.len(),
                ptr: self.as_mut_ptr(),
            }
        }
    }

    /// Clone host memory to this matrix's data.
    fn clone_from_host(&mut self, data: &[f32]) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(self.len(), "self.len()", data.len(), "data.len()");
        }
        cuda_memcpy2d(self.as_mut_ptr(),
                      self.leading_dimension() * size_of::<f32>(),
                      data.as_ptr(),
                      self.rows() * size_of::<f32>(),
                      self.rows() * size_of::<f32>(),
                      self.cols(),
                      cudaMemcpyKind::HostToDevice);
    }

    /// Clone a matrix's data to this matrix's data.
    fn clone_from_device(&mut self, data: &CuMatrixOp) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(self.len(), "self.len()", data.len(), "data.len()");
        }
        cuda_memcpy2d(self.as_mut_ptr(),
                      self.leading_dimension() * size_of::<f32>(),
                      data.as_ptr(),
                      data.leading_dimension() * size_of::<f32>(),
                      self.rows() * size_of::<f32>(),
                      self.cols(),
                      cudaMemcpyKind::DeviceToDevice);
    }

    /// Initializes the matrix with 'value'.
    fn init(&mut self, value: f32, stream: &CudaStream) {
        unsafe {
            MatrixKernel_init(self.as_mut_ptr(), self.leading_dimension() as i32,
                              self.rows() as i32, self.cols() as i32, value, stream.stream);
        }
    }

    /// Add value to each matrix of the vector.
    fn add_value(&mut self, value: f32, stream: &CudaStream) {
        unsafe {
            MatrixKernel_addValue(self.as_ptr(), self.leading_dimension() as i32,
                                  self.as_mut_ptr(), self.leading_dimension() as i32,
                                  self.rows() as i32, self.cols() as i32, value, stream.stream);
        }
    }

    /// Scale each element of the matrix by 'value'.
    fn scale(&mut self, value: f32, stream: &CudaStream) {
        unsafe {
            MatrixKernel_scale(self.as_ptr(), self.leading_dimension() as i32,
                               self.as_mut_ptr(), self.leading_dimension() as i32,
                               self.rows() as i32, self.cols() as i32, value, stream.stream);
        }
    }

    /// Add an other matrix to this one.
    fn add(&mut self, to_add: &CuMatrixOp, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(self.rows(), "self.rows()", to_add.rows(), "to_add.rows()");
            assert_eq_usize(self.cols(), "self.cols()", to_add.cols(), "to_add.cols()");
        }
        unsafe {
            MatrixKernel_add(self.as_ptr(), self.leading_dimension() as i32,
                             to_add.as_ptr(), to_add.leading_dimension() as i32,
                             self.as_mut_ptr(), self.leading_dimension() as i32,
                             self.rows() as i32, self.cols() as i32, stream.stream)
        }
    }

}




#[cfg(test)]
mod tests {

    use super::*;
    use meta::codec::Codec;

    #[test]
    fn codec() {
        let data = [1.2, -2.2656146, 7.12, 2.0, 4.5, 7.256];
        CuMatrix::decode(&CuMatrix::from_data(2, 3, &data).encode()).dev_assert_equals(&data);
    }

}