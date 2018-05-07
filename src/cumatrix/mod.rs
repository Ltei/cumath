
#[cfg(not(feature = "disable_checks"))]
use meta::assert::*;

use std::{marker::PhantomData, mem::size_of, fmt, os::raw::c_void};
use cuda_core::{cuda::*, cuda_ffi::*};
use CuDataType;



mod owned;
pub use self::owned::*;
mod slice;
pub use self::slice::*;
mod fragment;
pub use self::fragment::*;
mod ptr;
pub use self::ptr::*;
mod math;
pub use self::math::*;



/// Immutable matrix operator trait.
pub trait CuMatrixOp<T: CuDataType>: fmt::Debug {

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

    fn as_vector(&self) -> ::CuVectorSlice<T> {
        ::CuVectorSlice {
            _parent: PhantomData,
            ptr: self.as_ptr(),
            len: self.len(),
        }
    }

    /// Returns an immutable sub-matrix.
    fn slice<'a>(&'a self, row_offset: usize, col_offset: usize, nb_rows: usize, nb_cols: usize) -> CuMatrixFragment<'a, T> {
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
    fn clone_to_host(&self, data: &mut [T]) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(self.len(), "self.len()", data.len(), "data.len()");
        }
        cuda_memcpy2d(data.as_mut_ptr() as *mut c_void,
                      self.rows() * size_of::<T>(),
                      self.as_ptr() as *const c_void,
                      self.leading_dimension() * size_of::<T>(),
                      self.rows() * size_of::<T>(),
                      self.cols(),
                      cudaMemcpyKind::DeviceToHost);
    }

    /// [inline]
    /// Returns a pointer on the matrix's data.
    #[inline]
    fn as_ptr(&self) -> *const T;

    #[allow(dead_code)]
    fn dev_assert_equals(&self, data: &[T]) where Self: Sized {
        if self.len() != data.len() { panic!(); }
        let mut buffer = vec![T::zero(); self.len()];
        self.clone_to_host(buffer.as_mut_slice());
        for i in 0..data.len() {
            if data[i] != buffer[i] { panic!("At index {} : {:.8} != {:.8}", i, data[i], buffer[i]); }
        }
    }

}

/// Mutable matrix operator trait.
pub trait CuMatrixOpMut<T: CuDataType>: CuMatrixOp<T>  {

    /// [inline]
    /// Returns a mutable pointer on the matrix's data
    #[inline]
    fn as_mut_ptr(&mut self) -> *mut T;

    /// [inline]
    /// Supertrait upcasting
    #[inline]
    fn as_immutable(&self) -> &CuMatrixOp<T>;

    fn as_mut_vector(&mut self) -> ::CuVectorSliceMut<T> {
        ::CuVectorSliceMut {
            _parent: PhantomData,
            ptr: self.as_mut_ptr(),
            len: self.len(),
        }
    }

    /// Returns a mutable sub-matrix.
    fn slice_mut<'a>(&'a mut self, row_offset: usize, col_offset: usize, nb_rows: usize, nb_cols: usize) -> CuMatrixFragmentMut<'a, T> {
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

    /// Returns a pointer over a matrix :
    /// - It won't free the inner GPU-pointer when it goes out of scope
    /// - It won't check if the underlying memory is still allocated when used
    /// -> Use at your own risk
    fn as_wrapped_ptr(&mut self) -> CuMatrixPtr<T> {
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
    fn clone_from_host(&mut self, data: &[T]) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(self.len(), "self.len()", data.len(), "data.len()");
        }
        cuda_memcpy2d(self.as_mut_ptr() as *mut c_void,
                      self.leading_dimension() * size_of::<T>(),
                      data.as_ptr() as *const c_void,
                      self.rows() * size_of::<T>(),
                      self.rows() * size_of::<T>(),
                      self.cols(),
                      cudaMemcpyKind::HostToDevice);
    }

    /// Clone a matrix's data to this matrix's data.
    fn clone_from_device(&mut self, data: &CuMatrixOp<T>) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(self.len(), "self.len()", data.len(), "data.len()");
        }
        cuda_memcpy2d(self.as_mut_ptr() as *mut c_void,
                      self.leading_dimension() * size_of::<T>(),
                      data.as_ptr() as *const c_void,
                      data.leading_dimension() * size_of::<T>(),
                      self.rows() * size_of::<T>(),
                      self.cols(),
                      cudaMemcpyKind::DeviceToDevice);
    }

    /// Initializes the matrix with 'value'.
    fn init(&mut self, value: T, stream: &CudaStream);

    /// Add value to each matrix of the vector.
    fn add_value(&mut self, value: T, stream: &CudaStream);

    /// Scale each element of the matrix by 'value'.
    fn scl(&mut self, value: T, stream: &CudaStream);

    /// Add an other matrix to this one.
    fn add(&mut self, to_add: &CuMatrixOp<T>, stream: &CudaStream);

}