
#[cfg(not(feature = "disable_checks"))]
use meta::assert::*;

use std::{fmt::{self, Debug}, marker::PhantomData, mem::size_of, os::raw::c_void};
use cuda_core::{cuda::*, cuda_ffi::*};
use kernel::*;
use CuDataType;



mod owned;
pub use self::owned::*;
mod slice;
pub use self::slice::*;
mod fragment;
pub use self::fragment::*;
mod view;
pub use self::view::*;
mod ptr;
pub use self::ptr::*;
mod math;
pub use self::math::*;



const ERROR_MATRIX_VECTOR_CONVERT: &'static str = "Only matrices that are continuous in memory can be converted to a vector.";



pub struct CuMatrixDeref<T: CuDataType> {
    pub(crate) ptr: *mut T,
    pub(crate) len: usize,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) leading_dimension: usize,
}

impl<T: CuDataType> Debug for CuMatrixDeref<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let len = self.rows() * self.cols();
        let mut buffer = vec![T::zero(); len];
        self.clone_to_host(&mut buffer);
        write!(f, "Matrix ({},{}) :\n", self.rows, self.cols)?;
        if self.cols > 0 {
            for row in 0..self.rows() {
                write!(f, "[")?;
                for col in 0..self.cols()-1 {
                    write!(f, "{}, ", buffer[row+col*self.rows()])?;
                }
                if row == self.rows()-1 {
                    write!(f, "{}]", buffer[row+(self.cols()-1)*self.rows()])?;
                } else {
                    write!(f, "{}]\n", buffer[row+(self.cols()-1)*self.rows()])?;
                }
            }
        }
        Ok(())
    }
}

/// Immutable matrix operator trait.
impl<T: CuDataType> CuMatrixDeref<T> {

    /// [inline]
    /// Returns the number of rows in the matrix.
    #[inline]
    pub fn rows(&self) -> usize { self.rows }

    /// [inline]
    /// Returns the number of columns in the matrix.
    #[inline]
    pub fn cols(&self) -> usize { self.cols }

    /// [inline]
    /// Returns the number of elements in the matrix.
    #[inline]
    pub fn len(&self) -> usize { self.len }

    /// [inline]
    /// Returns the leading dimension of the matrix.
    #[inline]
    pub fn leading_dimension(&self) -> usize { self.leading_dimension }

    /// [inline]
    /// Returns a pointer on the matrix's data.
    #[inline]
    pub fn as_ptr(&self) -> *const T { self.ptr }

    /// [inline]
    /// Returns a mutable pointer on the matrix's data
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T { self.ptr }


    /// Returns a vector slice containing this matrix datas if this matrix is packed, else returns Err
    pub fn try_as_vector<'a>(&'a self) -> Result<::CuVectorSlice<'a, T>, &'static str> {
        if self.rows() != self.leading_dimension() {
            Err(ERROR_MATRIX_VECTOR_CONVERT)
        } else {
            Ok(::CuVectorSlice {
                _parent: PhantomData,
                deref: ::CuVectorDeref {
                    ptr: self.ptr,
                    len: self.len(),
                }
            })
        }
    }

    /// Returns a mutable vector slice containing this matrix datas if this matrix is packed, else returns Err
    pub fn try_as_mut_vector<'a>(&'a mut self) -> Result<::CuVectorSliceMut<'a, T>, &'static str> {
        if self.rows() != self.leading_dimension() {
            Err(ERROR_MATRIX_VECTOR_CONVERT)
        } else {
            Ok(::CuVectorSliceMut {
                _parent: PhantomData,
                deref: ::CuVectorDeref {
                    ptr: self.ptr,
                    len: self.len(),
                }
            })
        }
    }


    /// Returns an immutable sub-matrix.
    pub fn slice<'a>(&'a self, row_offset: usize, col_offset: usize, nb_rows: usize, nb_cols: usize) -> CuMatrixFragment<'a, T> {
        #[cfg(not(feature = "disable_checks"))] {
            assert_infeq_usize(row_offset + nb_rows, "row_offset+nb_rows", self.rows(), "self.rows()");
            assert_infeq_usize(col_offset + nb_cols, "col_offset+nb_cols", self.cols(), "self.cols()");
        }
        CuMatrixFragment {
            _parent: PhantomData,
            deref: ::CuMatrixDeref {
                ptr: unsafe { self.ptr.offset((row_offset + col_offset*self.leading_dimension()) as isize) },
                len: nb_rows*nb_cols,
                rows: nb_rows,
                cols: nb_cols,
                leading_dimension: self.leading_dimension,
            }
        }
    }

    /// Returns a mutable sub-matrix.
    pub fn slice_mut<'a>(&'a mut self, row_offset: usize, col_offset: usize, nb_rows: usize, nb_cols: usize) -> CuMatrixFragmentMut<'a, T> {
        #[cfg(not(feature = "disable_checks"))] {
            assert_infeq_usize(row_offset + nb_rows, "row_offset+nb_rows", self.rows(), "self.rows()");
            assert_infeq_usize(col_offset + nb_cols, "col_offset+nb_cols", self.cols(), "self.cols()");
        }
        CuMatrixFragmentMut {
            parent: PhantomData,
            deref: ::CuMatrixDeref {
                ptr: unsafe { self.ptr.offset((row_offset + col_offset*self.leading_dimension()) as isize) },
                len: nb_rows*nb_cols,
                rows: nb_rows,
                cols: nb_cols,
                leading_dimension: self.leading_dimension,
            }
        }
    }


    /// Returns an immutable pointer over a matrix :
    /// - It won't free the inner GPU-pointer when it goes out of scope
    /// - It won't check if the underlying memory is still allocated when used
    /// -> Use at your own risk
    pub fn as_wrapped_ptr(&self) -> CuMatrixPtr<T> {
        CuMatrixPtr {
            deref: CuMatrixDeref {
                ptr: self.ptr,
                len: self.len,
                rows: self.rows,
                cols: self.cols,
                leading_dimension: self.leading_dimension
            }
        }
    }

    /// Returns a mutable pointer over a matrix :
    /// - It won't free the inner GPU-pointer when it goes out of scope
    /// - It won't check if the underlying memory is still allocated when used
    /// -> Use at your own risk
    pub fn as_wrapped_mut_ptr(&mut self) -> CuMatrixMutPtr<T> {
        CuMatrixMutPtr {
            deref: CuMatrixDeref {
                ptr: self.ptr,
                len: self.len,
                rows: self.rows,
                cols: self.cols,
                leading_dimension: self.leading_dimension
            }
        }
    }


    /// Clone this matrix, returning a new GPU-allocated matrix
    pub fn clone(&self) -> CuMatrix<T> {
        let mut result = unsafe { CuMatrix::<T>::uninitialized(self.rows(), self.cols()) };
        result.clone_from_device(self);
        result
    }

    /// Clone this matrix's data to host memory.
    pub fn clone_to_host(&self, data: &mut [T]) {
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

    /// Clone host memory to this matrix's data.
    pub fn clone_from_host(&mut self, data: &[T]) {
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
    pub fn clone_from_device(&mut self, data: &CuMatrixDeref<T>) {
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

    pub fn dev_assert_equals(&self, data: &[T]) where Self: Sized {
        if self.len() != data.len() { panic!(); }
        let mut buffer = vec![T::zero(); self.len()];
        self.clone_to_host(buffer.as_mut_slice());
        for i in 0..data.len() {
            if data[i] != buffer[i] { panic!("At index {} : {:.8} != {:.8}", i, data[i], buffer[i]); }
        }
    }

}

impl CuMatrixDeref<f32> {

    /// Initializes the matrix with 'value'.
    pub fn init(&mut self, value: f32, stream: &CudaStream) {
        unsafe {
            VectorFragment_init_f32(self.as_mut_ptr(), self.leading_dimension() as i32, value,
                                    self.rows() as i32, self.cols() as i32, stream.stream);
        }
    }

    /// Add value to each matrix of the vector.
    pub fn add_value(&mut self, value: f32, stream: &CudaStream) {
        unsafe {
            VectorFragment_addValue_f32(self.as_ptr(), self.leading_dimension() as i32, value,
                                        self.as_mut_ptr(), self.leading_dimension() as i32,
                                        self.rows() as i32, self.cols() as i32, stream.stream);
        }
    }

    /// Scale each element of the matrix by 'value'.
    pub fn scl(&mut self, value: f32, stream: &CudaStream) {
        unsafe {
            VectorFragment_scl_f32(self.as_ptr(), self.leading_dimension() as i32, value,
                                   self.as_mut_ptr(), self.leading_dimension() as i32,
                                   self.rows() as i32, self.cols() as i32, stream.stream);
        }
    }

    /// Add an other matrix to this one.
    pub fn add(&mut self, to_add: &CuMatrixDeref<f32>, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(self.rows(), "self.rows()", to_add.rows(), "to_add.rows()");
            assert_eq_usize(self.cols(), "self.cols()", to_add.cols(), "to_add.cols()");
        }
        unsafe {
            VectorFragment_add_f32(self.as_ptr(), self.leading_dimension() as i32,
                                   to_add.as_ptr(), to_add.leading_dimension() as i32,
                                   self.as_mut_ptr(), self.leading_dimension() as i32,
                                   self.rows() as i32, self.cols() as i32, stream.stream)
        }
    }

}

impl CuMatrixDeref<i32> {

    /// Initializes the matrix with 'value'.
    pub fn init(&mut self, value: i32, stream: &CudaStream) {
        unsafe {
            VectorFragment_init_i32(self.as_mut_ptr(), self.leading_dimension() as i32, value,
                                    self.rows() as i32, self.cols() as i32, stream.stream);
        }
    }

    /// Add value to each matrix of the vector.
    pub fn add_value(&mut self, value: i32, stream: &CudaStream) {
        unsafe {
            VectorFragment_addValue_i32(self.as_ptr(), self.leading_dimension() as i32, value,
                                        self.as_mut_ptr(), self.leading_dimension() as i32,
                                        self.rows() as i32, self.cols() as i32, stream.stream);
        }
    }

    /// Scale each element of the matrix by 'value'.
    pub fn scl(&mut self, value: i32, stream: &CudaStream) {
        unsafe {
            VectorFragment_scl_i32(self.as_ptr(), self.leading_dimension() as i32, value,
                                   self.as_mut_ptr(), self.leading_dimension() as i32,
                                   self.rows() as i32, self.cols() as i32, stream.stream);
        }
    }

    /// Add an other matrix to this one.
    pub fn add(&mut self, to_add: &CuMatrixDeref<i32>, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(self.rows(), "self.rows()", to_add.rows(), "to_add.rows()");
            assert_eq_usize(self.cols(), "self.cols()", to_add.cols(), "to_add.cols()");
        }
        unsafe {
            VectorFragment_add_i32(self.as_ptr(), self.leading_dimension() as i32,
                                   to_add.as_ptr(), to_add.leading_dimension() as i32,
                                   self.as_mut_ptr(), self.leading_dimension() as i32,
                                   self.rows() as i32, self.cols() as i32, stream.stream)
        }
    }

}