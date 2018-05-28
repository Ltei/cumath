
#[cfg(not(feature = "disable_checks"))]
use meta::assert::*;

use std::{mem::size_of, os::raw::c_void, marker::PhantomData, fmt::{self, Debug}};
use cuda_core::{cuda::*, cuda_ffi::*};
use CuDataType;
use kernel::*;


mod owned;
pub use self::owned::*;
mod slice;
pub use self::slice::*;
mod view;
pub use self::view::*;
mod ptr;
pub use self::ptr::*;
mod slice_iter;
pub use self::slice_iter::*;
mod math;
pub use self::math::*;


pub struct CuVectorDeref<T: CuDataType> {
    pub(crate) len: usize,
    pub(crate) ptr: *mut T,
}

impl<T: CuDataType> Debug for CuVectorDeref<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let len = self.len as usize;
        let mut buffer = vec![T::zero(); len];
        cuda_memcpy(buffer.as_mut_ptr() as *mut c_void, self.as_ptr() as *const c_void, self.len()*size_of::<T>(), cudaMemcpyKind::DeviceToHost);
        if len > 0 {
            write!(f, "Vector ({}) : [", len)?;
            for i in 0..len-1 {
                write!(f, "{}, ", buffer[i])?;
            }
            write!(f, "{}]", buffer[len-1])
        } else {
            write!(f, "Vector ({}) : []", len,)
        }
    }
}

impl<T: CuDataType> CuVectorDeref<T> {

    /// [inline]
    /// Returns the vector's length.
    #[inline]
    pub fn len(&self) -> usize { self.len }

    /// [inline]
    /// Returns a pointer on the vector's data.
    #[inline]
    pub fn as_ptr(&self) -> *const T { self.ptr }

    /// [inline]
    /// Returns a mutable pointer on the vector's data
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T { self.ptr }

    /// Creates a vector slice starting from self.ptr()+offset, to self.ptr()+offset+len.
    pub fn slice<'a>(&'a self, offset: usize, len: usize) -> CuVectorSlice<'a, T> {
        #[cfg(not(feature = "disable_checks"))] {
            assert_infeq_usize(offset+len, "offset+len", self.len(), "self.len");
        }
        CuVectorSlice {
            _parent: PhantomData,
            deref: CuVectorDeref {
                len,
                ptr: unsafe { self.ptr.offset(offset as isize) },
            }
        }
    }

    /// Returns an iterator over a vector, returning vector slices.
    pub fn slice_iter<'a>(&'a self) -> CuVectorSliceIter<'a, T> {
        CuVectorSliceIter {
            _parent: PhantomData,
            len: self.len,
            ptr: self.ptr,
        }
    }

    /// Creates a matrix slice starting from self.ptr()+offset, to self.ptr()+offset+rows*cols.
    pub fn matrix_slice<'a>(&'a self, offset: usize, rows: usize, cols: usize) -> ::CuMatrixSlice<'a, T> {
        if offset + rows * cols > self.len() { panic!() }
        ::CuMatrixSlice {
            _parent: PhantomData,
            deref: ::CuMatrixDeref {
                ptr: unsafe { self.ptr.offset(offset as isize) },
                len: rows*cols,
                rows,
                cols,
                leading_dimension: rows,
            }
        }
    }

    /// Clone this vector, returning a new GPU-allocated vector
    pub fn clone(&self) -> CuVector<T> {
        let mut result = unsafe { CuVector::<T>::uninitialized(self.len()) };
        result.clone_from_device(self);
        result
    }

    /// Clone this vector's data to host memory.
    pub fn clone_to_host(&self, data: &mut [T]) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(self.len(), "self.len()", data.len(), "data.len()");
        }
        cuda_memcpy(data.as_mut_ptr() as *mut c_void, self.as_ptr() as *const c_void, data.len()*size_of::<T>(), cudaMemcpyKind::DeviceToHost);
    }
    
    /// Returns a mutable vector slice pointing to this vector's data :
    /// self[offset..offset+len]
    pub fn slice_mut<'a>(&'a mut self, offset: usize, len: usize) -> CuVectorSliceMut<'a, T> {
        #[cfg(not(feature = "disable_checks"))] {
            assert_infeq_usize(offset + len, "offset+len", self.len(), "self.len");
        }
        CuVectorSliceMut {
            _parent: PhantomData,
            deref: CuVectorDeref {
                len,
                ptr: unsafe { self.as_mut_ptr().offset(offset as isize) },
            }
        }
    }

    /// Returns two mutable vector slices pointing to this vector's data :
    /// The first one will be self[offset0..offset0+len0]
    /// The second one will be self[offset0+len0+offset1..offset0+len0+offset1+len1]
    pub fn slice_mut2<'a>(&'a mut self, offset0: usize, len0: usize, mut offset1: usize, len1: usize) -> (CuVectorSliceMut<'a, T>, CuVectorSliceMut<'a, T>) {
        offset1 += offset0+len0;
        #[cfg(not(feature = "disable_checks"))] {
            assert!(offset1 + len1 <= self.len());
        }

        (CuVectorSliceMut {
            _parent: PhantomData,
            deref: CuVectorDeref {
                len: len0,
                ptr: unsafe { self.as_mut_ptr().offset(offset0 as isize) },
            }
        },
         CuVectorSliceMut {
             _parent: PhantomData,
             deref: CuVectorDeref {
                 len: len1,
                 ptr: unsafe { self.as_mut_ptr().offset(offset1 as isize) },
             }
         })
    }

    /// Returns a vector mutable vector slices pointing to this vector's data,
    /// for a slice of tuples representing the offset from the end of the last
    /// slice (starting at 0), and the len.
    pub fn slice_mutn<'a>(&'a mut self, offsets_lens: &[(usize,usize)]) -> Vec<CuVectorSliceMut<'a, T>> {
        let mut offset = 0;
        let mut slices = Vec::with_capacity(offsets_lens.len());
        offsets_lens.iter().for_each(|&(off, len)| {
            #[cfg(not(feature = "disable_checks"))] {
                assert!(offset + off + len <= self.len());
            }
            slices.push(CuVectorSliceMut {
                _parent: PhantomData,
                deref: CuVectorDeref {
                    len,
                    ptr: unsafe { self.as_mut_ptr().offset((offset + off) as isize) }
                }
            });
            offset += off + len;
        });
        slices
    }

    /// Returns an iterator over a mutable vector, returning mutable vector slices.
    pub fn slice_mut_iter<'a>(&'a mut self) -> CuVectorSliceIterMut<'a, T> {
        CuVectorSliceIterMut {
            _parent: PhantomData,
            len: self.len(),
            ptr: self.as_mut_ptr(),
        }
    }

    /// Creates a mutable matrix slice starting from self.ptr()+offset, to self.ptr()+offset+rows*cols.
    pub fn matrix_slice_mut<'a>(&'a mut self, offset: usize, rows: usize, cols: usize) -> ::CuMatrixSliceMut<'a, T> {
        if offset + rows * cols > self.len() { panic!() }
        ::CuMatrixSliceMut {
            _parent: PhantomData,
            deref: ::CuMatrixDeref {
                ptr: unsafe { self.ptr.offset(offset as isize) },
                len: rows*cols,
                rows,
                cols,
                leading_dimension: rows,
            }
        }
    }

    /// Returns an immutable pointer over a vector :
    /// - It won't free the inner GPU-pointer when it goes out of scope
    /// - It won't check if the underlying memory is still allocated when used
    /// -> Use at your own risk
    pub fn as_wrapped_ptr(&self) -> CuVectorPtr<T> {
        CuVectorPtr {
            deref: CuVectorDeref {
                ptr: self.ptr,
                len: self.len,
            }
        }
    }

    /// Returns a mutable pointer over a vector :
    /// - It won't free the inner GPU-pointer when it goes out of scope
    /// - It won't check if the underlying memory is still allocated when used
    /// -> Use at your own risk
    pub fn as_wrapped_mut_ptr(&mut self) -> CuVectorMutPtr<T> {
        CuVectorMutPtr {
            deref: CuVectorDeref {
                ptr: self.ptr,
                len: self.len,
            }
        }
    }

    /// Copy host memory into this vector.
    pub fn clone_from_host(&mut self, data: &[T]) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(self.len(), "self.len()", data.len(), "data.len()");
        }
        cuda_memcpy(self.as_mut_ptr() as *mut c_void, data.as_ptr() as *const c_void, data.len()*size_of::<T>(), cudaMemcpyKind::HostToDevice);
    }

    /// Clone device memory into this vector.
    pub fn clone_from_device(&mut self, source: &CuVectorDeref<T>) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(self.len(), "self.len()", source.len(), "data.len()");
        }
        cuda_memcpy(self.as_mut_ptr() as *mut c_void, source.as_ptr() as *const c_void, self.len()*size_of::<T>(), cudaMemcpyKind::DeviceToDevice);
    }

    /// Method that test if this vector's data is equal to the parameter data
    /// This is slow as it has to copy the data back to the CPU in order to evaluate it.
    pub fn dev_assert_equals(&self, data: &[T]) where Self: Sized {
        if self.len() != data.len() { panic!(); }
        let mut buffer = vec![T::zero(); self.len()];
        self.clone_to_host(buffer.as_mut_slice());
        for i in 0..data.len() {
            if data[i] != buffer[i] { panic!("At index {} : {:.8} != {:.8}", i, data[i], buffer[i]); }
        }
    }

}

impl CuVectorDeref<f32> {

    /// Initializes the vector with value.
    pub fn init(&mut self, value: f32, stream: &CudaStream) {
        unsafe { VectorPacked_init_f32(self.as_mut_ptr(), value, self.len() as i32, stream.stream) }
    }

    /// Add value to each element of the vector.
    pub fn add_value(&mut self, value: f32, stream: &CudaStream) {
        unsafe { VectorPacked_addValue_f32(self.as_ptr(), value, self.as_mut_ptr(), self.len() as i32, stream.stream) }
    }

    /// Scale each element of the vector by value.
    pub fn scl(&mut self, value: f32, stream: &CudaStream) {
        unsafe { VectorPacked_scl_f32(self.as_ptr(), value, self.as_mut_ptr(), self.len() as i32, stream.stream) }
    }

    /// Add an other vector to this one.
    pub fn add(&mut self, right_op: &CuVectorDeref<f32>, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(self.len(), "self.len()", right_op.len(), "right_op.len()");
        }
        unsafe { VectorPacked_add_f32(self.as_ptr(), right_op.as_ptr(), self.as_mut_ptr(), self.len() as i32, stream.stream) }
    }

    /// Substract an other vector to this one.
    pub fn sub(&mut self, right_op: &CuVectorDeref<f32>, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(self.len(), "self.len()", right_op.len(), "right_op.len()");
        }
        unsafe { VectorPacked_sub_f32(self.as_ptr(), right_op.as_ptr(), self.as_mut_ptr(), self.len() as i32, stream.stream) }
    }

    /// Multiply each element of the vector by the corresponding element in the parameter vector.
    pub fn mult(&mut self, right_op: &CuVectorDeref<f32>, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(self.len(), "self.len()", right_op.len(), "right_op.len()");
        }
        unsafe { VectorPacked_mult_f32(self.as_ptr(), right_op.as_ptr(), self.as_mut_ptr(), self.len() as i32, stream.stream) }
    }

    /// Multiply each element of the vector by itself.
    pub fn square(&mut self, stream: &CudaStream) {
        unsafe { VectorPacked_square_f32(self.as_ptr(), self.as_mut_ptr(), self.len() as i32, stream.stream) }
    }

    /// vector[i] = sigmoid(vector[i])
    pub fn fast_sigmoid(&mut self, stream: &CudaStream) {
        unsafe { VectorPacked_sigmoid_f32(self.ptr, self.ptr, self.len as i32, stream.stream) }
    }

    /// vector[i] = sigmoid(vector[i])
    pub fn fast_sigmoid_deriv(&mut self, stream: &CudaStream) {
        unsafe { VectorPacked_sigmoidDeriv_f32(self.ptr, self.ptr, self.len as i32, stream.stream) }
    }

    /// vector[i] = tanh(vector[i])
    pub fn fast_tanh(&mut self, stream: &CudaStream) {
        unsafe { VectorPacked_tanh_f32(self.ptr, self.ptr, self.len as i32, stream.stream) }
    }

    /// vector[i] = tanh(vector[i])
    pub fn fast_tanh_deriv(&mut self, stream: &CudaStream) {
        unsafe { VectorPacked_tanhDeriv_f32(self.ptr, self.ptr, self.len as i32, stream.stream) }
    }

    /// vector[i] = sigmoid(vector[i])
    pub fn relu(&mut self, stream: &CudaStream) {
        unsafe { VectorPacked_relu_f32(self.ptr, self.ptr, self.len as i32, stream.stream) }
    }

    /// vector[i] = tanh(vector[i])
    pub fn relu_deriv(&mut self, stream: &CudaStream) {
        unsafe { VectorPacked_reluDeriv_f32(self.ptr, self.ptr, self.len as i32, stream.stream) }
    }

    /// self[i] = self[i] > threshold ? 1.0 : 0.0
    pub fn binarize(&mut self, threshold: f32, stream: &CudaStream) {
        unsafe { VectorPacked_binarize_f32(self.as_ptr(), threshold, self.as_mut_ptr(), self.len() as i32, stream.stream) }
    }
    
    
}

impl CuVectorDeref<i32> {

    /// Initializes the vector with value.
    pub fn init(&mut self, value: i32, stream: &CudaStream) {
        unsafe { VectorPacked_init_i32(self.as_mut_ptr(), value, self.len() as i32, stream.stream) }
    }

    /// Add value to each element of the vector.
    pub fn add_value(&mut self, value: i32, stream: &CudaStream) {
        unsafe { VectorPacked_addValue_i32(self.as_ptr(), value, self.as_mut_ptr(), self.len() as i32, stream.stream) }
    }

    /// Scale each element of the vector by value.
    pub fn scl(&mut self, value: i32, stream: &CudaStream) {
        unsafe { VectorPacked_scl_i32(self.as_ptr(), value, self.as_mut_ptr(), self.len() as i32, stream.stream) }
    }

    /// Add an other vector to this one.
    pub fn add(&mut self, right_op: &CuVectorDeref<i32>, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(self.len(), "self.len()", right_op.len(), "right_op.len()");
        }
        unsafe { VectorPacked_add_i32(self.as_ptr(), right_op.as_ptr(), self.as_mut_ptr(), self.len() as i32, stream.stream) }
    }

    /// Substract an other vector to this one.
    pub fn sub(&mut self, right_op: &CuVectorDeref<i32>, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(self.len(), "self.len()", right_op.len(), "right_op.len()");
        }
        unsafe { VectorPacked_sub_i32(self.as_ptr(), right_op.as_ptr(), self.as_mut_ptr(), self.len() as i32, stream.stream) }
    }

    /// Multiply each element of the vector by the corresponding element in the parameter vector.
    pub fn mult(&mut self, right_op: &CuVectorDeref<i32>, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(self.len(), "self.len()", right_op.len(), "right_op.len()");
        }
        unsafe { VectorPacked_mult_i32(self.as_ptr(), right_op.as_ptr(), self.as_mut_ptr(), self.len() as i32, stream.stream) }
    }

    /// Multiply each element of the vector by itself.
    pub fn square(&mut self, stream: &CudaStream) {
        unsafe { VectorPacked_square_i32(self.as_ptr(), self.as_mut_ptr(), self.len() as i32, stream.stream) }
    }

    /// self[i] = self[i] > threshold ? 1.0 : 0.0
    pub fn binarize(&mut self, threshold: i32, stream: &CudaStream) {
        unsafe { VectorPacked_binarize_i32(self.as_ptr(), threshold, self.as_mut_ptr(), self.len() as i32, stream.stream) }
    }

}