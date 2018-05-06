
#[cfg(not(feature = "disable_checks"))]
use meta::assert::*;

use std::{fmt, mem::size_of, os::raw::c_void, marker::PhantomData};
use cuda_core::{cuda::*, cuda_ffi::*};
use CuDataType;
use kernel::*;


mod owned;
pub use self::owned::*;
mod slice;
pub use self::slice::*;
mod ptr;
pub use self::ptr::*;
mod slice_iter;
pub use self::slice_iter::*;
mod math;
pub use self::math::*;



pub trait CuVectorOp<T: CuDataType>: fmt::Debug {

    /// [inline]
    /// Returns the vector's length.
    #[inline]
    fn len(&self) -> usize;

    /// [inline]
    /// Returns a pointer on the vector's data.
    #[inline]
    fn as_ptr(&self) -> *const T;

    /// Creates a vector slice starting from self.ptr()+offset, to self.ptr()+offset+len.
    fn slice<'a>(&'a self, offset: usize, len: usize) -> CuVectorSlice<'a, T> {
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
    fn slice_iter<'a>(&'a self) -> CuVectorSliceIter<'a, T> {
        CuVectorSliceIter {
            _parent: PhantomData,
            len: self.len(),
            ptr: self.as_ptr(),
        }
    }

    fn matrix_slice<'a>(&'a self, offset: usize, rows: usize, cols: usize) -> ::CuMatrixSlice<'a, T> {
        if offset + rows * cols >= self.len() { panic!() }
        ::CuMatrixSlice {
            _parent: PhantomData,
            ptr: unsafe { self.as_ptr().offset(offset as isize) },
            rows, cols, len: rows*cols
        }
    }

    /// Clone this vector's data to host memory.
    fn clone_to_host(&self, data: &mut [T]) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(self.len(), "self.len()", data.len(), "data.len()");
        }
        cuda_memcpy(data.as_mut_ptr() as *mut c_void, self.as_ptr() as *const c_void, data.len()*size_of::<T>(), cudaMemcpyKind::DeviceToHost);
    }

    fn dev_assert_equals(&self, data: &[T]) where Self: Sized {
        if self.len() != data.len() { panic!(); }
        let mut buffer = vec![T::zero(); self.len()];
        self.clone_to_host(buffer.as_mut_slice());
        for i in 0..data.len() {
            if data[i] != buffer[i] { panic!("At index {} : {:.8} != {:.8}", i, data[i], buffer[i]); }
        }
    }

}


/// Mutable vector operator trait.
pub trait CuVectorOpMut<T: CuDataType>: CuVectorOp<T> {

    /// [inline]
    /// Returns a mutable pointer on the vector's data
    #[inline]
    fn as_mut_ptr(&mut self) -> *mut T;

    /// [inline]
    /// Supertrait upcasting
    #[inline]
    fn as_immutable(&self) -> &CuVectorOp<T>;

    /// Returns a mutable vector slice pointing to this vector's data :
    /// self[offset..offset+len]
    fn slice_mut<'a>(&'a mut self, offset: usize, len: usize) -> CuVectorSliceMut<'a, T> {
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
    fn slice_mut2<'a>(&'a mut self, offset0: usize, len0: usize, mut offset1: usize, len1: usize) -> (CuVectorSliceMut<'a, T>, CuVectorSliceMut<'a, T>) {
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

    /// Returns a vector mutable vector slices pointing to this vector's data,
    /// for a slice of tuples representing the offset from the end of the last
    /// slice (starting at 0), and the len.
    fn slice_mutn<'a>(&'a mut self, offsets_lens: &[(usize,usize)]) -> Vec<CuVectorSliceMut<'a, T>> {
        let mut offset = 0;
        let mut slices = Vec::with_capacity(offsets_lens.len());
        offsets_lens.iter().for_each(|&(off, len)| {
            #[cfg(not(feature = "disable_checks"))] {
                assert!(offset + off + len <= self.len());
            }
            slices.push(CuVectorSliceMut {
                _parent: PhantomData, len,
                ptr: unsafe { self.as_mut_ptr().offset((offset+off) as isize) }
            });
            offset += off + len;
        });
        slices
    }

    /// Returns an iterator over a mutable vector, returning mutable vector slices.
    fn slice_mut_iter<'a>(&'a mut self) -> CuVectorSliceIterMut<'a, T> {
        CuVectorSliceIterMut {
            _parent: PhantomData,
            len: self.len(),
            ptr: self.as_mut_ptr(),
        }
    }

    fn matrix_slice_mut<'a>(&'a mut self, offset: usize, rows: usize, cols: usize) -> ::CuMatrixSliceMut<'a, T> {
        if offset + rows * cols >= self.len() { panic!() }
        ::CuMatrixSliceMut {
            _parent: PhantomData,
            ptr: unsafe { self.as_mut_ptr().offset(offset as isize) },
            rows, cols, len: rows*cols
        }
    }

    /// Returns a pointer over a vector :
    /// - It won't free the inner GPU-pointer when it goes out of scope
    /// - It won't check if the underlying memory is still allocated when used
    /// -> Use at your own risk
    fn as_wrapped_ptr(&mut self) -> CuVectorPtr<T> {
        unsafe { CuVectorPtr::<T>::from_raw_ptr(self.as_mut_ptr(), self.len()) }
    }

    /// Copy host memory into this vector.
    fn clone_from_host(&mut self, data: &[T]) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(self.len(), "self.len()", data.len(), "data.len()");
        }
        cuda_memcpy(self.as_mut_ptr() as *mut c_void, data.as_ptr() as *const c_void, data.len()*size_of::<T>(), cudaMemcpyKind::HostToDevice);
    }

    /// Clone device memory into this vector.
    fn clone_from_device(&mut self, source: &CuVectorOp<T>) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(self.len(), "self.len()", source.len(), "data.len()");
        }
        cuda_memcpy(self.as_mut_ptr() as *mut c_void, source.as_ptr() as *const c_void, self.len()*size_of::<T>(), cudaMemcpyKind::DeviceToDevice);
    }

    /// Initializes the vector with value.
    fn init(&mut self, value: T, stream: &CudaStream);

    /// Add value to each element of the vector.
    fn add_value(&mut self, value: T, stream: &CudaStream);

    /// Scale each element of the vector by value.
    fn scl(&mut self, value: T, stream: &CudaStream);

    /// Add an other vector to this one.
    fn add(&mut self, right_op: &CuVectorOp<T>, stream: &CudaStream);

    /// Substract an other vector to this one.
    fn sub(&mut self, right_op: &CuVectorOp<T>, stream: &CudaStream);

    /// Multiply each element of the vector by the corresponding element in the parameter vector.
    fn pmult(&mut self, right_op: &CuVectorOp<T>, stream: &CudaStream);

    /// Multiply each element of the vector by itself.
    fn psquare(&mut self, stream: &CudaStream);

    /// self[i] = self[i] > threshold ? 1.0 : 0.0
    fn binarize(&mut self, threshold: T, stream: &CudaStream);

}