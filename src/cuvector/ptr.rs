
use super::*;
use std::{ptr, mem::size_of};
use cuda_core::{cuda_ffi::{cudaMemcpyKind, cuda_malloc, cuda_memcpy}};
use CuDataType;



/// A pointer over a vector :
/// - It won't free the inner GPU-pointer when it goes out of scope
/// - It won't check if the underlying memory is still allocated when used
/// -> Use at your own risk
pub struct CuVectorPtr<T: CuDataType> {
    deref: CuVectorPtrDeref<T>,
}
pub struct CuVectorPtrDeref<T: CuDataType> {
    len: usize,
    ptr: *mut T,
}

impl<T: CuDataType> CuVectorPtr<T> {

    pub unsafe fn from_device_data(data: &CuVectorOp<T>) -> CuVectorPtr<T> {
        let mut ptr = ptr::null_mut();
        cuda_malloc(&mut ptr, data.len() * size_of::<T>());
        cuda_memcpy(ptr, data.as_ptr() as *const c_void, data.len() * size_of::<T>(), cudaMemcpyKind::DeviceToDevice);
        CuVectorPtr {
            deref: CuVectorPtrDeref {
                ptr: ptr as *mut T,
                len: data.len(),
            }
        }
    }
    pub unsafe fn from_host_data(data: &[T]) -> CuVectorPtr<T> {
        let mut ptr = ptr::null_mut();
        cuda_malloc(&mut ptr, data.len() * size_of::<T>());
        cuda_memcpy(ptr, data.as_ptr() as *const c_void, data.len() * size_of::<T>(), cudaMemcpyKind::HostToDevice);
        CuVectorPtr {
            deref: CuVectorPtrDeref {
                ptr: ptr as *mut T,
                len: data.len(),
            }
        }
    }
    pub unsafe fn from_raw_ptr(ptr: *mut T, len: usize) -> CuVectorPtr<T>  {
        CuVectorPtr {
            deref: CuVectorPtrDeref { ptr, len }
        }
    }

    /// [inline]
    /// Returns the length of the underlying vector (even if the pointed memory has been freed)
    #[inline]
    pub fn len(&self) -> usize { self.deref.len as usize }

    /// [inline]
    /// Returns an immutable reference to the underlying vector
    #[inline]
    pub unsafe fn deref(&self) -> &CuVectorPtrDeref<T> {
        &self.deref
    }

    /// [inline]
    /// Returns an mutable reference to the underlying vector
    #[inline]
    pub unsafe fn deref_mut(&mut self) -> &mut CuVectorPtrDeref<T> {
        &mut self.deref
    }
}

impl_mutable_vector_holder!(CuVectorPtrDeref);