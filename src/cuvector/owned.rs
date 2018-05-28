

use super::*;
use std::{ptr, mem::size_of, ops::{Deref, DerefMut}};
use CuDataType;



/// A GPU-allocated vector.
/// Holds a pointer to continuous GPU memory.
#[derive(Debug)]
pub struct CuVector<T: CuDataType> {
    deref: CuVectorDeref<T>,
}
impl<T: CuDataType> Drop for CuVector<T> {
    fn drop(&mut self) { cuda_free(self.deref.ptr as *mut c_void) }
}

impl<T: CuDataType> Deref for CuVector<T> {
    type Target = CuVectorDeref<T>;
    fn deref(&self) -> &CuVectorDeref<T> { &self.deref }
}
impl<T: CuDataType> DerefMut for CuVector<T> {
    fn deref_mut(&mut self) -> &mut CuVectorDeref<T> { &mut self.deref }
}

impl<T: CuDataType> CuVector<T> {

    /// Returns a new uninitialized GPU-allocated CuVector.
    pub unsafe fn uninitialized(len: usize) -> CuVector<T> {
        let mut ptr = ptr::null_mut();
        cuda_malloc(&mut ptr, len * size_of::<T>());
        CuVector {
            deref: CuVectorDeref {
                ptr: ptr as *mut T,
                len: len,
            }
        }
    }

    /// Creates a new CuVector containing data
    pub fn from_host_data(data: &[T]) -> CuVector<T> {
        let mut ptr = ptr::null_mut();
        cuda_malloc(&mut ptr, data.len() * size_of::<T>());
        cuda_memcpy(ptr, data.as_ptr() as *const c_void, data.len() * size_of::<T>(), cudaMemcpyKind::HostToDevice);
        CuVector {
            deref: CuVectorDeref {
                ptr: ptr as *mut T,
                len: data.len(),
            }
        }
    }

    /// Creates a new CuVector containing data
    pub fn from_device_data(data: &CuVectorDeref<T>) -> CuVector<T> {
        let mut ptr = ptr::null_mut();
        cuda_malloc(&mut ptr, data.len() * size_of::<T>());
        cuda_memcpy(ptr, data.as_ptr() as *const c_void, data.len() * size_of::<T>(), cudaMemcpyKind::DeviceToDevice);
        CuVector {
            deref: CuVectorDeref {
                ptr: ptr as *mut T,
                len: data.len(),
            }
        }
    }

    /// Creates a new CuVector from a pointer and a length
    pub unsafe fn from_raw_ptr(ptr: *mut T, len: usize) -> CuVector<T>  {
        CuVector {
            deref: CuVectorDeref {
                ptr,
                len,
            }
        }
    }

}

macro_rules! impl_CuVector {
    ( $inner_type:ident, $fn_init:tt) => {

        impl CuVector<$inner_type> {

            pub fn zero(len: usize) -> CuVector<$inner_type> {
                let mut ptr = ptr::null_mut();
                cuda_malloc(&mut ptr, len * size_of::<$inner_type>());
                unsafe { $fn_init(ptr as *mut $inner_type, $inner_type::zero(), len as i32, DEFAULT_STREAM.stream) }
                CuVector {
                    deref: CuVectorDeref {
                        ptr: ptr as *mut $inner_type,
                        len: len,
                    }
                }
            }
            pub fn new(value: $inner_type, len: usize) -> CuVector<$inner_type> {
                let mut ptr = ptr::null_mut();
                cuda_malloc(&mut ptr, len * size_of::<$inner_type>());
                unsafe { $fn_init(ptr as *mut $inner_type, value, len as i32, DEFAULT_STREAM.stream); }
                CuVector {
                    deref: CuVectorDeref {
                        ptr: ptr as *mut $inner_type,
                        len: len,
                    }
                }
            }

        }

    };
}

impl_CuVector!(i32, VectorPacked_init_i32);
impl_CuVector!(f32, VectorPacked_init_f32);
