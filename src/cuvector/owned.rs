

use super::*;
use std::{ptr, mem::size_of};
use CuDataType;



/// A GPU-allocated vector.
/// Holds a pointer to continuous GPU memory.
pub struct CuVector<T: CuDataType> {
    ptr: *mut T,
    len: usize,
}
impl<T: CuDataType> Drop for CuVector<T> {
    fn drop(&mut self) { cuda_free(self.ptr as *mut c_void) }
}

impl<T: CuDataType> CuVector<T> {

    /// Creates a new CuVector containing data
    pub fn from_host_data(data: &[T]) -> CuVector<T> {
        let mut ptr = ptr::null_mut();
        cuda_malloc(&mut ptr, data.len() * size_of::<T>());
        cuda_memcpy(ptr, data.as_ptr() as *const c_void, data.len() * size_of::<T>(), cudaMemcpyKind::HostToDevice);
        CuVector {
            ptr: ptr as *mut T,
            len: data.len(),
        }
    }

    /// Creates a new CuVector containing data
    pub fn from_device_data(data: &CuVectorOp<T>) -> CuVector<T> {
        let mut ptr = ptr::null_mut();
        cuda_malloc(&mut ptr, data.len() * size_of::<T>());
        cuda_memcpy(ptr, data.as_ptr() as *const c_void, data.len() * size_of::<T>(), cudaMemcpyKind::DeviceToDevice);
        CuVector {
            ptr: ptr as *mut T,
            len: data.len(),
        }
    }

    /// Creates a new CuVector from a pointer and a length
    pub unsafe fn from_raw_ptr(ptr: *mut T, len: usize) -> CuVector<T>  {
        CuVector { ptr, len }
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
                    ptr: ptr as *mut $inner_type,
                    len: len,
                }
            }
            pub fn new(value: $inner_type, len: usize) -> CuVector<$inner_type> {
                let mut ptr = ptr::null_mut();
                cuda_malloc(&mut ptr, len * size_of::<$inner_type>());
                unsafe { $fn_init(ptr as *mut $inner_type, value, len as i32, DEFAULT_STREAM.stream); }
                CuVector {
                    ptr: ptr as *mut $inner_type,
                    len: len,
                }
            }

        }

    };
}

impl_CuVector!(i32, VectorPacked_init_i32);
impl_CuVector!(f32, VectorPacked_init_f32);

impl_mutable_vector_holder!(CuVector);
