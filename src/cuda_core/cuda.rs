
use std::ptr;

use super::cuda_ffi::*;



pub const DEFAULT_STREAM : CudaStream = CudaStream { stream: ptr::null_mut() };



pub struct Cuda {}
impl Cuda {
    pub fn synchronize() {
        cuda_device_synchronize();
    }
}



pub struct CudaStream {
    pub(crate) stream: cudaStream_t,
}
impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.stream.is_null() { cuda_stream_destroy(self.stream) }
    }
}
impl CudaStream {
    pub fn new() -> CudaStream {
        let mut ptr = ptr::null_mut();
        cuda_stream_create(&mut ptr);
        CudaStream { stream: ptr }
    }
    pub fn default() -> CudaStream {
        CudaStream { stream: ptr::null_mut() }
    }

    pub fn synchronize(&self) {
        cuda_stream_synchronize(self.stream)
    }
}