
use std::ptr;

use ffi::cuda_ffi::*;
//use std::process::Command;

pub const DEFAULT_STREAM : CudaStream = CudaStream { stream: ptr::null_mut() };


pub struct Cuda {}
impl Cuda {
    pub fn synchronize() {
        cuda_device_synchronize();
    }

    /*pub fn create_kernel(params: &str, core: &str) {
        let kernel_str =  Self::get_kernel_str(params, core);

        let nvcc_exit_status = Command::new("nvcc -ccbin clang-3.8")
            .output().expect("Could not process nvcc command");
        if nvcc_exit_status != 0 { panic!("Could not compile cuda kernel") }
    }
    fn get_kernel_str(params: &str, core: &str) -> String {
        String::with_capacity(1024).add("extern \"C\" __global__ void kernel(").add(params).add(") {").add(core).push('}')
    }*/
}


#[derive(Eq, PartialEq, Debug)]
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