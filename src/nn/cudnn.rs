
use super::ffi::*;
use std::ptr;

//pub(crate) static mut NAN_PROPAGATION : CudnnNanPropagation = CudnnNanPropagation::Propagate;


pub struct Cudnn {
    pub(crate) handle: *mut _CudnnStruct,
}

impl Drop for Cudnn {
    fn drop(&mut self) {
        cudnn_destroy(self.handle)
    }
}

impl Cudnn {

    pub fn new() -> Cudnn {
        let mut data = ptr::null_mut();
        cudnn_create(&mut data);
        Cudnn { handle: data }
    }

}

/*impl Cudnn {
    pub fn set_nan_propagation(propagate: bool) {
        unsafe {
            NAN_PROPAGATION = if propagate {
                CudnnNanPropagation::Propagate
            } else {
                CudnnNanPropagation::NotPropagate
            }
        }
    }
    pub fn get_nan_propagation() -> bool {
        unsafe {
            match NAN_PROPAGATION {
                CudnnNanPropagation::NotPropagate => false,
                CudnnNanPropagation::Propagate => true,
            }
        }
    }
}*/