
use super::CudnnStatus;



pub enum _CudnnStruct {}




#[allow(non_snake_case)]
extern {

    fn cudnnCreate(handle: *mut *mut _CudnnStruct) -> CudnnStatus;

    fn cudnnDestroy(handle: *mut _CudnnStruct) -> CudnnStatus;

}





#[inline]
pub fn cudnn_create(handle: *mut *mut _CudnnStruct) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnCreate(handle) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnCreate(handle) };
    }
}

#[inline]
pub fn cudnn_destroy(handle: *mut _CudnnStruct) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnDestroy(handle) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnDestroy(handle) };
    }
}



