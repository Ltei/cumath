
use std::os::raw::c_void;
use super::{CudnnStatus, CudnnDataType, CudnnTensorFormat, cudnn::_CudnnStruct};



pub enum _TensorDescriptorStruct {}




#[allow(non_snake_case)]
extern {

    fn cudnnCreateTensorDescriptor(tensorDesc: *mut*mut _TensorDescriptorStruct) -> CudnnStatus;

    fn cudnnDestroyTensorDescriptor(tensorDesc: *mut _TensorDescriptorStruct) -> CudnnStatus;

    fn cudnnSetTensorNdDescriptor(
        tensorDesc: *mut _TensorDescriptorStruct,
        dataType: CudnnDataType,
        nbDims: i32,
        dimA: *const i32,
        strideA: *const i32
    ) -> CudnnStatus;

    fn cudnnSetTensor4dDescriptor(
        tensorDesc: *mut _TensorDescriptorStruct,
        format: CudnnTensorFormat,
        dataType: CudnnDataType,
        n: i32,
        c: i32,
        h: i32,
        w: i32
    ) -> CudnnStatus;

    fn cudnnGetTensorNdDescriptor(
        tensorDesc: *mut _TensorDescriptorStruct,
        nbDimsRequested: i32,
        dataType: *mut CudnnDataType,
        nbDims: *mut i32,
        dimA: *mut i32,
        strideA: *mut i32
    ) -> CudnnStatus;

    fn cudnnGetTensorSizeInBytes(
        tensorDesc: *mut _TensorDescriptorStruct,
        size: &mut usize
    ) -> CudnnStatus;

    fn cudnnSetTensor(
        handle: *mut _CudnnStruct,
        yDesc: *const _TensorDescriptorStruct,
        y: *mut c_void,
        valuePtr: *const c_void
    ) -> CudnnStatus;

}





#[inline]
pub fn cudnn_create_tensor_descriptor(tensor_desc: *mut*mut _TensorDescriptorStruct) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnCreateTensorDescriptor(tensor_desc) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnCreateTensorDescriptor(tensor_desc) };
    }
}

#[inline]
pub fn cudnn_destroy_tensor_descriptor(tensor_desc: *mut _TensorDescriptorStruct) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnDestroyTensorDescriptor(tensor_desc) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnDestroyTensorDescriptor(tensor_desc) };
    }
}

#[inline]
pub fn cudnn_set_tensor_nd_descriptor(tensor_desc: *mut _TensorDescriptorStruct, data_type: CudnnDataType, nb_dims: i32, dim_a: *const i32, stride_a: *const i32) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnSetTensorNdDescriptor(tensor_desc, data_type, nb_dims, dim_a, stride_a) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnSetTensorNdDescriptor(tensor_desc, data_type, nb_dims, dim_a, stride_a) };
    }
}

#[inline]
pub fn cudnn_set_tensor4d_descriptor(tensor_desc: *mut _TensorDescriptorStruct, format: CudnnTensorFormat, data_type: CudnnDataType, n: i32, c: i32, h: i32, w: i32) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnSetTensor4dDescriptor(tensor_desc, format, data_type, n, c, h, w) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnSetTensor4dDescriptor(tensor_desc, format, data_type, n, c, h, w) };
    }
}

#[inline]
pub fn cudnn_get_tensor_nd_descriptor(tensor_desc: *mut _TensorDescriptorStruct, nb_dims_requested: i32, data_type: *mut CudnnDataType, nb_dims: *mut i32, dim_a: *mut i32, stride_a: *mut i32) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnGetTensorNdDescriptor(tensor_desc, nb_dims_requested, data_type, nb_dims, dim_a, stride_a) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnGetTensorNdDescriptor(tensor_desc, nb_dims_requested, data_type, nb_dims, dim_a, stride_a) };
    }
}

#[inline]
pub fn cudnn_get_tensor_size_in_bytes(tensor_desc: *mut _TensorDescriptorStruct, size: &mut usize) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnGetTensorSizeInBytes(tensor_desc, size) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnGetTensorSizeInBytes(tensor_desc, size) };
    }
}

#[inline]
pub fn cudnn_set_tensor(handle: *mut _CudnnStruct, y_desc: *const _TensorDescriptorStruct, y: *mut c_void, value_ptr: *const c_void) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnSetTensor(handle, y_desc, y, value_ptr) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnSetTensor(handle, y_desc, y, value_ptr) };
    }
}



