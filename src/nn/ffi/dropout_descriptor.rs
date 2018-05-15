
use std::os::raw::c_void;
use super::{CudnnStatus};
use super::cudnn::_CudnnStruct;
use super::tensor_descriptor::_TensorDescriptorStruct;



pub enum _DropoutDescriptorStruct {}


#[allow(non_snake_case)]
extern {

    fn cudnnCreateDropoutDescriptor(dropoutDesc: *mut*mut _DropoutDescriptorStruct) -> CudnnStatus;

    fn cudnnDestroyDropoutDescriptor(dropoutDesc: *mut _DropoutDescriptorStruct) -> CudnnStatus;

    fn cudnnSetDropoutDescriptor(
        dropoutDesc: *mut _DropoutDescriptorStruct,
        handle: *mut _CudnnStruct,
        dropout: f32,
        states: *mut c_void,
        stateSizeInBytes: usize,
        seed: u64,
    ) -> CudnnStatus;

    fn cudnnGetDropoutDescriptor(
        dropoutDesc: *mut _DropoutDescriptorStruct,
        handle: *mut _CudnnStruct,
        dropout: *mut f32,
        states: *mut*mut c_void,
        seed: *mut u64
    ) -> CudnnStatus;

    fn cudnnDropoutGetReserveSpaceSize(
        xDesc: *const _TensorDescriptorStruct,
        sizeInBytes: *mut usize,
    ) -> CudnnStatus;

    fn cudnnDropoutGetStatesSize(
        handle: *mut _CudnnStruct,
        sizeInBytes: usize,
    ) -> CudnnStatus;

    fn cudnnDropoutForward(
        handle: *mut _CudnnStruct,
        dropoutDesc: *const _DropoutDescriptorStruct,
        xDesc: *const _TensorDescriptorStruct,
        x: *const c_void,
        yDesc: *const _TensorDescriptorStruct,
        y: *mut c_void,
        reserveSpace: *mut c_void,
        reserveSpaceSizeInBytes: usize,
    ) -> CudnnStatus;

    fn cudnnDropoutBackward(
        handle: *mut _CudnnStruct,
        dropoutDesc: *const _DropoutDescriptorStruct,
        dyDesc: *const _TensorDescriptorStruct,
        dy: *const c_void,
        dxDesc: *const _TensorDescriptorStruct,
        dx: *mut c_void,
        reserveSpace: *mut c_void,
        reserveSpaceSizeInBytes: usize,
    ) -> CudnnStatus;

}





#[inline]
pub fn cudnn_create_dropout_descriptor(dropout_desc: *mut*mut _DropoutDescriptorStruct) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnCreateDropoutDescriptor(dropout_desc) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnCreateDropoutDescriptor(dropout_desc) };
    }
}

#[inline]
pub fn cudnn_destroy_dropout_descriptor(dropout_desc: *mut _DropoutDescriptorStruct) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnDestroyDropoutDescriptor(dropout_desc) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnDestroyDropoutDescriptor(dropout_desc) };
    }
}

#[inline]
pub fn cudnn_set_dropout_descriptor(dropout_desc: *mut _DropoutDescriptorStruct, handle: *mut _CudnnStruct, dropout: f32, states: *mut c_void, state_size_in_bytes: usize, seed: u64) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnSetDropoutDescriptor(dropout_desc, handle, dropout, states, state_size_in_bytes, seed) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnSetDropoutDescriptor(dropout_desc, handle, dropout, states, state_size_in_bytes, seed) };
    }
}

#[inline]
pub fn cudnn_get_dropout_descriptor(dropout_desc: *mut _DropoutDescriptorStruct, handle: *mut _CudnnStruct, dropout: *mut f32, states: *mut*mut c_void, seed: *mut u64) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnGetDropoutDescriptor(dropout_desc, handle, dropout, states, seed) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnGetDropoutDescriptor(dropout_desc, handle, dropout, states, seed) };
    }
}

#[inline]
pub fn cudnn_dropout_get_reserve_space_size(x_desc: *const _TensorDescriptorStruct, size_in_bytes: *mut usize) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnDropoutGetReserveSpaceSize(x_desc, size_in_bytes) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnDropoutGetReserveSpaceSize(x_desc, size_in_bytes) };
    }
}

#[inline]
pub fn cudnn_dropout_get_states_size(handle: *mut _CudnnStruct, size_in_bytes: usize) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnDropoutGetStatesSize(handle, size_in_bytes) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnDropoutGetStatesSize(handle, size_in_bytes) };
    }
}

#[inline]
pub fn cudnn_dropout_forward(handle: *mut _CudnnStruct, dropout_desc: *const _DropoutDescriptorStruct, x_desc: *const _TensorDescriptorStruct, x: *const c_void, y_desc: *const _TensorDescriptorStruct, y: *mut c_void, reserve_space: *mut c_void, reserve_space_size_in_bytes: usize) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnDropoutForward(handle, dropout_desc, x_desc, x, y_desc, y, reserve_space, reserve_space_size_in_bytes) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnDropoutForward(handle, dropout_desc, x_desc, x, y_desc, y, reserve_space, reserve_space_size_in_bytes) };
    }
}

#[inline]
pub fn cudnn_dropout_backward(handle: *mut _CudnnStruct, dropout_desc: *const _DropoutDescriptorStruct, dy_desc: *const _TensorDescriptorStruct, dy: *const c_void, dx_desc: *const _TensorDescriptorStruct, dx: *mut c_void, reserve_space: *mut c_void, reserve_space_size_in_bytes: usize) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnDropoutBackward(handle, dropout_desc, dy_desc, dy, dx_desc, dx, reserve_space, reserve_space_size_in_bytes) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnDropoutBackward(handle, dropout_desc, dy_desc, dy, dx_desc, dx, reserve_space, reserve_space_size_in_bytes) };
    }
}



