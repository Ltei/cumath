
use std::os::raw::c_void;
use super::{CudnnStatus, CudnnActivationMode, CudnnNanPropagation};
use super::cudnn::_CudnnStruct;
use super::tensor_descriptor::_TensorDescriptorStruct;



pub enum _ActivationDescriptorStruct {}




#[allow(non_snake_case)]
extern {

    fn cudnnCreateActivationDescriptor(activationDesc: *mut*mut _ActivationDescriptorStruct) -> CudnnStatus;

    fn cudnnDestroyActivationDescriptor(activationDesc: *mut _ActivationDescriptorStruct) -> CudnnStatus;

    fn cudnnSetActivationDescriptor(
        activationDesc: *mut _ActivationDescriptorStruct,
        mode: CudnnActivationMode,
        reluNanOpt: CudnnNanPropagation,
        coef: f64
    ) -> CudnnStatus;

    fn cudnnGetActivationDescriptor(
        activationDesc: *const _ActivationDescriptorStruct,
        mode: *mut CudnnActivationMode,
        reluNanOpt: *mut CudnnNanPropagation,
        coef: *mut f64
    ) -> CudnnStatus;

    fn cudnnActivationForward(
        handle: *const _CudnnStruct,
        activationDesc: *const _ActivationDescriptorStruct,
        alpha: *const c_void,
        xDesc: *const _TensorDescriptorStruct,
        x: *const c_void,
        beta: *const c_void,
        yDesc: *const _TensorDescriptorStruct,
        y: *mut c_void
    ) -> CudnnStatus;

    fn cudnnActivationBackward(
        handle: *const _CudnnStruct,
        activationDesc: *const _ActivationDescriptorStruct,
        alpha: *const c_void,
        yDesc: *const _TensorDescriptorStruct,
        y: *const c_void,
        dyDesc: *const _TensorDescriptorStruct,
        dy: *const c_void,
        xDesc: *const _TensorDescriptorStruct,
        x: *const c_void,
        beta: *const c_void,
        dxDesc: *const _TensorDescriptorStruct,
        dx: *mut c_void,
    ) -> CudnnStatus;

}





#[inline]
pub fn cudnn_create_activation_descriptor(activation_desc: *mut*mut _ActivationDescriptorStruct) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnCreateActivationDescriptor(activation_desc) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnCreateActivationDescriptor(activation_desc) };
    }
}

#[inline]
pub fn cudnn_destroy_activation_descriptor(activation_desc: *mut _ActivationDescriptorStruct) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnDestroyActivationDescriptor(activation_desc) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnDestroyActivationDescriptor(activation_desc) };
    }
}

#[inline]
pub fn cudnn_set_activation_descriptor(activation_desc: *mut _ActivationDescriptorStruct, mode: CudnnActivationMode, relu_nan_opt: CudnnNanPropagation, coef: f64) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnSetActivationDescriptor(activation_desc, mode, relu_nan_opt, coef) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnSetActivationDescriptor(activation_desc, mode, relu_nan_opt, coef) };
    }
}

#[inline]
pub fn cudnn_get_activation_descriptor(activation_desc: *const _ActivationDescriptorStruct, mode: *mut CudnnActivationMode, relu_nan_opt: *mut CudnnNanPropagation, coef: *mut f64) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnGetActivationDescriptor(activation_desc, mode, relu_nan_opt, coef) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnGetActivationDescriptor(activation_desc, mode, relu_nan_opt, coef) };
    }
}

#[inline]
pub fn cudnn_activation_forward(handle: *const _CudnnStruct, activation_desc: *const _ActivationDescriptorStruct, alpha: *const c_void, x_desc: *const _TensorDescriptorStruct, x: *const c_void, beta: *const c_void, y_desc: *const _TensorDescriptorStruct, y: *mut c_void) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnActivationForward(handle, activation_desc, alpha, x_desc, x, beta, y_desc, y) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnActivationForward(handle, activation_desc, alpha, x_desc, x, beta, y_desc, y) };
    }
}

#[inline]
pub fn cudnn_activation_backward(handle: *const _CudnnStruct, activation_desc: *const _ActivationDescriptorStruct, alpha: *const c_void, y_desc: *const _TensorDescriptorStruct, y: *const c_void, dy_desc: *const _TensorDescriptorStruct, dy: *const c_void, x_desc: *const _TensorDescriptorStruct, x: *const c_void, beta: *const c_void, dx_desc: *const _TensorDescriptorStruct, dx: *mut c_void) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnActivationBackward(handle, activation_desc, alpha, y_desc, y, dy_desc, dy, x_desc, x, beta, dx_desc, dx) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnActivationBackward(handle, activation_desc, alpha, y_desc, y, dy_desc, dy, x_desc, x, beta, dx_desc, dx) };
    }
}



