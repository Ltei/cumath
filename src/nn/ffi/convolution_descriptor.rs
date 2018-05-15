
use std::os::raw::c_void;
use super::{CudnnStatus, CudnnDataType, CudnnConvolutionFwdAlgo, CudnnConvolutionMode, CudnnMathType};
use super::cudnn::_CudnnStruct;
use super::filter_descriptor::_FilterDescriptorStruct;
use super::tensor_descriptor::_TensorDescriptorStruct;



pub enum _ConvolutionDescriptorStruct {}




#[allow(non_snake_case)]
extern {

    fn cudnnCreateConvolutionDescriptor(convDesc: *mut*mut _ConvolutionDescriptorStruct) -> CudnnStatus;

    fn cudnnDestroyConvolutionDescriptor(convDesc: *mut _ConvolutionDescriptorStruct) -> CudnnStatus;

    fn cudnnSetConvolutionNdDescriptor(
        convDesc: *mut _ConvolutionDescriptorStruct,
        arrayLength: i32,
        padA: *const i32,
        filterStrideA: *const i32,
        dilatationA: *const i32,
        mode: CudnnConvolutionMode,
        dataType: CudnnDataType,
    ) -> CudnnStatus;

    fn cudnnSetConvolution2dDescriptor(
        convDesc: *mut _ConvolutionDescriptorStruct,
        pad_h: i32,
        pad_w: i32,
        u: i32,
        v: i32,
        dilatation_h: i32,
        dilatation_w: i32,
        mode: CudnnConvolutionMode,
        computeType: CudnnDataType,
    ) -> CudnnStatus;

    fn cudnnSetConvolutionGroupCount(
        convDesc: *mut _ConvolutionDescriptorStruct,
        groupCount: i32,
    ) -> CudnnStatus;

    fn cudnnSetConvolutionMathType(
        convDesc: *mut _ConvolutionDescriptorStruct,
        math_type: CudnnMathType,
    ) -> CudnnStatus;

    fn cudnnGetConvolutionNdDescriptor(
        convDesc: *const _ConvolutionDescriptorStruct,
        arrayLengthRequested: i32,
        arrayLength: *mut i32,
        padA: *mut i32,
        filterStrideA: *mut i32,
        dilatationA: *mut i32,
        mode: *mut CudnnConvolutionMode,
        dataType: *mut CudnnDataType,
    ) -> CudnnStatus;

    fn cudnnGetConvolutionForwardWorkspaceSize(
        handle: *mut _CudnnStruct,
        xDesc: *const _TensorDescriptorStruct,
        wDesc: *const _FilterDescriptorStruct,
        convDesc: *const _ConvolutionDescriptorStruct,
        yDesc: *const _TensorDescriptorStruct,
        algo: CudnnConvolutionFwdAlgo,
        sizeInBytes: *mut usize
    ) -> CudnnStatus;

    fn cudnnConvolutionForward(
        handle: *mut _CudnnStruct,
        alpha: *const c_void,
        xDesc: *const _TensorDescriptorStruct,
        x: *const c_void,
        wDesc: *const _FilterDescriptorStruct,
        w: *const c_void,
        convDesc: *const _ConvolutionDescriptorStruct,
        algo: CudnnConvolutionFwdAlgo,
        workspace: *mut c_void,
        workspaceSizeInBytes: usize,
        beta: *const c_void,
        yDesc: *const _TensorDescriptorStruct,
        y: *mut c_void,
    ) -> CudnnStatus;

}





#[inline]
pub fn cudnn_create_convolution_descriptor(conv_desc: *mut*mut _ConvolutionDescriptorStruct) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnCreateConvolutionDescriptor(conv_desc) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnCreateConvolutionDescriptor(conv_desc) };
    }
}

#[inline]
pub fn cudnn_destroy_convolution_descriptor(conv_desc: *mut _ConvolutionDescriptorStruct) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnDestroyConvolutionDescriptor(conv_desc) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnDestroyConvolutionDescriptor(conv_desc) };
    }
}

#[inline]
pub fn cudnn_set_convolution_nd_descriptor(conv_desc: *mut _ConvolutionDescriptorStruct, array_length: i32, pad_a: *const i32, filter_stride_a: *const i32, dilatation_a: *const i32, mode: CudnnConvolutionMode, data_type: CudnnDataType) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnSetConvolutionNdDescriptor(conv_desc, array_length, pad_a, filter_stride_a, dilatation_a, mode, data_type) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnSetConvolutionNdDescriptor(conv_desc, array_length, pad_a, filter_stride_a, dilatation_a, mode, data_type) };
    }
}

#[inline]
pub fn cudnn_set_convolution2d_descriptor(conv_desc: *mut _ConvolutionDescriptorStruct, pad_h: i32, pad_w: i32, u: i32, v: i32, dilatation_h: i32, dilatation_w: i32, mode: CudnnConvolutionMode, compute_type: CudnnDataType) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w, u, v, dilatation_h, dilatation_w, mode, compute_type) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w, u, v, dilatation_h, dilatation_w, mode, compute_type) };
    }
}

#[inline]
pub fn cudnn_set_convolution_group_count(conv_desc: *mut _ConvolutionDescriptorStruct, group_count: i32) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnSetConvolutionGroupCount(conv_desc, group_count) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnSetConvolutionGroupCount(conv_desc, group_count) };
    }
}

#[inline]
pub fn cudnn_set_convolution_math_type(conv_desc: *mut _ConvolutionDescriptorStruct, math_type: CudnnMathType) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnSetConvolutionMathType(conv_desc, math_type) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnSetConvolutionMathType(conv_desc, math_type) };
    }
}

#[inline]
pub fn cudnn_get_convolution_nd_descriptor(conv_desc: *const _ConvolutionDescriptorStruct, array_length_requested: i32, array_length: *mut i32, pad_a: *mut i32, filter_stride_a: *mut i32, dilatation_a: *mut i32, mode: *mut CudnnConvolutionMode, data_type: *mut CudnnDataType) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnGetConvolutionNdDescriptor(conv_desc, array_length_requested, array_length, pad_a, filter_stride_a, dilatation_a, mode, data_type) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnGetConvolutionNdDescriptor(conv_desc, array_length_requested, array_length, pad_a, filter_stride_a, dilatation_a, mode, data_type) };
    }
}

#[inline]
pub fn cudnn_get_convolution_forward_workspace_size(handle: *mut _CudnnStruct, x_desc: *const _TensorDescriptorStruct, w_desc: *const _FilterDescriptorStruct, conv_desc: *const _ConvolutionDescriptorStruct, y_desc: *const _TensorDescriptorStruct, algo: CudnnConvolutionFwdAlgo, size_in_bytes: *mut usize) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnGetConvolutionForwardWorkspaceSize(handle, x_desc, w_desc, conv_desc, y_desc, algo, size_in_bytes) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnGetConvolutionForwardWorkspaceSize(handle, x_desc, w_desc, conv_desc, y_desc, algo, size_in_bytes) };
    }
}

#[inline]
pub fn cudnn_convolution_forward(handle: *mut _CudnnStruct, alpha: *const c_void, x_desc: *const _TensorDescriptorStruct, x: *const c_void, w_desc: *const _FilterDescriptorStruct, w: *const c_void, conv_desc: *const _ConvolutionDescriptorStruct, algo: CudnnConvolutionFwdAlgo, workspace: *mut c_void, workspace_size_in_bytes: usize, beta: *const c_void, y_desc: *const _TensorDescriptorStruct, y: *mut c_void) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnConvolutionForward(handle, alpha, x_desc, x, w_desc, w, conv_desc, algo, workspace, workspace_size_in_bytes, beta, y_desc, y) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnConvolutionForward(handle, alpha, x_desc, x, w_desc, w, conv_desc, algo, workspace, workspace_size_in_bytes, beta, y_desc, y) };
    }
}



