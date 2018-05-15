#![allow(dead_code)]

use std::os::raw::c_void;


pub enum CudnnStruct {}
pub enum TensorDescriptorStruct {}
pub enum ReduceTensorDescriptorStruct {}
pub enum ActivationDescriptorStruct {}
pub enum ConvolutionDescriptorStruct {}



#[derive(PartialEq, Debug, Clone, Copy)]
#[repr(C)]
pub enum CudnnDataType {
    Float = 0,
    Double = 1,
    Half = 2,
    Int8 = 3,
    Int32 = 4,
    Int8x4 = 5,
    Uint8 = 6,
    Uint8x4 = 7,
}

#[derive(PartialEq, Debug, Clone, Copy)]
#[repr(C)]
pub enum CudnnNanPropagation {
    NotPropagate = 0,
    Propagate = 1,
}

#[derive(PartialEq, Debug, Clone, Copy)]
#[repr(C)]
pub enum CudnnStatus {
    Success = 0,
    NotInitialized = 1,
    AllocFailed = 2,
    BadParam = 3,
    InternalError = 4,
    InvalidValue = 5,
    ArchMismatch = 6,
    MappingError = 7,
    ExecutionFailed = 8,
    NotSupported = 9,
    LicenseError = 10,
    RuntimePrerequisiteMissing = 11,
    RuntimeInProgress = 12,
    RuntimeFPOverflow = 13,
}
impl CudnnStatus {
    fn assert_success(&self) {
        assert_eq!(self, &CudnnStatus::Success);
    }
    fn get_error_str(&self) -> Option<&'static str> {
        match *self {
            CudnnStatus::Success => None,
            CudnnStatus::NotInitialized => Some("NotInitialized"),
            CudnnStatus::AllocFailed => Some("AllocFailed"),
            CudnnStatus::BadParam => Some("BadParam"),
            CudnnStatus::InvalidValue => Some("InvalidValue"),
            CudnnStatus::ArchMismatch => Some("ArchMismatch"),
            CudnnStatus::MappingError => Some("MappingError"),
            CudnnStatus::ExecutionFailed => Some("ExecutionFailed"),
            CudnnStatus::InternalError => Some("InternalError"),
            CudnnStatus::NotSupported => Some("NotSupported"),
            CudnnStatus::LicenseError => Some("LicenseError"),
            CudnnStatus::RuntimePrerequisiteMissing => Some("RuntimePrerequisiteMissing"),
            CudnnStatus::RuntimeInProgress => Some("RuntimeInProgress"),
            CudnnStatus::RuntimeFPOverflow => Some("RuntimeFPOverflow"),
        }
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]
#[repr(C)]
pub enum CudnnReduceTensorOp {
    Add = 0,
    Mul = 1,
    Min = 2,
    Max = 3,
    Amax = 4,
    Avg = 5,
    Norm1 = 6,
    Norm2 = 7,
    MulNoZeros = 8,
}

#[derive(PartialEq, Debug, Clone, Copy)]
#[repr(C)]
pub enum CudnnReduceTensorIndices {
    NoIndices = 0,
    FlattenedIndices = 1,
}

#[derive(PartialEq, Debug, Clone, Copy)]
#[repr(C)]
pub enum CudnnIndicesType {
    Indices32bit = 0,
    Indices64bit = 1,
    Indices16bit = 2,
    Indices8bit = 3,
}

#[derive(PartialEq, Debug, Clone, Copy)]
#[repr(C)]
pub enum CudnnConvolutionMode {
    Convolution = 0,
    CrossCorrelation = 1,
}

#[derive(PartialEq, Debug, Clone, Copy)]
#[repr(C)]
pub enum CudnnActivationMode {
    Sigmoid = 0,
    Relu = 1,
    Tanh = 2,
    ClippedRelu = 3,
    Elu = 4,
    //Identity = 5, Doesn't work, but it is useless anyway
}

#[derive(PartialEq, Debug, Clone, Copy)]
#[repr(C)]
pub enum CudnnConvolutionFwdAlgo {
    ImplicitGemm = 0,
    ImplicitPrecompGemm = 1,
    Gemm = 2,
    Direct = 3,
    Fft = 4,
    FftTiling = 5,
    Winograd = 6,
    WinogradNonfused = 7,
    Count = 8,
}

#[derive(PartialEq, Debug, Clone, Copy)]
#[repr(C)]
pub enum CudnnMathType {
    Default = 0,
    TensorOp = 1,
}




#[allow(non_snake_case)]
extern {

    fn cudnnCreate(handle: *mut*mut CudnnStruct) -> CudnnStatus;

    fn cudnnDestroy(handle: *mut CudnnStruct) -> CudnnStatus;

    fn cudnnCreateTensorDescriptor(tensorDesc: *mut*mut TensorDescriptorStruct) -> CudnnStatus;

    fn cudnnDestroyTensorDescriptor(tensorDesc: *mut TensorDescriptorStruct) -> CudnnStatus;

    fn cudnnSetTensorNdDescriptor(
        tensorDesc: *mut TensorDescriptorStruct,
        dataType: CudnnDataType,
        nbDims: i32,
        dimA: *const i32,
        strideA: *const i32
    ) -> CudnnStatus;

    fn cudnnGetTensorNdDescriptor(
        tensorDesc: *mut TensorDescriptorStruct,
        nbDimsRequested: i32,
        dataType: *mut CudnnDataType,
        nbDims: *mut i32,
        dimA: *mut i32,
        strideA: *mut i32
    ) -> CudnnStatus;

    fn cudnnGetTensorSizeInBytes(
        tensorDesc: *mut TensorDescriptorStruct,
        size: &mut usize
    ) -> CudnnStatus;

    fn cudnnSetTensor(
        handle: *mut CudnnStruct,
        yDesc: *const TensorDescriptorStruct,
        y: *mut c_void,
        valuePtr: *const c_void
    ) -> CudnnStatus;

    fn cudnnCreateReduceTensorDescriptor(reduceTensorDesc: *mut*mut ReduceTensorDescriptorStruct) -> CudnnStatus;

    fn cudnnDestroyReduceTensorDescriptor(reduceTensorDesc: *mut ReduceTensorDescriptorStruct) -> CudnnStatus;

    fn cudnnSetReduceTensorDescriptor(
        reduceTensorDesc: *mut ReduceTensorDescriptorStruct,
        reduceTensorOp: CudnnReduceTensorOp,
        reduceTensorCompType: CudnnDataType,
        reduceTensorNanOpt: CudnnNanPropagation,
        reduceTensorIndices: CudnnReduceTensorIndices,
        reduceTensorIndicesType: CudnnIndicesType
    ) -> CudnnStatus;

    fn cudnnGetReduceTensorDescriptor(
        reduceTensorDesc: *const ReduceTensorDescriptorStruct,
        reduceTensorOp: &mut CudnnReduceTensorOp,
        reduceTensorCompType: &mut CudnnDataType,
        reduceTensorNanOpt: &mut CudnnNanPropagation,
        reduceTensorIndices: &mut CudnnReduceTensorIndices,
        reduceTensorIndicesType: &mut CudnnIndicesType
    ) -> CudnnStatus;

    fn cudnnCreateActivationDescriptor(activationDesc: *mut*mut ActivationDescriptorStruct) -> CudnnStatus;

    fn cudnnDestroyActivationDescriptor(activationDesc: *mut ActivationDescriptorStruct) -> CudnnStatus;

    fn cudnnSetActivationDescriptor(
        activationDesc: *mut ActivationDescriptorStruct,
        mode: CudnnActivationMode,
        reluNanOpt: CudnnNanPropagation,
        coef: f64
    ) -> CudnnStatus;

    fn cudnnGetActivationDescriptor(
        activationDesc: *const ActivationDescriptorStruct,
        mode: *mut CudnnActivationMode,
        reluNanOpt: *mut CudnnNanPropagation,
        coef: *mut f64
    ) -> CudnnStatus;

    fn cudnnActivationForward(
        handle: *const CudnnStruct,
        activationDesc: *const ActivationDescriptorStruct,
        alpha: *const c_void,
        xDesc: *const TensorDescriptorStruct,
        x: *const c_void,
        beta: *const c_void,
        yDesc: *const TensorDescriptorStruct,
        y: *mut c_void
    ) -> CudnnStatus;

    fn cudnnActivationBackward(
        handle: *const CudnnStruct,
        activationDesc: *const ActivationDescriptorStruct,
        alpha: *const c_void,
        yDesc: *const TensorDescriptorStruct,
        y: *const c_void,
        dyDesc: *const TensorDescriptorStruct,
        dy: *const c_void,
        beta: *const c_void,
        xDesc: *const TensorDescriptorStruct,
        x: *mut c_void
    ) -> CudnnStatus;

    fn cudnnCreateConvolutionDescriptor(convDesc: *mut*mut ConvolutionDescriptorStruct) -> CudnnStatus;

    fn cudnnDestroyConvolutionDescriptor(convDesc: *mut ConvolutionDescriptorStruct) -> CudnnStatus;

    fn cudnnSetConvolutionNdDescriptor(
        convDesc: *mut ConvolutionDescriptorStruct,
        arrayLength: i32,
        padA: *const i32,
        filterStrideA: *const i32,
        dilatationA: *const i32,
        mode: CudnnConvolutionMode,
        dataType: CudnnDataType,
    ) -> CudnnStatus;

    fn cudnnSetConvolutionGroupCount(
        convDesc: *mut ConvolutionDescriptorStruct,
        groupCount: i32,
    ) -> CudnnStatus;

    fn cudnnSetConvolutionMathType(
        convDesc: *mut ConvolutionDescriptorStruct,
        math_type: CudnnMathType,
    ) -> CudnnStatus;

}


#[inline]
pub fn cudnn_create(handle: *mut*mut CudnnStruct) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnCreate(handle) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnCreate(handle) };
    }
}
#[inline]
pub fn cudnn_destroy(handle: *mut CudnnStruct) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnDestroy(handle) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnDestroy(handle) };
    }
}
#[inline]
pub fn cudnn_create_tensor_descriptor(tensor_desc: *mut*mut TensorDescriptorStruct) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnCreateTensorDescriptor(tensor_desc) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnCreateTensorDescriptor(tensor_desc) };
    }
}
#[inline]
pub fn cudnn_destroy_tensor_descriptor(tensor_desc: *mut TensorDescriptorStruct) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnDestroyTensorDescriptor(tensor_desc) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnDestroyTensorDescriptor(tensor_desc) };
    }
}
#[inline]
pub fn cudnn_set_tensor_nd_descriptor(tensor_desc: *mut TensorDescriptorStruct, data_type: CudnnDataType, nb_dims: i32, dim_a: *const i32, stride_a: *const i32) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnSetTensorNdDescriptor(tensor_desc, data_type, nb_dims, dim_a, stride_a) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnSetTensorNdDescriptor(tensor_desc, data_type, nb_dims, dim_a, stride_a) };
    }
}
#[inline]
pub fn cudnn_get_tensor_nd_descriptor(tensor_desc: *mut TensorDescriptorStruct, nb_dims_requested: i32, data_type: *mut CudnnDataType, nb_dims: *mut i32, dim_a: *mut i32, stride_a: *mut i32) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnGetTensorNdDescriptor(tensor_desc, nb_dims_requested, data_type, nb_dims, dim_a, stride_a) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnGetTensorNdDescriptor(tensor_desc, nb_dims_requested, data_type, nb_dims, dim_a, stride_a) };
    }
}
#[inline]
pub fn cudnn_get_tensor_size_in_bytes(tensor_desc: *mut TensorDescriptorStruct, size: &mut usize) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnGetTensorSizeInBytes(tensor_desc, size) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnGetTensorSizeInBytes(tensor_desc, size) };
    }
}
#[inline]
pub fn cudnn_set_tensor(handle: *mut CudnnStruct, y_desc: *const TensorDescriptorStruct, y: *mut c_void, value_ptr: *const c_void) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnSetTensor(handle, y_desc, y, value_ptr) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnSetTensor(handle, y_desc, y, value_ptr) };
    }
}
#[inline]
pub fn cudnn_create_reduce_tensor_descriptor(reduce_tensor_desc: *mut*mut ReduceTensorDescriptorStruct) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnCreateReduceTensorDescriptor(reduce_tensor_desc) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnCreateReduceTensorDescriptor(reduce_tensor_desc) };
    }
}
#[inline]
pub fn cudnn_destroy_reduce_tensor_descriptor(reduce_tensor_desc: *mut ReduceTensorDescriptorStruct) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnDestroyReduceTensorDescriptor(reduce_tensor_desc) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnDestroyReduceTensorDescriptor(reduce_tensor_desc) };
    }
}
#[inline]
pub fn cudnn_set_reduce_tensor_descriptor(reduce_tensor_desc: *mut ReduceTensorDescriptorStruct, reduce_tensor_op: CudnnReduceTensorOp, reduce_tensor_comp_type: CudnnDataType, reduce_tensor_nan_opt: CudnnNanPropagation, reduce_tensor_indices: CudnnReduceTensorIndices, reduce_tensor_indices_type: CudnnIndicesType) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnSetReduceTensorDescriptor(reduce_tensor_desc, reduce_tensor_op,
                                                reduce_tensor_comp_type, reduce_tensor_nan_opt,
                                                reduce_tensor_indices, reduce_tensor_indices_type) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnSetReduceTensorDescriptor(reduce_tensor_desc, reduce_tensor_op,
                                                reduce_tensor_comp_type, reduce_tensor_nan_opt,
                                                reduce_tensor_indices, reduce_tensor_indices_type) };
    }
}
#[inline]
pub fn cudnn_get_reduce_tensor_descriptor(reduce_tensor_desc: *const ReduceTensorDescriptorStruct, reduce_tensor_op: &mut CudnnReduceTensorOp, reduce_tensor_comp_type: &mut CudnnDataType, reduce_tensor_nan_opt: &mut CudnnNanPropagation, reduce_tensor_indices: &mut CudnnReduceTensorIndices, reduce_tensor_indices_type: &mut CudnnIndicesType) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnGetReduceTensorDescriptor(reduce_tensor_desc, reduce_tensor_op,
                                                reduce_tensor_comp_type, reduce_tensor_nan_opt,
                                                reduce_tensor_indices, reduce_tensor_indices_type) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnGetReduceTensorDescriptor(reduce_tensor_desc, reduce_tensor_op,
                                                reduce_tensor_comp_type, reduce_tensor_nan_opt,
                                                reduce_tensor_indices, reduce_tensor_indices_type) };
    }
}
#[inline]
pub fn cudnn_create_activation_descriptor(activation_desc: *mut*mut ActivationDescriptorStruct) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnCreateActivationDescriptor(activation_desc) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnCreateActivationDescriptor(activation_desc) };
    }
}
#[inline]
pub fn cudnn_destroy_activation_descriptor(activation_desc: *mut ActivationDescriptorStruct) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnDestroyActivationDescriptor(activation_desc) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnDestroyActivationDescriptor(activation_desc) };
    }
}
#[inline]
pub fn cudnn_set_activation_descriptor(activation_desc: *mut ActivationDescriptorStruct, mode: CudnnActivationMode, relu_nan_opt: CudnnNanPropagation, coef: f64) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnSetActivationDescriptor(activation_desc, mode, relu_nan_opt, coef) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnSetActivationDescriptor(activation_desc, mode, relu_nan_opt, coef) };
    }
}
#[inline]
pub fn cudnn_get_activation_descriptor(activation_desc: *const ActivationDescriptorStruct, mode: *mut CudnnActivationMode, relu_nan_opt: *mut CudnnNanPropagation, coef: *mut f64) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnGetActivationDescriptor(activation_desc, mode, relu_nan_opt, coef) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnGetActivationDescriptor(activation_desc, mode, relu_nan_opt, coef) };
    }
}
#[inline]
pub fn cudnn_activation_forward(handle: *const CudnnStruct, activation_desc: *const ActivationDescriptorStruct, alpha: *const c_void, x_desc: *const TensorDescriptorStruct, x: *const c_void, beta: *const c_void, y_desc: *const TensorDescriptorStruct, y: *mut c_void) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnActivationForward(handle, activation_desc, alpha, x_desc, x, beta, y_desc, y) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnActivationForward(handle, activation_desc, alpha, x_desc, x, beta, y_desc, y) };
    }
}
#[inline]
pub fn cudnn_activation_backward(handle: *const CudnnStruct, activation_desc: *const ActivationDescriptorStruct, alpha: *const c_void, y_desc: *const TensorDescriptorStruct, y: *const c_void, dy_desc: *const TensorDescriptorStruct, dy: *const c_void, beta: *const c_void, x_desc: *const TensorDescriptorStruct, x: *mut c_void) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnActivationBackward(handle, activation_desc, alpha, y_desc, y, dy_desc, dy, beta, x_desc, x) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnActivationBackward(handle, activation_desc, alpha, y_desc, y, dy_desc, dy, beta, x_desc, x) };
    }
}

#[inline]
pub fn cudnn_create_convolution_descriptor(conv_desc: *mut*mut ConvolutionDescriptorStruct) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnCreateConvolutionDescriptor(conv_desc) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnCreateConvolutionDescriptor(conv_desc) };
    }
}
#[inline]
pub fn cudnn_destroy_convolution_descriptor(conv_desc: *mut ConvolutionDescriptorStruct) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnDestroyConvolutionDescriptor(conv_desc) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnDestroyConvolutionDescriptor(conv_desc) };
    }
}
#[inline]
pub fn cudnn_set_convolution_nd_descriptor(conv_desc: *mut ConvolutionDescriptorStruct, array_len: i32, pads: *const i32, filters_stride: *const i32, dilatations: *const i32, mode: CudnnConvolutionMode, data_type: CudnnDataType) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnSetConvolutionNdDescriptor(conv_desc, array_len, pads, filters_stride, dilatations, mode, data_type) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnSetConvolutionNdDescriptor(conv_desc, array_len, pads, filters_stride, dilatations, mode, data_type) };
    }
}
#[inline]
pub fn cudnn_set_convolution_group_count(conv_desc: *mut ConvolutionDescriptorStruct, group_count: i32) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnSetConvolutionGroupCount(conv_desc, group_count) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnSetConvolutionGroupCount(conv_desc, group_count) };
    }
}
#[inline]
pub fn cudnn_set_convolution_math_type(conv_desc: *mut ConvolutionDescriptorStruct, math_type: CudnnMathType) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnSetConvolutionMathType(conv_desc, math_type) }.assert_success()
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnSetConvolutionMathType(conv_desc, math_type) };
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use std::ptr;

    #[test]
    fn create_destroy() {
        let mut handle = ptr::null_mut();
        unsafe {
            cudnnCreate(&mut handle).assert_success();
            cudnnDestroy(handle).assert_success();
        }
    }

    #[test]
    fn tensor_desc_create_destroy() {
        let mut tensor_desc = ptr::null_mut();
        unsafe {
            cudnnCreateTensorDescriptor(&mut tensor_desc).assert_success();
            cudnnDestroyTensorDescriptor(tensor_desc).assert_success();
        }
    }

    fn get_fully_packed_strides(dims: &[i32]) -> Vec<i32> {
        use std::collections::VecDeque;
        let mut output = VecDeque::with_capacity(dims.len());
        output.push_front(1);
        let mut last = 1;
        for i in (1..dims.len()).rev() {
            last *= dims[i];
            output.push_front(last);
        }
        Vec::from(output)
    }

    #[test]
    fn tensor_desc_set_get() {
        let mut tensor_desc = ptr::null_mut();
        unsafe {
            let input_dima = [2, 3, 5, 4, 1];
            let input_stridea = get_fully_packed_strides(&input_dima);

            cudnnCreateTensorDescriptor(&mut tensor_desc).assert_success();
            cudnnSetTensorNdDescriptor(tensor_desc, CudnnDataType::Float, 5,
                                       input_dima.as_ptr(), input_stridea.as_ptr()).assert_success();
            let mut nb_dims = 0;
            let mut data_type = CudnnDataType::Int8x4;
            let mut dim_a = [0; 5];
            let mut stride_a = [0; 5];
            cudnnGetTensorNdDescriptor(tensor_desc, 4, &mut data_type,
                                       &mut nb_dims, dim_a.as_mut_ptr(), stride_a.as_mut_ptr()).assert_success();

            let mut size = 0;
            cudnnGetTensorSizeInBytes(tensor_desc, &mut size).assert_success();
            println!("size = {}", size);

            cudnnDestroyTensorDescriptor(tensor_desc).assert_success();
            assert_eq!(data_type, CudnnDataType::Float);
            assert_eq!(dim_a, [2, 3, 5, 4, 0]);
            assert_eq!(stride_a, [60, 20, 4, 1, 0]);
            assert_eq!(nb_dims, 5);
        }
    }

}