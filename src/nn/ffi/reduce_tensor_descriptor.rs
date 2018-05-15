
use super::{CudnnStatus, CudnnDataType, CudnnNanPropagation, CudnnReduceTensorOp, CudnnReduceTensorIndices, CudnnIndicesType};



pub enum _ReduceTensorDescriptorStruct {}




#[allow(non_snake_case)]
extern {

    fn cudnnCreateReduceTensorDescriptor(reduceTensorDesc: *mut*mut _ReduceTensorDescriptorStruct) -> CudnnStatus;

    fn cudnnDestroyReduceTensorDescriptor(reduceTensorDesc: *mut _ReduceTensorDescriptorStruct) -> CudnnStatus;

    fn cudnnSetReduceTensorDescriptor(
        reduceTensorDesc: *mut _ReduceTensorDescriptorStruct,
        reduceTensorOp: CudnnReduceTensorOp,
        reduceTensorCompType: CudnnDataType,
        reduceTensorNanOpt: CudnnNanPropagation,
        reduceTensorIndices: CudnnReduceTensorIndices,
        reduceTensorIndicesType: CudnnIndicesType
    ) -> CudnnStatus;

    fn cudnnGetReduceTensorDescriptor(
        reduceTensorDesc: *const _ReduceTensorDescriptorStruct,
        reduceTensorOp: &mut CudnnReduceTensorOp,
        reduceTensorCompType: &mut CudnnDataType,
        reduceTensorNanOpt: &mut CudnnNanPropagation,
        reduceTensorIndices: &mut CudnnReduceTensorIndices,
        reduceTensorIndicesType: &mut CudnnIndicesType
    ) -> CudnnStatus;

}





#[inline]
pub fn cudnn_create_reduce_tensor_descriptor(reduce_tensor_desc: *mut*mut _ReduceTensorDescriptorStruct) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnCreateReduceTensorDescriptor(reduce_tensor_desc) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnCreateReduceTensorDescriptor(reduce_tensor_desc) };
    }
}

#[inline]
pub fn cudnn_destroy_reduce_tensor_descriptor(reduce_tensor_desc: *mut _ReduceTensorDescriptorStruct) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnDestroyReduceTensorDescriptor(reduce_tensor_desc) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnDestroyReduceTensorDescriptor(reduce_tensor_desc) };
    }
}

#[inline]
pub fn cudnn_set_reduce_tensor_descriptor(reduce_tensor_desc: *mut _ReduceTensorDescriptorStruct, reduce_tensor_op: CudnnReduceTensorOp, reduce_tensor_comp_type: CudnnDataType, reduce_tensor_nan_opt: CudnnNanPropagation, reduce_tensor_indices: CudnnReduceTensorIndices, reduce_tensor_indices_type: CudnnIndicesType) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnSetReduceTensorDescriptor(reduce_tensor_desc, reduce_tensor_op, reduce_tensor_comp_type, reduce_tensor_nan_opt, reduce_tensor_indices, reduce_tensor_indices_type) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnSetReduceTensorDescriptor(reduce_tensor_desc, reduce_tensor_op, reduce_tensor_comp_type, reduce_tensor_nan_opt, reduce_tensor_indices, reduce_tensor_indices_type) };
    }
}

#[inline]
pub fn cudnn_get_reduce_tensor_descriptor(reduce_tensor_desc: *const _ReduceTensorDescriptorStruct, reduce_tensor_op: &mut CudnnReduceTensorOp, reduce_tensor_comp_type: &mut CudnnDataType, reduce_tensor_nan_opt: &mut CudnnNanPropagation, reduce_tensor_indices: &mut CudnnReduceTensorIndices, reduce_tensor_indices_type: &mut CudnnIndicesType) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnGetReduceTensorDescriptor(reduce_tensor_desc, reduce_tensor_op, reduce_tensor_comp_type, reduce_tensor_nan_opt, reduce_tensor_indices, reduce_tensor_indices_type) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnGetReduceTensorDescriptor(reduce_tensor_desc, reduce_tensor_op, reduce_tensor_comp_type, reduce_tensor_nan_opt, reduce_tensor_indices, reduce_tensor_indices_type) };
    }
}



