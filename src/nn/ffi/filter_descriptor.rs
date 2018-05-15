
use super::{CudnnStatus, CudnnDataType, CudnnTensorFormat};



pub enum _FilterDescriptorStruct {}



#[allow(non_snake_case)]
extern {

    fn cudnnCreateFilterDescriptor(filterDesc: *mut*mut _FilterDescriptorStruct) -> CudnnStatus;

    fn cudnnDestroyFilterDescriptor(filterDesc: *mut _FilterDescriptorStruct) -> CudnnStatus;

    fn cudnnSetFilterNdDescriptor(
        filterDesc: *mut _FilterDescriptorStruct,
        dataType: CudnnDataType,
        format: CudnnTensorFormat,
        nbDims: i32,
        filterDimA: *const i32
    ) -> CudnnStatus;

    fn cudnnSetFilter4dDescriptor(
        filterDesc: *mut _FilterDescriptorStruct,
        dataType: CudnnDataType,
        format: CudnnTensorFormat,
        k: i32,
        c: i32,
        h: i32,
        w: i32,
    ) -> CudnnStatus;

    fn cudnnGetFilterNdDescriptor(
        filterDesc: *const _FilterDescriptorStruct,
        nbDimsRequested: i32,
        dataType: *mut CudnnDataType,
        format: *mut CudnnTensorFormat,
        nbDims: *mut i32,
        filterDimA: *mut i32,
    ) -> CudnnStatus;

}





#[inline]
pub fn cudnn_create_filter_descriptor(filter_desc: *mut*mut _FilterDescriptorStruct) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnCreateFilterDescriptor(filter_desc) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnCreateFilterDescriptor(filter_desc) };
    }
}

#[inline]
pub fn cudnn_destroy_filter_descriptor(filter_desc: *mut _FilterDescriptorStruct) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnDestroyFilterDescriptor(filter_desc) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnDestroyFilterDescriptor(filter_desc) };
    }
}

#[inline]
pub fn cudnn_set_filter_nd_descriptor(filter_desc: *mut _FilterDescriptorStruct, data_type: CudnnDataType, format: CudnnTensorFormat, nb_dims: i32, filter_dim_a: *const i32) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnSetFilterNdDescriptor(filter_desc, data_type, format, nb_dims, filter_dim_a) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnSetFilterNdDescriptor(filter_desc, data_type, format, nb_dims, filter_dim_a) };
    }
}

#[inline]
pub fn cudnn_set_filter4d_descriptor(filter_desc: *mut _FilterDescriptorStruct, data_type: CudnnDataType, format: CudnnTensorFormat, k: i32, c: i32, h: i32, w: i32) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnSetFilter4dDescriptor(filter_desc, data_type, format, k, c, h, w) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnSetFilter4dDescriptor(filter_desc, data_type, format, k, c, h, w) };
    }
}

#[inline]
pub fn cudnn_get_filter_nd_descriptor(filter_desc: *const _FilterDescriptorStruct, nb_dims_requested: i32, data_type: *mut CudnnDataType, format: *mut CudnnTensorFormat, nb_dims: *mut i32, filter_dim_a: *mut i32) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnGetFilterNdDescriptor(filter_desc, nb_dims_requested, data_type, format, nb_dims, filter_dim_a) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnGetFilterNdDescriptor(filter_desc, nb_dims_requested, data_type, format, nb_dims, filter_dim_a) };
    }
}



