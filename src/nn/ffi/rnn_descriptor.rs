
use super::{CudnnStatus, CudnnRNNInputMode, CudnnRNNAlgo, CudnnRNNMode, CudnnDirectionMode, CudnnDataType};
use super::cudnn::_CudnnStruct;
use super::dropout_descriptor::_DropoutDescriptorStruct;



pub enum _RNNDescriptorStruct {}


#[allow(non_snake_case)]
extern {

    fn cudnnCreateRNNDescriptor(rnnDesc: *mut*mut _RNNDescriptorStruct) -> CudnnStatus;

    fn cudnnDestroyRNNDescriptor(rnnDesc: *mut _RNNDescriptorStruct) -> CudnnStatus;

    fn cudnnSetRNNDescriptor(
        handle: *mut _CudnnStruct,
        rnnDesc: *mut _RNNDescriptorStruct,
        hiddenSize: i32,
        numLayers: i32,
        dropoutDesc: *const _DropoutDescriptorStruct,
        inputMode: CudnnRNNInputMode,
        direction: CudnnDirectionMode,
        mode: CudnnRNNMode,
        algo: CudnnRNNAlgo,
        dataType: CudnnDataType,
    ) -> CudnnStatus;

    fn cudnnGetRNNDescriptor(
        handle: *mut _CudnnStruct,
        rnnDesc: *const _RNNDescriptorStruct,
        hiddenSize: *mut i32,
        numLayers: *mut i32,
        dropoutDesc: *mut _DropoutDescriptorStruct,
        inputMode: *mut CudnnRNNInputMode,
        direction: *mut CudnnDirectionMode,
        mode: *mut CudnnRNNMode,
        algo: *mut CudnnRNNAlgo,
        dataType: *mut CudnnDataType,
    ) -> CudnnStatus;

}






#[inline]
pub fn cudnn_create_rnn_descriptor(rnn_desc: *mut*mut _RNNDescriptorStruct) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnCreateRNNDescriptor(rnn_desc) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnCreateRNNDescriptor(rnn_desc) };
    }
}

#[inline]
pub fn cudnn_destroy_rnn_descriptor(rnn_desc: *mut _RNNDescriptorStruct) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnDestroyRNNDescriptor(rnn_desc) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnDestroyRNNDescriptor(rnn_desc) };
    }
}

#[inline]
pub fn cudnn_set_rnn_descriptor(handle: *mut _CudnnStruct, rnn_desc: *mut _RNNDescriptorStruct, hidden_size: i32, num_layers: i32, dropout_desc: *const _DropoutDescriptorStruct, input_mode: CudnnRNNInputMode, direction: CudnnDirectionMode, mode: CudnnRNNMode, algo: CudnnRNNAlgo, data_type: CudnnDataType) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnSetRNNDescriptor(handle, rnn_desc, hidden_size, num_layers, dropout_desc, input_mode, direction, mode, algo, data_type) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnSetRNNDescriptor(handle, rnn_desc, hidden_size, num_layers, dropout_desc, input_mode, direction, mode, algo, data_type) };
    }
}

#[inline]
pub fn cudnn_get_rnn_descriptor(handle: *mut _CudnnStruct, rnn_desc: *const _RNNDescriptorStruct, hidden_size: *mut i32, num_layers: *mut i32, dropout_desc: *mut _DropoutDescriptorStruct, input_mode: *mut CudnnRNNInputMode, direction: *mut CudnnDirectionMode, mode: *mut CudnnRNNMode, algo: *mut CudnnRNNAlgo, data_type: *mut CudnnDataType) {
    #[cfg(not(feature = "disable_checks"))] {
        unsafe { cudnnGetRNNDescriptor(handle, rnn_desc, hidden_size, num_layers, dropout_desc, input_mode, direction, mode, algo, data_type) }.assert_success();
    }
    #[cfg(feature = "disable_checks")] {
        unsafe { cudnnGetRNNDescriptor(handle, rnn_desc, hidden_size, num_layers, dropout_desc, input_mode, direction, mode, algo, data_type) };
    }
}



