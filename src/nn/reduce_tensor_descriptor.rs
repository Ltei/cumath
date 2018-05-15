
use super::ffi::*;
use std::ptr;
use std::marker::PhantomData;
use CuDataType;


pub struct CuReduceTensorDescriptor<T: CuDataType> {
    _phantom: PhantomData<T>,
    data: *mut _ReduceTensorDescriptorStruct,
}

impl<T: CuDataType> Drop for CuReduceTensorDescriptor<T> {
    fn drop(&mut self) {
        cudnn_destroy_reduce_tensor_descriptor(self.data)
    }
}

impl CuReduceTensorDescriptor<f32> {

    pub fn new(op: CudnnReduceTensorOp) -> CuReduceTensorDescriptor<f32> {
        let mut data = ptr::null_mut();
        cudnn_create_reduce_tensor_descriptor(&mut data);
        cudnn_set_reduce_tensor_descriptor(data, op,
                                           CudnnDataType::Float,
                                           CudnnNanPropagation::Propagate,
                                           CudnnReduceTensorIndices::NoIndices,
                                           CudnnIndicesType::Indices32bit);
        CuReduceTensorDescriptor { _phantom: PhantomData, data }
    }



}


#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn init() {
        let descriptor = CuReduceTensorDescriptor::new(CudnnReduceTensorOp::Max);
    }

}