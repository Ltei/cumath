
use super::ffi::*;
use std::ptr;
use std::marker::PhantomData;
use std::fmt::{self, Debug};
use CuDataType;


pub struct CuFilterDescriptor<T: CuDataType> {
    _phantom: PhantomData<T>,
    pub(crate) data: *mut _FilterDescriptorStruct,
}

impl<T: CuDataType> Drop for CuFilterDescriptor<T> {
    fn drop(&mut self) {
        cudnn_destroy_filter_descriptor(self.data)
    }
}

impl CuFilterDescriptor<f32> {

    pub fn new(format: CudnnTensorFormat, filter_dims: &[i32]) -> CuFilterDescriptor<f32> {
        let mut data = ptr::null_mut();
        cudnn_create_filter_descriptor(&mut data);
        cudnn_set_filter_nd_descriptor(data, CudnnDataType::Float, format, filter_dims.len() as i32, filter_dims.as_ptr());
        CuFilterDescriptor { _phantom: PhantomData, data }
    }

    pub fn new_4d(format: CudnnTensorFormat, k: i32, c: i32, h: i32, w: i32) -> CuFilterDescriptor<f32> {
        let mut data = ptr::null_mut();
        cudnn_create_filter_descriptor(&mut data);
        cudnn_set_filter_4d_descriptor(data, CudnnDataType::Float, format, k, c, h, w);
        CuFilterDescriptor { _phantom: PhantomData, data }
    }

    pub fn get_info(&self, nb_dims_requested: i32) -> CuFilterDescriptorInfo {
        let mut data_type = CudnnDataType::Int8x4;
        let mut format = CudnnTensorFormat::Nchw;
        let mut nb_dims = -1;
        let mut filter_dims = vec![-1; nb_dims_requested as usize];
        cudnn_get_filter_nd_descriptor(self.data, nb_dims_requested, &mut data_type, &mut format, &mut nb_dims, filter_dims.as_mut_ptr());
        CuFilterDescriptorInfo { data_type, format, nb_dims, filter_dims }
    }

}


pub struct CuFilterDescriptorInfo {
    pub data_type: CudnnDataType,
    pub format: CudnnTensorFormat,
    pub nb_dims: i32,
    pub filter_dims: Vec<i32>,
}

impl Debug for CuFilterDescriptorInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Data type:{:?}, Format = {:?}, Nb dims:{}, Filter dims:{:?}", self.data_type, self.format, self.nb_dims, self.filter_dims)
    }
}


#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn init() {
        let descriptor = CuFilterDescriptor::<f32>::new(CudnnTensorFormat::Nchw, &[2, 2, 2, 2]);
    }

}