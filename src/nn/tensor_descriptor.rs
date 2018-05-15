
use super::ffi::*;
use super::*;
use std::{ptr, marker::PhantomData, fmt::{self, Debug}, mem::size_of};
use CuDataType;
use CuVectorDeref;




pub struct CuTensorDescriptor<T: CuDataType> {
    _phantom: PhantomData<T>,
    pub(crate) data: *mut _TensorDescriptorStruct,
    nb_dimensions: i32,
}

impl<T: CuDataType> Drop for CuTensorDescriptor<T> {
    fn drop(&mut self) {
        cudnn_destroy_tensor_descriptor(self.data);
    }
}

impl<T: CuDataType> Clone for CuTensorDescriptor<T> {
    fn clone(&self) -> CuTensorDescriptor<T> {
        let info = self.get_info();
        let mut data = ptr::null_mut();
        cudnn_create_tensor_descriptor(&mut data);
        cudnn_set_tensor_nd_descriptor(data, info.data_type, info.nb_dims, info.dimensions.as_ptr(), info.strides.as_ptr());
        CuTensorDescriptor {
            _phantom: PhantomData,
            data,
            nb_dimensions: self.nb_dimensions,
        }
    }
    fn clone_from(&mut self, other: &CuTensorDescriptor<T>) {
        let info = other.get_info();
        cudnn_set_tensor_nd_descriptor(self.data, info.data_type, info.nb_dims, info.dimensions.as_ptr(), info.strides.as_ptr());
        self.nb_dimensions = other.nb_dimensions;
    }
}

impl<T: CuDataType> CuTensorDescriptor<T> {

    pub fn link<'a>(&'a self, data: &'a CuVectorDeref<T>) -> CuTensor<'a, T> {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq!(data.len(), self.data_len(), "data.len() != self.data_len()");
        }
        CuTensor { deref: CuTensorDeref { descriptor: self, data: data.ptr } }
    }

    pub fn link_mut<'a>(&'a self, data: &'a mut CuVectorDeref<T>) -> CuTensorMut<'a, T> {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq!(data.len(), self.data_len(), "data.len() != self.data_len()");
        }
        CuTensorMut { deref: CuTensorDeref { descriptor: self, data: data.ptr } }
    }

    pub fn data_len(&self) -> usize {
        let mut output = 0;
        cudnn_get_tensor_size_in_bytes(self.data, &mut output);
        output / size_of::<T>()
    }

    pub fn get_info(&self) -> CuTensorDescriptorInfo {
        let mut data_type = CudnnDataType::Int8x4;
        let mut nb_dims = -1;
        let mut dimensions = vec![-1; self.nb_dimensions as usize];
        let mut strides = vec![-1; self.nb_dimensions as usize];
        cudnn_get_tensor_nd_descriptor(self.data, self.nb_dimensions,
                                       &mut data_type, &mut nb_dims, dimensions.as_mut_ptr(), strides.as_mut_ptr());
        CuTensorDescriptorInfo { data_type, nb_dims, dimensions, strides }
    }

}

impl CuTensorDescriptor<f32> {

    pub fn new(dimensions: &[i32]) -> CuTensorDescriptor<f32> {
        #[cfg(not(feature = "disable_checks"))] {
            if dimensions.len() < 3 { panic!("dimensions.len() must be >= 3") }
        }
        let mut data = ptr::null_mut();
        cudnn_create_tensor_descriptor(&mut data);
        cudnn_set_tensor_nd_descriptor(data, CudnnDataType::Float, dimensions.len() as i32,
                                       dimensions.as_ptr(), get_fully_packed_strides(dimensions).as_ptr());
        CuTensorDescriptor {
            _phantom: PhantomData,
            data,
            nb_dimensions: dimensions.len() as i32,
        }
    }

    pub fn new_4d(format: CudnnTensorFormat, n: i32, c: i32, h: i32, w: i32) -> CuTensorDescriptor<f32> {
        let mut data = ptr::null_mut();
        cudnn_create_tensor_descriptor(&mut data);
        cudnn_set_tensor4d_descriptor(data, format, CudnnDataType::Float, n, c, h, w);
        CuTensorDescriptor {
            _phantom: PhantomData,
            data,
            nb_dimensions: 4,
        }
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


pub struct CuTensorDescriptorInfo {
    pub data_type: CudnnDataType,
    pub nb_dims: i32,
    pub dimensions: Vec<i32>,
    pub strides: Vec<i32>,
}

impl Debug for CuTensorDescriptorInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Data type:{:?}, Nb dims:{}, Dimensions:{:?}, Strides:{:?}", self.data_type, self.nb_dims, self.dimensions, self.strides)
    }
}