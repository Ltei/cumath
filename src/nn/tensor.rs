
use super::ffi::*;
use super::*;
use std::ops::{Deref, DerefMut};
use std::os::raw::c_void;
use CuDataType;




pub struct CuTensorDeref<'a, T: CuDataType + 'a> {
    pub(crate) descriptor: &'a CuTensorDescriptor<T>,
    pub(crate) data: *mut T,
}

impl<'a, T: CuDataType + 'a> CuTensorDeref<'a, T> {

    pub fn descriptor(&self) -> &CuTensorDescriptor<T> {
        self.descriptor
    }

    pub fn init(&mut self, cudnn: &Cudnn, value: T) {
        cudnn_set_tensor(cudnn.handle, self.descriptor.data, self.data as *mut c_void, &value as *const T as *const c_void);
    }

}


pub struct CuTensor<'a, T: CuDataType + 'a> {
    pub(crate) deref: CuTensorDeref<'a, T>,
}
impl<'a, T: CuDataType + 'a> Deref for CuTensor<'a, T> {
    type Target = CuTensorDeref<'a, T>;
    fn deref(&self) -> &CuTensorDeref<'a, T> { &self.deref }
}

pub struct CuTensorMut<'a, T: CuDataType + 'a> {
    pub(crate) deref: CuTensorDeref<'a, T>,
}
impl<'a, T: CuDataType + 'a> Deref for CuTensorMut<'a, T> {
    type Target = CuTensorDeref<'a, T>;
    fn deref(&self) -> &CuTensorDeref<'a, T> { &self.deref }
}
impl<'a, T: CuDataType + 'a> DerefMut for CuTensorMut<'a, T> {
    fn deref_mut(&mut self) -> &mut CuTensorDeref<'a, T> { &mut self.deref }
}
