
use std::mem::size_of;


use super::*;
use tags::*;
use ffi::vectorkernel_ffi::*;


// CuMatrixSlice

pub struct CuMatrixSlice<'a, T> where T: CuMatrixOp + 'a {
    pub(super) parent: PhantomData<&'a T>,
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) len: usize,
    pub(super) ptr: *const f32,
}
impl<'a, T> CuMatrixOp for CuMatrixSlice<'a, T> where T: CuMatrixOp + 'a  {
    fn rows(&self) -> usize { self.rows }
    fn cols(&self) -> usize { self.cols }
    fn len(&self) -> usize { self.len }
    fn leading_dimension(&self) -> usize { self.rows }
    fn ptr(&self) -> *const f32 { self.ptr }

    fn clone_to_host(&self, output: &mut [f32]) {
        cuda_memcpy(output.as_mut_ptr(), self.ptr(), self.len() * size_of::<f32>(), CudaMemcpyKind::DeviceToHost);
    }
}
impl<'a, T> CuPacked for CuMatrixSlice<'a, T> where T: CuMatrixOp + 'a {}
impl<'a, T> CuMatrixSlice<'a, T> where T: CuMatrixOp + 'a {
    pub fn slice_col<'b>(&'b self, col_offset: usize, nb_cols: usize) -> CuMatrixSlice<'b, Self> {
        CuMatrixSlice {
            parent: PhantomData,
            rows: self.leading_dimension(),
            cols: nb_cols,
            len: self.leading_dimension()*nb_cols,
            ptr: unsafe { self.ptr().offset((col_offset*self.leading_dimension()) as isize) },
        }
    }
}


// CuMatrixSliceMut

pub struct CuMatrixSliceMut<'a, T> where T: CuMatrixOp + 'a {
    pub(super) parent: PhantomData<&'a T>,
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) len: usize,
    pub(super) ptr: *mut f32,
}
impl<'a, T> CuMatrixOp for CuMatrixSliceMut<'a, T> where T: CuMatrixOp + 'a  {
    fn rows(&self) -> usize { self.rows }
    fn cols(&self) -> usize { self.cols }
    fn len(&self) -> usize { self.len }
    fn leading_dimension(&self) -> usize { self.rows }
    fn ptr(&self) -> *const f32 { self.ptr }

    fn clone_to_host(&self, output: &mut [f32]) {
        cuda_memcpy(output.as_mut_ptr(), self.ptr(), self.len() * size_of::<f32>(), CudaMemcpyKind::DeviceToHost);
    }
}
impl<'a, T> CuMatrixOpMut for CuMatrixSliceMut<'a, T> where T: CuMatrixOp + 'a  {
    fn ptr_mut(&mut self) -> *mut f32 { self.ptr }

    fn clone_from_host(&mut self, data: &[f32]) {
        cuda_memcpy(self.ptr_mut(), data.as_ptr(), self.len() * size_of::<f32>(), CudaMemcpyKind::HostToDevice);
    }

    fn init(&mut self, value: f32) {
        unsafe { VectorKernel_init(self.ptr_mut(), self.len as i32, value) }
    }
    fn add_value_self(&mut self, value: f32) {
        unsafe { VectorKernel_addValue(self.ptr, self.ptr, self.len as i32, value) }
    }
    fn scale_self(&mut self, value: f32) {
        unsafe { VectorKernel_scl(self.ptr, self.ptr, self.len as i32, value) }
    }
}
impl <'a, T> CuPacked for CuMatrixSliceMut<'a, T> where T: CuMatrixOp + 'a {}
impl <'a, T> CuMatrixSliceMut<'a, T> where T: CuMatrixOp + 'a {
    pub fn slice_col<'b>(&'b self, col_offset: usize, nb_cols: usize) -> CuMatrixSlice<'b, Self> {
        CuMatrixSlice {
            parent: PhantomData,
            rows: self.leading_dimension(),
            cols: nb_cols,
            len: self.leading_dimension()*nb_cols,
            ptr: unsafe { self.ptr().offset((col_offset*self.leading_dimension()) as isize) },
        }
    }
    pub fn slice_col_mut<'b>(&'b mut self, col_offset: usize, nb_cols: usize) -> CuMatrixSliceMut<'b, Self> {
        CuMatrixSliceMut {
            parent: PhantomData,
            rows: self.leading_dimension(),
            cols: nb_cols,
            len: self.leading_dimension()*nb_cols,
            ptr: unsafe { self.ptr_mut().offset((col_offset*self.leading_dimension()) as isize) },
        }
    }
}