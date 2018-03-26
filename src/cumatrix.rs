
use std;
use std::ops::{Deref, DerefMut};
use libc::c_void;

use ffi::cuda_ffi::*;
use ffi::kernel_ffi::*;

use assert::*;



pub struct CuMatrix {
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) len: usize,
    pub(super) data: *mut f32,
}
impl Drop for CuMatrix {
    fn drop(&mut self) {
        unsafe { cudaFree(self.data as *mut c_void) }.assert_success();
    }
}
impl CuMatrix {
    pub fn new(rows: usize, cols: usize, init_value: f32) -> CuMatrix {
        let len = rows*cols;
        let mut data = std::ptr::null_mut();
        unsafe { cudaMalloc(&mut data, len*std::mem::size_of::<f32>()) }.assert_success();
        unsafe { CudaKernel_vectorSet((data as *mut f32), (len as i32), init_value) }
        CuMatrix { rows, cols, len, data: (data as *mut f32) }
    }
    pub fn from_data(rows: usize, cols: usize, data: &[f32]) -> CuMatrix {
        assert_eq_usize(rows*cols, "rows*cols", data.len(), "data.len()");
        let mut output = {
            let len = rows*cols;
            let mut data = std::ptr::null_mut();
            unsafe { cudaMalloc(&mut data, len*std::mem::size_of::<f32>()) }.assert_success();
            CuMatrix { rows, cols, len, data: (data as *mut f32) }
        };
        output.copy_from_host(data);
        output
    }

    #[inline]
    pub fn len(&self) -> usize {
        return self.len;
    }
    #[inline]
    pub fn rows(&self) -> usize {
        return self.rows;
    }
    #[inline]
    pub fn cols(&self) -> usize {
        return self.cols;
    }
    pub fn slice(&self, col_offset: usize, nb_cols: usize) -> CuMatrixSlice {
        assert!(col_offset < self.cols);
        assert!(nb_cols <= self.cols-col_offset);
        CuMatrixSlice {
            _parent: self,
            value: CuMatrix {
                rows: self.rows,
                cols: nb_cols,
                len: self.rows*nb_cols,
                data: unsafe { self.data.offset((col_offset * self.rows) as isize) } }
        }
    }
    pub fn slice_mut(&mut self, col_offset: usize, nb_cols: usize) -> CuMatrixSliceMut {
        assert!(col_offset < self.cols);
        assert!(nb_cols <= self.cols-col_offset);
        CuMatrixSliceMut {
            _parent: self,
            value: CuMatrix {
                rows: self.rows,
                cols: nb_cols,
                len: self.rows*nb_cols,
                data: unsafe { self.data.offset((col_offset * self.rows) as isize) } }
        }
    }

    pub fn init(&mut self, value: f32) {
        unsafe { CudaKernel_vectorSet((self.data as *mut f32), (self.len as i32), value) }
    }
    pub fn copy_from_host(&mut self, data: &[f32]) {
        unsafe {
            cudaMemcpy((self.data as *mut c_void), (data.as_ptr() as *const c_void), self.len*std::mem::size_of::<f32>(), CudaMemcpyKind::HostToDevice)
        }.assert_success();
    }
    pub fn copy_to_host(&self, data: &mut [f32]) {
        unsafe {
            cudaMemcpy((data.as_mut_ptr() as *mut c_void), (self.data as *mut c_void), self.len*std::mem::size_of::<f32>(), CudaMemcpyKind::DeviceToHost)
        }.assert_success();
    }

    pub fn copy_from_device(&mut self, source: &CuMatrix) {
        assert_eq_usize(self.rows, "self.rows()", source.rows, "source.rows()");
        assert_eq_usize(self.cols, "self.cols()", source.cols, "source.cols()");
        unsafe {
            cudaMemcpy((self.data as *mut c_void), (source.data as *mut c_void),
                       self.len*std::mem::size_of::<f32>(), CudaMemcpyKind::DeviceToDevice)
        }.assert_success();
    }

    pub fn add_scl_self(&mut self, value: f32) {
        unsafe { CudaKernel_vectorAddSclSelf((self.data as *mut f32), (self.len as i32), value) }
    }
    pub fn scale_self(&mut self, value: f32) {
        unsafe { CudaKernel_vectorScaleSelf((self.data as *mut f32), (self.len as i32), value) }
    }
    pub fn add_self(&mut self, to_add: &Self) {
        unsafe { CudaKernel_vectorAdd(self.data, to_add.data, self.data, self.len as i32) }
    }

    #[allow(dead_code)]
    pub fn dev_print(&self, msg: &str) {
        let mut buffer = vec![0.0; self.len];
        self.copy_to_host(&mut buffer);
        println!("{}", msg);
        for row in 0..self.rows {
            for col in 0..self.cols {
                print!("{:.5}, ", buffer[row+col*self.rows])
            }
            println!()
        }
    }
}

pub struct CuMatrixSlice<'a> {
    _parent: &'a CuMatrix,
    value: CuMatrix,
}
impl<'a> Drop for CuMatrixSlice<'a> {
    fn drop(&mut self) {
        self.value.data = std::ptr::null_mut();
    }
}
impl<'a> Deref for CuMatrixSlice<'a> {
    type Target = CuMatrix;
    fn deref(&self) -> &CuMatrix {
        &self.value
    }
}

pub struct CuMatrixSliceMut<'a> {
    _parent: &'a CuMatrix,
    value: CuMatrix,
}
impl<'a> Drop for CuMatrixSliceMut<'a> {
    fn drop(&mut self) {
        self.value.data = std::ptr::null_mut();
    }
}
impl<'a> Deref for CuMatrixSliceMut<'a> {
    type Target = CuMatrix;
    fn deref(&self) -> &CuMatrix {
        &self.value
    }
}
impl<'a> DerefMut for CuMatrixSliceMut<'a> {
    fn deref_mut(&mut self) -> &mut CuMatrix {
        &mut self.value
    }
}



#[cfg(test)]
mod tests {
    fn assert_equals_float(a: f32, b: f32) {
        let d = a-b;
        if d < -0.000001 || d > 0.000001 {
            panic!("{} != {}", a, b);
        }
    }

    #[test]
    fn init() {
        let value0 = -0.23254;
        let value1 = 1.1852;
        let mut matrix = super::CuMatrix::new(2, 4, 0.0);

        matrix.init(value0);
        matrix.slice_mut(1, 2).init(value1);

        let mut output = vec![0.0; 8];
        matrix.copy_to_host(&mut output);

        assert_equals_float(output[0], value0);
        assert_equals_float(output[1], value0);
        assert_equals_float(output[2], value1);
        assert_equals_float(output[3], value1);
        assert_equals_float(output[4], value1);
        assert_equals_float(output[5], value1);
        assert_equals_float(output[6], value0);
        assert_equals_float(output[7], value0);
    }
    #[test]
    fn add_self() {
        let value0 = -0.23254;
        let value1 = 1.185254;
        let mut matrix0 = super::CuMatrix::new(2, 4, 0.0);
        let mut matrix1 = super::CuMatrix::new(2, 2, 0.0);

        matrix0.init(value0);
        matrix1.init(value1);
        matrix0.slice_mut(1, 2).add_self(&matrix1);

        let mut output = vec![0.0; 8];
        matrix0.copy_to_host(&mut output);

        assert_equals_float(output[0], value0);
        assert_equals_float(output[1], value0);
        assert_equals_float(output[2], value0+value1);
        assert_equals_float(output[3], value0+value1);
        assert_equals_float(output[4], value0+value1);
        assert_equals_float(output[5], value0+value1);
        assert_equals_float(output[6], value0);
        assert_equals_float(output[7], value0);
    }

}