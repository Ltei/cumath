
use std::ptr;
use std::marker::PhantomData;
use std::mem::size_of;
use libc::c_void;

use ffi::cuda_ffi::*;
use ffi::vectorkernel_ffi::*;
use ffi::matrixkernel_ffi::*;

use meta_tags::*;

use assert::*;


// Matrix traits

pub trait CuMatrixOp: Sized {
    #[inline]
    fn rows(&self) -> usize;
    #[inline]
    fn cols(&self) -> usize;
    #[inline]
    fn len(&self) -> usize;
    #[inline]
    fn leading_dimension(&self) -> usize;
    #[inline]
    fn ptr(&self) -> *const f32;

    fn slice<'a>(&'a self, row_offset: usize, col_offset: usize, nb_rows: usize, nb_cols: usize) -> CuSubMatrix<'a, Self> {
        CuSubMatrix {
            parent: PhantomData,
            rows: nb_rows,
            cols: nb_cols,
            leading_dimension: self.leading_dimension(),
            ptr: unsafe { self.ptr().offset((row_offset + col_offset*self.leading_dimension()) as isize) },
        }
    }

    fn clone_to_host(&self, data: &mut [f32]) {
        assert_eq_usize(self.len(), "self.len()", data.len(), "data.len()");
        unsafe {
            cudaMemcpy2D(data.as_ptr() as *mut c_void,
                         self.rows() * size_of::<f32>(),
                         self.ptr() as *const c_void,
                         self.leading_dimension() * size_of::<f32>(),
                         self.rows() * size_of::<f32>(),
                         self.cols(),
                         CudaMemcpyKind::DeviceToHost);
        }
    }

    #[allow(dead_code)]
    fn dev_print(&self, msg: &str) {
        let mut buffer = vec![0.0; self.len()];
        self.clone_to_host(&mut buffer);
        println!("{}", msg);
        for row in 0..self.rows() {
            for col in 0..self.cols() {
                print!("{:.5}, ", buffer[row+col*self.rows()])
            }
            println!()
        }
    }
    fn assert_equals_float(a: f32, b: f32) {
        let d = a-b;
        if d < -0.000001 || d > 0.000001 {
            panic!("{} != {}", a, b);
        }
    }
    #[allow(dead_code)]
    fn dev_equals(&self, data: &[f32]) {
        assert_eq_usize(self.len(), "self.len()", data.len(), "data.len()");
        let mut buffer = vec![0.0; self.len()];
        self.clone_to_host(&mut buffer);
        let mut iter = buffer.iter();
        data.iter().for_each(|x| { Self::assert_equals_float(*x, *iter.next().unwrap()) });
    }
}
pub trait CuMatrixOpMut: CuMatrixOp  {
    #[inline]
    fn ptr_mut(&mut self) -> *mut f32;

    fn slice_mut<'a>(&'a mut self, row_offset: usize, col_offset: usize, nb_rows: usize, nb_cols: usize) -> CuSubMatrixMut<'a, Self> {
        CuSubMatrixMut {
            parent: PhantomData,
            rows: nb_rows,
            cols: nb_cols,
            leading_dimension: self.leading_dimension(),
            ptr: unsafe { self.ptr_mut().offset((row_offset + col_offset*self.leading_dimension()) as isize) },
        }
    }

    fn clone_from_host(&mut self, data: &[f32]) {
        assert_eq_usize(self.len(), "self.len()", data.len(), "data.len()");
        unsafe {
            cudaMemcpy2D(self.ptr_mut() as *mut c_void,
                         self.leading_dimension() * size_of::<f32>(),
                         data.as_ptr() as *const c_void,
                         self.rows() * size_of::<f32>(),
                         self.rows() * size_of::<f32>(),
                         self.cols(),
                         CudaMemcpyKind::HostToDevice);
        }
    }
    fn clone_from_device<DataT: CuMatrixOp>(&mut self, data: &DataT) {
        assert_eq_usize(self.len(), "self.len()", data.len(), "data.len()");
        unsafe {
            cudaMemcpy2D(self.ptr_mut() as *mut c_void,
                         self.leading_dimension() * size_of::<f32>(),
                         data.ptr() as *const c_void,
                         data.leading_dimension() * size_of::<f32>(),
                         self.rows() * size_of::<f32>(),
                         self.cols(),
                         CudaMemcpyKind::DeviceToDevice);
        }
    }

    fn init(&mut self, value: f32) {
        unsafe {
            MatrixKernel_init(self.ptr_mut(), self.leading_dimension() as i32,
                              self.rows() as i32, self.cols() as i32, value);
        }
    }

    fn add_value_self(&mut self, value: f32) {
        unsafe {
            MatrixKernel_addValue(self.ptr(), self.leading_dimension() as i32,
                                  self.ptr_mut(), self.leading_dimension() as i32,
                                  self.rows() as i32, self.cols() as i32, value);
        }
    }
    fn scale_self(&mut self, value: f32) {
        unsafe {
            MatrixKernel_scale(self.ptr(), self.leading_dimension() as i32,
                               self.ptr_mut(), self.leading_dimension() as i32,
                               self.rows() as i32, self.cols() as i32, value);
        }
    }
    fn add_self<ToAddT: CuMatrixOp>(&mut self, to_add: &ToAddT) {
        unsafe {
            MatrixKernel_add(self.ptr(), self.leading_dimension() as i32,
                             to_add.ptr(), to_add.leading_dimension() as i32,
                             self.ptr_mut(), self.leading_dimension() as i32,
                             self.rows() as i32, self.cols() as i32)
        }
    }
}


// CuMatrix

pub struct CuMatrix {
    rows: usize,
    cols: usize,
    len: usize,
    ptr: *mut f32,
}
impl Drop for CuMatrix {
    fn drop(&mut self) {
        unsafe { cudaFree(self.ptr as *mut c_void) }.assert_success();
    }
}
impl CuMatrixOp for CuMatrix {
    fn rows(&self) -> usize { self.rows }
    fn cols(&self) -> usize { self.cols }
    fn len(&self) -> usize { self.rows*self.cols }
    fn leading_dimension(&self) -> usize { self.rows }
    fn ptr(&self) -> *const f32 { self.ptr }

    fn clone_to_host(&self, output: &mut [f32]) {
        unsafe {
            cudaMemcpy(output.as_mut_ptr() as *mut c_void, self.ptr as *const c_void, self.len * size_of::<f32>(), CudaMemcpyKind::DeviceToHost)
        }.assert_success();
    }
}
impl CuMatrixOpMut for CuMatrix {
    fn ptr_mut(&mut self) -> *mut f32 { self.ptr }

    fn clone_from_host(&mut self, data: &[f32]) {
        unsafe {
            cudaMemcpy(self.ptr as *mut c_void, data.as_ptr() as *const c_void, self.len() * size_of::<f32>(), CudaMemcpyKind::HostToDevice)
        }.assert_success();
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
impl CuPacked for CuMatrix {}
impl CuMatrix {
    pub fn new(rows: usize, cols: usize, init_value: f32) -> CuMatrix {
        let len = rows*cols;
        let mut data = ptr::null_mut();
        unsafe { cudaMalloc(&mut data, len*size_of::<f32>()) }.assert_success();
        unsafe { VectorKernel_init((data as *mut f32), (len as i32), init_value) }
        CuMatrix {
            rows, cols, len,
            ptr: data as *mut f32,
        }
    }
    pub fn from_data(rows: usize, cols: usize, data: &[f32]) -> CuMatrix {
        assert_eq_usize(rows*cols, "rows*cols", data.len(), "data.len()");
        let mut output = {
            let len = rows*cols;
            let mut data = ptr::null_mut();
            unsafe { cudaMalloc(&mut data, len*size_of::<f32>()) }.assert_success();
            CuMatrix {
                rows, cols, len,
                ptr: data as *mut f32,
            }
        };
        output.clone_from_host(data);
        output
    }

    pub fn slice_col<'a>(&'a self, col_offset: usize, nb_cols: usize) -> CuMatrixSlice<'a, Self> {
        CuMatrixSlice {
            parent: PhantomData,
            rows: self.leading_dimension(),
            cols: nb_cols,
            len: self.leading_dimension()*nb_cols,
            ptr: unsafe { self.ptr.offset((col_offset*self.leading_dimension()) as isize) },
        }
    }
    pub fn slice_col_mut<'a>(&'a mut self, col_offset: usize, nb_cols: usize) -> CuMatrixSliceMut<'a, Self> {
        CuMatrixSliceMut {
            parent: PhantomData,
            rows: self.leading_dimension(),
            cols: nb_cols,
            len: self.leading_dimension()*nb_cols,
            ptr: unsafe { self.ptr.offset((col_offset*self.leading_dimension()) as isize) },
        }
    }
}

#[cfg(test)]
mod matrix {
    use super::{CuMatrix, CuMatrixOp, CuMatrixOpMut};

    #[test]
    fn getters() {
        let matrix = CuMatrix::new(4, 8, 0.0);
        assert_eq!(matrix.rows(), 4);
        assert_eq!(matrix.cols(), 8);
        assert_eq!(matrix.len(), 4*8);
        assert_eq!(matrix.leading_dimension(), 4);
    }

    #[test]
    fn send_receive() {
        let data = &[1.0, 2.1, -1.7, 8.3, 1.0, -0.2];
        let mut matrix1 = CuMatrix::new(3, 2, 0.0);
        matrix1.clone_from_host(data);
        let matrix2 = CuMatrix::from_data(3, 2, data);

        let output1 = &mut [0.0; 6];
        matrix1.clone_to_host(output1);
        let output2 = &mut [0.0; 6];
        matrix2.clone_to_host(output2);

        for i in 0..data.len() {
            assert_eq!(output1[i], data[i]);
            assert_eq!(output2[i], data[i]);
        }
    }

    #[test]
    fn init() {
        let value = -1.254;
        let mut matrix = CuMatrix::new(2, 3, 0.0);
        matrix.init(value);

        let output = &mut[0.0; 6];
        matrix.clone_to_host(output);

        output.iter().for_each(|x| assert_eq!(*x, value));
    }

    #[test]
    fn slice() {
        let data = &[0.0, 1.0, 2.0, 0.1, 1.1, 2.1];
        let matrix = super::CuMatrix::from_data(3, 2, data);

        matrix.dev_equals(data);
        matrix.slice(0, 0, 3, 2).dev_equals(data);
        matrix.slice(1, 0, 1, 2).dev_equals(&[1.0, 1.1]);
        matrix.slice(0, 1, 3, 1).dev_equals(&[0.1, 1.1, 2.1]);
        matrix.slice(1, 0, 2, 2).slice(0, 1, 1, 1).dev_equals(&[1.1]);
        matrix.slice_col(1, 1).slice(1, 0, 1, 1).dev_equals(&[1.1]);
    }

}


// CuSubMatrix

pub struct CuSubMatrix<'a, T> where T: CuMatrixOp + 'a {
    parent: PhantomData<&'a T>,
    rows: usize,
    cols: usize,
    leading_dimension: usize,
    ptr: *const f32,
}
impl<'a, T> CuMatrixOp for CuSubMatrix<'a, T> where T: CuMatrixOp + 'a  {
    fn rows(&self) -> usize { self.rows }
    fn cols(&self) -> usize { self.cols }
    fn len(&self) -> usize { self.rows*self.cols }
    fn leading_dimension(&self) -> usize { self.leading_dimension }
    fn ptr(&self) -> *const f32 { self.ptr }
}


// CuSubMatrixMut

pub struct CuSubMatrixMut<'a, T> where T: CuMatrixOp + 'a {
    parent: PhantomData<&'a T>,
    rows: usize,
    cols: usize,
    leading_dimension: usize,
    ptr: *mut f32,
}
impl<'a, T> CuMatrixOp for CuSubMatrixMut<'a, T> where T: CuMatrixOp + 'a  {
    fn rows(&self) -> usize { self.rows }
    fn cols(&self) -> usize { self.cols }
    fn len(&self) -> usize { self.rows*self.cols }
    fn leading_dimension(&self) -> usize { self.leading_dimension }
    fn ptr(&self) -> *const f32 { self.ptr }
}
impl<'a, T> CuMatrixOpMut for CuSubMatrixMut<'a, T> where T: CuMatrixOp + 'a  {
    fn ptr_mut(&mut self) -> *mut f32 { self.ptr }
}


#[cfg(test)]
mod sub_matrix {
    use super::{CuMatrix, CuMatrixOp, CuMatrixOpMut};

    #[test]
    fn getters() {
        let initial_rows = 4;
        let initial_cols = 8;
        let mut matrix = CuMatrix::new(initial_rows, initial_cols, 0.0);

        {
            let slice = matrix.slice(2, 1, 2, 7);
            assert_eq!(slice.rows(), 2);
            assert_eq!(slice.cols(), 7);
            assert_eq!(slice.len(), 14);
            assert_eq!(slice.leading_dimension(), initial_rows);
        }

        {
            let slice = matrix.slice_mut(1, 3, 2, 2);
            assert_eq!(slice.rows(), 2);
            assert_eq!(slice.cols(), 2);
            assert_eq!(slice.len(), 4);
            assert_eq!(slice.leading_dimension(), initial_rows);
        }

        {
            let slice = matrix.slice_col(1, 7);
            assert_eq!(slice.rows(), initial_rows);
            assert_eq!(slice.cols(), 7);
            assert_eq!(slice.len(), initial_rows*7);
            assert_eq!(slice.leading_dimension(), initial_rows);
        }

        {
            let slice = matrix.slice_col_mut(3, 2);
            assert_eq!(slice.rows(), initial_rows);
            assert_eq!(slice.cols(), 2);
            assert_eq!(slice.len(), initial_rows*2);
            assert_eq!(slice.leading_dimension(), initial_rows);
        }
    }

    #[test]
    fn init() {
        let value = -1.254;
        let mut matrix = CuMatrix::new(2, 3, 0.0);
        matrix.slice_mut(0, 1, 1, 2).init(value);

        let output = &mut[0.0; 6];
        matrix.clone_to_host(output);

        assert_eq!(output[0], 0.0);
        assert_eq!(output[1], 0.0);
        assert_eq!(output[2], value);
        assert_eq!(output[3], 0.0);
        assert_eq!(output[4], value);
        assert_eq!(output[5], 0.0);
    }

}



// CuMatrixSlice

pub struct CuMatrixSlice<'a, T> where T: CuMatrixOp + 'a {
    parent: PhantomData<&'a T>,
    rows: usize,
    cols: usize,
    len: usize,
    ptr: *const f32,
}
impl<'a, T> CuMatrixOp for CuMatrixSlice<'a, T> where T: CuMatrixOp + 'a  {
    fn rows(&self) -> usize { self.rows }
    fn cols(&self) -> usize { self.cols }
    fn len(&self) -> usize { self.len }
    fn leading_dimension(&self) -> usize { self.rows }
    fn ptr(&self) -> *const f32 { self.ptr }

    fn clone_to_host(&self, output: &mut [f32]) {
        unsafe {
            cudaMemcpy(output.as_mut_ptr() as *mut c_void, self.ptr() as *const c_void, self.len() * size_of::<f32>(), CudaMemcpyKind::DeviceToHost)
        }.assert_success();
    }
}
impl <'a, T> CuPacked for CuMatrixSlice<'a, T> where T: CuMatrixOp + 'a {}
impl <'a, T> CuMatrixSlice<'a, T> where T: CuMatrixOp + 'a {
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
    parent: PhantomData<&'a T>,
    rows: usize,
    cols: usize,
    len: usize,
    ptr: *mut f32,
}
impl<'a, T> CuMatrixOp for CuMatrixSliceMut<'a, T> where T: CuMatrixOp + 'a  {
    fn rows(&self) -> usize { self.rows }
    fn cols(&self) -> usize { self.cols }
    fn len(&self) -> usize { self.len }
    fn leading_dimension(&self) -> usize { self.rows }
    fn ptr(&self) -> *const f32 { self.ptr }

    fn clone_to_host(&self, output: &mut [f32]) {
        unsafe {
            cudaMemcpy(output.as_mut_ptr() as *mut c_void, self.ptr() as *const c_void, self.len() * size_of::<f32>(), CudaMemcpyKind::DeviceToHost)
        }.assert_success();
    }
}
impl<'a, T> CuMatrixOpMut for CuMatrixSliceMut<'a, T> where T: CuMatrixOp + 'a  {
    fn ptr_mut(&mut self) -> *mut f32 { self.ptr }

    fn clone_from_host(&mut self, data: &[f32]) {
        unsafe {
            cudaMemcpy(self.ptr_mut() as *mut c_void, data.as_ptr() as *const c_void, self.len() * size_of::<f32>(), CudaMemcpyKind::HostToDevice)
        }.assert_success();
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



// Tests

#[cfg(test)]
mod tests {
    use super::{CuMatrixOp, CuMatrixOpMut};

    fn assert_equals_float(a: f32, b: f32) {
        let d = a-b;
        if d < -0.000001 || d > 0.000001 {
            panic!("{} != {}", a, b);
        }
    }

    #[test]
    fn slice() {
        let data = &[0.0, 1.0, 2.0, 0.1, 1.1, 2.1];
        let matrix = super::CuMatrix::from_data(3, 2, data);
        matrix.dev_equals(data);
        matrix.slice(0, 0, 3, 2).dev_equals(data);
        matrix.slice(1, 0, 1, 2).dev_equals(&[1.0, 1.1]);
        matrix.slice(0, 1, 3, 1).dev_equals(&[0.1, 1.1, 2.1]);
        matrix.slice(1, 0, 2, 2).slice(0, 1, 1, 1).dev_equals(&[1.1]);
    }
    #[test]
    fn clone_from_to_host() {
        let mut matrix = super::CuMatrix::new(2, 4, 0.0);
        let data = &[1.0, 2.0, -3.0, 0.5, 0.7, 0.3, 1.0, 3.0];
        matrix.clone_from_host(data);
        let output = &mut [0.0; 8];
        matrix.clone_to_host(output);

        for i in 0..8 { assert_equals_float(output[i], data[i]); }
    }

    #[test]
    fn init() {
        let value0 = -0.23254;
        let value1 = 1.1852;
        let mut matrix = super::CuMatrix::new(2, 3, 0.0);

        matrix.init(value0);
        matrix.slice_mut(0, 1, 2, 2).init(value1);

        let mut output = vec![0.0; 6];
        matrix.clone_to_host(&mut output);

        assert_equals_float(output[0], value0);
        assert_equals_float(output[1], value0);
        assert_equals_float(output[2], value1);
        assert_equals_float(output[3], value1);
        assert_equals_float(output[4], value1);
        assert_equals_float(output[5], value1);
    }

    #[test]
    fn add_value_self() {
        let value0 = -0.23254;
        let value1 = 1.185254;
        let mut matrix0 = super::CuMatrix::new(2, 4, 1.0);

        matrix0.init(value0);
        matrix0.slice_mut(0, 1, 2, 2).add_value_self(value1);

        let mut output = vec![0.0; 8];
        matrix0.clone_to_host(&mut output);

        assert_equals_float(output[0], value0);
        assert_equals_float(output[1], value0);
        assert_equals_float(output[2], value0+value1);
        assert_equals_float(output[3], value0+value1);
        assert_equals_float(output[4], value0+value1);
        assert_equals_float(output[5], value0+value1);
        assert_equals_float(output[6], value0);
        assert_equals_float(output[7], value0);
    }
    #[test]
    fn scale_self() {
        let value0 = -0.23254;
        let value1 = 1.185254;
        let mut matrix0 = super::CuMatrix::new(2, 4, 1.0);

        matrix0.init(value0);
        matrix0.slice_mut(0, 1, 2, 2).scale_self(value1);

        let mut output = vec![0.0; 8];
        matrix0.clone_to_host(&mut output);

        assert_equals_float(output[0], value0);
        assert_equals_float(output[1], value0);
        assert_equals_float(output[2], value0*value1);
        assert_equals_float(output[3], value0*value1);
        assert_equals_float(output[4], value0*value1);
        assert_equals_float(output[5], value0*value1);
        assert_equals_float(output[6], value0);
        assert_equals_float(output[7], value0);
    }
    #[test]
    fn add_self() {
        let value0 = -0.23254;
        let value1 = 1.185254;
        let mut matrix0 = super::CuMatrix::new(2, 4, 1.0);
        let mut matrix1 = super::CuMatrix::new(2, 2, 0.0);

        matrix0.init(value0);
        matrix1.init(value1);
        matrix0.slice_mut(0, 1, 2, 2).add_self(&matrix1);

        let mut output = vec![0.0; 8];
        matrix0.clone_to_host(&mut output);

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


/*
macro_rules! impl_op {
    ($struct_name: ident, $rows: ident, $cols: ident,
    $leading_dimension: ident, $ptr: ident) => {
        impl CuMatrixOp for $struct_name {
            #[inline]
            fn rows(&self) -> usize { self.$rows }
            #[inline]
            fn cols(&self) -> usize { self.$cols }
            #[inline]
            fn leading_dimension(&self) -> usize { self.$leading_dimension }
            #[inline]
            fn ptr(&self) -> *const f32 { self.$ptr }
        }
    }
}
macro_rules! impl_op_lifetime {
    ($struct_name: ident, $rows: ident, $cols: ident,
    $leading_dimension: ident, $ptr: ident) => {
        impl<'a> CuMatrixOp for $struct_name<'a> {
            #[inline]
            fn rows(&self) -> usize { self.$rows }
            #[inline]
            fn cols(&self) -> usize { self.$cols }
            #[inline]
            fn leading_dimension(&self) -> usize { self.$leading_dimension }
            #[inline]
            fn ptr(&self) -> *const f32 { self.$ptr }
        }
    }
}*/
/*
macro_rules! impl_op_mut {
    ($struct_name: ident, $ptr: ident) => {
        impl CuMatrixOpMut for $struct_name {
            #[inline]
            fn ptr_mut(&mut self) -> *mut f32 { self.$ptr }
        }
    }
}
macro_rules! impl_op_mut_lifetime {
    ($struct_name: ident, $ptr: ident) => {
        impl<'a> CuMatrixOpMut for $struct_name<'a> {
            #[inline]
            fn ptr_mut(&mut self) -> *mut f32 { self.$ptr }
        }
    }
}*/
/*
pub struct CuMatrix {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) leading_dimension: usize,
    pub(crate) data: *mut f32,
}
impl Drop for CuMatrix {
    fn drop(&mut self) {
        unsafe { cudaFree(self.data as *mut c_void) }.assert_success();
    }
}
impl CuMatrix {
    pub fn new(rows: usize, cols: usize, init_value: f32) -> CuPackedMatrix {
        let len = rows*cols;
        let mut data = std::ptr::null_mut();
        unsafe { cudaMalloc(&mut data, len*std::mem::size_of::<f32>()) }.assert_success();
        unsafe { VectorKernel_init((data as *mut f32), (len as i32), init_value) }
        CuPackedMatrix {
            matrix: CuMatrix {
                rows, cols,
                leading_dimension: rows,
                data: (data as *mut f32)
            },
            len,
        }
    }
    pub fn from_data(rows: usize, cols: usize, data: &[f32]) -> CuPackedMatrix {
        assert_eq_usize(rows*cols, "rows*cols", data.len(), "data.len()");
        let mut output = {
            let len = rows*cols;
            let mut data = std::ptr::null_mut();
            unsafe { cudaMalloc(&mut data, len*std::mem::size_of::<f32>()) }.assert_success();
            CuPackedMatrix {
                matrix: CuMatrix {
                    rows, cols,
                    leading_dimension: rows,
                    data: (data as *mut f32)
                },
                len,
            }
        };
        output.clone_from_host(data);
        output
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }
    #[inline]
    pub fn leading_dimension(&self) -> usize {
        self.leading_dimension
    }
    pub fn slice<'a>(&'a self, row_offset: usize, col_offset: usize, nb_rows: usize, nb_cols: usize) -> CuMatrixSlice<'a>  {
        assert!(row_offset < self.rows);
        assert!(nb_rows <= self.rows-row_offset);
        assert!(col_offset < self.cols);
        assert!(nb_cols <= self.cols-col_offset);
        CuMatrixSlice {
            parent: PhantomData,
            value: CuMatrix {
                rows: nb_rows,
                cols: nb_cols,
                leading_dimension: self.leading_dimension,
                data: unsafe { self.data.offset((row_offset + col_offset * self.rows) as isize) } }
        }
    }
    pub fn slice_mut<'a>(&'a mut self, row_offset: usize, col_offset: usize, nb_rows: usize, nb_cols: usize) -> CuMatrixSliceMut<'a> {
        assert!(row_offset < self.rows);
        assert!(nb_rows <= self.rows-row_offset);
        assert!(col_offset < self.cols);
        assert!(nb_cols <= self.cols-col_offset);
        CuMatrixSliceMut {
            parent: PhantomData,
            value: CuMatrix {
                rows: nb_rows,
                cols: nb_cols,
                leading_dimension: self.leading_dimension,
                data: unsafe { self.data.offset((row_offset + col_offset * self.rows) as isize) } }
        }
    }

    pub fn init(&mut self, value: f32) {
        unsafe { MatrixKernel_init(self.data,
                                   self.leading_dimension as i32,
                                   self.rows as i32,
                                   self.cols as i32,
                                   value) }
    }
    pub fn clone_from_host(&mut self, data: &[f32]) {
        assert_eq_usize(self.len(), "self.len()", data.len(), "data.len()");
        unsafe {
            cudaMemcpy2D(self.data as *mut c_void,
                         self.leading_dimension * size_of::<f32>(),
                         data.as_ptr() as *const c_void,
                         self.rows * size_of::<f32>(),
                         self.rows * size_of::<f32>(),
                         self.cols,
                         CudaMemcpyKind::HostToDevice);
        }
    }
    pub fn clone_to_host(&self, data: &mut [f32]) {
        assert_eq_usize(self.len(), "self.len()", data.len(), "data.len()");
        unsafe {
            cudaMemcpy2D(data.as_ptr() as *mut c_void,
                         self.rows * size_of::<f32>(),
                         self.data as *const c_void,
                         self.leading_dimension * size_of::<f32>(),
                         self.rows * size_of::<f32>(),
                         self.cols,
                         CudaMemcpyKind::DeviceToHost);
        }
    }
    pub fn clone_from_device(&mut self, source: &CuMatrix) {
        assert_eq_usize(self.len(), "self.len()", source.len(), "data.len()");
        unsafe {
            cudaMemcpy2D(self.data as *mut c_void,
                         self.leading_dimension * size_of::<f32>(),
                         source.data as *const c_void,
                         self.rows * size_of::<f32>(),
                         self.rows * size_of::<f32>(),
                         self.cols,
                         CudaMemcpyKind::DeviceToDevice);
        }
    }

    pub fn add_value_self(&mut self, value: f32) {
        unsafe { MatrixKernel_addValue(self. data, self.leading_dimension as i32,
                                       self.data, self.leading_dimension as i32,
                                       self.rows as i32, self.cols as i32,
                                       value) }
    }
    pub fn scale_self(&mut self, value: f32) {
        unsafe { MatrixKernel_scale(self. data, self.leading_dimension as i32,
                                       self.data, self.leading_dimension as i32,
                                       self.rows as i32, self.cols as i32,
                                       value) }
    }
    pub fn add_self(&mut self, to_add: &Self) {
        unsafe { MatrixKernel_add(self.data, self.leading_dimension as i32,
                                  to_add.data, to_add.leading_dimension as i32,
                                  self.data, self.leading_dimension as i32,
                                  self.rows as i32, self.cols as i32) }
    }

    #[allow(dead_code)]
    pub fn dev_print(&self, msg: &str) {
        let mut buffer = vec![0.0; self.len()];
        self.clone_to_host(&mut buffer);
        println!("{}", msg);
        for row in 0..self.rows {
            for col in 0..self.cols {
                print!("{:.5}, ", buffer[row+col*self.rows])
            }
            println!()
        }
    }
    fn assert_equals_float(a: f32, b: f32) {
        let d = a-b;
        if d < -0.000001 || d > 0.000001 {
            panic!("{} != {}", a, b);
        }
    }
    #[allow(dead_code)]
    pub fn dev_equals(&self, data: &[f32]) {
        assert_eq_usize(self.len(), "self.len()", data.len(), "data.len()");
        let mut buffer = vec![0.0; self.len()];
        self.clone_to_host(&mut buffer);
        let mut iter = buffer.iter();
        data.iter().for_each(|x| { Self::assert_equals_float(*x, *iter.next().unwrap()) });
    }
}


pub struct CuPackedMatrix {
    matrix: CuMatrix,
    len: usize,
}
impl Deref for CuPackedMatrix {
    type Target = CuMatrix;
    fn deref(&self) -> &CuMatrix {
        &self.matrix
    }
}
impl DerefMut for CuPackedMatrix {
    fn deref_mut(&mut self) -> &mut CuMatrix {
        &mut self.matrix
    }
}
impl CuPackedMatrix {
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }

    pub fn init(&mut self, value: f32) {
        unsafe { VectorKernel_init(self.data, self.len as i32, value) }
    }
    pub fn clone_from_host(&mut self, data: &[f32]) {
        assert_eq_usize(self.len, "self.len()", data.len(), "data.len()");
        unsafe {
            cudaMemcpy((self.data as *mut c_void),
                       (data.as_ptr() as *const c_void),
                       self.len*std::mem::size_of::<f32>(),
                       CudaMemcpyKind::HostToDevice)
        }.assert_success();
    }
    pub fn clone_to_host(&self, data: &mut [f32]) {
        assert_eq_usize(self.len, "self.len()", data.len(), "data.len()");
        unsafe {
            cudaMemcpy((data.as_mut_ptr() as *mut c_void),
                       (self.data as *mut c_void),
                       self.len*std::mem::size_of::<f32>(),
                       CudaMemcpyKind::DeviceToHost)
        }.assert_success();
    }
    pub fn clone_from_device(&mut self, source: &CuPackedMatrix) {
        assert_eq_usize(self.len, "self.len()", source.len, "source.len()");
        unsafe {
            cudaMemcpy((self.data as *mut c_void),
                       (source.data as *mut c_void),
                       self.len*std::mem::size_of::<f32>(),
                       CudaMemcpyKind::DeviceToDevice)
        }.assert_success();
    }

    pub fn add_value_self(&mut self, value: f32) {
        unsafe { VectorKernel_addValue(self.data, self.data,self.len as i32, value) }
    }
    pub fn scale_self(&mut self, value: f32) {
        unsafe { VectorKernel_scl(self.data, self.data,self.len as i32, value) }
    }
    pub fn add_self(&mut self, to_add: &Self) {
        unsafe { VectorKernel_add(self.data, to_add.data, self.data, self.len as i32) }
    }
}

pub struct CuMatrixSlice<'a> {
    parent: PhantomData<&'a CuMatrix>,
    value: CuMatrix,
}
impl<'a> Drop for CuMatrixSlice<'a> {
    fn drop(&mut self) {
        // Set value to null so it wont be dropped
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
    parent: PhantomData<&'a mut CuMatrix>,
    value: CuMatrix,
}
impl<'a> Drop for CuMatrixSliceMut<'a> {
    fn drop(&mut self) {
        // Set value to null so it wont be dropped
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
    fn slice() {
        let data = &[0.0, 1.0, 2.0, 0.1, 1.1, 2.1];
        let matrix = super::CuMatrix::from_data(3, 2, data);
        matrix.dev_equals(data);
        matrix.slice(0, 0, 3, 2).dev_equals(data);
        matrix.slice(1, 0, 1, 2).dev_equals(&[1.0, 1.1]);
        matrix.slice(0, 1, 3, 1).dev_equals(&[0.1, 1.1, 2.1]);
    }
    #[test]
    fn clone_from_to_host() {
        let mut matrix = super::CuMatrix::new(2, 4, 0.0);
        let data = &[1.0, 2.0, -3.0, 0.5, 0.7, 0.3, 1.0, 3.0];
        matrix.clone_from_host(data);
        let output = &mut [0.0; 8];
        matrix.clone_to_host(output);

        for i in 0..8 { assert_equals_float(output[i], data[i]); }
    }

    #[test]
    fn init() {
        let value0 = -0.23254;
        let value1 = 1.1852;
        let mut matrix = super::CuMatrix::new(2, 3, 0.0);

        matrix.init(value0);
        matrix.slice_mut(0, 1, 2, 2).init(value1);

        let mut output = vec![0.0; 6];
        matrix.clone_to_host(&mut output);

        assert_equals_float(output[0], value0);
        assert_equals_float(output[1], value0);
        assert_equals_float(output[2], value1);
        assert_equals_float(output[3], value1);
        assert_equals_float(output[4], value1);
        assert_equals_float(output[5], value1);
    }

    #[test]
    fn add_value_self() {
        let value0 = -0.23254;
        let value1 = 1.185254;
        let mut matrix0 = super::CuMatrix::new(2, 4, 1.0);

        matrix0.init(value0);
        matrix0.slice_mut(0, 1, 2, 2).add_value_self(value1);

        let mut output = vec![0.0; 8];
        matrix0.clone_to_host(&mut output);

        assert_equals_float(output[0], value0);
        assert_equals_float(output[1], value0);
        assert_equals_float(output[2], value0+value1);
        assert_equals_float(output[3], value0+value1);
        assert_equals_float(output[4], value0+value1);
        assert_equals_float(output[5], value0+value1);
        assert_equals_float(output[6], value0);
        assert_equals_float(output[7], value0);
    }
    #[test]
    fn scale_self() {
        let value0 = -0.23254;
        let value1 = 1.185254;
        let mut matrix0 = super::CuMatrix::new(2, 4, 1.0);

        matrix0.init(value0);
        matrix0.slice_mut(0, 1, 2, 2).scale_self(value1);

        let mut output = vec![0.0; 8];
        matrix0.clone_to_host(&mut output);

        assert_equals_float(output[0], value0);
        assert_equals_float(output[1], value0);
        assert_equals_float(output[2], value0*value1);
        assert_equals_float(output[3], value0*value1);
        assert_equals_float(output[4], value0*value1);
        assert_equals_float(output[5], value0*value1);
        assert_equals_float(output[6], value0);
        assert_equals_float(output[7], value0);
    }
    #[test]
    fn add_self() {
        let value0 = -0.23254;
        let value1 = 1.185254;
        let mut matrix0 = super::CuMatrix::new(2, 4, 1.0);
        let mut matrix1 = super::CuMatrix::new(2, 2, 0.0);

        matrix0.init(value0);
        matrix1.init(value1);
        matrix0.slice_mut(0, 1, 2, 2).add_self(&matrix1);

        let mut output = vec![0.0; 8];
        matrix0.clone_to_host(&mut output);

        assert_equals_float(output[0], value0);
        assert_equals_float(output[1], value0);
        assert_equals_float(output[2], value0+value1);
        assert_equals_float(output[3], value0+value1);
        assert_equals_float(output[4], value0+value1);
        assert_equals_float(output[5], value0+value1);
        assert_equals_float(output[6], value0);
        assert_equals_float(output[7], value0);
    }

}*/