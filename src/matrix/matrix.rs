
use std::{ptr, mem::size_of};

use assert::*;
use tags::*;

use super::*;
use ffi::vectorkernel_ffi::*;



// CuMatrix

pub struct CuMatrix {
    rows: usize,
    cols: usize,
    len: usize,
    ptr: *mut f32,
}
impl Drop for CuMatrix {
    fn drop(&mut self) {
        cuda_free(self.ptr);
    }
}
impl CuMatrixOp for CuMatrix {
    fn rows(&self) -> usize { self.rows }
    fn cols(&self) -> usize { self.cols }
    fn len(&self) -> usize { self.rows*self.cols }
    fn leading_dimension(&self) -> usize { self.rows }
    fn ptr(&self) -> *const f32 { self.ptr }

    fn clone_to_host(&self, output: &mut [f32]) {
        cuda_memcpy(output.as_mut_ptr(), self.ptr, self.len * size_of::<f32>(), CudaMemcpyKind::DeviceToHost);
    }
}
impl CuMatrixOpMut for CuMatrix {
    fn ptr_mut(&mut self) -> *mut f32 { self.ptr }

    fn clone_from_host(&mut self, data: &[f32]) {
        cuda_memcpy(self.ptr, data.as_ptr(), self.len() * size_of::<f32>(), CudaMemcpyKind::HostToDevice);
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
        cuda_malloc(&mut data, len*size_of::<f32>());
        unsafe { VectorKernel_init(data as *mut f32, len as i32, init_value) }
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
            cuda_malloc(&mut data, len*size_of::<f32>());
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

    pub fn aypb<CuMatrixOpMutT: CuMatrixOpMut>
    (a: f32, b: f32, y: &mut CuMatrixOpMutT) {
        unsafe { MatrixKernel_aYpb(a, b, y.ptr_mut(), y.leading_dimension() as i32, y.rows() as i32, y.cols() as i32) }
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