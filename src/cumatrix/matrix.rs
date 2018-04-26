
use std::{ptr, mem::size_of};

use super::*;
use ffi::vectorkernel_ffi::*;



/// A GPU-allocated matrix
/// Holds a pointer to continuous GPU memory.
pub struct CuMatrix {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) len: usize,
    pub(crate) ptr: *mut f32,
}
impl Drop for CuMatrix {
    fn drop(&mut self) {
        cuda_free(self.ptr);
    }
}
impl_CuPackedDataMut!(CuMatrix);
impl_CuMatrixOpMut_packed!(CuMatrix);

impl CuMatrix {

    /// Returns a new GPU-allocated matrix from a length and an initial value.
    pub fn new(rows: usize, cols: usize, init_value: f32) -> CuMatrix {
        let len = rows*cols;
        let mut data = ptr::null_mut();
        cuda_malloc(&mut data, len*size_of::<f32>());
        unsafe { VectorKernel_init(data as *mut f32, len as i32, init_value, DEFAULT_STREAM.stream) }
        CuMatrix {
            rows, cols, len,
            ptr: data as *mut f32,
        }
    }
    /// Returns a new GPU-allocated matrix from CPU data.
    pub fn from_data(rows: usize, cols: usize, data: &[f32]) -> CuMatrix {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(rows * cols, "rows*cols", data.len(), "data.len()");
        }
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

    pub fn slice_col<'a>(&'a self, col_offset: usize, nb_cols: usize) -> CuMatrixSlice<'a> {
        #[cfg(not(feature = "disable_checks"))] {
            assert_infeq_usize(col_offset + nb_cols, "col_offset+nb_cols", self.cols(), "self.cols()");
        }
        CuMatrixSlice {
            parent: PhantomData,
            rows: self.leading_dimension(),
            cols: nb_cols,
            len: self.leading_dimension()*nb_cols,
            ptr: unsafe { self.ptr.offset((col_offset*self.leading_dimension()) as isize) },
        }
    }
    pub fn slice_col_mut<'a>(&'a mut self, col_offset: usize, nb_cols: usize) -> CuMatrixSliceMut<'a> {
        #[cfg(not(feature = "disable_checks"))] {
            assert_infeq_usize(col_offset + nb_cols, "col_offset+nb_cols", self.cols(), "self.cols()");
        }
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
mod tests {
    use super::{CuMatrix, CuMatrixOp, CuMatrixOpMut};
    use cuda::*;

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
        matrix.init(value, &DEFAULT_STREAM);

        let output = &mut[0.0; 6];
        matrix.clone_to_host(output);

        output.iter().for_each(|x| assert_eq!(*x, value));
    }

    #[test]
    fn slice() {
        let data = &[0.0, 1.0, 2.0, 0.1, 1.1, 2.1];
        let matrix = super::CuMatrix::from_data(3, 2, data);

        matrix.dev_assert_equals(data);
        matrix.slice(0, 0, 3, 2).dev_assert_equals(data);
        matrix.slice(1, 0, 1, 2).dev_assert_equals(&[1.0, 1.1]);
        matrix.slice(0, 1, 3, 1).dev_assert_equals(&[0.1, 1.1, 2.1]);
        matrix.slice(1, 0, 2, 2).slice(0, 1, 1, 1).dev_assert_equals(&[1.1]);
        matrix.slice_col(1, 1).slice(1, 0, 1, 1).dev_assert_equals(&[1.1]);
    }

}