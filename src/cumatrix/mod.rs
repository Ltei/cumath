

use std::{marker::PhantomData, mem::size_of};

use meta::{codec::*, assert::*};
use ffi::{cuda_ffi::*, matrixkernel_ffi::*};



mod matrix;
pub use self::matrix::*;
mod matrix_slice;
pub use self::matrix_slice::*;
mod sub_matrix;
pub use self::sub_matrix::*;



/// Immutable matrix operator trait.
pub trait CuMatrixOp {

    /// [inline]
    /// Returns the number of rows in the matrix.
    #[inline]
    fn rows(&self) -> usize;

    /// [inline]
    /// Returns the number of columns in the matrix.
    #[inline]
    fn cols(&self) -> usize;

    /// [inline]
    /// Returns the number of elements in the matrix.
    #[inline]
    fn len(&self) -> usize;

    /// [inline]
    /// Returns the leading dimension of the matrix.
    #[inline]
    fn leading_dimension(&self) -> usize;

    /// [inline]
    /// Returns a pointer on the matrix's data.
    #[inline]
    fn ptr(&self) -> *const f32;

    /// Returns an immutable sub-matrix.
    fn slice<'a>(&'a self, row_offset: usize, col_offset: usize, nb_rows: usize, nb_cols: usize) -> CuSubMatrix<'a> {
        CuSubMatrix {
            parent: PhantomData,
            rows: nb_rows,
            cols: nb_cols,
            leading_dimension: self.leading_dimension(),
            ptr: unsafe { self.ptr().offset((row_offset + col_offset*self.leading_dimension()) as isize) },
        }
    }

    /// Clone this matrix's data to host memory.
    fn clone_to_host(&self, data: &mut [f32]) {
        assert_eq_usize(self.len(), "self.len()", data.len(), "data.len()");
        cuda_memcpy2d(data.as_mut_ptr(),
                      self.rows() * size_of::<f32>(),
                      self.ptr(),
                      self.leading_dimension() * size_of::<f32>(),
                      self.rows() * size_of::<f32>(),
                      self.cols(),
                      CudaMemcpyKind::DeviceToHost);
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
    #[allow(dead_code)]
    fn dev_assert_equals(&self, data: &[f32]) where Self: Sized {
        assert_eq_usize(self.len(), "self.len()", data.len(), "data.len()");
        let mut buffer = vec![0.0; self.len()];
        self.clone_to_host(&mut buffer);
        let mut iter = buffer.iter();
        data.iter().for_each(|x| { assert_equals_float(*x, *iter.next().unwrap()) });
    }

}

/// Mutable matrix operator trait.
pub trait CuMatrixOpMut: CuMatrixOp  {

    /// [inline]
    /// Returns a mutable pointer on the matrix's data
    #[inline]
    fn ptr_mut(&mut self) -> *mut f32;

    fn slice_mut<'a>(&'a mut self, row_offset: usize, col_offset: usize, nb_rows: usize, nb_cols: usize) -> CuSubMatrixMut<'a> {
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
        cuda_memcpy2d(self.ptr_mut(),
                      self.leading_dimension() * size_of::<f32>(),
                      data.as_ptr(),
                      self.rows() * size_of::<f32>(),
                      self.rows() * size_of::<f32>(),
                      self.cols(),
                      CudaMemcpyKind::HostToDevice);
    }
    fn clone_from_device(&mut self, data: &CuMatrixOp) {
        assert_eq_usize(self.len(), "self.len()", data.len(), "data.len()");
        cuda_memcpy2d(self.ptr_mut(),
                      self.leading_dimension() * size_of::<f32>(),
                      data.ptr(),
                      data.leading_dimension() * size_of::<f32>(),
                      self.rows() * size_of::<f32>(),
                      self.cols(),
                      CudaMemcpyKind::DeviceToDevice);
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
    fn add_self(&mut self, to_add: &CuMatrixOp) {
        unsafe {
            MatrixKernel_add(self.ptr(), self.leading_dimension() as i32,
                             to_add.ptr(), to_add.leading_dimension() as i32,
                             self.ptr_mut(), self.leading_dimension() as i32,
                             self.rows() as i32, self.cols() as i32)
        }
    }

}



impl Codec for CuMatrix {
    type OutputType = CuMatrix;

    fn encode(&self) -> String {
        let mut host_data = vec![0.0; self.len()];
        self.clone_to_host(host_data.as_mut_slice());

        let mut output = format!("{} {} ", self.rows(), self.cols());
        host_data.iter().for_each(|x| {
            output.push_str(&format!("{} ", x))
        });
        output
    }
    fn decode(data: &str) -> CuMatrix {
        let mut split = data.split_whitespace();
        let rows = split.next().unwrap().parse::<usize>().unwrap();
        let cols = split.next().unwrap().parse::<usize>().unwrap();
        CuMatrix::from_data(rows, cols,
            split.map(|x| {
                x.parse::<f32>().unwrap_or_else(|err| { panic!("{}", err) })
            }).collect::<Vec<f32>>().as_slice()
        )
    }
}




// TESTS

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn codec() {
        let data = [1.2, -2.2656146, 7.12, 2.0, 4.5, 7.256];
        CuMatrix::decode(&CuMatrix::from_data(2, 3, &data).encode()).dev_assert_equals(&data);
    }

}