

use std::{marker::PhantomData, mem::size_of};

use ffi::{cuda_ffi::*, matrixkernel_ffi::*};
use meta::codec::*;

#[cfg(not(feature = "disable_checks"))]
use meta::assert::*;




mod matrix;
pub use self::matrix::*;
mod matrix_slice;
pub use self::matrix_slice::*;
mod sub_matrix;
pub use self::sub_matrix::*;
mod matrix_ptr;
pub use self::matrix_ptr::*;




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
    fn as_ptr(&self) -> *const f32;

    /// Returns an immutable sub-matrix.
    fn slice<'a>(&'a self, row_offset: usize, col_offset: usize, nb_rows: usize, nb_cols: usize) -> CuSubMatrix<'a> {
        #[cfg(not(feature = "disable_checks"))] {
            assert_infeq_usize(row_offset + nb_rows, "row_offset+nb_rows", self.rows(), "self.rows()");
            assert_infeq_usize(col_offset + nb_cols, "col_offset+nb_cols", self.cols(), "self.cols()");
        }
        CuSubMatrix {
            parent: PhantomData,
            rows: nb_rows,
            cols: nb_cols,
            leading_dimension: self.leading_dimension(),
            ptr: unsafe { self.as_ptr().offset((row_offset + col_offset*self.leading_dimension()) as isize) },
        }
    }

    /// Clone this matrix's data to host memory.
    fn clone_to_host(&self, data: &mut [f32]) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(self.len(), "self.len()", data.len(), "data.len()");
        }
        cuda_memcpy2d(data.as_mut_ptr(),
                      self.rows() * size_of::<f32>(),
                      self.as_ptr(),
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
        if self.len() != data.len() { panic!(); }
        let mut buffer = vec![0.0; self.len()];
        self.clone_to_host(&mut buffer);
        for i in 0..data.len() {
            let delta = data[i]-buffer[i];
            if delta < -0.00001 || delta > 0.00001 { panic!("At index {} : {:.8} != {:.8}", i, data[i], buffer[i]); }
        }
    }

}

/// Mutable matrix operator trait.
pub trait CuMatrixOpMut: CuMatrixOp  {

    /// [inline]
    /// Returns a mutable pointer on the matrix's data
    #[inline]
    fn as_mut_ptr(&mut self) -> *mut f32;

    /// Returns a mutable sub-matrix.
    fn slice_mut<'a>(&'a mut self, row_offset: usize, col_offset: usize, nb_rows: usize, nb_cols: usize) -> CuSubMatrixMut<'a> {
        #[cfg(not(feature = "disable_checks"))] {
            assert_infeq_usize(row_offset + nb_rows, "row_offset+nb_rows", self.rows(), "self.rows()");
            assert_infeq_usize(col_offset + nb_cols, "col_offset+nb_cols", self.cols(), "self.cols()");
        }
        CuSubMatrixMut {
            parent: PhantomData,
            rows: nb_rows,
            cols: nb_cols,
            leading_dimension: self.leading_dimension(),
            ptr: unsafe { self.as_mut_ptr().offset((row_offset + col_offset*self.leading_dimension()) as isize) },
        }
    }

    unsafe fn force_ownership(&mut self) -> CuMatrixPtr {
        CuMatrixPtr {
            rows: self.rows(),
            cols: self.cols(),
            len: self.len(),
            ptr: self.as_mut_ptr(),
        }
    }

    /// Clone host memory to this matrix's data.
    fn clone_from_host(&mut self, data: &[f32]) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(self.len(), "self.len()", data.len(), "data.len()");
        }
        cuda_memcpy2d(self.as_mut_ptr(),
                      self.leading_dimension() * size_of::<f32>(),
                      data.as_ptr(),
                      self.rows() * size_of::<f32>(),
                      self.rows() * size_of::<f32>(),
                      self.cols(),
                      CudaMemcpyKind::HostToDevice);
    }

    /// Clone a matrix's data to this matrix's data.
    fn clone_from_device(&mut self, data: &CuMatrixOp) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(self.len(), "self.len()", data.len(), "data.len()");
        }
        cuda_memcpy2d(self.as_mut_ptr(),
                      self.leading_dimension() * size_of::<f32>(),
                      data.as_ptr(),
                      data.leading_dimension() * size_of::<f32>(),
                      self.rows() * size_of::<f32>(),
                      self.cols(),
                      CudaMemcpyKind::DeviceToDevice);
    }

    /// Initializes the matrix with value.
    fn init(&mut self, value: f32) {
        unsafe {
            MatrixKernel_init(self.as_mut_ptr(), self.leading_dimension() as i32,
                              self.rows() as i32, self.cols() as i32, value);
        }
    }

    /// Add value to each matrix of the vector.
    fn add_value_self(&mut self, value: f32) {
        unsafe {
            MatrixKernel_addValue(self.as_ptr(), self.leading_dimension() as i32,
                                  self.as_mut_ptr(), self.leading_dimension() as i32,
                                  self.rows() as i32, self.cols() as i32, value);
        }
    }

    /// Scale each element of the matrix by value.
    fn scale_self(&mut self, value: f32) {
        unsafe {
            MatrixKernel_scale(self.as_ptr(), self.leading_dimension() as i32,
                               self.as_mut_ptr(), self.leading_dimension() as i32,
                               self.rows() as i32, self.cols() as i32, value);
        }
    }

    /// Add an other matrix to this one.
    fn add_self(&mut self, to_add: &CuMatrixOp) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(self.rows(), "self.rows", to_add.rows(), "to_add.rows()");
            assert_eq_usize(self.cols(), "self.cols", to_add.cols(), "to_add.cols()");
        }
        unsafe {
            MatrixKernel_add(self.as_ptr(), self.leading_dimension() as i32,
                             to_add.as_ptr(), to_add.leading_dimension() as i32,
                             self.as_mut_ptr(), self.leading_dimension() as i32,
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