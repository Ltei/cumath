
use std::{ptr, mem::size_of};

use super::*;
use kernel::*;


/// A GPU-allocated matrix
/// Holds a pointer to continuous GPU memory.
pub struct CuMatrix<T: CuDataType> {
    ptr: *mut T,
    len: usize,
    rows: usize,
    cols: usize,
}
impl<T: CuDataType> Drop for CuMatrix<T> {
    fn drop(&mut self) { cuda_free(self.ptr as *mut c_void) }
}

macro_rules! impl_CuMatrix {
    ($inner_type:ident, $fn_init:ident) => {
        impl CuMatrix<$inner_type> {
            /// Returns a new zero GPU-allocated matrix from a dimension.
            pub fn zero(rows: usize, cols: usize) -> CuMatrix<$inner_type> {
                let len = rows * cols;
                let mut ptr = ptr::null_mut();
                cuda_malloc(&mut ptr, len * size_of::<$inner_type>());
                unsafe { $fn_init(ptr as *mut $inner_type, $inner_type::zero(), len as i32, DEFAULT_STREAM.stream) }
                CuMatrix { rows, cols, len, ptr: ptr as *mut $inner_type }
            }
            /// Returns a new GPU-allocated matrix from a dimension and an initial value.
            pub fn new(value: $inner_type, rows: usize, cols: usize) -> CuMatrix<$inner_type> {
                let len = rows * cols;
                let mut ptr = ptr::null_mut();
                cuda_malloc(&mut ptr, len * size_of::<$inner_type>());
                unsafe { $fn_init(ptr as *mut $inner_type, value, len as i32, DEFAULT_STREAM.stream) }
                CuMatrix { rows, cols, len, ptr: ptr as *mut $inner_type }
            }
            /// Returns a new mutable slice
            pub fn slice_col_mut<'a>(&'a mut self, col_offset: usize, nb_cols: usize) -> CuMatrixSliceMut<'a, $inner_type> {
                #[cfg(not(feature = "disable_checks"))] {
                    assert_infeq_usize(col_offset + nb_cols, "col_offset+nb_cols", self.cols(), "self.cols()");
                }
                CuMatrixSliceMut {
                    _parent: PhantomData,
                    rows: self.leading_dimension(),
                    cols: nb_cols,
                    len: self.rows * nb_cols,
                    ptr: unsafe { self.ptr.offset((col_offset*self.rows) as isize) },
                }
            }
        }
    };
}

impl_CuMatrix!(i32, VectorPacked_init_i32);
impl_CuMatrix!(f32, VectorPacked_init_f32);
impl_mutable_packed_matrix_holder!(CuMatrix);

impl<T: CuDataType> CuMatrix<T> {

    /// Returns a new GPU-allocated matrix from CPU data.
    pub fn from_host_data(rows: usize, cols: usize, data: &[T]) -> CuMatrix<T> {
        let len = rows*cols;
        let mut ptr = ptr::null_mut();
        cuda_malloc(&mut ptr, data.len() * size_of::<T>());
        cuda_memcpy(ptr, data.as_ptr() as *const c_void, data.len() * size_of::<T>(), cudaMemcpyKind::HostToDevice);
        CuMatrix {
            ptr: ptr as *mut T,
            len, rows, cols
        }
    }
    /// Returns a new GPU-allocated matrix from CPU data.
    pub fn from_device_data(data: &CuMatrixOp<T>) -> CuMatrix<T> {
        let mut ptr = ptr::null_mut();
        cuda_malloc(&mut ptr, data.len() * size_of::<T>());
        cuda_memcpy(ptr, data.as_ptr() as *const c_void, data.len() * size_of::<T>(), cudaMemcpyKind::DeviceToDevice);
        CuMatrix {
            ptr: ptr as *mut T,
            len: data.len(),
            rows: data.rows(),
            cols: data.cols(),
        }
    }

    /// Consume this Matrix to return a new CuVector holding its data
    pub fn into_vector(mut self) -> ::CuVector<T> {
        let vector = unsafe { ::CuVector::from_raw_ptr(self.ptr, self.len) };
        self.ptr = ptr::null_mut(); // So it won't be freed
        vector
    }

    pub fn slice_col<'a>(&'a self, col_offset: usize, nb_cols: usize) -> CuMatrixSlice<'a, T> {
        #[cfg(not(feature = "disable_checks"))] {
            assert_infeq_usize(col_offset + nb_cols, "col_offset+nb_cols", self.cols(), "self.cols()");
        }
        CuMatrixSlice {
            _parent: PhantomData,
            ptr: unsafe { self.ptr.offset((col_offset*self.rows) as isize) },
            len: self.rows * nb_cols,
            rows: self.rows,
            cols: nb_cols,
        }
    }

}