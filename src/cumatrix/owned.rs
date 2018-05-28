
use std::{ptr, mem::size_of, ops::{Deref, DerefMut}};
use super::*;



/// A GPU-allocated matrix
/// Holds a pointer to continuous GPU memory.
#[derive(Debug)]
pub struct CuMatrix<T: CuDataType> {
    deref: CuMatrixDeref<T>
}

impl<T: CuDataType> Drop for CuMatrix<T> {
    fn drop(&mut self) { cuda_free(self.deref.ptr as *mut c_void) }
}

impl<T: CuDataType> Deref for CuMatrix<T> {
    type Target = CuMatrixDeref<T>;
    fn deref(&self) -> &CuMatrixDeref<T> { &self.deref }
}
impl<T: CuDataType> DerefMut for CuMatrix<T> {
    fn deref_mut(&mut self) -> &mut CuMatrixDeref<T> { &mut self.deref }
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
                CuMatrix { deref: CuMatrixDeref { rows, cols, len, ptr: ptr as *mut $inner_type, leading_dimension: rows } }
            }
            /// Returns a new GPU-allocated matrix from a dimension and an initial value.
            pub fn new(value: $inner_type, rows: usize, cols: usize) -> CuMatrix<$inner_type> {
                let len = rows * cols;
                let mut ptr = ptr::null_mut();
                cuda_malloc(&mut ptr, len * size_of::<$inner_type>());
                unsafe { $fn_init(ptr as *mut $inner_type, value, len as i32, DEFAULT_STREAM.stream) }
                CuMatrix { deref: CuMatrixDeref { rows, cols, len, ptr: ptr as *mut $inner_type, leading_dimension: rows } }
            }
        }
    };
}

impl_CuMatrix!(i32, VectorPacked_init_i32);
impl_CuMatrix!(f32, VectorPacked_init_f32);

impl<T: CuDataType> CuMatrix<T> {

    /// Returns a new uninitialized GPU-allocated matrix.
    pub unsafe fn uninitialized(rows: usize, cols: usize) -> CuMatrix<T> {
        let len = rows * cols;
        let mut ptr = ptr::null_mut();
        cuda_malloc(&mut ptr, len * size_of::<T>());
        CuMatrix { deref: CuMatrixDeref { rows, cols, len, ptr: ptr as *mut T, leading_dimension: rows } }
    }

    /// Creates a Matrix from a raw pointer.
    pub unsafe fn from_raw_ptr(ptr: *mut T, rows: usize, cols: usize) -> CuMatrix<T> {
        CuMatrix { deref: CuMatrixDeref { rows, cols, len: rows*cols, ptr, leading_dimension: rows } }
    }

    /// Returns a new GPU-allocated matrix from CPU data.
    pub fn from_host_data(rows: usize, cols: usize, data: &[T]) -> CuMatrix<T> {
        let len = rows*cols;
        let mut ptr = ptr::null_mut();
        cuda_malloc(&mut ptr, data.len() * size_of::<T>());
        cuda_memcpy(ptr, data.as_ptr() as *const c_void, data.len() * size_of::<T>(), cudaMemcpyKind::HostToDevice);
        CuMatrix {
            deref: CuMatrixDeref {
                ptr: ptr as *mut T,
                len,
                rows,
                cols,
                leading_dimension: rows,
            }
        }
    }
    /// Returns a new GPU-allocated matrix from CPU data.
    pub fn from_device_data(data: &CuMatrixDeref<T>) -> CuMatrix<T> {
        let mut ptr = ptr::null_mut();
        cuda_malloc(&mut ptr, data.len() * size_of::<T>());
        let mut output = CuMatrix {
            deref: CuMatrixDeref {
                ptr: ptr as *mut T,
                len: data.len,
                rows: data.rows,
                cols: data.cols,
                leading_dimension: data.rows,
            }
        };
        output.clone_from_device(data);
        output
    }

    /// Consume this Matrix to return a new CuVector holding its data
    pub fn into_vector(mut self) -> ::CuVector<T> {
        let vector = unsafe { ::CuVector::from_raw_ptr(self.ptr, self.len) };
        self.ptr = ptr::null_mut(); // So it won't be freed
        vector
    }

    /// Returns a vector slice containing this matrix datas
    pub fn as_vector(&self) -> ::CuVectorSlice<T> {
        ::CuVectorSlice {
            _parent: PhantomData,
            deref: ::CuVectorDeref {
                ptr: self.ptr,
                len: self.len,
            }
        }
    }

    /// Returns a mutable vector slice containing this matrix datas
    pub fn as_mut_vector(&mut self) -> ::CuVectorSliceMut<T> {
        ::CuVectorSliceMut {
            _parent: PhantomData,
            deref: ::CuVectorDeref {
                ptr: self.as_mut_ptr(),
                len: self.len(),
            }
        }
    }

    /// Returns a new slice
    pub fn slice_col<'a>(&'a self, col_offset: usize, nb_cols: usize) -> CuMatrixSlice<'a, T> {
        #[cfg(not(feature = "disable_checks"))] {
            assert_infeq_usize(col_offset + nb_cols, "col_offset+nb_cols", self.cols(), "self.cols()");
        }
        CuMatrixSlice {
            _parent: PhantomData,
            deref: CuMatrixDeref {
                ptr: unsafe { self.ptr.offset((col_offset*self.rows) as isize) },
                len: self.rows * nb_cols,
                rows: self.rows,
                cols: nb_cols,
                leading_dimension: self.leading_dimension,
            }
        }
    }

    /// Returns a new mutable slice
    pub fn slice_col_mut<'a>(&'a mut self, col_offset: usize, nb_cols: usize) -> CuMatrixSliceMut<'a, T> {
        #[cfg(not(feature = "disable_checks"))] {
            assert_infeq_usize(col_offset + nb_cols, "col_offset+nb_cols", self.cols(), "self.cols()");
        }
        CuMatrixSliceMut {
            _parent: PhantomData,
            deref: CuMatrixDeref {
                rows: self.leading_dimension(),
                cols: nb_cols,
                len: self.rows * nb_cols,
                ptr: unsafe { self.ptr.offset((col_offset*self.rows) as isize) },
                leading_dimension: self.rows,
            }
        }
    }

}