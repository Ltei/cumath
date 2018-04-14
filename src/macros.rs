


macro_rules! impl_CuPackedData {
    ( $name:ty $( , $lifetimes:tt )* ) => {
        impl<$($lifetimes),*> $crate::cudata::CuPackedData for $name {
            fn len(&self) -> usize { self.len }
            fn as_ptr(&self) -> *const f32 { self.ptr }
        }
    };
}
macro_rules! impl_CuPackedDataMut {
    ( $name:ty $( , $lifetimes:tt )* ) => {
        impl<$($lifetimes),*> $crate::cudata::CuPackedDataMut for $name {
            fn as_mut_ptr(&mut self) -> *mut f32 { self.ptr }
        }
    };
}



macro_rules! impl_CuVectorOp {
    ( $name:ty $( , $lifetimes:tt )* ) => {
        impl<$($lifetimes),*> $crate::CuVectorOp for $name {
            fn len(&self) -> usize { self.len }
            fn as_ptr(&self) -> *const f32 { self.ptr }
        }
    };
}
macro_rules! impl_CuVectorOpMut {
    ( $name:ty $( , $lifetimes:tt )* ) => {
        impl<$($lifetimes),*> $crate::CuVectorOpMut for $name {
            fn as_mut_ptr(&mut self) -> *mut f32 { self.ptr }
        }
    };
}




macro_rules! impl_CuMatrixOp_fragmented {
    ( $name:ty $( , $lifetimes:tt )* ) => {
        impl<$($lifetimes),*> $crate::CuMatrixOp for $name {
            fn rows(&self) -> usize { self.rows }
            fn cols(&self) -> usize { self.cols }
            fn len(&self) -> usize { self.rows*self.cols }
            fn leading_dimension(&self) -> usize { self.leading_dimension }
            fn as_ptr(&self) -> *const f32 { self.ptr }
        }
    };
}
macro_rules! impl_CuMatrixOpMut_fragmented {
    ( $name:ty $( , $lifetimes:tt )* ) => {
        impl<$($lifetimes),*> $crate::CuMatrixOpMut for $name {
            fn as_mut_ptr(&mut self) -> *mut f32 { self.ptr }
        }
    };
}

macro_rules! impl_CuMatrixOp_packed {
    ( $name:ty $( , $lifetimes:tt )* ) => {
        impl<$($lifetimes),*> $crate::CuMatrixOp for $name {
            fn rows(&self) -> usize { self.rows }
            fn cols(&self) -> usize { self.cols }
            fn len(&self) -> usize { self.len }
            fn leading_dimension(&self) -> usize { self.rows }
            fn as_ptr(&self) -> *const f32 { self.ptr }

            fn clone_to_host(&self, output: &mut [f32]) {
                $crate::ffi::cuda_ffi::cuda_memcpy(output.as_mut_ptr(), self.ptr, self.len * $crate::std::mem::size_of::<f32>(), $crate::ffi::cuda_ffi::CudaMemcpyKind::DeviceToHost);
            }
        }
    };
}
macro_rules! impl_CuMatrixOpMut_packed {
    ( $name:ty $( , $lifetimes:tt )* ) => {

        impl<$($lifetimes),*> $crate::CuMatrixOpMut for $name {
            fn as_mut_ptr(&mut self) -> *mut f32 { self.ptr }

            fn clone_from_host(&mut self, data: &[f32]) {
                $crate::ffi::cuda_ffi::cuda_memcpy(self.ptr, data.as_ptr(), self.len * $crate::std::mem::size_of::<f32>(), $crate::ffi::cuda_ffi::CudaMemcpyKind::HostToDevice);
            }

            fn init(&mut self, value: f32) {
                unsafe { $crate::ffi::vectorkernel_ffi::VectorKernel_init(self.ptr, self.len as i32, value) }
            }
            fn add_value_self(&mut self, value: f32) {
                unsafe { $crate::ffi::vectorkernel_ffi::VectorKernel_addValue(self.ptr, self.ptr, self.len as i32, value) }
            }
            fn scale_self(&mut self, value: f32) {
                unsafe { $crate::ffi::vectorkernel_ffi::VectorKernel_scl(self.ptr, self.ptr, self.len as i32, value) }
            }
        }

    };
}