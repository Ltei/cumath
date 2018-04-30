

macro_rules! impl_CuMatrixOp_fragmented {
    ( $name:ty $( , $lifetimes:tt )* ) => {
        impl<$($lifetimes),*> $crate::CuMatrixOp for $name {
            fn rows(&self) -> usize { self.rows }
            fn cols(&self) -> usize { self.cols }
            fn len(&self) -> usize { self.rows*self.cols }
            fn leading_dimension(&self) -> usize { self.leading_dimension }
            fn as_ptr(&self) -> *const f32 { self.ptr }
        }

        impl<$($lifetimes),*> ::std::fmt::Debug for $name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
                let len = self.rows * self.cols;
                let mut buffer = vec![0.0; len];
                ::CuMatrixOp::clone_to_host(self, &mut buffer);
                write!(f, "Matrix ({},{}) [{:p}] :\n", self.rows, self.cols, self.ptr)?;
                if self.cols > 0 {
                    for row in 0..self.rows {
                        write!(f, "[")?;
                        for col in 0..self.cols-1 {
                            write!(f, "{}, ", buffer[row+col*self.rows])?;
                        }
                        write!(f, "{}]\n", buffer[row+(self.cols-1)*self.rows])?;
                    }
                }
                Ok(())
            }
        }
    };
}
macro_rules! impl_CuMatrixOpMut_fragmented {
    ( $name:ty $( , $lifetimes:tt )* ) => {
        impl_CuMatrixOp_fragmented!($name $(,$lifetimes)*);
        impl<$($lifetimes),*> $crate::CuMatrixOpMut for $name {
            fn as_mut_ptr(&mut self) -> *mut f32 { self.ptr }
            fn as_immutable(&self) -> &::cumatrix::CuMatrixOp { self }
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
                ::cuda_core::cuda_ffi::cuda_memcpy(output.as_mut_ptr(), self.ptr, self.len * $crate::std::mem::size_of::<f32>(), ::cuda_core::cuda_ffi::cudaMemcpyKind::DeviceToHost);
            }
        }

        impl<$($lifetimes),*> ::std::fmt::Debug for $name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
                let len = self.len;
                let mut buffer = vec![0.0; len];
                ::CuMatrixOp::clone_to_host(self, &mut buffer);
                write!(f, "Matrix ({},{}) [{:p}] :\n", self.rows, self.cols, self.ptr)?;
                if self.cols > 0 {
                    for row in 0..self.rows {
                        write!(f, "[")?;
                        for col in 0..self.cols-1 {
                            write!(f, "{}, ", buffer[row+col*self.rows])?;
                        }
                        write!(f, "{}]\n", buffer[row+(self.cols-1)*self.rows])?;
                    }
                }
                Ok(())
            }
        }
    };
}
macro_rules! impl_CuMatrixOpMut_packed {
    ( $name:ty $( , $lifetimes:tt )* ) => {
        impl_CuMatrixOp_packed!($name $(,$lifetimes)*);
        impl<$($lifetimes),*> $crate::CuMatrixOpMut for $name {
            fn as_mut_ptr(&mut self) -> *mut f32 { self.ptr }
            fn as_immutable(&self) -> &::cumatrix::CuMatrixOp { self }

            fn clone_from_host(&mut self, data: &[f32]) {
                ::cuda_core::cuda_ffi::cuda_memcpy(self.ptr, data.as_ptr(), self.len * $crate::std::mem::size_of::<f32>(), ::cuda_core::cuda_ffi::cudaMemcpyKind::HostToDevice);
            }

            fn init(&mut self, value: f32, stream: &CudaStream) {
                unsafe { ::cuvector::ffi::VectorKernel_init(self.ptr, self.len as i32, value, stream.stream) }
            }
            fn add_value(&mut self, value: f32, stream: &CudaStream) {
                unsafe { ::cuvector::ffi::VectorKernel_addValue(self.ptr, self.ptr, self.len as i32, value, stream.stream) }
            }
            fn scale(&mut self, value: f32, stream: &CudaStream) {
                unsafe { ::cuvector::ffi::VectorKernel_scl(self.ptr, self.ptr, self.len as i32, value, stream.stream) }
            }
        }

    };
}