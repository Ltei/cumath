

macro_rules! impl_CuMatrixOpMut_fragmented_f32 {
    ( $name:ident $( , $lifetimes:tt )* ) => {
        impl<$($lifetimes),*> CuMatrixOpMut<f32> for $name<$($lifetimes,)* f32> {
            fn as_mut_ptr(&mut self) -> *mut f32 { self.ptr }
            fn as_immutable(&self) -> &CuMatrixOp<f32> { self }

            fn init(&mut self, value: f32, stream: &CudaStream) {
                unsafe {
                    VectorFragment_init_f32(self.as_mut_ptr(), self.leading_dimension() as i32, value,
                                      self.rows() as i32, self.cols() as i32, stream.stream);
                }
            }

            fn add_value(&mut self, value: f32, stream: &CudaStream) {
                unsafe {
                    VectorFragment_addValue_f32(self.as_ptr(), self.leading_dimension() as i32, value,
                                          self.as_mut_ptr(), self.leading_dimension() as i32,
                                          self.rows() as i32, self.cols() as i32, stream.stream);
                }
            }

            fn scl(&mut self, value: f32, stream: &CudaStream) {
                unsafe {
                    VectorFragment_scl_f32(self.as_ptr(), self.leading_dimension() as i32, value,
                                       self.as_mut_ptr(), self.leading_dimension() as i32,
                                       self.rows() as i32, self.cols() as i32, stream.stream);
                }
            }

            fn add(&mut self, to_add: &CuMatrixOp<f32>, stream: &CudaStream) {
                #[cfg(not(feature = "disable_checks"))] {
                    assert_eq_usize(self.rows(), "self.rows()", to_add.rows(), "to_add.rows()");
                    assert_eq_usize(self.cols(), "self.cols()", to_add.cols(), "to_add.cols()");
                }
                unsafe {
                    VectorFragment_add_f32(self.as_ptr(), self.leading_dimension() as i32,
                                     to_add.as_ptr(), to_add.leading_dimension() as i32,
                                     self.as_mut_ptr(), self.leading_dimension() as i32,
                                     self.rows() as i32, self.cols() as i32, stream.stream)
                }
            }
        }
    };
}


macro_rules! impl_CuMatrixOpMut_packed_f32 {
    ( $name:ident $( , $lifetimes:tt )* ) => {
        impl<$($lifetimes,)*> ::cumatrix::CuMatrixOpMut<f32> for $name<$($lifetimes,)* f32> {
            fn as_mut_ptr(&mut self) -> *mut f32 { self.ptr }
            fn as_immutable(&self) -> &::cumatrix::CuMatrixOp<f32> { self }

            fn clone_from_host(&mut self, data: &[f32]) {
                #[cfg(not(feature = "disable_checks"))] {
                    assert_eq_usize(self.len(), "self.len()", data.len(), "data.len()");
                }
                cuda_memcpy(self.as_mut_ptr() as *mut c_void, data.as_ptr() as *const c_void, data.len()*size_of::<f32>(), ::cuda_core::cuda_ffi::cudaMemcpyKind::HostToDevice);
            }

            fn init(&mut self, value: f32, stream: &CudaStream) {
                unsafe { VectorPacked_init_f32(self.as_mut_ptr(), value, self.len() as i32, stream.stream) }
            }
            fn add_value(&mut self, value: f32, stream: &CudaStream) {
                unsafe { VectorPacked_addValue_f32(self.as_ptr(), value, self.as_mut_ptr(), self.len() as i32, stream.stream) }
            }
            fn scl(&mut self, value: f32, stream: &CudaStream) {
                unsafe { VectorPacked_scl_f32(self.as_ptr(), value, self.as_mut_ptr(), self.len() as i32, stream.stream) }
            }
            fn add(&mut self, to_add: &CuMatrixOp<f32>, stream: &CudaStream) {
                #[cfg(not(feature = "disable_checks"))] {
                    assert_eq_usize(self.rows(), "self.rows()", to_add.rows(), "to_add.rows()");
                    assert_eq_usize(self.cols(), "self.cols()", to_add.cols(), "to_add.cols()");
                }
                unsafe {
                    VectorFragment_add_f32(self.as_ptr(), self.leading_dimension() as i32,
                                     to_add.as_ptr(), to_add.leading_dimension() as i32,
                                     self.as_mut_ptr(), self.leading_dimension() as i32,
                                     self.rows() as i32, self.cols() as i32, stream.stream)
                }
            }
        }

    };
}