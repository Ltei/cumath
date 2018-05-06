


macro_rules! impl_CuMatrixOpMut_fragmented_i32 {
    ( $name:ident $( , $lifetimes:tt )* ) => {
        impl<$($lifetimes),*> CuMatrixOpMut<i32> for $name<$($lifetimes,)* i32> {
            fn as_mut_ptr(&mut self) -> *mut i32 { self.ptr }
            fn as_immutable(&self) -> &CuMatrixOp<i32> { self }

            fn init(&mut self, value: i32, stream: &CudaStream) {
                unsafe {
                    VectorFragment_init_i32(self.as_mut_ptr(), self.leading_dimension() as i32, value,
                                      self.rows() as i32, self.cols() as i32, stream.stream);
                }
            }
        
            fn add_value(&mut self, value: i32, stream: &CudaStream) {
                unsafe {
                    VectorFragment_addValue_i32(self.as_ptr(), self.leading_dimension() as i32, value,
                                          self.as_mut_ptr(), self.leading_dimension() as i32,
                                          self.rows() as i32, self.cols() as i32, stream.stream);
                }
            }

            fn scl(&mut self, value: i32, stream: &CudaStream) {
                unsafe {
                    VectorFragment_scl_i32(self.as_ptr(), self.leading_dimension() as i32, value,
                                       self.as_mut_ptr(), self.leading_dimension() as i32,
                                       self.rows() as i32, self.cols() as i32, stream.stream);
                }
            }

            fn add(&mut self, to_add: &CuMatrixOp<i32>, stream: &CudaStream) {
                #[cfg(not(feature = "disable_checks"))] {
                    assert_eq_usize(self.rows(), "self.rows()", to_add.rows(), "to_add.rows()");
                    assert_eq_usize(self.cols(), "self.cols()", to_add.cols(), "to_add.cols()");
                }
                unsafe {
                    VectorFragment_add_i32(self.as_ptr(), self.leading_dimension() as i32,
                                     to_add.as_ptr(), to_add.leading_dimension() as i32,
                                     self.as_mut_ptr(), self.leading_dimension() as i32,
                                     self.rows() as i32, self.cols() as i32, stream.stream)
                }
            }
        }
    };
}


macro_rules! impl_CuMatrixOpMut_packed_i32 {
    ( $name:ident $( , $lifetimes:tt )* ) => {
        impl<$($lifetimes,)*> ::cumatrix::CuMatrixOpMut<i32> for $name<$($lifetimes,)* i32> {
            fn as_mut_ptr(&mut self) -> *mut i32 { self.ptr }
            fn as_immutable(&self) -> &::cumatrix::CuMatrixOp<i32> { self }

            fn clone_from_host(&mut self, data: &[i32]) {
                #[cfg(not(feature = "disable_checks"))] {
                    assert_eq_usize(self.len(), "self.len()", data.len(), "data.len()");
                }
                cuda_memcpy(self.as_mut_ptr() as *mut c_void, data.as_ptr() as *const c_void, data.len()*size_of::<i32>(), ::cuda_core::cuda_ffi::cudaMemcpyKind::HostToDevice);
            }

            fn init(&mut self, value: i32, stream: &CudaStream) {
                unsafe { VectorPacked_init_i32(self.as_mut_ptr(), value, self.len() as i32, stream.stream) }
            }
            fn add_value(&mut self, value: i32, stream: &CudaStream) {
                unsafe { VectorPacked_addValue_i32(self.as_ptr(), value, self.as_mut_ptr(), self.len() as i32, stream.stream) }
            }
            fn scl(&mut self, value: i32, stream: &CudaStream) {
                unsafe { VectorPacked_scl_i32(self.as_ptr(), value, self.as_mut_ptr(), self.len() as i32, stream.stream) }
            }
            fn add(&mut self, to_add: &CuMatrixOp<i32>, stream: &CudaStream) {
                #[cfg(not(feature = "disable_checks"))] {
                    assert_eq_usize(self.rows(), "self.rows()", to_add.rows(), "to_add.rows()");
                    assert_eq_usize(self.cols(), "self.cols()", to_add.cols(), "to_add.cols()");
                }
                unsafe {
                    VectorFragment_add_i32(self.as_ptr(), self.leading_dimension() as i32,
                                     to_add.as_ptr(), to_add.leading_dimension() as i32,
                                     self.as_mut_ptr(), self.leading_dimension() as i32,
                                     self.rows() as i32, self.cols() as i32, stream.stream)
                }
            }
        }

    };
}