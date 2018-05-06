

macro_rules! impl_CuVectorOpMut_i32 {
    ( $name:ident $( , $lifetimes:tt )* ) => {
        impl<$($lifetimes,)*> ::cuvector::CuVectorOpMut<i32> for $name<$($lifetimes,)* i32>  {
            fn as_mut_ptr(&mut self) -> *mut i32 { self.ptr }
            fn as_immutable(&self) -> &::cuvector::CuVectorOp<i32> { self }

            fn init(&mut self, value: i32, stream: &CudaStream) {
                unsafe { VectorPacked_init_i32(self.as_mut_ptr(), value, self.len() as i32, stream.stream) }
            }

            fn add_value(&mut self, value: i32, stream: &CudaStream) {
                unsafe { VectorPacked_addValue_i32(self.as_ptr(), value, self.as_mut_ptr(), self.len() as i32, stream.stream) }
            }

            fn scl(&mut self, value: i32, stream: &CudaStream) {
                unsafe { VectorPacked_scl_i32(self.as_ptr(), value, self.as_mut_ptr(), self.len() as i32, stream.stream) }
            }

            fn add(&mut self, right_op: &CuVectorOp<i32>, stream: &CudaStream) {
                #[cfg(not(feature = "disable_checks"))] {
                    assert_eq_usize(self.len(), "self.len()", right_op.len(), "right_op.len()");
                }
                unsafe { VectorPacked_add_i32(self.as_ptr(), right_op.as_ptr(), self.as_mut_ptr(), self.len() as i32, stream.stream) }
            }

            fn sub(&mut self, right_op: &CuVectorOp<i32>, stream: &CudaStream) {
                #[cfg(not(feature = "disable_checks"))] {
                    assert_eq_usize(self.len(), "self.len()", right_op.len(), "right_op.len()");
                }
                unsafe { VectorPacked_sub_i32(self.as_ptr(), right_op.as_ptr(), self.as_mut_ptr(), self.len() as i32, stream.stream) }
            }

            fn pmult(&mut self, right_op: &CuVectorOp<i32>, stream: &CudaStream) {
                #[cfg(not(feature = "disable_checks"))] {
                    assert_eq_usize(self.len(), "self.len()", right_op.len(), "right_op.len()");
                }
                unsafe { VectorPacked_mult_i32(self.as_ptr(), right_op.as_ptr(), self.as_mut_ptr(), self.len() as i32, stream.stream) }
            }

            fn psquare(&mut self, stream: &CudaStream) {
                unsafe { VectorPacked_square_i32(self.as_ptr(), self.as_mut_ptr(), self.len() as i32, stream.stream) }
            }

            fn binarize(&mut self, threshold: i32, stream: &CudaStream) {
                unsafe { VectorPacked_binarize_i32(self.as_ptr(), threshold, self.as_mut_ptr(), self.len() as i32, stream.stream) }
            }
        }
    };
}