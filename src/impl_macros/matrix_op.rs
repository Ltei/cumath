


macro_rules! impl_CuMatrixOp_fragmented {
    ( $name:ident $( , $lifetimes:tt )* ) => {
        impl<$($lifetimes,)* T: CuDataType> CuMatrixOp<T> for $name<$($lifetimes,)* T> {
            fn rows(&self) -> usize { self.rows }
            fn cols(&self) -> usize { self.cols }
            fn len(&self) -> usize { self.rows*self.cols }
            fn leading_dimension(&self) -> usize { self.leading_dimension }

            fn as_ptr(&self) -> *const T { self.ptr }
        }

    };
}

macro_rules! impl_CuMatrixOp_packed {
    ( $name:ident $( , $lifetimes:tt )* ) => {
        impl<$($lifetimes,)* T: CuDataType> CuMatrixOp<T> for $name<$($lifetimes,)* T> {
            fn rows(&self) -> usize { self.rows }
            fn cols(&self) -> usize { self.cols }
            fn len(&self) -> usize { self.len }
            fn leading_dimension(&self) -> usize { self.rows }

            fn clone_to_host(&self, output: &mut [T]) {
                cuda_memcpy(output.as_mut_ptr() as *mut c_void, self.as_ptr() as *const c_void, self.len() * size_of::<T>(), cudaMemcpyKind::DeviceToHost);
            }

            fn as_ptr(&self) -> *const T { self.ptr }
        }

    };
}