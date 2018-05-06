

macro_rules! impl_CuVectorOp {
    ( $name:ident $( , $lifetimes:tt )* ) => {
        impl<$($lifetimes,)* T: CuDataType> ::cuvector::CuVectorOp<T> for $name<$($lifetimes,)* T> {
            fn len(&self) -> usize { self.len }
            fn as_ptr(&self) -> *const T { self.ptr }
        }

    };
}