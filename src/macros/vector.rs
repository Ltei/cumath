



macro_rules! impl_CuVectorOp {
    ( $name:ty $( , $lifetimes:tt )* ) => {
        impl<$($lifetimes),*> $crate::CuVectorOp for $name {
            fn len(&self) -> usize { self.len }
            fn as_ptr(&self) -> *const f32 { self.ptr }
        }
        /*impl<$($lifetimes),*> Index<Range> for $name {

        }*/
    };
}
macro_rules! impl_CuVectorOpMut {
    ( $name:ty $( , $lifetimes:tt )* ) => {
        impl<$($lifetimes),*> $crate::CuVectorOpMut for $name {
            fn as_mut_ptr(&mut self) -> *mut f32 { self.ptr }
        }
    };
}