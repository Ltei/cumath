
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
        impl_CuPackedData!($name $(,$lifetimes)*);
        impl<$($lifetimes),*> $crate::cudata::CuPackedDataMut for $name {
            fn as_mut_ptr(&mut self) -> *mut f32 { self.ptr }
        }
    };
}