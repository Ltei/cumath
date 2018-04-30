macro_rules! impl_CuVectorOp {
    ( $name:ty $( , $lifetimes:tt )* ) => {
        impl<$($lifetimes),*> ::cuvector::CuVectorOp for $name {
            fn len(&self) -> usize { self.len }
            fn as_ptr(&self) -> *const f32 { self.ptr }
        }

        impl<$($lifetimes),*> ::std::fmt::Debug for $name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
                let len = self.len;
                let mut buffer = vec![0.0; len];
                ::CuVectorOp::clone_to_host(self, &mut buffer);
                if len > 0 {
                    write!(f, "Vector ({}) [{:p}] : [", len, self.ptr)?;
                    for i in 0..len-1 {
                        write!(f, "{}, ", buffer[i])?;
                    }
                    write!(f, "{}]", buffer[len-1])
                } else {
                    write!(f, "Vector ({}) [{:p}] : []", len, self.ptr)
                }
            }
        }
    };
}
macro_rules! impl_CuVectorOpMut {
    ( $name:ty $( , $lifetimes:tt )* ) => {
        impl_CuVectorOp!($name $(,$lifetimes)*);
        impl<$($lifetimes),*> ::cuvector::CuVectorOpMut for $name {
            fn as_mut_ptr(&mut self) -> *mut f32 { self.ptr }
            fn as_immutable(&self) -> &::cuvector::CuVectorOp { self }
        }
    };
}