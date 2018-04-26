macro_rules! impl_CuVectorOp {
    ( $name:ty $( , $lifetimes:tt )* ) => {
        impl<$($lifetimes),*> ::cuvector::CuVectorOp for $name {
            fn len(&self) -> usize { self.len }
            fn as_ptr(&self) -> *const f32 { self.ptr }
        }

        impl<$($lifetimes),*> ::meta::codec::Codec for $name {
            type OutputType = ::cuvector::CuVector;

            fn encode(&self) -> String {
                let mut host_data = vec![0.0; self.len];
                ::CuVectorOp::clone_to_host(self, &mut host_data);

                host_data.iter().map(|x| {
                    format!("{} ", x)
                }).collect::<String>()
            }
            fn decode(data: &str) -> ::cuvector::CuVector {
                ::cuvector::CuVector::from_data(
                    data.split_whitespace().map(|x| {
                        x.parse::<f32>().unwrap_or_else(|err| { panic!("{}", err) })
                    }).collect::<Vec<f32>>().as_slice()
                )
            }
        }

        impl<$($lifetimes),*> ::std::fmt::Debug for $name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
                let len = self.len;
                let mut buffer = vec![0.0; len];
                ::CuVectorOp::clone_to_host(self, &mut buffer);
                write!(f, "Vector ({}) : [", len)?;
                for i in 0..len-1 {
                    write!(f, "{}, ", buffer[i])?;
                }
                write!(f, "{}]", buffer[len-1])
            }
        }
    };
}
macro_rules! impl_CuVectorOpMut {
    ( $name:ty $( , $lifetimes:tt )* ) => {
        impl_CuVectorOp!($name $(,$lifetimes)*);
        impl<$($lifetimes),*> ::cuvector::CuVectorOpMut for $name {
            fn as_mut_ptr(&mut self) -> *mut f32 { self.ptr }
        }
    };
}