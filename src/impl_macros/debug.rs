

macro_rules! impl_Debug_vector {
    ( $name:ident $( , $lifetimes:tt )* ) => {
        impl<$($lifetimes,)* T: CuDataType> ::std::fmt::Debug for $name<$($lifetimes,)* T> {
            fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
                let len = self.len as usize;
                let mut buffer = vec![T::zero(); len];
                cuda_memcpy(buffer.as_mut_ptr() as *mut c_void, self.as_ptr() as *const c_void, self.len()*size_of::<T>(), cudaMemcpyKind::DeviceToHost);
                if len > 0 {
                    write!(f, "Vector ({}) [{:p}] : [", len, self.ptr)?;
                    for i in 0..len-1 {
                        write!(f, "{:.25}, ", buffer[i])?;
                    }
                    write!(f, "{:.25}]", buffer[len-1])
                } else {
                    write!(f, "Vector ({}) [{:p}] : []", len, self.ptr)
                }
            }
        }
    };
}

macro_rules! impl_Debug_matrix_packed {
    ( $name:ident $( , $lifetimes:tt )* ) => {
        impl<$($lifetimes,)* T: CuDataType> ::std::fmt::Debug for $name<$($lifetimes,)* T> {
            fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
                let mut buffer = vec![T::zero(); self.len()];
                cuda_memcpy(buffer.as_mut_ptr() as *mut c_void, self.as_ptr() as *const c_void, self.len()*size_of::<T>(), cudaMemcpyKind::DeviceToHost);
                self.clone_to_host(&mut buffer);
                write!(f, "Matrix ({},{}) [{:p}] :\n", self.rows, self.cols, self.as_ptr())?;
                if self.cols > 0 {
                    for row in 0..self.rows() {
                        write!(f, "[")?;
                        for col in 0..self.cols()-1 {
                            write!(f, "{}, ", buffer[row+col*self.rows()])?;
                        }
                        if row == self.rows()-1 {
                            write!(f, "{}]", buffer[row+(self.cols()-1)*self.rows()])?;
                        } else {
                            write!(f, "{}]\n", buffer[row+(self.cols()-1)*self.rows()])?;
                        }
                    }
                }
                Ok(())
            }
        }
    };
}

macro_rules! impl_Debug_matrix_fragmented {
    ( $name:ident $( , $lifetimes:tt )* ) => {
        impl<$($lifetimes,)* T: CuDataType> ::std::fmt::Debug for $name<$($lifetimes,)* T> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                let len = self.rows() * self.cols();
                let mut buffer = vec![T::zero(); len];
                self.clone_to_host(&mut buffer);
                write!(f, "Matrix ({},{}) [{:p}] :\n", self.rows, self.cols, self.ptr)?;
                if self.cols > 0 {
                    for row in 0..self.rows() {
                        write!(f, "[")?;
                        for col in 0..self.cols()-1 {
                            write!(f, "{}, ", buffer[row+col*self.rows()])?;
                        }
                        if row == self.rows()-1 {
                            write!(f, "{}]", buffer[row+(self.cols()-1)*self.rows()])?;
                        } else {
                            write!(f, "{}]\n", buffer[row+(self.cols()-1)*self.rows()])?;
                        }
                    }
                }
                Ok(())
            }
        }
    };
}

