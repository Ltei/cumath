

mod ffi;
mod meta;

mod cuda;
pub use cuda::*;

mod cublas;
pub use cublas::*;

mod curand;
pub use curand::*;


mod cuvector;
pub use cuvector::*;

mod cumatrix;
pub use cumatrix::*;




mod cudata {
    pub trait CuPackedData {
        fn len(&self) -> usize;
        fn as_ptr(&self) -> *const f32;
    }
    pub trait CuPackedDataMut {
        fn len(&self) -> usize;
        fn as_mut_ptr(&mut self) -> *mut f32;
    }
}
