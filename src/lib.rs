

mod ffi;
mod meta;

mod cuda;
pub use cuda::*;
mod cublas;
pub use cublas::*;
mod curand;
pub use curand::*;
mod vector;
pub use vector::*;
mod matrix;
pub use matrix::*;



