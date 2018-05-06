
#[macro_use]
pub(crate) mod impl_macros;

mod meta;
mod cuda_core;
mod kernel;
mod cuvector;
mod cumatrix;
mod data_iter;

pub use cuda_core::{cuda::*, cublas::*, curand::*};
pub use cuvector::*;
pub use cumatrix::*;
pub use data_iter::*;




use std::fmt::Display;



pub trait CuDataType: Clone + PartialEq + Display {
    #[inline]
    fn zero() -> Self;
}
impl CuDataType for i32 {
    fn zero() -> i32 { 0 }
}
impl CuDataType for f32 {
    fn zero() -> f32 { 0.0 }
}