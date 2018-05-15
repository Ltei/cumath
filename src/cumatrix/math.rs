
use std::marker::PhantomData;
use super::CuMatrixDeref;
use CuDataType;
use kernel::*;
use cuda_core::cuda::{CudaStream};

#[cfg(not(feature = "disable_checks"))]
use meta::assert::*;



pub struct CuMatrixMath<T: CuDataType> {
    _phantom: PhantomData<T>
}

impl CuMatrixMath<f32> {

    /// y[i][j] = a*y[i][j]+b
    pub fn aypb(a: f32, y: &mut CuMatrixDeref<f32>, b: f32, stream: &CudaStream) {
        unsafe { VectorFragment_aypb_f32(a, y.as_mut_ptr(), y.leading_dimension() as i32, b, y.rows() as i32, y.cols() as i32, stream.stream) }
    }

    pub fn convolution(input: &CuMatrixDeref<f32>, kernel: &CuMatrixDeref<f32>, output: &mut CuMatrixDeref<f32>, row_step: i32, col_step: i32, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize((input.rows()-kernel.rows()) % row_step as usize, "(input.rows()-kernel.rows()) % row_step", 0, "0");
            assert_eq_usize((input.cols()-kernel.cols()) % col_step as usize, "(input.cols()-kernel.cols()) % col_step", 0, "0");
        }
        unsafe { Matrix_convolution(input.as_ptr(), input.rows() as i32,
                                    input.cols() as i32, input.leading_dimension() as i32,
                                    kernel.as_ptr(), kernel.rows() as i32,
                                    kernel.cols() as i32, kernel.leading_dimension() as i32,
                                    row_step, col_step, output.as_mut_ptr(),
                                    output.leading_dimension() as i32, stream.stream) }
    }

}