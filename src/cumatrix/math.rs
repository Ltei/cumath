
use super::{CuMatrixOp, CuMatrixOpMut, ffi::*};
use cuda_core::cuda::{CudaStream};
use meta::assert::*;



pub struct CuMatrixMath;

impl CuMatrixMath {

    /// y[i][j] = a*y[i][j]+b
    pub fn aypb(a: f32, b: f32, y: &mut CuMatrixOpMut, stream: &CudaStream) {
        unsafe { MatrixKernel_aYpb(a, b, y.as_mut_ptr(), y.leading_dimension() as i32, y.rows() as i32, y.cols() as i32, stream.stream) }
    }

    pub fn convolution(input: &CuMatrixOp, kernel: &CuMatrixOp, output: &mut CuMatrixOpMut, row_step: i32, col_step: i32, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize((input.rows()-kernel.rows()) % row_step as usize, "(input.rows()-kernel.rows()) % row_step", 0, "0");
            assert_eq_usize((input.cols()-kernel.cols()) % col_step as usize, "(input.cols()-kernel.cols()) % col_step", 0, "0");
        }
        unsafe { MatrixKernel_convolution(input.as_ptr(), input.rows() as i32,
                                        input.cols() as i32, input.leading_dimension() as i32,
                                        kernel.as_ptr(), kernel.rows() as i32,
                                        kernel.cols() as i32, kernel.leading_dimension() as i32,
                                        row_step, col_step, output.as_mut_ptr(),
                                        output.leading_dimension() as i32, stream.stream) }
    }

}


#[cfg(test)]
mod tests {

    use super::*;
    use ::cumatrix::{CuMatrix, CuMatrixOp};
    use ::cuda_core::cuda::*;

    #[test]
    fn aypb() {
        let a = -1.0;
        let b = 22.142544;
        let data = &[0.0, 1.0, 2.0, 0.1, 1.1, 2.1];
        let mut matrix = CuMatrix::from_data(3, 2, data);

        CuMatrixMath::aypb(a, b, &mut matrix, &DEFAULT_STREAM);

        let output = &mut [0.0; 6];
        matrix.clone_to_host(output);

        for i in 0..data.len() {
            assert_equals_float(output[i], a*data[i]+b);
        }
    }

}