
use super::*;



pub struct CuMatrixMath;

impl CuMatrixMath {

    /// y[i][j] = a*y[i][j]+b
    pub fn aypb(a: f32, b: f32, y: &mut CuMatrixOpMut, stream: &CudaStream) {
        unsafe { MatrixKernel_aYpb(a, b, y.as_mut_ptr(), y.leading_dimension() as i32, y.rows() as i32, y.cols() as i32, stream.stream) }
    }

}


#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn aypb() {
        let a = -1.0;
        let b = 22.142544;
        let data = &[0.0, 1.0, 2.0, 0.1, 1.1, 2.1];
        let mut matrix = super::CuMatrix::from_data(3, 2, data);

        CuMatrixMath::aypb(a, b, &mut matrix, &DEFAULT_STREAM);

        let output = &mut [0.0; 6];
        matrix.clone_to_host(output);

        for i in 0..data.len() {
            assert_equals_float(output[i], a*data[i]+b);
        }
    }

}