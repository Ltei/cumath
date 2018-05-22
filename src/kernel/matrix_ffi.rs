

use cuda_core::cuda_ffi::cudaStream_t;




#[allow(dead_code)]
extern {
    pub fn Matrix_convolution(input: *const f32, input_row: i32, input_cols: i32, input_ld: i32,
                              kernel: *const f32, kernel_rows: i32, kernel_cols: i32, kernel_ld: i32,
                              row_step: i32, col_step: i32, output: *mut f32, output_ld: i32, stream: cudaStream_t);
}



#[cfg(test)]
mod tests {

    use cuda_core::{cuda::*};
    use cumatrix::*;
    use meta::assert::*;

    #[test]
    fn convolution() {
        let input = CuMatrix::from_host_data(3, 3, &[0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, -2.0]);
        let kernel = CuMatrix::from_host_data(2, 1, &[-1.0, 1.0]);
        let mut output = CuMatrix::<f32>::zero(2, 3);
        unsafe { super::Matrix_convolution(input.as_ptr(), input.rows() as i32,
                                           input.cols() as i32, input.leading_dimension() as i32,
                                           kernel.as_ptr(), kernel.rows() as i32,
                                           kernel.cols() as i32, kernel.leading_dimension() as i32,
                                           1, 1, output.as_mut_ptr(),
                                           output.leading_dimension() as i32, DEFAULT_STREAM.stream) }
        let mut buffer = vec![0.0; 6];
        output.clone_to_host(&mut buffer);

        assert_equals_float(buffer[0], 1.0);
        assert_equals_float(buffer[1], 1.0);
        assert_equals_float(buffer[2], -1.0);
        assert_equals_float(buffer[3], 1.0);
        assert_equals_float(buffer[4], -1.0);
        assert_equals_float(buffer[5], -3.0);
    }

}