
use cuda_core::cuda_ffi::*;


#[link(name = "matrixkernel")]
extern {
    pub fn MatrixKernel_init(matrix: *mut f32, ld: i32, rows: i32, cols: i32, value: f32, stream: cudaStream_t);

    pub fn MatrixKernel_addValue(matrix: *const f32, ld: i32, output: *mut f32, output_ld: i32, rows: i32, cols: i32, value: f32, stream: cudaStream_t);
    pub fn MatrixKernel_scale(matrix: *const f32, ld: i32, output: *mut f32, output_ld: i32, rows: i32, cols: i32, value: f32, stream: cudaStream_t);
    pub fn MatrixKernel_add(left_op: *const f32, left_op_ld: i32, right_op: *const f32, right_op_ld: i32, output: *mut f32, output_ld: i32, rows: i32, cols: i32, stream: cudaStream_t);

    pub fn MatrixKernel_aYpb(a: f32, b: f32, Y: *mut f32, Y_ld: i32, rows: i32, cols: i32, stream: cudaStream_t);

    pub fn MatrixKernel_convolution(input: *const f32, input_row: i32, input_cols: i32, input_ld: i32,
                                    kernel: *const f32, kernel_rows: i32, kernel_cols: i32, kernel_ld: i32,
                                    row_step: i32, col_step: i32, output: *mut f32, output_ld: i32, stream: cudaStream_t);
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]

    use super::*;
    use std::{ptr, mem::size_of};
    use ::cuda_core::{cuda::*};
    use ::cumatrix::*;
    use ::meta::assert::*;

    #[test]
    fn MatrixKernel_init() {
        let rows = 2;
        let cols = 2;
        let ld = 3;
        let len = ld*cols;
        let value1 = 1.0;
        let value2 = 2.0;
        let mut buffer = vec![0.0; len as usize];

        let mut matrix = ptr::null_mut();
        cuda_malloc(&mut matrix, (len as usize)*size_of::<f32>());

        unsafe { super::MatrixKernel_init(matrix, ld, rows, cols, value1, ptr::null_mut()) }
        unsafe { super::MatrixKernel_init(matrix.offset(rows as isize), ld, ld-rows, cols, value2, ptr::null_mut()) }
        cuda_memcpy(buffer.as_mut_ptr(), matrix, (len as usize)*size_of::<f32>(), cudaMemcpyKind::DeviceToHost);

        cuda_free(matrix);

        assert_equals_float(buffer[0], value1);
        assert_equals_float(buffer[1], value1);
        assert_equals_float(buffer[2], value2);
        assert_equals_float(buffer[3], value1);
        assert_equals_float(buffer[4], value1);
        assert_equals_float(buffer[5], value2);
    }

    #[test]
    fn MatrixKernel_aYpb() {
        let rows = 5;
        let cols = 3;
        let len = rows*cols;
        let matrix_value = 1.0;
        let a = 2.0;
        let b = -2.0;
        let mut buffer = vec![0.0; len];

        let mut matrix = ptr::null_mut();
        cuda_malloc(&mut matrix, len*size_of::<f32>());
        unsafe { super::MatrixKernel_init(matrix as *mut f32, rows as i32, rows as i32, cols as i32, matrix_value, ptr::null_mut()) }

        unsafe { super::MatrixKernel_aYpb(a, b, matrix as *mut f32, rows as i32, rows as i32, cols as i32, ptr::null_mut()) }

        cuda_memcpy(buffer.as_mut_ptr(), matrix, len*size_of::<f32>(), cudaMemcpyKind::DeviceToHost);
        cuda_free(matrix);

        buffer.iter().for_each(|x| { assert_equals_float(*x, a*matrix_value+b) });
    }

    #[test]
    fn convolution() {

        let input = ::CuMatrix::from_data(3, 3, &[0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, -2.0]);
        let kernel = ::CuMatrix::from_data(2, 1, &[-1.0, 1.0]);
        let mut output = ::CuMatrix::new(2, 3, 0.0);
        unsafe { super::MatrixKernel_convolution(input.as_ptr(), input.rows() as i32,
                                               input.cols() as i32, input.leading_dimension() as i32,
                                               kernel.as_ptr(), kernel.rows() as i32,
                                               kernel.cols() as i32, kernel.leading_dimension() as i32,
                                               1, 1, output.as_mut_ptr(),
                                               output.leading_dimension() as i32, DEFAULT_STREAM.stream) }

        let mut buffer = vec![0.0; 9];
        output.clone_to_host(&mut buffer);

        assert_equals_float(buffer[0], 1.0);
        assert_equals_float(buffer[1], 1.0);
        assert_equals_float(buffer[2], -1.0);
        assert_equals_float(buffer[3], 1.0);
        assert_equals_float(buffer[4], -1.0);
        assert_equals_float(buffer[5], -3.0);

    }
}
