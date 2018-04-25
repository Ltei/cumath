
use ffi::cuda_ffi::*;


#[link(name = "matrixkernel")]
extern {
    pub fn MatrixKernel_init(matrix: *mut f32, ld: i32, rows: i32, cols: i32, value: f32, stream: cudaStream_t);

    pub fn MatrixKernel_addValue(matrix: *const f32, ld: i32, output: *mut f32, output_ld: i32, rows: i32, cols: i32, value: f32, stream: cudaStream_t);
    pub fn MatrixKernel_scale(matrix: *const f32, ld: i32, output: *mut f32, output_ld: i32, rows: i32, cols: i32, value: f32, stream: cudaStream_t);
    pub fn MatrixKernel_add(left_op: *const f32, left_op_ld: i32, right_op: *const f32, right_op_ld: i32, output: *mut f32, output_ld: i32, rows: i32, cols: i32, stream: cudaStream_t);

    pub fn MatrixKernel_aYpb(a: f32, b: f32, Y: *mut f32, Y_ld: i32, rows: i32, cols: i32, stream: cudaStream_t);
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]

    use std::{ptr, mem::size_of};
    use ffi::cuda_ffi::*;

    fn assert_equals_float(a: f32, b: f32) {
        let d = a - b;
        if d < -0.000001 || d > 0.000001 {
            panic!("{} != {}", a, b);
        }
    }

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
}


/*
#[cfg(test)]
mod tests {
    use cumatrix::*;

    fn assert_equals_float(a: f32, b: f32) {
        let d = a - b;
        if d < -0.000001 || d > 0.000001 {
            panic!("{} != {}", a, b);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn MatrixKernel_init() {
        let matrix = CuMatrix::new(3, 3, 0.0);
        unsafe { super::MatrixKernel_init(matrix.data, 3, 3, 3, 5.0) }
        let mut buffer = [0.0; 9];
        matrix.clone_to_host(&mut buffer);
        buffer.iter().for_each(|x| { assert_equals_float(*x, 5.0) })
    }
}*/