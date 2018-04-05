

#[link(name = "matrixkernel")]
extern {
    pub fn MatrixKernel_init(matrix: *mut f32, ld: i32,
                             rows: i32, cols: i32,
                             value: f32);

    pub fn MatrixKernel_addValue(matrix: *const f32, ld: i32,
                                 output: *mut f32, output_ld: i32,
                                 rows: i32, cols: i32,
                                 value: f32);

    pub fn MatrixKernel_scale(matrix: *const f32, ld: i32,
                                 output: *mut f32, output_ld: i32,
                                 rows: i32, cols: i32,
                                 value: f32);

    pub fn MatrixKernel_add(left_op: *const f32, left_op_ld: i32,
                            right_op: *const f32, right_op_ld: i32,
                            output: *mut f32, output_ld: i32,
                            rows: i32, cols: i32);

    pub fn MatrixKernel_aYpb(a: f32, b: f32, Y: *mut f32, Y_ld: i32, rows: i32, cols: i32);
}

#[cfg(test)]
mod tests {
    use std::{ptr, mem::size_of};
    use ffi::cuda_ffi::*;

    fn assert_equals_float(a: f32, b: f32) {
        let d = a - b;
        if d < -0.000001 || d > 0.000001 {
            panic!("{} != {}", a, b);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn MatrixKernel_aYpb() {
        let a = 2.0;
        let b = -2.0;

        let rows = 5;
        let cols = 3;
        let len = rows*cols;
        let matrix_value = 1.0;
        let mut buffer = vec![0.0; len];

        let mut matrix = ptr::null_mut();
        cuda_malloc(&mut matrix, len*size_of::<f32>());
        unsafe { super::MatrixKernel_init(matrix as *mut f32, rows as i32, rows as i32, cols as i32, matrix_value) }

        unsafe { super::MatrixKernel_aYpb(a, b, matrix as *mut f32, rows as i32, rows as i32, cols as i32) }

        cuda_memcpy(buffer.as_mut_ptr(), matrix, len*size_of::<f32>(), CudaMemcpyKind::DeviceToHost);
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