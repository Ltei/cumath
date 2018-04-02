

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