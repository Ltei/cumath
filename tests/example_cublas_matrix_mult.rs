


#[cfg(test)]
mod example_cublas_matrix_mult {

    extern crate cumath;
    use self::cumath::*;

    fn assert_equals_float(a: f32, b: f32) {
        let d = a-b;
        if d < -0.000001 || d > 0.000001 {
            panic!("{} != {}", a, b);
        }
    }

    #[test]
    fn main() {
        // Create an instance of CuBLAS
        let cublas = Cublas::new().unwrap();

        // Create a 2*2 Matrix containing [1.0, 2.0, -2.0, 4.0] (matrices are row-ordered)
        let matrix1 = CuMatrix::<f32>::from_host_data(2, 2, &[1.0, 2.0, -2.0, 4.0]);
        // Create a 2*2 Matrix containing [2.0, -1.0, 0.0, 1.0]
        let matrix2 = CuMatrix::<f32>::from_host_data(2, 2, &[2.0, -1.0, 0.0, 1.0]);

        // Create a Zero 2*2 Matrix
        let mut output = CuMatrix::<f32>::zero(2, 2);

        // Matrix-Matrix multiplication
        cublas.mult_m_m(&matrix1, &matrix2, &mut output);

        // Copy the data to host memory
        let mut cpu_output = vec![0.0; 4];
        output.clone_to_host(&mut cpu_output);

        assert_equals_float(cpu_output[0], 4.0);
        assert_equals_float(cpu_output[1], 0.0);
        assert_equals_float(cpu_output[2], -2.0);
        assert_equals_float(cpu_output[3], 4.0);
    }

}