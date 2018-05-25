


#[cfg(test)]
mod cublas {

    extern crate cumath;
    use self::cumath::*;

    fn assert_equals_float(a: f32, b: f32) {
        let d = a-b;
        if d < -0.000001 || d > 0.000001 {
            panic!("{} != {}", a, b);
        }
    }

    // Level 1

    #[test]
    fn amax_idx() {
        let input_data = vec![-1.0, 2.0, 1.0, -2.0, 7.0, 5.5, -3.7, 1.1, 0.7];

        let cublas = Cublas::new().unwrap();
        let vector = CuVector::<f32>::from_host_data(input_data.as_slice());
        let idx = cublas.amax_idx(&vector);

        assert_eq!(4, idx -1);
    }

    #[test]
    fn amin_idx() {
        let input_data = vec![-1.0, 2.0, 1.0, -2.0, 7.0, 5.5, -3.7, 1.1, -0.7];

        let cublas = Cublas::new().unwrap();
        let vector = CuVector::<f32>::from_host_data(input_data.as_slice());
        let idx = cublas.amin_idx(&vector);

        assert_eq!(8, idx -1);
    }

    #[test]
    fn asum() {
        let input_data = vec![-1.0, 2.0, 1.0, -2.0, 7.0, 5.5, -3.7, 1.1, 0.7];

        let cublas = Cublas::new().unwrap();
        let vector = CuVector::<f32>::from_host_data(input_data.as_slice());
        let asum = cublas.asum(&vector);

        assert_eq!(24.0, asum);
    }

    #[test]
    fn dot() {
        let x_data = vec![-1.0, 1.5, 0.2, -2.0];
        let y_data = vec![-2.0, 7.5, -245.2, -5.0];

        let cublas = Cublas::new().unwrap();
        let mut x = CuVector::<f32>::from_host_data(x_data.as_slice());
        let mut y = CuVector::<f32>::from_host_data(y_data.as_slice());
        let result = cublas.dot(&mut x, &mut y);
        assert_equals_float(result, (0..x_data.len()).fold(0.0, |acc, x| acc + x_data[x] * y_data[x]));
    }

    #[test]
    fn axpy() {
        let x_data = vec![-1.0, 1.5, 0.2, -2.0];
        let y_data = vec![1.0, 0.0, -0.15, 5.0];

        let cublas = Cublas::new().unwrap();
        let x = CuVector::<f32>::from_host_data(x_data.as_slice());
        let mut y = CuVector::<f32>::from_host_data(y_data.as_slice());
        cublas.axpy(2.0, &x, &mut y);

        x.dev_assert_equals(&[-1.0, 1.5, 0.2, -2.0]);
        y.dev_assert_equals(&[-1.0, 3.0, 0.25, 1.0]);
    }

    #[test]
    fn scal() {
        let x_data = vec![-1.0, 1.5, 0.2, -2.0];

        let cublas = Cublas::new().unwrap();
        let mut x = CuVector::<f32>::from_host_data(x_data.as_slice());
        cublas.scal(&mut x, 2.0);

        x.dev_assert_equals(&[-2.0, 3.0, 0.4, -4.0]);
    }

    #[test]
    fn swap() {
        let x_data = vec![-1.0, 1.5, 0.2, -2.0];
        let y_data = vec![-2.0, 7.5, -245.2, -5.0];

        let cublas = Cublas::new().unwrap();
        let mut x = CuVector::<f32>::from_host_data(x_data.as_slice());
        let mut y = CuVector::<f32>::from_host_data(y_data.as_slice());
        cublas.swap(&mut x, &mut y);

        x.dev_assert_equals(&y_data);
        y.dev_assert_equals(&x_data);
    }


    // Level 2

    #[test]
    fn mult_m_col() {
        let col_vector_data = vec![1.0, -2.0, 3.0];
        let matrix_data = vec![-1.5, 2.0, 1.5, -0.5, 1.0, 3.5];

        let cublas = Cublas::new().unwrap();
        let col_vector = CuVector::<f32>::from_host_data(col_vector_data.as_slice());
        let matrix = CuMatrix::<f32>::from_host_data(2, 3, matrix_data.as_slice());
        let mut output = CuVector::<f32>::zero(2);

        cublas.mult_m_col(&matrix, &col_vector, &mut output);

        let mut output_buffer = vec![0.0; 2];
        output.clone_to_host(output_buffer.as_mut_slice());

        assert_equals_float(output_buffer[0], -1.5);
        assert_equals_float(output_buffer[1], 13.5);
    }

    #[test]
    fn mult_col_row() {
        let col_vector_data = vec![2.2, -3.2, 1.1];
        let row_vector_data = vec![-1.0, 2.0];

        let cublas = Cublas::new().unwrap();
        let col_vector = CuVector::<f32>::from_host_data(col_vector_data.as_slice());
        let row_vector = CuVector::<f32>::from_host_data(row_vector_data.as_slice());
        let mut output = CuMatrix::<f32>::zero(3, 2);

        cublas.mult_col_row(&col_vector, &row_vector, &mut output);

        let mut output_buffer = vec![0.0; 6];
        output.clone_to_host(output_buffer.as_mut_slice());

        assert_equals_float(output_buffer[0], -2.2);
        assert_equals_float(output_buffer[1], 3.2);
        assert_equals_float(output_buffer[2], -1.1);
        assert_equals_float(output_buffer[3], 4.4);
        assert_equals_float(output_buffer[4], -6.4);
        assert_equals_float(output_buffer[5], 2.2);
    }

    #[test]
    fn mult_row_m() {
        let vector_data = vec![2.2, -3.2, 1.1];
        let matrix_data = vec![-1.0, 2.0, 1.0, -2.0, 7.0, 5.5];

        let cublas = Cublas::new().unwrap();
        let vector = CuVector::<f32>::from_host_data(vector_data.as_slice());
        let matrix = CuMatrix::<f32>::from_host_data(3, 2, matrix_data.as_slice());
        let mut output = CuVector::<f32>::zero(2);

        cublas.mult_row_m(&vector, &matrix, &mut output);

        let mut output_buffer = vec![0.0; 2];
        output.clone_to_host(output_buffer.as_mut_slice());

        assert_equals_float(output_buffer[0], -7.5);
        assert_equals_float(output_buffer[1], -20.75);
    }


    // Level 3

    #[test]
    fn mult_m_m() {
        let input_data = vec![-1.0, 2.0, 1.0, -2.0, 7.0, 5.5];

        let cublas = Cublas::new().unwrap();
        let matrix1 = CuMatrix::<f32>::from_host_data(2, 3, input_data.as_slice());
        let matrix2 = CuMatrix::<f32>::from_host_data(3, 2, input_data.as_slice());
        let mut output = CuMatrix::<f32>::zero(2, 2);

        cublas.mult_m_m(&matrix1, &matrix2, &mut output);

        let mut output_buffer = vec![0.0; 4];
        output.clone_to_host(output_buffer.as_mut_slice());

        assert_equals_float(output_buffer[0], 10.0);
        assert_equals_float(output_buffer[1], -0.5);
        assert_equals_float(output_buffer[2], 47.5);
        assert_equals_float(output_buffer[3], 12.25);
    }

    /*
    #[test]
    fn benchmark_stream() {
        use std::time::Instant;

        let input_data = vec![-1.0; 900];

        let mut cublas0 = Cublas::new();
        let mut cublas1 = Cublas::new();

        let stream0 = CudaStream::new();
        let stream1 = CudaStream::new();

        let matrix1 = CuMatrix::from_data(30, 30, input_data.as_slice());
        let matrix2 = CuMatrix::from_data(30, 30, input_data.as_slice());
        let mut output0 = CuMatrix::new(30, 30, 0.0);
        let mut output1 = CuMatrix::new(30, 30, 0.0);


        let t0 = Instant::now();
        cublas0.set_stream(&stream0);
        for _ in 0..100000 {
            cublas0.mult_m_m(&matrix1, &matrix2, &mut output0);
        }
        stream0.synchronize();
        let dt = t0.elapsed();
        println!("1 finished in {}.{}", dt.as_secs(), dt.subsec_nanos());

        let t0 = Instant::now();
        for _ in 0..50000 {
            cublas0.set_stream(&stream0);
            cublas0.mult_m_m(&matrix1, &matrix2, &mut output0);
            cublas0.set_stream(&stream1);
            cublas0.mult_m_m(&matrix1, &matrix2, &mut output1);
        }
        stream0.synchronize();
        stream1.synchronize();
        let dt = t0.elapsed();
        println!("2 finished in {}.{}", dt.as_secs(), dt.subsec_nanos());

        let t0 = Instant::now();
        cublas0.set_stream(&stream0);
        cublas1.set_stream(&stream1);
        for _ in 0..50000 {
            cublas0.mult_m_m(&matrix1, &matrix2, &mut output0);
            cublas1.mult_m_m(&matrix1, &matrix2, &mut output1);
        }
        stream0.synchronize();
        stream1.synchronize();
        let dt = t0.elapsed();
        println!("3 finished in {}.{}", dt.as_secs(), dt.subsec_nanos());
    }
    */

}