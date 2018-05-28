


#[cfg(test)]
mod example_advanced {

    extern crate cumath;
    use self::cumath::*;

    #[test]
    fn main() {
        // Create a CuRAND generator
        let mut generator = CurandGenerator::new(CurandRngType::PseudoDefault).unwrap();
        // Create an instance of CuBLAS
        let cublas = Cublas::new().unwrap();

        // Create a random vector
        let mut data = CuVector::<f32>::zero(125);
        generator.generate_uniform_range(&mut data, -1.0, 1.0, &DEFAULT_STREAM);

        // Create a slice iterator over data
        let mut iter = data.slice_mut_iter();

        // Take slices
        iter.skip(2);                             // Skip the 2 first elements
        let slice1 = iter.next(10).unwrap();      // Take the next 10 elements
        let slice2 = iter.next(100).unwrap();     // Take the next 100 elements
        let mut slice3 = iter.last(10).unwrap();  // Take the last 10 elements

        // Convert slice2 into a matrix by taking a matrix slice
        let matrix = slice2.matrix_slice(0 /*slice offset*/, 10 /*rows*/, 10 /*cols*/);

        // Matrix-matrix multiplication with slice1 as a row-matrix (using cublas gemv)
        cublas.mult_row_m(&slice1, &matrix, &mut slice3);
    }

}