


#[cfg(test)]
mod example_simple_addition {

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
        let value0 = -0.23254;
        let value1 = 1.185254;

        // Create a vector containing [value0, value0, value0, value0, value0]
        let mut vector0 = CuVector::<f32>::new(value0, 5);
        // Create a vector containing [value1]
        let vector1 = CuVector::<f32>::new(value1, 1);

        {
            // Borrow a slice of vector0 with offset 2 and length 1
            let mut slice = vector0.slice_mut(2, 1);
            // Add vector1 to the slice
            slice.add(&vector1, &DEFAULT_STREAM);
        }

        // Copy the data to host memory
        let mut output = vec![0.0; 5];
        vector0.clone_to_host(&mut output);

        assert_equals_float(output[0], value0);
        assert_equals_float(output[1], value0);
        assert_equals_float(output[2], value0+value1);
        assert_equals_float(output[3], value0);
        assert_equals_float(output[4], value0);
    }

}