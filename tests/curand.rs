
#[cfg(test)]
mod tests {

    extern crate cumath;
    use self::cumath::*;

    #[test]
    fn curand_generate_uniform() {
        let mut generator = CurandGenerator::new(CurandRngType::PseudoDefault).unwrap();

        let mut vector = CuVector::<f32>::new(0.0, 10);
        generator.generate_uniform(&mut vector);

        let mut buffer = vec![0.0; 10];
        vector.clone_to_host(&mut buffer);

        buffer.iter().for_each(|x| {
            assert!(x >= &0.0 && x <= &1.0);
        });
    }

    #[test]
    fn curand_generate_uniform_range() {
        let min = -5.0;
        let max = 15.0;

        let mut generator = CurandGenerator::new(CurandRngType::PseudoDefault).unwrap();

        let mut vector = CuVector::<f32>::new(0.0, 10);
        generator.generate_uniform_range(&mut vector, min, max, &DEFAULT_STREAM);

        let mut buffer = vec![0.0; 10];
        vector.clone_to_host(&mut buffer);

        buffer.iter().for_each(|x| {
            assert!(*x >= min && *x <= max);
        });
    }

}