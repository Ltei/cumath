# cumath

Cumath is a safe cuda wrapper for Rust : The goal is to make a zero-cost wrapper that allows you tu use cuda, cublas, and curand easily.

[Install cuda before using cumath](https://developer.nvidia.com/cuda-toolkit)

/!\ This library is still under developement!

/!\ Cumath look for cuda libraries in /usr/bin/loca/cuda/lib64 (the default cuda path on linux)
    If it doesn't work for you, you can [explicitly specify cuda path](https://stackoverflow.com/questions/26246849/how-to-i-tell-rust-where-to-look-for-a-static-library)


## Implemented :

- GPU memory management
- Vector
- Matrix
- CuBLAS
- CuRAND
- Cuda streams
- Data type genericity

## To be implemented :

- [cumath_nn : a Cudnn wrapper based on cumath](https://github.com/Ltei/cumath_nn)
- User-definable Cuda kernels
- More built-in functions

## Won't be implemented

- Backend choice with CPU (This would rather be a higher level library)

## Getting started

Add Cumath to your Cargo.toml:

    [dependencies]
    cumath = "0.2.7"

Then in your main.rs :

    extern crate cumath;


## Examples

### Simple vector addition

```rust
extern crate cumath;
use cumath::*;

fn assert_equals_float(a: f32, b: f32) {
    let d = a-b;
    if d < -0.000001 || d > 0.000001 {
        panic!("{} != {}", a, b);
    }
}

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

```

### Matrix multiplication using CuBLAS 

```rust
extern crate cumath;
use cumath::*;

fn assert_equals_float(a: f32, b: f32) {
    let d = a-b;
    if d < -0.000001 || d > 0.000001 {
        panic!("{} != {}", a, b);
    }
}

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
```

### Advanced example

```rust
extern crate cumath;
use self::cumath::*;

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
    iter.skip(2);                             // Skip the first 3 elements
    let slice1 = iter.next(10).unwrap();      // Take the next 10 elements
    let slice2 = iter.next(100).unwrap();
    let mut slice3 = iter.last(10).unwrap();  // Take the last 10 elements

    // Convert slice2 into a matrix by taking a matrix slice
    let matrix = slice2.matrix_slice(0 /*slice offset*/, 10 /*rows*/, 10 /*cols*/);

    // Matrix-matrix multiplication with slice1 as a row-matrix
    cublas.mult_row_m(&slice1, &matrix, &mut slice3);
}
```

For more info, run 'cargo doc --open'
