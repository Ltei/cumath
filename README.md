# cumath

Cumath is a safe cuda wrapper for Rust : The goal is to make a low-cost abstraction wrapper in order tu use cuda, cublas, curand, and cudnn easily.

Install nvcc before : https://developer.nvidia.com/cuda-toolkit

/!\ This library is still under developement!

/!\ The detection of cuda compatible c compiler is very basic and only works for linux, if it doesn't work for you, you can try to comment this line in build.rs : "std::env::set_var("CXX", get_ccbin().unwrap());"


## Implemented :

- GPU memory management
- Vector
- Matrix
- CuBLAS
- CuRAND
- Cuda streams
- Data type genericity

## To be implemented :

- Finish Cudnn
- User-definable Cuda kernels
- More built-in functions
- Improve automatic cuda-compatible c compiler detection

## Won't be implemented

- Backend choice with CPU (This would rather be a higher level library)

## Getting started

Add Cumath to your Cargo.toml:

    [dependencies]
    cumath = "0.2.2"

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

For more info, run 'cargo doc --open'
