# cumath

Cuda-based matrix/vector computations

Install nvcc before using this library

CuVector and CuMatrix are allocated on device during their lifetime.

/!\ Cuda-compatible c compiler detection is for now very basic, probably unstable and works only for linux anyway. It it doesn't work for you you can try to modify this line in Build.rs "std::env::set_var("CXX", ccbin);" to put your compiler instead of ccbin (gcc, clang...)


## Implemented :

- GPU memory management
- Vector
- Matrix
- CuBLAS
- CuRAND
- Cuda streams

## To be implemented :

- Data type genericity (being able to use integer vectors)
- User-defined Cuda kernels
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
    let mut vector0 = CuVector::new(5, value0);
    // Create a vector containing [value1]
    let vector1 = CuVector::new(1, value1);

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
    let cublas = Cublas::new();

    // Create a 2*2 Matrix containing [2.0, -1.0, 0.0, 1.0] (matrices are row-ordered)
    let matrix1 = CuMatrix::from_data(2, 2, &[1.0, 2.0, -2.0, 4.0]);
    // Create a 2*2 Matrix containing [2.0, -1.0, 0.0, 1.0]
    let matrix2 = CuMatrix::from_data(2, 2, &[2.0, -1.0, 0.0, 1.0]);

    // Create a Zero 2*2 Matrix
    let mut output = CuMatrix::new(2, 2, 0.0);

    // Matrix-Matrix multiplication
    cublas.mult_m_m(&matrix1, &matrix2, &mut output);

    // Copy the data to host memory
    let mut cpu_output = vec![0.0; 6];
    output.clone_to_host(&mut cpu_output);

    assert_equals_float(cpu_output[0], 4.0);
    assert_equals_float(cpu_output[1], 0.0);
    assert_equals_float(cpu_output[2], -2.0);
    assert_equals_float(cpu_output[3], 4.0);
}

```

For more info, run 'cargo doc --open'
