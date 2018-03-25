# cumath
Cuda-based matrix/vector computations

CuVector and CuMatrix are allocated on device during their lifetime.

Example :
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
    let mut vector0 = CuVector::new(5);
    let mut vector1 = CuVector::new(2);

    vector0.init(value0);
    vector1.init(value1);
    vector0.slice_mut(2, 2).add_self(&vector1);

    let mut output = vec![0.0; 5];
    vector0.copy_to_host(&mut output);

    assert_equals_float(output[0], value0);
    assert_equals_float(output[1], value0);
    assert_equals_float(output[2], value0+value1);
    assert_equals_float(output[3], value0+value1);
    assert_equals_float(output[4], value0);
}

```
