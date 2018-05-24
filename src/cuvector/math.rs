
use super::*;


/// Vector math operations
pub struct CuVectorMath<T: CuDataType> {
    _phantom: PhantomData<T>
}

impl CuVectorMath<i32> {

    /// output[i] = vector[i] + value
    pub fn add_value(vector: &CuVectorDeref<i32>, value: i32, output: &mut CuVectorDeref<i32>, stream: &CudaStream) {
        unsafe { VectorPacked_addValue_i32(vector.as_ptr(), value, output.as_mut_ptr(), vector.len() as i32, stream.stream) }
    }

    /// output[i] = vector[i] * value
    pub fn scale(vector: &CuVectorDeref<i32>, value: i32, output: &mut CuVectorDeref<i32>, stream: &CudaStream) {
        unsafe { VectorPacked_scl_i32(vector.as_ptr(), value, output.as_mut_ptr(), vector.len() as i32, stream.stream) }
    }

    /// output[i] = left_op[i] + right_op[i]
    pub fn add(left_op: &CuVectorDeref<i32>, right_op: &CuVectorDeref<i32>, output: &mut CuVectorDeref<i32>, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(left_op.len(), "left_op.len()", right_op.len(), "right_op.len()");
            assert_eq_usize(left_op.len(), "left_op.len()", output.len(), "output.len()");
        }
        unsafe { VectorPacked_add_i32(left_op.as_ptr(), right_op.as_ptr(), output.as_mut_ptr(), left_op.len() as i32, stream.stream) }
    }

    /// output[i] = left_op[i] - right_op[i]
    pub fn sub(left_op: &CuVectorDeref<i32>, right_op: &CuVectorDeref<i32>, output: &mut CuVectorDeref<i32>, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(left_op.len(), "left_op.len()", right_op.len(), "right_op.len()");
            assert_eq_usize(left_op.len(), "left_op.len()", output.len(), "output.len()");
        }
        unsafe { VectorPacked_sub_i32(left_op.as_ptr(),
                                      right_op.as_ptr(),
                                      output.as_mut_ptr(),
                                      left_op.len() as i32, stream.stream) }
    }

    /// output[i] = left_op[i] * right_op[i]
    pub fn mult(left_op: &CuVectorDeref<i32>, right_op: &CuVectorDeref<i32>, output: &mut CuVectorDeref<i32>, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(left_op.len(), "left_op.len()", right_op.len(), "right_op.len()");
            assert_eq_usize(left_op.len(), "left_op.len()", output.len(), "output.len()");
        }
        unsafe { VectorPacked_mult_i32(left_op.as_ptr(),
                                       right_op.as_ptr(),
                                       output.as_mut_ptr(),
                                       left_op.len() as i32, stream.stream) }
    }


    /// y[i] = a*y[i]+b
    pub fn aypb(a: i32, b: i32, y: &mut CuVectorDeref<i32>, stream: &CudaStream) {
        unsafe { VectorPacked_aypb_i32(a, y.as_mut_ptr(), b, y.len() as i32, stream.stream) }
    }

    /// y[i] *= (a*x[i])+b
    pub fn axpb_y(a: i32, x: &CuVectorDeref<i32>, b: i32, y: &mut CuVectorDeref<i32>, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(x.len(), "x.len()", y.len(), "y.len()");
        }
        unsafe { VectorPacked_axpb_y_i32(a, x.as_ptr(), b, y.as_mut_ptr(), x.len() as i32, stream.stream) }
    }

    /// y[i] += x[i] * v[i]
    pub fn xvpy(x: &CuVectorDeref<i32>, v: &CuVectorDeref<i32>, y: &mut CuVectorDeref<i32>, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(x.len(), "x.len()", v.len(), "v.len()");
            assert_eq_usize(x.len(), "x.len()", y.len(), "y.len()");
        }
        unsafe { VectorPacked_xvpy_i32(x.as_ptr(), v.as_ptr(), y.as_mut_ptr(), x.len() as i32, stream.stream) }
    }

    /// y[i] += x[i] * (a*v[i]+b)
    pub fn x_avpb_py(x: &CuVectorDeref<i32>, a: i32, v: &CuVectorDeref<i32>, b: i32, y: &mut CuVectorDeref<i32>, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(x.len(), "x.len()", v.len(), "v.len()");
            assert_eq_usize(x.len(), "x.len()", y.len(), "y.len()");
        }
        unsafe { VectorPacked_x_avpb_py_i32(x.as_ptr(), a, v.as_ptr(), b, y.as_mut_ptr(), x.len() as i32, stream.stream) }
    }
    
}

impl CuVectorMath<f32> {

    /// output[i] = vector[i] + value
    pub fn add_value(vector: &CuVectorDeref<f32>, value: f32, output: &mut CuVectorDeref<f32>, stream: &CudaStream) {
        unsafe { VectorPacked_addValue_f32(vector.as_ptr(), value, output.as_mut_ptr(), vector.len() as i32, stream.stream) }
    }

    /// output[i] = vector[i] * value
    pub fn scale(vector: &CuVectorDeref<f32>, value: f32, output: &mut CuVectorDeref<f32>, stream: &CudaStream) {
        unsafe { VectorPacked_scl_f32(vector.as_ptr(), value, output.as_mut_ptr(), vector.len() as i32, stream.stream) }
    }

    /// output[i] = left_op[i] + right_op[i]
    pub fn add(left_op: &CuVectorDeref<f32>, right_op: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(left_op.len(), "left_op.len()", right_op.len(), "right_op.len()");
            assert_eq_usize(left_op.len(), "left_op.len()", output.len(), "output.len()");
        }
        unsafe { VectorPacked_add_f32(left_op.as_ptr(), right_op.as_ptr(), output.as_mut_ptr(), left_op.len() as i32, stream.stream) }
    }

    /// output[i] = left_op[i] - right_op[i]
    pub fn sub(left_op: &CuVectorDeref<f32>, right_op: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(left_op.len(), "left_op.len()", right_op.len(), "right_op.len()");
            assert_eq_usize(left_op.len(), "left_op.len()", output.len(), "output.len()");
        }
        unsafe { VectorPacked_sub_f32(left_op.as_ptr(),
                                  right_op.as_ptr(),
                                  output.as_mut_ptr(),
                                  left_op.len() as i32, stream.stream) }
    }

    /// output[i] = left_op[i] * right_op[i]
    pub fn mult(left_op: &CuVectorDeref<f32>, right_op: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(left_op.len(), "left_op.len()", right_op.len(), "right_op.len()");
            assert_eq_usize(left_op.len(), "left_op.len()", output.len(), "output.len()");
        }
        unsafe { VectorPacked_mult_f32(left_op.as_ptr(),
                                    right_op.as_ptr(),
                                    output.as_mut_ptr(),
                                    left_op.len() as i32, stream.stream) }
    }


    /// y[i] = a*y[i]+b
    pub fn aypb(a: f32, b: f32, y: &mut CuVectorDeref<f32>, stream: &CudaStream) {
        unsafe { VectorPacked_aypb_f32(a, y.as_mut_ptr(), b, y.len() as i32, stream.stream) }
    }

    /// y[i] *= (a*x[i])+b
    pub fn axpb_y(a: f32, x: &CuVectorDeref<f32>, b: f32, y: &mut CuVectorDeref<f32>, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(x.len(), "x.len()", y.len(), "y.len()");
        }
        unsafe { VectorPacked_axpb_y_f32(a, x.as_ptr(), b, y.as_mut_ptr(), x.len() as i32, stream.stream) }
    }

    /// y[i] += x[i] * v[i]
    pub fn xvpy(x: &CuVectorDeref<f32>, v: &CuVectorDeref<f32>, y: &mut CuVectorDeref<f32>, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(x.len(), "x.len()", v.len(), "v.len()");
            assert_eq_usize(x.len(), "x.len()", y.len(), "y.len()");
        }
        unsafe { VectorPacked_xvpy_f32(x.as_ptr(), v.as_ptr(), y.as_mut_ptr(), x.len() as i32, stream.stream) }
    }

    /// y[i] += x[i] * (a*v[i]+b)
    pub fn x_avpb_py(x: &CuVectorDeref<f32>, a: f32, v: &CuVectorDeref<f32>, b: f32, y: &mut CuVectorDeref<f32>, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(x.len(), "x.len()", v.len(), "v.len()");
            assert_eq_usize(x.len(), "x.len()", y.len(), "y.len()");
        }
        unsafe { VectorPacked_x_avpb_py_f32(x.as_ptr(), a, v.as_ptr(), b, y.as_mut_ptr(), x.len() as i32, stream.stream) }
    }


    /// output[i] = sigmoid(vector[i])
    pub fn sigmoid(vector: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(vector.len(), "vector.len()", output.len(), "output.len()");
        }
        unsafe { VectorPacked_sigmoid_f32(vector.as_ptr(), output.as_mut_ptr(), vector.len() as i32, stream.stream) }
    }

    /// output[i] = sigmoid_deriv(vector[i])
    pub fn sigmoid_deriv(vector: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(vector.len(), "vector.len()", output.len(), "output.len()");
        }
        unsafe { VectorPacked_sigmoidDeriv_f32(vector.as_ptr(), output.as_mut_ptr(), vector.len() as i32, stream.stream) }
    }

    /// output[i] = tanh(vector[i])
    pub fn tanh(vector: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(vector.len(), "vector.len()", output.len(), "output.len()");
        }
        unsafe { VectorPacked_tanh_f32(vector.as_ptr(), output.as_mut_ptr(), vector.len() as i32, stream.stream) }
    }

    /// output[i] = tanh(vector[i])
    pub fn tanh_deriv(vector: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(vector.len(), "vector.len()", output.len(), "output.len()");
        }
        unsafe { VectorPacked_tanhDeriv_f32(vector.as_ptr(), output.as_mut_ptr(), vector.len() as i32, stream.stream) }
    }

    /// output[i] = tanh(vector[i])
    pub fn relu(vector: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(vector.len(), "vector.len()", output.len(), "output.len()");
        }
        unsafe { VectorPacked_relu_f32(vector.as_ptr(), output.as_mut_ptr(), vector.len() as i32, stream.stream) }
    }

    /// output[i] = tanh(vector[i])
    pub fn relu_deriv(vector: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(vector.len(), "vector.len()", output.len(), "output.len()");
        }
        unsafe { VectorPacked_reluDeriv_f32(vector.as_ptr(), output.as_mut_ptr(), vector.len() as i32, stream.stream) }
    }

    pub fn custom_error_calc(vector: &mut CuVectorDeref<f32>, ideal_vector: &CuVectorDeref<f32>, threshold: f32, scale_foff: f32, scale_fon: f32, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(vector.len(), "vector.len()", ideal_vector.len(), "ideal_vector.len()");
        }
        unsafe { VectorPacked_customErrorCalc_f32(vector.as_ptr(), ideal_vector.as_ptr(),
                                                  threshold, scale_foff, scale_fon,
                                                  vector.as_mut_ptr(), vector.len() as i32, stream.stream) }
    }

}