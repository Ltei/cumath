
use super::*;



pub struct CuVectorMath;

impl CuVectorMath {

    /// output[i] = vector[i] + value
    pub fn add_value(vector: &CuVectorOp, value: f32, output: &mut CuVectorOpMut, stream: &CudaStream) {
        unsafe { VectorKernel_addValue(vector.as_ptr(), output.as_mut_ptr(), vector.len() as i32, value, stream.stream) }
    }

    /// output[i] = vector[i] * value
    pub fn scale(vector: &CuVectorOp, value: f32, output: &mut CuVectorOpMut, stream: &CudaStream) {
        unsafe { VectorKernel_scl(vector.as_ptr(), output.as_mut_ptr(), vector.len() as i32, value, stream.stream) }
    }

    /// output[i] = left_op[i] + right_op[i]
    pub fn add(left_op: &CuVectorOp, right_op: &CuVectorOp, output: &mut CuVectorOpMut, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(left_op.len(), "left_op.len()", right_op.len(), "right_op.len()");
            assert_eq_usize(left_op.len(), "left_op.len()", output.len(), "output.len()");
        }
        unsafe { VectorKernel_add(left_op.as_ptr(), right_op.as_ptr(), output.as_mut_ptr(), left_op.len() as i32, stream.stream) }
    }

    /// output[i] = left_op[i] - right_op[i]
    pub fn sub(left_op: &CuVectorOp, right_op: &CuVectorOp, output: &mut CuVectorOpMut, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(left_op.len(), "left_op.len()", right_op.len(), "right_op.len()");
            assert_eq_usize(left_op.len(), "left_op.len()", output.len(), "output.len()");
        }
        unsafe { VectorKernel_sub(left_op.as_ptr(),
                                  right_op.as_ptr(),
                                  output.as_mut_ptr(),
                                  left_op.len() as i32, stream.stream) }
    }

    /// output[i] = left_op[i] * right_op[i]
    pub fn pmult(left_op: &CuVectorOp, right_op: &CuVectorOp, output: &mut CuVectorOpMut, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(left_op.len(), "left_op.len()", right_op.len(), "right_op.len()");
            assert_eq_usize(left_op.len(), "left_op.len()", output.len(), "output.len()");
        }
        unsafe { VectorKernel_pmult(left_op.as_ptr(),
                                    right_op.as_ptr(),
                                    output.as_mut_ptr(),
                                    left_op.len() as i32, stream.stream) }
    }

    /// output[i] = sigmoid(vector[i])
    pub fn sigmoid(vector: &CuVectorOp, output: &mut CuVectorOpMut, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(vector.len(), "vector.len()", output.len(), "output.len()");
        }
        unsafe { VectorKernel_sigmoid(vector.as_ptr(), output.as_mut_ptr(), vector.len() as i32, stream.stream) }
    }

    /// output[i] = sigmoid_deriv(vector[i])
    pub fn sigmoid_deriv(vector: &CuVectorOp, output: &mut CuVectorOpMut, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(vector.len(), "vector.len()", output.len(), "output.len()");
        }
        unsafe { VectorKernel_sigmoidDeriv(vector.as_ptr(), output.as_mut_ptr(), vector.len() as i32, stream.stream) }
    }

    pub fn binarize_one_max(vector: &CuVectorOp, output: &mut CuVectorOpMut, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(vector.len(), "vector.len()", output.len(), "output.len()");
        }
        unsafe { VectorKernel_binarizeOneMax(vector.as_ptr(), output.as_mut_ptr(), vector.len() as i32, stream.stream) }
    }

    /// y[i] = a*y[i]+b
    pub fn aypb(a: f32, b: f32, y: &mut CuVectorOpMut, stream: &CudaStream) {
        unsafe { VectorKernel_aYpb(a, b, y.as_mut_ptr(), y.len() as i32, stream.stream) }
    }

    /// y[i] *= (a*x[i])+b
    pub fn axpb_y(a: f32, x: &CuVectorOp, b: f32, y: &mut CuVectorOpMut, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(x.len(), "x.len()", y.len(), "y.len()");
        }
        unsafe { VectorKernel_aXpb_Y(a, x.as_ptr(), b, y.as_mut_ptr(), x.len() as i32, stream.stream) }
    }

    /// y[i] += x[i] * v[i]
    pub fn xvpy(x: &CuVectorOp, v: &CuVectorOp, y: &mut CuVectorOpMut, stream: &CudaStream) {
        #[cfg(not(feature = "disable_checks"))] {
            assert_eq_usize(x.len(), "x.len()", v.len(), "v.len()");
            assert_eq_usize(x.len(), "x.len()", y.len(), "y.len()");
        }
        unsafe { VectorKernel_XVpY(x.as_ptr(), v.as_ptr(), y.as_mut_ptr(), x.len() as i32, stream.stream) }
    }


}