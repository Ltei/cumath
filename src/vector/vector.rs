

use std::{ptr, mem::size_of};
use super::*;



pub struct CuVector {
    len: usize,
    ptr: *mut f32,
}
impl Drop for CuVector {
    fn drop(&mut self) {
        cuda_free(self.ptr);
    }
}
impl CuVectorOp for CuVector {
    fn len(&self) -> usize { self.len }
    fn ptr(&self) -> *const f32 { self.ptr }
}
impl CuVectorOpMut for CuVector {
    fn ptr_mut(&mut self) -> *mut f32 { self.ptr }
}
impl CuVector {
    pub fn new(len: usize, init_value: f32) -> CuVector {
        let mut data = ptr::null_mut();
        cuda_malloc(&mut data, len*size_of::<f32>());
        unsafe { VectorKernel_init(data as *mut f32, len as i32, init_value) }
        CuVector { len, ptr: (data as *mut f32) }
    }
    pub fn from_data(data: &[f32]) -> CuVector {
        let mut output = {
            let len = data.len();
            let mut data = ptr::null_mut();
            cuda_malloc(&mut data, len*size_of::<f32>());
            CuVector { len, ptr: (data as *mut f32) }
        };
        output.clone_from_host(data);
        output
    }

    pub fn add_value(vector: &CuVectorOp, value: f32, output: &mut CuVectorOpMut) {
        unsafe { VectorKernel_addValue(vector.ptr(), output.ptr_mut(), vector.len() as i32, value) }
    }
    pub fn scale(vector: &CuVectorOp, value: f32, output: &mut CuVectorOpMut) {
        unsafe { VectorKernel_scl(vector.ptr(), output.ptr_mut(), vector.len() as i32, value) }
    }
    pub fn add(left_op: &CuVectorOp, right_op: &CuVectorOp, output: &mut CuVectorOpMut) {
        assert_eq_usize(left_op.len(), "left_op.len()", right_op.len(), "right_op.len()");
        assert_eq_usize(left_op.len(), "left_op.len()", output.len(), "output.len()");
        unsafe { VectorKernel_add(left_op.ptr(), right_op.ptr(), output.ptr_mut(), left_op.len() as i32) }
    }
    pub fn sub(left_op: &CuVectorOp, right_op: &CuVectorOp, output: &mut CuVectorOpMut) {
        assert_eq_usize(left_op.len(), "left_op.len()", right_op.len(), "right_op.len()");
        assert_eq_usize(left_op.len(), "left_op.len()", output.len(), "output.len()");
        unsafe { VectorKernel_sub(left_op.ptr(),
                                  right_op.ptr(),
                                  output.ptr_mut(),
                                  left_op.len() as i32) }
    }
    pub fn pmult(left_op: &CuVectorOp, right_op: &CuVectorOp, output: &mut CuVectorOpMut) {
        assert_eq_usize(left_op.len(), "left_op.len()", right_op.len(), "right_op.len()");
        assert_eq_usize(left_op.len(), "left_op.len()", output.len(), "output.len()");
        unsafe { VectorKernel_pmult(left_op.ptr(),
                                    right_op.ptr(),
                                    output.ptr_mut(),
                                    left_op.len() as i32) }
    }
    pub fn sigmoid(vector: &CuVectorOp, output: &mut CuVectorOpMut) {
        assert_eq_usize(vector.len(), "vector.len()", output.len(), "output.len()");
        unsafe { VectorKernel_sigmoid(vector.ptr(),output.ptr_mut(), vector.len() as i32) }
    }
    pub fn sigmoid_deriv(vector: &CuVectorOp, output: &mut CuVectorOpMut) {
        assert_eq_usize(vector.len(), "vector.len()", output.len(), "output.len()");
        unsafe { VectorKernel_sigmoidDeriv(vector.ptr(), output.ptr_mut(), vector.len() as i32) }
    }

    /** y[i] = a*y[i]+b */
    pub fn aypb(a: f32, b: f32, y: &mut CuVectorOpMut) {
        unsafe { VectorKernel_aYpb(a, b, y.ptr_mut(), y.len() as i32) }
    }
    /** y[i] *= (a*x[i])+b */
    pub fn axpb_y(a: f32, x: &CuVectorOp, b: f32, y: &mut CuVectorOpMut) {
        assert_eq_usize(x.len(), "x.len()", y.len(), "y.len()");
        unsafe { VectorKernel_aXpb_Y(a, x.ptr(), b, y.ptr_mut(), x.len() as i32) }
    }
    /** y[i] += x[i] * v[i] */
    pub fn xvpy(x: &CuVectorOp, v: &CuVectorOp, y: &mut CuVectorOpMut) {
        assert_eq_usize(x.len(), "x.len()", v.len(), "v.len()");
        assert_eq_usize(x.len(), "x.len()", y.len(), "y.len()");
        unsafe { VectorKernel_XVpY(x.ptr(), v.ptr(), y.ptr_mut(), x.len() as i32) }
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    fn assert_equals_float(a: f32, b: f32) {
        let d = a-b;
        if d < -0.000001 || d > 0.000001 {
            panic!("{} != {}", a, b);
        }
    }

    #[test]
    fn init() {
        let value0 = -0.23254;
        let value1 = 1.1852;
        let mut vector = super::CuVector::new(5, 0.0);

        vector.init(value0);
        vector.slice_mut(1, 3).init(value1);

        let mut output = vec![0.0; 5];
        vector.clone_to_host(&mut output);

        assert_equals_float(output[0], value0);
        assert_equals_float(output[1], value1);
        assert_equals_float(output[2], value1);
        assert_equals_float(output[3], value1);
        assert_equals_float(output[4], value0);
    }
    #[test]
    fn add_self() {
        let value0 = -0.23254;
        let value1 = 1.185254;
        let mut vector0 = super::CuVector::new(5, 0.0);
        let mut vector1 = super::CuVector::new(2, 0.0);

        vector0.init(value0);
        vector1.init(value1);
        vector0.slice_mut(2, 2).add_self(&vector1);

        let mut output = vec![0.0; 5];
        vector0.clone_to_host(&mut output);

        assert_equals_float(output[0], value0);
        assert_equals_float(output[1], value0);
        assert_equals_float(output[2], value0+value1);
        assert_equals_float(output[3], value0+value1);
        assert_equals_float(output[4], value0);
    }
}
