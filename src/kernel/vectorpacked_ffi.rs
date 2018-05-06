
use cuda_core::cuda_ffi::cudaStream_t;





#[allow(dead_code)]
extern {
    pub fn VectorPacked_init_i32(vector: *mut i32, value: i32, len: i32, stream: cudaStream_t);
    pub fn VectorPacked_init_f32(vector: *mut f32, value: f32, len: i32, stream: cudaStream_t);
    pub fn VectorPacked_addValue_i32(vector: *const i32, value: i32, output: *mut i32, len: i32, stream: cudaStream_t);
    pub fn VectorPacked_addValue_f32(vector: *const f32, value: f32, output: *mut f32, len: i32, stream: cudaStream_t);
    pub fn VectorPacked_scl_i32(vector: *const i32, value: i32, output: *mut i32, len: i32, stream: cudaStream_t);
    pub fn VectorPacked_scl_f32(vector: *const f32, value: f32, output: *mut f32, len: i32, stream: cudaStream_t);
    pub fn VectorPacked_add_i32(left_op: *const i32, right_op: *const i32, output: *mut i32, len: i32, stream: cudaStream_t);
    pub fn VectorPacked_add_f32(left_op: *const f32, right_op: *const f32, output: *mut f32, len: i32, stream: cudaStream_t);
    pub fn VectorPacked_sub_i32(left_op: *const i32, right_op: *const i32, output: *mut i32, len: i32, stream: cudaStream_t);
    pub fn VectorPacked_sub_f32(left_op: *const f32, right_op: *const f32, output: *mut f32, len: i32, stream: cudaStream_t);
    pub fn VectorPacked_mult_i32(left_op: *const i32, right_op: *const i32, output: *mut i32, len: i32, stream: cudaStream_t);
    pub fn VectorPacked_mult_f32(left_op: *const f32, right_op: *const f32, output: *mut f32, len: i32, stream: cudaStream_t);
    pub fn VectorPacked_square_i32(vector: *const i32, output: *mut i32, len: i32, stream: cudaStream_t);
    pub fn VectorPacked_square_f32(vector: *const f32, output: *mut f32, len: i32, stream: cudaStream_t);
    pub fn VectorPacked_binarize_i32(vector: *const i32, threshold: i32, output: *mut i32, len: i32, stream: cudaStream_t);
    pub fn VectorPacked_binarize_f32(vector: *const f32, threshold: f32, output: *mut f32, len: i32, stream: cudaStream_t);
    pub fn VectorPacked_aypb_i32(a: i32, y: *mut i32, b: i32, len: i32, stream: cudaStream_t);
    pub fn VectorPacked_aypb_f32(a: f32, y: *mut f32, b: f32, len: i32, stream: cudaStream_t);
    pub fn VectorPacked_axpb_y_i32(a: i32, x: *const i32, b: i32, y: *mut i32, len: i32, stream: cudaStream_t);
    pub fn VectorPacked_axpb_y_f32(a: f32, x: *const f32, b: f32, y: *mut f32, len: i32, stream: cudaStream_t);
    pub fn VectorPacked_xvpy_i32(x: *const i32, v: *const i32, y: *mut i32, len: i32, stream: cudaStream_t);
    pub fn VectorPacked_xvpy_f32(x: *const f32, v: *const f32, y: *mut f32, len: i32, stream: cudaStream_t);
    pub fn VectorPacked_x_avpb_py_i32(x: *const i32, a: i32, v: *const i32, b: i32, y: *mut i32, len: i32, stream: cudaStream_t);
    pub fn VectorPacked_x_avpb_py_f32(x: *const f32, a: f32, v: *const f32, b: f32, y: *mut f32, len: i32, stream: cudaStream_t);
    pub fn VectorPacked_sigmoid_f32(vector: *const f32, output: *mut f32, len: i32, stream: cudaStream_t);
    pub fn VectorPacked_sigmoidDeriv_f32(vector: *const f32, output: *mut f32, len: i32, stream: cudaStream_t);
    pub fn VectorPacked_tanh_f32(vector: *const f32, output: *mut f32, len: i32, stream: cudaStream_t);
    pub fn VectorPacked_tanhDeriv_f32(vector: *const f32, output: *mut f32, len: i32, stream: cudaStream_t);
    pub fn VectorPacked_relu_f32(vector: *const f32, output: *mut f32, len: i32, stream: cudaStream_t);
    pub fn VectorPacked_reluDeriv_f32(vector: *const f32, output: *mut f32, len: i32, stream: cudaStream_t);
    pub fn VectorPacked_customErrorCalc_f32(vector: *const f32, ideal_vector: *const f32, threshold: f32, scaleFoff: f32, scaleFon: f32, output: *mut f32, len: i32, stream: cudaStream_t);
}



#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]

    use std::os::raw::c_void;
    use std::{ptr, mem::size_of};
    use cuda_core::{cuda::*, cuda_ffi::*};
    use meta::assert::*;

    #[test]
    fn init() {
        let len = 10;
        let value = 5.0;
        let mut buffer = vec![0.0; len];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, len*size_of::<f32>());
        unsafe { super::VectorPacked_init_f32(vector as *mut f32, value, len as i32, DEFAULT_STREAM.stream) }

        cuda_memcpy(buffer.as_mut_ptr() as *mut c_void, vector, len*size_of::<f32>(), cudaMemcpyKind::DeviceToHost);

        cuda_free(vector);

        buffer.iter().for_each(|x| { assert_equals_float(*x, value) });
    }

    #[test]
    fn addValue() {
        let len = 10;
        let value0 = 1.0;
        let add_value = 5.0;
        let mut buffer = vec![0.0; len];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, len*size_of::<f32>());
        unsafe { super::VectorPacked_init_f32(vector as *mut f32, value0, len as i32, DEFAULT_STREAM.stream) }

        unsafe { super::VectorPacked_addValue_f32(vector as *mut f32, add_value, vector as *mut f32, len as i32, DEFAULT_STREAM.stream) }
        cuda_memcpy(buffer.as_mut_ptr() as *mut c_void, vector, len*size_of::<f32>(), cudaMemcpyKind::DeviceToHost);

        cuda_free(vector);

        buffer.iter().for_each(|x| { assert_equals_float(*x, value0+add_value) });
    }
    #[test]
    fn scl() {
        let len = 10;
        let value0 = 1.0;
        let scl_value = 5.0;
        let mut buffer = vec![0.0; len];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, len*size_of::<f32>());
        unsafe { super::VectorPacked_init_f32(vector as *mut f32, value0, len as i32, DEFAULT_STREAM.stream) }

        unsafe { super::VectorPacked_scl_f32(vector as *mut f32, scl_value, vector as *mut f32, len as i32, DEFAULT_STREAM.stream) }
        cuda_memcpy(buffer.as_mut_ptr() as *mut c_void, vector, len*size_of::<f32>(), cudaMemcpyKind::DeviceToHost);

        cuda_free(vector);

        buffer.iter().for_each(|x| { assert_equals_float(*x, value0*scl_value) });
    }
    #[test]
    fn add() {
        let len = 10;
        let value0 = 1.0;
        let add_value = 5.0;
        let mut buffer = vec![0.0; len];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, len*size_of::<f32>());
        unsafe { super::VectorPacked_init_f32(vector as *mut f32, value0, len as i32, DEFAULT_STREAM.stream) }

        let mut add_vector = ptr::null_mut();
        cuda_malloc(&mut add_vector, len*size_of::<f32>());
        unsafe { super::VectorPacked_init_f32(add_vector as *mut f32, add_value, len as i32, DEFAULT_STREAM.stream) }

        unsafe { super::VectorPacked_add_f32(vector as *const f32, add_vector as *const f32, vector as *mut f32, len as i32, DEFAULT_STREAM.stream) }

        cuda_memcpy(buffer.as_mut_ptr() as *mut c_void, vector, len*size_of::<f32>(), cudaMemcpyKind::DeviceToHost);
        cuda_free(vector);
        cuda_free(add_vector);

        buffer.iter().for_each(|x| { assert_equals_float(*x, value0+add_value) });
    }
    #[test]
    fn sub() {
        let len = 10;
        let value0 = 1.0;
        let sub_value = 5.0;
        let mut buffer = vec![0.0; len];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, len*size_of::<f32>());
        unsafe { super::VectorPacked_init_f32(vector as *mut f32, value0, len as i32, DEFAULT_STREAM.stream) }

        let mut sub_vector = ptr::null_mut();
        cuda_malloc(&mut sub_vector, len*size_of::<f32>());
        unsafe { super::VectorPacked_init_f32(sub_vector as *mut f32, sub_value, len as i32, DEFAULT_STREAM.stream) }

        unsafe { super::VectorPacked_sub_f32(vector as *const f32, sub_vector as *const f32, vector as *mut f32, len as i32, DEFAULT_STREAM.stream) }

        cuda_memcpy(buffer.as_mut_ptr() as *mut c_void, vector, len*size_of::<f32>(), cudaMemcpyKind::DeviceToHost);
        cuda_free(vector);
        cuda_free(sub_vector);

        buffer.iter().for_each(|x| { assert_equals_float(*x, value0-sub_value) });
    }
    #[test]
    fn mult() {
        let len = 10;
        let value0 = 1.0;
        let mult_value = 5.0;
        let mut buffer = vec![0.0; len];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, len*size_of::<f32>());
        unsafe { super::VectorPacked_init_f32(vector as *mut f32, value0, len as i32, DEFAULT_STREAM.stream) }

        let mut mult_vector = ptr::null_mut();
        cuda_malloc(&mut mult_vector, len*size_of::<f32>());
        unsafe { super::VectorPacked_init_f32(mult_vector as *mut f32, mult_value, len as i32, DEFAULT_STREAM.stream) }

        unsafe { super::VectorPacked_mult_f32(vector as *const f32, mult_vector as *const f32, vector as *mut f32, len as i32, DEFAULT_STREAM.stream) }

        cuda_memcpy(buffer.as_mut_ptr() as *mut c_void, vector, len*size_of::<f32>(), cudaMemcpyKind::DeviceToHost);
        cuda_free(vector);
        cuda_free(mult_vector);

        buffer.iter().for_each(|x| { assert_equals_float(*x, value0*mult_value) });
    }
    #[test]
    fn square() {
        let len = 10;
        let value0 = -54.0105;
        let mut buffer = vec![0.0; len];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, len*size_of::<f32>());
        unsafe { super::VectorPacked_init_f32(vector as *mut f32, value0, len as i32, DEFAULT_STREAM.stream) }

        unsafe { super::VectorPacked_square_f32(vector as *const f32, vector as *mut f32, len as i32, DEFAULT_STREAM.stream) }

        cuda_memcpy(buffer.as_mut_ptr() as *mut c_void, vector, len*size_of::<f32>(), cudaMemcpyKind::DeviceToHost);
        cuda_free(vector);

        buffer.iter().for_each(|x| { assert_equals_float(*x, value0*value0) });
    }

    #[test]
    fn aYpb() {
        let a = 2.0;
        let b = -2.0;

        let len = 10;
        let vector_value = 1.0;
        let mut buffer = vec![0.0; len];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, len*size_of::<f32>());
        unsafe { super::VectorPacked_init_f32(vector as *mut f32, vector_value, len as i32, DEFAULT_STREAM.stream) }

        unsafe { super::VectorPacked_aypb_f32(a, vector as *mut f32, b, len as i32, DEFAULT_STREAM.stream) }

        cuda_memcpy(buffer.as_mut_ptr() as *mut c_void, vector, len*size_of::<f32>(), cudaMemcpyKind::DeviceToHost);
        cuda_free(vector);

        buffer.iter().for_each(|x| { assert_equals_float(*x, a*vector_value+b) });
    }
    #[test]
    fn aXpb_Y() {
        let a = 2.0;
        let b = -2.0;

        let len = 10;
        let vector_value = 1.0;
        let operator_value = 5.0;
        let mut buffer = vec![0.0; len];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, len*size_of::<f32>());
        unsafe { super::VectorPacked_init_f32(vector as *mut f32, vector_value, len as i32, DEFAULT_STREAM.stream) }

        let mut operator = ptr::null_mut();
        cuda_malloc(&mut operator, len*size_of::<f32>());
        unsafe { super::VectorPacked_init_f32(operator as *mut f32, operator_value, len as i32, DEFAULT_STREAM.stream) }

        unsafe { super::VectorPacked_axpb_y_f32(a, operator as *mut f32, b, vector as *mut f32, len as i32, DEFAULT_STREAM.stream) }

        cuda_memcpy(buffer.as_mut_ptr() as *mut c_void, vector, len*size_of::<f32>(), cudaMemcpyKind::DeviceToHost);
        cuda_free(vector);
        cuda_free(operator);

        buffer.iter().for_each(|x| { assert_equals_float(*x, (a*operator_value+b)*vector_value) });
    }
    #[test]
    fn XVpY() {
        let len = 10;
        let vector_value = 1.0;
        let operator1_value = 5.0;
        let operator2_value = 5.0;
        let mut buffer = vec![0.0; len];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, len*size_of::<f32>());
        unsafe { super::VectorPacked_init_f32(vector as *mut f32, vector_value, len as i32, DEFAULT_STREAM.stream) }

        let mut operator1 = ptr::null_mut();
        cuda_malloc(&mut operator1, len*size_of::<f32>());
        unsafe { super::VectorPacked_init_f32(operator1 as *mut f32, operator1_value, len as i32, DEFAULT_STREAM.stream) }

        let mut operator2 = ptr::null_mut();
        cuda_malloc(&mut operator2, len*size_of::<f32>());
        unsafe { super::VectorPacked_init_f32(operator2 as *mut f32, operator2_value, len as i32, DEFAULT_STREAM.stream) }

        unsafe { super::VectorPacked_xvpy_f32(operator1 as *mut f32, operator2 as *mut f32, vector as *mut f32, len as i32, DEFAULT_STREAM.stream) }

        cuda_memcpy(buffer.as_mut_ptr() as *mut c_void, vector, len*size_of::<f32>(), cudaMemcpyKind::DeviceToHost);
        cuda_free(vector);
        cuda_free(operator1);
        cuda_free(operator2);

        buffer.iter().for_each(|x| { assert_equals_float(*x, operator1_value*operator2_value+vector_value) });
    }
    #[test]
    fn x_avpb_py() {
        let len = 10;
        let vector_value = 1.0;
        let operator1_value = 5.0;
        let operator2_value = 5.0;
        let a = 2.0;
        let b = 3.0;
        let mut buffer = vec![0.0; len];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, len*size_of::<f32>());
        unsafe { super::VectorPacked_init_f32(vector as *mut f32, vector_value, len as i32, DEFAULT_STREAM.stream) }

        let mut operator1 = ptr::null_mut();
        cuda_malloc(&mut operator1, len*size_of::<f32>());
        unsafe { super::VectorPacked_init_f32(operator1 as *mut f32, operator1_value, len as i32, DEFAULT_STREAM.stream) }

        let mut operator2 = ptr::null_mut();
        cuda_malloc(&mut operator2, len*size_of::<f32>());
        unsafe { super::VectorPacked_init_f32(operator2 as *mut f32, operator2_value, len as i32, DEFAULT_STREAM.stream) }

        unsafe { super::VectorPacked_x_avpb_py_f32(operator1 as *mut f32, a, operator2 as *mut f32, b, vector as *mut f32, len as i32, DEFAULT_STREAM.stream) }

        cuda_memcpy(buffer.as_mut_ptr() as *mut c_void, vector, len*size_of::<f32>(), cudaMemcpyKind::DeviceToHost);
        cuda_free(vector);
        cuda_free(operator1);
        cuda_free(operator2);

        buffer.iter().for_each(|x| { assert_equals_float(*x, operator1_value*(a*operator2_value+b)+vector_value) });
    }

    #[test]
    fn binarize() {
        let len = 10;
        let threshold = 4.0;
        let mut buffer = vec![0.0; len];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, len*size_of::<f32>());
        unsafe { super::VectorPacked_init_f32(vector as *mut f32, 2.0, len as i32, DEFAULT_STREAM.stream) }

        unsafe { super::VectorPacked_binarize_f32(vector as *mut f32, threshold, vector as *mut f32, len as i32, DEFAULT_STREAM.stream) }

        cuda_memcpy(buffer.as_mut_ptr() as *mut c_void, vector, len*size_of::<f32>(), cudaMemcpyKind::DeviceToHost);
        cuda_free(vector);

        buffer.iter().for_each(|x| { assert_equals_float(*x, 0.0) });
    }

    /*#[test]
    fn VectorPacked_customErrorCalc() {
        let mut buffer = vec![-2.0, 1.2, -5.2, 1.0326, 3.56, 1.0, 1.0];
        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, buffer.len()*size_of::<f32>());
        cuda_memcpy(vector, buffer.as_ptr(), buffer.len()*size_of::<f32>(), cudaMemcpyKind::HostToDevice);
        unsafe { super::VectorPacked_customErrorCalc(vector as *mut f32, vector as *mut f32, buffer.len() as i32, DEFAULT_STREAM.stream) }
        cuda_memcpy(buffer.as_mut_ptr(), vector, buffer.len()*size_of::<f32>(), cudaMemcpyKind::DeviceToHost);
        cuda_free(vector);
        buffer.iter().for_each(|x| print!("{}, ", x));
        println!();
        for i in 0..buffer.len() {
            if i == 4 {
                assert_equals_float(buffer[i], 1.0);
            } else {
                assert_equals_float(buffer[i], 0.0);
            }
        }
    }*/

}