
use cuda_core::cuda_ffi::*;


#[link(name = "vectorkernel")]
extern {
    pub fn VectorKernel_init(vector: *mut f32, len: i32, value: f32, stream: cudaStream_t);

    pub fn VectorKernel_addValue(vector: *const f32, output: *mut f32, len: i32, value: f32, stream: cudaStream_t);
    pub fn VectorKernel_scl(vector: *const f32, output: *mut f32, len: i32, value: f32, stream: cudaStream_t);
    pub fn VectorKernel_add(left_op: *const f32, right_op: *const f32, output: *mut f32, len: i32, stream: cudaStream_t);
    pub fn VectorKernel_sub(left_op: *const f32, right_op: *const f32, output: *mut f32, len: i32, stream: cudaStream_t);
    pub fn VectorKernel_pmult(left_op: *const f32, right_op: *const f32, output: *mut f32, len: i32, stream: cudaStream_t);
    pub fn VectorKernel_psquare(vector: *const f32, output: *mut f32, len: i32, stream: cudaStream_t);
    pub fn VectorKernel_sigmoid(vector: *const f32, output: *mut f32, len: i32, stream: cudaStream_t);
    pub fn VectorKernel_sigmoidDeriv(vector: *const f32, output: *mut f32, len: i32, stream: cudaStream_t);
    pub fn VectorKernel_tanh(vector: *const f32, output: *mut f32, len: i32, stream: cudaStream_t);


    pub fn VectorKernel_binarize(vector: *const f32, threshold: f32, output: *mut f32, len: i32, stream: cudaStream_t);
    pub fn VectorKernel_binarizeOneMax(vector: *const f32, output: *mut f32, len: i32, stream: cudaStream_t);
    pub fn VectorKernel_customErrorCalc(vector: *const f32, ideal_vector: *const f32,
                                        threshold: f32, scale_foff: f32, scale_fon: f32,
                                        output: *mut f32, len: i32, stream: cudaStream_t);

    pub fn VectorKernel_aYpb(a: f32, b: f32, Y: *mut f32, len: i32, stream: cudaStream_t);
    pub fn VectorKernel_aXpb_Y(a: f32, X: *const f32, b: f32, Y: *mut f32, len: i32, stream: cudaStream_t);
    pub fn VectorKernel_XVpY(X: *const f32, V: *const f32, Y: *mut f32, len: i32, stream: cudaStream_t);
    pub fn VectorKernel_X_aVpb_Y(X: *const f32, a: f32, V: *const f32, b: f32, Y: *mut f32, len: i32, stream: cudaStream_t);
}



#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]

    use super::*;
    use std::{ptr, mem::size_of};
    use ::cuda_core::{cuda::*};
    use meta::assert::*;

    #[test]
    fn VectorKernel_init() {
        let len = 10;
        let value = 5.0;
        let mut buffer = vec![0.0; len];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(vector as *mut f32, len as i32, value, DEFAULT_STREAM.stream) }

        cuda_memcpy(buffer.as_mut_ptr(), vector, len*size_of::<f32>(), cudaMemcpyKind::DeviceToHost);

        cuda_free(vector);

        buffer.iter().for_each(|x| { assert_equals_float(*x, value) });
    }

    #[test]
    fn VectorKernel_addValue() {
        let len = 10;
        let value0 = 1.0;
        let add_value = 5.0;
        let mut buffer = vec![0.0; len];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(vector as *mut f32, len as i32, value0, DEFAULT_STREAM.stream) }

        unsafe { super::VectorKernel_addValue(vector as *mut f32, vector as *mut f32, len as i32, add_value, DEFAULT_STREAM.stream) }
        cuda_memcpy(buffer.as_mut_ptr(), vector, len*size_of::<f32>(), cudaMemcpyKind::DeviceToHost);

        cuda_free(vector);

        buffer.iter().for_each(|x| { assert_equals_float(*x, value0+add_value) });
    }
    #[test]
    fn VectorKernel_scl() {
        let len = 10;
        let value0 = 1.0;
        let scl_value = 5.0;
        let mut buffer = vec![0.0; len];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(vector as *mut f32, len as i32, value0, DEFAULT_STREAM.stream) }

        unsafe { super::VectorKernel_scl(vector as *mut f32, vector as *mut f32, len as i32, scl_value, DEFAULT_STREAM.stream) }
        cuda_memcpy(buffer.as_mut_ptr(), vector, len*size_of::<f32>(), cudaMemcpyKind::DeviceToHost);

        cuda_free(vector);

        buffer.iter().for_each(|x| { assert_equals_float(*x, value0*scl_value) });
    }
    #[test]
    fn VectorKernel_add() {
        let len = 10;
        let value0 = 1.0;
        let add_value = 5.0;
        let mut buffer = vec![0.0; len];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(vector as *mut f32, len as i32, value0, DEFAULT_STREAM.stream) }

        let mut add_vector = ptr::null_mut();
        cuda_malloc(&mut add_vector, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(add_vector as *mut f32, len as i32, add_value, DEFAULT_STREAM.stream) }

        unsafe { super::VectorKernel_add(vector as *mut f32, add_vector as *mut f32, vector as *mut f32, len as i32, DEFAULT_STREAM.stream) }

        cuda_memcpy(buffer.as_mut_ptr(), vector, len*size_of::<f32>(), cudaMemcpyKind::DeviceToHost);
        cuda_free(vector);
        cuda_free(add_vector);

        buffer.iter().for_each(|x| { assert_equals_float(*x, value0+add_value) });
    }
    #[test]
    fn VectorKernel_sub() {
        let len = 10;
        let value0 = 1.0;
        let sub_value = 5.0;
        let mut buffer = vec![0.0; len];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(vector as *mut f32, len as i32, value0, DEFAULT_STREAM.stream) }

        let mut sub_vector = ptr::null_mut();
        cuda_malloc(&mut sub_vector, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(sub_vector as *mut f32, len as i32, sub_value, DEFAULT_STREAM.stream) }

        unsafe { super::VectorKernel_sub(vector as *mut f32, sub_vector as *mut f32, vector as *mut f32, len as i32, DEFAULT_STREAM.stream) }

        cuda_memcpy(buffer.as_mut_ptr(), vector, len*size_of::<f32>(), cudaMemcpyKind::DeviceToHost);
        cuda_free(vector);
        cuda_free(sub_vector);

        buffer.iter().for_each(|x| { assert_equals_float(*x, value0-sub_value) });
    }
    #[test]
    fn VectorKernel_pmult() {
        let len = 10;
        let value0 = 1.0;
        let mult_value = 5.0;
        let mut buffer = vec![0.0; len];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(vector as *mut f32, len as i32, value0, DEFAULT_STREAM.stream) }

        let mut mult_vector = ptr::null_mut();
        cuda_malloc(&mut mult_vector, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(mult_vector as *mut f32, len as i32, mult_value, DEFAULT_STREAM.stream) }

        unsafe { super::VectorKernel_pmult(vector as *mut f32, mult_vector as *mut f32, vector as *mut f32, len as i32, DEFAULT_STREAM.stream) }

        cuda_memcpy(buffer.as_mut_ptr(), vector, len*size_of::<f32>(), cudaMemcpyKind::DeviceToHost);
        cuda_free(vector);
        cuda_free(mult_vector);

        buffer.iter().for_each(|x| { assert_equals_float(*x, value0*mult_value) });
    }
    #[test]
    fn VectorKernel_psquare() {
        let len = 10;
        let value0 = -54.0105;
        let mut buffer = vec![0.0; len];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(vector as *mut f32, len as i32, value0, DEFAULT_STREAM.stream) }

        unsafe { super::VectorKernel_psquare(vector as *mut f32,vector as *mut f32, len as i32, DEFAULT_STREAM.stream) }

        cuda_memcpy(buffer.as_mut_ptr(), vector, len*size_of::<f32>(), cudaMemcpyKind::DeviceToHost);
        cuda_free(vector);

        buffer.iter().for_each(|x| { assert_equals_float(*x, value0*value0) });
    }

    #[test]
    fn VectorKernel_binarize() {
        let len = 10;
        let threshold = 4.0;
        let mut buffer = vec![0.0; len];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(vector as *mut f32, len as i32, 2.0, DEFAULT_STREAM.stream) }

        unsafe { super::VectorKernel_binarize(vector as *mut f32, threshold, vector as *mut f32, len as i32, DEFAULT_STREAM.stream) }

        cuda_memcpy(buffer.as_mut_ptr(), vector, len*size_of::<f32>(), cudaMemcpyKind::DeviceToHost);
        cuda_free(vector);

        buffer.iter().for_each(|x| { assert_equals_float(*x, 0.0) });
    }
    #[test]
    fn VectorKernel_binarizeOneMax() {
        let mut buffer = vec![-2.0, 1.2, -5.2, 1.0326, 3.56, 1.0, 1.0];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, buffer.len()*size_of::<f32>());
        cuda_memcpy(vector, buffer.as_ptr(), buffer.len()*size_of::<f32>(), cudaMemcpyKind::HostToDevice);

        unsafe { super::VectorKernel_binarizeOneMax(vector as *mut f32, vector as *mut f32, buffer.len() as i32, DEFAULT_STREAM.stream) }

        cuda_memcpy(buffer.as_mut_ptr(), vector, buffer.len()*size_of::<f32>(), cudaMemcpyKind::DeviceToHost);
        cuda_free(vector);

        for i in 0..buffer.len() {
            if i == 4 {
                assert_equals_float(buffer[i], 1.0);
            } else {
                assert_equals_float(buffer[i], 0.0);
            }
        }
    }
    /*#[test]
    fn VectorKernel_customErrorCalc() {
        let mut buffer = vec![-2.0, 1.2, -5.2, 1.0326, 3.56, 1.0, 1.0];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, buffer.len()*size_of::<f32>());
        cuda_memcpy(vector, buffer.as_ptr(), buffer.len()*size_of::<f32>(), cudaMemcpyKind::HostToDevice);

        unsafe { super::VectorKernel_customErrorCalc(vector as *mut f32, vector as *mut f32, buffer.len() as i32, DEFAULT_STREAM.stream) }

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

    #[test]
    fn VectorKernel_aYpb() {
        let a = 2.0;
        let b = -2.0;

        let len = 10;
        let vector_value = 1.0;
        let mut buffer = vec![0.0; len];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(vector as *mut f32, len as i32, vector_value, DEFAULT_STREAM.stream) }

        unsafe { super::VectorKernel_aYpb(a, b, vector as *mut f32, len as i32, DEFAULT_STREAM.stream) }

        cuda_memcpy(buffer.as_mut_ptr(), vector, len*size_of::<f32>(), cudaMemcpyKind::DeviceToHost);
        cuda_free(vector);

        buffer.iter().for_each(|x| { assert_equals_float(*x, a*vector_value+b) });
    }
    #[test]
    fn VectorKernel_aXpb_Y() {
        let a = 2.0;
        let b = -2.0;

        let len = 10;
        let vector_value = 1.0;
        let operator_value = 5.0;
        let mut buffer = vec![0.0; len];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(vector as *mut f32, len as i32, vector_value, DEFAULT_STREAM.stream) }

        let mut operator = ptr::null_mut();
        cuda_malloc(&mut operator, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(operator as *mut f32, len as i32, operator_value, DEFAULT_STREAM.stream) }

        unsafe { super::VectorKernel_aXpb_Y(a, operator as *mut f32, b, vector as *mut f32, len as i32, DEFAULT_STREAM.stream) }

        cuda_memcpy(buffer.as_mut_ptr(), vector, len*size_of::<f32>(), cudaMemcpyKind::DeviceToHost);
        cuda_free(vector);
        cuda_free(operator);

        buffer.iter().for_each(|x| { assert_equals_float(*x, (a*operator_value+b)*vector_value) });
    }
    #[test]
    fn VectorKernel_XVpY() {
        let len = 10;
        let vector_value = 1.0;
        let operator1_value = 5.0;
        let operator2_value = 5.0;
        let mut buffer = vec![0.0; len];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(vector as *mut f32, len as i32, vector_value, DEFAULT_STREAM.stream) }

        let mut operator1 = ptr::null_mut();
        cuda_malloc(&mut operator1, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(operator1 as *mut f32, len as i32, operator1_value, DEFAULT_STREAM.stream) }

        let mut operator2 = ptr::null_mut();
        cuda_malloc(&mut operator2, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(operator2 as *mut f32, len as i32, operator2_value, DEFAULT_STREAM.stream) }

        unsafe { super::VectorKernel_XVpY(operator1 as *mut f32, operator2 as *mut f32, vector as *mut f32, len as i32, DEFAULT_STREAM.stream) }

        cuda_memcpy(buffer.as_mut_ptr(), vector, len*size_of::<f32>(), cudaMemcpyKind::DeviceToHost);
        cuda_free(vector);
        cuda_free(operator1);
        cuda_free(operator2);

        buffer.iter().for_each(|x| { assert_equals_float(*x, operator1_value*operator2_value+vector_value) });
    }
    #[test]
    fn VectorKernel_X_aVpb_Y() {
        let len = 10;
        let vector_value = 1.0;
        let operator1_value = 5.0;
        let operator2_value = 5.0;
        let a = 2.0;
        let b = 3.0;
        let mut buffer = vec![0.0; len];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(vector as *mut f32, len as i32, vector_value, DEFAULT_STREAM.stream) }

        let mut operator1 = ptr::null_mut();
        cuda_malloc(&mut operator1, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(operator1 as *mut f32, len as i32, operator1_value, DEFAULT_STREAM.stream) }

        let mut operator2 = ptr::null_mut();
        cuda_malloc(&mut operator2, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(operator2 as *mut f32, len as i32, operator2_value, DEFAULT_STREAM.stream) }

        unsafe { super::VectorKernel_X_aVpb_Y(operator1 as *mut f32, a, operator2 as *mut f32, b, vector as *mut f32, len as i32, DEFAULT_STREAM.stream) }

        cuda_memcpy(buffer.as_mut_ptr(), vector, len*size_of::<f32>(), cudaMemcpyKind::DeviceToHost);
        cuda_free(vector);
        cuda_free(operator1);
        cuda_free(operator2);

        buffer.iter().for_each(|x| { assert_equals_float(*x, operator1_value*(a*operator2_value+b)+vector_value) });
    }
}