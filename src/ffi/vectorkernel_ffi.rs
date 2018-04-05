


#[link(name = "vectorkernel")]
extern {
    pub fn VectorKernel_init(vector: *mut f32, len: i32, value: f32);

    pub fn VectorKernel_addValue(vector: *const f32, output: *mut f32, len: i32, value: f32);
    pub fn VectorKernel_scl(vector: *const f32, output: *mut f32, len: i32, value: f32);
    pub fn VectorKernel_add(left_op: *const f32, right_op: *const f32, output: *mut f32, len: i32);
    pub fn VectorKernel_sub(left_op: *const f32, right_op: *const f32, output: *mut f32, len: i32);
    pub fn VectorKernel_pmult(left_op: *const f32, right_op: *const f32, output: *mut f32, len: i32);
    pub fn VectorKernel_sigmoid(vector: *const f32, output: *mut f32, len: i32);
    pub fn VectorKernel_sigmoidDeriv(vector: *const f32, output: *mut f32, len: i32);

    pub fn VectorKernel_aYpb(a: f32, b: f32, Y: *mut f32, len: i32);
    pub fn VectorKernel_aXpb_Y(a: f32, X: *const f32, b: f32, Y: *mut f32, len: i32);
    pub fn VectorKernel_XVpY(X: *const f32, V: *const f32, Y: *mut f32, len: i32);
}



#[cfg(test)]
mod tests {
    use std::{ptr, mem::size_of};
    use ffi::cuda_ffi::*;

    fn assert_equals_float(a: f32, b: f32) {
        let d = a-b;
        if d < -0.000001 || d > 0.000001 {
            panic!("{} != {}", a, b);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn VectorKernel_init() {
        let len = 10;
        let value = 5.0;
        let mut buffer = vec![0.0; len];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(vector as *mut f32, len as i32, value) }

        cuda_memcpy(buffer.as_mut_ptr(), vector, len*size_of::<f32>(), CudaMemcpyKind::DeviceToHost);

        cuda_free(vector);

        buffer.iter().for_each(|x| { assert_equals_float(*x, value) });
    }
    #[test]
    #[allow(non_snake_case)]
    fn VectorKernel_addValue() {
        let len = 10;
        let value0 = 1.0;
        let add_value = 5.0;
        let mut buffer = vec![0.0; len];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(vector as *mut f32, len as i32, value0) }

        unsafe { super::VectorKernel_addValue(vector as *mut f32, vector as *mut f32, len as i32, add_value) }
        cuda_memcpy(buffer.as_mut_ptr(), vector, len*size_of::<f32>(), CudaMemcpyKind::DeviceToHost);

        cuda_free(vector);

        buffer.iter().for_each(|x| { assert_equals_float(*x, value0+add_value) });
    }
    #[test]
    #[allow(non_snake_case)]
    fn VectorKernel_scl() {
        let len = 10;
        let value0 = 1.0;
        let scl_value = 5.0;
        let mut buffer = vec![0.0; len];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(vector as *mut f32, len as i32, value0) }

        unsafe { super::VectorKernel_scl(vector as *mut f32, vector as *mut f32, len as i32, scl_value) }
        cuda_memcpy(buffer.as_mut_ptr(), vector, len*size_of::<f32>(), CudaMemcpyKind::DeviceToHost);

        cuda_free(vector);

        buffer.iter().for_each(|x| { assert_equals_float(*x, value0*scl_value) });
    }
    #[test]
    #[allow(non_snake_case)]
    fn VectorKernel_add() {
        let len = 10;
        let value0 = 1.0;
        let add_value = 5.0;
        let mut buffer = vec![0.0; len];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(vector as *mut f32, len as i32, value0) }

        let mut add_vector = ptr::null_mut();
        cuda_malloc(&mut add_vector, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(add_vector as *mut f32, len as i32, add_value) }

        unsafe { super::VectorKernel_add(vector as *mut f32, add_vector as *mut f32, vector as *mut f32, len as i32) }

        cuda_memcpy(buffer.as_mut_ptr(), vector, len*size_of::<f32>(), CudaMemcpyKind::DeviceToHost);
        cuda_free(vector);
        cuda_free(add_vector);

        buffer.iter().for_each(|x| { assert_equals_float(*x, value0+add_value) });
    }
    #[test]
    #[allow(non_snake_case)]
    fn VectorKernel_sub() {
        let len = 10;
        let value0 = 1.0;
        let sub_value = 5.0;
        let mut buffer = vec![0.0; len];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(vector as *mut f32, len as i32, value0) }

        let mut sub_vector = ptr::null_mut();
        cuda_malloc(&mut sub_vector, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(sub_vector as *mut f32, len as i32, sub_value) }

        unsafe { super::VectorKernel_sub(vector as *mut f32, sub_vector as *mut f32, vector as *mut f32, len as i32) }

        cuda_memcpy(buffer.as_mut_ptr(), vector, len*size_of::<f32>(), CudaMemcpyKind::DeviceToHost);
        cuda_free(vector);
        cuda_free(sub_vector);

        buffer.iter().for_each(|x| { assert_equals_float(*x, value0-sub_value) });
    }
    #[test]
    #[allow(non_snake_case)]
    fn VectorKernel_pmult() {
        let len = 10;
        let value0 = 1.0;
        let mult_value = 5.0;
        let mut buffer = vec![0.0; len];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(vector as *mut f32, len as i32, value0) }

        let mut mult_vector = ptr::null_mut();
        cuda_malloc(&mut mult_vector, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(mult_vector as *mut f32, len as i32, mult_value) }

        unsafe { super::VectorKernel_pmult(vector as *mut f32, mult_vector as *mut f32, vector as *mut f32, len as i32) }

        cuda_memcpy(buffer.as_mut_ptr(), vector, len*size_of::<f32>(), CudaMemcpyKind::DeviceToHost);
        cuda_free(vector);
        cuda_free(mult_vector);

        buffer.iter().for_each(|x| { assert_equals_float(*x, value0*mult_value) });
    }
    #[test]
    #[allow(non_snake_case)]
    fn VectorKernel_aYpb() {
        let a = 2.0;
        let b = -2.0;

        let len = 10;
        let vector_value = 1.0;
        let mut buffer = vec![0.0; len];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(vector as *mut f32, len as i32, vector_value) }

        unsafe { super::VectorKernel_aYpb(a, b, vector as *mut f32, len as i32) }

        cuda_memcpy(buffer.as_mut_ptr(), vector, len*size_of::<f32>(), CudaMemcpyKind::DeviceToHost);
        cuda_free(vector);

        buffer.iter().for_each(|x| { assert_equals_float(*x, a*vector_value+b) });
    }
    #[test]
    #[allow(non_snake_case)]
    fn VectorKernel_aXpb_Y() {
        /*let vector1 = CuVector::new(10, 5.0);
        let vector2 = CuVector::new(10, 7.0);
        unsafe { super::VectorKernel_aXpb_Y(2.0, vector1.data, -2.0, vector2.data, 10) }
        let mut buffer = [0.0; 10];
        vector2.clone_to_host(&mut buffer);
        buffer.iter().for_each(|x| { assert_equals_float(*x, 56.0) })*/
        let a = 2.0;
        let b = -2.0;

        let len = 10;
        let vector_value = 1.0;
        let operator_value = 5.0;
        let mut buffer = vec![0.0; len];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(vector as *mut f32, len as i32, vector_value) }

        let mut operator = ptr::null_mut();
        cuda_malloc(&mut operator, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(operator as *mut f32, len as i32, operator_value) }

        unsafe { super::VectorKernel_aXpb_Y(a, operator as *mut f32, b, vector as *mut f32, len as i32) }

        cuda_memcpy(buffer.as_mut_ptr(), vector, len*size_of::<f32>(), CudaMemcpyKind::DeviceToHost);
        cuda_free(vector);
        cuda_free(operator);

        buffer.iter().for_each(|x| { assert_equals_float(*x, (a*operator_value+b)*vector_value) });
    }
    #[test]
    #[allow(non_snake_case)]
    fn VectorKernel_XVpY() {
        let len = 10;
        let vector_value = 1.0;
        let operator1_value = 5.0;
        let operator2_value = 5.0;
        let mut buffer = vec![0.0; len];

        let mut vector = ptr::null_mut();
        cuda_malloc(&mut vector, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(vector as *mut f32, len as i32, vector_value) }

        let mut operator1 = ptr::null_mut();
        cuda_malloc(&mut operator1, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(operator1 as *mut f32, len as i32, operator1_value) }

        let mut operator2 = ptr::null_mut();
        cuda_malloc(&mut operator2, len*size_of::<f32>());
        unsafe { super::VectorKernel_init(operator2 as *mut f32, len as i32, operator2_value) }

        unsafe { super::VectorKernel_XVpY(operator1 as *mut f32, operator2 as *mut f32, vector as *mut f32, len as i32) }

        cuda_memcpy(buffer.as_mut_ptr(), vector, len*size_of::<f32>(), CudaMemcpyKind::DeviceToHost);
        cuda_free(vector);
        cuda_free(operator1);
        cuda_free(operator2);

        buffer.iter().for_each(|x| { assert_equals_float(*x, operator1_value*operator2_value+vector_value) });
    }
}