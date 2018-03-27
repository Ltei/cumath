


#[link(name = "cudakernel")]
extern {
    pub fn CudaKernel_vectorSet(vector: *mut f32, len: i32, value: f32);

    pub fn CudaKernel_vectorAddScl(vector: *const f32, output: *mut f32, len: i32, value: f32);
    pub fn CudaKernel_vectorScl(vector: *const f32, output: *mut f32, len: i32, value: f32);
    pub fn CudaKernel_vectorAdd(left_op: *const f32, right_op: *const f32, output: *mut f32, len: i32);
    pub fn CudaKernel_vectorSub(left_op: *const f32, right_op: *const f32, output: *mut f32, len: i32);
    pub fn CudaKernel_vectorPMult(left_op: *const f32, right_op: *const f32, output: *mut f32, len: i32);
    pub fn CudaKernel_vectorSigmoid(vector: *const f32, output: *mut f32, len: i32);
    pub fn CudaKernel_vectorSigmoidDeriv(vector: *const f32, output: *mut f32, len: i32);

    pub fn CudaKernel_aXpb_Y(a: f32, X: *const f32, b: f32, Y: *mut f32, len: i32);
    pub fn CudaKernel_XVpY(X: *const f32, V: *const f32, Y: *mut f32, len: i32);
}



#[cfg(test)]
mod tests {
    use cuvector::*;

    fn assert_equals_float(a: f32, b: f32) {
        let d = a-b;
        if d < -0.000001 || d > 0.000001 {
            panic!("{} != {}", a, b);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn CudaKernel_vectorSet() {
        let vector = CuVector::new(10, 0.0);
        unsafe { super::CudaKernel_vectorSet(vector.data, 10, 5.0) }
        let mut buffer = [0.0; 10];
        vector.clone_to_host(&mut buffer);
        buffer.iter().for_each(|x| { assert_equals_float(*x, 5.0) })
    }
    #[test]
    #[allow(non_snake_case)]
    fn CudaKernel_vectorAddScl() {
        let vector = CuVector::new(10, 0.0);
        unsafe { super::CudaKernel_vectorAddScl(vector.data, vector.data, 10, 5.0) }
        let mut buffer = [0.0; 10];
        vector.clone_to_host(&mut buffer);
        buffer.iter().for_each(|x| { assert_equals_float(*x, 5.0) })
    }
    #[test]
    #[allow(non_snake_case)]
    fn CudaKernel_vectorScl() {
        let vector = CuVector::new(10, 5.0);
        unsafe { super::CudaKernel_vectorScl(vector.data, vector.data, 10, 5.0) }
        let mut buffer = [0.0; 10];
        vector.clone_to_host(&mut buffer);
        buffer.iter().for_each(|x| { assert_equals_float(*x, 25.0) })
    }
    #[test]
    #[allow(non_snake_case)]
    fn CudaKernel_vectorAdd() {
        let vector1 = CuVector::new(10, 5.0);
        let vector2 = CuVector::new(10, 7.0);
        unsafe { super::CudaKernel_vectorAdd(vector1.data, vector2.data, vector1.data, 10) }
        let mut buffer = [0.0; 10];
        vector1.clone_to_host(&mut buffer);
        buffer.iter().for_each(|x| { assert_equals_float(*x, 12.0) })
    }
    #[test]
    #[allow(non_snake_case)]
    fn CudaKernel_vectorSub() {
        let vector1 = CuVector::new(10, 5.0);
        let vector2 = CuVector::new(10, 7.0);
        unsafe { super::CudaKernel_vectorSub(vector1.data, vector2.data, vector1.data, 10) }
        let mut buffer = [0.0; 10];
        vector1.clone_to_host(&mut buffer);
        buffer.iter().for_each(|x| { assert_equals_float(*x, -2.0) })
    }
    #[test]
    #[allow(non_snake_case)]
    fn CudaKernel_vectorPMult() {
        let vector1 = CuVector::new(10, 5.0);
        let vector2 = CuVector::new(10, 7.0);
        unsafe { super::CudaKernel_vectorPMult(vector1.data, vector2.data, vector1.data, 10) }
        let mut buffer = [0.0; 10];
        vector1.clone_to_host(&mut buffer);
        buffer.iter().for_each(|x| { assert_equals_float(*x, 35.0) })
    }

    #[test]
    #[allow(non_snake_case)]
    fn CudaKernel_aXpb_Y() {
        let vector1 = CuVector::new(10, 5.0);
        let vector2 = CuVector::new(10, 7.0);
        unsafe { super::CudaKernel_aXpb_Y(2.0, vector1.data, -2.0, vector2.data, 10) }
        let mut buffer = [0.0; 10];
        vector2.clone_to_host(&mut buffer);
        buffer.iter().for_each(|x| { assert_equals_float(*x, 56.0) })
    }
    #[test]
    #[allow(non_snake_case)]
    fn CudaKernel_XVpY() {
        let vector1 = CuVector::new(10, 5.0);
        let vector2 = CuVector::new(10, 7.0);
        let vector3 = CuVector::new(10, -3.0);
        unsafe { super::CudaKernel_XVpY(vector1.data, vector2.data,vector3.data, 10) }
        let mut buffer = [0.0; 10];
        vector3.clone_to_host(&mut buffer);
        buffer.iter().for_each(|x| { assert_equals_float(*x, 32.0) })
    }
}