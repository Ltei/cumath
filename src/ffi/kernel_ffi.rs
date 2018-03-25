


#[link(name = "cudakernel")]
extern {
    pub fn CudaKernel_vectorSet(vector: *mut f32, len: i32, value: f32);

    pub fn CudaKernel_vectorAddSclSelf(vector: *mut f32, len: i32, value: f32);
    pub fn CudaKernel_vectorScaleSelf(vector: *mut f32, len: i32, value: f32);

    pub fn CudaKernel_vectorAdd(left_op: *const f32, right_op: *const f32, output: *mut f32, len: i32);
    pub fn CudaKernel_vectorSub(left_op: *const f32, right_op: *const f32, output: *mut f32, len: i32);
    pub fn CudaKernel_vectorPMult(left_op: *const f32, right_op: *const f32, output: *mut f32, len: i32);
    //pub fn CudaKernel_vectorAddThenScl(vector: *const f32, output: *mut f32, len: i32, addValue: f32, sclValue: f32);
    pub fn CudaKernel_vectorSigmoid(vector: *const f32, output: *mut f32, len: i32);
    pub fn CudaKernel_vectorSigmoidDeriv(vector: *const f32, output: *mut f32, len: i32);
}