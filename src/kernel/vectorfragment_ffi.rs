
use cuda_core::cuda_ffi::cudaStream_t;




#[allow(dead_code)]
extern {
    pub fn VectorFragment_init_i32(vector: *mut i32, vector_ld: i32, value: i32, rows: i32, cols: i32, stream: cudaStream_t);
    pub fn VectorFragment_init_f32(vector: *mut f32, vector_ld: i32, value: f32, rows: i32, cols: i32, stream: cudaStream_t);
    pub fn VectorFragment_addValue_i32(vector: *const i32, vector_ld: i32, value: i32, output: *mut i32, output_ld: i32, rows: i32, cols: i32, stream: cudaStream_t);
    pub fn VectorFragment_addValue_f32(vector: *const f32, vector_ld: i32, value: f32, output: *mut f32, output_ld: i32, rows: i32, cols: i32, stream: cudaStream_t);
    pub fn VectorFragment_scl_i32(vector: *const i32, vector_ld: i32, value: i32, output: *mut i32, output_ld: i32, rows: i32, cols: i32, stream: cudaStream_t);
    pub fn VectorFragment_scl_f32(vector: *const f32, vector_ld: i32, value: f32, output: *mut f32, output_ld: i32, rows: i32, cols: i32, stream: cudaStream_t);
    pub fn VectorFragment_add_i32(left_op: *const i32, left_op_ld: i32, right_op: *const i32, right_op_ld: i32, output: *mut i32, output_ld: i32, rows: i32, cols: i32, stream: cudaStream_t);
    pub fn VectorFragment_add_f32(left_op: *const f32, left_op_ld: i32, right_op: *const f32, right_op_ld: i32, output: *mut f32, output_ld: i32, rows: i32, cols: i32, stream: cudaStream_t);
    pub fn VectorFragment_sub_i32(left_op: *const i32, left_op_ld: i32, right_op: *const i32, right_op_ld: i32, output: *mut i32, output_ld: i32, rows: i32, cols: i32, stream: cudaStream_t);
    pub fn VectorFragment_sub_f32(left_op: *const f32, left_op_ld: i32, right_op: *const f32, right_op_ld: i32, output: *mut f32, output_ld: i32, rows: i32, cols: i32, stream: cudaStream_t);
    pub fn VectorFragment_mult_i32(left_op: *const i32, left_op_ld: i32, right_op: *const i32, right_op_ld: i32, output: *mut i32, output_ld: i32, rows: i32, cols: i32, stream: cudaStream_t);
    pub fn VectorFragment_mult_f32(left_op: *const f32, left_op_ld: i32, right_op: *const f32, right_op_ld: i32, output: *mut f32, output_ld: i32, rows: i32, cols: i32, stream: cudaStream_t);
    pub fn VectorFragment_square_i32(vector: *const i32, vector_ld: i32, output: *mut i32, output_ld: i32, rows: i32, cols: i32, stream: cudaStream_t);
    pub fn VectorFragment_square_f32(vector: *const f32, vector_ld: i32, output: *mut f32, output_ld: i32, rows: i32, cols: i32, stream: cudaStream_t);
    pub fn VectorFragment_binarize_i32(vector: *const i32, vector_ld: i32, threshold: i32, output: *mut i32, output_ld: i32, rows: i32, cols: i32, stream: cudaStream_t);
    pub fn VectorFragment_binarize_f32(vector: *const f32, vector_ld: i32, threshold: f32, output: *mut f32, output_ld: i32, rows: i32, cols: i32, stream: cudaStream_t);
    pub fn VectorFragment_aypb_i32(a: i32, y: *mut i32, y_ld: i32, b: i32, rows: i32, cols: i32, stream: cudaStream_t);
    pub fn VectorFragment_aypb_f32(a: f32, y: *mut f32, y_ld: i32, b: f32, rows: i32, cols: i32, stream: cudaStream_t);
    pub fn VectorFragment_axpb_y_i32(a: i32, x: *const i32, x_ld: i32, b: i32, y: *mut i32, y_ld: i32, rows: i32, cols: i32, stream: cudaStream_t);
    pub fn VectorFragment_axpb_y_f32(a: f32, x: *const f32, x_ld: i32, b: f32, y: *mut f32, y_ld: i32, rows: i32, cols: i32, stream: cudaStream_t);
    pub fn VectorFragment_xvpy_i32(x: *const i32, x_ld: i32, v: *const i32, v_ld: i32, y: *mut i32, y_ld: i32, rows: i32, cols: i32, stream: cudaStream_t);
    pub fn VectorFragment_xvpy_f32(x: *const f32, x_ld: i32, v: *const f32, v_ld: i32, y: *mut f32, y_ld: i32, rows: i32, cols: i32, stream: cudaStream_t);
    pub fn VectorFragment_x_avpb_py_i32(x: *const i32, x_ld: i32, a: i32, v: *const i32, v_ld: i32, b: i32, y: *mut i32, y_ld: i32, rows: i32, cols: i32, stream: cudaStream_t);
    pub fn VectorFragment_x_avpb_py_f32(x: *const f32, x_ld: i32, a: f32, v: *const f32, v_ld: i32, b: f32, y: *mut f32, y_ld: i32, rows: i32, cols: i32, stream: cudaStream_t);
    pub fn VectorFragment_sigmoid_f32(vector: *const f32, vector_ld: i32, output: *mut f32, output_ld: i32, rows: i32, cols: i32, stream: cudaStream_t);
    pub fn VectorFragment_sigmoidDeriv_f32(vector: *const f32, vector_ld: i32, output: *mut f32, output_ld: i32, rows: i32, cols: i32, stream: cudaStream_t);
    pub fn VectorFragment_tanh_f32(vector: *const f32, vector_ld: i32, output: *mut f32, output_ld: i32, rows: i32, cols: i32, stream: cudaStream_t);
    pub fn VectorFragment_tanhDeriv_f32(vector: *const f32, vector_ld: i32, output: *mut f32, output_ld: i32, rows: i32, cols: i32, stream: cudaStream_t);
    pub fn VectorFragment_relu_f32(vector: *const f32, vector_ld: i32, output: *mut f32, output_ld: i32, rows: i32, cols: i32, stream: cudaStream_t);
    pub fn VectorFragment_reluDeriv_f32(vector: *const f32, vector_ld: i32, output: *mut f32, output_ld: i32, rows: i32, cols: i32, stream: cudaStream_t);
    pub fn VectorFragment_customErrorCalc_f32(vector: *const f32, vector_ld: i32, ideal_vector: *const f32, ideal_vector_ld: i32, threshold: f32, scaleFoff: f32, scaleFon: f32, output: *mut f32, output_ld: i32, rows: i32, cols: i32, stream: cudaStream_t);
}