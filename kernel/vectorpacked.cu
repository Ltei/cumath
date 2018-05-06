__global__ void init_i32 (int* vector, int value, int len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
vector[idx] = value;
	}
}
extern "C" {
	void VectorPacked_init_i32 (int* vector, int value, int len, cudaStream_t stream) {
		dim3 gridDim;
		dim3 blockDim;
		blockDim.x = 1024;
		gridDim.x = (len + blockDim.x - 1) / blockDim.x;
		init_i32 <<<gridDim, blockDim, 0, stream>>> (vector, value, len);
	}
}
__global__ void init_f32 (float* vector, float value, int len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
vector[idx] = value;
	}
}
extern "C" {
	void VectorPacked_init_f32 (float* vector, float value, int len, cudaStream_t stream) {
		dim3 gridDim;
		dim3 blockDim;
		blockDim.x = 1024;
		gridDim.x = (len + blockDim.x - 1) / blockDim.x;
		init_f32 <<<gridDim, blockDim, 0, stream>>> (vector, value, len);
	}
}
__global__ void addValue_i32 (int* vector, int value, int* output, int len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
output[idx] = vector[idx] + value;
	}
}
extern "C" {
	void VectorPacked_addValue_i32 (int* vector, int value, int* output, int len, cudaStream_t stream) {
		dim3 gridDim;
		dim3 blockDim;
		blockDim.x = 1024;
		gridDim.x = (len + blockDim.x - 1) / blockDim.x;
		addValue_i32 <<<gridDim, blockDim, 0, stream>>> (vector, value, output, len);
	}
}
__global__ void addValue_f32 (float* vector, float value, float* output, int len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
output[idx] = vector[idx] + value;
	}
}
extern "C" {
	void VectorPacked_addValue_f32 (float* vector, float value, float* output, int len, cudaStream_t stream) {
		dim3 gridDim;
		dim3 blockDim;
		blockDim.x = 1024;
		gridDim.x = (len + blockDim.x - 1) / blockDim.x;
		addValue_f32 <<<gridDim, blockDim, 0, stream>>> (vector, value, output, len);
	}
}
__global__ void scl_i32 (int* vector, int value, int* output, int len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
output[idx] = vector[idx] * value;
	}
}
extern "C" {
	void VectorPacked_scl_i32 (int* vector, int value, int* output, int len, cudaStream_t stream) {
		dim3 gridDim;
		dim3 blockDim;
		blockDim.x = 1024;
		gridDim.x = (len + blockDim.x - 1) / blockDim.x;
		scl_i32 <<<gridDim, blockDim, 0, stream>>> (vector, value, output, len);
	}
}
__global__ void scl_f32 (float* vector, float value, float* output, int len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
output[idx] = vector[idx] * value;
	}
}
extern "C" {
	void VectorPacked_scl_f32 (float* vector, float value, float* output, int len, cudaStream_t stream) {
		dim3 gridDim;
		dim3 blockDim;
		blockDim.x = 1024;
		gridDim.x = (len + blockDim.x - 1) / blockDim.x;
		scl_f32 <<<gridDim, blockDim, 0, stream>>> (vector, value, output, len);
	}
}
__global__ void add_i32 (int* left_op, int* right_op, int* output, int len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
output[idx] = left_op[idx] + right_op[idx];
	}
}
extern "C" {
	void VectorPacked_add_i32 (int* left_op, int* right_op, int* output, int len, cudaStream_t stream) {
		dim3 gridDim;
		dim3 blockDim;
		blockDim.x = 1024;
		gridDim.x = (len + blockDim.x - 1) / blockDim.x;
		add_i32 <<<gridDim, blockDim, 0, stream>>> (left_op, right_op, output, len);
	}
}
__global__ void add_f32 (float* left_op, float* right_op, float* output, int len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
output[idx] = left_op[idx] + right_op[idx];
	}
}
extern "C" {
	void VectorPacked_add_f32 (float* left_op, float* right_op, float* output, int len, cudaStream_t stream) {
		dim3 gridDim;
		dim3 blockDim;
		blockDim.x = 1024;
		gridDim.x = (len + blockDim.x - 1) / blockDim.x;
		add_f32 <<<gridDim, blockDim, 0, stream>>> (left_op, right_op, output, len);
	}
}
__global__ void sub_i32 (int* left_op, int* right_op, int* output, int len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
output[idx] = left_op[idx] - right_op[idx];
	}
}
extern "C" {
	void VectorPacked_sub_i32 (int* left_op, int* right_op, int* output, int len, cudaStream_t stream) {
		dim3 gridDim;
		dim3 blockDim;
		blockDim.x = 1024;
		gridDim.x = (len + blockDim.x - 1) / blockDim.x;
		sub_i32 <<<gridDim, blockDim, 0, stream>>> (left_op, right_op, output, len);
	}
}
__global__ void sub_f32 (float* left_op, float* right_op, float* output, int len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
output[idx] = left_op[idx] - right_op[idx];
	}
}
extern "C" {
	void VectorPacked_sub_f32 (float* left_op, float* right_op, float* output, int len, cudaStream_t stream) {
		dim3 gridDim;
		dim3 blockDim;
		blockDim.x = 1024;
		gridDim.x = (len + blockDim.x - 1) / blockDim.x;
		sub_f32 <<<gridDim, blockDim, 0, stream>>> (left_op, right_op, output, len);
	}
}
__global__ void mult_i32 (int* left_op, int* right_op, int* output, int len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
output[idx] = left_op[idx] * right_op[idx];
	}
}
extern "C" {
	void VectorPacked_mult_i32 (int* left_op, int* right_op, int* output, int len, cudaStream_t stream) {
		dim3 gridDim;
		dim3 blockDim;
		blockDim.x = 1024;
		gridDim.x = (len + blockDim.x - 1) / blockDim.x;
		mult_i32 <<<gridDim, blockDim, 0, stream>>> (left_op, right_op, output, len);
	}
}
__global__ void mult_f32 (float* left_op, float* right_op, float* output, int len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
output[idx] = left_op[idx] * right_op[idx];
	}
}
extern "C" {
	void VectorPacked_mult_f32 (float* left_op, float* right_op, float* output, int len, cudaStream_t stream) {
		dim3 gridDim;
		dim3 blockDim;
		blockDim.x = 1024;
		gridDim.x = (len + blockDim.x - 1) / blockDim.x;
		mult_f32 <<<gridDim, blockDim, 0, stream>>> (left_op, right_op, output, len);
	}
}
__global__ void square_i32 (int* vector, int* output, int len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
output[idx] = vector[idx] * vector[idx];
	}
}
extern "C" {
	void VectorPacked_square_i32 (int* vector, int* output, int len, cudaStream_t stream) {
		dim3 gridDim;
		dim3 blockDim;
		blockDim.x = 1024;
		gridDim.x = (len + blockDim.x - 1) / blockDim.x;
		square_i32 <<<gridDim, blockDim, 0, stream>>> (vector, output, len);
	}
}
__global__ void square_f32 (float* vector, float* output, int len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
output[idx] = vector[idx] * vector[idx];
	}
}
extern "C" {
	void VectorPacked_square_f32 (float* vector, float* output, int len, cudaStream_t stream) {
		dim3 gridDim;
		dim3 blockDim;
		blockDim.x = 1024;
		gridDim.x = (len + blockDim.x - 1) / blockDim.x;
		square_f32 <<<gridDim, blockDim, 0, stream>>> (vector, output, len);
	}
}
__global__ void binarize_i32 (int* vector, int threshold, int* output, int len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
output[idx] = vector[idx] > threshold ? 1 : 0;
	}
}
extern "C" {
	void VectorPacked_binarize_i32 (int* vector, int threshold, int* output, int len, cudaStream_t stream) {
		dim3 gridDim;
		dim3 blockDim;
		blockDim.x = 1024;
		gridDim.x = (len + blockDim.x - 1) / blockDim.x;
		binarize_i32 <<<gridDim, blockDim, 0, stream>>> (vector, threshold, output, len);
	}
}
__global__ void binarize_f32 (float* vector, float threshold, float* output, int len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
output[idx] = vector[idx] > threshold ? 1 : 0;
	}
}
extern "C" {
	void VectorPacked_binarize_f32 (float* vector, float threshold, float* output, int len, cudaStream_t stream) {
		dim3 gridDim;
		dim3 blockDim;
		blockDim.x = 1024;
		gridDim.x = (len + blockDim.x - 1) / blockDim.x;
		binarize_f32 <<<gridDim, blockDim, 0, stream>>> (vector, threshold, output, len);
	}
}
__global__ void aypb_i32 (int a, int* y, int b, int len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
y[idx] = a * y[idx] + b;
	}
}
extern "C" {
	void VectorPacked_aypb_i32 (int a, int* y, int b, int len, cudaStream_t stream) {
		dim3 gridDim;
		dim3 blockDim;
		blockDim.x = 1024;
		gridDim.x = (len + blockDim.x - 1) / blockDim.x;
		aypb_i32 <<<gridDim, blockDim, 0, stream>>> (a, y, b, len);
	}
}
__global__ void aypb_f32 (float a, float* y, float b, int len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
y[idx] = a * y[idx] + b;
	}
}
extern "C" {
	void VectorPacked_aypb_f32 (float a, float* y, float b, int len, cudaStream_t stream) {
		dim3 gridDim;
		dim3 blockDim;
		blockDim.x = 1024;
		gridDim.x = (len + blockDim.x - 1) / blockDim.x;
		aypb_f32 <<<gridDim, blockDim, 0, stream>>> (a, y, b, len);
	}
}
__global__ void axpb_y_i32 (int a, int* x, int b, int* y, int len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
y[idx] *= a * x[idx] + b;
	}
}
extern "C" {
	void VectorPacked_axpb_y_i32 (int a, int* x, int b, int* y, int len, cudaStream_t stream) {
		dim3 gridDim;
		dim3 blockDim;
		blockDim.x = 1024;
		gridDim.x = (len + blockDim.x - 1) / blockDim.x;
		axpb_y_i32 <<<gridDim, blockDim, 0, stream>>> (a, x, b, y, len);
	}
}
__global__ void axpb_y_f32 (float a, float* x, float b, float* y, int len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
y[idx] *= a * x[idx] + b;
	}
}
extern "C" {
	void VectorPacked_axpb_y_f32 (float a, float* x, float b, float* y, int len, cudaStream_t stream) {
		dim3 gridDim;
		dim3 blockDim;
		blockDim.x = 1024;
		gridDim.x = (len + blockDim.x - 1) / blockDim.x;
		axpb_y_f32 <<<gridDim, blockDim, 0, stream>>> (a, x, b, y, len);
	}
}
__global__ void xvpy_i32 (int* x, int* v, int* y, int len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
y[idx] += x[idx] * v[idx];
	}
}
extern "C" {
	void VectorPacked_xvpy_i32 (int* x, int* v, int* y, int len, cudaStream_t stream) {
		dim3 gridDim;
		dim3 blockDim;
		blockDim.x = 1024;
		gridDim.x = (len + blockDim.x - 1) / blockDim.x;
		xvpy_i32 <<<gridDim, blockDim, 0, stream>>> (x, v, y, len);
	}
}
__global__ void xvpy_f32 (float* x, float* v, float* y, int len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
y[idx] += x[idx] * v[idx];
	}
}
extern "C" {
	void VectorPacked_xvpy_f32 (float* x, float* v, float* y, int len, cudaStream_t stream) {
		dim3 gridDim;
		dim3 blockDim;
		blockDim.x = 1024;
		gridDim.x = (len + blockDim.x - 1) / blockDim.x;
		xvpy_f32 <<<gridDim, blockDim, 0, stream>>> (x, v, y, len);
	}
}
__global__ void x_avpb_py_i32 (int* x, int a, int* v, int b, int* y, int len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
y[idx] += x[idx] * (a * v[idx] + b);
	}
}
extern "C" {
	void VectorPacked_x_avpb_py_i32 (int* x, int a, int* v, int b, int* y, int len, cudaStream_t stream) {
		dim3 gridDim;
		dim3 blockDim;
		blockDim.x = 1024;
		gridDim.x = (len + blockDim.x - 1) / blockDim.x;
		x_avpb_py_i32 <<<gridDim, blockDim, 0, stream>>> (x, a, v, b, y, len);
	}
}
__global__ void x_avpb_py_f32 (float* x, float a, float* v, float b, float* y, int len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
y[idx] += x[idx] * (a * v[idx] + b);
	}
}
extern "C" {
	void VectorPacked_x_avpb_py_f32 (float* x, float a, float* v, float b, float* y, int len, cudaStream_t stream) {
		dim3 gridDim;
		dim3 blockDim;
		blockDim.x = 1024;
		gridDim.x = (len + blockDim.x - 1) / blockDim.x;
		x_avpb_py_f32 <<<gridDim, blockDim, 0, stream>>> (x, a, v, b, y, len);
	}
}
__global__ void sigmoid_f32 (float* vector, float* output, int len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
float tmp = vector[idx];   output[idx] = 0.5 - 0.5 * tmp / (1.0 + (tmp < 0.0 ? -tmp : tmp));
	}
}
extern "C" {
	void VectorPacked_sigmoid_f32 (float* vector, float* output, int len, cudaStream_t stream) {
		dim3 gridDim;
		dim3 blockDim;
		blockDim.x = 1024;
		gridDim.x = (len + blockDim.x - 1) / blockDim.x;
		sigmoid_f32 <<<gridDim, blockDim, 0, stream>>> (vector, output, len);
	}
}
__global__ void sigmoidDeriv_f32 (float* vector, float* output, int len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
float tmp = vector[idx];   float tmp2 = 1.0 + (tmp < 0.0 ? -tmp : tmp);   output[idx] = 0.5 / (tmp2*tmp2);
	}
}
extern "C" {
	void VectorPacked_sigmoidDeriv_f32 (float* vector, float* output, int len, cudaStream_t stream) {
		dim3 gridDim;
		dim3 blockDim;
		blockDim.x = 1024;
		gridDim.x = (len + blockDim.x - 1) / blockDim.x;
		sigmoidDeriv_f32 <<<gridDim, blockDim, 0, stream>>> (vector, output, len);
	}
}
__global__ void tanh_f32 (float* vector, float* output, int len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
float tmp = vector[idx];   output[idx] = tmp / (1.0 + (tmp < 0.0 ? -tmp : tmp));
	}
}
extern "C" {
	void VectorPacked_tanh_f32 (float* vector, float* output, int len, cudaStream_t stream) {
		dim3 gridDim;
		dim3 blockDim;
		blockDim.x = 1024;
		gridDim.x = (len + blockDim.x - 1) / blockDim.x;
		tanh_f32 <<<gridDim, blockDim, 0, stream>>> (vector, output, len);
	}
}
__global__ void tanhDeriv_f32 (float* vector, float* output, int len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
float tmp = vector[idx] < 0.0 ? -vector[idx] : vector[idx];   output[idx] =  1.0 / ((1.0+tmp)*(1.0+tmp));
	}
}
extern "C" {
	void VectorPacked_tanhDeriv_f32 (float* vector, float* output, int len, cudaStream_t stream) {
		dim3 gridDim;
		dim3 blockDim;
		blockDim.x = 1024;
		gridDim.x = (len + blockDim.x - 1) / blockDim.x;
		tanhDeriv_f32 <<<gridDim, blockDim, 0, stream>>> (vector, output, len);
	}
}
__global__ void relu_f32 (float* vector, float* output, int len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
output[idx] = vector[idx] > 0.0 ? vector[idx] : 0.0;
	}
}
extern "C" {
	void VectorPacked_relu_f32 (float* vector, float* output, int len, cudaStream_t stream) {
		dim3 gridDim;
		dim3 blockDim;
		blockDim.x = 1024;
		gridDim.x = (len + blockDim.x - 1) / blockDim.x;
		relu_f32 <<<gridDim, blockDim, 0, stream>>> (vector, output, len);
	}
}
__global__ void reluDeriv_f32 (float* vector, float* output, int len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
output[idx] = vector[idx] > 0.0 ? 1.0 : 0.0;
	}
}
extern "C" {
	void VectorPacked_reluDeriv_f32 (float* vector, float* output, int len, cudaStream_t stream) {
		dim3 gridDim;
		dim3 blockDim;
		blockDim.x = 1024;
		gridDim.x = (len + blockDim.x - 1) / blockDim.x;
		reluDeriv_f32 <<<gridDim, blockDim, 0, stream>>> (vector, output, len);
	}
}
__global__ void customErrorCalc_f32 (float* vector, float* ideal_vector, float threshold, float scaleFoff, float scaleFon, float* output, int len) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
float vectorValue = vector[idx];
if (ideal_vector[idx] > threshold) {
    output[idx] = 1.0 - vectorValue;
    if (vectorValue < threshold) {
        output[idx] *= scaleFoff;
    }
} else {
    output[idx] = vectorValue * vectorValue;
    if (vectorValue > threshold) {
        output[idx] *= scaleFon;
    }
}
	}
}
extern "C" {
	void VectorPacked_customErrorCalc_f32 (float* vector, float* ideal_vector, float threshold, float scaleFoff, float scaleFon, float* output, int len, cudaStream_t stream) {
		dim3 gridDim;
		dim3 blockDim;
		blockDim.x = 1024;
		gridDim.x = (len + blockDim.x - 1) / blockDim.x;
		customErrorCalc_f32 <<<gridDim, blockDim, 0, stream>>> (vector, ideal_vector, threshold, scaleFoff, scaleFon, output, len);
	}
}
