__global__ void init_i32 (int* vector, int vector_ld, int value, int rows, int cols) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < rows && col < cols) {
vector[row+col*vector_ld] = value;
	}
}
extern "C" {
	void VectorFragment_init_i32 (int* vector, int vector_ld, int value, int rows, int cols, cudaStream_t stream) {
		dim3 blockDim(32, 32);
		dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);
		init_i32 <<<gridDim, blockDim, 0, stream>>> (vector, vector_ld, value, rows, cols);
	}
}
__global__ void init_f32 (float* vector, int vector_ld, float value, int rows, int cols) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < rows && col < cols) {
vector[row+col*vector_ld] = value;
	}
}
extern "C" {
	void VectorFragment_init_f32 (float* vector, int vector_ld, float value, int rows, int cols, cudaStream_t stream) {
		dim3 blockDim(32, 32);
		dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);
		init_f32 <<<gridDim, blockDim, 0, stream>>> (vector, vector_ld, value, rows, cols);
	}
}
__global__ void addValue_i32 (int* vector, int vector_ld, int value, int* output, int output_ld, int rows, int cols) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < rows && col < cols) {
output[row+col*output_ld] = vector[row+col*vector_ld] + value;
	}
}
extern "C" {
	void VectorFragment_addValue_i32 (int* vector, int vector_ld, int value, int* output, int output_ld, int rows, int cols, cudaStream_t stream) {
		dim3 blockDim(32, 32);
		dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);
		addValue_i32 <<<gridDim, blockDim, 0, stream>>> (vector, vector_ld, value, output, output_ld, rows, cols);
	}
}
__global__ void addValue_f32 (float* vector, int vector_ld, float value, float* output, int output_ld, int rows, int cols) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < rows && col < cols) {
output[row+col*output_ld] = vector[row+col*vector_ld] + value;
	}
}
extern "C" {
	void VectorFragment_addValue_f32 (float* vector, int vector_ld, float value, float* output, int output_ld, int rows, int cols, cudaStream_t stream) {
		dim3 blockDim(32, 32);
		dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);
		addValue_f32 <<<gridDim, blockDim, 0, stream>>> (vector, vector_ld, value, output, output_ld, rows, cols);
	}
}
__global__ void scl_i32 (int* vector, int vector_ld, int value, int* output, int output_ld, int rows, int cols) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < rows && col < cols) {
output[row+col*output_ld] = vector[row+col*vector_ld] * value;
	}
}
extern "C" {
	void VectorFragment_scl_i32 (int* vector, int vector_ld, int value, int* output, int output_ld, int rows, int cols, cudaStream_t stream) {
		dim3 blockDim(32, 32);
		dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);
		scl_i32 <<<gridDim, blockDim, 0, stream>>> (vector, vector_ld, value, output, output_ld, rows, cols);
	}
}
__global__ void scl_f32 (float* vector, int vector_ld, float value, float* output, int output_ld, int rows, int cols) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < rows && col < cols) {
output[row+col*output_ld] = vector[row+col*vector_ld] * value;
	}
}
extern "C" {
	void VectorFragment_scl_f32 (float* vector, int vector_ld, float value, float* output, int output_ld, int rows, int cols, cudaStream_t stream) {
		dim3 blockDim(32, 32);
		dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);
		scl_f32 <<<gridDim, blockDim, 0, stream>>> (vector, vector_ld, value, output, output_ld, rows, cols);
	}
}
__global__ void add_i32 (int* left_op, int left_op_ld, int* right_op, int right_op_ld, int* output, int output_ld, int rows, int cols) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < rows && col < cols) {
output[row+col*output_ld] = left_op[row+col*left_op_ld] + right_op[row+col*right_op_ld];
	}
}
extern "C" {
	void VectorFragment_add_i32 (int* left_op, int left_op_ld, int* right_op, int right_op_ld, int* output, int output_ld, int rows, int cols, cudaStream_t stream) {
		dim3 blockDim(32, 32);
		dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);
		add_i32 <<<gridDim, blockDim, 0, stream>>> (left_op, left_op_ld, right_op, right_op_ld, output, output_ld, rows, cols);
	}
}
__global__ void add_f32 (float* left_op, int left_op_ld, float* right_op, int right_op_ld, float* output, int output_ld, int rows, int cols) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < rows && col < cols) {
output[row+col*output_ld] = left_op[row+col*left_op_ld] + right_op[row+col*right_op_ld];
	}
}
extern "C" {
	void VectorFragment_add_f32 (float* left_op, int left_op_ld, float* right_op, int right_op_ld, float* output, int output_ld, int rows, int cols, cudaStream_t stream) {
		dim3 blockDim(32, 32);
		dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);
		add_f32 <<<gridDim, blockDim, 0, stream>>> (left_op, left_op_ld, right_op, right_op_ld, output, output_ld, rows, cols);
	}
}
__global__ void sub_i32 (int* left_op, int left_op_ld, int* right_op, int right_op_ld, int* output, int output_ld, int rows, int cols) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < rows && col < cols) {
output[row+col*output_ld] = left_op[row+col*left_op_ld] - right_op[row+col*right_op_ld];
	}
}
extern "C" {
	void VectorFragment_sub_i32 (int* left_op, int left_op_ld, int* right_op, int right_op_ld, int* output, int output_ld, int rows, int cols, cudaStream_t stream) {
		dim3 blockDim(32, 32);
		dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);
		sub_i32 <<<gridDim, blockDim, 0, stream>>> (left_op, left_op_ld, right_op, right_op_ld, output, output_ld, rows, cols);
	}
}
__global__ void sub_f32 (float* left_op, int left_op_ld, float* right_op, int right_op_ld, float* output, int output_ld, int rows, int cols) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < rows && col < cols) {
output[row+col*output_ld] = left_op[row+col*left_op_ld] - right_op[row+col*right_op_ld];
	}
}
extern "C" {
	void VectorFragment_sub_f32 (float* left_op, int left_op_ld, float* right_op, int right_op_ld, float* output, int output_ld, int rows, int cols, cudaStream_t stream) {
		dim3 blockDim(32, 32);
		dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);
		sub_f32 <<<gridDim, blockDim, 0, stream>>> (left_op, left_op_ld, right_op, right_op_ld, output, output_ld, rows, cols);
	}
}
__global__ void mult_i32 (int* left_op, int left_op_ld, int* right_op, int right_op_ld, int* output, int output_ld, int rows, int cols) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < rows && col < cols) {
output[row+col*output_ld] = left_op[row+col*left_op_ld] * right_op[row+col*right_op_ld];
	}
}
extern "C" {
	void VectorFragment_mult_i32 (int* left_op, int left_op_ld, int* right_op, int right_op_ld, int* output, int output_ld, int rows, int cols, cudaStream_t stream) {
		dim3 blockDim(32, 32);
		dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);
		mult_i32 <<<gridDim, blockDim, 0, stream>>> (left_op, left_op_ld, right_op, right_op_ld, output, output_ld, rows, cols);
	}
}
__global__ void mult_f32 (float* left_op, int left_op_ld, float* right_op, int right_op_ld, float* output, int output_ld, int rows, int cols) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < rows && col < cols) {
output[row+col*output_ld] = left_op[row+col*left_op_ld] * right_op[row+col*right_op_ld];
	}
}
extern "C" {
	void VectorFragment_mult_f32 (float* left_op, int left_op_ld, float* right_op, int right_op_ld, float* output, int output_ld, int rows, int cols, cudaStream_t stream) {
		dim3 blockDim(32, 32);
		dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);
		mult_f32 <<<gridDim, blockDim, 0, stream>>> (left_op, left_op_ld, right_op, right_op_ld, output, output_ld, rows, cols);
	}
}
__global__ void square_i32 (int* vector, int vector_ld, int* output, int output_ld, int rows, int cols) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < rows && col < cols) {
output[row+col*output_ld] = vector[row+col*vector_ld] * vector[row+col*vector_ld];
	}
}
extern "C" {
	void VectorFragment_square_i32 (int* vector, int vector_ld, int* output, int output_ld, int rows, int cols, cudaStream_t stream) {
		dim3 blockDim(32, 32);
		dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);
		square_i32 <<<gridDim, blockDim, 0, stream>>> (vector, vector_ld, output, output_ld, rows, cols);
	}
}
__global__ void square_f32 (float* vector, int vector_ld, float* output, int output_ld, int rows, int cols) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < rows && col < cols) {
output[row+col*output_ld] = vector[row+col*vector_ld] * vector[row+col*vector_ld];
	}
}
extern "C" {
	void VectorFragment_square_f32 (float* vector, int vector_ld, float* output, int output_ld, int rows, int cols, cudaStream_t stream) {
		dim3 blockDim(32, 32);
		dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);
		square_f32 <<<gridDim, blockDim, 0, stream>>> (vector, vector_ld, output, output_ld, rows, cols);
	}
}
__global__ void binarize_i32 (int* vector, int vector_ld, int threshold, int* output, int output_ld, int rows, int cols) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < rows && col < cols) {
output[row+col*output_ld] = vector[row+col*vector_ld] > threshold ? 1 : 0;
	}
}
extern "C" {
	void VectorFragment_binarize_i32 (int* vector, int vector_ld, int threshold, int* output, int output_ld, int rows, int cols, cudaStream_t stream) {
		dim3 blockDim(32, 32);
		dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);
		binarize_i32 <<<gridDim, blockDim, 0, stream>>> (vector, vector_ld, threshold, output, output_ld, rows, cols);
	}
}
__global__ void binarize_f32 (float* vector, int vector_ld, float threshold, float* output, int output_ld, int rows, int cols) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < rows && col < cols) {
output[row+col*output_ld] = vector[row+col*vector_ld] > threshold ? 1 : 0;
	}
}
extern "C" {
	void VectorFragment_binarize_f32 (float* vector, int vector_ld, float threshold, float* output, int output_ld, int rows, int cols, cudaStream_t stream) {
		dim3 blockDim(32, 32);
		dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);
		binarize_f32 <<<gridDim, blockDim, 0, stream>>> (vector, vector_ld, threshold, output, output_ld, rows, cols);
	}
}
__global__ void aypb_i32 (int a, int* y, int y_ld, int b, int rows, int cols) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < rows && col < cols) {
y[row+col*y_ld] = a * y[row+col*y_ld] + b;
	}
}
extern "C" {
	void VectorFragment_aypb_i32 (int a, int* y, int y_ld, int b, int rows, int cols, cudaStream_t stream) {
		dim3 blockDim(32, 32);
		dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);
		aypb_i32 <<<gridDim, blockDim, 0, stream>>> (a, y, y_ld, b, rows, cols);
	}
}
__global__ void aypb_f32 (float a, float* y, int y_ld, float b, int rows, int cols) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < rows && col < cols) {
y[row+col*y_ld] = a * y[row+col*y_ld] + b;
	}
}
extern "C" {
	void VectorFragment_aypb_f32 (float a, float* y, int y_ld, float b, int rows, int cols, cudaStream_t stream) {
		dim3 blockDim(32, 32);
		dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);
		aypb_f32 <<<gridDim, blockDim, 0, stream>>> (a, y, y_ld, b, rows, cols);
	}
}
__global__ void axpb_y_i32 (int a, int* x, int x_ld, int b, int* y, int y_ld, int rows, int cols) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < rows && col < cols) {
y[row+col*y_ld] *= a * x[row+col*x_ld] + b;
	}
}
extern "C" {
	void VectorFragment_axpb_y_i32 (int a, int* x, int x_ld, int b, int* y, int y_ld, int rows, int cols, cudaStream_t stream) {
		dim3 blockDim(32, 32);
		dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);
		axpb_y_i32 <<<gridDim, blockDim, 0, stream>>> (a, x, x_ld, b, y, y_ld, rows, cols);
	}
}
__global__ void axpb_y_f32 (float a, float* x, int x_ld, float b, float* y, int y_ld, int rows, int cols) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < rows && col < cols) {
y[row+col*y_ld] *= a * x[row+col*x_ld] + b;
	}
}
extern "C" {
	void VectorFragment_axpb_y_f32 (float a, float* x, int x_ld, float b, float* y, int y_ld, int rows, int cols, cudaStream_t stream) {
		dim3 blockDim(32, 32);
		dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);
		axpb_y_f32 <<<gridDim, blockDim, 0, stream>>> (a, x, x_ld, b, y, y_ld, rows, cols);
	}
}
__global__ void xvpy_i32 (int* x, int x_ld, int* v, int v_ld, int* y, int y_ld, int rows, int cols) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < rows && col < cols) {
y[row+col*y_ld] += x[row+col*x_ld] * v[row+col*v_ld];
	}
}
extern "C" {
	void VectorFragment_xvpy_i32 (int* x, int x_ld, int* v, int v_ld, int* y, int y_ld, int rows, int cols, cudaStream_t stream) {
		dim3 blockDim(32, 32);
		dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);
		xvpy_i32 <<<gridDim, blockDim, 0, stream>>> (x, x_ld, v, v_ld, y, y_ld, rows, cols);
	}
}
__global__ void xvpy_f32 (float* x, int x_ld, float* v, int v_ld, float* y, int y_ld, int rows, int cols) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < rows && col < cols) {
y[row+col*y_ld] += x[row+col*x_ld] * v[row+col*v_ld];
	}
}
extern "C" {
	void VectorFragment_xvpy_f32 (float* x, int x_ld, float* v, int v_ld, float* y, int y_ld, int rows, int cols, cudaStream_t stream) {
		dim3 blockDim(32, 32);
		dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);
		xvpy_f32 <<<gridDim, blockDim, 0, stream>>> (x, x_ld, v, v_ld, y, y_ld, rows, cols);
	}
}
__global__ void x_avpb_py_i32 (int* x, int x_ld, int a, int* v, int v_ld, int b, int* y, int y_ld, int rows, int cols) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < rows && col < cols) {
y[row+col*y_ld] += x[row+col*x_ld] * (a * v[row+col*v_ld] + b);
	}
}
extern "C" {
	void VectorFragment_x_avpb_py_i32 (int* x, int x_ld, int a, int* v, int v_ld, int b, int* y, int y_ld, int rows, int cols, cudaStream_t stream) {
		dim3 blockDim(32, 32);
		dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);
		x_avpb_py_i32 <<<gridDim, blockDim, 0, stream>>> (x, x_ld, a, v, v_ld, b, y, y_ld, rows, cols);
	}
}
__global__ void x_avpb_py_f32 (float* x, int x_ld, float a, float* v, int v_ld, float b, float* y, int y_ld, int rows, int cols) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < rows && col < cols) {
y[row+col*y_ld] += x[row+col*x_ld] * (a * v[row+col*v_ld] + b);
	}
}
extern "C" {
	void VectorFragment_x_avpb_py_f32 (float* x, int x_ld, float a, float* v, int v_ld, float b, float* y, int y_ld, int rows, int cols, cudaStream_t stream) {
		dim3 blockDim(32, 32);
		dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);
		x_avpb_py_f32 <<<gridDim, blockDim, 0, stream>>> (x, x_ld, a, v, v_ld, b, y, y_ld, rows, cols);
	}
}
__global__ void sigmoid_f32 (float* vector, int vector_ld, float* output, int output_ld, int rows, int cols) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < rows && col < cols) {
float tmp = vector[row+col*vector_ld];   output[row+col*output_ld] = 0.5 - 0.5 * tmp / (1.0 + (tmp < 0.0 ? -tmp : tmp));
	}
}
extern "C" {
	void VectorFragment_sigmoid_f32 (float* vector, int vector_ld, float* output, int output_ld, int rows, int cols, cudaStream_t stream) {
		dim3 blockDim(32, 32);
		dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);
		sigmoid_f32 <<<gridDim, blockDim, 0, stream>>> (vector, vector_ld, output, output_ld, rows, cols);
	}
}
__global__ void sigmoidDeriv_f32 (float* vector, int vector_ld, float* output, int output_ld, int rows, int cols) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < rows && col < cols) {
float tmp = 1.0 + (vector[row+col*vector_ld] < 0.0 ? -vector[row+col*vector_ld] : vector[row+col*vector_ld]);   output[row+col*output_ld] = - 0.5 / (tmp*tmp);
	}
}
extern "C" {
	void VectorFragment_sigmoidDeriv_f32 (float* vector, int vector_ld, float* output, int output_ld, int rows, int cols, cudaStream_t stream) {
		dim3 blockDim(32, 32);
		dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);
		sigmoidDeriv_f32 <<<gridDim, blockDim, 0, stream>>> (vector, vector_ld, output, output_ld, rows, cols);
	}
}
__global__ void tanh_f32 (float* vector, int vector_ld, float* output, int output_ld, int rows, int cols) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < rows && col < cols) {
float tmp = vector[row+col*vector_ld];   output[row+col*output_ld] = tmp / (1.0 + (tmp < 0.0 ? -tmp : tmp));
	}
}
extern "C" {
	void VectorFragment_tanh_f32 (float* vector, int vector_ld, float* output, int output_ld, int rows, int cols, cudaStream_t stream) {
		dim3 blockDim(32, 32);
		dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);
		tanh_f32 <<<gridDim, blockDim, 0, stream>>> (vector, vector_ld, output, output_ld, rows, cols);
	}
}
__global__ void tanhDeriv_f32 (float* vector, int vector_ld, float* output, int output_ld, int rows, int cols) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < rows && col < cols) {
float tmp = vector[row+col*vector_ld] < 0.0 ? -vector[row+col*vector_ld] : vector[row+col*vector_ld];   output[row+col*output_ld] =  1.0 / ((1.0+tmp)*(1.0+tmp));
	}
}
extern "C" {
	void VectorFragment_tanhDeriv_f32 (float* vector, int vector_ld, float* output, int output_ld, int rows, int cols, cudaStream_t stream) {
		dim3 blockDim(32, 32);
		dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);
		tanhDeriv_f32 <<<gridDim, blockDim, 0, stream>>> (vector, vector_ld, output, output_ld, rows, cols);
	}
}
__global__ void relu_f32 (float* vector, int vector_ld, float* output, int output_ld, int rows, int cols) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < rows && col < cols) {
output[row+col*output_ld] = vector[row+col*vector_ld] > 0.0 ? vector[row+col*vector_ld] : 0.0;
	}
}
extern "C" {
	void VectorFragment_relu_f32 (float* vector, int vector_ld, float* output, int output_ld, int rows, int cols, cudaStream_t stream) {
		dim3 blockDim(32, 32);
		dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);
		relu_f32 <<<gridDim, blockDim, 0, stream>>> (vector, vector_ld, output, output_ld, rows, cols);
	}
}
__global__ void reluDeriv_f32 (float* vector, int vector_ld, float* output, int output_ld, int rows, int cols) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < rows && col < cols) {
output[row+col*output_ld] = vector[row+col*vector_ld] > 0.0 ? 1.0 : 0.0;
	}
}
extern "C" {
	void VectorFragment_reluDeriv_f32 (float* vector, int vector_ld, float* output, int output_ld, int rows, int cols, cudaStream_t stream) {
		dim3 blockDim(32, 32);
		dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);
		reluDeriv_f32 <<<gridDim, blockDim, 0, stream>>> (vector, vector_ld, output, output_ld, rows, cols);
	}
}
__global__ void customErrorCalc_f32 (float* vector, int vector_ld, float* ideal_vector, int ideal_vector_ld, float threshold, float scaleFoff, float scaleFon, float* output, int output_ld, int rows, int cols) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < rows && col < cols) {
float vectorValue = vector[row+col*vector_ld];
if (ideal_vector[row+col*vector_ld] > threshold) {
    output[row+col*output_ld] = 1.0 - vectorValue;
    if (vectorValue < threshold) {
        output[row+col*output_ld] *= scaleFoff;
    }
} else {
    output[row+col*output_ld] = vectorValue * vectorValue;
    if (vectorValue > threshold) {
        output[row+col*output_ld] *= scaleFon;
    }
}
	}
}
extern "C" {
	void VectorFragment_customErrorCalc_f32 (float* vector, int vector_ld, float* ideal_vector, int ideal_vector_ld, float threshold, float scaleFoff, float scaleFon, float* output, int output_ld, int rows, int cols, cudaStream_t stream) {
		dim3 blockDim(32, 32);
		dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);
		customErrorCalc_f32 <<<gridDim, blockDim, 0, stream>>> (vector, vector_ld, ideal_vector, ideal_vector_ld, threshold, scaleFoff, scaleFon, output, output_ld, rows, cols);
	}
}
