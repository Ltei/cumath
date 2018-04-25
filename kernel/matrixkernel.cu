

__global__ void kernel_init(float* matrix, int ld, int rows, int cols, float value) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        matrix[row+col*ld] = value;
    }
}

__global__ void kernel_addValue(float* matrix, int ld, float* output, int output_ld, int rows, int cols, float value) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        output[row+col*output_ld] = matrix[row+col*ld] + value;
    }
}
__global__ void kernel_scale(float* matrix, int ld, float* output, int output_ld, int rows, int cols, float value) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        output[row+col*output_ld] = matrix[row+col*ld] * value;
    }
}
__global__ void kernel_add(float* left_op, int left_op_ld, float* right_op, int right_op_ld, float* output, int output_ld, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        output[row+col*output_ld] = left_op[row+col*left_op_ld] + right_op[row+col*right_op_ld];
    }
}

__global__ void kernel_aYpb(float a, float b, float* Y, int Y_ld, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        Y[row+col*Y_ld] = a*Y[row+col*Y_ld] + b;
    }
}

extern "C" {
    void MatrixKernel_init(float* matrix, int ld, int rows, int cols, float value, cudaStream_t stream) {
        dim3 blockDim(32, 32);
        dim3 gridDim((rows + blockDim.x - 1) / blockDim.x,
                     (cols + blockDim.y - 1) / blockDim.y);

        kernel_init <<<gridDim, blockDim, 0, stream>>> (matrix, ld, rows, cols, value);
    }

    void MatrixKernel_addValue(float* matrix, int ld, float* output, int output_ld, int rows, int cols, float value, cudaStream_t stream) {
        dim3 blockDim(32, 32);
        dim3 gridDim((rows + blockDim.x - 1) / blockDim.x,
                     (cols + blockDim.y - 1) / blockDim.y);

        kernel_addValue <<<gridDim, blockDim, 0, stream>>> (matrix, ld, output, output_ld, rows, cols, value);
    }
    void MatrixKernel_scale(float* matrix, int ld, float* output, int output_ld, int rows, int cols, float value, cudaStream_t stream) {
        dim3 blockDim(32, 32);
        dim3 gridDim((rows + blockDim.x - 1) / blockDim.x,
                     (cols + blockDim.y - 1) / blockDim.y);

        kernel_scale <<<gridDim, blockDim, 0, stream>>> (matrix, ld, output, output_ld, rows, cols, value);
    }
    void MatrixKernel_add(float* left_op, int left_op_ld, float* right_op, int right_op_ld, float* output, int output_ld, int rows, int cols, cudaStream_t stream) {
        dim3 blockDim(32, 32);
        dim3 gridDim((rows + blockDim.x - 1) / blockDim.x,
                     (cols + blockDim.y - 1) / blockDim.y);

        kernel_add <<<gridDim, blockDim, 0, stream>>> (left_op, left_op_ld, right_op, right_op_ld, output, output_ld, rows, cols);
    }

    void MatrixKernel_aYpb(float a, float b, float* Y, int Y_ld, int rows, int cols, cudaStream_t stream) {
        dim3 blockDim(32, 32);
        dim3 gridDim((rows + blockDim.x - 1) / blockDim.x,
                     (cols + blockDim.y - 1) / blockDim.y);

        kernel_aYpb <<<gridDim, blockDim, 0, stream>>> (a, b, Y, Y_ld, rows, cols);
    }
}