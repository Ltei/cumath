

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

extern "C" {
    void MatrixKernel_init(float* matrix, int ld, int rows, int cols, float value) {
        dim3 blockDim(32, 32);
        dim3 gridDim((rows + blockDim.x - 1) / blockDim.x,
                     (cols + blockDim.y - 1) / blockDim.y);

        kernel_init <<<gridDim, blockDim>>> (matrix, ld, rows, cols, value);
    }

    void MatrixKernel_addValue(float* matrix, int ld, float* output, int output_ld, int rows, int cols, float value) {
        dim3 blockDim(32, 32);
        dim3 gridDim((rows + blockDim.x - 1) / blockDim.x,
                     (cols + blockDim.y - 1) / blockDim.y);

        kernel_addValue <<<gridDim, blockDim>>> (matrix, ld, output, output_ld, rows, cols, value);
    }
    void MatrixKernel_scale(float* matrix, int ld, float* output, int output_ld, int rows, int cols, float value) {
        dim3 blockDim(32, 32);
        dim3 gridDim((rows + blockDim.x - 1) / blockDim.x,
                     (cols + blockDim.y - 1) / blockDim.y);

        kernel_scale <<<gridDim, blockDim>>> (matrix, ld, output, output_ld, rows, cols, value);
    }
    void MatrixKernel_add(float* left_op, int left_op_ld, float* right_op, int right_op_ld, float* output, int output_ld, int rows, int cols) {
        dim3 blockDim(32, 32);
        dim3 gridDim((rows + blockDim.x - 1) / blockDim.x,
                     (cols + blockDim.y - 1) / blockDim.y);

        kernel_add <<<gridDim, blockDim>>> (left_op, left_op_ld, right_op, right_op_ld, output, output_ld, rows, cols);
    }
}