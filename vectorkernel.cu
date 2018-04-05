

__global__ void kernel_init(float* vector, int len, float value) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) { vector[tid] = value; }
}

__global__ void kernel_addValue(float* vector, float* output, int len, float value) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) { output[tid] = vector[tid] + value; }
}
__global__ void kernel_scl(float* vector, float* output, int len, float value) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) { output[tid] = vector[tid] * value; }
}
__global__ void kernel_add(float* left_op, float* right_op, float* output, int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) { output[tid] = left_op[tid] + right_op[tid]; }
}
__global__ void kernel_sub(float* left_op, float* right_op, float* output, int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) { output[tid] = left_op[tid] - right_op[tid]; }
}
__global__ void kernel_pmult(float* left_op, float* right_op, float* output, int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) { output[tid] = left_op[tid] * right_op[tid]; }
}
__global__ void kernel_sigmoid(float* vector, float* output, int len) {
    float tmp;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) {
      tmp = vector[tid];
      output[tid] = 0.5 - 0.5 * tmp / (1.0 + (tmp < 0.0 ? -tmp : tmp));
    }
}
__global__ void kernel_sigmoidDeriv(float* vector, float* output, int len) {
    float tmp, tmp2;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) {
      tmp = vector[tid];
      tmp2 = 1.0 + (tmp < 0.0 ? -tmp : tmp);
      output[tid] = - 0.5 / (tmp2*tmp2);
    }
}

__global__ void kernel_aYpb(float a, float b, float* Y, int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) {
        Y[tid] = a*Y[tid] + b;
    }
}
__global__ void kernel_aXpb_Y(float a, float* X, float b, float* Y, int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) {
        Y[tid] *= a*X[tid] + b;
    }
}
__global__ void kernel_XVpY(float* X, float* V, float* Y, int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) {
        Y[tid] += X[tid] * V[tid];
    }
}

extern "C" {
    void VectorKernel_init(float* vector, int len, float value) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        kernel_init <<<gridDim, blockDim>>> (vector, len, value);
    }

    void VectorKernel_addValue(float* vector, float* output, int len, float value) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        kernel_addValue <<<gridDim, blockDim>>> (vector, output, len, value);
    }
    void VectorKernel_scl(float* vector, float* output, int len, float value) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        kernel_scl <<<gridDim, blockDim>>> (vector, output, len, value);
    }
    void VectorKernel_add(float* left_op, float* right_op, float* output, int len) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        kernel_add <<<gridDim, blockDim>>> (left_op, right_op, output, len);
    }
    void VectorKernel_sub(float* left_op, float* right_op, float* output, int len) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        kernel_sub <<<gridDim, blockDim>>> (left_op, right_op, output, len);
    }
    void VectorKernel_pmult(float* left_op, float* right_op, float* output, int len) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        kernel_pmult <<<gridDim, blockDim>>> (left_op, right_op, output, len);
    }
    void VectorKernel_sigmoid(float* vector, float* output, int len) {
       dim3 gridDim;
       dim3 blockDim;

       blockDim.x = 1024;
       gridDim.x = (len + blockDim.x - 1) / blockDim.x;

      kernel_sigmoid <<<gridDim, blockDim>>> (vector, output, len);
    }
    void VectorKernel_sigmoidDeriv(float* vector, float* output, int len) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        kernel_sigmoidDeriv <<<gridDim, blockDim>>> (vector, output, len);
    }

    void VectorKernel_aYpb(float a, float b, float* Y, int len) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        kernel_aYpb <<<gridDim, blockDim>>> (a, b, Y, len);
    }
    void VectorKernel_aXpb_Y(float a, float* X, float b, float* Y, int len) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        kernel_aXpb_Y <<<gridDim, blockDim>>> (a, X, b, Y, len);
    }
    void VectorKernel_XVpY(float* X, float* V, float* Y, int len) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        kernel_XVpY <<<gridDim, blockDim>>> (X, V, Y, len);
    }
}

// aXpbpy
