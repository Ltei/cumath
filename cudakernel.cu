

__global__ void vectorSet_ker(float* vector, int len, float value) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) { vector[tid] = value; }
}

__global__ void vectorAddScl_ker(float* vector, float* output, int len, float value) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) { output[tid] = vector[tid] + value; }
}
__global__ void vectorScl_ker(float* vector, float* output, int len, float value) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) { output[tid] = vector[tid] * value; }
}
__global__ void vectorAdd_ker(float* left_op, float* right_op, float* output, int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) { output[tid] = left_op[tid] + right_op[tid]; }
}
__global__ void vectorSub_ker(float* left_op, float* right_op, float* output, int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) { output[tid] = left_op[tid] - right_op[tid]; }
}
__global__ void vectorPMult_ker(float* left_op, float* right_op, float* output, int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) { output[tid] = left_op[tid] * right_op[tid]; }
}
__global__ void vectorSigmoid_ker(float* vector, float* output, int len) {
    float tmp;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) {
      tmp = vector[tid];
      output[tid] = 0.5 - 0.5 * tmp / (1.0 + (tmp < 0.0 ? -tmp : tmp));
    }
}
__global__ void vectorSigmoidDeriv_ker(float* vector, float* output, int len) {
    float tmp, tmp2;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) {
      tmp = vector[tid];
      tmp2 = 1.0 + (tmp < 0.0 ? -tmp : tmp);
      output[tid] = - 0.5 / (tmp2*tmp2);
    }
}

__global__ void aXpb_Y_ker(float a, float* X, float b, float* Y, int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) {
        Y[tid] *= a*X[tid] + b;
    }
}
__global__ void XVpY_ker(float* X, float* V, float* Y, int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) {
        Y[tid] += X[tid] * V[tid];
    }
}

extern "C" {
    void CudaKernel_vectorSet(float* vector, int len, float value) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        vectorSet_ker <<<gridDim, blockDim>>> (vector, len, value);
    }

    void CudaKernel_vectorAddScl(float* vector, float* output, int len, float value) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        vectorAddScl_ker <<<gridDim, blockDim>>> (vector, output, len, value);
    }
    void CudaKernel_vectorScl(float* vector, float* output, int len, float value) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        vectorScl_ker <<<gridDim, blockDim>>> (vector, output, len, value);
    }
    void CudaKernel_vectorAdd(float* left_op, float* right_op, float* output, int len) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        vectorAdd_ker <<<gridDim, blockDim>>> (left_op, right_op, output, len);
    }
    void CudaKernel_vectorSub(float* left_op, float* right_op, float* output, int len) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        vectorSub_ker <<<gridDim, blockDim>>> (left_op, right_op, output, len);
    }
    void CudaKernel_vectorPMult(float* left_op, float* right_op, float* output, int len) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        vectorPMult_ker <<<gridDim, blockDim>>> (left_op, right_op, output, len);
    }
    void CudaKernel_vectorSigmoid(float* vector, float* output, int len) {
       dim3 gridDim;
       dim3 blockDim;

       blockDim.x = 1024;
       gridDim.x = (len + blockDim.x - 1) / blockDim.x;

      vectorSigmoid_ker <<<gridDim, blockDim>>> (vector, output, len);
    }
    void CudaKernel_vectorSigmoidDeriv(float* vector, float* output, int len) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        vectorSigmoidDeriv_ker <<<gridDim, blockDim>>> (vector, output, len);
    }

    void CudaKernel_aXpb_Y(float a, float* X, float b, float* Y, int len) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        aXpb_Y_ker <<<gridDim, blockDim>>> (a, X, b, Y, len);
    }
    void CudaKernel_XVpY(float* X, float* V, float* Y, int len) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        XVpY_ker <<<gridDim, blockDim>>> (X, V, Y, len);
    }

}

// aXpbpy
