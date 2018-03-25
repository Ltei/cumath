

__global__ void vectorSet_ker(float* vector, int len, float value) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) { vector[tid] = value; }
}

__global__ void vectorAddSclSelf_ker(float* vector, int len, float value) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) { vector[tid] += value; }
}
__global__ void vectorScaleSelf_ker(float* vector, int len, float value) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) { vector[tid] *= value; }
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

__global__ void vectorAddThenScl_ker(float* vector, float* output, int len, float addValue, float sclValue) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) { output[tid] = sclValue*(vector[tid]+addValue); }
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

extern "C" {
    void CudaKernel_vectorSet(float* vector, int len, float value) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        vectorSet_ker <<<gridDim, blockDim>>> (vector, len, value);
    }

    void CudaKernel_vectorScaleSelf(float* vector, int len, float value) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        vectorScaleSelf_ker <<<gridDim, blockDim>>> (vector, len, value);
    }
    void CudaKernel_vectorAddSclSelf(float* vector, int len, float value) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        vectorAddSclSelf_ker <<<gridDim, blockDim>>> (vector, len, value);
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

    void CudaKernel_vectorAddThenScl(float* vector, float* output, int len, float addValue, float sclValue) {
       dim3 gridDim;
       dim3 blockDim;

       blockDim.x = 1024;
       gridDim.x = (len + blockDim.x - 1) / blockDim.x;

      vectorAddThenScl_ker <<<gridDim, blockDim>>> (vector, output, len, addValue, sclValue);
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
}
