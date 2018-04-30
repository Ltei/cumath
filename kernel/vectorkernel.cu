

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
__global__ void kernel_psquare(float* vector, float* output, int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) { output[tid] = vector[tid] * vector[tid]; }
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
__global__ void kernel_tanh(float* vector, float* output, int len) {
    float tmp;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) {
      tmp = vector[tid];
      output[tid] = tmp / (1.0 + (tmp < 0.0 ? -tmp : tmp));
    }
}

__global__ void kernel_binarize(float* vector, float threshold, float* output, int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) {
        output[tid] = vector[tid] > threshold ? 1.0 : 0.0;
    }
}
__global__ void kernel_binarizeOneMax(float* vector, float* output, int len) {
    int i, maxI = 0;
    float tmpV, maxV = vector[0];
    output[0] = 0.0;
    for (i=1; i<len; i++) {
        tmpV = vector[i];
        output[i] = 0.0;
        if (tmpV > maxV) {
            maxI = i;
            maxV = tmpV;
        }
    }
    output[maxI] = 1.0;
}
__global__ void kernel_customErrorCalc(float* vector, float* ideal_vector, float threshold, float scaleFoff, float scaleFon, float* output, int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) {
        float vectorValue = vector[tid];
        if (ideal_vector[tid] > threshold) {
            output[tid] = 1.0 - vectorValue;
            if (vectorValue < threshold) {
                output[tid] *= scaleFoff;
            }
        } else {
            output[tid] = vectorValue * vectorValue;
            if (vectorValue > threshold) {
                output[tid] *= scaleFon;
            }
        }
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
__global__ void kernel_X_aVpb_pY(float* X, float a, float* V, float b, float* Y, int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) {
        Y[tid] += X[tid] * (a*V[tid]+b);
    }
}

extern "C" {
    void VectorKernel_init(float* vector, int len, float value, cudaStream_t stream) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        kernel_init <<<gridDim, blockDim, 0, stream>>> (vector, len, value);
    }

    void VectorKernel_addValue(float* vector, float* output, int len, float value, cudaStream_t stream) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        kernel_addValue <<<gridDim, blockDim, 0, stream>>> (vector, output, len, value);
    }
    void VectorKernel_scl(float* vector, float* output, int len, float value, cudaStream_t stream) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        kernel_scl <<<gridDim, blockDim, 0, stream>>> (vector, output, len, value);
    }
    void VectorKernel_add(float* left_op, float* right_op, float* output, int len, cudaStream_t stream) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        kernel_add <<<gridDim, blockDim, 0, stream>>> (left_op, right_op, output, len);
    }
    void VectorKernel_sub(float* left_op, float* right_op, float* output, int len, cudaStream_t stream) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        kernel_sub <<<gridDim, blockDim, 0, stream>>> (left_op, right_op, output, len);
    }
    void VectorKernel_pmult(float* left_op, float* right_op, float* output, int len, cudaStream_t stream) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        kernel_pmult <<<gridDim, blockDim, 0, stream>>> (left_op, right_op, output, len);
    }
    void VectorKernel_psquare(float* vector, float* output, int len, cudaStream_t stream) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        kernel_psquare <<<gridDim, blockDim, 0, stream>>> (vector, output, len);
    }
    void VectorKernel_sigmoid(float* vector, float* output, int len, cudaStream_t stream) {
       dim3 gridDim;
       dim3 blockDim;

       blockDim.x = 1024;
       gridDim.x = (len + blockDim.x - 1) / blockDim.x;

      kernel_sigmoid <<<gridDim, blockDim, 0, stream>>> (vector, output, len);
    }
    void VectorKernel_sigmoidDeriv(float* vector, float* output, int len, cudaStream_t stream) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        kernel_sigmoidDeriv <<<gridDim, blockDim, 0, stream>>> (vector, output, len);
    }
    void VectorKernel_tanh(float* vector, float* output, int len, cudaStream_t stream) {
       dim3 gridDim;
       dim3 blockDim;

       blockDim.x = 1024;
       gridDim.x = (len + blockDim.x - 1) / blockDim.x;

      kernel_tanh <<<gridDim, blockDim, 0, stream>>> (vector, output, len);
    }

    void VectorKernel_binarize(float* vector, float threshold, float* output, int len, cudaStream_t stream) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        kernel_binarize <<<gridDim, blockDim, 0, stream>>> (vector, threshold, output, len);
    }
    void VectorKernel_binarizeOneMax(float* vector, float* output, int len, cudaStream_t stream) {
        kernel_binarizeOneMax <<<1, 1, 0, stream>>> (vector, output, len);
    }
    void VectorKernel_customErrorCalc(float* vector, float* ideal_vector, float threshold, float scaleFoff, float scaleFon, float* output, int len, cudaStream_t stream) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        kernel_customErrorCalc <<<gridDim, blockDim, 0, stream>>> (vector, ideal_vector, threshold, scaleFoff, scaleFon, output, len);
    }

    void VectorKernel_aYpb(float a, float b, float* Y, int len, cudaStream_t stream) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        kernel_aYpb <<<gridDim, blockDim, 0, stream>>> (a, b, Y, len);
    }
    void VectorKernel_aXpb_Y(float a, float* X, float b, float* Y, int len, cudaStream_t stream) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        kernel_aXpb_Y <<<gridDim, blockDim, 0, stream>>> (a, X, b, Y, len);
    }
    void VectorKernel_XVpY(float* X, float* V, float* Y, int len, cudaStream_t stream) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        kernel_XVpY <<<gridDim, blockDim, 0, stream>>> (X, V, Y, len);
    }
    void VectorKernel_X_aVpb_Y(float* X, float a, float* V, float b, float* Y, int len, cudaStream_t stream) {
        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = 1024;
        gridDim.x = (len + blockDim.x - 1) / blockDim.x;

        kernel_X_aVpb_pY <<<gridDim, blockDim, 0, stream>>> (X, a, V, b, Y, len);
    }
}
