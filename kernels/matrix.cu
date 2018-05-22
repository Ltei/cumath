

__global__ void convolution(float* input, int inputRows, int inputCols, int inputLd,
                            float* kernel, int kernelRows, int kernelCols, int kernelLd,
                            int rowStep, int colStep, float* output, int outputLd) {

    int row = (blockIdx.y * blockDim.y + threadIdx.y) * rowStep;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * colStep;

    if (row <= inputRows - kernelRows && col <= inputCols - kernelCols) {
        int i, j;
        output[row+col*outputLd] = 0;
        for (i=0; i<kernelRows; i++) {
            for (j=0; j<kernelCols; j++) {
                output[row+col*outputLd] += kernel[i+j*kernelLd] * input[(row+i)+(col+j)*inputLd];
            }
        }
    }

}

extern "C" {
    void Matrix_convolution(float* input, int inputRows, int inputCols, int inputLd,
                                float* kernel, int kernelRows, int kernelCols, int kernelLd,
                                int rowStep, int colStep, float* output, int outputLd, cudaStream_t stream) {

        dim3 blockDim(32, 32);
        dim3 gridDim((inputRows + blockDim.x - 1) / blockDim.x, (inputCols + blockDim.y - 1) / blockDim.y);
        convolution <<<gridDim, blockDim, 0, stream>>> (input, inputRows, inputCols, inputLd,
                                                        kernel, kernelRows, kernelCols, kernelLd,
                                                        rowStep, colStep, output, outputLd);
    }
}