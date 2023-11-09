
for (int i = 0; i<10000; ++i){
    C[i] = A[i] + B[i];
}

__global__ void KernelFuncion(...) {
    int tid = blockDim.x * blockIdx.x + thredIdx.x;
    int varA = a[tid];
    int varB = b[tid];
    varC[tid] = varA + varB;
}

