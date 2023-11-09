
__global__ add_matrix(float* a, float* b, float* c, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int index = i + j * N;
    if (i < N && j < N){
        c[index] = a[index] + b[index];
    }
}

int main() {
    dim3 dimBlock(blocksize, blocksize);
    dim3 dimGrad(N/dimBlock.x, N/dimBlock.y);
    add_matrix<<<dimGrad, dimBlock>>>(a, b, c, N);
}

