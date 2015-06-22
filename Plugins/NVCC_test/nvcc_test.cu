#include "nvcc_test.cuh"

__global__ void kernel()
{

}
__global__ void kernel(unsigned char* data, int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= N)
        return;
    data[tid] *= 2;
}

void run_kernel()
{
    kernel<<<1,1>>>();
}
void run_kernel(unsigned char *data, int pixels, cudaStream_t stream)
{
    int threads = 1024;
    int blocks = pixels / 1024;
    kernel<<<blocks, threads, 0, stream>>>(data, pixels);  
}
