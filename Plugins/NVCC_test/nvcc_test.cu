#include "nvcc_test.cuh"

__global__ void kernel()
{

}

void run_kernel()
{
    kernel<<<1,1>>>();
}
