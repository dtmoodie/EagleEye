#include <cuda.h>
#include <cuda_runtime.h>


void run_kernel();
void run_kernel(unsigned char* data, int pixels, cudaStream_t stream); 
