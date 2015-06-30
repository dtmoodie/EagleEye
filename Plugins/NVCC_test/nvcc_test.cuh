#include <cuda.h>
#include <cuda_runtime.h>
#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE


void run_kernel();
void run_kernel(unsigned char* data, int pixels, cudaStream_t stream); 
