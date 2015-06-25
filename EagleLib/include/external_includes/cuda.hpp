#pragma once
#include "RuntimeLinkLibrary.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#ifdef _MSC_VER // Windows

RUNTIME_COMPILER_LINKLIBRARY("cudart.lib")
RUNTIME_COMPILER_LINKLIBRARY("cuda.lib")
RUNTIME_COMPILER_LINKLIBRARY("cublas.lib")

#define CALL

#else // Linux
RUNTIME_COMPILER_LINKLIBRARY("-lcudart")
#define CALL
#endif


