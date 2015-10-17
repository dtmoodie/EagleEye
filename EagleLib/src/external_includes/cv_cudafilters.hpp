#pragma once
#include "opencv2/cudafilters.hpp"
#include "RuntimeLinkLibrary.h"
#ifdef _MSC_VER // Windows

#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_cudafilters300d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_cudafilters300.lib")
#endif

#else // Linux
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_cudafilters")
#define CALL
#endif