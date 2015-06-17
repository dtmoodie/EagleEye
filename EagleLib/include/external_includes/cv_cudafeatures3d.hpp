#pragma once
#include "opencv2/cudafeatures2d.hpp"
#include "../../RuntimeObjectSystem/RuntimeLinkLibrary.h"
#ifdef _MSC_VER // Windows

#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_cudafeatures2d300d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_cudafeatures2d300.lib")
#endif

#else // Linux
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_cudafeatures2d")
#define CALL
#endif
