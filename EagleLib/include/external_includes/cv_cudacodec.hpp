#pragma once
#include "opencv2/cudacodec.hpp"
#include "RuntimeLinkLibrary.h"
#ifdef _MSC_VER // Windows

#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_cudacodec300d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_cudacodec300.lib")
#endif

#else // Linux
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_cudacodec")
#define CALL
#endif
