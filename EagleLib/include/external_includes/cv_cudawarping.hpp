#pragma once
#include "opencv2/cudawarping.hpp"
#include "../../RuntimeObjectSystem/RuntimeLinkLibrary.h"
#ifdef _MSC_VER // Windows

#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_cudawarping300d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_cudawarping300.lib")
#endif

#else // Linux
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_cudawarping")
#define CALL
#endif