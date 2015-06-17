#pragma once
#include "opencv2/cudaoptflow.hpp"
#include "../../RuntimeObjectSystem/RuntimeLinkLibrary.h"
#ifdef _MSC_VER // Windows

#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_cudaoptflow300d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_cudaoptflow300.lib")
#endif

#else // Linux
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_cudaoptflow")
#define CALL
#endif
