#pragma once
#include "opencv2/cudaarithm.hpp"
#include "../../RuntimeObjectSystem/RuntimeLinkLibrary.h"
#ifdef _MSC_VER // Windows

#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_cudaarithm300d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_cudaarithm300.lib")
#endif

#else // Linux
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_cudaarithm")
#define CALL
#endif