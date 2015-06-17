#pragma once
#include "opencv2/features2d.hpp"
#include "../../RuntimeObjectSystem/RuntimeLinkLibrary.h"
#ifdef _MSC_VER // Windows

#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_features2d300d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_features2d300.lib")
#endif

#else // Linux
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_features2d")
#define CALL
#endif
