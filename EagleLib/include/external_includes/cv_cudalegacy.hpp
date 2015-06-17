#pragma once
#include "opencv2/cudalegacy.hpp"
#include "../../RuntimeObjectSystem/RuntimeLinkLibrary.h"
#ifdef _MSC_VER // Windows

#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_cudalegacy300d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_cudalegacy300.lib")
#endif

#else // Linux
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_cudalegacy")
#define CALL
#endif
