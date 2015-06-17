#pragma once
#include "opencv2/hal.hpp"
#include "../../RuntimeObjectSystem/RuntimeLinkLibrary.h"
#ifdef _MSC_VER // Windows

#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_hal300d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_hal300.lib")
#endif

#else // Linux
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_hal")
#define CALL
#endif
