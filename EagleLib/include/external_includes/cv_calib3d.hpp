
#pragma once
#include "opencv2/calib3d.hpp"
#include "../../RuntimeObjectSystem/RuntimeLinkLibrary.h"
#ifdef _MSC_VER // Windows

#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_calib3d300d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_calib3d300.lib")
#endif

#else // Linux
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_calib3d")
#define CALL
#endif
