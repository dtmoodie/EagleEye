#pragma once
#include "opencv2/flann.hpp"
#include "../../RuntimeObjectSystem/RuntimeLinkLibrary.h"
#ifdef _MSC_VER // Windows

#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_flann300d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_flann300.lib")
#endif

#else // Linux
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_flann")
#define CALL
#endif
