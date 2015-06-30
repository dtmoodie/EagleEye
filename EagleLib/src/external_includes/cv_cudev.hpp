#pragma once
#include "opencv2/cudev.hpp"
#include "RuntimeLinkLibrary.h"
#ifdef _MSC_VER // Windows

#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_cudev300d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_cudev300.lib")
#endif

#else // Linux
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_cudev")
#define CALL
#endif
