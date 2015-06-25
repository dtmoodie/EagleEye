#pragma once
#include "opencv2/imgproc.hpp"
#include "RuntimeLinkLibrary.h"
#ifdef _MSC_VER // Windows

#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_imgproc300d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_imgproc300.lib")
#endif

#else // Linux
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_imgproc")
#define CALL
#endif
