#pragma once
#include "RuntimeLinkLibrary.h"

#include <opencv2/videoio.hpp>

#ifdef _MSC_VER // Windows

#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_videoio300d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_videoio300.lib")
#endif

#else // Linux
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_videoio")
#define CALL
#endif