#pragma once
#include "cv_link_config.hpp"
#include "RuntimeLinkLibrary.h"
#include "cv_core.hpp"
#include <opencv2/videoio.hpp>

#ifdef _MSC_VER // Windows

#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_videoio" CV_VERSION_ "d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_videoio" CV_VERSION_ ".lib")
#endif

#else // Linux
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_videoio")
#define CALL
#endif