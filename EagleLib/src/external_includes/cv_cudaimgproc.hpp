#pragma once
#include "cv_link_config.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "RuntimeLinkLibrary.h"
#ifdef _MSC_VER // Windows

#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_cudaimgproc" CV_VERSION_ "d.lib")
RUNTIME_COMPILER_LINKLIBRARY("opencv_imgproc" CV_VERSION_ "d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_cudaimgproc" CV_VERSION_ ".lib")
RUNTIME_COMPILER_LINKLIBRARY("opencv_imgproc" CV_VERSION_ ".lib")
#endif

#else // Linux
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_cudaimgproc")
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_imgproc")
#define CALL
#endif
