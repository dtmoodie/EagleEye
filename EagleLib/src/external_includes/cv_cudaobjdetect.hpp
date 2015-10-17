#pragma once
#include "opencv2/cudaobjdetect.hpp"
#include "RuntimeLinkLibrary.h"
#ifdef _MSC_VER // Windows

#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_cudaobjdetect300d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_cudaobjdetect300.lib")
#endif

#else // Linux
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_cudaobjdetect")
#define CALL
#endif
