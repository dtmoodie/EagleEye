#pragma once
#include "opencv2/cudastereo.hpp"
#include "RuntimeLinkLibrary.h"
#ifdef _MSC_VER // Windows

#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_cudastereo300d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_cudastereo300.lib")
#endif

#else // Linux
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_cudastereo")
#define CALL
#endif
