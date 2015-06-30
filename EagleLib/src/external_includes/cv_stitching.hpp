#pragma once
#include "RuntimeLinkLibrary.h"
#include <opencv2/stitching.hpp>
#if _WIN32
#if _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_stitching300d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_stitching300.lib")
#endif
#else
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_stitching")
#endif
