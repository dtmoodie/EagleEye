#pragma once
#include "RuntimeLinkLibrary.h"
#include <opencv2/ts.hpp>
#if _WIN32
#if _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_ts300d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_ts300.lib")
#endif
#else
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_ts")
#endif
