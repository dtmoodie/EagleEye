#pragma once
#include "RuntimeLinkLibrary.h"
#include <opencv2/videostab.hpp>
#if _WIN32
#if _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_videostab300d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_videostab300.lib")
#endif
#else
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_videostab")
#endif
