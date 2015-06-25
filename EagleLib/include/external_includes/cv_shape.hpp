#pragma once
#include "RuntimeLinkLibrary.h"
#include <opencv2/shape.hpp>
#if _WIN32
#if _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_shape300d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_shape300.lib")
#endif
#else
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_shape")
#endif
