#pragma once
#include "RuntimeLinkLibrary.h"
#include <opencv2/ml.hpp>
#if _WIN32
#if _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_ml300d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_ml300.lib")
#endif
#else
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_ml")
#endif
