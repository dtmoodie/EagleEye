#pragma once
#include "RuntimeLinkLibrary.h"
#include <opencv2/viz.hpp>
#if _WIN32
#if _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_viz300d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_viz300.lib")
#endif
#else
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_viz")
#endif
