#pragma once
#include "../../RuntimeObjectSystem/RuntimeLinkLibrary.h"
#include <opencv2/imgcodecs.hpp>
#if _WIN32
#if _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_imgcodecs300d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_imgcodecs300.lib")
#endif
#else
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_imgcodecs")
#endif
