#pragma once
#include "RuntimeLinkLibrary.h"
#include <opencv2/video.hpp>
#if _WIN32
#if _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_video300d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_video300.lib")
#endif
#else
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_video")
#endif
