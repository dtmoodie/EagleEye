#pragma once
#include "cv_link_config.hpp"
#include "cv_core.hpp"
#include "RuntimeLinkLibrary.h"
#include <opencv2/imgcodecs.hpp>
#if _WIN32
#if _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_imgcodecs" CV_VERSION_ "d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_imgcodecs" CV_VERSION_ ".lib")
#endif
#else
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_imgcodecs")
#endif
