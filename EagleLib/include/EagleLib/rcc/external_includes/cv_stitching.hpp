#pragma once
#include "cv_link_config.hpp"
#include "RuntimeLinkLibrary.h"
#include <opencv2/stitching.hpp>
#if _WIN32
#if _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_stitching" CV_VERSION_ "d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_stitching" CV_VERSION_ ".lib")
#endif
#else
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_stitching")
#endif
