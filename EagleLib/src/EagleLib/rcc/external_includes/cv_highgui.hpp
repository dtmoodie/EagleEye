#pragma once
#include "cv_link_config.hpp"
#include <opencv2/highgui.hpp>
#include "RuntimeLinkLibrary.h"
#if _WIN32
#if _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_highgui" CV_VERSION_ "d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_highgui" CV_VERSION_ ".lib")
#endif
#else
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_highgui")
#endif
