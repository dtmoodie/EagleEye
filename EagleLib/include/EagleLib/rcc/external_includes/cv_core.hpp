#pragma once
#include "cv_link_config.hpp"
#include "RuntimeLinkLibrary.h"
#include "cv_cudev.hpp"
#include "cv_hal.hpp"
#include <opencv2/core.hpp>

#if _WIN32
  #if _DEBUG
    RUNTIME_COMPILER_LINKLIBRARY("opencv_core" CV_VERSION_ "d.lib")
  #else
    RUNTIME_COMPILER_LINKLIBRARY("opencv_core" CV_VERSION_ ".lib")
  #endif
#else
  RUNTIME_COMPILER_LINKLIBRARY("-lopencv_core")
#endif
