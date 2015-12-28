
#pragma once
#include "cv_link_config.hpp"
#include "opencv2/calib3d.hpp"
#include <external_includes/cv_ml.hpp>
#include <external_includes/cv_videoio.hpp>
#include <external_includes/cv_features2d.hpp>
#include <external_includes/cv_flann.hpp>
#include <external_includes/cv_core.hpp>
#include "RuntimeLinkLibrary.h"
#ifdef _MSC_VER // Windows
// "opencv_cudev;opencv_hal;opencv_core;opencv_flann;opencv_imgproc;opencv_ml;opencv_imgcodecs;opencv_videoio;opencv_highgui;opencv_features2d"
#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_calib3d" CV_VERSION_ "d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_calib3d" CV_VERSION_ ".lib")
#endif

#else // Linux
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_calib3d")
#define CALL
#endif
