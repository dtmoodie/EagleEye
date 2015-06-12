#pragma once
#include <opencv2/highgui.hpp>
#if _WIN32
#if _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_highgui300d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_highgui300.lib")
#endif
#else
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_highgui")
#endif