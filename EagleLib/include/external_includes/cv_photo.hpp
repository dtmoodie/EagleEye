#pragma once
#include "../../RuntimeObjectSystem/RuntimeLinkLibrary.h"
#include <opencv2/photo.hpp>
#if _WIN32
#if _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_photo300d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_photo300.lib")
#endif
#else
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_photo")
#endif
