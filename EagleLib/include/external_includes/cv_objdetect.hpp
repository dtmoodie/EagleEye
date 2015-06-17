#pragma once
#include "../../RuntimeObjectSystem/RuntimeLinkLibrary.h"
#include <opencv2/objdetect.hpp>
#if _WIN32
#if _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_objdetect300d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_objdetect300.lib")
#endif
#else
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_objdetect")
#endif
