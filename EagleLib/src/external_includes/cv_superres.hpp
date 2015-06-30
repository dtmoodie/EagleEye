#pragma once
#include "RuntimeLinkLibrary.h"
#include <opencv2/superres.hpp>
#if _WIN32
#if _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("opencv_superres300d.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("opencv_superres300.lib")
#endif
#else
RUNTIME_COMPILER_LINKLIBRARY("-lopencv_superres")
#endif
