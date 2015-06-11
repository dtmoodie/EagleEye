#pragma once
#include <opencv2/core.hpp>
#include "../../RuntimeObjectSystem/RuntimeLinkLibrary.h"
#ifdef _MSC_VER
#define CALL __stdcall
#ifdef _DEBUG
	RUNTIME_COMPILER_LINKLIBRARY("opencv_core300d.lib")
#else
	RUNTIME_COMPILER_LINKLIBRARY("opencv_core300.lib")
#endif
#else // _MSC_VER
	RUNTIME_COMPILER_LINKLIBRARY("-lopencv_core")
#define CALL
#endif // _MSC_VER