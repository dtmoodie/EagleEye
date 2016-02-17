#pragma once
#include "RuntimeLinkLibrary.h"
#include "parameters/Parameter.hpp"
#if _WIN32
#if _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("libParameterd.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("libParameter.lib")
#endif
#else
RUNTIME_COMPILER_LINKLIBRARY("-lParameter")
#endif
