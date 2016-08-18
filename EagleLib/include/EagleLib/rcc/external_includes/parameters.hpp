#pragma once
#include "RuntimeLinkLibrary.h"
#include "parameters/Parameter.hpp"
#if _WIN32
#if _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("parametersd.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("parameters.lib")
#endif
#else
RUNTIME_COMPILER_LINKLIBRARY("-lparameters")
#endif
