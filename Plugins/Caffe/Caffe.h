#pragma once

#include "EagleLib/Project_defs.hpp"

SETUP_PROJECT_DEF
#include "RuntimeLinkLibrary.h"

#ifdef _MSC_VER
#  ifdef _DEBUG
    RUNTIME_COMPILER_LINKLIBRARY("libcaffed.lib")
#  else
    RUNTIME_COMPILER_LINKLIBRARY("libcaffe.lib")
#  endif
#else



#endif


