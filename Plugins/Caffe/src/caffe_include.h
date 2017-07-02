#pragma once

#include "RuntimeObjectSystem/RuntimeLinkLibrary.h"
#ifdef _MSC_VER
  #ifdef _DEBUG
    RUNTIME_COMPILER_LINKLIBRARY("libcaffe-d.lib");
    RUNTIME_COMPILER_LINKLIBRARY("proto-d.lib");
    RUNTIME_COMPILER_LINKLIBRARY("libprotobufd.lib");
    RUNTIME_COMPILER_LINKLIBRARY("libglog.lib");
  #else
    RUNTIME_COMPILER_LINKLIBRARY("libcaffe.lib");
    RUNTIME_COMPILER_LINKLIBRARY("proto.lib");
    RUNTIME_COMPILER_LINKLIBRARY("libprotobuf.lib");
    RUNTIME_COMPILER_LINKLIBRARY("libglog.lib");
  #endif
#else
#endif