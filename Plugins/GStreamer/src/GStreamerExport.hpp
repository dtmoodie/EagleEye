#pragma once

#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__) && defined GStreamer_EXPORTS
#  define GStreamer_EXPORT __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#  define GStreamer_EXPORT __attribute__ ((visibility ("default")))
#else
#  define GStreamer_EXPORT
#endif

#if _WIN32
#pragma warning(disable: 4251)
#pragma warning(disable: 4275)
#endif


#include "RuntimeObjectSystem/RuntimeLinkLibrary.h"
#ifdef WIN32
#ifdef _DEBUG
  RUNTIME_COMPILER_LINKLIBRARY("GStreamerd.lib")
#else
  RUNTIME_COMPILER_LINKLIBRARY("GStreamer.lib")
#endif
#else // Unix
#ifdef NDEBUG
  RUNTIME_COMPILER_LINKLIBRARY("-lGStreamer")
#else
  RUNTIME_COMPILER_LINKLIBRARY("-lGStreamerd")
#endif
#endif
