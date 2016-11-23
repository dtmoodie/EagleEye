#pragma once

#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__) && defined Vclick3d_EXPORTS
#  define Vclick3d_EXPORT __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#  define Vclick3d_EXPORT __attribute__ ((visibility ("default")))
#else
#  define Vclick3d_EXPORT
#endif

#if _WIN32
#pragma warning(disable: 4251)
#pragma warning(disable: 4275)
#endif


#include "RuntimeLinkLibrary.h"
#ifdef WIN32
#ifdef _DEBUG
  RUNTIME_COMPILER_LINKLIBRARY("Vclick3dd.lib")
#else
  RUNTIME_COMPILER_LINKLIBRARY("Vclick3d.lib")
#endif
#else // Unix
#ifdef NDEBUG
  RUNTIME_COMPILER_LINKLIBRARY("-lVclick3d")
#else
  RUNTIME_COMPILER_LINKLIBRARY("-lVclick3dd")
#endif
#endif
