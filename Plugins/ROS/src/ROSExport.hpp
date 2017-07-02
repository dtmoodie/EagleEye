#pragma once

#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__) && defined ROS_EXPORTS
#  define ROS_EXPORT __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#  define ROS_EXPORT __attribute__ ((visibility ("default")))
#else
#  define ROS_EXPORT
#endif

#if _WIN32
#pragma warning(disable: 4251)
#pragma warning(disable: 4275)
#endif


#include "RuntimeObjectSystem/RuntimeLinkLibrary.h"
#ifdef WIN32
#ifdef _DEBUG
  RUNTIME_COMPILER_LINKLIBRARY("ROSd.lib")
#else
  RUNTIME_COMPILER_LINKLIBRARY("ROS.lib")
#endif
#else // Unix
#ifdef NDEBUG
  RUNTIME_COMPILER_LINKLIBRARY("-lROS")
#else
  RUNTIME_COMPILER_LINKLIBRARY("-lROSd")
#endif
#endif
