#pragma once

#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__)
  #define EAGLE_EXPORTS __declspec(dllexport)
  #define TEMPLATE_EXTERN extern
#elif defined __GNUC__ && __GNUC__ >= 4
  #define EAGLE_EXPORTS __attribute__ ((visibility ("default")))
  #define TEMPLATE_EXTERN 
#else
  #define EAGLE_EXPORTS
  #define TEMPLATE_EXTERN 
#endif

#ifdef _MSC_VER
  #ifndef EagleLib_EXPORTS
    #ifdef _DEBUG
      #pragma comment(lib, "EagleLibd.lib")
      #pragma comment(lib, "pplx_2_7d.lib")
    #else
      #pragma comment(lib, "EagleLib.lib")
      #pragma comment(lib, "pplx_2_7.lib")
    #endif
  #endif
#endif

#ifndef _MSC_VER
  #include "RuntimeLinkLibrary.h"
  #ifdef NDEBUG
    RUNTIME_COMPILER_LINKLIBRARY("-lEagleLib")
  #else
    RUNTIME_COMPILER_LINKLIBRARY("-lEagleLibd")
  #endif
#endif

#ifndef BUILD_TYPE
#ifdef _DEBUG
#define BUILD_TYPE 0
#else
#define BUILD_TYPE 1
#endif
#endif
