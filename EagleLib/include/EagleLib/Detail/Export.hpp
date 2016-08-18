#pragma once

#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__)
  #define EAGLE_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
  #define EAGLE_EXPORTS __attribute__ ((visibility ("default")))
#else
  #define EAGLE_EXPORTS
#endif

#ifdef _MSC_VER
  #ifndef EagleLib_EXPORTS
    #ifdef _DEBUG
      #pragma comment(lib, "EagleLibd.lib")
      #pragma comment(lib, "signalsd.lib")
      #pragma comment(lib, "pplx_2_7d.lib")
    #else
      #pragma comment(lib, "EagleLib.lib")
      #pragma comment(lib, "signals.lib")
      #pragma comment(lib, "pplx_2_7.lib")
    #endif
  #endif
#endif

#ifndef _MSC_VER
  #include "RuntimeLinkLibrary.h"
  #ifdef _DEBUG
    RUNTIME_COMPILER_LINKLIBRARY("EagleLibd.lib");
  #else
    RUNTIME_COMPILER_LINKLIBRARY("EagleLib.lib");
  #endif
#endif