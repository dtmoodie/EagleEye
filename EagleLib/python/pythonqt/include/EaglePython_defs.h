#pragma once

#ifdef EaglePython_EXPORTS && (defined( WIN32 ) || defined( _WIN32 ) || defined( WINCE ) || defined( __CYGWIN__ ))
#  define EAGLEPYTHON_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#  define EAGLEPYTHON_EXPORTS __attribute__ ((visibility ("default")))
#else
#  define EAGLEPYTHON_EXPORTS
#endif

#ifndef EaglePython_EXPORTS
  #ifdef _MSC_VER
    #ifdef _DEBUG
      #pragma comment(lib "EaglePythond.lib")
    #else
      #pragma comment(lib "EaglePython.lib")
    #endif
  #else
  #endif
#endif