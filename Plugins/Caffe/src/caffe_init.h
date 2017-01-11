#pragma once
#include <RuntimeLinkLibrary.h>

#ifdef _MSC_VER
  #ifdef _DEBUG

  #else

  #endif
#else
  #ifdef NDEBUG
    RUNTIME_COMPILER_LINKLIBRARY("-lCaffe")
  #else
    RUNTIME_COMPILER_LINKLIBRARY("-lCaffed")
  #endif
#endif
namespace EagleLib
{
    class PLUGIN_EXPORTS caffe_init_singleton
    {
        caffe_init_singleton();
    public:
        static caffe_init_singleton* inst();
    };    
}
