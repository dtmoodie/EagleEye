#pragma once
#include <RuntimeObjectSystem/RuntimeLinkLibrary.h>
#include "CaffeExport.hpp"
namespace aq
{
    class Caffe_EXPORT caffe_init_singleton
    {
        caffe_init_singleton();
    public:
        static caffe_init_singleton* inst();
    };    
}
